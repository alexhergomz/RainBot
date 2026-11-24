"""Behaviour cloning from recorded human play, vision masked.

The recorded human sessions (`RainBotData/session-*.obs`, mod log mode) predate the screen-based
vision: they carry the game-state rows and the consumed actions, but no frames. So the policy is
cloned with vision masked (every token's ``frame_valid`` is False, the cross-attention reads
nothing) and learns through the state path (body, needs, food / creature / item entities, room).
GRPO then continues with vision on, with a fading BC term so the clone is not forgotten at once.

Older protocols lack fields (protocol 2 has no food perception, no eat counter, no cycle clock);
``adapt`` maps every session onto the current row layout, deriving what it can and leaving the
rest at neutral defaults. Labels: token n (ticks 8n..8n+7) is labelled with the human's consumed
inputs on the first tick of each decision of the *next* chunk (ticks 8(n+1) + {0, 2, 4, 6}), the
same decision the policy makes at that point. Humans change input on odd ticks too; that is the
price of a 2-tick decision.

    python -m rainbot.train.bc --out ../runs/bc-1 [--steps 3000] [--device cuda]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .. import obs as O
from ..data import session as S
from ..env.fake_game import build_layout, schema_json
from ..nn.agent import Agent, AgentConfig, n_params, split_logits, to_tensors
from ..optim import make_optimizer
from ..proto import parse_schema

RB = Path(os.path.expanduser("~/.steam/steam/steamapps/common/Rain World/RainBotData"))
#: Current row layout (the fake game mirrors the mod's v8 row field for field).
ROW_DTYPE = parse_schema(json.dumps(schema_json(build_layout())).encode()).dtype
#: Food meter size from the hibernation threshold (Survivor 4/7, Monk 3/5, Hunter 6/9).
MAX_FOOD = {3: 5, 4: 7, 6: 9}


def human_sessions(root: Path = RB, min_ticks: int = 2000) -> list[Path]:
    """Sessions played by a person: no injected input (`inject_fired` ~ 0)."""
    out = []
    for p in sorted(root.glob("session-*.obs")):
        try:
            s = S.load(p)
        except Exception:
            continue
        o = s.obs
        if len(o) < min_ticks:
            continue
        inj = float(np.mean(o["inject_fired"])) if "inject_fired" in (o.dtype.names or ()) else 0.0
        if inj < 0.05:
            out.append(p)
    return out


def adapt(o: np.ndarray) -> np.ndarray:
    """Old session rows -> current row dtype; derived and default fields filled."""
    n = len(o)
    r = np.zeros(n, dtype=ROW_DTYPE)
    have = set(o.dtype.names or ())
    for name in ROW_DTYPE.names:
        if name in have:
            try:
                r[name] = o[name]
            except (ValueError, TypeError):
                pass                                   # shape changed across protocols: keep default
    if "action" not in have and "prev_action" in have:
        r["action"] = o["prev_action"]                 # the log patches in the consumed action
    room = np.asarray(o["room"], np.int64)
    r["room_changed"] = np.concatenate([[False], room[1:] != room[:-1]])
    if "food_to_hibernate" not in have:
        r["food_to_hibernate"] = 4                     # Survivor (every recorded session)
    r["max_food"] = [MAX_FOOD.get(int(h), 7) for h in r["food_to_hibernate"]]
    for k in ("cr_type", "cr_ai_kind", "cr_behavior", "food_type", "item_type"):
        if k not in have:
            r[k] = -1                                  # absent entities
    r["consumed"] = 1
    r["alive"] = o["alive"] if "alive" in have else 1
    for k in ("food_path", "shelter_path"):
        if k not in have:
            r[k] = -1                                  # recorded before path lengths existed
    if "eat_speed" not in have:
        r["eat_speed"] = 1
    if "food_hibernate_base" not in have:
        r["food_hibernate_base"] = r["food_to_hibernate"]
    return r


def session_index_map(sess, vocab: O.Vocab, world: dict | None) -> O.IndexMap:
    ix = O.IndexMap(vocab)
    ix.from_schema(sess.schema.enums)
    if world:
        ix.set_world(world["first_room_index"], world["rooms"])
    return ix


def load_world_names(root: Path = RB, region: str = "SU") -> dict | None:
    p = root / f"graph-{region}.json"
    if not p.exists():
        return None
    g = json.loads(p.read_text())
    rooms = sorted(g["rooms"], key=lambda r: r["index"])
    first = g.get("first_room_index", rooms[0]["index"])
    names = [""] * (rooms[-1]["index"] - first + 1)
    for r in rooms:
        names[r["index"] - first] = r["name"]
    return {"first_room_index": first, "rooms": names}


#: Keyframe weighting (Wen et al. 2021, "Keyframe-Focused Visual Imitation Learning", arXiv 2106.06452,
#: ICML): decisions in the top THR of the copycat predictor's error weigh W, the rest 1. Their reported
#: step setting for both CARLA and MuJoCo (THR 10%, W 5; W 4-6 measured about equal), not tuned here.
KEYFRAME_THR, KEYFRAME_W = 0.10, 5.0


def keyframe_weights(labels: list, valid: list, seg_sess: list) -> list:
    """Per-decision weights [T, 4] per segment. The copycat predictor sees only the previous decision
    (H = 1, as their MuJoCo runs): a count table p(factor value | previous joint decision), Laplace
    smoothed, cross-validated by session as the paper recommends (each session scored by the table
    of the others). Error = surprise -log p of the human's decision (sum over factors)."""
    dims = (3, 3, 2, 2, 2)
    n_prev = int(np.prod(dims)) + 1                                   # + 'none' (segment start)
    def joint(a):                                                     # [..., 5] -> [...]
        j = np.zeros(a.shape[:-1], np.int64)
        for i, d in enumerate(dims):
            j = j * d + a[..., i]
        return j
    per = []
    for lab, val in zip(labels, valid):
        a = lab.reshape(-1, 5)                                        # decisions in order
        prev = np.concatenate([[n_prev - 1], joint(a[:-1])])
        ok = np.repeat(val, 4)
        per.append((a, prev, ok))
    sess = sorted(set(seg_sess))
    counts = {s: [np.zeros((n_prev, d)) for d in dims] for s in sess}
    for (a, prev, ok), s in zip(per, seg_sess):
        for i in range(5):
            np.add.at(counts[s][i], (prev[ok], a[ok, i]), 1.0)
    total = [sum(counts[s][i] for s in sess) for i in range(5)]
    ape = []
    for (a, prev, ok), s in zip(per, seg_sess):
        e = np.zeros(len(a))
        for i in range(5):
            c = total[i] - counts[s][i] + 1.0                          # the other sessions, Laplace
            p = c / c.sum(1, keepdims=True)
            e -= np.log(p[prev, a[:, i]])
        ape.append(np.where(ok, e, np.nan))
    allv = np.concatenate([x[~np.isnan(x)] for x in ape])
    thr = np.quantile(allv, 1.0 - KEYFRAME_THR)
    out = [np.where(np.nan_to_num(x, nan=-np.inf) > thr, KEYFRAME_W, 1.0).reshape(-1, 4) for x in ape]
    frac = float(np.mean(allv > thr))
    print(f"keyframe weights: {frac:.1%} of decisions above the copycat-error threshold weigh {KEYFRAME_W:g}",
          flush=True)
    return out


class DemoSet:
    """Featurised human tokens + next-chunk decision labels, as contiguous segments."""

    def __init__(self, paths: list[Path], vocab: O.Vocab, held_out: float = 0.1):
        world = load_world_names()
        self.segments = []              # (features dict [T, ...], labels [T, 4, 5], valid [T])
        seg_sess = []                   # session index of each segment (cross-validated weights)
        for si, p in enumerate(paths):
            sess = S.load(p)
            rows = adapt(sess.obs)
            ix = session_index_map(sess, vocab, world)
            feat = O.Featurizer(ix)
            tick = np.asarray(rows["tick"], np.int64)
            # Split at tick gaps (pauses, menus, relaunches).
            cuts = np.nonzero(np.diff(tick) != 1)[0] + 1
            for seg in np.split(np.arange(len(rows)), cuts):
                ntok = len(seg) // O.TICKS
                if ntok < 16:
                    continue
                R = rows[seg[:ntok * O.TICKS]].reshape(ntok, O.TICKS)
                f = {}
                for b0 in range(0, ntok, 512):
                    part = feat.chunk(R[b0:b0 + 512])
                    part.pop("frame")
                    for k, v in part.items():
                        f.setdefault(k, []).append(v)
                f = {k: np.concatenate(v) for k, v in f.items()}
                f["frame_valid"][:] = False                 # vision masked
                a = np.asarray(R["action"], np.int64)[:, ::O.TICKS_PER_DECISION].copy()  # [ntok, 4, 5]
                a[..., :2] = np.clip(a[..., :2] + 1, 0, 2)
                a[..., 2:] = np.clip(a[..., 2:], 0, 1)
                labels = np.zeros_like(a)
                labels[:-1] = a[1:]                         # decisions made in the next chunk
                # Only decisions the slugcat could act on: the next chunk alive and not asleep.
                nxt_ok = (np.asarray(R["alive"]).all(1) & ~np.asarray(R["sleeping"]).astype(bool).any(1))
                valid = np.zeros(ntok, bool)
                valid[:-1] = nxt_ok[1:]
                self.segments.append((f, labels, valid))
                seg_sess.append(si)
        weights = keyframe_weights([s[1] for s in self.segments], [s[2] for s in self.segments], seg_sess)
        # Held out: the last `held_out` fraction of every segment (a whole held-out segment can be
        # all sleeping / dead and carry no valid decision at all).
        self.train, self.val, self.train_w, self.val_w = [], [], [], []
        for (f, lab, val), w in zip(self.segments, weights):
            cut = int(len(val) * (1 - held_out))
            if len(val) - cut >= 16 and cut >= 16:
                self.train.append(({k: v[:cut] for k, v in f.items()}, lab[:cut], val[:cut]))
                self.val.append(({k: v[cut:] for k, v in f.items()}, lab[cut:], val[cut:]))
                self.train_w.append(w[:cut]); self.val_w.append(w[cut:])
            else:
                self.train.append((f, lab, val))
                self.train_w.append(w)
        self.rng = np.random.default_rng(1)

    def tokens(self, split="train") -> int:
        return sum(len(s[2]) for s in (self.train if split == "train" else self.val))

    def batch(self, n: int, L: int, split: str = "train"):
        """n windows of L tokens (shorter segments are used whole and padded as invalid)."""
        segs = self.train if split == "train" else self.val
        segw = self.train_w if split == "train" else self.val_w
        feats, labels, valid, wts = [], [], [], []
        for _ in range(n):
            j = int(self.rng.integers(len(segs)))
            f, lab, val = segs[j]
            seg_w = segw[j]
            T = len(val)
            s0 = int(self.rng.integers(0, max(T - L, 0) + 1))
            sl = slice(s0, s0 + L)
            pad = L - min(L, T - s0)
            fw = {k: v[sl] for k, v in f.items()}
            lw, vw, ww = lab[sl], val[sl], seg_w[sl]
            if pad:
                fw = {k: np.concatenate([v, np.repeat(v[-1:], pad, 0)]) for k, v in fw.items()}
                lw = np.concatenate([lw, np.repeat(lw[-1:], pad, 0)])
                vw = np.concatenate([vw, np.zeros(pad, bool)])
                ww = np.concatenate([ww, np.zeros((pad,) + ww.shape[1:], ww.dtype)])
            feats.append(fw)
            labels.append(lw)
            valid.append(vw)
            wts.append(ww)
        f = {k: np.stack([x[k] for x in feats]) for k in feats[0]}
        return f, np.stack(labels), np.stack(valid), np.stack(wts)


def bc_loss(model: Agent, f: dict, labels: torch.Tensor, valid: torch.Tensor,
            weights: torch.Tensor | None = None):
    """Cross-entropy of the human's decisions under the policy (sum over factors, weighted mean
    over valid decisions; ``weights`` [B, T, 4] = keyframe weights), plus per-factor accuracy."""
    B, T = valid.shape
    reset = torch.zeros(B, T, dtype=torch.bool, device=valid.device)
    reset[:, 0] = True
    logits, _, _, _ = model(f, model.initial_state(B, valid.device), reset, act=labels)
    loss, acc = 0.0, []
    m = valid.float().unsqueeze(-1)                                   # [B, T, 1]
    wm = m * (weights.float() if weights is not None else 1.0)       # [B, T, 4]
    for i, lg in enumerate(split_logits(logits)):
        ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), labels[..., i].reshape(-1),
                             reduction="none").view(B, T, -1)
        denom = wm.sum() if weights is not None else m.sum() * lg.shape[2]
        loss = loss + (ce * wm).sum() / denom.clamp_min(1e-9)
        hits = ((lg.argmax(-1) == labels[..., i]).float() * m).sum()
        acc.append(float(hits / (m.sum() * lg.shape[2]).clamp_min(1)))
    return loss, acc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Behaviour cloning from human sessions, vision masked")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--sgd-lr", type=float, default=3e-3)
    ap.add_argument("--optim", default="musgd")
    ap.add_argument("--keyframe", action=argparse.BooleanOptionalAction, default=True,
                    help="keyframe weighting (Wen et al. 2021): the top 10%% of decisions by copycat error weigh 5x")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--eval-every", type=int, default=250,
                    help="full held-out evaluation; the best checkpoint by change-decision accuracy is kept")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    vocab = O.Vocab()
    paths = human_sessions()
    print(f"human sessions: {len(paths)}")
    ds = DemoSet(paths, vocab)
    print(f"demo tokens: train {ds.tokens('train')}, val {ds.tokens('val')} "
          f"({len(ds.train)} / {len(ds.val)} segments)")
    dev = torch.device(args.device)
    model = Agent(AgentConfig()).to(dev)
    print(f"agent: {n_params(model):,} parameters on {dev}")
    # Observation statistics from the demonstrations, before training.
    for key, norm in model.tokenizer.norms().items():
        x = np.concatenate([s[0][key].reshape(-1, s[0][key].shape[-1]) for s in ds.train])
        norm.update(torch.as_tensor(x, dtype=torch.float32, device=dev))
    opt = make_optimizer(model, args.optim, args.lr, args.sgd_lr)
    log = open(out / "bc_metrics.jsonl", "a")
    t0 = time.time()
    best = -1.0
    for step in range(args.steps + 1):
        model.train()
        f, lab, val, wts = ds.batch(args.batch, args.seq)
        ft = to_tensors(f, dev)
        loss, acc = bc_loss(model, ft, torch.as_tensor(lab, device=dev), torch.as_tensor(val, device=dev),
                            torch.as_tensor(wts, device=dev) if args.keyframe else None)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step and step % args.eval_every == 0:
            ev = evaluate(model, ds, dev)
            score = ev["joint_acc_on_changes"] + 0.1 * float(np.mean(ev["acc"]))
            if score > best:
                best = score
                torch.save({"model": model.state_dict(), "cfg": AgentConfig().to_dict(),
                            "vocab": vocab.state_dict(), "bc": vars(args), "eval": ev, "step": step},
                           out / "bc_best.pt")
            log.write(json.dumps({"step": step, "full_eval": ev, "best": best}) + "\n")
            print(f"step {step} held-out: {json.dumps(ev)}", flush=True)
        if step % 100 == 0:
            model.eval()
            with torch.no_grad():
                fv, lv, vv, _ = ds.batch(args.batch, args.seq, "val")
                vl, vacc = bc_loss(model, to_tensors(fv, dev), torch.as_tensor(lv, device=dev),
                                   torch.as_tensor(vv, device=dev))
            rec = {"step": step, "loss": float(loss), "acc": acc, "val_loss": float(vl), "val_acc": vacc,
                   "s": round(time.time() - t0, 1)}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else
                                  [round(a, 3) for a in v] if isinstance(v, list) else v) for k, v in rec.items()}),
                  flush=True)
    ev = evaluate(model, ds, dev)
    print("held-out:", json.dumps(ev))
    log.write(json.dumps({"final_eval": ev}) + "\n")
    torch.save({"model": model.state_dict(), "cfg": AgentConfig().to_dict(), "vocab": vocab.state_dict(),
                "bc": vars(args), "eval": ev}, out / "bc.pt")
    print(f"saved {out / 'bc.pt'}")
    return 0


@torch.no_grad()
def evaluate(model: Agent, ds: DemoSet, device, split: str = "val", L: int = 128) -> dict:
    """Accuracy over the whole split, overall and on *change* decisions (where the human's input
    differs from the last tick of the current chunk). Persistence alone gets ~93% overall and 0% on
    changes, so the change accuracy is the number that says whether anything was learned."""
    model.eval()
    segs = ds.val if split == "val" else ds.train
    hit_all, hit_chg, n_all, n_chg = np.zeros(5), np.zeros(5), 0, 0
    joint_chg = 0
    for f, lab, val in segs:
        T = len(val)
        for s0 in range(0, T, L):
            sl = slice(s0, min(T, s0 + L))
            ft = to_tensors({k: v[sl][None] for k, v in f.items()}, device)
            n = sl.stop - sl.start
            reset = torch.zeros(1, n, dtype=torch.bool, device=device)
            reset[:, 0] = True
            # Teacher-forced: each decision sees the human's decisions before it.
            logits, _, _, _ = model(ft, model.initial_state(1, device), reset,
                                    act=torch.as_tensor(lab[sl][None], device=device))
            pred = torch.stack([lg.argmax(-1) for lg in split_logits(logits)], -1)[0].cpu().numpy()  # [n,4,5]
            y, v = lab[sl], val[sl]
            cur = f["action"][sl][:, -1]                                   # [n, 5]
            for t in np.nonzero(v)[0]:
                for j in range(4):
                    ok = pred[t, j] == y[t, j]
                    hit_all += ok
                    n_all += 1
                    if not np.array_equal(y[t, j], cur[t]):
                        hit_chg += ok
                        n_chg += 1
                        joint_chg += ok.all()
    return {"decisions": n_all, "acc": (hit_all / max(n_all, 1)).round(4).tolist(),
            "changes": n_chg, "acc_on_changes": (hit_chg / max(n_chg, 1)).round(4).tolist(),
            "joint_acc_on_changes": round(joint_chg / max(n_chg, 1), 4)}


if __name__ == "__main__":
    raise SystemExit(main())
