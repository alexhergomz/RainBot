"""Training on the real game: GRPO (default) or PPO with a critic.

One step is one token: one action chunk of 4 decisions (8 ticks). PPO uses gamma 0.999 per token,
GAE lambda 0.95, clip 0.2, value coefficient 0.5 and entropy coefficient 0.01, with an alarm when
entropy falls below 30% of its maximum. GRPO (train/grpo.py) replaces the critic with a group of
members that start from the same seeded state.

Compute split: rollouts run while the games step; for each update the games stay frozen in lockstep
(keepalive pings hold their links) and the learner trains on the GPU. The learner never trains while
a game is mid-step: the learner and the game both busy on this GPU have crashed the machine.

Rollouts: all games go through one batched forward pass per step. With two or more games (PPO path)
they are split into two halves, so one half's games simulate while the other half's forward pass
runs. Recurrent state carries across rollouts; the state at the start of every sequence is stored
and the learner starts from it. ``burn_in`` > 0 masks the first tokens of each sequence from the
losses (R2D2-style).

    python -m rainbot.train.ppo --instances N [--init BC_CKPT] [--run-dir DIR] [--resume CKPT]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .. import obs as O
from .. import reward as R
from ..env.rain_env import NEUTRAL, RainEnv, Step
from ..nn.agent import Agent, AgentConfig, log_prob_entropy, max_entropy, n_params, to_tensors, to_wire
from ..optim import make_optimizer
from . import plasticity as P
from .clips import ClipRecorder
from .curriculum import Curriculum, EpisodeStats
from .episodes import EpisodeWriter
from .grpo import GroupCollector
from .rollout import EnvTrack, Rollout, clone_state, gae, has_event, scatter_state, stack_states, state_to, stomach

REPO = Path(__file__).resolve().parents[3]


@dataclass
class PPOConfig:
    instances: int = 1
    base_port: int = 54321
    seq_len: int = 256
    seqs_per_env: int = 2            # PPO rollout length per game = seqs_per_env * seq_len tokens
    epochs: int = 4
    minibatch_seqs: int = 4
    dual_clip: float = 3.0           # dual-clip PPO bound for negative advantages (Ye et al. 2020)
    accum_seqs: int = 1              # sequences per forward/backward inside a minibatch (gradient accumulation)
    vision_checkpoint: bool = False  # recompute vision activations in backward
    fp16: bool = False               # fp16 autocast + GradScaler (RDNA2 has double-rate fp16)
    vision_grad_frac: float = 1.0    # share of frames that send gradient into the CNN in the update
    gamma: float = 0.999
    lam: float = 0.95
    clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    entropy_alarm: float = 0.30
    max_grad_norm: float = 1.0
    optim: str = "musgd"             # or "adamw"
    lr: float = 1.5e-3
    sgd_lr: float = 3e-3
    weight_decay: float = 0.01
    burn_in: int = 0
    max_cycles: int = 3              # PPO: truncate an episode after this many cycles
    max_episode_tokens: int = 20000  # PPO: safety truncation
    learner_device: str = "cuda"
    actor_device: str = "cuda"
    bf16: bool = False               # off: bf16 autocast is 2x slower on RDNA2 (no bf16 matrix units)
    compile: bool = False
    stage: str = "L"
    stage_level: int = 0             # start inside the stage (a ladder rung); ignored on resume
    iterations: int = 1000000
    plasticity_every: int = 10
    eval_every: int = 1
    run_dir: str = ""
    seed: int = 0
    sim_speed: float = 16.0
    algo: str = "grpo"               # "grpo" (group baseline, no critic) or "ppo" (GAE + PopArt critic)
    group_tokens: int = 128          # GRPO: chunk length of the update, and the short-episode length
    groups_per_update: int = 2       # GRPO: token budget per iteration, in groups of group_tokens
    group_repeats: int = 2           # GRPO: passes per group over the games with the same seeded start
    shaping: bool = True
    init: str = ""                   # start from these weights (e.g. a BC checkpoint), not a resume
    bc_coef: float = 0.0             # weight of the human-demonstration BC loss during RL ...
    bc_decay: float = 0.97           # ... multiplied by this every iteration
    bc_batch: int = 2
    kl_ref: float = 0.0              # GRPO: KL penalty toward a frozen reference policy (the --init weights)
    member_temps: bool = True        # GRPO: each group member samples at its own temperature
    forage_window: int = 10          # food forecast refitted from the episodes of the last N iterations
    snapshot_every: int = 10         # keep ckpt-iterNNNN.pt every N iterations, plus ckpt-best.pt
    snapshot_keep_every: int = 50    # never delete the snapshots at multiples of this
    region: str = "SU"               # games found in another region are moved back
    home_den: str = "SU_S01"
    render: bool = True              # must stay on: the policy sees the rendered screen


def _batch(feats: list[dict]) -> dict:
    return {k: np.concatenate([f[k] for f in feats], axis=0) for k in feats[0]}


class Trainer(GroupCollector):
    def __init__(self, cfg: PPOConfig, envs: list[RainEnv]):
        self.cfg = cfg
        self.envs = envs
        torch.manual_seed(cfg.seed)
        self.dev_l = torch.device(cfg.learner_device if torch.cuda.is_available() or
                                  cfg.learner_device == "cpu" else "cpu")
        self.dev_a = torch.device(cfg.actor_device)
        self.learner = Agent(AgentConfig()).to(self.dev_l)
        self.actor = Agent(AgentConfig()).to(self.dev_a)
        self.actor.load_state_dict(self.learner.state_dict())
        self.actor.eval()
        if cfg.compile:
            _try_compile(self.learner, "learner")
            _try_compile(self.actor, "actor")
        self.opt = make_optimizer(self.learner, cfg.optim, cfg.lr, cfg.sgd_lr, cfg.weight_decay)
        self.scaler = torch.amp.GradScaler("cuda", enabled=cfg.fp16 and self.dev_l.type == "cuda")
        self.learner.cfg.checkpoint_vision = cfg.vision_checkpoint
        self.learner.cfg.vision_grad_frac = cfg.vision_grad_frac
        self.vocab = envs[0].vocab
        self.curriculum = Curriculum(cfg.stage, seed=cfg.seed)
        self.curriculum.level = cfg.stage_level
        self.tracks = [EnvTrack(stage=cfg.stage) for _ in envs]
        self.drives = R.DriveTracker(len(envs))
        self.state = self.actor.initial_state(len(envs), self.dev_a)
        self.finished: list[EpisodeStats] = []
        self.iteration = 0
        self.pool = ThreadPoolExecutor(max_workers=max(1, len(envs)))
        self.run_dir = Path(cfg.run_dir or REPO / "runs" / time.strftime("%Y%m%d-%H%M%S"))
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_f = open(self.run_dir / "metrics.jsonl", "a")
        self.demos = None
        self.ref = None
        self.best_food = -1.0
        self.clips = ClipRecorder(self.run_dir / "clips", len(envs))
        self._episodes = EpisodeWriter(self.run_dir)
        # GRPO state
        self._tau_max = 1.0
        self._rep = 0
        self._start_tile: dict[int, list] = {}
        self._memreport_sent = False
        self.zero_adv_frac: list[float] = []
        self._refills = 0
        self._path_diag: list = []
        self._forage_diag: dict = {}

    def load_demos(self) -> None:
        """Featurise the human demonstrations with the current vocabulary. Call after any init/resume
        has loaded the checkpoint's vocabulary, or the ids would not match."""
        from .bc import DemoSet, human_sessions
        self.demos = DemoSet(human_sessions(), self.vocab)
        print(f"BC regulariser: {self.demos.tokens()} demo tokens, coef {self.cfg.bc_coef} x "
              f"{self.cfg.bc_decay}^iter")
        print(f"agent: {n_params(self.learner):,} parameters; learner on {self.dev_l}, actor on {self.dev_a}")

    # ---- checkpoints ------------------------------------------------------------------------
    def save(self) -> None:
        torch.save({"model": self.learner.state_dict(), "cfg": AgentConfig().to_dict(),
                    "ppo": asdict(self.cfg), "vocab": self.vocab.state_dict(),
                    "curriculum": self.curriculum.state_dict(), "optim": self.opt.state_dict(),
                    "iteration": self.iteration, "tau_max": self._tau_max}, self.run_dir / "ckpt.pt.tmp")
        os.replace(self.run_dir / "ckpt.pt.tmp", self.run_dir / "ckpt.pt")

    def load(self, path: Path) -> None:
        ck = torch.load(path, map_location=self.dev_l, weights_only=False)
        self.learner.load_state_dict(ck["model"])
        self.actor.load_state_dict(self.learner.state_dict())
        self.vocab.load_state_dict(ck["vocab"])
        self.curriculum.load_state_dict(ck["curriculum"])
        try:
            self.opt.load_state_dict(ck["optim"])
            # load_state_dict also restores the old learning rates: set the configured ones again.
            other_lr = self.cfg.sgd_lr if self.cfg.optim == "musgd" else self.cfg.lr
            for g, lr in zip(self.opt.param_groups, (self.cfg.lr, other_lr)):
                g["lr"] = lr
        except (ValueError, KeyError) as e:
            print(f"optimizer state not restored: {e}")
        # Optimizer state saved before a parameter grew (the vision stem's distance channel) is
        # zero-padded to the parameter's shape, like the parameter itself.
        for param, state in self.opt.state.items():
            for key, val in list(state.items()):
                if torch.is_tensor(val) and val.dim() == param.dim() and val.shape != param.shape:
                    pad = torch.zeros_like(param)
                    pad[tuple(slice(0, n) for n in val.shape)] = val
                    state[key] = pad
        self.iteration = ck.get("iteration", 0)
        self._tau_max = float(ck.get("tau_max", 1.0))

    def init_from(self, path: Path) -> None:
        """Weights and vocabulary only (a BC clone): optimizer, curriculum and iteration start fresh."""
        ck = torch.load(path, map_location=self.dev_l, weights_only=False)
        self.learner.load_state_dict(ck["model"])
        self.actor.load_state_dict(self.learner.state_dict())
        self.vocab.load_state_dict(ck["vocab"])
        print(f"initialised from {path}")

    def load_ref(self, path: Path) -> None:
        """The frozen reference policy for the KL penalty: the --init weights, also on resume."""
        ck = torch.load(path, map_location=self.dev_l, weights_only=False)
        self.ref = Agent(AgentConfig()).to(self.dev_l)
        self.ref.load_state_dict(ck["model"])
        self.ref.eval()
        for p in self.ref.parameters():
            p.requires_grad_(False)
        print(f"KL-to-reference: beta {self.cfg.kl_ref} toward {path}")

    # ---- game plumbing ----------------------------------------------------------------------
    def start(self) -> None:
        for e in self.envs:
            if e.sock is None:
                e.connect(retries=30)
            e.command(f"speed {self.cfg.sim_speed}")
            e.command("render " + ("on" if self.cfg.render else "off"))
            e.command("autopilot on")
            # GRPO episodes end with the cycle: the mod books the night and freezes the world instead of
            # running the death or sleep screens and a new cycle, so a reset lands at once.
            e.command("endings " + ("freeze" if self.cfg.algo == "grpo" else "game"))
            # Ping until every game has sent its first step. A new game can take minutes to load its
            # save, and the games that answered at once would drop their links meanwhile.
            e.keepalive(True)
        steps = list(self.pool.map(lambda e: e.recv_step(), self.envs))
        for e in self.envs:
            e.keepalive(False)
        for i, s in enumerate(steps):
            self.tracks[i].step = s
            if self.envs[i].world:
                self.curriculum.set_world(self.envs[i].world)

    def _new_spec(self, world=None) -> tuple[str, str, int]:
        """A reset spec from the curriculum: (stage, spec, ladder rung or -1)."""
        stage, spec = self.curriculum.reset_spec(world=world)
        return stage, spec, (self.curriculum.last_rung if stage == "L" else -1)

    def _advance(self, i: int, actions: np.ndarray | None) -> Step:
        e = self.envs[i]
        if actions is not None:
            e.send_actions(actions)
        return e.recv_step()

    def _phase(self, name: str) -> None:
        """Write the training phase for outside tools. scripts/replay.py renders only during
        "collect": the learner and a rendering game must never use the GPU together."""
        try:
            (self.run_dir / "phase").write_text(name)
        except OSError:
            pass

    def _feat(self, i: int) -> dict:
        st = self.tracks[i].step
        return self.envs[i].feat.chunk(st.rows[None], st.frame[None], np.array([st.frame_tick]))

    # ---- acting -----------------------------------------------------------------------------
    @torch.no_grad()
    def _act(self, idx: list[int], ro: Rollout, t: int):
        """One batched forward pass for games ``idx``; stores their token t and returns the actions."""
        feats = [self._feat(i) for i in idx]
        fb = _batch(feats)
        f = to_tensors({k: v[:, None] for k, v in fb.items()}, self.dev_a)
        reset = torch.tensor([[self.tracks[i].step.episode_start] for i in idx], device=self.dev_a)
        sub = Agent.index_state(self.state, torch.tensor(idx, device=self.dev_a))
        # Each member samples from logits / T_i, and its log-prob is stored under the same T_i.
        taus = torch.tensor([self._tau(i) for i in idx], device=self.dev_a, dtype=torch.float32)
        logits, a, v, new_sub, _ = self.actor.act(f, sub, reset, tau=taus)
        a = a[:, 0]
        lp, _ = log_prob_entropy(logits / taus.to(logits.dtype).view(-1, 1, 1, 1), a.unsqueeze(1))
        v_raw = self.actor.value.denormalize(v)[:, 0].cpu()
        lp, a = lp.cpu(), a.cpu()
        if ro.logits is None:
            ro.logits = np.zeros((ro.n, ro.T) + tuple(logits.shape[2:]), np.float16)
        for j, i in enumerate(idx):
            for blk_state, new in zip(self.state, new_sub):
                scatter_state(blk_state, new, j, i)
            s = self.tracks[i].step
            last = s.rows[s.n_real - 1]
            ro.put_feat(i, t, {k: x[j:j + 1] for k, x in fb.items()})
            ro.reset[i, t] = s.episode_start
            ro.act[i, t] = a[j].numpy()
            ro.logp[i, t] = float(lp[j, 0])
            ro.temp[i, t] = float(taus[j])
            ro.logits[i, t] = logits[j, 0].float().cpu().numpy()
            ro.value[i, t] = float(v_raw[j])
            ro.deficits[i, t] = R.deficits(last)
            ro.stale[i, t] = int(last["tick"]) - s.frame_tick if s.frame_tick >= 0 else -1
            ro.brightness[i, t] = float(s.frame.mean()) if s.frame is not None else 0.0
        return a

    def _receive(self, i: int, s: Step, ro: Rollout, t: int) -> None:
        """Store game i's new step and the reward of token t."""
        prev = self.tracks[i].step
        if s.episode_start:
            # An episode change mid-segment ends the old episode here: no base reward across the jump,
            # and the old episode's shaping closes with Phi(terminal) = 0 (exact telescoping).
            ro.rew[i, t] = (-float(R.potential(prev.rows[prev.n_real - 1], self.cfg.gamma))
                            if self.cfg.shaping and self.cfg.algo == "grpo" else 0.0)
            ro.next_start[i, t] = True
        else:
            a, b = prev.rows[prev.n_real - 1], s.rows[s.n_real - 1]
            boundary = has_event(prev, "session_end") or has_event(s, "session_start")
            # The night's outcome as the mod reports it. Never inferred from karma: at the cap a sleep adds none.
            slept = has_event(prev, "session_end", "sleep")
            if self.cfg.shaping:
                r = R.shaped_reward(a, b, self.cfg.gamma, cycle_boundary=boundary, slept=slept)
            else:
                r = R.reward_across_cycle(a, b, slept) if boundary else R.reward(a, b, slept)
            ro.rew[i, t] = float(r)
        self.tracks[i].step = s

    # ---- PPO collection ---------------------------------------------------------------------
    def _request_reset(self, i: int, episode: int) -> None:
        tr = self.tracks[i]
        tr.stage, spec, tr.rung = self._new_spec(self.envs[i].world)
        self.envs[i].request_reset(spec)
        tr.awaiting_reset, tr.requested, tr.reset_from_episode = True, True, episode

    def _begin_step(self, i: int) -> bool:
        """Bookkeeping before acting on game i's current step (PPO path). Returns whether the step is valid."""
        tr, s, e = self.tracks[i], self.tracks[i].step, self.envs[i]
        if e.world:
            self.curriculum.set_world(e.world)
        ep_id = int(s.rows[0]["episode"])
        if has_event(s, "reset_failed"):
            print(f"env {i}: reset failed: {[ev for ev in s.events if ev.get('type') == 'reset_failed']}")
            self._request_reset(i, ep_id)
            return False
        region = (e.world or {}).get("region")
        if region and region != self.cfg.region:
            # Out of scope (a save that ended elsewhere, or a gate crossed): set the den in the home
            # region and die once; the game's own respawn carries the slugcat across regions.
            if not tr.relocating:
                print(f"env {i}: in region {region}, moving to {self.cfg.home_den}")
                e.command(f"den {self.cfg.home_den}")
                e.command("kill")
                tr.relocating = True
            tr.awaiting_reset, tr.requested = True, False
            return False
        tr.relocating = False
        if tr.awaiting_reset and tr.requested and ep_id != tr.reset_from_episode:
            tr.awaiting_reset, tr.requested = False, False
        if s.episode_start or (not tr.awaiting_reset and tr.stats.tokens == 0):
            self.drives.reset(i)
            tr.stats = EpisodeStats(stage=tr.stage, rung=tr.rung)
            tr.last_food, tr.last_pos, tr.last_room = None, None, -1
        if tr.awaiting_reset:
            # Connected but no episode of ours yet, or a reset still in flight: request one once.
            if not tr.requested:
                self._request_reset(i, ep_id)
            return False
        last = s.rows[s.n_real - 1]
        st = tr.stats
        food = stomach(last)
        boundary = any(ev.get("type") in ("session_end", "session_start", "reset_done") for ev in s.events)
        if tr.last_food is not None and food > tr.last_food and not boundary:
            st.pips += food - tr.last_food
        tr.last_food = food
        st.sleeps += s.sleeps
        st.deaths += s.deaths + sum(ev.get("reason") == "starve" for ev in s.events
                                    if ev.get("type") == "session_end")
        st.ticks += s.n_real
        st.tokens += 1
        st.rooms.add(int(last["room"]))
        pos = np.asarray(last["pos0"], np.float64)
        if tr.last_pos is not None and int(last["room"]) == tr.last_room:
            d = float(np.linalg.norm(pos - tr.last_pos))
            st.displacement += d
            st.idle_tokens += d < 2.0
        tr.last_pos, tr.last_room = pos, int(last["room"])
        # Truncate when the cycle budget is spent, or at the stage's own episode length.
        limit = self.curriculum.episode_tokens(st) or self.cfg.max_episode_tokens
        if st.sleeps + st.deaths >= self.cfg.max_cycles or st.tokens >= min(limit, self.cfg.max_episode_tokens):
            self.finished.append(st)
            self._request_reset(i, ep_id)
            return False
        return True

    def collect(self) -> tuple[Rollout, list]:
        """PPO rollout: T tokens from every game, two halves stepping while the other half thinks."""
        cfg, N = self.cfg, len(self.envs)
        T = cfg.seq_len * cfg.seqs_per_env
        ro = Rollout(N, T)
        seq_states = []
        halves = [list(range(N))] if N < 2 else [list(range(N // 2)), list(range(N // 2, N))]
        futures: dict[int, object] = {}
        for t in range(T):
            if t % cfg.seq_len == 0:
                seq_states.append(Agent.detach_state([clone_state(s) for s in self.state]))
            for half in halves:
                for i in half:
                    if i in futures:
                        self._receive(i, futures.pop(i).result(), ro, t - 1)
                for i in half:
                    ro.valid[i, t] = self._begin_step(i)
                a = self._act(half, ro, t)
                for j, i in enumerate(half):
                    s = self.tracks[i].step
                    ro.acted[i, t] = s.needs_action and ro.valid[i, t]
                    wire = to_wire(a[j]).numpy().astype(np.int8) if ro.valid[i, t] else NEUTRAL
                    futures[i] = self.pool.submit(self._advance, i, wire if s.needs_action else None)
        for i, fu in futures.items():
            self._receive(i, fu.result(), ro, T - 1)
        # Bootstrap values for the final tokens, without committing the recurrent state.
        with torch.no_grad():
            feats = _batch([self._feat(i) for i in range(N)])
            f = to_tensors({k: v[:, None] for k, v in feats.items()}, self.dev_a)
            reset = torch.tensor([[self.tracks[i].step.episode_start] for i in range(N)], device=self.dev_a)
            _, _, v, _, _ = self.actor.act(f, self.state, reset)
            ro.boot[:] = self.actor.value.denormalize(v)[:, 0].cpu().numpy()
            ro.boot[reset[:, 0].cpu().numpy()] = 0.0
        return ro, seq_states

    # ---- update -----------------------------------------------------------------------------
    def update(self, ro: Rollout, seq_states: list | None, adv_n: np.ndarray | None = None,
               seq_len: int | None = None) -> dict:
        """PPO-clip update. PPO: advantages from GAE with the critic. GRPO: ``adv_n`` is given
        (group-normalised) and there is no critic."""
        cfg = self.cfg
        use_critic = adv_n is None
        pol_mask = ro.valid & ro.acted
        if use_critic:
            adv, ret = gae(ro, cfg.gamma, cfg.lam)
            sel = adv[pol_mask]
            adv_n = (adv - sel.mean()) / (sel.std() + 1e-8) if sel.size > 1 else adv
            self.learner.value.update(torch.as_tensor(ret[ro.valid], device=self.dev_l))
        else:
            ret = np.zeros_like(ro.rew)

        L = seq_len or cfg.seq_len
        seqs = [(i, k) for k in range(ro.T // L) for i in range(ro.n) if ro.valid[i, k * L:(k + 1) * L].any()]
        stats: dict = defaultdict(list)
        probe = P.DormantProbe(self.learner) if self.iteration % cfg.plasticity_every == 0 else None
        rng = np.random.default_rng(self.iteration)
        last_h = None
        for _ in range(cfg.epochs):
            order = rng.permutation(len(seqs))
            for b0 in range(0, len(seqs), cfg.minibatch_seqs):
                mb = [seqs[j] for j in order[b0:b0 + cfg.minibatch_seqs]]
                # Gradient accumulation: forward/backward accum_seqs sequences at a time, every loss
                # normalised by the whole minibatch's mask counts. Same gradient as one pass, with only
                # one micro-batch's activations alive.
                pm = np.stack([pol_mask[i, k * L:(k + 1) * L] for i, k in mb])
                vm = np.stack([ro.valid[i, k * L:(k + 1) * L] for i, k in mb])
                n_pol = max(int(pm[:, cfg.burn_in:].sum()), 1)
                n_val = max(int(vm[:, cfg.burn_in:].sum()), 1)
                self.opt.zero_grad(set_to_none=True)
                acc: dict = defaultdict(float)
                step = max(1, cfg.accum_seqs)
                for m0 in range(0, len(mb), step):
                    parts, last_h = self._micro_batch(mb[m0:m0 + step], L, ro, seq_states, adv_n, ret, pol_mask,
                                                      use_critic, n_pol, n_val, with_bc=m0 == 0, stats=stats)
                    for k, v in parts.items():
                        acc[k] += v
                self._optimizer_step(stats)
                for k in ("pg", "vf", "ent", "kl", "clipfrac"):
                    stats[k].append(acc[k])
                if self.ref is not None:
                    stats["kl_ref"].append(acc["kl_ref"])
                if probe is not None:
                    probe.close()   # one minibatch is enough for the activation statistics
                    masks = probe.dormant()
                    probe = None
                    stats["dormant"] = P.dormant_ratio(masks)
                    if stats["dormant"] > P.RESET_ABOVE:
                        stats["redo_units"] = P.redo_reset(self.learner, masks, self.opt)
        # Observation statistics advance after the update, so the old log-probs stayed exact during it.
        for key, norm in self.learner.tokenizer.norms().items():
            norm.update(torch.as_tensor(ro.feat[key][ro.valid], device=self.dev_l, dtype=torch.float32))
        self.actor.load_state_dict({k: v.to(self.dev_a) for k, v in self.learner.state_dict().items()})
        if self.dev_l.type == "cuda":
            # Give the cached blocks back while the games run: the mod's clip ring needs the VRAM.
            torch.cuda.empty_cache()
        out = {k: float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")
               for k, v in stats.items() if isinstance(v, list) and v}
        out.update({k: v for k, v in stats.items() if not isinstance(v, list)})
        if last_h is not None and self.iteration % cfg.plasticity_every == 0:
            out["feature_rank"] = P.feature_effective_rank(last_h)
            out["stable_rank_mean"] = float(np.mean(list(P.weight_stable_ranks(self.learner).values())))
        out["value_mu"] = float(self.learner.value.mu)
        out["value_sigma"] = float(self.learner.value.sigma)
        return out

    def _sync(self) -> None:
        if self.dev_l.type == "cuda":
            torch.cuda.synchronize()

    def _micro_batch(self, sub: list, L: int, ro: Rollout, seq_states: list | None, adv_n: np.ndarray,
                     ret: np.ndarray, pol_mask: np.ndarray, use_critic: bool, n_pol: int, n_val: int,
                     with_bc: bool, stats: dict) -> tuple[dict, torch.Tensor]:
        """Forward and backward over the sequences ``sub`` = [(game, sequence index)]. Returns the loss
        terms (already divided by the minibatch's mask counts) and the last hidden state."""
        cfg, dev = self.cfg, self.dev_l
        take = lambda a: np.stack([a[i, k * L:(k + 1) * L] for i, k in sub])
        tt = lambda a, dt=torch.float32: torch.as_tensor(take(a), device=dev, dtype=dt)
        f = to_tensors({k: take(v) for k, v in ro.feat.items()}, dev)
        if seq_states is None:
            init = self.learner.initial_state(len(sub), dev)
        else:
            init = [state_to(stack_states([seq_states[k][b].index(torch.tensor([i])) for i, k in sub]), dev)
                    for b in range(len(seq_states[0]))]
        reset = tt(ro.reset, torch.bool)
        act, old_lp = tt(ro.act, torch.long), tt(ro.logp)
        A, Rt = tt(adv_n), tt(ret)
        m_pol, m_val = tt(pol_mask, torch.bool), tt(ro.valid, torch.bool)
        m_pol[:, :cfg.burn_in] = False
        m_val[:, :cfg.burn_in] = False
        autocast = torch.autocast(dev.type, dtype=torch.float16 if cfg.fp16 else torch.bfloat16,
                                  enabled=(cfg.bf16 or cfg.fp16) and dev.type == "cuda")

        self._sync()
        t0 = time.perf_counter()
        with autocast:
            logits, v, _, h = self.learner(f, init, reset, act=act)
        self._sync()
        stats["t_fwd"].append(time.perf_counter() - t0)
        stats["t_vis_fwd"].append(getattr(self.learner, "_t_vision", 0.0))
        lp, _ = log_prob_entropy(logits.float() / tt(ro.temp)[..., None, None], act)   # each member's own policy
        _, ent = log_prob_entropy(logits.float(), act)                                   # the shared policy (T = 1)
        # Bounded log-ratio: on masked tokens the learner can disagree with the stored log-prob by 90+
        # nats, and exp() would overflow to inf; inf * 0 (the mask) is NaN.
        ratio = torch.exp((lp - old_lp).clamp(-20.0, 20.0))
        kl_ref = torch.zeros((), device=dev)
        if self.ref is not None:
            # Unbiased per-token estimator (DeepSeekMath): exp(r) - r - 1, r = log pi_ref - log pi.
            with torch.no_grad(), autocast:
                ref_init = init if seq_states is not None else self.ref.initial_state(len(sub), dev)
                ref_logits, _, _, _ = self.ref(f, ref_init, reset, act=act)
            lp_ref, _ = log_prob_entropy(ref_logits.float(), act)
            r = (lp_ref - lp).clamp(-20.0, 20.0)
            kl_ref = torch.where(m_pol, torch.exp(r) - r - 1, 0.0).sum() / n_pol
        obj = torch.min(ratio * A, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * A)
        # Dual-clip PPO (Ye et al. 2020): for A < 0 the pessimistic min is unbounded in the ratio; cap
        # the loss at dual_clip * |A| so one stale token cannot dominate.
        obj = torch.where(A < 0, torch.maximum(obj, cfg.dual_clip * A), obj)
        pg = -obj
        pol = m_pol & torch.isfinite(pg)
        pg_l = torch.where(pol, pg, 0.0).sum() / n_pol
        ent_l = torch.where(pol, ent, 0.0).sum() / n_pol
        if use_critic:
            vf_l = ((v.float() - self.learner.value.normalize(Rt)).square() * m_val.float()).sum() / n_val
        else:
            vf_l = torch.zeros((), device=dev)
        loss = pg_l + cfg.vf_coef * vf_l - cfg.ent_coef * ent_l + cfg.kl_ref * kl_ref
        coef = cfg.bc_coef * cfg.bc_decay ** self.iteration
        if with_bc and self.demos is not None and coef > 1e-4:
            from .bc import bc_loss
            fd, ld, vd = self.demos.batch(cfg.bc_batch, L)
            bl, _ = bc_loss(self.learner, to_tensors(fd, dev), torch.as_tensor(ld, device=dev),
                            torch.as_tensor(vd, device=dev))
            loss = loss + coef * bl
            stats["bc"].append(float(bl))
        self._sync()
        t0 = time.perf_counter()
        self.scaler.scale(loss).backward()
        self._sync()
        stats["t_bwd"].append(time.perf_counter() - t0)
        with torch.no_grad():
            parts = {"pg": float(pg_l), "vf": float(vf_l), "ent": float(ent_l), "kl_ref": float(kl_ref),
                     "kl": float(torch.where(pol, (ratio - 1) - torch.log(ratio), 0.0).sum()) / n_pol,
                     "clipfrac": float(torch.where(pol, ((ratio - 1).abs() > cfg.clip).float(), 0.0).sum()) / n_pol}
        return parts, h.detach()

    def _optimizer_step(self, stats: dict) -> None:
        self._sync()
        t0 = time.perf_counter()
        self.scaler.unscale_(self.opt)
        with torch.no_grad():
            sq = [p.grad.float().norm() ** 2 for p in self.learner.vision.parameters() if p.grad is not None]
            stats["vision_grad_norm"].append(float(sum(sq)) ** 0.5)
        torch.nn.utils.clip_grad_norm_(self.learner.parameters(), self.cfg.max_grad_norm)
        scale0 = self.scaler.get_scale() if self.scaler.is_enabled() else 1.0
        self.scaler.step(self.opt)
        self.scaler.update()
        if self.scaler.is_enabled():
            # GradScaler skips a step whose gradients overflowed and halves its scale.
            stats["fp16_skipped"].append(float(self.scaler.get_scale() < scale0))
            stats["fp16_scale"] = float(self.scaler.get_scale())
        self._sync()
        stats["t_opt"].append(time.perf_counter() - t0)

    # ---- diagnostics ------------------------------------------------------------------------
    def rollout_metrics(self, ro: Rollout) -> dict:
        m = ro.valid & ro.acted
        a = ro.act[m]                                                   # [n, 4, 5]
        noop = (a[..., 0] == 1) & (a[..., 1] == 1) & (a[..., 2:].sum(-1) == 0) if a.size else np.zeros(0)
        think = [x for e in self.envs for x in e.think_ms[-ro.T:]]
        stale = ro.stale[ro.stale >= 0]
        return {
            "reward_sum": float(ro.rew[ro.valid].sum()),
            "reward_mean": float(ro.rew[ro.valid].mean()) if ro.valid.any() else 0.0,
            "noop_frac": float(noop.mean()) if noop.size else 0.0,
            "deficit_food": float(ro.deficits[..., 0].mean()),
            "deficit_karma": float(ro.deficits[..., 1].mean()),
            "deficit_breath": float(ro.deficits[..., 2].mean()),
            "think_p50_ms": float(np.percentile(think, 50)) if think else 0.0,
            "think_p95_ms": float(np.percentile(think, 95)) if think else 0.0,
            "frame_stale_p50": float(np.percentile(stale, 50)) if stale.size else -1.0,
            "frame_stale_max": float(ro.stale.max()),
            "frame_brightness": float(ro.brightness.mean()),
            "frame_black_frac": float((ro.brightness < 1.0).mean()),
        }

    def _path_metrics(self) -> dict:
        if not self._path_diag:
            return {}
        pd = np.array(self._path_diag, np.float64)
        fp, sp = pd[:, 0], pd[:, 1]
        return {"food_path_valid": float((fp >= 0).mean()),
                "food_path_p50": float(np.median(fp[fp >= 0])) if (fp >= 0).any() else -1.0,
                "shelter_path_valid": float((sp >= 0).mean()),
                "shelter_path_p50": float(np.median(sp[sp >= 0])) if (sp >= 0).any() else -1.0,
                "phi_mean": float(pd[:, 2].mean())}

    def _refit_forage(self) -> None:
        """Refit the food forecast (train/forage_fit.py) from the recent episode records before each
        collection; it stays fixed during the collection so the shaping telescopes. Records without
        the game's rows are skipped. With no data the previous table (or forage_table.json) stays."""
        from . import forage_fit as FF
        d = self.run_dir / "episodes"
        path = self.run_dir / "forage_table.json"
        recs = []
        if d.exists():
            lo = self.iteration - self.cfg.forage_window
            for p in d.glob("it*.npz"):
                try:
                    if int(p.name[2:6]) >= lo:
                        recs.append(p)
                except ValueError:
                    continue
        eps, cens = [], []
        for p in sorted(recs):
            try:
                z = np.load(p, allow_pickle=False)
                if "rows" not in z.files or not z["rows"].dtype.names:
                    continue
                eps.append(z["rows"])
                cens.append(str(z["ended"]) == "")     # cut, not ended by the cycle
            except (OSError, ValueError, KeyError):
                continue
        prev = R.FORAGE_TABLE
        if prev is None and path.exists():
            prev = FF.ForageTable.load(path)
        tab = prev
        if eps:
            tab = FF.fit(eps, self.cfg.gamma, censored=cens, prior=prev)
            tab.save(path)
        if tab is not None:
            R.set_forage_table(tab)
            self._forage_diag = {"forage_episodes": tab.episodes, "forage_bites": tab.bites,
                                 "forage_adj": tab.adj, "forage_held_full": tab.held[40], "forage_held_0": tab.held[0]}

    def _save_logits(self, ro: Rollout) -> None:
        """The iteration's policy outputs along its own rollout, for offline study of sampling schemes
        (scripts/sampling_study.py)."""
        if ro.logits is None:
            return
        frames = ro.feat.get("frame") if ro.feat else None
        np.savez_compressed(self.run_dir / "logits_last.npz", logits=ro.logits, temp=ro.temp,
                            acted=ro.acted & ro.valid, act=ro.act.astype(np.int8),
                            frames=frames[0, ::64] if frames is not None else np.zeros(0))

    def _keep_snapshots(self, rec: dict) -> None:
        """Rollback points: ckpt-best.pt (best food_need_met, a diagnostic choice that training does
        not react to) and ckpt-iterNNNN.pt every snapshot_every iterations (the last 5 plus milestones)."""
        cfg = self.cfg
        fn = (rec.get("curriculum") or {}).get("food_need_met")
        if fn is not None and fn > self.best_food:
            self.best_food = fn
            shutil.copyfile(self.run_dir / "ckpt.pt", self.run_dir / "ckpt-best.pt")
        if cfg.snapshot_every and self.iteration % cfg.snapshot_every == 0:
            shutil.copyfile(self.run_dir / "ckpt.pt", self.run_dir / f"ckpt-iter{self.iteration:04d}.pt")
            keep = cfg.snapshot_keep_every
            snaps = [p for p in sorted(self.run_dir.glob("ckpt-iter*.pt"))
                     if not (keep and int(p.stem[len("ckpt-iter"):]) % keep == 0)]
            for old in snaps[:-5]:
                old.unlink()

    # ---- main loop --------------------------------------------------------------------------
    def run(self) -> None:
        cfg = self.cfg
        grpo = cfg.algo == "grpo"
        self.start()
        maxent = max_entropy()
        while self.iteration < cfg.iterations:
            t0 = time.time()
            self.clips.new_iteration()
            self._refit_forage()
            tau_used = self._tau_max
            if grpo:
                ro, adv, seq_states = self.collect_group()
                if cfg.member_temps:
                    self._fit_tau_max(ro)          # for the next iteration
                self._save_logits(ro)
            else:
                ro, seq_states = self.collect()
                adv = None
            t1 = time.time()
            for e in self.envs:
                e.keepalive(True)                  # the games stay frozen in lockstep during the update
            self._phase("update")
            try:
                upd = self.update(ro, seq_states, adv, cfg.group_tokens if grpo else None)
            finally:
                self._phase("collect")
                for e in self.envs:
                    e.keepalive(False)
            t2 = time.time()

            rec = {"iter": self.iteration, "stage": self.curriculum.stage,
                   "tokens": int(ro.n * ro.T), "steps_per_s": ro.n * ro.T / max(t1 - t0, 1e-9),
                   "collect_s": t1 - t0, "update_s": t2 - t1, **self.rollout_metrics(ro), **upd}
            rec["entropy_frac"] = rec.get("ent", 0.0) / maxent
            if grpo:
                change = ro.action_change_rate()
                if change is not None:
                    rec["action_change_frac"] = change
                    rec["refills"] = self._refills
                if cfg.member_temps:
                    rec["tau_max"], rec["tau_max_next"] = tau_used, self._tau_max
                if self.zero_adv_frac:
                    rec["zero_adv_frac"] = float(np.mean(self.zero_adv_frac))
            rec.update(self._path_metrics())
            if rec["frame_black_frac"] > 0.5:
                # A locked or blanked session stops the game drawing, and the agent goes blind with no error.
                rec["ALARM_FRAMES"] = f"{rec['frame_black_frac']:.0%} of frames black"
                print(f"!! FRAME ALARM: {rec['frame_black_frac']:.0%} of frames are black: is the session locked?")
            if rec["entropy_frac"] < cfg.entropy_alarm:
                rec["ALARM"] = f"entropy {rec['entropy_frac']:.2f} of max"
                print(f"!! ENTROPY ALARM: {rec['entropy_frac']:.2%} of maximum")
            rec.update(self._forage_diag)
            if self.iteration % cfg.eval_every == 0:
                cur = self.curriculum.evaluate(self.finished)
                rec["curriculum"] = cur
                self.finished = []
                if cur.get("event"):
                    print(f"curriculum: {cur['event']}")
            np.savez_compressed(self.run_dir / "drive_trace_last.npz", deficits=ro.deficits[0],
                                reward=ro.rew[0], valid=ro.valid[0])
            self.log_f.write(json.dumps(rec, default=float) + "\n")
            self.log_f.flush()
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()
                              if k != "curriculum"}, default=str))
            self.iteration += 1
            self.save()
            self._keep_snapshots(rec)


def _try_compile(agent: Agent, label: str) -> None:
    """torch.compile the stateless pieces (vision, body GLU, head GLUs). The recurrent mixers stay eager:
    their Python loops are data-dependent. Module.compile works in place, so state_dict keys are
    unchanged. Off by default: Triton is unreliable on gfx1030."""
    for mod in (agent.vision, agent.tokenizer.body, agent.actor_glu, agent.slot_glu, agent.critic_glu):
        mod.compile(dynamic=True)
    print(f"[{label}] torch.compile enabled for vision, body GLU and head GLUs")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for f in PPOConfig.__dataclass_fields__.values():
        typ = f.type if not isinstance(f.type, str) else eval(f.type)
        if typ is bool:
            ap.add_argument(f"--{f.name.replace('_', '-')}", action=argparse.BooleanOptionalAction, default=f.default)
        else:
            ap.add_argument(f"--{f.name.replace('_', '-')}", type=typ, default=f.default)
    ap.add_argument("--fake", action="store_true", help="serve FakeGame instances instead of Rain World")
    ap.add_argument("--resume", default="", help="checkpoint to resume from")
    args = ap.parse_args(argv)
    cfg = PPOConfig(**{k: getattr(args, k) for k in PPOConfig.__dataclass_fields__})
    vocab = O.Vocab()
    if args.fake:
        from ..env.fake_game import FakeGame
        for i in range(cfg.instances):
            FakeGame(cfg.base_port + i, seed=cfg.seed + i).start()
    envs = [RainEnv(instance=i, port=cfg.base_port + i, vocab=vocab) for i in range(cfg.instances)]
    tr = Trainer(cfg, envs)
    if args.resume:
        tr.load(Path(args.resume))
    elif cfg.init:
        tr.init_from(Path(cfg.init))
    if cfg.kl_ref > 0:
        if not cfg.init:
            raise SystemExit("--kl-ref needs --init (the reference policy), also when resuming")
        tr.load_ref(Path(cfg.init))
    if cfg.bc_coef > 0:
        tr.load_demos()
    tr.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
