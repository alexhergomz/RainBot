"""The whole network: temporal tokenizer, hybrid backbone, actor and critic heads.

    8 ticks of rows --tokenizer--> 1 token (d=256) --[KDA,KDA,KDA,SWA]x2--> actor: 4 decisions
                                                                         critic: 1 value

One token = one action chunk = one PPO step. The trunk is shared by actor and critic.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .. import obs as O
from ..proto import ACTION_DIMS
from .action_head import ActionHead
from .kda import KDA
from .layers import GLU, ConvResidual, DWGLU1d, LayerScale, ScaleNorm
from .swa import SWA
from .vision import Vision
from .xattn import XAttn

#: Factored action ``ACTION_DIMS``: x (3), y (3), jmp (2), thrw (2), pckp (2), in wire order.
N_LOGITS = sum(ACTION_DIMS)


@dataclass
class AgentConfig:
    d_model: int = 256
    tick_c: int = 64               # per-tick width C; token = Linear(8C -> d_model)
    vision_d: int = 128
    room_dim: int = 32
    action_dim: int = 16
    pattern: tuple = ("kda", "kda", "kda", "swa") * 2
    kda_heads: int = 4
    kda_head_dim: int = 64
    swa_q_heads: int = 8
    swa_kv_heads: int = 2
    swa_head_dim: int = 32
    window: int = 128
    sinks: int = 4
    glu_order: int = 2             # 1 = the paper's first-order xATGLU fallback
    adaln_room: bool = False       # optional component, off by default
    checkpoint_vision: bool = True  # activation checkpointing for the vision stack when training
    vision_micro_batch: int = 64    # frames per checkpointed vision chunk (160x96: ~1 GB per 256)
    vision_grad_frac: float = 1.0   # training: share of frames whose CNN pass keeps a graph (the rest no_grad)
    frame_hw: tuple = (O.FRAME_H, O.FRAME_W)
    frame_c: int = O.FRAME_C          # RGB + the movement-graph distance field (protocol v11)
    use_entities: bool = True      # entity tokens alongside the image tokens in vision
    xattn_window: int = 4          # frames the cross-attention sees (current + 3 previous)
    xattn_on: tuple = ("swa",)     # which block kinds get a cross-attention sublayer
    decisions: int = O.DECISIONS
    # Autoregressive action head: causal conv dilations over the decision timeline
    # (nn/action_head.py); () = independent slots, no action head.
    action_dilations: tuple = (1, 2)
    # The head's history is its own chunk only. Earlier chunks' decisions already reach the trunk as
    # consumed actions, and a cross-chunk path lets a cloned policy copy its previous action.
    # Dilations 1, 2 give a receptive field of 4 decisions = one chunk.
    action_chunk_only: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


class RunningNorm(nn.Module):
    """Running mean/variance observation normalisation, clipped to +-5. Frozen unless ``update``d."""

    def __init__(self, dim: int, clip: float = 5.0):
        super().__init__()
        self.clip = clip
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.reshape(-1, x.shape[-1]).float()
        n = x.shape[0]
        if n == 0:
            return
        m, v = x.mean(0), x.var(0, unbiased=False)
        tot = self.count + n
        delta = m - self.mean
        self.mean += delta * n / tot
        self.var = (self.var * self.count + v * n + delta.square() * self.count * n / tot) / tot
        self.count = tot

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ((x - self.mean) / (self.var + 1e-8).sqrt()).clamp(-self.clip, self.clip)


class Tokenizer(nn.Module):
    """8 ticks -> 1 token."""

    def __init__(self, cfg: AgentConfig):
        super().__init__()
        C = cfg.tick_c
        self.cfg = cfg
        cap = O.CAPACITY
        self.body_norm = RunningNorm(O.BODY_DIM)
        self.needs_norm = RunningNorm(O.NEEDS_DIM)
        self.ent_norm = RunningNorm(O.N_ENT_CONT)
        self.anim = nn.Embedding(cap["animation"], 16)
        self.mode = nn.Embedding(cap["body_mode"], 8)
        self.grasp = nn.Embedding(cap["object_type"], 8)
        self.body = GLU(O.BODY_DIM + 16 + 8 + 16, C, hidden=2 * C, order=cfg.glu_order)
        self.action = nn.ModuleList([nn.Embedding(n, cfg.action_dim) for n in ACTION_DIMS])
        self.room = nn.Embedding(cap["room"], cfg.room_dim)
        tick_in = C + O.NEEDS_DIM + cfg.action_dim + cfg.room_dim
        self.tick_proj = nn.Linear(tick_in, C, bias=False)
        self.mix = nn.ModuleList([ConvResidual(C, DWGLU1d(C, 3)) for _ in range(2)])
        self.to_token = nn.Linear(O.TICKS * C, cfg.d_model, bias=False)
        self.entered_room = nn.Parameter(torch.randn(cfg.d_model))        # N(0,1), like nn.Embedding

    def norms(self):
        return {"body": self.body_norm, "needs": self.needs_norm, "ent_cont": self.ent_norm}

    def forward(self, f: dict) -> torch.Tensor:
        B, T = f["body"].shape[:2]
        body = torch.cat([self.body_norm(f["body"]), self.anim(f["animation"]), self.mode(f["body_mode"]),
                          self.grasp(f["grasp"]).flatten(-2)], dim=-1)
        act = sum(e(f["action"][..., i]) for i, e in enumerate(self.action))
        x = torch.cat([self.body(body), self.needs_norm(f["needs"]), act, self.room(f["room"])], -1)
        x = self.tick_proj(x)                                       # [B, T, 8, C]
        x = x.reshape(B * T, O.TICKS, -1).transpose(1, 2)           # [BT, C, 8]
        for blk in self.mix:
            x = blk(x)
        tok = self.to_token(x.transpose(1, 2).reshape(B, T, -1))    # time-to-channel unshuffle
        entered = f["room_changed"].any(-1, keepdim=True).to(tok.dtype)
        return tok + entered * self.entered_room

    def token_room(self, f: dict) -> torch.Tensor:
        return f["room"][..., -1]


class Block(nn.Module):
    """Pre-norm residual pair: x += g1 * Mixer(SN(x)); x += g2 * GLU(SN(x)). Optional AdaLN-on-room."""

    def __init__(self, kind: str, cfg: AgentConfig):
        super().__init__()
        d = cfg.d_model
        self.kind = kind
        self.mixer = (KDA(d, cfg.kda_heads, cfg.kda_head_dim) if kind == "kda" else
                      SWA(d, cfg.swa_q_heads, cfg.swa_kv_heads, cfg.swa_head_dim, cfg.window, cfg.sinks))
        self.n1, self.n2 = ScaleNorm(d), ScaleNorm(d)
        self.s1, self.s2 = LayerScale(d), LayerScale(d)
        self.ffn = GLU(d, order=cfg.glu_order)
        self.xattn = None
        if kind in cfg.xattn_on:
            n_tok = (cfg.frame_hw[0] // 16) * (cfg.frame_hw[1] // 16) + (O.N_ENTITIES if cfg.use_entities else 0)
            self.nx, self.sx = ScaleNorm(d), LayerScale(d)
            self.xattn = XAttn(d, cfg.vision_d, n_tok, cfg.swa_q_heads, cfg.swa_kv_heads, cfg.swa_head_dim,
                               cfg.xattn_window, grid_hw=(cfg.frame_hw[0] // 16, cfg.frame_hw[1] // 16))
        self.ada = None
        if cfg.adaln_room:
            self.ada = nn.Linear(cfg.room_dim, 4 * d)

    def initial_state(self, batch: int, device=None) -> list:
        """[mixer state] or [mixer state, cross-attention state]."""
        st = [self.mixer.initial_state(batch, device)]
        if self.xattn is not None:
            st.append(self.xattn.initial_state(batch, device))
        return st

    def forward(self, x, state, reset, room_emb=None, vis=None, frame_valid=None):
        mix_state = state[0]
        h1, h2 = self.n1(x), None
        if self.ada is not None:
            a1, b1, a2, b2 = self.ada(room_emb).chunk(4, -1)
            h1 = h1 * (1 + a1) + b1
        y, mix_state = self.mixer(h1, mix_state, reset)
        x = x + self.s1(y)
        new = [mix_state]
        if self.xattn is not None:
            # Pre-norm residual with LayerScale, QK-norm inside: x += g * XAttn(ScaleNorm(x)).
            y, xs = self.xattn(self.nx(x), vis, frame_valid, state[1], reset)
            x = x + self.sx(y)
            new.append(xs)
        h2 = self.n2(x)
        if self.ada is not None:
            h2 = h2 * (1 + a2) + b2
        x = x + self.s2(self.ffn(h2))
        return x, new


class PopArtValue(nn.Module):
    """Linear value output with PopArt normalisation (van Hasselt et al. 2016).

    The layer predicts a *normalised* value; ``mu, sigma`` track the return statistics and every
    statistics update rescales the weights so unnormalised predictions are preserved exactly.
    """

    def __init__(self, d: int, beta: float = 3e-4):
        super().__init__()
        self.lin = nn.Linear(d, 1)
        self.beta = beta
        self.register_buffer("mu", torch.zeros(()))
        self.register_buffer("nu", torch.ones(()))       # second moment

    @property
    def sigma(self) -> torch.Tensor:
        return (self.nu - self.mu.square()).clamp_min(1e-6).sqrt()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.lin(h).squeeze(-1)

    def denormalize(self, v: torch.Tensor) -> torch.Tensor:
        return v * self.sigma + self.mu

    def normalize(self, v: torch.Tensor) -> torch.Tensor:
        return (v - self.mu) / self.sigma

    @torch.no_grad()
    def update(self, targets: torch.Tensor) -> None:
        t = targets.float().flatten()
        if t.numel() == 0:
            return
        old_mu, old_sigma = self.mu.clone(), self.sigma.clone()
        self.mu.lerp_(t.mean(), self.beta)
        self.nu.lerp_(t.square().mean(), self.beta)
        s = self.sigma
        self.lin.weight.mul_(old_sigma / s)
        self.lin.bias.copy_((old_sigma * self.lin.bias + old_mu - self.mu) / s)


class Agent(nn.Module):

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Load a checkpoint, upgrading two older layouts.

        A 3-channel vision stem (before the distance channel) gets zero input weights for the new
        channel, so the loaded policy behaves exactly as before until training gives it weight."""
        sd = dict(state_dict)
        k = "vision.stem.proj.weight"
        if k in sd:
            w, cur = sd[k], self.vision.stem.proj.weight
            if w.shape[1] < cur.shape[1]:
                pad = torch.zeros(w.shape[0], cur.shape[1] - w.shape[1], *w.shape[2:], dtype=w.dtype, device=w.device)
                sd[k] = torch.cat([w, pad], 1)
        # A checkpoint without the autoregressive action head: the head starts fresh (behaviour
        # cloning then trains it); the rest of the policy loads unchanged.
        if self.action_head is not None and not any(k_.startswith("action_head.") for k_ in sd):
            for k_, v_ in self.action_head.state_dict().items():
                sd["action_head." + k_] = v_
        return super().load_state_dict(sd, strict=strict, assign=assign)

    def __init__(self, cfg: AgentConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or AgentConfig()
        d = cfg.d_model
        self.tokenizer = Tokenizer(cfg)
        cap = O.CAPACITY
        self.vision = Vision(O.n_entity_types(), cap["behavior"], O.N_ENT_CONT, d=cfg.vision_d,
                             frame_hw=tuple(cfg.frame_hw), use_entities=cfg.use_entities, frame_c=cfg.frame_c)
        self.blocks = nn.ModuleList([Block(k, cfg) for k in cfg.pattern])
        self.out_norm = ScaleNorm(d)
        # Actor: SN -> GLU -> 4x64 -> channel-to-time shuffle -> shared small GLU -> factored logits.
        self.actor_norm = ScaleNorm(d)
        self.actor_glu = GLU(d, order=cfg.glu_order)
        self.actor_proj = nn.Linear(d, cfg.decisions * 64, bias=False)
        # ScaleNorm before the slot GLU, like every other pre-norm block. Without it two second-order
        # GLUs are chained back to back, and a product-of-branches GLU squashes small inputs
        # quadratically: slot features reach the logits at ~1e-3 and the policy gradient is ~1000x
        # too small to move.
        self.slot_norm = ScaleNorm(64)
        self.slot_glu = GLU(64, order=cfg.glu_order)
        self.actor_logits = nn.Linear(64, N_LOGITS)
        self.action_head = ActionHead(64, tuple(cfg.action_dilations), cfg.glu_order) if cfg.action_dilations else None
        self.critic_norm = ScaleNorm(d)
        self.critic_glu = GLU(d, order=cfg.glu_order)
        self.value = PopArtValue(d)

    # ---- state ------------------------------------------------------------------------------
    def initial_state(self, batch: int, device=None) -> list:
        """Flat list of per-layer states (mixer states, then cross-attention states, per block)."""
        st = [s for b in self.blocks for s in b.initial_state(batch, device)]
        if self.action_head is not None:
            st.append(self.action_head.initial_state(batch, device))
        return st

    @staticmethod
    def detach_state(state: list) -> list:
        return [s.detach() for s in state]

    @staticmethod
    def index_state(state: list, idx) -> list:
        return [s.index(idx) for s in state]

    def encode_frames(self, f: dict) -> torch.Tensor:
        """Vision for every step's frame, batched across instances and time: [B, T, N, d_v]."""
        frame = f["frame"]                                           # [B, T, C, H, W] uint8
        B, T = frame.shape[:2]
        cuda = frame.is_cuda
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        flat = lambda t: t.reshape(B * T, *t.shape[2:])
        tk = self.tokenizer
        args = [flat(frame)]
        if self.cfg.use_entities:
            args += [flat(f["ent_kind"]), flat(f["ent_type"]), flat(f["ent_behavior"]),
                     tk.ent_norm(flat(f["ent_cont"]))]
        frac = self.cfg.vision_grad_frac
        if self.training and torch.is_grad_enabled() and frac < 1.0:
            # Vision gradient from a random subset of frames: every frame still goes through the CNN
            # and reaches the policy; only the chosen ones keep activations and send gradient into
            # the CNN. Consecutive frames are 8 ticks apart and nearly identical, so the subset keeps
            # most of the signal at a fraction of the backward cost.
            n = args[0].shape[0]
            k = max(1, int(round(frac * n)))
            perm = torch.randperm(n, device=frame.device)
            g_idx, ng_idx = perm[:k], perm[k:]
            c = self.cfg.vision_micro_batch * 4
            run = lambda idx: torch.cat([self.vision(*(a[idx[i:i + c]] for a in args))
                                         for i in range(0, len(idx), c)])
            out_g = run(g_idx)
            out = out_g.new_empty((n,) + out_g.shape[1:])
            out[g_idx] = out_g
            if len(ng_idx):
                with torch.no_grad():
                    out[ng_idx] = run(ng_idx).to(out.dtype)
        elif self.training and torch.is_grad_enabled() and self.cfg.checkpoint_vision:
            # Activations of B*T frames dominate learner memory; checkpoint in micro-batches.
            n, c = args[0].shape[0], self.cfg.vision_micro_batch
            out = torch.cat([torch.utils.checkpoint.checkpoint(
                self.vision, *(a[i:i + c] for a in args), use_reentrant=False) for i in range(0, n, c)])
        else:
            # Inference also runs in micro-batches: a reference-policy pass over a 4x128 minibatch
            # is 512 frames, and one CNN pass over all of them runs an 8 GB card out of memory.
            n, c = args[0].shape[0], self.cfg.vision_micro_batch * 4
            out = torch.cat([self.vision(*(a[i:i + c] for a in args)) for i in range(0, n, c)])
        if cuda:
            torch.cuda.synchronize()
        self._t_vision = time.perf_counter() - t0                    # update-time breakdown (ppo.py)
        return out.view(B, T, *out.shape[1:])

    # ---- forward ----------------------------------------------------------------------------
    def trunk(self, f: dict, state: list, reset: torch.Tensor):
        x = self.tokenizer(f)
        room = self.tokenizer.room(self.tokenizer.token_room(f)) if self.cfg.adaln_room else None
        fvalid = f["frame_valid"].bool()
        if "frame" in f and bool(fvalid.any()):
            vis = self.encode_frames(f)
        else:
            # No valid frame anywhere (vision-masked cloning, or before the first capture): skip the
            # CNN. The cross-attention masks invalid frames, so these zeros are never read.
            B, T = fvalid.shape
            n_tok = self.vision.n_tokens(O.N_ENTITIES)
            vis = x.new_zeros(B, T, n_tok, self.cfg.vision_d)
        new_state, k = [], 0
        for blk in self.blocks:
            n = 2 if blk.xattn is not None else 1
            x, st = blk(x, state[k:k + n], reset, room, vis, fvalid)
            new_state += st
            k += n
        return self.out_norm(x), new_state

    def _slot_logits(self, a: torch.Tensor) -> torch.Tensor:
        return self.actor_logits(self.slot_glu(self.slot_norm(a)))

    def heads(self, h: torch.Tensor, act: torch.Tensor | None = None, head_state=None, reset=None):
        # Always fp32: a very peaked policy's logits and their log-softmax gradients exceed fp16's
        # range, which makes every gradient non-finite under fp16 autocast. The heads are tiny, so
        # fp32 costs nothing.
        B, T, _ = h.shape
        with torch.autocast(device_type=h.device.type, enabled=False):
            h = h.float()
            a = self.actor_proj(self.actor_glu(self.actor_norm(h))).view(B, T, self.cfg.decisions, 64)
            if self.action_head is not None:
                if act is None:
                    raise ValueError("the autoregressive action head needs the decisions taken (act=); "
                                     "to choose decisions use Agent.act")
                a, head_state = self.action_head(a, act, head_state, self._head_reset(reset))
            logits = self._slot_logits(a)                                   # [B, T, 4, 12]
            v = self.value(self.critic_glu(self.critic_norm(h)))            # normalised value [B, T]
        return logits, v, head_state

    def _head_reset(self, reset: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(reset, dtype=torch.bool) if self.cfg.action_chunk_only else reset

    def _split_state(self, state: list):
        if self.action_head is None:
            return state, None
        return state[:-1], state[-1]

    def forward(self, f: dict, state: list, reset: torch.Tensor, act: torch.Tensor | None = None):
        """Teacher-forced: logits of the decisions ``act`` [B, T, 4, 5] were drawn from (each
        decision conditioned on the ones before it)."""
        bstate, hstate = self._split_state(state)
        h, state = self.trunk(f, bstate, reset)
        logits, v, hstate = self.heads(h, act, hstate, reset)
        if hstate is not None:
            state = state + [hstate]
        return logits, v, state, h

    @torch.no_grad()
    def act(self, f: dict, state: list, reset: torch.Tensor, tau=1.0, greedy: bool = False):
        """Choose one token's decisions ([B, 1] inputs). ``tau`` = temperature, a float or [B].
        Returns raw logits [B, 1, 4, 12] (each decision's, given the ones chosen before it),
        actions [B, 1, 4, 5], value, state, trunk features."""
        if self.action_head is None:
            logits, v, state, h = self.forward(f, state, reset)
            t = torch.as_tensor(tau, dtype=logits.dtype, device=logits.device).reshape(-1, 1, 1, 1)
            return logits, sample(logits / t, greedy), v, state, h
        bstate, hstate = self._split_state(state)
        h, state = self.trunk(f, bstate, reset)
        B, T, _ = h.shape
        with torch.autocast(device_type=h.device.type, enabled=False):
            hf = h.float()
            a = self.actor_proj(self.actor_glu(self.actor_norm(hf))).view(B, T, self.cfg.decisions, 64)
            t = torch.as_tensor(tau, dtype=torch.float32, device=h.device).reshape(-1, 1)

            def choose(x):
                lg = self._slot_logits(x)
                return lg, sample(lg / t, greedy)
            logits, acts, hstate = self.action_head.step(a, hstate, self._head_reset(reset), choose)
            v = self.value(self.critic_glu(self.critic_norm(hf)))
        return logits, acts, v, state + [hstate], h


# ---- factored categorical over a chunk of decisions ---------------------------------------------

def split_logits(logits: torch.Tensor):
    return torch.split(logits.float(), ACTION_DIMS, dim=-1)


def sample(logits: torch.Tensor, greedy: bool = False) -> torch.Tensor:
    """logits [..., 4, 12] -> actions [..., 4, 5] in {0..n-1} per factor."""
    parts = []
    for lg in split_logits(logits):
        if greedy:
            parts.append(lg.argmax(-1))
        else:
            parts.append(torch.distributions.Categorical(logits=lg).sample())
    return torch.stack(parts, dim=-1)


def log_prob_entropy(logits: torch.Tensor, actions: torch.Tensor):
    """Chunk log-probability (sum over 4 decisions x 5 factors) and entropy (same sum)."""
    lp, ent = 0.0, 0.0
    for i, lg in enumerate(split_logits(logits)):
        logp = F.log_softmax(lg, dim=-1)
        lp = lp + logp.gather(-1, actions[..., i:i + 1].long()).squeeze(-1).sum(-1)
        ent = ent - (logp.exp() * logp).sum(-1).sum(-1)
    return lp, ent


def max_entropy(decisions: int = O.DECISIONS) -> float:
    return decisions * sum(math.log(n) for n in ACTION_DIMS)


def to_wire(actions: torch.Tensor):
    """[..., 4, 5] indices -> wire values: x,y in {-1,0,1}; buttons {0,1}."""
    a = actions.clone()
    a[..., :2] -= 1
    return a


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# ---- numpy features -> device tensors ------------------------------------------------------------

FLOAT_KEYS = ("body", "needs", "ent_cont")


def to_tensors(f: dict, device) -> dict:
    """Featurizer output (numpy, any leading dims) -> torch on ``device``; frames stay uint8."""
    out = {}
    for k, v in f.items():
        t = torch.as_tensor(np.ascontiguousarray(v)) if isinstance(v, np.ndarray) else torch.as_tensor(v)
        if k == "frame":
            out[k] = t.to(device=device, dtype=torch.uint8)            # scaled to [0,1] in vision
        elif k in FLOAT_KEYS:
            out[k] = t.to(device=device, dtype=torch.float32)
        elif k in ("room_changed", "frame_valid"):
            out[k] = t.to(device=device, dtype=torch.bool)
        else:
            out[k] = t.to(device=device, dtype=torch.long)
    return out
