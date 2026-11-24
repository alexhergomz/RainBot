"""Run a policy (or a fixed action source) against one environment, outside training.

Used by the pre-training checks, the sanity checks and the probes. Unlike the trainer it gives the
caller control over when the recurrent state is reset (the memory probe keeps it across a
teleport), and it records the trunk features for linear probes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .. import obs as O
from ..env.rain_env import NEUTRAL, RainEnv, Step
from ..nn.agent import Agent, AgentConfig, to_tensors, to_wire


def load_agent(path: str | Path, device="cpu", vocab: O.Vocab | None = None) -> Agent:
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = AgentConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in ck["cfg"].items()})
    if "action_dilations" not in ck["cfg"]:
        cfg.action_dilations = ()           # saved before the autoregressive head: keep its old actor
    if "action_chunk_only" not in ck["cfg"]:
        cfg.action_chunk_only = False       # saved before chunk-only history
    m = Agent(cfg).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    if vocab is not None:
        vocab.load_state_dict(ck["vocab"])
    return m


@dataclass
class Trace:
    rows: list = field(default_factory=list)        # last real row per token
    h: list = field(default_factory=list)           # trunk features per token (policy runs only)
    events: list = field(default_factory=list)      # (token index, event)
    actions: list = field(default_factory=list)

    def array(self) -> np.ndarray:
        return np.stack(self.rows) if self.rows else np.zeros(0)


class PolicyRunner:
    def __init__(self, env: RainEnv, agent: Agent | None = None, greedy: bool = False,
                 action_fn: Callable[[int, Step], np.ndarray] | None = None, device="cpu", tau: float = 1.0):
        self.env, self.agent, self.greedy, self.action_fn, self.device = env, agent, greedy, action_fn, device
        self.tau = tau
        self.state = agent.initial_state(1, device) if agent is not None else None
        self.step: Step | None = None
        self.t = 0

    def reset_memory(self) -> None:
        if self.agent is not None:
            self.state = self.agent.initial_state(1, self.device)

    def first(self) -> Step:
        self.step = self.env.recv_step()
        return self.step

    @torch.no_grad()
    def act(self, reset_state: bool | None = None) -> tuple[np.ndarray, torch.Tensor | None]:
        s = self.step
        h = None
        if self.action_fn is not None:
            wire = np.asarray(self.action_fn(self.t, s), np.int8)
        else:
            feat = self.env.feat.chunk(s.rows[None], s.frame[None], np.array([s.frame_tick]))
            f = to_tensors({k: v[:, None] for k, v in feat.items()}, self.device)
            r = s.episode_start if reset_state is None else reset_state
            _, a, _, self.state, h = self.agent.act(f, self.state, torch.tensor([[r]], device=self.device),
                                                     tau=self.tau, greedy=self.greedy)
            wire = to_wire(a[0, 0]).cpu().numpy().astype(np.int8)
        return wire, h

    def advance(self, wire: np.ndarray | None, trace: Trace | None = None, h=None) -> Step:
        s = self.step
        if trace is not None:
            trace.rows.append(s.rows[s.n_real - 1].copy())
            trace.actions.append(wire)
            if h is not None:
                trace.h.append(h[0, 0].float().cpu().numpy())
            for e in s.events:
                trace.events.append((len(trace.rows) - 1, e))
        if s.needs_action:
            self.env.send_actions(wire if wire is not None else NEUTRAL)
        self.step = self.env.recv_step()
        self.t += 1
        return self.step

    def run(self, n: int, trace: Trace | None = None, until: Callable[[Step], bool] | None = None,
            reset_state: bool | None = None) -> Trace:
        trace = trace if trace is not None else Trace()
        for _ in range(n):
            wire, h = self.act(reset_state)
            self.advance(wire, trace, h)
            if until is not None and until(self.step):
                break
        return trace

    def reset(self, spec: str, max_wait: int = 2000, retries: int = 5) -> Step:
        """Request a reset and advance (neutral input) until its first chunk arrives.

        A reset is refused while the slugcat is dead; then this waits for the respawn (a new game
        session) and asks again.
        """
        for _ in range(retries):
            self.env.request_reset(spec)
            ep0 = int(self.step.rows[0]["episode"])
            failed = None
            for _ in range(max_wait):
                self.advance(NEUTRAL)
                if int(self.step.rows[0]["episode"]) != ep0:
                    return self.step
                failed = [e for e in self.step.events if e.get("type") == "reset_failed"]
                if failed:
                    break
            if not failed:
                raise TimeoutError("reset did not complete")
            # Dead (or between sessions): play neutral until alive again, then retry.
            for _ in range(max_wait):
                if int(self.step.rows[-1]["alive"]) and not any(
                        e.get("type") == "session_end" for e in self.step.events):
                    break
                self.advance(NEUTRAL)
        raise RuntimeError(f"reset failed {retries} times: {failed}")
