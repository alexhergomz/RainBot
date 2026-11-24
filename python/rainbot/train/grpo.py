"""GRPO collection: groups of members that start from one seeded state.

Every game resets with the same seeded spec, so step t of every member is comparable. The advantage
is the per-step group-normalised discounted return-to-go:

    A[m, t] = (G[m, t] - mean_j G[j, t]) / (std_j G[j, t] + eps)

There is no critic: the group is the baseline. A group whose members all did the same thing gets
zero advantage and no gradient.

A pass is a time budget of L tokens (the cycle plus the rain's longest time to kill, so a member that
starts with the pass always finishes its night). Every game whose member finishes resets to the same
start and plays another member (a refill) until the budget runs out. Members still alive then are cut
and left out, because their return is incomplete. No game waits for another. Members are compared by
their own episode step.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import reward as R
from ..env.rain_env import NEUTRAL
from ..nn.agent import max_entropy, to_wire
from .curriculum import EpisodeStats
from .episodes import new_record
from .rollout import (Member, Rollout, clone_state, death_cause, fit_tau_max, group_advantages, has_event,
                      member_tau, state_to, stomach)


@dataclass
class _Game:
    """One game's state during a pass."""

    member: Member
    food0: float              # stomach at the previous token, for counting pips eaten
    prev_last: object         # the previous token's last row
    resetting: bool = False   # a refill's reset is in flight
    fresh: bool = False       # the next token starts a new member
    ep_before: int = 0        # the mod's episode id when the refill was requested
    cycle_ended: bool = False
    done: bool = False


class GroupCollector:
    """GRPO collection, mixed into the Trainer (uses its envs, tracks, actor, curriculum, clips)."""

    # ---- per-member temperature -----------------------------------------------------------------
    def _tau(self, i: int) -> float:
        cfg = self.cfg
        if cfg.algo != "grpo" or not cfg.member_temps or self._tau_max <= 1.0:
            return 1.0
        reps = max(1, cfg.group_repeats)
        return member_tau(self._tau_max, i * reps + self._rep, len(self.envs) * reps)

    def _fit_tau_max(self, ro: Rollout) -> None:
        if ro.logits is None:
            return
        tau = fit_tau_max(ro.logits, ro.valid & ro.acted, self.cfg.entropy_alarm * max_entropy())
        if tau is not None:
            self._tau_max = tau

    # ---- collection -------------------------------------------------------------------------------
    def collect_group(self) -> tuple[Rollout, np.ndarray, list]:
        """Collect this iteration's groups: (rollout [N, T], advantages [N, T], actor state at every
        group_tokens chunk start)."""
        cfg, N = self.cfg, len(self.envs)
        reps = max(1, cfg.group_repeats)
        for e in self.envs:                    # rooms with water, from the mod's survey
            if getattr(e, "water_rooms", None) is not None:
                self.curriculum.water_rooms = set(e.water_rooms)
                break
        groups = self._plan_groups()
        T = sum(g[3] for g in groups) * reps
        ro = Rollout(N, T)
        adv = np.zeros((N, T), np.float32)
        seq_states: list = []
        self.zero_adv_frac, self._refills, self._path_diag = [], 0, []
        off = 0
        for g, (stage, spec, rung, L) in enumerate(groups):
            passes: list = []
            for rep in range(reps):
                self._rep = rep
                got = self._group_reset(spec, stage, rung)
                if got[1] != spec:             # a failed reset swapped the start: a new group from here
                    self._finish_group(passes, adv, ro)
                    passes = []
                    stage, spec, rung = got
                members = self._play_pass(ro, seq_states, off, L, g, stage, spec, rung)
                self._episodes.save(members, spec=spec, stage=stage, rung=rung, iteration=self.iteration,
                                    tag_prefix=f"it{self.iteration:04d}_g{g * reps + rep}", ro=ro, off=off,
                                    start_tile=self._start_tile)
                ro.reset[:, off] = True
                passes.append((off, L, members))
                off += L
            self._finish_group(passes, adv, ro)
        return ro, adv, seq_states

    def _plan_groups(self) -> list[tuple[str, str, int, int]]:
        """The iteration's groups as (stage, spec, rung, length), decided up front so the rollout can be
        allocated. Groups are added up to groups_per_update x group_tokens tokens; one whole-cycle group
        can fill that alone."""
        budget, groups = self.cfg.groups_per_update * self.cfg.group_tokens, []
        while not groups or sum(g[3] for g in groups) < budget:
            stage, spec, rung = self._new_spec(self.envs[0].world)
            L = self.curriculum.group_tokens(stage, spec, self.cfg.group_tokens)
            if groups and sum(g[3] for g in groups) + L > budget:
                break
            groups.append((stage, spec, rung, L))
        return groups

    def _finish_group(self, passes: list, adv: np.ndarray, ro: Rollout) -> None:
        if not passes:
            return
        frac = group_advantages(ro.rew, passes, self.cfg.gamma, adv)
        self.zero_adv_frac.append(1.0 if frac is None else frac)

    # ---- group reset ------------------------------------------------------------------------------
    def _group_reset(self, spec: str, stage: str, rung: int) -> tuple[str, str, int]:
        """Reset every game with the same seeded spec and wait until all are in the new episode.
        A failed reset retries with a new spec, up to five times. Returns the start that was used."""
        for e in self.envs:
            e.keepalive(False)
        for _ in range(5):
            self._settle_all()
            for e in self.envs:
                e.request_reset(spec)
            res = list(self.pool.map(self._wait_reset, range(len(self.envs))))
            if all(r == "ok" for r in res):
                if not self._memreport_sent:
                    self._memreport_sent = True      # each game logs where its memory goes, once per run
                    for e in self.envs:
                        e.command("memreport")
                for tr in self.tracks:
                    tr.stage, tr.rung = stage, rung
                return stage, spec, rung
            print(f"group reset: {res}; retrying with a new spec")
            stage, spec, rung = self._new_spec(self.envs[0].world)
        raise RuntimeError("group reset failed five times")

    def _settle_all(self) -> None:
        """Step every game with neutral input until all have a living slugcat in a dry cycle.

        An episode can end with the slugcat dead or in the rain, and the mod does not reset a dead
        slugcat. Every game keeps stepping until all are ready at once: a game left idle while another
        one loads its save loses its link. Keepalive pings cover rows that ask for no action (menus).
        """
        ready = [False] * len(self.envs)

        def settle(i):
            e = self.envs[i]
            e.keepalive(True)
            for _ in range(20000):
                s = self.tracks[i].step
                last = s.rows[s.n_real - 1]
                ready[i] = bool(int(last["alive"]) and not int(last["rain_hit"]) and s.needs_action)
                if all(ready):
                    return
                if s.needs_action:
                    e.send_actions(NEUTRAL)
                self.tracks[i].step = e.recv_step()
        list(self.pool.map(settle, range(len(self.envs))))
        for e in self.envs:
            e.keepalive(False)

    def _wait_reset(self, i: int) -> str:
        """Step game i until its requested reset lands: "ok", "failed" or "timeout"."""
        e, ep0 = self.envs[i], int(self.tracks[i].step.rows[0]["episode"])
        for _ in range(4000):
            if self.tracks[i].step.needs_action:
                e.send_actions(NEUTRAL)
            s = self.tracks[i].step = e.recv_step()
            for ev in s.events:
                if ev.get("type") == "reset_failed":
                    reason = str(ev.get("reason", ""))
                    if reason.startswith("no climb-free start in "):
                        # No start in this room reaches a shelter without a pole climb: the ladder's
                        # no-climb rungs stop drawing it.
                        self.curriculum.climb_rooms.add(reason.rsplit(" ", 1)[-1])
                    return "failed"
            for ev in s.events:
                if ev.get("type") == "reset_done" and ev.get("tile") is not None:
                    self._start_tile[i] = list(ev["tile"])      # for exact replays
            if int(s.rows[0]["episode"]) != ep0:
                return "ok"
        return "timeout"

    # ---- one pass -----------------------------------------------------------------------------------
    def _play_pass(self, ro: Rollout, seq_states: list, off: int, L: int, g: int,
                   stage: str, spec: str, rung: int) -> list[Member]:
        """Play one pass of L tokens into ro[:, off:off + L], with refills. Returns its members."""
        N, C = len(self.envs), self.cfg.group_tokens
        self.state = self.actor.initial_state(N, self.dev_a)
        members = [Member(i, stats=EpisodeStats(stage=stage, rung=rung), rec=new_record()) for i in range(N)]
        games = [_Game(members[i], stomach(tr.step.rows[-1]), tr.step.rows[-1]) for i, tr in enumerate(self.tracks)]
        for k in range(L):
            t = off + k
            if k % C == 0:
                seq_states.append([state_to(clone_state(x), "cpu") for x in self.state])
            if all(gm.done for gm in games) and not any(gm.resetting for gm in games):
                ro.valid[:, t] = False
                ro.acted[:, t] = False
                continue
            for i, (tr, gm) in enumerate(zip(self.tracks, games)):
                tr.step.episode_start = k == 0 or gm.fresh
                ro.valid[i, t] = not gm.done and not gm.resetting
            a = self._act(list(range(N)), ro, t)
            futures = {}
            for i, (tr, gm) in enumerate(zip(self.tracks, games)):
                gm.fresh = False
                s = tr.step
                if gm.resetting:                # stepping a refill's reset in: neutral input, not data
                    ro.acted[i, t] = False
                    futures[i] = self.pool.submit(self._advance, i, NEUTRAL if s.needs_action else None)
                elif gm.done:
                    ro.acted[i, t] = False
                else:
                    ro.acted[i, t] = s.needs_action
                    wire = to_wire(a[i]).numpy().astype(np.int8)
                    gm.member.rec["wire"].append(wire if s.needs_action else np.zeros_like(wire))
                    gm.member.rec["acted"].append(bool(s.needs_action))
                    futures[i] = self.pool.submit(self._advance, i, wire if s.needs_action else None)
            for i, fu in futures.items():
                self._receive(i, fu.result(), ro, t)
                gm = games[i]
                if gm.resetting:
                    self._refill_started(i, gm, k, members, stage, rung)
                    continue
                finished = self._book_token(i, gm, ro, t, k, L, g, stage, rung)
                if finished and k < L - 2:
                    # Refill. Requested after the token's bookkeeping: that token carries the night's booking.
                    gm.ep_before = int(self.tracks[i].step.rows[0]["episode"])
                    self.envs[i].keepalive(False)
                    self.envs[i].request_reset(spec)
                    gm.resetting = True
        for m in members:
            if m.n is None:
                # Still playing when the budget ran out: cut. A member that ran the whole budget is only
                # possible without a rain timer (stage F); there the budget is the designed segment cut.
                m.n, m.cut = L - m.s, m.s > 0
                if m.cut:
                    ro.valid[m.i, off + m.s:off + L] = False
        for i, gm in enumerate(games):
            if gm.resetting:                    # a refill that never got to play
                self.envs[i].keepalive(True)
        self.finished.extend(m.stats for m in members if not m.cut)
        self.clips.flush()
        self._refills += sum(m.refill and not m.cut for m in members)
        return members

    def _refill_started(self, i: int, gm: _Game, k: int, members: list, stage: str, rung: int) -> None:
        s = self.tracks[i].step
        if has_event(s, "reset_failed"):
            gm.resetting = False                # no refill on this game this pass
            self.envs[i].keepalive(True)
        elif int(s.rows[0]["episode"]) != gm.ep_before and s.needs_action:
            m = Member(i, s=k + 1, refill=True, stats=EpisodeStats(stage=stage, rung=rung), rec=new_record())
            members.append(m)
            gm.member, gm.resetting, gm.done, gm.cycle_ended, gm.fresh = m, False, False, False, True
            last = s.rows[s.n_real - 1]
            gm.prev_last, gm.food0 = last, stomach(last)

    def _book_token(self, i: int, gm: _Game, ro: Rollout, t: int, k: int, L: int, g: int,
                    stage: str, rung: int) -> bool:
        """Book game i's token k into its member's stats and record. Returns whether the member finished."""
        cfg, s = self.cfg, self.tracks[i].step
        m, st, rec = gm.member, gm.member.stats, gm.member.rec
        finished = False
        # A member finishes on the first token after its cycle ends: that token's reward carries what
        # the game books on the sleep or death screen (karma, the stomach spent). Its game then stays
        # frozen in lockstep until the next reset, which can take minutes, so its link needs pings.
        ends = [ev for ev in s.events if ev.get("type") == "session_end"]
        if ends:
            gm.cycle_ended = True
            rec["ended"] = str(ends[-1].get("reason", ""))
        elif gm.cycle_ended:
            gm.done = finished = True
            m.n = k - m.s + 1
            self.envs[i].keepalive(True)
        if k == L - 1 and not finished and cfg.shaping and not ro.next_start[i, t]:
            # Cut at the budget, with no bootstrap: close the shaping with Phi = 0. The shaped return then
            # telescopes to -Phi(start), which every member shares and the normalisation removes. A cycle
            # end keeps Phi: the forecast on waking is part of the objective.
            ro.rew[i, t] -= cfg.gamma * float(R.potential(s.rows[s.n_real - 1], cfg.gamma))

        last, prev = s.rows[-1], gm.prev_last
        st.tokens += 1
        st.ticks += s.n_real
        st.rooms.add(int(last["room"]))
        st.sleeps += s.sleeps
        if s.sleeps and stomach(last) >= float(last["food_to_hibernate"]):
            st.fed_sleeps += 1
        st.deaths += s.deaths
        st.food_to_hibernate = int(last["food_to_hibernate"]) or st.food_to_hibernate
        st.cycle_length = int(last["cycle_length"]) or st.cycle_length
        if s.sleeps and not int(last["rain_hit"]):
            st.sheltered_before_rain += 1
        if s.deaths:
            cause = death_cause(prev)
            st.death_causes[cause] = st.death_causes.get(cause, 0) + 1
        if float(last["air_in_lungs"]) <= 0.0:
            st.drowned = 1
        if int(last["room"]) == int(prev["room"]):
            moved = float(np.linalg.norm(np.asarray(last["pos0"], np.float64)
                                         - np.asarray(prev["pos0"], np.float64))) / 20.0     # tiles
            st.displacement += moved
            st.idle_tokens += moved < 0.1
        if "food_path" in (last.dtype.names or ()):
            self._path_diag.append((int(last["food_path"]), int(last["shelter_path"]),
                                    float(R.potential(last, cfg.gamma))))

        tag = f"it{self.iteration:04d}_g{g}_env{i}_t{k:03d}"
        event = None
        if s.sleeps:
            event = f"hibernate_{stage}"
            self.clips.mark(i, f"{tag}_{event}", priority=True)
        elif int(last["in_shelter"]) and not int(prev["in_shelter"]) and \
                int(last["food_in_stomach"]) >= int(last["food_to_hibernate"]):
            event = f"shelter_fed_{stage}"
            self.clips.mark(i, f"{tag}_{event}", priority=True)
        gm.prev_last = last
        food = stomach(last)
        if st.need < 0:                         # pips the cycle starts short of, at its first token
            st.need = max(0.0, float(last["food_to_hibernate"]) - food)
        self.clips.push(i, s.frame)
        # Pips eaten: the stomach rises. Not across a session boundary (a reset sets the stomach, a death
        # rewinds it), except a sleep: its booking eats the food carried into the shelter.
        ended = has_event(s, "session_start") or any(
            ev.get("type") == "session_end" and ev.get("reason") != "sleep" for ev in s.events)
        if food > gm.food0 and not ended:
            st.pips += food - gm.food0
            self.clips.mark(i, f"{tag}_pip_{stage}{rung}")
            event = event or f"pip_{stage}{rung}"
        gm.food0 = food
        rec["pos"].append(np.asarray(last["pos0"], np.float32))
        rec["room"].append(int(last["room"]))
        rec["food"].append(food)
        rec["rows"].append(last.copy())         # the game's full state at the token (what Phi reads)
        if event is not None:
            rec["events"].append(f"{k}:{event}")
        return finished
