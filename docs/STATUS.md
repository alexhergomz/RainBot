# Status

## Current run

- **Run.** GRPO from the keyframe-weighted behaviour clone, with the current reward.
- **Progress.** Rungs 0 to 2 of the ladder pass: food, eating speed and a shelter in reach.
- **Next.** The first climbing rung (start within 6 tiles of the shelter pipe) is in progress.

Restart command:

    tools/run_training.sh --instances 4 --run-dir runs/<run> --stall-timeout 1800 --stage L \
        --init runs/bc-kf/bc.pt --lr 2e-4 --sgd-lr 5e-4 --ent-coef 0 --groups-per-update 2 \
        --group-repeats 2 --no-fp16 --accum-seqs 1 --no-vision-checkpoint --epochs 2 --vision-grad-frac 0.5

Watchers:

    tail -F runs/<run>/metrics.jsonl | python3 -u tools/fmt_metrics.py
    python3 -u tools/watch_games.py runs/<run> 4

## Open problem: leaving a pole

Every start in a climbing room fails at the last 8 to 10 tiles up to the shelter.

- **What the slugcat does.** It reaches the pole below the shelter and holds the direction on it for
  most of the cycle. It never jumps.
- **What the game needs.** On a vertical pole, up only climbs. To leave the pole toward a ledge, the
  player holds the direction and then jumps.
- **Next step.** Count direction + jump presses while body_mode is ClimbingOnBeam, for the agent and
  for the human recordings. Then decide whether the near-shelter climbing rung is enough, or whether
  starts must be at the top of the pole.

## Known limits

- **One start per iteration.** About half the iterations cannot be judged for the pass rule (fed
  starts, replays of earlier rungs).
- **Temperatures stay spread.** Equal-temperature groups made no progress, so the temperature spread
  is kept.
- **Determinism.** The game is not fully deterministic, so long open-loop replays can drift.

## Ideas noted, not built

- Running standardization of the deficits.
- A learned hold time for actions (TempoRL-like), if jitter returns.
- Jump links painted as their measured arc in the distance channel.

## Possible bugs to check

- **Fake game.** The fake slugcat can never jump: `FakeGame._serve` writes the action into
  `history[0]` before `tick_once` tests `history[0]` for the press edge.
- **Clip ring.** `FrameCapture.ClipPush` does not reset the ring head and count when the screen size
  changes while a clip is armed.
- **Two start filters.** `ShelterWithoutClimb` counts pipes into rooms next to a shelter as goals.
  `ShelterDistance` (used by `shelter_within`) counts only pipes straight into a shelter.
- **Old capture check.** `tools/check_capture.py` reads the v6 field names `prev_action` and
  `has_controller`.
