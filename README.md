# RainBot

An agent that learns to survive in Rain World by playing the real game.

- A BepInEx mod runs the game in lockstep with a Python trainer. It exports the game state and the
  rendered screen, and sets up randomised starts.
- One recurrent network (Kimi Delta Attention + sliding-window attention, a CNN on the screen)
  trains with GRPO on several game instances that run faster than real time.
- The reward is homeostatic. The agent keeps food, karma and breath near the game's own setpoints.

Documents:
- `docs/DESIGN.md`: the design and the code map.
- `docs/PROTOCOL.md`: the wire contract between the mod and Python.
- `docs/STATUS.md`: the current state and the open problems.

## Layout

    mod/RainBot/      the mod (netstandard2.0): lockstep, state export, frames, resets, path distances
    mod/Loader/       a small loader that hot-swaps the mod into running games
    python/rainbot/   obs.py, reward.py, optim.py, nn/ (network), env/ (game client, fake game),
                      train/ (trainer, GRPO, curriculum, behaviour cloning), scripts/ (checks, tools)
    python/tests/     unit tests (CPU only)
    tools/            launch, supervise and stop the games; watch a run

## Setup

Requirements:
- Rain World v1.11.8 with BepInEx 5.4.17 (Linux: under Proton).
- The .NET SDK, to build the mod.
- Python 3.12 and PyTorch (`python/requirements.txt`).

    ./tools/dump_signatures.sh     # reference signatures of the game's code, after any game update
    ./mod/build.sh                 # build and install into the game's BepInEx folder
    cd python && python -m venv .venv && .venv/bin/pip install -r requirements.txt

The game folder defaults to `~/.steam/steam/steamapps/common/Rain World`. Set `RAINWORLD_DIR` to
change it.

If the plugin never appears in `BepInEx/LogOutput.log`, check `BepInEx/backup/`. Rain World's Remix
loader moves plugins that are not whitelisted there at launch. `build.sh` adds the whitelist entry,
but the game must be restarted after the first install.

## Test without the game

    cd python
    .venv/bin/python -m pytest -q tests
    .venv/bin/python -m rainbot.train.ppo --fake --instances 3 --learner-device cpu --actor-device cpu

## Train on the game

Set `mode = drive` in `BepInEx/config/rainbot.bridge.cfg`. The mod listens on
`127.0.0.1:54321+instance` and does nothing until a client connects.

    ./tools/launch_instances.sh 4                  # N isolated instances, ports 54321+i
    cd python
    .venv/bin/python -m rainbot.scripts.checks physics       # speed-up keeps the physics
    .venv/bin/python -m rainbot.scripts.checks determinism   # same seed and actions, same result
    .venv/bin/python -m rainbot.scripts.checks throughput --instances 4
    .venv/bin/python -m rainbot.scripts.checks noop          # a NO-OP policy must score negative
    cd ..
    ./tools/run_training.sh --instances 4 --run-dir runs/<name> --init runs/<bc>/bc.pt   # supervised
    ./tools/stop_all.sh

A run writes `runs/<name>/`:
- `metrics.jsonl`: one record per iteration;
- checkpoints: `ckpt.pt`, `ckpt-best.pt`, `ckpt-iterNNNN.pt`;
- `episodes/*.npz`: every episode, enough to replay it exactly;
- `clips/`: short videos around events.

Watch a run:

    tail -F runs/<name>/metrics.jsonl | python3 -u tools/fmt_metrics.py
    python3 -u tools/watch_games.py runs/<name> 4

Replay recorded episodes at full resolution on a spare game instance (it renders only while the
trainer collects, never during an update):

    cd python && .venv/bin/python -m rainbot.scripts.replay --run-dir ../runs/<name> [episode.npz ...]

## Behaviour cloning

The starting policy is a clone of human play:

    cd python && .venv/bin/python -m rainbot.train.bc --out ../runs/bc

Human play is recorded with `mode = log`. The game plays normally and never blocks. Sessions land in
`<game>/RainBotData/session-<timestamp>.{obs,events.jsonl,schema.json}` (about 41 MB per hour).
Check a session with:

    cd python && .venv/bin/python -m rainbot.scripts.check_session "<game>/RainBotData"

The check tests action-label alignment: the input must cause horizontal acceleration at lag 0. It does
not test replay determinism, because Rain World uses a lot of randomness and a replay diverges even with
correct labels.

## Video alignment

OBS records at 60 fps and the game runs at 40 Hz, so frames do not map to ticks in a fixed way. The
mod draws the tick number as a binary strip in the top-left corner (`[marker]` in the config), and
each frame decodes to an exact tick:

    python -m rainbot.scripts.align_video capture.mkv     # -> capture.ticks.npz

Keep `cell_px` at 16 or more so the cells survive video compression. Keep the marker on for anything
recorded as training data.

`tick` restarts at 1 on every game launch. To join a video to a session, pick the session whose
`wall_ms` range covers the video's creation time. After that, row index = tick - 1.

## Probes (after training)

    cd python
    .venv/bin/python -m rainbot.scripts.checks shuffle-rooms --ckpt ../runs/X/ckpt.pt
    .venv/bin/python -m rainbot.scripts.checks memory-probe  --ckpt ../runs/X/ckpt.pt
    .venv/bin/python -m rainbot.scripts.checks room-probe    --ckpt ../runs/X/ckpt.pt

## License

Apache 2.0. Rain World is the property of its owners. This repository holds no game files.
