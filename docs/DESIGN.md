# Design

RainBot is one recurrent policy network that learns to survive in Rain World. It plays the real game
through a BepInEx mod. It trains with GRPO (default) or PPO on several game instances that run
faster than real time.

## Overview

- **Reward.** Homeostatic drive reduction over three variables the game already tracks: food, karma
  and breath. Nothing else enters the reward.
- **Network.** A hybrid sequence model: three Kimi Delta Attention (KDA) layers per sliding-window
  softmax attention (SWA) layer. KDA carries long memory at constant cost. The window handles recent
  detail.
- **Time.** One token covers 8 game ticks. The agent acts in chunks of 4 decisions (2 ticks each).
- **Starts.** Episodes start from randomised states that the mod sets up (room, position, food,
  karma, rain timer, fruit).

Principles:
- Inputs and reward come from what the game itself tracks.
- One reward term, so there are no weights to tune.
- One network.
- Bounded compute per step.
- Add a component only when there is evidence for it.

## Code map

| part | where |
|---|---|
| Lockstep (8 ticks per step), state export, room id, action override, randomised reset | `mod/RainBot/Lockstep.cs`, `ObsWriter.cs`, `Reset.cs`; wire contract in `docs/PROTOCOL.md` |
| Parallel instances (port and save isolation) | `mod/RainBot/Instance.cs`, `tools/launch_instances.sh` |
| Path distances (AI map, observed moves, jump links) | `mod/RainBot/PathInfo.cs` |
| Checks before training: physics under speed-up, determinism, throughput | `python -m rainbot.scripts.checks physics / determinism / throughput` |
| Observations to model inputs (categoricals by name) | `python/rainbot/obs.py` |
| Homeostatic reward and shaping | `python/rainbot/reward.py`, `train/forage_fit.py` |
| Layers: ScaleNorm, LayerScale, GLU | `python/rainbot/nn/layers.py` |
| Vision: CNN on the rendered screen | `python/rainbot/nn/vision.py` |
| Cross-attention from each step to recent frames | `python/rainbot/nn/xattn.py` |
| Tokenizer, backbone, heads | `python/rainbot/nn/agent.py`, `kda.py`, `swa.py`, `action_head.py` |
| Optimizer (MuSGD, AdamW control) | `python/rainbot/optim.py` |
| Trainer (PPO path, update, main loop) | `python/rainbot/train/ppo.py` |
| GRPO collection (groups, refills, temperatures) | `python/rainbot/train/grpo.py`, `rollout.py` |
| Episode records | `python/rainbot/train/episodes.py` |
| Curriculum ladder | `python/rainbot/train/curriculum.py` |
| Behaviour cloning on human recordings | `python/rainbot/train/bc.py` |
| Plasticity: dormant ratio, ReDo, ranks | `python/rainbot/train/plasticity.py` |
| Supervisor (restarts games and trainer) | `python/rainbot/train/supervisor.py`, `tools/run_training.sh` |
| Fake game that follows the protocol (tests without Rain World) | `python/rainbot/env/fake_game.py` |

## Observations

- **Vision is the rendered screen**, downscaled in the mod to 160x96. There is no tile map.
  - A fourth channel holds a movement-graph distance field from the slugcat. The graph combines the
    game's AI connections, moves recorded from human play and jump links.
  - The field is an arrival-time flood. Air tiles take the time of the jumps and drops that cross them.
  - Each room is scaled on its own: 255 at the slugcat, 1 at the farthest reachable tile, 0 where the
    slugcat cannot reach.
- **No pooling.** The CNN feature map (10x6) is flattened to 60 tokens. 14 entity tokens are added,
  and one fusion layer runs over them.
- **Only the latest frame goes into vision.** The backbone reads frames through sliding-window
  cross-attention over the last 4 frames' tokens, placed in the SWA blocks. KDA carries everything
  older.
  - Attention uses QK-norm and ScaleNorm pre-norm, with a residual and LayerScale around it.
  - 2D RoPE on image tokens.
- **Entities carry more than type, position and velocity:** the creature's AI behaviour, flags
  (dead, stunned, grabbing the slugcat), food points and a weapon flag. A lizard's commitment to a
  lunge cannot be learned without the behaviour.
- **Categoricals are keyed by name.** ExtEnum indices depend on registration order and room indices
  are per region. So every embedding (animation, body mode, creature and object type, behaviour,
  room) goes through a vocabulary (name to id) that the checkpoint saves.

## Reward

The reward is the drop in a drive D:

- **Necessity deficits** d_i are each in [0, 1], measured over their own range:
  - food: up to tonight's base requirement;
  - karma: up to 9, the game's maximum;
  - air: up to full lungs.
- **Drive:** D = sum of d_i². Each need's marginal weight is its own size, so the biggest deficit
  counts most and every deficit counts.
- **Reward per step:** r = D(before) - D(after).
- **Extras.** Pips above the requirement are not a need. They are paid only when the game books a
  sleep, as 1 - ((req - e) / req)², which is tomorrow's starting food. Nothing is paid at a death.
- **Sleep detection.** A sleep comes from the mod's session_end reason. It is never inferred from
  karma, because at the karma cap a sleep adds none.
- **Shaping.** Potential-based shaping is on every variable:
  - food: what is held, plus eating progress and a forecast;
  - karma: fed in a shelter;
  - breath: its own level.
- **Exact telescoping.** A segment cut and an unexpected episode change both close with Phi = 0. The
  shaped return is then the real return minus Phi(start), which the whole group shares.
- **Food bookkeeping.** Food is held across cycle boundaries: the hibernation spend and a death's
  rewind are bookkeeping, not eating.
- **Food variable.** It is the pips in the stomach, quarter pips included. That is what the game
  tracks and what hibernation consumes, so extra pips carry over.

The study on recorded episodes is `python -m rainbot.scripts.reward_study`.

## Network

- **Backbone:** `[KDA, KDA, KDA, SWA] x 2`, d = 256, about 8.4M parameters.
- **Actor:** an autoregressive action head. Each of the 4 decisions in a chunk sees the decisions
  before it in the same chunk through a causal convolution stack (kernel 2, dilations 1 and 2).
  Decisions drawn independently jittered in a 4-decision rhythm.
- **Critic:** a PopArt critic, used only on the PPO path.
- **Initialisation:** PyTorch default initialisations everywhere.
- **Norm epsilon:** the epsilon is on the mean square (RMSNorm convention). A floor on the norm itself
  gave zero inputs (black patches) a gain of about 1e7 per norm.
- **Frame pixels:** centred to [-0.5, 0.5].

## Training

- **GRPO (default).**
  - Every game resets with the same seeded spec and forms a group.
  - Advantage = per-step group-normalised discounted return-to-go. There is no critic. PPO-clip and
    entropy are kept.
  - A pass is a time budget. Every game whose member finishes resets to the same start and plays
    another member (a refill), until the budget runs out. Members still alive then are cut and left
    out. No game waits for another. Members are compared by their own episode step.
- **Member temperatures.** Group members sample at temperatures spread geometrically from 1 to
  tau_max. tau_max is fitted each iteration so the hottest member samples at 30% of the maximum
  entropy. Each member trains on-policy at its own temperature.
- **PPO path** (`--algo ppo`): gamma 0.999 per token, GAE lambda 0.95, clip 0.2, value coefficient
  0.5, entropy coefficient 0.01, PopArt critic.
- **Compute split.**
  - Rollouts run while the games step.
  - Updates run on the GPU while the games stay frozen in lockstep; keepalive pings hold their
    links.
  - The learner never trains while a game is mid-step: the learner and a game both busy on this GPU
    have crashed the machine.
- **Learning rates:** `--lr` for the Muon group, `--sgd-lr` for the rest (defaults in `PPOConfig`).
  Fine-tuning from a clone uses lower rates (see `docs/STATUS.md`).
- **Start from a clone.** Behaviour cloning on human recordings gives the starting policy (`train/bc.py`).
  - Decisions are keyframe-weighted (Wen et al. 2021, arXiv 2106.06452).
  - Without the weighting, the clone copies its previous action.

## Curriculum: the ladder

The episodes run in story mode. The mod's reset teleports the slugcat into a room and sets up
creatures, fruit, food, karma and the timer, so karma is real.

- **Every rung is a whole cycle:** food, eating, a shelter and the rain together. Rungs get harder in
  one dimension at a time:
  - pips missing;
  - eating speed (eating and the shelter door's counters are sped up by K, walked from 20 to 1);
  - fruit in hand, then only placed;
  - climbing;
  - water rooms;
  - the shelter 1 to 3 rooms away.
- **Rain timer:** 9000 ticks at every rung.
- **Stomach start:** anywhere from the rung's hungriest case up to exactly fed, never with extras
  (extras are earned).
- **No-climb rungs:** rungs 0 and 1 start only where the game's AI map reaches a shelter pipe without
  a pole climb. Rooms with no such start are remembered and skipped.
- **Climbing as a reverse curriculum:** three rungs play only the climbing rooms. Their starts begin
  within 6 tiles of path from the shelter pipe, then 12, then anywhere. The start may be on a pole,
  holding it.
- **Pass rule:** fed sleeps above 80%, counted on hungry starts only, over 3 windows. 20% of episodes
  replay earlier rungs.
- **Metrics only diagnose.** They choose the rung, never the reward or the gradients.

## Platform notes

- **Game:** Rain World v1.11.8 under Proton, BepInEx 5.4.17.
- **Hot reload:** `BepInEx/plugins/RainBot.dll` is only a loader (`mod/Loader`). The mod itself
  (`BepInEx/rainbot/RainBotCore.dll`) is swapped into running games when no agent is connected, or
  on the `reload` command. Only a change to the loader needs a game restart.
- **Starting the games:** they need the Steam client running and an unlocked desktop session.
  `tools/run_training.sh` holds an inhibitor for the run.
- **GPU:** AMD RX 6650 XT (gfx1030), ROCm, 8 GB.
  - fla's KDA kernels do not compile on this card, so a PyTorch chunkwise implementation
    (`kda_chunked`) is the default. `RAINBOT_KDA=fla` switches to fla where it works.
  - flash-attn is not available, so SWA uses PyTorch SDPA with an explicit mask; attention sinks are
    always-visible key columns.
  - bf16 autocast is off: it is 2x slower on RDNA2.
- **Determinism:** the same seed and actions at 1x and 16x are identical for about 400 ticks. After
  that, runs drift by fractions of a unit, and two 1x runs drift the same way. The game itself is
  not fully deterministic.
- **Throughput:** one instance, random actions, runs at 600 ticks/s (15x) at speed 16.

## Optional components (off until a symptom calls for them)

| component | symptom | status |
|---|---|---|
| METRA skill pretraining | the agent cannot cross gaps or climb even with food beyond | not built |
| AdaLN room conditioning | a probe shows room information lost in deep layers | built, `AgentConfig.adaln_room` |
| Room-graph memory | the memory probe drops off at region scale | not built |
| World model | throughput too low, or long-horizon behaviour stalls | not built |
