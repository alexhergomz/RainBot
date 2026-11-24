#!/usr/bin/env bash
# Start an unattended PPO run under the supervisor, with the machine held awake for its duration.
#
#   tools/run_training.sh --instances 2 --run-dir runs/ladder --stage L [any rainbot.train.ppo flag]
#
# systemd-inhibit and kde-inhibit are scoped to this process, so the inhibitors end with the run.
# Both are needed: KDE's screen locker ignores systemd's idle inhibitor, and a locked session
# stops Proton games from starting at all, with no log line. The session is unlocked once at
# start (loginctl, no PIN needed). Screen blanking is disabled because DPMS blanking interferes
# with the game's rendering under Proton.
# Refuses to start a second supervisor: two of them each own the game's lifecycle and kill each
# other's instances.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DISPLAY=${DISPLAY:-:0} xset s off 2>/dev/null || true
DISPLAY=${DISPLAY:-:0} xset -dpms 2>/dev/null || true
for sid in $(loginctl list-sessions --no-legend 2>/dev/null | awk -v u="$(id -u)" '$2 == u {print $1}'); do
    loginctl unlock-session "$sid" 2>/dev/null || true
done

for orphan in $(pgrep -f "python[^ ]* -u -m rainbot\.train\.ppo" || true); do
    if ! pgrep -f "python[^ ]* -m rainbot\.train\.supervisor" >/dev/null 2>&1; then
        echo "sweeping orphaned trainer $orphan" >&2
        kill "$orphan" 2>/dev/null || true
    fi
done
existing=$(pgrep -f "python[^ ]* -m rainbot\.train\.supervisor" || true)
if [ -n "$existing" ] && [ "${RAINBOT_FORCE:-0}" != "1" ]; then
    echo "a supervisor is already running (pid $(echo "$existing" | tr '\n' ' ')); stop it first" >&2
    exit 1
fi
cd "$REPO/python"
exec systemd-inhibit --what=idle:sleep:handle-lid-switch --who="RainBot" --why="PPO training run" \
  kde-inhibit --power --screenSaver \
  "$REPO/python/.venv/bin/python" -m rainbot.train.supervisor "$@"
