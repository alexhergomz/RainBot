#!/usr/bin/env bash
# Stop every RainBot process, in an order that does not leave orphans, and check that it worked.
#
# Orphans do real damage: a trainer that outlives its supervisor keeps writing rollouts or
# checkpoints, and two supervisors fight over the game's lifecycle (which looks like the game
# crashing). The order matters: supervisors first, or a surviving supervisor relaunches the game
# that was just killed; then the trainer, a child that can outlive a SIGTERMed parent; then the
# game; then any check scripts.
set -uo pipefail

kill_all() {
    local pattern="$1" label="$2" sig="${3:-TERM}"
    local pids
    pids=$(pgrep -f "$pattern" | grep -v "^$$\$" || true)
    [ -z "$pids" ] && return 0
    echo "  stopping $label: $(echo "$pids" | tr '\n' ' ')"
    # shellcheck disable=SC2086
    kill -"$sig" $pids 2>/dev/null || true
}

echo "stopping RainBot"
kill_all "python[^ ]* -m rainbot\.train\.supervisor" "supervisor"
sleep 3
kill_all "python[^ ]* (-u )?-m rainbot\.train\.ppo" "trainer"
sleep 2
pgrep -x "RainWorld.exe" >/dev/null && { echo "  stopping game"; pkill -x "RainWorld.exe"; }
sleep 5
kill_all "python[^ ]* (-u )?-m rainbot\.scripts\.checks" "checks"
sleep 2

PATTERNS=("python[^ ]* -m rainbot\.train\.supervisor" "python[^ ]* (-u )?-m rainbot\.train\.ppo"
          "python[^ ]* (-u )?-m rainbot\.scripts\.checks")

# Anything still alive gets SIGKILL, then check.
for p in "${PATTERNS[@]}"; do
    kill_all "$p" "$p (forcing)" KILL
done
pgrep -x "RainWorld.exe" >/dev/null && pkill -9 -x "RainWorld.exe"
sleep 2

fail=0
for p in "${PATTERNS[@]}"; do
    n=$(pgrep -f "$p" | grep -v "^$$\$" | wc -l)
    [ "$n" -gt 0 ] && { echo "  STILL RUNNING: $p ($n)"; fail=1; }
done
n=$(pgrep -x "RainWorld.exe" | wc -l)
[ "$n" -gt 0 ] && { echo "  STILL RUNNING: RainWorld.exe ($n)"; fail=1; }

if [ "$fail" -eq 0 ]; then
    echo "all stopped; port 54321 $(ss -tln 2>/dev/null | grep -q ':54321' && echo 'STILL BOUND' || echo 'free')"
else
    echo "something survived; do not relaunch until it is gone" >&2
fi
exit "$fail"
