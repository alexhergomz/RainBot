#!/usr/bin/env bash
# Switch the mod between agent-driven and human-played, and launch for a human session.
#
#   tools/play_mode.sh human    # record a human session, no agent, no autopilot
#   tools/play_mode.sh agent    # put it back the way the collector expects
#
# Three settings matter and all three are wrong for a human by default:
#
#   mode=both        blocks inside Player.checkInput waiting for the agent to answer, and
#                    injects its reply. A human would be playing against a frozen game and
#                    their own inputs would be overridden. `log` records without either.
#   autopilot=true   synthesises button presses to walk menus. It fights you at every screen.
#   speed            the sim multiplier; a human needs 1.
#
# Recording is on in both modes: the mod writes the same .obs stream either way, so a human
# session is directly comparable with an agent one and lands in the same directory.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
RW="${RAINWORLD_DIR:-$HOME/.steam/steam/steamapps/common/Rain World}"
CFG="$RW/BepInEx/config/rainbot.bridge.cfg"
MODE="${1:-}"

[ -f "$CFG" ] || { echo "no mod config at $CFG" >&2; exit 1; }

# Section-aware, because the keys are NOT unique: `enabled` appears under both [autopilot] and
# [marker]. A blind match would turn the tick marker off along with the autopilot, and the marker
# is what aligns a recorded video back to ticks.
set_key() {
    local section="$1" key="$2" val="$3"
    # The file is written by BepInEx under Proton and has CRLF endings, so a section header
    # reads as "[marker]\r" and never equals "[marker]". Strip the carriage return before
    # comparing and put it back on the way out; otherwise nothing matches and nothing is said.
    awk -v sec="[$section]" -v k="$key" -v v="$val" '
        { line = $0; sub(/\r$/, "", line) }
        line ~ /^\[/ { in_sec = (line == sec) }
        in_sec && $1 == k && $2 == "=" { printf "%s = %s\r\n", k, v; next }
        { printf "%s\r\n", line }
    ' "$CFG" > "$CFG.tmp" && mv "$CFG.tmp" "$CFG"
}

case "$MODE" in
  prep)
    # Config only, no launch: the mod reads these at startup, so they must be right before the
    # game is opened. Separate from `human` so the person playing can start the game themselves.
    "$REPO/tools/stop_all.sh" || { echo "something is still running; fix that first" >&2; exit 1; }
    set_key bridge mode log
    set_key autopilot enabled false
    set_key marker enabled true          # keep the tick marker: it is what aligns video to ticks
    echo "mod set for human play: mode=log, autopilot=off, tick marker on"
    grep -vE '^#|^$' "$CFG" | sed 's/^/  /'
    echo
    echo "safe to open the game now. every tick is recorded to RainBotData/session-*.obs"
    echo "a minute in, check capture with: ./python/.venv/bin/python tools/check_capture.py"
    ;;
  human)
    "$REPO/tools/stop_all.sh" || { echo "something is still running; not launching" >&2; exit 1; }
    set_key bridge mode log
    set_key autopilot enabled false
    set_key marker enabled true
    echo "mod set for human play: mode=log, autopilot=off, tick marker on"
    grep -vE '^#|^$' "$CFG" | sed 's/^/  /'
    echo
    echo "launching. play normally; every tick is recorded to RainBotData/session-*.obs"
    echo "when you are done, close the game and run: tools/play_mode.sh agent"
    exec "$REPO/tools/launch.sh"
    ;;
  agent)
    set_key bridge mode both
    set_key autopilot enabled true
    set_key marker enabled true
    echo "mod restored for the agent: mode=both, autopilot=on"
    grep -vE '^#|^$' "$CFG" | sed 's/^/  /'
    ;;
  *)
    echo "usage: $0 {prep|human|agent}" >&2
    exit 2
    ;;
esac
