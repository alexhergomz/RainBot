#!/usr/bin/env bash
# Launch Rain World under Proton without going through the Steam client.
# steam_appid.txt sits next to the exe, so Steamworks initialises standalone.
# Single instance (instance 0). For several in parallel use tools/launch_instances.sh N;
# RAINBOT_INSTANCE, if set, is passed through to the game on the direct-Proton path only.
set -uo pipefail
STEAM="${STEAM_ROOT:-$HOME/.steam/steam}"
RW="${RAINWORLD_DIR:-$STEAM/steamapps/common/Rain World}"
PROTON="${PROTON_DIR:-$STEAM/steamapps/common/Proton - Experimental}"

export STEAM_COMPAT_DATA_PATH="${STEAM_COMPAT_DATA_PATH:-$STEAM/steamapps/compatdata/312520}"
export STEAM_COMPAT_CLIENT_INSTALL_PATH="$STEAM"
export SteamAppId=312520
# Doorstop hooks BepInEx in by shadowing winhttp.dll. Wine prefers its own builtin
# unless told otherwise, so without this override the plugin silently never loads:
# the game starts fine and BepInEx/LogOutput.log stays empty. Steam sets this via the
# game's launch options; a direct Proton launch must set it here.
export WINEDLLOVERRIDES="${WINEDLLOVERRIDES:-winhttp=n,b}"
# Prefer launching THROUGH the Steam client when it is running.
#
# Steam Game Recording only hooks titles Steam itself launches. With `proton run` the game is
# invisible to Steam and the rolling background recording stops without any error. The video
# dataset needs the recording, so Steam wins; without Steam this falls back to a direct launch.
if [ "${RAINBOT_FORCE_PROTON:-0}" != "1" ] && [ "${RAINBOT_INSTANCE:-0}" = "0" ] \
   && command -v steam >/dev/null 2>&1 && pgrep -x steam >/dev/null 2>&1; then
    echo "launching via Steam (background recording enabled)"
    steam -applaunch 312520 >/dev/null 2>&1 &
    # Steam returns immediately; wait for the game process so the supervisor can track it.
    for _ in $(seq 1 90); do
        pgrep -f "RainWorld[.]exe" >/dev/null 2>&1 && break
        sleep 2
    done
    # Stay in the foreground for as long as the game lives, so the supervisor's process
    # lifetime still matches the game's.
    while pgrep -f "RainWorld[.]exe" >/dev/null 2>&1; do sleep 5; done
    exit 0
fi

echo "launching via Proton directly (Steam not running; no background recording)"
cd "$RW"
exec "$PROTON/proton" run "$RW/RainWorld.exe" "$@"
