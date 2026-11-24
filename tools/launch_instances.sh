#!/usr/bin/env bash
# Launch N Rain World instances side by side under Proton, one per RAINBOT_INSTANCE.
#
#   tools/launch_instances.sh N [extra game args...]
#
# Instance i listens on port 54321+i (mod/RainBot/Instance.cs) and saves to its own file
# `sav_rainboti` in the shared Wine prefix; instance 0 keeps the normal `sav`. Each isolated
# save is seeded from `sav` on first launch, because a fresh save starts in the tutorial room
# SU_C04, whose start-up script pins the rain clock.
#
# Always a direct Proton launch: `steam -applaunch` starts one copy only. steam_appid.txt next
# to the exe lets Steamworks initialise standalone. Small windowed resolution, since rendering
# is not what the agent learns from (and `render off` skips most of it anyway).
#
# Stays in the foreground until every instance has exited; Ctrl-C stops them all.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
N="${1:?usage: launch_instances.sh N [game args...]}"
shift
STEAM="${STEAM_ROOT:-$HOME/.steam/steam}"
RW="${RAINWORLD_DIR:-$STEAM/steamapps/common/Rain World}"
PROTON="${PROTON_DIR:-$STEAM/steamapps/common/Proton - Experimental}"
export STEAM_COMPAT_DATA_PATH="${STEAM_COMPAT_DATA_PATH:-$STEAM/steamapps/compatdata/312520}"
export STEAM_COMPAT_CLIENT_INSTALL_PATH="$STEAM"
export SteamAppId=312520
export WINEDLLOVERRIDES="${WINEDLLOVERRIDES:-winhttp=n,b}"   # BepInEx doorstop, see launch.sh
SAVES="$STEAM_COMPAT_DATA_PATH/pfx/drive_c/users/steamuser/AppData/LocalLow/Videocult/Rain World"
W="${RAINBOT_WIDTH:-640}" H="${RAINBOT_HEIGHT:-360}"

[ -f "$RW/steam_appid.txt" ] || echo 312520 > "$RW/steam_appid.txt"

# Keep the session awake and unlocked while the games run, whoever launched them (not only
# run_training.sh). A locked session stops Proton games from starting; KDE's locker ignores
# systemd's idle inhibitor, so both are held. Unlock once now (loginctl, no PIN needed).
for sid in $(loginctl list-sessions --no-legend 2>/dev/null | awk -v u="$(id -u)" '$2 == u {print $1}'); do
    loginctl unlock-session "$sid" 2>/dev/null || true
done
DISPLAY=${DISPLAY:-:0} xset s off -dpms 2>/dev/null || true
systemd-inhibit --what=idle:sleep:handle-lid-switch --who=RainBot --why="Rain World instances running" \
    kde-inhibit --power --screenSaver sleep infinity >/dev/null 2>&1 &
inhibitor=$!

pids=()
cleanup() { [ ${#pids[@]} -gt 0 ] && kill "${pids[@]}" 2>/dev/null; kill "$inhibitor" 2>/dev/null; }
trap 'cleanup; exit 130' INT TERM

cd "$RW"
for ((i = 0; i < N; i++)); do
    if [ "$i" -gt 0 ] && [ ! -f "$SAVES/sav_rainbot$i" ]; then
        if [ -f "$SAVES/sav" ]; then
            cp "$SAVES/sav" "$SAVES/sav_rainbot$i"
            echo "instance $i: seeded sav_rainbot$i from sav"
        else
            echo "instance $i: no sav to seed from; it will start a fresh game (tutorial room)" >&2
        fi
    fi
    echo "instance $i: port $((54321 + i))"
    RAINBOT_INSTANCE=$i "$HERE/ksm_exec.py" "$PROTON/proton" run "$RW/RainWorld.exe" \
        -screen-fullscreen 0 -screen-width "$W" -screen-height "$H" "$@" \
        >"/tmp/rainbot-instance-$i.log" 2>&1 &
    pids+=($!)
    # Next instance only once this one's BepInEx is up (its mod listens on its port): instances
    # loading together write BepInEx's shared config at the same moment, and one preloader then
    # dies on the locked file (the game shows a crash screen). A fixed stagger is not always
    # enough. Wait up to 180 s, then go on anyway.
    if [ "$i" -lt $((N - 1)) ]; then
        for _ in $(seq 1 90); do
            (exec 3<>"/dev/tcp/127.0.0.1/$((54321 + i))") 2>/dev/null && break
            sleep 2
        done
        sleep "${RAINBOT_STAGGER:-5}"
    fi
done
wait "${pids[@]}"
kill "$inhibitor" 2>/dev/null
