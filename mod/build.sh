#!/usr/bin/env bash
# Build RainBot and install it into the game.
#
#   BepInEx/plugins/RainBot.dll      the loader (mod/Loader): permanent, whitelisted. A change to it
#                                    needs a game restart (it only loads the core).
#   BepInEx/rainbot/RainBotCore.dll  the mod itself (mod/RainBot). Running games swap it in within
#                                    a second, as soon as no agent is connected (or on `reload`).
set -euo pipefail
export DOTNET_ROOT="${DOTNET_ROOT:-$HOME/.dotnet}"
export PATH="$DOTNET_ROOT:$PATH"

RW="${RAINWORLD_DIR:-$HOME/.steam/steam/steamapps/common/Rain World}"
[ -d "$RW/BepInEx/plugins" ] || { echo "no BepInEx at: $RW" >&2; exit 1; }
HERE="$(cd "$(dirname "$0")" && pwd)"

dotnet build "$HERE/RainBot/RainBot.csproj" -c Release -v minimal -nologo -p:RainWorldDir="$RW"
# Rain World's Remix loader (BepInEx.MultiFolderLoader) sweeps BepInEx/plugins on every launch
# and silently moves anything not whitelisted into BepInEx/backup/. The game then starts but the
# plugin never loads. The whitelist is the concatenation of every StreamingAssets/whitelist*.txt,
# so the mod ships its own file instead of editing the game's.
SA="$RW/RainWorld_Data/StreamingAssets"
echo "rainbot.dll" > "$SA/whitelist-rainbot.txt"

LOADER="$HERE/Loader/bin/Release/RainBot.dll"
if ! cmp -s "$LOADER" "$RW/BepInEx/plugins/RainBot.dll"; then
    install -m 644 "$LOADER" "$RW/BepInEx/plugins/RainBot.dll"
    echo "loader changed -> $RW/BepInEx/plugins/RainBot.dll (running games need a restart for this one)"
fi
rm -f "$RW/BepInEx/backup/RainBot.dll"

mkdir -p "$RW/BepInEx/rainbot"
install -m 644 "$HERE/RainBot/bin/Release/RainBotCore.dll" "$RW/BepInEx/rainbot/RainBotCore.dll.tmp"
mv -f "$RW/BepInEx/rainbot/RainBotCore.dll.tmp" "$RW/BepInEx/rainbot/RainBotCore.dll"   # atomic rename: a game never reads a half-written file
echo "core -> $RW/BepInEx/rainbot/RainBotCore.dll (running games swap it in when no agent is connected)"
