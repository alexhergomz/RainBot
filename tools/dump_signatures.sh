#!/usr/bin/env bash
# Decompile the game types the mod hooks, into reference/types/.
# Re-run after any game update: every hook signature in mod/RainBot/ was verified against these.
set -euo pipefail
export DOTNET_ROOT="${DOTNET_ROOT:-$HOME/.dotnet}"
export PATH="$DOTNET_ROOT:$DOTNET_ROOT/tools:$PATH"

RW="${RAINWORLD_DIR:-$HOME/.steam/steam/steamapps/common/Rain World}"
M="$RW/RainWorld_Data/Managed"
OUT="$(dirname "$0")/../reference/types"
mkdir -p "$OUT"

command -v ilspycmd >/dev/null ||
    { echo "need: dotnet tool install -g ilspycmd --version 9.0.0.7889" >&2; exit 1; }

for t in Player RWInput RainWorld MainLoopProcess RainWorldGame Room AbstractRoom Creature; do
  ilspycmd -t "$t" -r "$M" "$M/Assembly-CSharp.dll" > "$OUT/$t.cs" 2>/dev/null \
    && printf "  %-18s %8d bytes\n" "$t" "$(stat -c%s "$OUT/$t.cs")" \
    || echo "  $t: FAILED"
done

# The MonoMod HookGen delegate names: overload suffixes cannot be guessed.
ilspycmd -t "On.Player" -t "On.RWInput" -t "On.MainLoopProcess" \
  -r "$M" -r "$RW/BepInEx/core" "$RW/BepInEx/plugins/HOOKS-Assembly-CSharp.dll" \
  > "$OUT/HOOKS.cs" 2>/dev/null && echo "  HOOKS.cs written"
