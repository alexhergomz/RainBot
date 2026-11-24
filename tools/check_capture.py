"""Check that the human's input is being recorded. Run this a minute into a session.

The failure is silent: the mod hooks `Player.checkInput` and reads the InputPackage the game
consumed, which is device-agnostic. But if the mod is in the wrong mode, or a controller is not
seen, the stream records a player who never pressed anything, and the session looks fine until
it is used.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from rainbot.data import session as S              # noqa: E402

RB = Path.home() / ".steam/steam/steamapps/common/Rain World/RainBotData"

path = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted(Path(RB).glob("session-*.obs"))[-1]
sess = S.load(path)
o = sess.obs
a = np.asarray(o["prev_action"]).reshape(len(o), -1)
names = ["x", "y", "jump", "throw", "pickup"]

print(f"{path.stem}: {len(o)} ticks ({len(o)/40/60:.1f} min)")
print(f"  mode check: inject_fired on {float(np.asarray(o['inject_fired']).mean())*100:.0f}% "
      f"of ticks (should be 0% for a human session)")
print(f"  has_controller on {float(np.asarray(o['has_controller']).mean())*100:.0f}% of ticks")
print()
dead = 0
for i, nm in enumerate(names):
    vals = Counter(int(v) for v in a[:, i])
    nonzero = 1.0 - vals.get(0, 0) / len(a)
    flag = "" if nonzero > 0.001 else "   <- never pressed"
    if nonzero <= 0.001:
        dead += 1
    print(f"  {nm:7s} pressed on {nonzero*100:5.1f}% of ticks   {dict(sorted(vals.items()))}{flag}")

moved = float(np.linalg.norm(np.diff(np.asarray(o["pos0"], dtype=float), axis=0), axis=1).mean())
print(f"\n  movement {moved:.2f} units/tick")
if dead >= 3 or moved < 0.05:
    print("\n  LOOKS WRONG: little or no input is reaching the recorder.")
    print("  Check the mod is in log mode (tools/play_mode.sh human) and that no collector is running.")
else:
    print("\n  capture looks healthy")
