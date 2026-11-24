using System;
using System.Reflection;

namespace RainBot
{
    /// <summary>Log why the rain clock stopped, the first time it does.</summary>
    /// <remarks>
    /// A frozen clock makes a world where nothing is urgent and nothing dies: the clock ends
    /// episodes and drives the death detector and every time-pressure input. One known cause is
    /// the tutorial room's start-up script, which pins the clock while SU_C04 stays loaded.
    /// `AllowRainCounterToTick()` can say no for several unrelated reasons, so each is logged.
    /// Logged once, not per tick.
    /// </remarks>
    internal static class Diagnostics
    {
        private static int _lastTimer = -1, _stuckFor;
        private static bool _diagnosed;

        public static void FrozenCycle(Player self, int timer)
        {
            if (timer < 0 || _diagnosed) { _lastTimer = timer; return; }
            if (timer != _lastTimer) { _lastTimer = timer; _stuckFor = 0; return; }
            if (++_stuckFor < 400) return;      // 10 s of sim with no movement
            _diagnosed = true;
            try
            {
                RainWorldGame game = self.abstractCreature.world.game;
                World pw = self.abstractCreature.world;
                Plugin.Log.LogWarning("rain clock frozen at " + timer + " for 400 ticks:"
                    + " AllowRainCounterToTick=" + game.AllowRainCounterToTick()
                    + " disableRain=" + game.setupValues.disableRain
                    + " GamePaused=" + game.GamePaused
                    + " pause=" + pw.rainCycle.pause
                    + " preTimer=" + pw.rainCycle.preTimer
                    + " cycleLength=" + pw.rainCycle.cycleLength
                    + " room=" + (self.room != null && self.room.abstractRoom != null ? self.room.abstractRoom.name : "?")
                    + " ripple=" + RippleLayer(game)
                    + " player.world==game.world=" + ReferenceEquals(pw, game.world));
            }
            catch (Exception e) { Plugin.Log.LogWarning("cycle diagnosis failed: " + e.Message); }
        }

        /// <summary>Watcher's ripple layer, by reflection: absent on non-Watcher builds.</summary>
        private static string RippleLayer(RainWorldGame game)
        {
            try
            {
                PropertyInfo pi = typeof(RainWorldGame).GetProperty("ActiveRippleLayer");
                if (pi != null) return Convert.ToString(pi.GetValue(game, null));
                FieldInfo fi = typeof(RainWorldGame).GetField("ActiveRippleLayer");
                if (fi != null) return Convert.ToString(fi.GetValue(game));
            }
            catch (Exception) { }
            return "n/a";
        }
    }
}
