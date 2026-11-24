using System;

namespace RainBot
{
    /// <summary>
    /// Which of several parallel game instances this is, and the per-instance isolation that
    /// lets them run side by side.
    ///
    /// Selected by the `RAINBOT_INSTANCE` environment variable (default 0). Proton passes the
    /// variable through to the Windows process, so `tools/launch_instances.sh` sets it per launch.
    ///
    /// Isolation. All instances share one Wine prefix, so they share `persistentDataPath`. Without
    /// isolation every instance writes the same `sav` file and corrupts the others. In 1.11.8 the
    /// save goes through Kittehface `UserData` under the name returned by
    /// `Options.GetSaveFileName_SavOrExp()` ("sav", "sav2", ... per save slot). Hooking that one
    /// method gives instance N its own file `sav_rainbotN` and leaves the normal save slots alone.
    /// Instance 0 keeps the normal save, which has a den past the tutorial (a fresh save starts in
    /// SU_C04, whose start-up script pins the rain clock). The launcher seeds `sav_rainbotN` from
    /// `sav`.
    ///
    /// Instances above 0 also never write the shared `options` file, so N processes cannot race
    /// on it, and keep their session recordings in `RainBotData/instance-N`.
    /// </summary>
    internal static class Instance
    {
        public static readonly int Index = Read();

        public static bool Isolated => Index > 0;

        public static string SaveFileName => "sav_rainbot" + Index;

        private static int Read()
        {
            try
            {
                string v = Environment.GetEnvironmentVariable("RAINBOT_INSTANCE");
                int i;
                if (!string.IsNullOrEmpty(v) && int.TryParse(v.Trim(), out i) && i >= 0) return i;
            }
            catch (Exception) { }
            return 0;
        }

        private static On.Options.hook_GetSaveFileName_SavOrExp _saveName;
        private static On.Options.hook_Save _save;

        public static void Hook()
        {
            if (!Isolated) return;
            _saveName = (orig, self) => SaveFileName;
            _save = (orig, self) => { };
            On.Options.GetSaveFileName_SavOrExp += _saveName;
            On.Options.Save += _save;
            Plugin.Log.LogMessage("instance " + Index + ": save file '" + SaveFileName + "', options writes disabled");
        }

        public static void Unhook()
        {
            if (_saveName != null) { On.Options.GetSaveFileName_SavOrExp -= _saveName; _saveName = null; }
            if (_save != null) { On.Options.Save -= _save; _save = null; }
        }
    }
}
