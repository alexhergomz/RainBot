using System;
using System.Collections.Generic;
using UnityEngine;

namespace RainBot
{
    /// <summary>
    /// Keeps the game running without a human: clears death prompts, advances the karma /
    /// sleep / death screens, and optionally fast-forwards the sim.
    ///
    /// Unattended training needs this: an agent learning from scratch dies often, and every death
    /// parks the game on a prompt that waits for a button.
    ///
    /// Two mechanisms, chosen per screen so the save state stays correct:
    ///
    ///   Game-over prompt: call RainWorldGame.GoToDeathScreen() directly. That is what
    ///   TextPrompt.Update does when it sees the button, so nothing is skipped.
    ///
    ///   Menu screens: synthesise a UI button press through RWInput.PlayerUIInput instead of
    ///   forcing a process switch. The death and sleep screens commit karma loss and the cycle
    ///   counter to the save; jumping straight back to Game would skip that and make progression
    ///   differ from what the agent observes.
    ///
    /// The press is pulsed, not held. TextPrompt and the menus edge-detect (`flag &amp;&amp;
    /// !lastAnyKey`), so a held button registers once and then never again, which looks like a
    /// hang.
    /// </summary>
    internal sealed class AutoPilot
    {
        public bool Enabled;
        /// <summary>Sim speed multiplier. 1 = normal 40 Hz.</summary>
        public float SpeedMultiplier = 1f;
        /// <summary>Ticks to wait before acting on a screen, so its own animations can settle.</summary>
        public int SettleTicks = 60;
        /// <summary>Frames to hold the synthesised button. Must exceed the longest hold-to-
        /// continue meter in the game; slugcat select is roughly a second and a half.</summary>
        public int HoldFrames = 150;
        /// <summary>Frames released between holds, so edge-detecting screens re-arm.</summary>
        public int ReleaseFrames = 40;

        /// <summary>
        /// Screens the autopilot must never blind-press, because a press there can do something
        /// destructive or irreversible.
        ///
        /// A denylist, not an allowlist: an allowlist needs a new entry for every unexpected screen
        /// (such as Tips after a death), and each missing entry parks an unattended run. With a
        /// denylist an unknown screen costs a stray button press.
        ///
        /// MainMenu is excluded because its button list contains EXIT; it is navigated by
        /// signalling STORY instead. The options screens are excluded because a press there
        /// rewrites the player's settings.
        /// </summary>
        private static readonly HashSet<string> NeverAdvance = new HashSet<string>
        {
            "MainMenu", "OptionsMenu", "InputOptions", "ConsoleOptionsMenu", "InputSelect",
            "ModListButton", "Credits", "RegionSelect", "RegionsOverviewScreen",
            "MultiplayerMenu", "ArenaMenu", "Initialization", "RainWorldSteamManager",
        };

        private string _lastProcess = "";
        private int _framesOnProcess;
        private int _pulse;
        private int _deathsHandled, _screensAdvanced;

        public int DeathsHandled { get { return _deathsHandled; } }
        public int ScreensAdvanced { get { return _screensAdvanced; } }
        public string CurrentProcess { get { return _lastProcess; } }

        /// <summary>True while a synthetic UI press should be reported this frame.</summary>
        public bool WantUiPress { get; private set; }

        /// <summary>Call once per rendered frame, before the game reads input.</summary>
        public void Update(ProcessManager manager)
        {
            WantUiPress = false;
            if (!Enabled || manager == null || manager.currentMainLoop == null) return;

            string id = manager.currentMainLoop.ID != null
                ? manager.currentMainLoop.ID.ToString() : "?";
            if (id != _lastProcess)
            {
                Plugin.Log.LogInfo("autopilot: process -> " + id);
                _lastProcess = id;
                _framesOnProcess = 0;
                _pulse = 0;
            }
            _framesOnProcess++;

            // Main menu: name the destination instead of pressing whatever is highlighted.
            // Menu.MainMenu.Singal(sender, "STORY") runs the STORY button's own callback, which
            // switches to SlugcatSelect. A blind press here is dangerous: the same list holds
            // EXIT, and a mistimed press on a save slot can start a new game over an existing one.
            if (id == "MainMenu" && _framesOnProcess > SettleTicks)
            {
                Menu.MainMenu mm = manager.currentMainLoop as Menu.MainMenu;
                if (mm != null)
                {
                    Plugin.Log.LogMessage("autopilot: main menu -> STORY");
                    try { mm.Singal(null, "STORY"); }
                    catch (Exception e) { Plugin.Log.LogWarning("STORY signal failed: " + e.Message); }
                    _framesOnProcess = 0;
                }
                return;
            }

            RainWorldGame game = manager.currentMainLoop as RainWorldGame;
            if (game != null)
            {
                ApplySpeed(game);
                // TextPrompt ignores the button for its first 40 ticks, so wait that out.
                if (_framesOnProcess > SettleTicks && IsGameOver(game))
                {
                    _deathsHandled++;
                    Plugin.Log.LogMessage("autopilot: clearing death prompt (#" + _deathsHandled + ")");
                    try { game.GoToDeathScreen(); }
                    catch (Exception e) { Plugin.Log.LogWarning("GoToDeathScreen failed: " + e.Message); }
                    _framesOnProcess = 0;
                }
                return;
            }

            // A notification dialog is a side process, so `currentMainLoop.ID` reports the screen
            // underneath. If that screen is on the denylist, the dialog would never be answered
            // (for example the "mods have been updated" dialog over Initialization after a mod
            // rebuild).
            //
            // A denylist again, for the reason given above NeverAdvance. The two excluded dialogs
            // ask real questions about save data. Matched by name, not by type, so every dialog
            // class need not resolve at compile time (several are not under Menu, one is not a
            // Dialog). `Menu.Menu.Singal` ignores messages it does not recognise, so a wrong
            // guess does nothing.
            Menu.Dialog dlg = manager.dialog;
            if (dlg != null)
            {
                string dname = dlg.GetType().Name;
                if (dname != "DialogConfirm" && dname != "DialogBackupSaveInfo")
                {
                    // DialogNotify greys out its OK button for `timeOut` seconds and ignores
                    // signals until then, so wait.
                    if (_framesOnProcess > SettleTicks)
                    {
                        Plugin.Log.LogMessage("autopilot: dismissing " + dname);
                        try { dlg.Singal(null, "OK"); }
                        catch (Exception e) { Plugin.Log.LogWarning("dialog OK failed: " + e.Message); }
                        _framesOnProcess = 0;
                    }
                }
                else if (_framesOnProcess == SettleTicks)
                {
                    // Not answered, but logged, so a parked run can be diagnosed from the log.
                    Plugin.Log.LogWarning("autopilot: waiting on " + dname + " over " + id
                                          + "; it asks about save data and will not be answered");
                }
                return;
            }

            if (NeverAdvance.Contains(id)) return;
            if (_framesOnProcess < SettleTicks) return;

            // A screen with a CONTINUE button (dream, sleep, death, karma screens): send its own
            // signal, which is what clicking it does. A synthesised press only works when the
            // button is selected, and the dream screen can show CONTINUE with nothing selected.
            // Sent once per second, in case the screen ignores it while animating.
            Menu.Menu menu = manager.currentMainLoop as Menu.Menu;
            if (menu != null && _framesOnProcess % 40 == 0)
            {
                Menu.SimpleButton b = FindButton(menu, "CONTINUE");
                if (b != null && !b.buttonBehav.greyedOut)
                {
                    Plugin.Log.LogMessage("autopilot: CONTINUE on " + id);
                    try { menu.Singal(b, "CONTINUE"); } catch (Exception e) { Plugin.Log.LogWarning("CONTINUE failed: " + e.Message); }
                    return;
                }
            }

            // Long hold, then a clean release. The death and karma screens edge-detect a button
            // (`flag && !lastAnyKey`), while slugcat select is hold-to-continue and ignores a tap.
            // A long hold serves both: the leading edge fires the edge detectors and the held
            // press fills the hold meter. The release re-arms the edge for the next screen.
            _pulse = (_pulse + 1) % (HoldFrames + ReleaseFrames);
            WantUiPress = _pulse < HoldFrames;
            if (_pulse == 0) _screensAdvanced++;
        }

        private static Menu.SimpleButton FindButton(Menu.Menu menu, string signal)
        {
            if (menu.pages == null) return null;
            foreach (Menu.Page page in menu.pages)
            {
                Menu.SimpleButton b = FindButton(page, signal);
                if (b != null) return b;
            }
            return null;
        }

        private static Menu.SimpleButton FindButton(Menu.MenuObject o, string signal)
        {
            Menu.SimpleButton sb = o as Menu.SimpleButton;
            if (sb != null && sb.signalText == signal) return sb;
            if (o.subObjects != null)
                foreach (Menu.MenuObject c in o.subObjects)
                {
                    Menu.SimpleButton r = FindButton(c, signal);
                    if (r != null) return r;
                }
            return null;
        }

        private static bool IsGameOver(RainWorldGame game)
        {
            try { return game.GameOverModeActive; }
            catch (Exception) { return false; }
        }

        public void ApplySpeed(RainWorldGame game)
        {
            // framesPerSecond sets how many Update()s RawUpdate runs per second of real time, so
            // raising it fast-forwards the sim. This is not frameskip: the agent still sees every
            // tick, sooner.
            int want = Mathf.Clamp(Mathf.RoundToInt(40f * SpeedMultiplier), 40, 40 * 16);
            if (game.framesPerSecond != want) game.framesPerSecond = want;
        }

        /// <summary>The package to hand back from RWInput.PlayerUIInput while advancing a screen.</summary>
        public Player.InputPackage UiPress(Player.InputPackage baseline)
        {
            baseline.jmp = true;
            baseline.thrw = true;   // screens differ in which button they accept; press both
            baseline.pckp = false;
            baseline.x = 0;
            baseline.y = 0;
            baseline.analogueDir = Vector2.zero;
            return baseline;
        }
    }
}
