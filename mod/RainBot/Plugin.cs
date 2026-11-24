using System;
using System.Collections.Generic;
using System.IO;
using BepInEx;
using BepInEx.Configuration;
using BepInEx.Logging;
using UnityEngine;

namespace RainBot
{
    /// <summary>
    /// RainBot: observation logging and chunked lockstep control for a Rain World agent.
    /// The wire contract is docs/PROTOCOL.md; the tick loop lives in <see cref="Lockstep"/>.
    ///
    /// Main hooks (checked against the decompiled v1.11.8 assembly):
    ///
    ///   RainWorldGame.Update        one simulation tick: the lockstep, rows, resets.
    ///   Player.checkInput           records the input the sim consumed.
    ///   RWInput.PlayerInput(int)    injection point.
    ///   MainLoopProcess.RawUpdate   dt clamp, sim speed, extra ticks per frame.
    ///   RainWorldGame.GrafUpdate    skipped when rendering is off.
    ///   RainWorldGame.ShutDownProcess / Win / GoTo*Screen: session end and its reason.
    ///   Creature.Violence           exact damage attribution.
    ///
    /// Injecting in RWInput.PlayerInput, not overwriting input[0], gives the agent the same
    /// post-processing a human gets (superLaunchJump zeroing x, Roll substituting downDiagonal).
    /// On random play that rewrites ~13% of actions.
    ///
    /// Loaded and swapped by the permanent loader plugin (mod/Loader, see Loader.cs). Everything
    /// started in Start is released in Stop, so a new build can replace this one in a running game.
    /// </summary>
    public sealed class Plugin : ICore
    {
        internal static ManualLogSource Log;

        private ConfigEntry<string> _mode;
        private ConfigEntry<int> _port;
        private ConfigEntry<int> _timeoutMs;
        private ConfigEntry<string> _logDir;
        private ConfigEntry<bool> _marker;
        private ConfigEntry<int> _markerCell;
        private ConfigEntry<bool> _autoEnabled;
        private ConfigEntry<float> _speed;
        private ConfigEntry<int> _maxTicksPerFrame;
        private readonly TickMarker _tickMarker = new TickMarker();

        internal readonly AutoPilot AutoPilot = new AutoPilot();
        internal Bridge Bridge { get; private set; }
        internal bool DriveEnabled { get; private set; }

        private Lockstep _lockstep;
        private ConfigEntry<int> _frameW, _frameH;
        /// <summary>Screen capture for the agent's vision (null when frames are off).</summary>
        internal FrameCapture Frame;
        private Recorder _recorder;
        private bool _logEnabled, _hooked;
        private string _dumpedWorld;
        private int _dumpedRoom = -1;
        private IHost _host;
        private ConfigFile Config => _host.Config;
        private InstanceLog _instanceLog;
        private Coroutine _frameLoop;

        public bool CanReload => Bridge == null || !Bridge.Alive();
        internal void RequestReload() { if (_host != null) _host.RequestReload(); }

        public void Start(IHost host)
        {
            _host = host;
            Log = host.Log;
            // Per-instance log that survives relaunches (LogOutput.log is shared by every instance
            // and truncated at each launch): warnings and errors, timestamped, appended.
            try { _instanceLog = new InstanceLog(); BepInEx.Logging.Logger.Listeners.Add(_instanceLog); } catch (Exception) { }
            _mode = Config.Bind("bridge", "mode", "log",
                "off | log | drive | both.  log = write play to disk, never blocks. " +
                "drive = lockstep control from Python. both = log while driving.");
            _port = Config.Bind("bridge", "port", 54321,
                "Base loopback TCP port for drive mode; instance N listens on port + N.");
            // Not `timeout_ms`: older config files hold a 30 s value under that key, which would
            // override this default.
            _timeoutMs = Config.Bind("bridge", "recv_timeout_ms", 120000,
                "Give up on a silent agent after this long and resume free play. Generous because " +
                "PPO updates hold the game frozen; the agent sends `ping` to stay inside it.");
            _logDir = Config.Bind("bridge", "log_dir", "",
                "Where session logs go. Empty = <game>/RainBotData (instance-N subfolder for N > 0).");
            _autoEnabled = Config.Bind("autopilot", "enabled", false,
                "Clear death prompts and advance karma/sleep screens automatically, so the game "
                + "keeps running unattended. Leave off while a human is playing.");
            _speed = Config.Bind("autopilot", "speed", 1.0f,
                "Sim speed multiplier (1 = 40Hz). Raises framesPerSecond; not frameskip, the "
                + "agent still sees every tick.");
            _maxTicksPerFrame = Config.Bind("autopilot", "max_ticks_per_frame", 3,
                "Most simulation ticks run per rendered frame. The game's own catch-up stops at 3; "
                + "raise it (with `render off`) when the frame rate, not the sim, is the limit.");
            _marker = Config.Bind("marker", "enabled", true,
                "Draw the tick as a binary strip in the top-left corner, so recorded video " +
                "frames can be mapped to exact ticks. Turn off for normal play.");
            _markerCell = Config.Bind("marker", "cell_px", 12,
                "Cell size in screen pixels. Larger survives video compression better.");
            _frameW = Config.Bind("vision", "frame_w", 160,
                "Width of the downscaled screen frame sent to the agent each step (0 = no frames). "
                + "Width and height divisible by 16 for the vision CNN.");
            _frameH = Config.Bind("vision", "frame_h", 96, "Height of the frame sent to the agent.");

            AutoPilot.Enabled = _autoEnabled.Value;
            AutoPilot.SpeedMultiplier = _speed.Value;
            _tickMarker.Enabled = _marker.Value;
            _tickMarker.CellPx = Math.Max(4, _markerCell.Value);

            string m = (_mode.Value ?? "log").Trim().ToLowerInvariant();
            DriveEnabled = (m == "drive" || m == "both");
            _logEnabled = (m == "log" || m == "both");

            if (!DriveEnabled && !_logEnabled)
            {
                Log.LogMessage("mode=off, doing nothing");
                return;
            }

            try
            {
                Instance.Hook();
                _lockstep = new Lockstep(this);
                RestoreRuntime();
                if (_frameW.Value > 0 && _frameH.Value > 0 && DriveEnabled)
                {
                    Observation.FrameW = _frameW.Value;
                    Observation.FrameH = _frameH.Value;
                    Frame = new FrameCapture(_frameW.Value, _frameH.Value);
                    Frame.FillDistance = FillDistance;
                    _frameLoop = host.Mono.StartCoroutine(Frame.Loop(() => Lockstep.Tick));
                    // The tick marker would be drawn into every frame the agent sees.
                    _tickMarker.Enabled = false;
                }
                int port = _port.Value + Instance.Index;
                if (DriveEnabled)
                {
                    Bridge = new Bridge(port, SchemaNow, _timeoutMs.Value);
                    Bridge.Start();
                }

                On.RainWorldGame.Update += _lockstep.OnGameUpdate;
                On.RainWorldGame.GrafUpdate += OnGameGrafUpdate;
                On.Player.GrabUpdate += OnGrabUpdate;
                On.OverseerTutorialBehavior.Update += OnOverseerTutorial;
                On.RainWorldGame.ShutDownProcess += OnGameShutDown;
                On.RainWorldGame.Win += OnWin;
                On.RainWorldGame.GoToDeathScreen += OnDeathScreen;
                On.RainWorldGame.GoToStarveScreen += OnStarveScreen;
                On.RainWorldGame.GoToRedsGameOver += OnRedsGameOver;
                On.Player.checkInput += OnCheckInput;
                On.Creature.Violence += OnViolence;
                On.RWInput.PlayerInput_int += OnPlayerInput;
                On.RWInput.PlayerUIInput_int += OnPlayerUIInput;
                On.MainLoopProcess.RawUpdate += OnRawUpdate;
                On.Player.Die += OnPlayerDie;
                On.Player.Update += OnPlayerUpdate;
                On.Player.Destroy += OnPlayerDestroy;
                On.AbstractWorldEntity.Destroy += OnAbstractDestroy;
                On.ShelterDoor.Close += OnShelterClose;
                _hooked = true;

                Log.LogMessage("mode=" + m + "  protocol=" + Observation.Protocol + "  row=" + Observation.Size + "B  "
                               + RainWorld.GAME_VERSION_STRING + "  instance=" + Instance.Index
                               + (DriveEnabled ? ("  listening 127.0.0.1:" + port) : ""));
            }
            catch (Exception e)
            {
                Log.LogError("failed to start: " + e);
            }
        }

        public void OnGUI()
        {
            if (Event.current != null && Event.current.type == EventType.Repaint)
                _tickMarker.Draw(Lockstep.Tick);
        }

        // Settings the agent changes at runtime (commands) survive a hot reload: the old core
        // leaves them in the AppDomain, and the new one applies them over the config defaults.
        private const string RuntimeKey = "rainbot.runtime";

        private void SaveRuntime()
        {
            if (_lockstep == null) return;
            AppDomain.CurrentDomain.SetData(RuntimeKey, new object[] {
                AutoPilot.Enabled, AutoPilot.SpeedMultiplier, _lockstep.RenderOff, _lockstep.Peaceful, _lockstep.FreezeEndings });
        }

        private void RestoreRuntime()
        {
            object[] v = AppDomain.CurrentDomain.GetData(RuntimeKey) as object[];
            if (v == null || v.Length != 5) return;
            AutoPilot.Enabled = (bool)v[0];
            AutoPilot.SpeedMultiplier = (float)v[1];
            _lockstep.RenderOff = (bool)v[2];
            _lockstep.Peaceful = (bool)v[3];
            _lockstep.FreezeEndings = (bool)v[4];
            Log.LogMessage("runtime settings kept across the reload: autopilot " + AutoPilot.Enabled + ", speed "
                           + AutoPilot.SpeedMultiplier + ", render " + (_lockstep.RenderOff ? "off" : "on")
                           + ", peaceful " + _lockstep.Peaceful + ", endings " + (_lockstep.FreezeEndings ? "freeze" : "game"));
        }

        public void Stop()
        {
            SaveRuntime();
            Instance.Unhook();
            if (_frameLoop != null && _host != null) { _host.Mono.StopCoroutine(_frameLoop); _frameLoop = null; }
            if (Frame != null) { Frame.Dispose(); Frame = null; }
            if (_hooked)
            {
                On.RainWorldGame.Update -= _lockstep.OnGameUpdate;
                On.RainWorldGame.GrafUpdate -= OnGameGrafUpdate;
                On.Player.GrabUpdate -= OnGrabUpdate;
                On.OverseerTutorialBehavior.Update -= OnOverseerTutorial;
                On.RainWorldGame.ShutDownProcess -= OnGameShutDown;
                On.RainWorldGame.Win -= OnWin;
                On.RainWorldGame.GoToDeathScreen -= OnDeathScreen;
                On.RainWorldGame.GoToStarveScreen -= OnStarveScreen;
                On.RainWorldGame.GoToRedsGameOver -= OnRedsGameOver;
                On.Player.checkInput -= OnCheckInput;
                On.Creature.Violence -= OnViolence;
                On.RWInput.PlayerInput_int -= OnPlayerInput;
                On.RWInput.PlayerUIInput_int -= OnPlayerUIInput;
                On.MainLoopProcess.RawUpdate -= OnRawUpdate;
                On.Player.Die -= OnPlayerDie;
                On.Player.Update -= OnPlayerUpdate;
                On.Player.Destroy -= OnPlayerDestroy;
                On.AbstractWorldEntity.Destroy -= OnAbstractDestroy;
                On.ShelterDoor.Close -= OnShelterClose;
                _hooked = false;
            }
            CloseRecorder();
            if (Bridge != null) { Bridge.Dispose(); Bridge = null; }
            if (_instanceLog != null) { BepInEx.Logging.Logger.Listeners.Remove(_instanceLog); _instanceLog = null; }
        }

        /// <summary>The frame's distance channel. Each pixel maps through the camera to its room tile
        /// and takes the slugcat's movement-graph distance there (PathInfo.HereDist), scaled per room:
        /// 255 at the slugcat, 1 at the farthest reachable tile, 0 unreachable or another room. Pixel
        /// (x, y), row 0 at the top, covers room position cam.pos plus its share of the camera's view
        /// (sSize), the same view the RGB was downscaled from.</summary>
        private void FillDistance(byte[] block, int off, int W, int H)
        {
            RainWorldGame game = _lockstep != null ? _lockstep.Game : null;
            if (game == null || game.cameras == null || game.cameras.Length == 0) return;
            RoomCamera cam = game.cameras[0];
            PathInfo pi = _lockstep.Obs.Paths;
            if (cam == null || cam.room == null || cam.room.abstractRoom.index != pi.HereRoom || pi.HereDist.Length == 0) return;
            int TW = pi.HereW, TH = pi.HereH;
            float sx = cam.sSize.x / W, sy = cam.sSize.y / H;
            for (int y = 0; y < H; y++)
            {
                int ty = (int)Math.Floor((cam.pos.y + (H - 1 - y + 0.5f) * sy) / 20f);
                for (int x = 0; x < W; x++)
                {
                    int tx = (int)Math.Floor((cam.pos.x + (x + 0.5f) * sx) / 20f);
                    byte v = 0;
                    if (tx >= 0 && ty >= 0 && tx < TW && ty < TH)
                    {
                        int dd = pi.HereDist[tx * TH + ty];
                        if (dd >= 0) v = (byte)(255 - (int)Math.Round(254.0 * dd / Math.Max(1, pi.HereMax)));
                    }
                    block[off + (y * W + x) * 4 + 3] = v;
                }
            }
        }

        private static string SchemaNow() { return Observation.SchemaJson(RainWorld.GAME_VERSION_STRING); }

        // -- recording and events ----------------------------------------------------------

        internal void Record(byte[] row)
        {
            if (!_logEnabled) return;
            try
            {
                if (_recorder == null)
                {
                    _recorder = new Recorder(DataDir(), SchemaNow);
                    Log.LogMessage("recording -> " + _recorder.BasePath + ".obs");
                }
                _recorder.Write(row, Observation.Size);
            }
            catch (Exception e) { Log.LogError("recorder write failed: " + e.Message); _logEnabled = false; }
        }

        internal void Emit(string json)
        {
            if (_recorder != null) _recorder.Event(json);
            if (Bridge != null && Bridge.Connected) Bridge.SendEvent(json);
        }

        private void CloseRecorder()
        {
            if (_recorder == null) return;
            _recorder.WriteSchemaSidecar();
            Log.LogMessage("session closed: " + _recorder.Records + " ticks -> " + _recorder.BasePath);
            _recorder.Dispose();
            _recorder = null;
        }

        private string DataDir()
        {
            string root = string.IsNullOrEmpty(_logDir.Value) ? Path.Combine(Paths.GameRootPath, "RainBotData") : _logDir.Value;
            return Instance.Isolated ? Path.Combine(root, "instance-" + Instance.Index) : root;
        }

        internal void MaybeDumpWorld(World world)
        {
            if (world == null || world.name == _dumpedWorld) return;
            _dumpedWorld = world.name;
            WorldDump.Dump(world, DataDir());
        }

        /// <summary>Export a room's static geometry the first time it is entered, once the AI map
        /// exists and any shelter door is fully open (a shut door bakes its tiles as rock).</summary>
        internal void MaybeDumpRoom(Room room)
        {
            if (room == null || room.abstractRoom == null || room.abstractRoom.index == _dumpedRoom) return;
            if (!room.readyForAI) return;
            if (room.shelterDoor != null && (room.shelterDoor.IsOpening || room.shelterDoor.aiMapper != null)) return;
            _dumpedRoom = room.abstractRoom.index;
            RoomDump.Dump(room, DataDir());
        }

        // -- hooks -------------------------------------------------------------------------

        private static bool IsAgentPlayer(Player self)
        {
            if (self == null || self.abstractCreature == null || self.abstractCreature.world == null) return false;
            if (self.abstractCreature.world.game == null) return false;
            PlayerState ps = self.playerState;
            return ps != null && ps.playerNumber == 0;
        }

        private void OnCheckInput(On.Player.orig_checkInput orig, Player self)
        {
            orig(self);
            if (IsAgentPlayer(self) && self.input != null)
            {
                _lockstep.NoteConsumed(self.input[0]);
                // Log why an injected action was dropped, once per episode. Player.checkInput zeroes
                // input when the slugcat is stunned, dead or asleep, the map button is held, or a
                // controller or AI drives the body.
                ActionPackage a = _lockstep.Pending;
                bool sent = _lockstep.Injecting && (a.x != 0 || a.y != 0 || a.jmp || a.thrw || a.pckp);
                Player.InputPackage got = self.input[0];
                if (sent && got.x == 0 && got.y == 0 && !got.jmp && !got.thrw && !got.pckp
                    && _droppedLogged != _lockstep.Episode)
                {
                    _droppedLogged = _lockstep.Episode;
                    Log.LogWarning("injected input dropped: stun=" + self.stun + " dead=" + self.dead + " sleeping=" + self.Sleeping
                                   + " mapHeld=" + (self.standStillOnMapButton && self.mapInput.mp) + " controller=" + (self.controller != null)
                                   + " AI=" + (self.AI != null) + " injectFired=" + _lockstep.InjectFired + " episode=" + _lockstep.Episode);
                }
            }
        }
        private int _droppedLogged = -1;

        /// <summary>
        /// Injection. Not scoped to checkInput: Player polls RWInput.PlayerInput from several
        /// places (SleepUpdate decides whether to wake in a shelter from it), and an agent whose
        /// input those cannot see never leaves the first shelter.
        /// </summary>
        private Player.InputPackage OnPlayerInput(On.RWInput.orig_PlayerInput_int orig, int playerNumber)
        {
            if (!_lockstep.Injecting || playerNumber != 0) return orig(playerNumber);
            _lockstep.InjectFired = true;
            Player.InputPackage pkg = orig(playerNumber);
            ActionPackage a = _lockstep.Pending;
            pkg.x = a.x;
            pkg.y = a.y;
            pkg.jmp = a.jmp;
            pkg.thrw = a.thrw;
            pkg.pckp = a.pckp;
            pkg.analogueDir = Vector2.zero;
            // Buttons the agent never presses. From the window's real input they can only interfere
            // (a held map button makes Player.checkInput drop all movement).
            pkg.mp = false;
            pkg.spec = false;
            pkg.crouchToggle = false;
            return pkg;
        }

        private Player.InputPackage OnPlayerUIInput(On.RWInput.orig_PlayerUIInput_int orig, int playerNumber)
        {
            Player.InputPackage pkg = orig(playerNumber);
            // Menus poll PlayerUIInput(-1) ("any player"), not (0).
            if (AutoPilot.Enabled && AutoPilot.WantUiPress && (playerNumber == 0 || playerNumber < 0))
                return AutoPilot.UiPress(pkg);
            return pkg;
        }

        private const float MaxDt = 0.1f;   // 4 frames at 40 Hz

        /// <summary>
        /// Runs at the end of RainWorldGame.RawUpdate (the `base.RawUpdate` call), after the game
        /// has set framesPerSecond, so the speed multiplier applied here is not overwritten.
        ///
        /// The dt clamp bounds the catch-up burst after a lockstep block (the game itself stops at
        /// 3 Updates; lockstep correctness does not depend on either). Ticks beyond the game's
        /// 3-per-frame cap run here when `max_ticks_per_frame` allows, so the render rate does not
        /// limit a fast sim.
        /// </summary>
        private void OnRawUpdate(On.MainLoopProcess.orig_RawUpdate orig, MainLoopProcess self, float dt)
        {
            bool isCurrent = self != null && self.manager != null && ReferenceEquals(self, self.manager.currentMainLoop);
            RainWorldGame game = self as RainWorldGame;
            if (DriveEnabled && Bridge != null && Bridge.Connected && dt > MaxDt) dt = MaxDt;
            if (game != null && (AutoPilot.Enabled || AutoPilot.SpeedMultiplier != 1f)) AutoPilot.ApplySpeed(game);
            // A native clip needs every tick drawn: the game's own RawUpdate runs up to 3 Updates
            // per rendered frame at a raised framesPerSecond, so drop to real time while armed.
            if (game != null && Frame != null && Frame.ClipArmed && game.framesPerSecond != 40) game.framesPerSecond = 40;
            if (AutoPilot.Enabled && isCurrent) AutoPilot.Update(self.manager);

            _lockstep.UpdatesThisFrame = 0;
            orig(self, dt);

            if (game != null && isCurrent && game.processActive && !game.GamePaused)
            {
                // A clip needs every tick drawn: one tick per frame.
                int cap = (Frame != null && Frame.ClipArmed) ? 1 : _maxTicksPerFrame.Value;
                int want = Math.Min(Math.Max(1, cap), (int)(dt * game.framesPerSecond));
                for (int i = _lockstep.UpdatesThisFrame; i < want && ReferenceEquals(game, game.manager.currentMainLoop); i++)
                {
                    try { game.manager.rainWorld.RunRewiredUpdate(); } catch (Exception) { }
                    game.Update();
                }
            }
        }

        /// <summary>Eating-speed curriculum (`eat_speed=K` reset key, default 1 = the real game).</summary>
        /// <remarks>
        /// While the slugcat holds food, `eatCounter` runs down by one on each qualifying tick (grab
        /// held, body still, every other input neutral; Player.cs ~9833/~10069/~10260). A bite fires
        /// at 0, with a 15-tick countdown after each bite. A stochastic policy rarely sustains that
        /// (DreamerV3, Hafner 2023, sped up block breaking in Minecraft for the same reason). With
        /// K &gt; 1, every tick the game counts down removes K-1 more, so the action is K times
        /// shorter and the game's own rules still decide which ticks count. The curriculum walks K
        /// back down to 1.
        /// </remarks>
        private void OnGrabUpdate(On.Player.orig_GrabUpdate orig, Player self, bool eu)
        {
            int before = self.eatCounter;
            orig(self, eu);
            int k = Lockstep.EatSpeed;
            if (k > 1 && IsAgentPlayer(self) && self.eatCounter == before - 1)
                self.eatCounter = Math.Max(0, self.eatCounter - (k - 1));
        }

        /// <summary>Sleep-speed curriculum (`sleep_speed=K` reset key, default 1 = the real game).</summary>
        /// <remarks>A shelter's door closes after the slugcat has stayed without any input for 41 ticks
        /// (touchedNoInputCounter, fed) or held down 261 ticks (forceSleepCounter, a malnourished sleep;
        /// Player.Update). Like eating, a sampling policy rarely sustains that: a fed slugcat on a sleep
        /// spot can outlive the rain without closing the door. Each tick the game counts adds K-1 more,
        /// so the game's rules still decide which ticks count. The curriculum walks K back to 1.</remarks>
        private void OnPlayerUpdate(On.Player.orig_Update orig, Player self, bool eu)
        {
            int k = Lockstep.SleepSpeed;
            if (k <= 1 || !IsAgentPlayer(self)) { orig(self, eu); return; }
            int still = self.touchedNoInputCounter, force = self.forceSleepCounter;
            orig(self, eu);
            if (self.touchedNoInputCounter == still + 1) self.touchedNoInputCounter += k - 1;
            if (self.forceSleepCounter == force + 1) self.forceSleepCounter += k - 1;
        }

        /// <summary>No controls-tutorial holograms while the agent drives.</summary>
        /// <remarks>On cycle 0 of a fresh save the Outskirts overseer projects a keyboard diagram of
        /// the controls (OverseerAI attaches OverseerTutorialBehavior in the tutorial rooms). Resets never
        /// advance the save past cycle 0, so it would appear in the agent's frames all the time, and
        /// it carries nothing the agent needs. Human play (no agent connected) keeps it.</remarks>
        private void OnOverseerTutorial(On.OverseerTutorialBehavior.orig_Update orig, OverseerTutorialBehavior self)
        {
            if (DriveEnabled && Bridge != null && Bridge.Connected) return;
            orig(self);
        }

        /// <summary>Where this instance's memory goes, by Unity asset type (the `memreport` command).
        /// Shows which per-instance memory saving is worth doing.</summary>
        internal static string MemoryReport()
        {
            System.Text.StringBuilder sb = new System.Text.StringBuilder("memreport:");
            sb.Append(" unity_allocated=").Append(UnityEngine.Profiling.Profiler.GetTotalAllocatedMemoryLong() >> 20).Append("MB");
            sb.Append(" unity_reserved=").Append(UnityEngine.Profiling.Profiler.GetTotalReservedMemoryLong() >> 20).Append("MB");
            sb.Append(" mono_used=").Append(UnityEngine.Profiling.Profiler.GetMonoUsedSizeLong() >> 20).Append("MB");
            sb.Append(" mono_heap=").Append(UnityEngine.Profiling.Profiler.GetMonoHeapSizeLong() >> 20).Append("MB");
            // Every live object, grouped by type, largest 12 types (count after the slash).
            Dictionary<string, long> bytes = new Dictionary<string, long>();
            Dictionary<string, int> count = new Dictionary<string, int>();
            foreach (UnityEngine.Object o in Resources.FindObjectsOfTypeAll(typeof(UnityEngine.Object)))
            {
                string n = o.GetType().Name;
                long sz = UnityEngine.Profiling.Profiler.GetRuntimeMemorySizeLong(o);
                bytes[n] = (bytes.TryGetValue(n, out long b0) ? b0 : 0) + sz;
                count[n] = (count.TryGetValue(n, out int c0) ? c0 : 0) + 1;
            }
            List<KeyValuePair<string, long>> top = new List<KeyValuePair<string, long>>(bytes);
            top.Sort((x, y) => y.Value.CompareTo(x.Value));
            for (int i = 0; i < Math.Min(12, top.Count); i++)
                sb.Append(' ').Append(top[i].Key).Append('=').Append(top[i].Value >> 20).Append("MB/").Append(count[top[i].Key]);
            long all = 0; foreach (long v in bytes.Values) all += v;
            sb.Append(" objects_total=").Append(all >> 20).Append("MB");
            return sb.ToString();
        }

        private void OnGameGrafUpdate(On.RainWorldGame.orig_GrafUpdate orig, RainWorldGame self, float timeStacker)
        {
            if (_lockstep.RenderOff && !self.GamePaused) return;
            // Drawing must not consume the simulation's random numbers. Graphics code calls
            // UnityEngine.Random once per drawn frame, and the number of frames per tick depends on
            // speed and load, so runs at >1x would diverge (docs/PROTOCOL.md, Determinism). Saving
            // and restoring the generator's state around the draw makes the simulation's random
            // sequence depend on ticks alone. Drawing still gets varied numbers: it continues from
            // the simulation's current state each frame.
            UnityEngine.Random.State sim = UnityEngine.Random.state;
            try { orig(self, timeStacker); }
            finally { UnityEngine.Random.state = sim; }
        }

        private void OnGameShutDown(On.RainWorldGame.orig_ShutDownProcess orig, RainWorldGame self)
        {
            try { _lockstep.OnShutDown(self); } catch (Exception e) { Log.LogWarning("session end failed: " + e.Message); }
            orig(self);
        }

        private void OnWin(On.RainWorldGame.orig_Win orig, RainWorldGame self, bool malnourished, bool fromWarpPoint)
        {
            _lockstep.NoteEndReason("sleep");
            orig(self, malnourished, fromWarpPoint);
        }

        // -- endings taken over by the mod (Lockstep.EndNow; `endings freeze`) -------------------

        /// <summary>A death of the agent's slugcat: booked and frozen instead of run (the game's own
        /// invincibility setting returns from Player.Die at the same point).</summary>
        private void OnPlayerDie(On.Player.orig_Die orig, Player self)
        {
            if (IsAgentPlayer(self) && !self.dead && _lockstep.ResetShielded) return;
            if (IsAgentPlayer(self) && !self.dead && (_lockstep.Frozen || _lockstep.Intercepts(self)))
            {
                _lockstep.EndNow(self.abstractCreature.world.game, "death", false);
                return;
            }
            orig(self);
        }

        /// <summary>Falling out of the room calls Die then Destroy: the slugcat stays (frozen).</summary>
        private void OnPlayerDestroy(On.Player.orig_Destroy orig, Player self)
        {
            // Also during a reset: the slugcat still lies below the room until it is placed, and the
            // game re-runs its fall-out code (Die, Destroy) every tick.
            if (IsAgentPlayer(self) && (_lockstep.Frozen || _lockstep.ResetShielded)) return;
            orig(self);
        }

        private void OnAbstractDestroy(On.AbstractWorldEntity.orig_Destroy orig, AbstractWorldEntity self)
        {
            AbstractCreature ac = self as AbstractCreature;
            if (ac != null && (_lockstep.Frozen || _lockstep.ResetShielded) && ac.realizedCreature is Player && IsAgentPlayer((Player)ac.realizedCreature)) return;
            orig(self);
        }

        /// <summary>The shelter door starting to close is the night's decision; the game settles it
        /// when the door is shut (ShelterDoor.Update / DoorClosed): food in the room below 1 (below
        /// max food if already malnourished) starves, otherwise a sleep, malnourished below the
        /// hibernation requirement. Booked now; the door does not move.</summary>
        private void OnShelterClose(On.ShelterDoor.orig_Close orig, ShelterDoor self)
        {
            Player p = self.room != null ? self.room.game.RealizedPlayerOfPlayerNumber(0) : null;
            if (_lockstep.ResetShielded || _lockstep.Frozen) return;          // the door stays open
            if (p != null && IsAgentPlayer(p) && p.room == self.room && !self.Broken && _lockstep.Intercepts(p))
            {
                try
                {
                    bool maln = self.room.game.GetStorySession.saveState.malnourished;
                    int food = p.FoodInRoom(self.room, false);
                    if (food < (maln ? p.slugcatStats.maxFood : 1)) _lockstep.EndNow(self.room.game, "starve", maln);
                    else
                    {
                        // The night's food as SaveState books it: FoodInRoom(eatAndDestroy: true) eats
                        // every edible in the shelter (the fruit in hand too), and that count is the
                        // stomach. Without it a slugcat that carries its fruit in is booked fed (the
                        // door counts the fruit) but its stomach stays a pip short.
                        int eaten = p.FoodInRoom(self.room, true);
                        if (eaten > p.playerState.foodInStomach)
                        {
                            p.playerState.foodInStomach = Math.Min(eaten, p.slugcatStats.maxFood);
                            p.playerState.quarterFoodPoints = 0;
                        }
                        _lockstep.EndNow(self.room.game, "sleep", food < p.slugcatStats.foodToHibernate);
                    }
                    return;
                }
                catch (Exception e) { Log.LogWarning("shelter ending not taken over: " + e.Message); }
            }
            orig(self);
        }

        private void OnDeathScreen(On.RainWorldGame.orig_GoToDeathScreen orig, RainWorldGame self)
        {
            _lockstep.NoteEndReason("death");
            orig(self);
        }

        private void OnStarveScreen(On.RainWorldGame.orig_GoToStarveScreen orig, RainWorldGame self)
        {
            _lockstep.NoteEndReason("starve");
            orig(self);
        }

        private void OnRedsGameOver(On.RainWorldGame.orig_GoToRedsGameOver orig, RainWorldGame self)
        {
            _lockstep.NoteEndReason("death");
            orig(self);
        }

        /// <summary>Record damage the slugcat dealt or took, attributed exactly: the damaging
        /// chunk's owner is the attacker, or a weapon that remembers who threw it.</summary>
        private void OnViolence(On.Creature.orig_Violence orig, Creature self, BodyChunk source,
                                Vector2? directionAndMomentum, BodyChunk hitChunk,
                                PhysicalObject.Appendage.Pos hitAppendage,
                                Creature.DamageType type, float damage, float stunBonus)
        {
            orig(self, source, directionAndMomentum, hitChunk, hitAppendage, type, damage, stunBonus);
            try
            {
                Player me = (self != null && self.abstractCreature != null && self.abstractCreature.world != null
                             && self.abstractCreature.world.game != null)
                    ? self.abstractCreature.world.game.RealizedPlayerOfPlayerNumber(0) : null;
                if (me == null) return;
                ObsWriter o = _lockstep.Obs;
                if (self == me) { o.HitTaken++; return; }
                PhysicalObject owner = (source != null) ? source.owner : null;
                bool ours = owner == me;
                Weapon w = owner as Weapon;
                if (w != null && w.thrownBy == me) ours = true;
                if (!ours) return;
                o.HitDealt++;
                if (self.abstractCreature.creatureTemplate != null)
                    o.HitVictim = self.abstractCreature.creatureTemplate.type.Index;
            }
            catch (Exception) { }
        }
    }

    internal sealed class InstanceLog : BepInEx.Logging.ILogListener
    {
        private readonly string _path;
        public InstanceLog()
        {
            string dir = System.IO.Path.Combine(BepInEx.Paths.GameRootPath, "RainBotData");
            System.IO.Directory.CreateDirectory(dir);
            _path = System.IO.Path.Combine(dir, "instance" + Instance.Index + ".log");
        }
        public void LogEvent(object sender, BepInEx.Logging.LogEventArgs e)
        {
            if (e.Level > BepInEx.Logging.LogLevel.Warning && !(e.Data is string str && (str.Contains("agent") || str.Contains("reset"))))
                return;
            try
            {
                System.IO.File.AppendAllText(_path, DateTime.Now.ToString("HH:mm:ss.fff") + " [" + e.Level + "] " + e.Data + "\n");
            }
            catch (Exception) { }
        }
        public void Dispose() { }
    }
}
