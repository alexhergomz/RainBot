using System;
using System.Globalization;
using System.Text;

namespace RainBot
{
    /// <summary>
    /// Chunked lockstep at simulation-tick granularity (docs/PROTOCOL.md "Timing").
    ///
    /// A tick is one un-paused `RainWorldGame.Update` of a story session whose player 0 is
    /// realized. It is counted here, not in `Player.checkInput`, because checkInput does not run
    /// while the slugcat is inside a shortcut pipe: an exchange per checkInput would go silent for
    /// 54-80 frames on every room change and give the agent one transition spanning all of them.
    ///
    /// Every 8 ticks the 8 post-tick rows go out as one MSG_CHUNK and the main thread blocks until
    /// 4 decisions come back; decision k is injected for ticks 2k and 2k+1 of the next chunk.
    /// </summary>
    internal sealed class Lockstep
    {
        private readonly Plugin _plugin;
        private readonly ObsWriter _obs = new ObsWriter();
        private readonly byte[] _chunk = new byte[Observation.TicksPerStep * Observation.Size];
        private int _count;
        private readonly ActionPackage[] _decisions = new ActionPackage[Observation.DecisionsPerStep];
        private bool _haveDecisions;

        // Process-lifetime counters: they survive game-session changes (a respawn or a new cycle
        // is a new RainWorldGame). `tick` never resets, so it is unique per launch and is what the
        // video tick marker draws. `episode` changes only on `reset` (episodes run through deaths
        // and hibernation), and `session` on every new RainWorldGame.
        private static int _tick, _episode, _session;
        public static int Tick => _tick;
        public int UpdatesThisFrame;
        private RainWorldGame _game;
        public RainWorldGame Game => _game;
        private World _worldSent;
        private int _connSeen;
        private Reset _reset;
        private bool _peaceful;
        private string _endReason;

        // Endings without the game's screens. With `endings freeze` and an agent driving, a death,
        // a starvation or a sleep does not leave the game: the mod books the night by the game's
        // own rules (karma, malnourishment), sends the ending's rows, then one booked token (the
        // state the next cycle's first token would carry), and holds the world frozen until the
        // next reset. No death screen, no cycle reload, no waking in a shelter. Off by default
        // (the game's own flow).
        public bool FreezeEndings;
        private string _frozen;
        private bool _frozenAnnounced;
        public bool Frozen => _frozen != null;
        /// <summary>A reset is running while endings are frozen: nothing may end the episode mid-teleport.</summary>
        public bool ResetShielded => FreezeEndings && _reset != null;

        // Per-tick injection state, read by the RWInput hook and the checkInput hook.
        public bool Injecting;
        public ActionPackage Pending;
        public int Episode => _episode;
        public bool InjectFired;
        private bool _consumed;
        private readonly sbyte[] _action = new sbyte[Observation.ActionBytes];

        // Event bookkeeping.
        private int _lastRoom = -1, _lastRowRoom = -1, _lastKarma = -1;
        private bool _wasDead;

        public bool RenderOff;
        public bool Peaceful { get { return _peaceful; } set { _peaceful = value; } }

        public Lockstep(Plugin plugin) { _plugin = plugin; }

        public ObsWriter Obs => _obs;

        /// <summary>Player 0 if this Update is a simulation tick, else null.</summary>
        private static Player SimPlayer(RainWorldGame g)
        {
            if (g == null || !g.IsStorySession || g.GamePaused || !g.processActive) return null;
            if (g.world == null || g.session == null || g.Players == null) return null;
            Player p = g.RealizedPlayerOfPlayerNumber(0);
            if (p == null || p.abstractCreature == null || p.abstractCreature.world == null) return null;
            if (p.room == null && !p.inShortcut) return null;     // not placed anywhere yet
            return p;
        }

        // -- the tick ----------------------------------------------------------------------

        public void OnGameUpdate(On.RainWorldGame.orig_Update orig, RainWorldGame game)
        {
            UpdatesThisFrame++;
            Player p = SimPlayer(game);
            if (_reset != null && ++_resetAge % 400 == 0) ResetStuckReport(game, p);
            // A living slugcat in no room (it fell out of the bottom of one) still has to finish a
            // frozen ending (its last rows and booked token) and take its next reset. Skipping it
            // leaves the trainer waiting and the game running free in the rain.
            if (p == null && (_reset != null || _frozen != null)) p = RoomlessPlayer(game);
            if (p == null)
            {
                Injecting = false;
                orig(game);
                return;
            }

            Bridge bridge = _plugin.Bridge;
            bool connected = bridge != null && bridge.Connected;

            if (!ReferenceEquals(game, _game)) StartSession(game);
            if (game.world != _worldSent) SendWorld(game.world);
            if (bridge != null && bridge.ConnectionId != _connSeen)
            {
                // A new agent: give it the world table and start its first chunk clean.
                _connSeen = bridge.ConnectionId;
                _count = 0;
                _haveDecisions = false;
                SendWorld(game.world);
            }

            if (bridge != null) bridge.PollCommands();
            ApplyCommands(p, game);
            if (_obs.Paths.SurveyComplete && _waterSentConn != _connSeen) EmitWaterRooms();
            FlushEnumEvents();
            if (_frozen != null && _reset != null) Unfreeze();
            if (_frozen != null && !connected)
            {
                // The agent left mid-ending: hold the world frozen (no rows, nothing to send) until
                // one connects and resets it. Handing the ending back to the game would run its
                // death/sleep/dream screens, which freezing avoids.
                Injecting = false;
                _frozenAnnounced = true;
                _count = 0;
                return;
            }
            if (_frozen != null) { FrozenTick(p, game, bridge); return; }

            if (_reset != null)
            {
                Reset.Status st = _reset.Step(p, game);
                if (st == Reset.Status.Pending)
                {
                    // Not lockstepped: the game runs freely, with neutral input, until it lands.
                    Injecting = _plugin.DriveEnabled && connected;
                    Pending = default(ActionPackage);
                    InjectFired = false;
                    orig(game);
                    if (_peaceful) Pacify(p);
                    return;
                }
                if (st == Reset.Status.Done)
                {
                    ZeroPhaseCounters(game);
                    NewEpisode();
                    if (!_reset.HasSeed) ApplyGlobalSeed("reset, episode " + _episode);
                    Emit("{\"type\":\"reset_done\",\"t\":" + _tick + ",\"episode\":" + _episode + ",\"session\":" + _session
                         + ",\"room\":\"" + Esc(_reset.RoomName) + "\""
                         + ",\"food\":" + p.playerState.foodInStomach
                         + ",\"karma\":" + Karma(game)
                         + ",\"timer\":" + game.world.rainCycle.timer
                         + ",\"spawned\":[" + _reset.SpawnedJson + "]"
                         + ",\"tile\":" + _reset.StartTileJson + "}");
                    Plugin.Log.LogMessage("reset done -> " + _reset.RoomName + " (episode " + _episode + ")");
                }
                else
                {
                    Emit("{\"type\":\"reset_failed\",\"reason\":\"" + Esc(_reset.Error) + "\"}");
                    Plugin.Log.LogWarning("reset failed: " + _reset.Error);
                }
                _reset = null;
            }

            // Exchange: the previous chunk out, the next 4 decisions in.
            if (_plugin.DriveEnabled && connected && _count == Observation.TicksPerStep)
            {
                FlushEnumEvents();
                FrameCapture fc = _plugin.Frame;
                bool ok = bridge.ExchangeChunk(_chunk, _count, Observation.Size, _decisions,
                                               fc != null ? fc.Block : null, FrameLen(fc));
                _count = 0;
                _haveDecisions = ok;
                connected = ok;
                // Commands that arrived while blocked take effect on the tick the agent meant.
                ApplyCommands(p, game);
                if (_reset != null)
                {
                    // Started by this exchange: it runs from the next tick, and this tick's row
                    // must not leak into the new episode's first chunk.
                    Injecting = false;
                    orig(game);
                    return;
                }
            }
            else if (_count == Observation.TicksPerStep)
            {
                _count = 0;          // no agent to send it to; keep going
            }

            int stepTick = _count;
            Injecting = _plugin.DriveEnabled && connected;
            Pending = (Injecting && _haveDecisions) ? _decisions[stepTick / Observation.TicksPerDecision]
                                                     : default(ActionPackage);
            InjectFired = false;
            _consumed = false;

            orig(game);

            // The tick may have ended the session (death screen request etc.), but the player
            // object is still valid for one more row.
            if (_peaceful) Pacify(p);

            if (!_consumed)
            {
                if (Injecting) Put(_action, Pending);
                else Array.Clear(_action, 0, _action.Length);
            }

            AbstractRoom aroom; Room room; UnityEngine.Vector2 pos; bool pipe;
            ObsWriter.Locate(p, out room, out aroom, out pos, out pipe);
            int roomIdx = aroom != null ? aroom.index : -1;
            bool changed = _lastRowRoom != -1 && roomIdx != _lastRowRoom;
            _lastRowRoom = roomIdx;

            _obs.Write(p, game, _tick, _episode, _session, stepTick, changed, _consumed, _action, InjectFired);
            Buffer.BlockCopy(_obs.Row, 0, _chunk, _count * Observation.Size, Observation.Size);
            _count++;
            _tick++;
            _plugin.Record(_obs.Row);
            MaybeDump(p, room, game);
            EmitEvents(p, game, aroom);
        }

        /// <summary>checkInput ran for player 0: record what the sim consumed. The game rewrites
        /// input[0] on advanced-movement frames (pole launches, rolls), so the requested action is
        /// not a correct label there.</summary>
        public void NoteConsumed(Player.InputPackage c)
        {
            _consumed = true;
            _action[0] = (sbyte)c.x;
            _action[1] = (sbyte)c.y;
            _action[2] = (sbyte)(c.jmp ? 1 : 0);
            _action[3] = (sbyte)(c.thrw ? 1 : 0);
            _action[4] = (sbyte)(c.pckp ? 1 : 0);
        }

        private static void Put(sbyte[] a, ActionPackage p)
        {
            a[0] = (sbyte)p.x; a[1] = (sbyte)p.y;
            a[2] = (sbyte)(p.jmp ? 1 : 0); a[3] = (sbyte)(p.thrw ? 1 : 0); a[4] = (sbyte)(p.pckp ? 1 : 0);
        }

        // -- sessions and episodes ---------------------------------------------------------

        private static readonly System.Reflection.FieldInfo[] PhaseFields =
        {
            typeof(RainWorldGame).GetField("updateShortCut",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance),
            typeof(RainWorldGame).GetField("updateAbstractRoom",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance),
        };

        /// <summary>Zero the game's update-phase counters when a reset lands.</summary>
        /// <remarks>
        /// RainWorldGame.Update moves creatures through pipes only every third tick
        /// (`updateShortCut`, mod 3) and walks abstract rooms round-robin (`updateAbstractRoom`).
        /// A reset does not touch either, and the reset itself runs outside lockstep for a number of
        /// ticks that depends on sim speed, so each episode would start at an arbitrary phase. Then
        /// the same seed and actions at 1x and 16x diverge on the first pipe entry. Zeroing both
        /// makes a reset reproducible.
        /// </remarks>
        private static void ZeroPhaseCounters(RainWorldGame game)
        {
            foreach (System.Reflection.FieldInfo f in PhaseFields)
            {
                try { if (f != null) f.SetValue(game, 0); }
                catch (Exception e) { Plugin.Log.LogWarning("phase reset failed: " + e.Message); }
            }
        }

        /// <summary>Bytes of frame to append: always the full block when frames are configured, so
        /// the chunk size is constant; before the first capture it carries frame_tick -1 and black.</summary>
        private static int FrameLen(FrameCapture fc)
        {
            if (fc == null) return 0;
            if (!fc.HaveFrame) Observation.PutI32(fc.Block, 0, -1);
            return fc.Block.Length;
        }

        /// <summary>Run-level seed (`global_seed N`, -1 = off). When set, every new game session and
        /// every reset re-seeds UnityEngine.Random from (seed, episode, session), so a run depends only
        /// on the seed and the agent's actions. This allows replays, GRPO groups from identical states,
        /// and reproducible bugs.</summary>
        public static int GlobalSeed = -1;
        /// <summary>Eating-speed multiplier for the agent's slugcat; set by `reset eat_speed=K`.</summary>
        public static int EatSpeed = 1;
        /// <summary>Sleep-counter multiplier (`reset sleep_speed=K`): the game's still-for-the-door and
        /// hold-down-to-sleep counters advance K per qualifying tick. 1 = the real game.</summary>
        public static int SleepSpeed = 1;

        public static void ApplyGlobalSeed(string why)
        {
            if (GlobalSeed < 0) return;
            unchecked
            {
                int s = GlobalSeed * 1000003 + _episode * 7919 + _session * 104729;
                UnityEngine.Random.InitState(s);
                Plugin.Log.LogMessage("global seed -> " + s + " (" + why + ")");
            }
        }

        private void StartSession(RainWorldGame game)
        {
            _game = game;
            _endReason = null;
            _lastRoom = -1;
            _wasDead = false;
            _session++;
            ApplyGlobalSeed("session " + _session);
            Reset.Spawned.Clear();
            ClearChunk();
            Emit("{\"type\":\"session_start\",\"t\":" + _tick + ",\"session\":" + _session + ",\"episode\":" + _episode + "}");
        }

        private void NewEpisode()
        {
            _episode++;
            ClearChunk();
        }

        private void ClearChunk()
        {
            _count = 0;
            _haveDecisions = false;
            _lastRowRoom = -1;
        }

        private int _resetAge;
        private int _waterSentConn = -1;

        private void EmitWaterRooms()
        {
            StringBuilder sb = new StringBuilder("{\"type\":\"water_rooms\",\"complete\":");
            sb.Append(_obs.Paths.SurveyComplete ? "true" : "false").Append(",\"rooms\":[");
            for (int i = 0; i < _obs.Paths.WaterRooms.Count; i++)
            {
                if (i > 0) sb.Append(',');
                sb.Append('"').Append(Esc(_obs.Paths.WaterRooms[i])).Append('"');
            }
            Emit(sb.Append("]}").ToString());
            Plugin.Log.LogMessage("water rooms sent: " + _obs.Paths.WaterRooms.Count + " (" + string.Join(",", _obs.Paths.WaterRooms.ToArray()) + ")");
            if (_obs.Paths.SurveyComplete) _waterSentConn = _connSeen;
        }

        /// <summary>A living player 0 that is in no room and not in a pipe. After it falls out of the
        /// bottom of a room, the game takes it out of the room (and the mod's frozen ending keeps it
        /// from being destroyed). SimPlayer skips it, so a pending reset would never step. The reset's
        /// own entry path brings it into the target room through a pipe.</summary>
        private static Player RoomlessPlayer(RainWorldGame g)
        {
            if (g == null || !g.IsStorySession || g.GamePaused || !g.processActive || g.world == null) return null;
            Player p = g.RealizedPlayerOfPlayerNumber(0);
            if (p == null || p.dead || p.abstractCreature == null || p.abstractCreature.world == null) return null;
            return (p.room == null && !p.inShortcut) ? p : null;
        }

        /// <summary>Log what a reset waits on, every 400 ticks (10 s) while it is pending, so a reset
        /// that never finishes can be diagnosed.</summary>
        private void ResetStuckReport(RainWorldGame g, Player sim)
        {
            try
            {
                Player p0 = g != null ? g.RealizedPlayerOfPlayerNumber(0) : null;
                Plugin.Log.LogWarning("reset pending " + _resetAge + " ticks: phase " + _reset.PhaseName
                    + " simPlayer=" + (sim != null) + " paused=" + (g != null && g.GamePaused) + " active=" + (g != null && g.processActive)
                    + " player=" + (p0 != null) + " room=" + (p0 != null && p0.room != null ? p0.room.abstractRoom.name : "null")
                    + " inShortcut=" + (p0 != null && p0.inShortcut) + " dead=" + (p0 != null && p0.dead)
                    + " slated=" + (p0 != null && p0.slatedForDeletetion)
                    + " target=" + _reset.RoomName);
            }
            catch (Exception e) { Plugin.Log.LogWarning("reset pending " + _resetAge + " ticks (report failed: " + e.Message + ")"); }
        }

        public void NoteEndReason(string reason) { if (_endReason == null) _endReason = reason; }

        /// <summary>Whether an ending of this player is taken over by the mod right now.</summary>
        public bool Intercepts(Player p)
        {
            Bridge b = _plugin.Bridge;
            return FreezeEndings && _plugin.DriveEnabled && b != null && b.Connected && _game != null && _reset == null
                   && p != null && p.abstractCreature != null && ReferenceEquals(p.abstractCreature.world.game, _game);
        }

        /// <summary>The night, booked as the game would (SaveState.SessionEnded and the death save in
        /// DeathPersistentSaveData): a death or a starvation costs one karma unless a karma flower
        /// protects it (then the protection is spent); a sleep gives one up to the cap, and a
        /// malnourished sleep also spends the protection and leaves the slugcat malnourished.</summary>
        public void EndNow(RainWorldGame game, string reason, bool malnourished)
        {
            if (_frozen != null) return;
            try
            {
                SaveState ss = (game.session as StoryGameSession).saveState;
                DeathPersistentSaveData d = ss.deathPersistentSaveData;
                if (reason == "sleep")
                {
                    d.karma = Math.Min(d.karma + 1, d.karmaCap);
                    if (malnourished) d.reinforcedKarma = false;
                    ss.malnourished = malnourished;
                }
                else if (d.reinforcedKarma) d.reinforcedKarma = false;
                else d.karma = Math.Max(0, d.karma - 1);
            }
            catch (Exception e) { Plugin.Log.LogWarning("ending: karma not booked: " + e.Message); }
            _frozen = reason;
            _frozenAnnounced = false;
            _obs.ForceDead = reason != "sleep";
            _obs.WorldFrozen = true;
            Plugin.Log.LogMessage("ending frozen: " + reason + (malnourished ? " (malnourished)" : ""));
        }

        private void Unfreeze()
        {
            _obs.WorldFrozen = false;
            _obs.ForceDead = false;
            _frozen = null;
            _frozenAnnounced = false;
            _count = 0;
            _haveDecisions = false;
        }

        /// <summary>One tick of a frozen ending: the world does not advance; rows of the frozen state
        /// are sent in lockstep chunks (the trainer has stopped that member and waits for its group).</summary>
        private void FrozenTick(Player p, RainWorldGame game, Bridge bridge)
        {
            Injecting = false;
            if (!_frozenAnnounced)
            {
                // The ending's own rows (1..8), then the booked token, as for a session change.
                Emit("{\"type\":\"session_end\",\"t\":" + _tick + ",\"reason\":\"" + _frozen + "\",\"session\":" + _session + ",\"episode\":" + _episode + ",\"frozen\":true}");
                SendFinalChunk(bridge);
                _count = 0;
                _haveDecisions = false;
                _session++;
                Emit("{\"type\":\"session_start\",\"t\":" + _tick + ",\"session\":" + _session + ",\"episode\":" + _episode + ",\"frozen\":true}");
                _frozenAnnounced = true;
            }
            if (_count == Observation.TicksPerStep)
            {
                FlushEnumEvents();
                FrameCapture fc = _plugin.Frame;
                bool ok = bridge != null && bridge.ExchangeChunk(_chunk, _count, Observation.Size, _decisions,
                                                                 fc != null ? fc.Block : null, FrameLen(fc));
                _count = 0;
                _haveDecisions = ok;
                ApplyCommands(p, game);
                if (_reset != null) { Unfreeze(); return; }
                if (!ok) return;                                   // agent gone: stay frozen (above)
            }
            Array.Clear(_action, 0, _action.Length);
            _obs.Write(p, game, _tick, _episode, _session, _count, false, false, _action, false);
            Buffer.BlockCopy(_obs.Row, 0, _chunk, _count * Observation.Size, Observation.Size);
            _count++;
            _tick++;
            _plugin.Record(_obs.Row);
            EmitKarmaChange(game);
        }

        /// <summary>Send the buffered rows (1..8) of an ending session as a final, unanswered chunk.</summary>
        private void SendFinalChunk(Bridge bridge)
        {
            if (_plugin.DriveEnabled && bridge != null && bridge.Connected && _count > 0)
                bridge.SendFinalChunk(_chunk, _count, Observation.Size, _plugin.Frame != null ? _plugin.Frame.Block : null,
                                     FrameLen(_plugin.Frame));
        }

        /// <summary>The game process is leaving RainWorldGame: emit `session_end` first, then flush
        /// the buffered rows (1..8) as a final MSG_CHUNK that is not an exchange: the mod does not
        /// wait for an action after it. Nothing is sent if the buffer is empty.</summary>
        public void OnShutDown(RainWorldGame game)
        {
            if (!ReferenceEquals(game, _game)) return;
            string reason = _endReason;
            if (reason == null)
            {
                Player p = game.RealizedPlayerOfPlayerNumber(0);
                reason = (p != null && p.dead) ? "death" : "quit";
            }
            Emit("{\"type\":\"session_end\",\"t\":" + _tick + ",\"reason\":\"" + reason + "\",\"session\":" + _session + ",\"episode\":" + _episode + "}");
            SendFinalChunk(_plugin.Bridge);
            if (_plugin.Frame != null) _plugin.Frame.ClipStop();
            _count = 0;
            _haveDecisions = false;
            _game = null;
            _reset = null;
            Injecting = false;
        }

        // -- commands ----------------------------------------------------------------------

        private void ApplyCommands(Player self, RainWorldGame game)
        {
            Bridge bridge = _plugin.Bridge;
            if (bridge == null) return;
            string cmd;
            while ((cmd = bridge.NextCommand()) != null)
            {
                string[] parts = cmd.Trim().Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length == 0) continue;
                try
                {
                    switch (parts[0])
                    {
                        case "ping":
                            break;
                        case "kill":
                            Plugin.Log.LogMessage("cmd: kill");
                            self.Die();
                            break;
                        case "speed":
                            float sp;
                            if (parts.Length > 1 && float.TryParse(parts[1], NumberStyles.Float, CultureInfo.InvariantCulture, out sp))
                            {
                                _plugin.AutoPilot.SpeedMultiplier = sp;
                                Plugin.Log.LogMessage("cmd: speed " + sp);
                            }
                            break;
                        case "global_seed":
                            int gs;
                            if (parts.Length > 1 && int.TryParse(parts[1], NumberStyles.Integer, CultureInfo.InvariantCulture, out gs))
                            {
                                GlobalSeed = gs;
                                Plugin.Log.LogMessage("cmd: global_seed " + gs);
                            }
                            break;
                        case "clip":
                            // clip arm | clip off | clip dump <path>   (native-resolution clips)
                            if (_plugin.Frame == null) break;
                            if (parts.Length > 1 && parts[1] == "arm")
                                _plugin.Frame.ClipArm(parts.Length > 2 ? int.Parse(parts[2]) : 400);
                            else if (parts.Length > 1 && parts[1] == "off") _plugin.Frame.ClipStop(true);
                            else if (parts.Length > 2 && parts[1] == "dump")
                            {
                                string path = cmd.Trim().Substring(cmd.Trim().IndexOf("dump", StringComparison.Ordinal) + 5);
                                int n = 0;
                                try { n = _plugin.Frame.ClipDump(path); }
                                catch (Exception e) { Plugin.Log.LogWarning("clip dump failed: " + e.Message); }
                                Emit("{\"type\":\"clip_dumped\",\"path\":\"" + Esc(path) + "\",\"frames\":" + n + "}");
                            }
                            break;
                        case "timer_left":
                            // Set the rain timer of the running cycle to N ticks left (a reset can also
                            // set it from its spec).
                            int tl;
                            if (parts.Length > 1 && int.TryParse(parts[1], NumberStyles.Integer, CultureInfo.InvariantCulture, out tl))
                            {
                                RainCycle rc = game.world.rainCycle;
                                rc.timer = Math.Max(0, rc.cycleLength - Math.Max(0, tl));
                                Plugin.Log.LogMessage("cmd: timer_left " + tl);
                            }
                            break;
                        case "water_rooms":
                            EmitWaterRooms();
                            break;
                        case "endings":
                            FreezeEndings = parts.Length > 1 && parts[1] == "freeze";
                            Plugin.Log.LogMessage("cmd: endings " + (FreezeEndings ? "freeze" : "game"));
                            break;
                        case "reload":
                            Plugin.Log.LogMessage("cmd: reload (the agent link closes; reconnect)");
                            _plugin.RequestReload();
                            break;
                        case "pathdebug":
                            string pd = _obs.Paths.Debug(self, self.room);
                            Plugin.Log.LogMessage(pd);
                            Emit("{\"type\":\"memreport\",\"text\":\"" + Esc(pd) + "\"}");
                            break;
                        case "memreport":
                            string rep = Plugin.MemoryReport();
                            Plugin.Log.LogMessage(rep);
                            Emit("{\"type\":\"memreport\",\"text\":\"" + Esc(rep) + "\"}");
                            break;
                        case "frame":
                            if (_plugin.Frame != null) _plugin.Frame.Enabled = parts.Length > 1 && parts[1] == "on";
                            break;
                        case "render":
                            RenderOff = parts.Length > 1 && parts[1] == "off";
                            Plugin.Log.LogMessage("cmd: render " + (RenderOff ? "off" : "on"));
                            break;
                        case "seed":
                            int seed;
                            if (parts.Length > 1 && int.TryParse(parts[1], NumberStyles.Integer, CultureInfo.InvariantCulture, out seed))
                            {
                                UnityEngine.Random.InitState(seed);
                                Plugin.Log.LogMessage("cmd: seed " + seed);
                            }
                            break;
                        case "den":
                            // Set the respawn shelter in both save-state objects: the respawn path
                            // restores from progression.currentSaveState, so setting only the
                            // session's copy is undone by the death that should apply it. With `kill`
                            // after it, this moves the slugcat to another region (`reset` works only
                            // within the loaded world).
                            if (parts.Length > 1 && game.GetStorySession != null)
                            {
                                game.GetStorySession.saveState.denPosition = parts[1];
                                SaveState prog = game.rainWorld.progression != null
                                    ? game.rainWorld.progression.currentSaveState : null;
                                if (prog != null) prog.denPosition = parts[1];
                                Plugin.Log.LogMessage("cmd: den " + parts[1]);
                            }
                            break;
                        case "peaceful":
                            _peaceful = parts.Length > 1 && parts[1] == "on";
                            Plugin.Log.LogMessage("cmd: peaceful " + (_peaceful ? "on" : "off"));
                            break;
                        case "autopilot":
                            _plugin.AutoPilot.Enabled = parts.Length > 1 && parts[1] == "on";
                            Plugin.Log.LogMessage("cmd: autopilot " + (_plugin.AutoPilot.Enabled ? "on" : "off"));
                            break;
                        case "reset":
                            string err;
                            Reset r = Reset.Parse(parts, out err);
                            if (r == null)
                            {
                                Emit("{\"type\":\"reset_failed\",\"reason\":\"" + Esc(err) + "\"}");
                                Plugin.Log.LogWarning("reset rejected: " + err);
                            }
                            else
                            {
                                _reset = r; _resetAge = 0;
                                Plugin.Log.LogMessage("cmd: " + cmd.Trim());
                            }
                            break;
                        default:
                            Plugin.Log.LogWarning("unknown command: " + cmd);
                            break;
                    }
                }
                catch (Exception e)
                {
                    Plugin.Log.LogWarning("command '" + cmd + "' failed: " + e.Message);
                }
            }
        }

        private void Pacify(Player p)
        {
            try
            {
                if (p.room != null) Reset.Pacify(p.room, p);
            }
            catch (Exception e) { Plugin.Log.LogWarning("peaceful failed: " + e.Message); }
        }

        // -- events ------------------------------------------------------------------------

        private void FlushEnumEvents()
        {
            if (Observation.PendingEnumEvents.Count == 0) return;
            for (int i = 0; i < Observation.PendingEnumEvents.Count; i++) Emit(Observation.PendingEnumEvents[i]);
            Observation.PendingEnumEvents.Clear();
        }

        private void EmitEvents(Player self, RainWorldGame game, AbstractRoom aroom)
        {
            int room = aroom != null ? aroom.index : -1;
            if (room != _lastRoom && aroom != null)
            {
                Emit("{\"t\":" + _tick + ",\"type\":\"room\",\"index\":" + room + ",\"name\":\"" + Esc(aroom.name)
                     + "\",\"shelter\":" + (aroom.shelter ? "true" : "false") + "}");
                _lastRoom = room;
            }
            if (self.dead != _wasDead)
            {
                if (self.dead) Emit("{\"t\":" + _tick + ",\"type\":\"death\",\"clock\":" + game.clock + "}");
                _wasDead = self.dead;
            }
            EmitKarmaChange(game);
        }

        private void EmitKarmaChange(RainWorldGame game)
        {
            int k = Karma(game);
            if (k >= 0 && k != _lastKarma)
            {
                if (_lastKarma >= 0)
                    Emit("{\"t\":" + _tick + ",\"type\":\"karma\",\"from\":" + _lastKarma + ",\"to\":" + k + "}");
                _lastKarma = k;
            }
        }

        private static int Karma(RainWorldGame game)
        {
            try
            {
                StoryGameSession s = game.session as StoryGameSession;
                return s != null ? s.saveState.deathPersistentSaveData.karma : -1;
            }
            catch (Exception) { return -1; }
        }

        /// <summary>The region's room table: names in index order, shelters, gates, connections.
        /// Sent on connect and whenever the world changes, so the agent never depends on the
        /// on-disk dumps to know room names.</summary>
        private void SendWorld(World world)
        {
            _worldSent = world;
            if (world == null || world.abstractRooms == null) return;
            StringBuilder sb = new StringBuilder(1 << 15);
            sb.Append("{\"type\":\"world\",\"region\":\"").Append(Esc(world.name)).Append('"');
            sb.Append(",\"first_room_index\":").Append(world.firstRoomIndex);
            sb.Append(",\"rooms\":[");
            for (int i = 0; i < world.abstractRooms.Length; i++)
            {
                if (i > 0) sb.Append(',');
                AbstractRoom r = world.abstractRooms[i];
                sb.Append('"').Append(r != null ? Esc(r.name) : "").Append('"');
            }
            sb.Append("],\"shelters\":[");
            bool first = true;
            foreach (AbstractRoom r in world.abstractRooms)
                if (r != null && r.shelter) { if (!first) sb.Append(','); first = false; sb.Append('"').Append(Esc(r.name)).Append('"'); }
            sb.Append("],\"gates\":[");
            first = true;
            foreach (AbstractRoom r in world.abstractRooms)
                if (r != null && r.gate) { if (!first) sb.Append(','); first = false; sb.Append('"').Append(Esc(r.name)).Append('"'); }
            sb.Append("],\"connections\":{");
            first = true;
            foreach (AbstractRoom r in world.abstractRooms)
            {
                if (r == null) continue;
                if (!first) sb.Append(',');
                first = false;
                sb.Append('"').Append(Esc(r.name)).Append("\":[");
                bool cf = true;
                if (r.connections != null)
                    foreach (int c in r.connections)
                    {
                        if (c < 0) continue;
                        AbstractRoom o = world.GetAbstractRoom(c);
                        if (o == null) continue;
                        if (!cf) sb.Append(',');
                        cf = false;
                        sb.Append('"').Append(Esc(o.name)).Append('"');
                    }
                sb.Append(']');
            }
            sb.Append("}}");
            Emit(sb.ToString());
        }

        private void Emit(string json) { _plugin.Emit(json); }

        private void MaybeDump(Player p, Room room, RainWorldGame game)
        {
            _plugin.MaybeDumpWorld(game.world);
            if (room != null && p.room == room) _plugin.MaybeDumpRoom(room);
        }

        private static string Esc(string s)
        {
            return s == null ? "" : s.Replace("\\", "\\\\").Replace("\"", "\\\"");
        }
    }
}
