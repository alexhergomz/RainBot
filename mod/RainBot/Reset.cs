using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;
using RWCustom;
using UnityEngine;

namespace RainBot
{
    /// <summary>
    /// Randomized initial state (docs/PROTOCOL.md "Commands", `reset`).
    ///
    /// Moving the slugcat to another room goes through the game's own shortcut machinery
    /// (`ShortcutHandler.CreatureEnterFromAbstractRoom`), the same path a creature arriving from
    /// an abstract room takes. That path already realizes the destination, waits until it is
    /// ready for a player, moves the abstract creature, puts the slugcat out of an entrance and
    /// moves the camera; a hand-written teleport would have to get each of these right. The cost
    /// is a few ticks of pipe travel, which are not exchanged: a reset is not lockstepped, and the
    /// game runs freely until it completes.
    ///
    /// A multi-tick state machine, advanced once per simulation tick by the lockstep:
    ///   Loading  -> target room realized and ready for AI
    ///   Transit  -> slugcat out of the pipe in the target room (skipped for the current room)
    ///   (apply)  -> position, food, karma, timer, creatures, objects, seed
    /// Any exception or timeout fails the reset and leaves the game as it is.
    /// </summary>
    internal sealed class Reset
    {
        private enum Phase { Loading, Transit, Settle }

        // -- request -----------------------------------------------------------------------

        private string _room, _pos = "random";
        private int? _food, _karma, _timerLeft, _seed;
        // Malnourishment is part of the start (default: not malnourished). Otherwise one
        // malnourished sleep would make every later start need a full food bar.
        private bool _maln;
        private bool _clear;
        private readonly List<KeyValuePair<string, int>> _spawn = new List<KeyValuePair<string, int>>();
        private int _fruit, _spears, _rocks, _fruitNear;
        private bool _holdFruit;
        private int? _eatCounter, _bites;
        private int _eatSpeed = 1, _sleepSpeed = 1;
        // `climb=0`: the start must reach a shelter without climbing a pole (the curriculum's early
        // rungs). A room where no standable tile does fails the reset with that reason.
        private bool _noClimb;
        // `shelter_within=N`: the start is at most N tiles (movement-graph path, climbing included) from
        // the pipe into the shelter. The curriculum's climbing rungs start on the climb and move back
        // (reverse curriculum). No such tile fails the reset with that reason.
        private int? _shelterWithin;
        private string _applyError;

        // -- state -------------------------------------------------------------------------

        private Phase _phase;
        private AbstractRoom _target;
        private int _ticks;
        private System.Random _rng;

        public string Error { get; private set; }

        /// <summary>Creatures a reset spawned; `peaceful` mode spares exactly these.</summary>
        public static readonly HashSet<AbstractCreature> Spawned = new HashSet<AbstractCreature>();

        private const int LoadTimeout = 40 * 30, TransitTimeout = 40 * 20;

        public static Reset Parse(string[] parts, out string error)
        {
            error = null;
            Reset r = new Reset();
            for (int i = 1; i < parts.Length; i++)
            {
                if (parts[i].Length == 0) continue;
                int eq = parts[i].IndexOf('=');
                if (eq <= 0) { error = "bad token '" + parts[i] + "'"; return null; }
                string k = parts[i].Substring(0, eq).ToLowerInvariant(), v = parts[i].Substring(eq + 1);
                try
                {
                    switch (k)
                    {
                        case "room": r._room = v; break;
                        case "pos": r._pos = v.ToLowerInvariant(); break;
                        case "food": r._food = Int(v); break;
                        case "karma": r._karma = Int(v); break;
                        case "timer_left": r._timerLeft = Int(v); break;
                        case "malnourished": r._maln = Int(v) != 0; break;
                        case "seed": r._seed = Int(v); break;
                        case "fruit": r._fruit = Math.Max(0, Int(v)); break;
                        // Eating curriculum: fruit within 6 tiles, a fruit already in hand, and the
                        // game's own eatCounter.
                        case "fruit_near": r._fruitNear = Math.Max(0, Int(v)); break;
                        case "hold":
                            if (v == "fruit") r._holdFruit = true;
                            else if (v != "none") { error = "hold must be fruit|none"; return null; }
                            break;
                        case "eat_counter": r._eatCounter = Mathf.Clamp(Int(v), 1, 40); break;
                        case "bites": r._bites = Mathf.Clamp(Int(v), 1, 3); break;
                        case "eat_speed": r._eatSpeed = Mathf.Clamp(Int(v), 1, 40); break;
                        case "sleep_speed": r._sleepSpeed = Mathf.Clamp(Int(v), 1, 40); break;
                        case "climb": r._noClimb = Int(v) == 0; break;
                        case "shelter_within": r._shelterWithin = Math.Max(1, Int(v)); break;
                        case "creatures":
                            if (v == "clear") r._clear = true;
                            else if (v != "keep") { error = "creatures must be keep|clear"; return null; }
                            break;
                        case "spawn":
                            foreach (KeyValuePair<string, int> kv in Pairs(v)) r._spawn.Add(kv);
                            break;
                        case "items":
                            foreach (KeyValuePair<string, int> kv in Pairs(v))
                            {
                                string n = kv.Key.ToLowerInvariant();
                                if (n == "spear") r._spears += kv.Value;
                                else if (n == "rock") r._rocks += kv.Value;
                                else { error = "unknown item '" + kv.Key + "'"; return null; }
                            }
                            break;
                        default: error = "unknown key '" + k + "'"; return null;
                    }
                }
                catch (Exception e) { error = "bad value for " + k + ": " + e.Message; return null; }
            }
            return r;
        }

        private static int Int(string v) { return int.Parse(v, NumberStyles.Integer, CultureInfo.InvariantCulture); }

        private static IEnumerable<KeyValuePair<string, int>> Pairs(string v)
        {
            foreach (string p in v.Split(','))
            {
                if (p.Length == 0) continue;
                int c = p.IndexOf(':');
                if (c < 0) yield return new KeyValuePair<string, int>(p, 1);
                else yield return new KeyValuePair<string, int>(p.Substring(0, c), Int(p.Substring(c + 1)));
            }
        }

        // -- running -----------------------------------------------------------------------

        public enum Status { Pending, Done, Failed }

        /// <summary>Advance one tick. Called before the tick runs.</summary>
        public Status Step(Player p, RainWorldGame game)
        {
            try
            {
                if (_ticks == 0)
                {
                    Recover(p);
                    int gone = RemoveOurs(game.world);
                    if (gone > 0) Plugin.Log.LogMessage("reset: removed " + gone + " objects left from earlier episodes");
                }
                if (_ticks++ == 0 && !Begin(p, game)) return Status.Failed;
                if (p.dead) return Fail("slugcat died during reset");

                if (_phase == Phase.Loading)
                {
                    if (_ticks > LoadTimeout) return Fail("room " + _target.name + " did not load");
                    Room r = _target.realizedRoom;
                    if (r == null) { game.world.ActivateRoom(_target); return Status.Pending; }
                    if (!r.readyForAI || !r.ReadyForPlayer) return Status.Pending;
                    if (p.inShortcut) return Status.Pending;          // let a pipe trip finish first
                    if (p.room == r) { _phase = Phase.Transit; return Status.Pending; }

                    int node = PickExitNode(_target);
                    if (node < 0) return Fail("room " + _target.name + " has no exit node to enter by");
                    FreeHands(p);
                    if (p.room == null)
                    {
                        // In no room at all (it fell out of the bottom of one; the frozen ending kept
                        // it): a pipe entry does not move an object that is in no room. Put it in the
                        // target room at the entrance directly; Apply places it as for any arrival.
                        WorldCoordinate c = r.LocalCoordinateOfNode(node);
                        p.abstractCreature.Move(c);
                        p.PlaceInRoom(r);
                        RoomCamera cam = game.cameras[0];
                        if (cam.room != r) cam.MoveCamera(r, Math.Max(0, r.CameraViewingPoint(r.MiddleOfTile(c.Tile))));
                        _phase = Phase.Transit;
                        _ticks = 1;
                        return Status.Pending;
                    }
                    p.room.RemoveObject(p);
                    p.RemoveFromRoom();
                    game.shortcuts.CreatureEnterFromAbstractRoom(p, _target, node);
                    _phase = Phase.Transit;
                    _ticks = 1;
                    return Status.Pending;
                }

                if (_phase == Phase.Settle)
                {
                    // Neutral input for SettleTicks after placement, so the body reaches the same
                    // rest state whichever way it arrived (through a pipe or already in the room).
                    // Otherwise can_jump, animation frame and chunk velocities carry over, and the
                    // same seed gives different trajectories from tick 0.
                    if (--_settleLeft > 0) return Status.Pending;
                    if (!_resettled && _placedTile.HasValue)
                    {
                        // Second round from an exact placement: the first settle makes the hidden
                        // counters agree but leaves positions a fraction of a unit apart (~0.2 u),
                        // which a chaotic soft body amplifies. Placing again on the same tile and
                        // settling again starts both arrival paths from one state.
                        _resettled = true;
                        Place(p, game, p.room, _placedTile.Value);
                        _settleLeft = SettleTicks;
                        return Status.Pending;
                    }
                    Finish(p, game);
                    return Status.Done;
                }

                // Transit
                if (_ticks > TransitTimeout) return Fail("slugcat never arrived in " + _target.name);
                if (p.inShortcut || p.room == null || p.room != _target.realizedRoom) return Status.Pending;
                Apply(p, game, p.room);
                if (_applyError != null) return Fail(_applyError);
                _phase = Phase.Settle;
                _settleLeft = SettleTicks;
                return Status.Pending;
            }
            catch (Exception e)
            {
                return Fail("exception: " + e.Message);
            }
        }

        private const int SettleTicks = 40;
        private int _settleLeft;
        private bool _resettled;
        private IntVector2? _placedTile;

        /// <summary>The parts of the state that settling would disturb are written last: the rain
        /// timer (40 settle ticks would otherwise eat into it) and the Unity RNG seed.</summary>
        private void Finish(Player p, RainWorldGame game)
        {
            // The game's sleep counters start fresh: the settle ticks (no input) count toward the
            // door closing, so a reset onto a sleep spot with enough food would sleep at once.
            // Not the shelter the cycle started in: that only applies to waking up in one.
            p.touchedNoInputCounter = 0;
            p.forceSleepCounter = 0;
            // Awake: a slugcat realized in a shelter room starts asleep (Player ctor: sleepCounter =
            // 100) and wakes only on an input read through RWInput.PlayerInput, which the action
            // override does not feed. It would sleep until the rain killed it.
            p.sleepCounter = 0;
            p.sleepCurlUp = 0f;
            // No scripted control: SU_C04's intro script (RoomSpecificScript.SU_C04StartUp, saves at
            // cycle < 2) puts a NullController on the slugcat while it pushes it out of the start pipe,
            // and releases it only for players still in SU_C04. After a reset moves the slugcat away,
            // the controller would stay, and Player.checkInput would ignore every injected input.
            p.controller = null;
            p.readyForWin = false;
            p.stillInStartShelter = false;
            if (_timerLeft.HasValue) SetTimerLeft(game.world.rainCycle, _timerLeft.Value);
            if (_holdFruit)
            {
                // Spawned after settling, straight into the free hand: during settling a neutral
                // input might drop it.
                Room room = p.room;
                AbstractPhysicalObject o = NewFruit(game, room, room.GetWorldCoordinate(p.mainBodyChunk.pos));
                AddObject(room, o);
                if (o.realizedObject != null)
                {
                    o.realizedObject.firstChunk.HardSetPosition(p.mainBodyChunk.pos);
                    p.SlugcatGrab(o.realizedObject, p.grasps[0] == null ? 0 : 1);
                    // Bites left before the pip: a DangleFruit takes 3, each followed by a
                    // 15-tick countdown of standing still.
                    DangleFruit df = o.realizedObject as DangleFruit;
                    if (df != null && _bites.HasValue) df.bites = _bites.Value;
                }
            }
            // eatCounter idles at 40 and must count down to 0 while grab is held, the body still
            // and y neutral (Player.cs ~10069); starting it lower makes the first bite reachable.
            if (_eatCounter.HasValue) p.eatCounter = _eatCounter.Value;
            Lockstep.EatSpeed = _eatSpeed;          // every reset sets it; omitted = 1, the real game
            Lockstep.SleepSpeed = _sleepSpeed;
            if (_seed.HasValue) UnityEngine.Random.InitState(_seed.Value);
        }

        private Status Fail(string why) { Error = why; return Status.Failed; }

        private bool Begin(Player p, RainWorldGame game)
        {
            _rng = _seed.HasValue ? new System.Random(_seed.Value) : new System.Random();
            if (!game.IsStorySession) { Error = "reset needs a story session"; return false; }
            if (p.dead) { Error = "slugcat is dead"; return false; }
            World world = game.world;
            // A fresh cycle's rain: the game's own GlobalRain.ResetRain (death rain, flood speed,
            // timer, deathRainHasHit) and no flood. With frozen endings the game never starts a new
            // cycle, so without this a reset after the rain hit would keep the flood and the death
            // rain.
            try { game.globalRain.ResetRain(); game.globalRain.flood = 0f; }
            catch (Exception e) { Plugin.Log.LogWarning("reset: rain not reset: " + e.Message); }
            // The flood's water stays in a normally dry room: RoomRain only pulls a room's water level
            // back while raining or when the room has water of its own, so the next start could drown.
            // Every loaded room goes back to its own level, as Water's constructor sets it without the
            // flood.
            try
            {
                foreach (AbstractRoom ar in world.abstractRooms)
                {
                    Room rr = ar != null ? ar.realizedRoom : null;
                    if (rr != null && rr.waterObject != null) rr.waterObject.fWaterLevel = rr.waterObject.originalWaterLevel;
                }
            }
            catch (Exception e) { Plugin.Log.LogWarning("reset: water not drained: " + e.Message); }
            // The new rain timer from the start, too (Finish sets it exactly): a reset that begins
            // after the rain would otherwise run its loading and settling in the rain.
            if (_timerLeft.HasValue) SetTimerLeft(world.rainCycle, _timerLeft.Value);
            game.GetStorySession.saveState.malnourished = _maln;

            if (string.IsNullOrEmpty(_room))
            {
                AbstractRoom cur;
                Room rr; Vector2 pos; bool pipe;
                ObsWriter.Locate(p, out rr, out cur, out pos, out pipe);
                _target = cur;
            }
            else if (_room == "shelter" || _room == "any")
            {
                List<AbstractRoom> pool = new List<AbstractRoom>();
                for (int i = 0; i < world.abstractRooms.Length; i++)
                {
                    AbstractRoom a = world.abstractRooms[i];
                    if (a == null || a.gate || a.offScreenDen) continue;
                    if (_room == "shelter" && !a.shelter) continue;
                    if (PickExitNode(a, probeOnly: true) < 0) continue;
                    pool.Add(a);
                }
                if (pool.Count == 0) { Error = "no " + _room + " room in " + world.name; return false; }
                _target = pool[_rng.Next(pool.Count)];
            }
            else
            {
                _target = world.GetAbstractRoom(_room);
            }
            if (_target == null) { Error = "no room '" + _room + "' in " + world.name; return false; }
            if (_target.realizedRoom == null) world.ActivateRoom(_target);
            _phase = Phase.Loading;
            return true;
        }

        private int PickExitNode(AbstractRoom a, bool probeOnly = false)
        {
            if (a.nodes == null) return -1;
            List<int> exits = new List<int>();
            for (int i = 0; i < a.nodes.Length; i++)
                if (a.nodes[i].type == AbstractRoomNode.Type.Exit) exits.Add(i);
            if (exits.Count == 0) return -1;
            return probeOnly ? exits[0] : exits[_rng.Next(exits.Count)];
        }

        /// <summary>After an ending the mod took over (Lockstep.EndNow), the slugcat never died: undo
        /// what led there (held by a predator, stunned, out of breath), so the reset starts from an
        /// ordinary living slugcat.</summary>
        private static void Recover(Player p)
        {
            if (p.dead) return;
            // Falling out of a room flags the slugcat for deletion before Die/Destroy are reached
            // (the frozen ending blocks those). A room drops any object so flagged on its next
            // update, so a re-placed slugcat would vanish again at once.
            p.slatedForDeletetion = false;
            if (p.abstractCreature != null) p.abstractCreature.slatedForDeletion = false;
            FreeHands(p);
            p.stun = 0;
            p.airInLungs = 1f;
            p.lungsExhausted = false;
            p.drown = 0f;
        }

        /// <summary>Drop what the slugcat holds and make anything holding it let go: a reset is a
        /// fresh start, and a lizard's jaws would otherwise come along through the pipe.</summary>
        private static void FreeHands(Player p)
        {
            p.LoseAllGrasps();
            ReleaseGrabbers(p);
        }

        /// <summary>Make every creature holding `o` let go of it.</summary>
        private static void ReleaseGrabbers(PhysicalObject o)
        {
            if (o.grabbedBy != null)
                foreach (Creature.Grasp g in o.grabbedBy.ToArray())
                    if (g != null && g.grabber != null) g.grabber.ReleaseGrasp(g.graspUsed);
        }

        private static void SetTimerLeft(RainCycle rc, int ticksLeft)
        {
            rc.timer = Math.Max(0, rc.cycleLength - Math.Max(0, ticksLeft));
        }

        private static AbstractPhysicalObject NewFruit(RainWorldGame game, Room room, WorldCoordinate pos)
        {
            return new DangleFruit.AbstractDangleFruit(game.world, null, pos, game.GetNewID(), room.abstractRoom.index, -1, false, null);
        }

        // -- applying the state ------------------------------------------------------------

        private void Apply(Player p, RainWorldGame game, Room room)
        {
            StringBuilder spawned = new StringBuilder();
            // A fresh stream for everything placed in the room. The arrival path (choosing an
            // entrance pipe, or none when already in the room) consumes draws from `_rng`, so a
            // shared stream would put the same seed on different tiles depending on where the
            // slugcat started.
            if (_seed.HasValue) _rng = new System.Random(_seed.Value * 7919 + 17);
            List<IntVector2> standable = StandableTiles(room);

            // Position.
            if (_pos == "random")
            {
                if (standable.Count > 0)
                {
                    // An unsafe tile is redrawn from the same seeded stream, so the first draw, when
                    // safe, is kept.
                    List<IntVector2> pool = standable;
                    if (_shelterWithin.HasValue)
                    {
                        int[] dist = ShelterDistance(room);
                        int H = room.TileHeight;
                        // Floor, and the poles: the climb itself has no floor (the nearest floor can be
                        // the climb's foot), so a start can be on a pole, holding it.
                        List<IntVector2> cand = new List<IntVector2>(standable);
                        cand.AddRange(PoleTiles(room));
                        pool = cand.FindAll(s => dist[s.x * H + s.y] > 0 && dist[s.x * H + s.y] <= _shelterWithin.Value);
                        if (pool.Count == 0)
                        {
                            int reach = 0, sMin = int.MaxValue, goals = 0;
                            for (int i = 0; i < dist.Length; i++) { if (dist[i] >= 0) reach++; if (dist[i] == 0) goals++; }
                            foreach (IntVector2 s in standable) { int d = dist[s.x * H + s.y]; if (d > 0 && d < sMin) sMin = d; }
                            _applyError = "no start within " + _shelterWithin.Value + " tiles of a shelter in " + room.abstractRoom.name
                                          + " (pipe tiles " + goals + ", tiles reaching it " + reach + ", standable " + standable.Count
                                          + ", nearest standable " + (sMin == int.MaxValue ? "none" : sMin.ToString()) + ")";
                            return;
                        }
                    }
                    if (_shelterWithin.HasValue)
                        Plugin.Log.LogMessage("reset: " + pool.Count + " starts within " + _shelterWithin.Value + " tiles of a shelter in " + room.abstractRoom.name);
                    IntVector2 t = pool[_rng.Next(pool.Count)];
                    int tries = 0;
                    for (; tries < 200 && !(SafeStart(room, t, p) && (!_noClimb || ShelterWithoutClimb(room, t))); tries++)
                        t = pool[_rng.Next(pool.Count)];
                    if (_noClimb && tries >= 200 && !ShelterWithoutClimb(room, t))
                    {
                        _applyError = "no climb-free start in " + room.abstractRoom.name;
                        return;
                    }
                    _placedTile = t;
                    Place(p, game, room, _placedTile.Value);
                }
            }
            else if (_pos != "den")
            {
                string[] xy = _pos.Split(',');
                if (xy.Length == 2)
                {
                    _placedTile = new IntVector2(Int(xy[0]), Int(xy[1]));
                    Place(p, game, room, _placedTile.Value);
                }
            }
            IntVector2 me = room.GetTilePosition(p.mainBodyChunk.pos);
            // Everything after the start tile draws from its own seeded stream. The number of draws
            // the start takes depends on the start checks, which change between mod versions. A
            // shared stream would move every fruit and item with them, and a replay with the
            // recorded start tile (pos=x,y, no draws) would place food elsewhere.
            if (_seed.HasValue) _rng = new System.Random(_seed.Value * 7919 + 101);

            // Needs.
            if (_food.HasValue)
            {
                int max = p.slugcatStats != null ? p.slugcatStats.maxFood : 7;
                int f = Mathf.Clamp(_food.Value, 0, max);
                p.playerState.foodInStomach = f;
                p.playerState.quarterFoodPoints = 0;
                try
                {
                    if (game.cameras[0].hud != null && game.cameras[0].hud.foodMeter != null)
                        game.cameras[0].hud.foodMeter.NewShowCount(f);
                }
                catch (Exception) { }
            }
            if (_karma.HasValue)
            {
                DeathPersistentSaveData d = game.GetStorySession.saveState.deathPersistentSaveData;
                d.karma = Mathf.Clamp(_karma.Value, 0, d.karmaCap);
            }
            if (_timerLeft.HasValue) SetTimerLeft(game.world.rainCycle, _timerLeft.Value);

            // Creatures.
            if (_clear) ClearCreatures(room, p);
            foreach (KeyValuePair<string, int> kv in _spawn)
            {
                CreatureTemplate tpl = StaticWorld.GetCreatureTemplate(kv.Key);
                if (tpl == null)
                {
                    ExtEnumBase e;
                    if (ExtEnumBase.TryParse(typeof(CreatureTemplate.Type), kv.Key, true, out e))
                        tpl = StaticWorld.GetCreatureTemplate(e as CreatureTemplate.Type);
                }
                if (tpl == null) { Plugin.Log.LogWarning("reset: unknown creature '" + kv.Key + "'"); continue; }
                for (int i = 0; i < kv.Value; i++)
                {
                    IntVector2 t = FarTile(standable, me);
                    AbstractCreature ac = new AbstractCreature(game.world, tpl, null, room.GetWorldCoordinate(t), game.GetNewID());
                    room.abstractRoom.AddEntity(ac);
                    ac.RealizeInRoom();
                    Spawned.Add(ac);
                    Note(spawned, tpl.type.value, t);
                }
            }

            // Objects.
            for (int i = 0; i < _fruit; i++)
            {
                IntVector2 t = AnyTile(standable, me);
                AddObject(room, NewFruit(game, room, room.GetWorldCoordinate(t)));
                Note(spawned, "DangleFruit", t);
            }
            for (int i = 0; i < _fruitNear; i++)
            {
                IntVector2 t = NearTile(standable, me, 6);
                AddObject(room, NewFruit(game, room, room.GetWorldCoordinate(t)));
                Note(spawned, "DangleFruit", t);
            }
            for (int i = 0; i < _spears; i++)
            {
                IntVector2 t = AnyTile(standable, me);
                AddObject(room, new AbstractSpear(game.world, null, room.GetWorldCoordinate(t), game.GetNewID(), false));
                Note(spawned, "Spear", t);
            }
            for (int i = 0; i < _rocks; i++)
            {
                IntVector2 t = AnyTile(standable, me);
                AddObject(room, new AbstractPhysicalObject(game.world, AbstractPhysicalObject.AbstractObjectType.Rock,
                    null, room.GetWorldCoordinate(t), game.GetNewID()));
                Note(spawned, "Rock", t);
            }

            SpawnedJson = spawned.ToString();
            RoomName = room.abstractRoom.name;
        }

        public string SpawnedJson { get; private set; } = "";
        public bool HasSeed { get { return _seed.HasValue; } }
        /// <summary>The start tile the reset placed the slugcat on (-1,-1 if none): recorded so a replay
        /// can ask for exactly it.</summary>
        public string StartTileJson { get { return _placedTile.HasValue ? "[" + _placedTile.Value.x + "," + _placedTile.Value.y + "]" : "[-1,-1]"; } }
        public string PhaseName { get { return _phase + " (" + _ticks + " ticks, settle " + _settleLeft + ")"; } }
        public string RoomName { get; private set; } = "";

        private static void Note(StringBuilder sb, string what, IntVector2 t)
        {
            if (sb.Length > 0) sb.Append(',');
            sb.Append("{\"type\":\"").Append(what).Append("\",\"x\":").Append(t.x).Append(",\"y\":").Append(t.y).Append('}');
        }

        private static void AddObject(Room room, AbstractPhysicalObject o)
        {
            room.abstractRoom.AddEntity(o);
            o.RealizeInRoom();
            _ours.Add(o);
        }

        // Objects this mod spawned (fruit, spears, rocks), removed at the next reset wherever they
        // ended up. Frozen endings never reload the world, so without this every uneaten spawn
        // would stay for good (mostly in shelters, carried there by slugcats the rain then killed),
        // plug the shelters and leak extra food into later episodes. Objects the level itself
        // placed are left alone.
        private static readonly List<AbstractPhysicalObject> _ours = new List<AbstractPhysicalObject>();

        private static int RemoveOurs(World world)
        {
            // Plus every DangleFruit with no placement index anywhere in the world: the game saves the
            // objects lying in shelters (RegionState.savedObjects) and puts them back on every launch,
            // so spawns from before this process are not in _ours. Level fruit hangs on vines and
            // always carries its placement index.
            if (world != null && world.abstractRooms != null)
                foreach (AbstractRoom ar in world.abstractRooms)
                {
                    if (ar == null) continue;
                    foreach (AbstractWorldEntity ent in ar.entities)
                    {
                        AbstractConsumable c = ent as AbstractConsumable;
                        if (c != null && c.type == AbstractPhysicalObject.AbstractObjectType.DangleFruit
                            && c.placedObjectIndex < 0 && !_ours.Contains(c))
                            _ours.Add(c);
                    }
                }
            int n = 0;
            foreach (AbstractPhysicalObject o in _ours)
            {
                try
                {
                    if (o == null || o.slatedForDeletion) continue;
                    PhysicalObject r = o.realizedObject;
                    if (r != null)
                    {
                        ReleaseGrabbers(r);
                        if (r.room != null) r.room.RemoveObject(r);
                        r.Destroy();
                    }
                    if (o.Room != null) o.Room.RemoveEntity(o);
                    o.Destroy();
                    n++;
                }
                catch (Exception e) { Plugin.Log.LogWarning("reset: could not remove a spawned object: " + e.Message); }
            }
            _ours.Clear();
            return n;
        }

        /// <summary>A start the game itself would not hurt at once: no bottomless pit beside it, the room's
        /// own accessibility for the Slugcat template (every IAccessibilityModifier: worm grass, water legality),
        /// and, while the room's creatures are kept, no creature that eats or attacks slugcats
        /// within 10 tiles (the distance this reset keeps spawned predators away).</summary>
        private bool SafeStart(Room room, IntVector2 t, Player p)
        {
            CreatureTemplate slug = StaticWorld.GetCreatureTemplate(CreatureTemplate.Type.Slugcat);
            if (slug != null && room.aimap != null && !room.aimap.TileAccessibleToCreature(t, slug)) return false;
            // Next to a bottomless pit: the AI map's own fall-risk marker (AImapper: a fall from this
            // column, or the one beside it, leaves the room; fallRiskTile.y < 0), the test the game's
            // creatures use.
            if (room.aimap != null)
                for (int dx = -1; dx <= 1; dx++)
                {
                    IntVector2 n = new IntVector2(t.x + dx, t.y);
                    if (n.x < 0 || n.x >= room.TileWidth) continue;
                    AItile a = room.aimap.getAItile(n);
                    if (a != null && a.fallRiskTile.y < 0 && !room.GetTile(n).Solid) return false;
                }
            if (_clear || slug == null) return true;
            foreach (AbstractCreature c in room.abstractRoom.creatures)
            {
                if (c == null || c.realizedCreature == p || c.state == null || c.state.dead || c.creatureTemplate == null) continue;
                CreatureTemplate.Relationship.Type rel = c.creatureTemplate.CreatureRelationship(slug).type;
                if (rel != CreatureTemplate.Relationship.Type.Eats && rel != CreatureTemplate.Relationship.Type.Attacks) continue;
                IntVector2 ct = c.pos.Tile;
                if (Math.Abs(ct.x - t.x) + Math.Abs(ct.y - t.y) < 10) return false;
            }
            return true;
        }

        /// <summary>A start from which a shelter needs no pole climb, now or after any wrong turn: on the
        /// room's own AI map (the moves the Slugcat template allows, never onto a pole tile,
        /// AItile.Accessibility.Climb), every tile reachable from t can still reach a shelter pipe
        /// (the in-room start tiles of the exits that lead to a shelter or to a room next to one).
        /// "A route exists" is not enough: from a ledge the slugcat can drop into a pit from which
        /// only a pole leads out.</summary>
        private static bool ShelterWithoutClimb(Room room, IntVector2 t)
        {
            if (room.abstractRoom.shelter) return true;
            CreatureTemplate slug = StaticWorld.GetCreatureTemplate(CreatureTemplate.Type.Slugcat);
            AImap map = room.aimap;
            if (map == null || slug == null) return false;
            int W = room.TileWidth, H = room.TileHeight, N = W * H;
            AbstractRoom a = room.abstractRoom;
            bool[] goal = new bool[N];
            bool anyGoal = false;
            for (int i = 0; i < room.shortcuts.Length; i++)
            {
                int n = room.shortcuts[i].destNode;
                if (n < 0 || n >= a.connections.Length || a.connections[n] < 0) continue;
                AbstractRoom to = room.world.GetAbstractRoom(a.connections[n]);
                if (to == null || !(to.shelter || NextToShelter(room.world, to))) continue;
                IntVector2 st = room.shortcuts[i].StartTile;
                for (int dx = -1; dx <= 1; dx++)
                    for (int dy = -2; dy <= 1; dy++)
                    {
                        int x = st.x + dx, y = st.y + dy;
                        if (x >= 0 && y >= 0 && x < W && y < H) { goal[x * H + y] = true; anyGoal = true; }
                    }
            }
            if (!anyGoal) return false;
            // Climb-free moves, both ways.
            List<int>[] fwd = new List<int>[N], back = new List<int>[N];
            for (int u = 0; u < N; u++)
            {
                AItile tile = map.getAItile(u / H, u % H);
                if (tile == null || tile.outgoingPaths == null || tile.acc == AItile.Accessibility.Climb) continue;
                foreach (MovementConnection c in tile.outgoingPaths)
                {
                    if (c.destinationCoord.room != a.index || !c.destinationCoord.TileDefined) continue;
                    IntVector2 d = c.DestTile;
                    if (d.x < 0 || d.y < 0 || d.x >= W || d.y >= H) continue;
                    if (!map.IsConnectionAllowedForCreature(c, slug)) continue;
                    AItile dt = map.getAItile(d);
                    if (dt == null || dt.acc == AItile.Accessibility.Climb) continue;
                    int v = d.x * H + d.y;
                    (fwd[u] ?? (fwd[u] = new List<int>())).Add(v);
                    (back[v] ?? (back[v] = new List<int>())).Add(u);
                }
            }
            // Tiles that reach a shelter pipe without climbing.
            bool[] ok = new bool[N];
            Queue<int> q = new Queue<int>();
            for (int u = 0; u < N; u++) if (goal[u]) { ok[u] = true; q.Enqueue(u); }
            while (q.Count > 0)
            {
                int v = q.Dequeue();
                if (back[v] == null) continue;
                foreach (int u in back[v]) if (!ok[u]) { ok[u] = true; q.Enqueue(u); }
            }
            int s0 = t.x * H + t.y;
            if (!ok[s0]) return false;
            // Every tile reachable from the start (without climbing) must still be one of them.
            bool[] seen = new bool[N];
            q.Enqueue(s0); seen[s0] = true;
            while (q.Count > 0)
            {
                int u = q.Dequeue();
                if (!ok[u]) return false;
                if (fwd[u] == null) continue;
                foreach (int v in fwd[u]) if (!seen[v]) { seen[v] = true; q.Enqueue(v); }
            }
            return true;
        }

        /// <summary>Path length (tiles, the game's movement connections, climbing included) from every
        /// tile of the room to the nearest pipe into a shelter; -1 where none is reachable. Dijkstra on
        /// the reversed graph from the tiles around those pipes' entrances.</summary>
        private static int[] ShelterDistance(Room room)
        {
            int W = room.TileWidth, H = room.TileHeight, N = W * H;
            int[] dist = new int[N];
            for (int i = 0; i < N; i++) dist[i] = -1;
            CreatureTemplate slug = StaticWorld.GetCreatureTemplate(CreatureTemplate.Type.Slugcat);
            AImap map = room.aimap;
            if (map == null || slug == null) return dist;
            AbstractRoom a = room.abstractRoom;
            List<int>[] back = new List<int>[N];
            List<int>[] backLen = new List<int>[N];
            for (int u = 0; u < N; u++)
            {
                AItile tile = map.getAItile(u / H, u % H);
                if (tile == null || tile.outgoingPaths == null) continue;
                foreach (MovementConnection c in tile.outgoingPaths)
                {
                    if (c.destinationCoord.room != a.index || !c.destinationCoord.TileDefined) continue;
                    IntVector2 d = c.DestTile;
                    if (d.x < 0 || d.y < 0 || d.x >= W || d.y >= H) continue;
                    if (!map.IsConnectionAllowedForCreature(c, slug)) continue;
                    int v = d.x * H + d.y;
                    (back[v] ?? (back[v] = new List<int>())).Add(u);
                    (backLen[v] ?? (backLen[v] = new List<int>())).Add(Math.Max(1, c.distance));
                }
            }
            // The same graph as shelter_path: plus the observed human moves and the jump links.
            Dictionary<int, List<int>> extra = PathInfo.Current != null ? PathInfo.Current.ExtraEdgesOf(room) : null;
            if (extra != null)
                foreach (KeyValuePair<int, List<int>> kv in extra)
                    for (int j = 0; j + 1 < kv.Value.Count; j += 2)
                    {
                        int v = kv.Value[j];
                        if (v < 0 || v >= N) continue;
                        (back[v] ?? (back[v] = new List<int>())).Add(kv.Key);
                        (backLen[v] ?? (backLen[v] = new List<int>())).Add(Math.Max(1, kv.Value[j + 1]));
                    }
            SortedSet<long> open = new SortedSet<long>();
            for (int i = 0; i < room.shortcuts.Length; i++)
            {
                int n = room.shortcuts[i].destNode;
                if (n < 0 || n >= a.connections.Length || a.connections[n] < 0) continue;
                AbstractRoom to = room.world.GetAbstractRoom(a.connections[n]);
                if (to == null || !to.shelter) continue;
                IntVector2 st = room.shortcuts[i].StartTile;
                for (int dx = -1; dx <= 1; dx++)
                    for (int dy = -2; dy <= 1; dy++)
                    {
                        int x = st.x + dx, y = st.y + dy;
                        if (x < 0 || y < 0 || x >= W || y >= H) continue;
                        int g = x * H + y;
                        if (dist[g] == 0) continue;
                        dist[g] = 0;
                        open.Add(g);
                    }
            }
            while (open.Count > 0)
            {
                long top = open.Min;
                open.Remove(top);
                int v = (int)(top & 0xffffffffL), dv = (int)(top >> 32);
                if (dv > dist[v] && dist[v] >= 0) continue;
                if (back[v] == null) continue;
                for (int j = 0; j < back[v].Count; j++)
                {
                    int u = back[v][j], du = dv + backLen[v][j];
                    if (dist[u] < 0 || du < dist[u])
                    {
                        dist[u] = du;
                        open.Add(((long)du << 32) | (uint)u);
                    }
                }
            }
            return dist;
        }

        /// <summary>A room with an exit straight into a shelter (for starts two rooms away: the pipe
        /// toward it is this room's way to a shelter).</summary>
        private static bool NextToShelter(World world, AbstractRoom r)
        {
            foreach (int c in r.connections)
            {
                if (c < 0) continue;
                AbstractRoom n = world.GetAbstractRoom(c);
                if (n != null && n.shelter) return true;
            }
            return false;
        }

        /// <summary>Vertical pole tiles in open air, out of water, that are not floor (a slugcat placed
        /// there holds the pole).</summary>
        private static List<IntVector2> PoleTiles(Room room)
        {
            List<IntVector2> l = new List<IntVector2>();
            if (room.aimap == null) return l;
            int w = room.TileWidth, h = room.TileHeight;
            for (int x = 1; x < w - 1; x++)
                for (int y = 1; y < h - 2; y++)
                {
                    Room.Tile t = room.GetTile(x, y);
                    AItile a = room.aimap.getAItile(x, y);
                    if (!t.verticalBeam || t.Terrain != Room.Tile.TerrainType.Air || a == null || a.AnyWater) continue;
                    if (a.acc == AItile.Accessibility.Floor) continue;          // already a standable start
                    if (!room.GetTile(x, y + 1).verticalBeam && !room.GetTile(x, y - 1).verticalBeam) continue;
                    l.Add(new IntVector2(x, y));
                }
            return l;
        }

        private static void Place(Player p, RainWorldGame game, Room room, IntVector2 tile)
        {
            Vector2 pos = room.MiddleOfTile(tile);
            p.SuperHardSetPosition(pos);
            for (int i = 0; i < p.bodyChunks.Length; i++) p.bodyChunks[i].vel = Vector2.zero;
            if (room.GetTile(tile).verticalBeam && room.aimap != null && room.aimap.getAItile(tile).acc != AItile.Accessibility.Floor)
            {
                // On a pole: holding it, as Player.UpdateAnimation does when up is pressed on one.
                p.bodyChunks[0].HardSetPosition(pos + new Vector2(0f, 6f));
                p.bodyChunks[1].HardSetPosition(pos - new Vector2(0f, 6f));
                p.animation = Player.AnimationIndex.ClimbOnBeam;
                p.bodyMode = Player.BodyModeIndex.ClimbingOnBeam;
            }
            // One posture whichever way the slugcat arrived: a reset in the room it is already in
            // would keep its old posture (neutral input never changes it), 16 px off a pipe arrival,
            // and the same seed would diverge from the first token. This is the posture the game's
            // own Player constructor gives a new slugcat.
            p.standing = false;
            p.abstractCreature.pos.Tile = tile;
            RoomCamera cam = game.cameras[0];
            int cp = room.CameraViewingPoint(pos);
            if (cam.room == room && cp >= 0 && cp != cam.currentCameraPosition) cam.MoveCamera(cp);
        }

        /// <summary>Tiles a slugcat can stand on: the AI map calls it floor, it is open air with
        /// headroom, not narrow, not under water.</summary>
        private static List<IntVector2> StandableTiles(Room room)
        {
            List<IntVector2> l = new List<IntVector2>();
            if (room.aimap == null) return l;
            int w = room.TileWidth, h = room.TileHeight;
            for (int x = 1; x < w - 1; x++)
                for (int y = 1; y < h - 2; y++)
                {
                    AItile a = room.aimap.getAItile(x, y);
                    if (a == null || a.acc != AItile.Accessibility.Floor || a.narrowSpace || a.AnyWater) continue;
                    if (room.GetTile(x, y).Terrain != Room.Tile.TerrainType.Air) continue;
                    if (room.GetTile(x, y + 1).Terrain != Room.Tile.TerrainType.Air) continue;
                    l.Add(new IntVector2(x, y));
                }
            return l;
        }

        /// <summary>A random standable tile, preferring ones at least 10 tiles from the slugcat so a
        /// spawned predator does not start on top of it.</summary>
        private IntVector2 FarTile(List<IntVector2> s, IntVector2 me)
        {
            if (s.Count == 0) return me;
            List<IntVector2> far = new List<IntVector2>();
            foreach (IntVector2 t in s) if (Math.Abs(t.x - me.x) + Math.Abs(t.y - me.y) >= 10) far.Add(t);
            if (far.Count > 0) return far[_rng.Next(far.Count)];
            IntVector2 best = s[0];
            int bd = -1;
            foreach (IntVector2 t in s)
            {
                int d = Math.Abs(t.x - me.x) + Math.Abs(t.y - me.y);
                if (d > bd) { bd = d; best = t; }
            }
            return best;
        }

        /// <summary>A random standable tile within `radius` (Manhattan) of the slugcat, not its own.</summary>
        private IntVector2 NearTile(List<IntVector2> s, IntVector2 me, int radius)
        {
            List<IntVector2> near = new List<IntVector2>();
            foreach (IntVector2 t in s)
            {
                int d = Math.Abs(t.x - me.x) + Math.Abs(t.y - me.y);
                if (d > 0 && d <= radius) near.Add(t);
            }
            return near.Count > 0 ? near[_rng.Next(near.Count)] : AnyTile(s, me);
        }

        private IntVector2 AnyTile(List<IntVector2> s, IntVector2 me)
        {
            if (s.Count == 0) return me;
            for (int tries = 0; tries < 8; tries++)
            {
                IntVector2 t = s[_rng.Next(s.Count)];
                if (t.x != me.x || t.y != me.y) return t;
            }
            return s[_rng.Next(s.Count)];
        }

        // -- creature removal (also used by peaceful mode) ---------------------------------

        public static void ClearCreatures(Room room, Player keep)
        {
            AbstractRoom a = room.abstractRoom;
            foreach (AbstractCreature ac in a.creatures.ToArray())
                if (!IsPlayer(ac, keep)) Remove(ac);
            foreach (AbstractWorldEntity e in a.entitiesInDens.ToArray())
            {
                AbstractCreature ac = e as AbstractCreature;
                if (ac != null && !IsPlayer(ac, keep))
                {
                    a.entitiesInDens.Remove(ac);
                    ac.Destroy();
                }
            }
        }

        /// <summary>Peaceful mode: remove non-player creatures in the slugcat's room that a reset
        /// did not spawn. Returns how many went.</summary>
        public static int Pacify(Room room, Player keep)
        {
            int n = 0;
            foreach (AbstractCreature ac in room.abstractRoom.creatures.ToArray())
            {
                if (IsPlayer(ac, keep) || Spawned.Contains(ac)) continue;
                Remove(ac);
                n++;
            }
            return n;
        }

        private static bool IsPlayer(AbstractCreature ac, Player keep)
        {
            if (ac == null) return true;
            if (keep != null && ac == keep.abstractCreature) return true;
            return ac.creatureTemplate != null && ac.creatureTemplate.type == CreatureTemplate.Type.Slugcat;
        }

        private static void Remove(AbstractCreature ac)
        {
            Creature c = ac.realizedCreature;
            if (c != null)
            {
                try { c.LoseAllGrasps(); } catch (Exception) { }
                ReleaseGrabbers(c);
                c.Destroy();
            }
            AbstractRoom r = ac.Room;
            if (r != null) r.RemoveEntity(ac);
            ac.Destroy();
            Spawned.Remove(ac);
        }
    }
}
