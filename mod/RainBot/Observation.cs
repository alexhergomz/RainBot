using System;
using System.Collections.Generic;
using System.Reflection;
using System.Text;

namespace RainBot
{
    /// <summary>One decision as sent by Python: five MultiDiscrete components.</summary>
    internal struct ActionPackage
    {
        public int x, y;                 // {-1, 0, 1}, matching InputPackage directly
        public bool jmp, thrw, pckp;

        public static ActionPackage FromBytes(byte[] b, int off = 0)
        {
            ActionPackage a;
            a.x = Math.Max(-1, Math.Min(1, (int)(sbyte)b[off]));
            a.y = Math.Max(-1, Math.Min(1, (int)(sbyte)b[off + 1]));
            a.jmp = b[off + 2] != 0;
            a.thrw = b[off + 3] != 0;
            a.pckp = b[off + 4] != 0;
            return a;
        }
    }

    /// <summary>
    /// The observation blob, its self-description, and the ExtEnum maps needed to interpret it.
    ///
    /// AnimationIndex, BodyModeIndex, CreatureTemplate.Type and the AI Behavior enums are
    /// ExtEnums: string-backed and registered at load time, so their integer indices depend on
    /// the installed mods and change between game versions. Rows carry the index, and the schema
    /// carries the full index-to-name table (from ExtEnumBase.GetNames), so recorded data stays
    /// readable after a game update.
    /// </summary>
    internal static class Observation
    {
        public const int Protocol = 11;
        /// <summary>Frame size sent after the rows of every chunk (FrameCapture); 0 = no frames.</summary>
        public static int FrameW = 0, FrameH = 0;
        public const int ActionBytes = 5;
        /// <summary>docs/PROTOCOL.md "Timing": one agent step = 8 ticks = 4 decisions x 2 ticks.</summary>
        public const int TicksPerStep = 8;
        public const int TicksPerDecision = 2;
        public const int DecisionsPerStep = TicksPerStep / TicksPerDecision;
        public const int NumCreatures = 6;     // nearest creatures
        public const int NumFoods = 4;         // nearest edible items
        public const int NumItems = 4;         // nearest carryable non-food objects
        public const int InputHistory = 10;    // Player.input is a 10-deep ring buffer

        // Creature flag bits
        public const byte FlagPresent = 1, FlagDead = 2, FlagStunned = 4, FlagGrabbingMe = 8;

        private static readonly ObsLayout L = new ObsLayout();

        // -- layout, declared once ---------------------------------------------------------
        public static readonly int OffTick = L.Add("tick", "i4");
        public static readonly int OffClock = L.Add("clock", "i4");
        /// <summary>Unix ms when this tick was observed. Gives coarse video alignment, and shows
        /// when the recorder stalls during a long session.</summary>
        public static readonly int OffWallMs = L.Add("wall_ms", "i8");
        public static readonly int OffRoom = L.Add("room", "i4");
        public static readonly int OffCamera = L.Add("camera", "i4");
        /// <summary>Increments on every `reset` only; PPO episodes continue through deaths and
        /// hibernation. Rows of one chunk never span a change.</summary>
        public static readonly int OffEpisode = L.Add("episode", "i4");
        /// <summary>Increments on every new RainWorldGame (respawn after death, new cycle after
        /// hibernation). Rows of one chunk never span a change.</summary>
        public static readonly int OffSession = L.Add("session", "i4");
        /// <summary>Position of this row inside its chunk (0..7).</summary>
        public static readonly int OffStepTick = L.Add("step_tick", "u1");
        public static readonly int OffRoomChanged = L.Add("room_changed", "u1");
        /// <summary>Inside a shortcut pipe. Player.checkInput does not run then, which is why the
        /// lockstep counts simulation ticks, not checkInput calls.</summary>
        public static readonly int OffInShortcut = L.Add("in_shortcut", "u1");
        /// <summary>checkInput ran this tick. When it did not (pipe, stun-free gaps), `action` is
        /// the injected input, not one the sim consumed.</summary>
        public static readonly int OffConsumed = L.Add("consumed", "u1");
        /// <summary>The input that drove this row's transition (post-tick row).</summary>
        public static readonly int OffAction = L.Add("action", "i1", ActionBytes);

        public static readonly int OffAlive = L.Add("alive", "u1");
        public static readonly int OffSleeping = L.Add("sleeping", "u1");
        public static readonly int OffStanding = L.Add("standing", "u1");
        public static readonly int OffExhausted = L.Add("exhausted", "u1");
        public static readonly int OffGrabbedBy = L.Add("grabbed_by", "u1");
        public static readonly int OffCanJump = L.Add("can_jump", "u1");
        public static readonly int OffStun = L.Add("stun", "u1");
        public static readonly int OffRoll = L.Add("roll_counter", "u1");
        public static readonly int OffSlide = L.Add("slide_counter", "u1");

        public static readonly int OffAnimation = L.Add("animation", "i2");
        public static readonly int OffAnimFrame = L.Add("animation_frame", "i2");
        public static readonly int OffBodyMode = L.Add("body_mode", "i2");
        public static readonly int OffGrasp0 = L.Add("grasp0_type", "i2");
        public static readonly int OffGrasp1 = L.Add("grasp1_type", "i2");

        public static readonly int OffAerobic = L.Add("aerobic_level", "f4");
        public static readonly int OffAir = L.Add("air_in_lungs", "f4");
        public static readonly int OffPos0 = L.Add("pos0", "f4", 2);
        public static readonly int OffVel0 = L.Add("vel0", "f4", 2);
        public static readonly int OffPos1 = L.Add("pos1", "f4", 2);
        public static readonly int OffVel1 = L.Add("vel1", "f4", 2);

        // The cycle clock. Without it the agent cannot tell "plenty of time to explore" from
        // "rain is coming, go to a shelter".
        public static readonly int OffCycleTimer = L.Add("cycle_timer", "i4");
        public static readonly int OffCycleLength = L.Add("cycle_length", "i4");
        public static readonly int OffRainHit = L.Add("rain_hit", "u1");
        public static readonly int OffInShelter = L.Add("in_shelter", "u1");

        // Food needed before the shelter lets the slugcat hibernate. ShelterDoor only sets
        // sleepWhenStill once FoodInRoom >= this. Below it a normal sleep cannot end the cycle,
        // and the slugcat waits in the shelter while the rain timer runs past its length.
        public static readonly int OffFoodToHibernate = L.Add("food_to_hibernate", "i2");
        public static readonly int OffFood = L.Add("food_in_stomach", "i2");
        public static readonly int OffQuarterFood = L.Add("quarter_food", "i2");
        public static readonly int OffKarma = L.Add("karma", "i2");
        public static readonly int OffKarmaCap = L.Add("karma_cap", "i2");
        /// <summary>Food meter size, `slugcatStats.maxFood`: the reward's food scale.</summary>
        public static readonly int OffMaxFood = L.Add("max_food", "i2");
        /// <summary>Karma flower effect (`reinforcedKarma`): the next death costs no karma.</summary>
        public static readonly int OffKarmaProtected = L.Add("karma_protected", "u1");

        /// <summary>The game's own 10-deep InputPackage ring buffer, oldest last: recent input
        /// context that the game already keeps.</summary>
        public static readonly int OffInputHist = L.Add("input_history", "i1", InputHistory * ActionBytes);

        // K nearest creatures, distance-sorted, masked via FlagPresent.
        public static readonly int OffCrType = L.Add("cr_type", "i2", NumCreatures);
        public static readonly int OffCrAiKind = L.Add("cr_ai_kind", "i2", NumCreatures);
        public static readonly int OffCrBehavior = L.Add("cr_behavior", "i2", NumCreatures);
        public static readonly int OffCrFlags = L.Add("cr_flags", "u1", NumCreatures);
        public static readonly int OffCrPos = L.Add("cr_rel_pos", "f4", NumCreatures * 2);
        public static readonly int OffCrVel = L.Add("cr_rel_vel", "f4", NumCreatures * 2);

        // Nearest edible items, so the agent can see where food is, not only how much it has.
        public static readonly int OffFoodType = L.Add("food_type", "i2", NumFoods);
        public static readonly int OffFoodPoints = L.Add("food_points", "i1", NumFoods);
        public static readonly int OffFoodFlags = L.Add("food_flags", "u1", NumFoods);
        public static readonly int OffFoodPos = L.Add("food_rel_pos", "f4", NumFoods * 2);

        // Nearest carryable non-food objects, mainly rocks and spears: the slugcat's only weapons.
        // Without them the agent sees a weapon only once it is in hand.
        public static readonly int OffItemType = L.Add("item_type", "i2", NumItems);
        public static readonly int OffItemFlags = L.Add("item_flags", "u1", NumItems);
        public static readonly int OffItemPos = L.Add("item_rel_pos", "f4", NumItems * 2);

        // Eating state. Eating requires holding an edible while nearly still
        // (Player.cs: DistLess(pos, lastPos, 3.6f)), so the agent needs to know that it holds food
        // and that a bite is in progress.
        public static readonly int OffHoldingEdible = L.Add("holding_edible", "u1");
        public static readonly int OffEatCounter = L.Add("eat_counter", "u1");
        public static readonly int OffSpeed = L.Add("speed", "f4");

        // Diagnostics: which branch checkInput took, and whether the injected input landed.
        public static readonly int OffHasController = L.Add("has_controller", "u1");
        public static readonly int OffHasAI = L.Add("has_ai", "u1");
        public static readonly int OffInjectFired = L.Add("inject_fired", "u1");

        /// <summary>Damage events this tick: dealt by the slugcat, and taken by it.</summary>
        /// <remarks>
        /// Read from `Creature.Violence`, not inferred from stun flags. Stun flags over-count:
        /// the nearest-creature list is distance-sorted, so a stunned creature that changes slot
        /// looks like a new stun, and creatures are also stunned by falls and by each other.
        ///
        /// Attribution is exact: the damaging `BodyChunk` has an owner, and a thrown weapon
        /// records `thrownBy`. The hit is the slugcat's only if that is the slugcat.
        /// </remarks>
        public static readonly int OffHitDealt = L.Add("hit_dealt", "u1");
        public static readonly int OffHitTaken = L.Add("hit_taken", "u1");
        /// <summary>Creature type the slugcat last hit, or -1.</summary>
        public static readonly int OffHitVictim = L.Add("hit_victim", "i2");

        // Path lengths for the reward's potential-based shaping (PathInfo.cs; not policy inputs):
        // tiles along the game's own Slugcat movement graph (-1 = none or unknown), and what is
        // at the end.
        public static readonly int OffFoodPath = L.Add("food_path", "i2");
        public static readonly int OffFoodPathPoints = L.Add("food_path_points", "i1");
        public static readonly int OffFoodPathBites = L.Add("food_path_bites", "u1");
        public static readonly int OffShelterPath = L.Add("shelter_path", "i2");
        /// <summary>The edible in hand: food points and bites left (0 when none).</summary>
        public static readonly int OffHeldPoints = L.Add("held_points", "i1");
        public static readonly int OffHeldBites = L.Add("held_bites", "u1");
        /// <summary>The eat-speed curriculum's K (mod `eat_speed`; 1 = the real game).</summary>
        public static readonly int OffEatSpeed = L.Add("eat_speed", "u1");
        /// <summary>The save's malnourished flag (a malnourished sleep last night): tonight's
        /// requirement is then max food, and below it the slugcat starves (ShelterDoor, SlugcatStats).</summary>
        public static readonly int OffMalnourished = L.Add("malnourished", "u1");
        /// <summary>The species' normal hibernation threshold (SlugcatFoodMeter.y), whatever the
        /// malnourished requirement: the reward's hunger setpoint.</summary>
        public static readonly int OffFoodHibernateBase = L.Add("food_hibernate_base", "i2");
        // Sleep (v9), by the game's rules (Player.Update, ShelterDoor): the door closes
        // with the slugcat on a sleep spot (more than 6 tiles from the entrance, inside the shelter
        // range) and either still for the no-input count while `readyForWin` (fed and out of the
        // start shelter, or the rain has come), or holding down for 260 ticks (a malnourished sleep:
        // 1 pip up to the requirement, or max food when already malnourished).
        /// <summary>Tiles along the movement graph to the nearest sleep spot of this shelter
        /// (0 on one; -1 not in a shelter).</summary>
        public static readonly int OffSleepPath = L.Add("sleep_path", "i2");
        public static readonly int OffStartShelter = L.Add("start_shelter", "u1");
        public static readonly int OffReadyForWin = L.Add("ready_for_win", "u1");
        /// <summary>Ticks still to stay without input for the door to close, once readyForWin
        /// (-1 outside a shelter).</summary>
        public static readonly int OffStillTicksLeft = L.Add("still_ticks_left", "i2");
        /// <summary>Ticks still to hold down for a forced (malnourished) sleep (-1 when the food
        /// does not allow one).</summary>
        public static readonly int OffForceTicksLeft = L.Add("force_ticks_left", "i2");
        /// <summary>The sleep-speed curriculum's K (mod `sleep_speed`; 1 = the real game), v10.</summary>
        public static readonly int OffSleepSpeed = L.Add("sleep_speed", "u1");

        public static int Size { get { return L.Size; } }

        // -- ExtEnum registry --------------------------------------------------------------

        private static readonly List<string> _aiKinds = new List<string>();
        /// <summary>`enum` events not yet sent; drained on the main thread before each exchange.</summary>
        public static readonly List<string> PendingEnumEvents = new List<string>();
        private static readonly Dictionary<Type, int> _aiKindIndex = new Dictionary<Type, int>();
        private static readonly Dictionary<Type, FieldInfo> _behaviorField = new Dictionary<Type, FieldInfo>();

        /// <summary>
        /// Map an AI instance to a small stable id, and cache the reflection needed to read its
        /// `behavior` field. Every AI type declares its own Behavior ExtEnum (LizardAI.Behavior,
        /// ScavengerAI.Behavior, ...) with no common base. Reflection once per type avoids listing
        /// every creature class by hand, and also covers creature types not listed anywhere.
        /// </summary>
        public static int AiKind(object ai, out FieldInfo behaviorField)
        {
            Type t = ai.GetType();
            int idx;
            if (!_aiKindIndex.TryGetValue(t, out idx))
            {
                idx = _aiKinds.Count;
                _aiKinds.Add(t.FullName);
                _aiKindIndex[t] = idx;

                FieldInfo fi = t.GetField("behavior", BindingFlags.Public | BindingFlags.Instance);
                if (fi != null && !typeof(ExtEnumBase).IsAssignableFrom(fi.FieldType)) fi = null;
                _behaviorField[t] = fi;

                // The ai_kind table grows after the handshake, so each new entry goes to the agent as
                // an event. Queued here (this runs mid-row) and flushed by the lockstep before the
                // chunk that first uses the index.
                StringBuilder ev = new StringBuilder(512);
                ev.Append("{\"type\":\"enum\",\"table\":\"ai_kind\",\"index\":").Append(idx)
                  .Append(",\"name\":\"").Append(Escape(t.FullName)).Append("\",\"behaviors\":");
                if (fi != null) AppendNameArray(ev, fi.FieldType); else ev.Append("[]");
                ev.Append('}');
                PendingEnumEvents.Add(ev.ToString());
            }
            behaviorField = _behaviorField[t];
            return idx;
        }

        /// <summary>index -> name tables for every ExtEnum the observation references.</summary>
        private static string EnumsJson()
        {
            StringBuilder sb = new StringBuilder(8192);
            sb.Append('{');
            bool first = true;
            AppendNames(sb, ref first, "animation", typeof(Player.AnimationIndex));
            AppendNames(sb, ref first, "body_mode", typeof(Player.BodyModeIndex));
            AppendNames(sb, ref first, "creature_type", typeof(CreatureTemplate.Type));
            // food_type, item_type, grasp0_type and grasp1_type all index this table.
            AppendNames(sb, ref first, "object_type", typeof(AbstractPhysicalObject.AbstractObjectType));

            // AI kinds seen so far, and each one's behavior table. This grows as new creatures
            // appear, so the recorder rewrites its schema sidecar on close.
            if (!first) sb.Append(',');
            sb.Append("\"ai_kind\":[");
            for (int i = 0; i < _aiKinds.Count; i++)
            {
                if (i > 0) sb.Append(',');
                sb.Append('"').Append(Escape(_aiKinds[i])).Append('"');
            }
            sb.Append(']');
            sb.Append(",\"ai_behavior\":{");
            bool bfirst = true;
            foreach (KeyValuePair<Type, FieldInfo> kv in _behaviorField)
            {
                if (kv.Value == null) continue;
                if (!bfirst) sb.Append(',');
                bfirst = false;
                sb.Append('"').Append(Escape(kv.Key.FullName)).Append("\":");
                AppendNameArray(sb, kv.Value.FieldType);
            }
            sb.Append('}');
            sb.Append('}');
            return sb.ToString();
        }

        private static void AppendNames(StringBuilder sb, ref bool first, string key, Type t)
        {
            if (!first) sb.Append(',');
            first = false;
            sb.Append('"').Append(key).Append("\":");
            AppendNameArray(sb, t);
        }

        private static void AppendNameArray(StringBuilder sb, Type t)
        {
            sb.Append('[');
            try
            {
                string[] names = ExtEnumBase.GetNames(t);
                for (int i = 0; i < names.Length; i++)
                {
                    if (i > 0) sb.Append(',');
                    sb.Append('"').Append(Escape(names[i])).Append('"');
                }
            }
            catch (Exception e)
            {
                Plugin.Log.LogWarning("could not enumerate " + t.Name + ": " + e.Message);
            }
            sb.Append(']');
        }

        // -- schema ------------------------------------------------------------------------

        public static string SchemaJson(string gameVersion)
        {
            StringBuilder sb = new StringBuilder(16384);
            sb.Append("{\"protocol\":").Append(Protocol);
            sb.Append(",\"game_version\":\"").Append(Escape(gameVersion)).Append('"');
            sb.Append(",\"obs_bytes\":").Append(Size);
            sb.Append(",\"k_creatures\":").Append(NumCreatures);
            sb.Append(",\"k_foods\":").Append(NumFoods);
            sb.Append(",\"k_items\":").Append(NumItems);
            // The game refuses to eat while moving faster than this.
            sb.Append(",\"eat_speed_limit\":3.6");
            sb.Append(",\"input_history\":").Append(InputHistory);
            sb.Append(",\"ticks_per_step\":").Append(TicksPerStep);
            sb.Append(",\"ticks_per_decision\":").Append(TicksPerDecision);
            sb.Append(",\"decisions_per_step\":").Append(DecisionsPerStep);
            // v8: the latest rendered frame follows the rows of every chunk (FrameCapture).
            sb.Append(",\"frame\":{\"w\":").Append(FrameW).Append(",\"h\":").Append(FrameH)
              .Append(",\"c\":").Append(FrameCapture.Channels).Append(",\"header_bytes\":").Append(FrameCapture.HeaderBytes).Append('}');
            sb.Append(",\"instance\":").Append(Instance.Index);
            sb.Append(",\"action\":{\"dims\":[3,3,2,2,2],\"bytes\":").Append(ActionBytes).Append('}');
            sb.Append(",\"creature_flags\":{\"present\":1,\"dead\":2,\"stunned\":4,\"grabbing_me\":8}");
            sb.Append(",\"enums\":").Append(EnumsJson());
            sb.Append(",\"fields\":[").Append(L.FieldsJson()).Append("]}");
            return sb.ToString();
        }

        private static string Escape(string s)
        {
            return s == null ? "" : s.Replace("\\", "\\\\").Replace("\"", "\\\"");
        }

        // -- writing -----------------------------------------------------------------------

        public static void PutI64(byte[] b, int off, long v)
        {
            for (int i = 0; i < 8; i++) b[off + i] = (byte)(v >> (8 * i));
        }

        public static void PutI32(byte[] b, int off, int v)
        {
            b[off] = (byte)v; b[off + 1] = (byte)(v >> 8); b[off + 2] = (byte)(v >> 16); b[off + 3] = (byte)(v >> 24);
        }

        public static void PutI16(byte[] b, int off, int v)
        {
            b[off] = (byte)v; b[off + 1] = (byte)(v >> 8);
        }

        public static void PutF32(byte[] b, int off, float v)
        {
            byte[] t = BitConverter.GetBytes(v);
            b[off] = t[0]; b[off + 1] = t[1]; b[off + 2] = t[2]; b[off + 3] = t[3];
        }

        public static void PutU8(byte[] b, int off, bool v) { b[off] = v ? (byte)1 : (byte)0; }

        public static void PutClampedU8(byte[] b, int off, int v)
        {
            b[off] = (byte)(v < 0 ? 0 : (v > 255 ? 255 : v));
        }
    }
}
