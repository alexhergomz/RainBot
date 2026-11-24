using System;
using System.Collections.Generic;
using System.Reflection;
using UnityEngine;

namespace RainBot
{
    /// <summary>
    /// Fills one observation row (docs/PROTOCOL.md) from the live game.
    ///
    /// Works whether or not the slugcat is in a room. Inside a shortcut pipe `Player.room` is
    /// null and the body chunks hold wherever it entered, so the row is built from the pipe
    /// vessel instead: its abstract room, the tile it is passing through, zero velocity.
    /// checkInput does not run in a pipe, so rows are written per simulation tick.
    /// </summary>
    internal sealed class ObsWriter
    {
        public readonly byte[] Row = new byte[Observation.Size];

        // Damage counters, filled by the Violence hook between rows.
        public int HitDealt, HitTaken, HitVictim = -1;

        private readonly List<Creature> _scratch = new List<Creature>();
        private readonly List<PhysicalObject> _objScratch = new List<PhysicalObject>();
        private readonly List<PhysicalObject> _foodPick = new List<PhysicalObject>();
        private readonly HashSet<int> _foodSeenTypes = new HashSet<int>();
        private static readonly DateTime Epoch = new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc);

        /// <summary>The room the slugcat is in or travelling through, and where it is.</summary>
        public static bool Locate(Player p, out Room room, out AbstractRoom aroom, out Vector2 pos, out bool inPipe)
        {
            inPipe = p.inShortcut;
            room = p.room;
            aroom = room != null ? room.abstractRoom : null;
            pos = p.mainBodyChunk != null ? p.mainBodyChunk.pos : Vector2.zero;
            if (inPipe)
            {
                ShortcutHandler.ShortCutVessel v = p.inShortcutVessel as ShortcutHandler.ShortCutVessel;
                AbstractRoom vr = p.inShortcutVessel != null ? p.inShortcutVessel.room : null;
                if (vr != null) aroom = vr;
                room = aroom != null ? aroom.realizedRoom : null;
                if (v != null) pos = new Vector2(v.pos.x * 20f + 10f, v.pos.y * 20f + 10f);
            }
            if (aroom == null && p.abstractCreature != null && p.abstractCreature.world != null)
                aroom = p.abstractCreature.world.GetAbstractRoom(p.abstractCreature.pos.room);
            return aroom != null;
        }

        public readonly PathInfo Paths = new PathInfo();
        /// <summary>A death the mod took over (Lockstep.EndNow): the game never ran Die, the row says dead.</summary>
        public bool ForceDead;
        /// <summary>The world is held frozen after an ending (Lockstep.FrozenTick).</summary>
        public bool WorldFrozen;

        public void Write(Player self, RainWorldGame game, int tick, int episode, int session, int stepTick,
                          bool roomChanged, bool consumed, sbyte[] action, bool injectFired)
        {
            Array.Clear(Row, 0, Row.Length);
            Room room; AbstractRoom aroom; Vector2 me; bool inPipe;
            Locate(self, out room, out aroom, out me, out inPipe);

            Observation.PutI32(Row, Observation.OffTick, tick);
            Observation.PutI32(Row, Observation.OffClock, game.clock);
            Observation.PutI64(Row, Observation.OffWallMs, (long)(DateTime.UtcNow - Epoch).TotalMilliseconds);
            Observation.PutI32(Row, Observation.OffRoom, aroom != null ? aroom.index : -1);
            Observation.PutI32(Row, Observation.OffCamera,
                (game.cameras != null && game.cameras.Length > 0) ? game.cameras[0].currentCameraPosition : -1);
            Observation.PutI32(Row, Observation.OffEpisode, episode);
            Observation.PutI32(Row, Observation.OffSession, session);
            Row[Observation.OffStepTick] = (byte)stepTick;
            Observation.PutU8(Row, Observation.OffRoomChanged, roomChanged);
            Observation.PutU8(Row, Observation.OffInShortcut, inPipe);
            Observation.PutU8(Row, Observation.OffConsumed, consumed);
            for (int i = 0; i < Observation.ActionBytes; i++) Row[Observation.OffAction + i] = (byte)action[i];

            Observation.PutU8(Row, Observation.OffAlive, !self.dead && !ForceDead);
            Observation.PutU8(Row, Observation.OffSleeping, self.Sleeping);
            Observation.PutU8(Row, Observation.OffStanding, self.standing);
            Observation.PutU8(Row, Observation.OffExhausted, self.exhausted);
            Observation.PutU8(Row, Observation.OffGrabbedBy, self.grabbedBy != null && self.grabbedBy.Count > 0);
            Observation.PutClampedU8(Row, Observation.OffCanJump, self.canJump);
            Observation.PutClampedU8(Row, Observation.OffStun, self.stun);
            Observation.PutClampedU8(Row, Observation.OffRoll, self.rollCounter);
            Observation.PutClampedU8(Row, Observation.OffSlide, self.slideCounter);

            Observation.PutI16(Row, Observation.OffAnimation, self.animation != null ? self.animation.Index : -1);
            Observation.PutI16(Row, Observation.OffAnimFrame, self.animationFrame);
            Observation.PutI16(Row, Observation.OffBodyMode, self.bodyMode != null ? self.bodyMode.Index : -1);
            Observation.PutI16(Row, Observation.OffGrasp0, GraspedType(self, 0));
            Observation.PutI16(Row, Observation.OffGrasp1, GraspedType(self, 1));

            Observation.PutF32(Row, Observation.OffAerobic, self.aerobicLevel);
            Observation.PutF32(Row, Observation.OffAir, self.airInLungs);
            if (inPipe)
            {
                // Body chunks are stale inside a pipe; both report the vessel's tile, at rest.
                Observation.PutF32(Row, Observation.OffPos0, me.x);
                Observation.PutF32(Row, Observation.OffPos0 + 4, me.y);
                Observation.PutF32(Row, Observation.OffPos1, me.x);
                Observation.PutF32(Row, Observation.OffPos1 + 4, me.y);
            }
            else
            {
                WriteChunk(self, 0, Observation.OffPos0, Observation.OffVel0);
                WriteChunk(self, 1, Observation.OffPos1, Observation.OffVel1);
            }

            WriteCycle(self, aroom);

            PlayerState ps = self.playerState;
            Observation.PutI16(Row, Observation.OffFoodToHibernate,
                self.slugcatStats != null ? self.slugcatStats.foodToHibernate : 4);
            Observation.PutI16(Row, Observation.OffMaxFood, self.slugcatStats != null ? self.slugcatStats.maxFood : 7);
            Observation.PutI16(Row, Observation.OffFood, ps.foodInStomach);
            Observation.PutI16(Row, Observation.OffQuarterFood, ps.quarterFoodPoints);
            WriteKarma(game);

            WriteInputHistory(self);
            if (room != null)
            {
                WriteCreatures(self, room, me);
                WriteFoods(room, me);
                WriteItems(room, me);
            }
            Row[Observation.OffHitDealt] = (byte)Math.Min(HitDealt, 255);
            Row[Observation.OffHitTaken] = (byte)Math.Min(HitTaken, 255);
            Observation.PutI16(Row, Observation.OffHitVictim, HitVictim);
            HitDealt = HitTaken = 0;
            HitVictim = -1;
            WriteEatState(self);

            Observation.PutU8(Row, Observation.OffHasController, self.controller != null);
            Observation.PutU8(Row, Observation.OffHasAI, self.AI != null);
            Observation.PutU8(Row, Observation.OffInjectFired, injectFired);

            if (stepTick == Observation.TicksPerStep - 1 || roomChanged)
            {
                try { Paths.Update(self, room, aroom, inPipe); }
                catch (Exception e) { Plugin.Log.LogWarning("paths: " + e.Message); }
            }
            Observation.PutI16(Row, Observation.OffFoodPath, Paths.FoodPath);
            Row[Observation.OffFoodPathPoints] = (byte)(sbyte)Math.Max(-128, Math.Min(127, Paths.FoodPoints));
            Observation.PutClampedU8(Row, Observation.OffFoodPathBites, Paths.FoodBites);
            Observation.PutI16(Row, Observation.OffShelterPath, Math.Min(Paths.ShelterPath, 32767));
            Row[Observation.OffHeldPoints] = (byte)(sbyte)Math.Max(-128, Math.Min(127, Paths.HeldPoints));
            Observation.PutClampedU8(Row, Observation.OffHeldBites, Paths.HeldBites);
            Observation.PutClampedU8(Row, Observation.OffEatSpeed, Lockstep.EatSpeed);
        }

        // -- pieces ------------------------------------------------------------------------

        private static short GraspedType(Player self, int i)
        {
            if (self.grasps == null || i >= self.grasps.Length) return -1;
            Creature.Grasp g = self.grasps[i];
            if (g == null || g.grabbed == null || g.grabbed.abstractPhysicalObject == null) return -1;
            return (short)g.grabbed.abstractPhysicalObject.type.Index;
        }

        private void WriteCycle(Player self, AbstractRoom aroom)
        {
            int timer = -1, length = -1;
            bool hit = false;
            try
            {
                World world = self.abstractCreature.world;
                if (world != null && world.rainCycle != null)
                {
                    timer = world.rainCycle.timer;
                    length = world.rainCycle.cycleLength;
                    hit = world.rainCycle.deathRainHasHit;
                }
            }
            catch (Exception) { }
            // A frozen ending stops the clock by design.
            if (!ForceDead && !WorldFrozen) Diagnostics.FrozenCycle(self, timer);
            Observation.PutI32(Row, Observation.OffCycleTimer, timer);
            Observation.PutI32(Row, Observation.OffCycleLength, length);
            Observation.PutU8(Row, Observation.OffRainHit, hit);
            Observation.PutU8(Row, Observation.OffInShelter, aroom != null && aroom.shelter);
            int still = -1, force = -1;
            try
            {
                if (aroom != null && aroom.shelter && self.room != null)
                {
                    // Ticks still to go at the sleep-speed curriculum's K. `still` counts while ready_for_win.
                    int ks = Math.Max(1, Lockstep.SleepSpeed);
                    still = Math.Max(0, ((ModManager.MMF ? 40 : 20) + 1 - self.touchedNoInputCounter + ks - 1) / ks);
                    bool maln = self.room.game.GetStorySession.saveState.malnourished;
                    int food = self.FoodInRoom(self.room, false);
                    if ((!maln && food > 0 && food < self.slugcatStats.foodToHibernate) || (maln && food >= self.MaxFoodInStomach))
                        force = Math.Max(0, (261 - self.forceSleepCounter + ks - 1) / ks);
                }
            }
            catch (Exception) { }
            Observation.PutI16(Row, Observation.OffSleepPath, aroom != null && aroom.shelter ? Paths.SleepPath : -1);
            Observation.PutU8(Row, Observation.OffStartShelter, self.stillInStartShelter);
            Observation.PutU8(Row, Observation.OffReadyForWin, self.readyForWin);
            Observation.PutI16(Row, Observation.OffStillTicksLeft, still);
            Observation.PutI16(Row, Observation.OffForceTicksLeft, force);
            Observation.PutClampedU8(Row, Observation.OffSleepSpeed, Lockstep.SleepSpeed);
        }

        private void WriteKarma(RainWorldGame game)
        {
            int karma = -1, cap = -1;
            bool prot = false;
            try
            {
                StoryGameSession story = game.session as StoryGameSession;
                if (story != null)
                {
                    DeathPersistentSaveData d = story.saveState.deathPersistentSaveData;
                    karma = d.karma;
                    cap = d.karmaCap;
                    prot = d.reinforcedKarma;
                }
            }
            catch (Exception) { }
            Observation.PutI16(Row, Observation.OffKarma, karma);
            Observation.PutI16(Row, Observation.OffKarmaCap, cap);
            Observation.PutU8(Row, Observation.OffKarmaProtected, prot);
            bool maln = false;
            int baseHib = 4;
            try
            {
                StoryGameSession story = game.session as StoryGameSession;
                if (story != null)
                {
                    maln = story.saveState.malnourished;
                    baseHib = SlugcatStats.SlugcatFoodMeter(story.saveStateNumber).y;
                }
            }
            catch (Exception) { }
            Observation.PutU8(Row, Observation.OffMalnourished, maln);
            Observation.PutI16(Row, Observation.OffFoodHibernateBase, baseHib);
        }

        private void WriteChunk(Player self, int idx, int posOff, int velOff)
        {
            if (self.bodyChunks == null || idx >= self.bodyChunks.Length || self.bodyChunks[idx] == null) return;
            Vector2 p = self.bodyChunks[idx].pos, v = self.bodyChunks[idx].vel;
            Observation.PutF32(Row, posOff, p.x);
            Observation.PutF32(Row, posOff + 4, p.y);
            Observation.PutF32(Row, velOff, v.x);
            Observation.PutF32(Row, velOff + 4, v.y);
        }

        /// <summary>The game's own 10-deep input ring buffer.</summary>
        private void WriteInputHistory(Player self)
        {
            if (self.input == null) return;
            int n = Math.Min(Observation.InputHistory, self.input.Length);
            for (int i = 0; i < n; i++)
            {
                Player.InputPackage p = self.input[i];
                int o = Observation.OffInputHist + i * Observation.ActionBytes;
                Row[o] = (byte)(sbyte)p.x;
                Row[o + 1] = (byte)(sbyte)p.y;
                Row[o + 2] = (byte)(p.jmp ? 1 : 0);
                Row[o + 3] = (byte)(p.thrw ? 1 : 0);
                Row[o + 4] = (byte)(p.pckp ? 1 : 0);
            }
        }

        // -- entities ----------------------------------------------------------------------

        /// <summary>The body chunk of `c` closest to `from`. mainBodyChunk is wrong for long
        /// creatures: a PoleMimic's main chunk is the top of the pole.</summary>
        private static BodyChunk NearestChunk(Creature c, Vector2 from, out float sqrDist)
        {
            BodyChunk best = c.mainBodyChunk;
            sqrDist = (best != null) ? (best.pos - from).sqrMagnitude : float.MaxValue;
            if (c.bodyChunks != null)
                for (int i = 0; i < c.bodyChunks.Length; i++)
                {
                    BodyChunk ch = c.bodyChunks[i];
                    if (ch == null) continue;
                    float d = (ch.pos - from).sqrMagnitude;
                    if (best == null || d < sqrDist) { best = ch; sqrDist = d; }
                }
            return best;
        }

        /// <summary>K nearest creatures in the room, distance-sorted, with AI.behavior. The
        /// behaviour is present-tense state a player reads from body language; without it a
        /// lizard's commitment to a lunge looks random.</summary>
        private void WriteCreatures(Player self, Room room, Vector2 me)
        {
            _scratch.Clear();
            List<AbstractCreature> all = room.abstractRoom.creatures;
            for (int i = 0; i < all.Count; i++)
            {
                Creature c = all[i].realizedCreature;
                if (c == null || c == self || c.room != room || c.mainBodyChunk == null) continue;
                _scratch.Add(c);
            }
            _scratch.Sort(delegate (Creature a, Creature b)
            {
                float da, db;
                NearestChunk(a, me, out da);
                NearestChunk(b, me, out db);
                return da.CompareTo(db);
            });

            Vector2 myVel = self.mainBodyChunk != null && !self.inShortcut ? self.mainBodyChunk.vel : Vector2.zero;
            int n = Math.Min(Observation.NumCreatures, _scratch.Count);
            for (int i = 0; i < n; i++)
            {
                Creature c = _scratch[i];
                byte flags = Observation.FlagPresent;
                if (c.dead) flags |= Observation.FlagDead;
                if (c.stun > 0) flags |= Observation.FlagStunned;
                if (IsGrabbingMe(c, self)) flags |= Observation.FlagGrabbingMe;

                Observation.PutI16(Row, Observation.OffCrType + i * 2, c.abstractCreature.creatureTemplate.type.Index);
                Row[Observation.OffCrFlags + i] = flags;

                int aiKind = -1, behavior = -1;
                object ai = (c.abstractCreature.abstractAI != null) ? c.abstractCreature.abstractAI.RealAI : null;
                if (ai != null)
                {
                    FieldInfo bf;
                    aiKind = Observation.AiKind(ai, out bf);
                    if (bf != null)
                    {
                        ExtEnumBase v = bf.GetValue(ai) as ExtEnumBase;
                        if (v != null) behavior = v.Index;
                    }
                }
                Observation.PutI16(Row, Observation.OffCrAiKind + i * 2, aiKind);
                Observation.PutI16(Row, Observation.OffCrBehavior + i * 2, behavior);

                float nearSq;
                BodyChunk near = NearestChunk(c, me, out nearSq);
                Vector2 rp = near.pos - me;
                Vector2 rv = near.vel - myVel;
                Observation.PutF32(Row, Observation.OffCrPos + i * 8, rp.x);
                Observation.PutF32(Row, Observation.OffCrPos + i * 8 + 4, rp.y);
                Observation.PutF32(Row, Observation.OffCrVel + i * 8, rv.x);
                Observation.PutF32(Row, Observation.OffCrVel + i * 8 + 4, rv.y);
            }
        }

        /// <summary>Nearest edible objects; nearest of each distinct type first so one abundant
        /// food cannot crowd a scarce one out of the four slots.</summary>
        private void WriteFoods(Room room, Vector2 me)
        {
            _objScratch.Clear();
            if (room.physicalObjects != null)
                for (int layer = 0; layer < room.physicalObjects.Length; layer++)
                {
                    List<PhysicalObject> objs = room.physicalObjects[layer];
                    for (int i = 0; i < objs.Count; i++)
                    {
                        IPlayerEdible e = objs[i] as IPlayerEdible;
                        if (e == null || !e.Edible) continue;
                        if (objs[i].bodyChunks == null || objs[i].bodyChunks.Length == 0 || objs[i].bodyChunks[0] == null) continue;
                        _objScratch.Add(objs[i]);
                    }
                }
            SortScratchByDistance(me);

            _foodPick.Clear();
            _foodSeenTypes.Clear();
            for (int i = 0; i < _objScratch.Count && _foodPick.Count < Observation.NumFoods; i++)
            {
                AbstractPhysicalObject ap = _objScratch[i].abstractPhysicalObject;
                int ty = ap != null ? ap.type.Index : -1;
                if (_foodSeenTypes.Contains(ty)) continue;
                _foodSeenTypes.Add(ty);
                _foodPick.Add(_objScratch[i]);
            }
            for (int i = 0; i < _objScratch.Count && _foodPick.Count < Observation.NumFoods; i++)
                if (!_foodPick.Contains(_objScratch[i])) _foodPick.Add(_objScratch[i]);

            for (int i = 0; i < _foodPick.Count; i++)
            {
                PhysicalObject o = _foodPick[i];
                IPlayerEdible e = o as IPlayerEdible;
                Observation.PutI16(Row, Observation.OffFoodType + i * 2,
                    o.abstractPhysicalObject != null ? o.abstractPhysicalObject.type.Index : -1);
                Row[Observation.OffFoodPoints + i] = (byte)(sbyte)Math.Max(-128, Math.Min(127, e.FoodPoints));
                byte flags = Observation.FlagPresent;
                if (e.AutomaticPickUp) flags |= 2;
                Row[Observation.OffFoodFlags + i] = flags;
                Vector2 rp = o.bodyChunks[0].pos - me;
                Observation.PutF32(Row, Observation.OffFoodPos + i * 8, rp.x);
                Observation.PutF32(Row, Observation.OffFoodPos + i * 8 + 4, rp.y);
            }
        }

        /// <summary>Nearest carryable non-edible objects: rocks, spears, anything throwable.</summary>
        private void WriteItems(Room room, Vector2 me)
        {
            _objScratch.Clear();
            if (room.physicalObjects != null)
                for (int layer = 0; layer < room.physicalObjects.Length; layer++)
                {
                    List<PhysicalObject> objs = room.physicalObjects[layer];
                    for (int i = 0; i < objs.Count; i++)
                    {
                        PhysicalObject o = objs[i];
                        if (o is Creature || o is IPlayerEdible) continue;
                        if (o.bodyChunks == null || o.bodyChunks.Length == 0 || o.abstractPhysicalObject == null) continue;
                        _objScratch.Add(o);
                    }
                }
            SortScratchByDistance(me);
            int n = Math.Min(Observation.NumItems, _objScratch.Count);
            for (int i = 0; i < n; i++)
            {
                PhysicalObject o = _objScratch[i];
                Observation.PutI16(Row, Observation.OffItemType + i * 2, o.abstractPhysicalObject.type.Index);
                byte flags = Observation.FlagPresent;
                if (o is Weapon) flags |= 2;
                Row[Observation.OffItemFlags + i] = flags;
                Vector2 rp = o.bodyChunks[0].pos - me;
                Observation.PutF32(Row, Observation.OffItemPos + i * 8, rp.x);
                Observation.PutF32(Row, Observation.OffItemPos + i * 8 + 4, rp.y);
            }
        }

        /// <summary>Sort `_objScratch` by the distance of each object's first body chunk to `me`.</summary>
        private void SortScratchByDistance(Vector2 me)
        {
            _objScratch.Sort(delegate (PhysicalObject a, PhysicalObject b)
            {
                return (a.bodyChunks[0].pos - me).sqrMagnitude.CompareTo((b.bodyChunks[0].pos - me).sqrMagnitude);
            });
        }

        /// <summary>Whether food is in hand, whether a bite is under way, and current speed.</summary>
        private void WriteEatState(Player self)
        {
            bool holdingEdible = false;
            if (self.grasps != null)
                for (int i = 0; i < self.grasps.Length; i++)
                    if (self.grasps[i] != null && self.grasps[i].grabbed is IPlayerEdible
                        && (self.grasps[i].grabbed as IPlayerEdible).Edible)
                        holdingEdible = true;
            Observation.PutU8(Row, Observation.OffHoldingEdible, holdingEdible);
            Observation.PutClampedU8(Row, Observation.OffEatCounter, self.eatCounter);
            float sp = 0f;
            if (self.mainBodyChunk != null && !self.inShortcut)
                sp = (self.mainBodyChunk.pos - self.mainBodyChunk.lastPos).magnitude;
            Observation.PutF32(Row, Observation.OffSpeed, sp);
        }

        private static bool IsGrabbingMe(Creature c, Player self)
        {
            if (c.grasps == null) return false;
            for (int i = 0; i < c.grasps.Length; i++)
                if (c.grasps[i] != null && c.grasps[i].grabbed == self) return true;
            return false;
        }
    }
}
