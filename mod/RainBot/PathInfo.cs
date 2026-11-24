using System;
using System.Collections.Generic;
using RWCustom;
using UnityEngine;

namespace RainBot
{
    /// <summary>
    /// Path lengths, in tiles, from the slugcat to the nearest edible and to the nearest shelter,
    /// using only the game's own movement model. They feed the potential-based shaping
    /// (python/rainbot/reward.py); they are not policy inputs.
    ///
    /// In the room: Dijkstra from the slugcat's tile over the room's AI map
    /// (<c>AItile.outgoingPaths</c>), keeping the connections the game allows the Slugcat creature
    /// template (<c>AImap.IsConnectionAllowedForCreature</c>), each weighted by its own
    /// <c>MovementConnection.distance</c>. Only the length is used, not the template's resistances:
    /// those are AI preferences (a drop "costs" 3000), while the potential needs travel time.
    ///
    /// Between rooms: exit-to-exit lengths from the same in-room search, for every room of the
    /// region. A background survey loads each room once, because the game's prebaked
    /// <c>AbstractRoom.ConnectionLength</c> for the Slugcat template reports "no path" between some
    /// exits a player can cross. A room not surveyed yet falls back to the prebaked length. Each
    /// pipe adds its <c>shortCutLength</c>, and the result is solved into "tiles from entering exit
    /// n of room r to arriving in a shelter". Loading a room also places its own fruit in the world
    /// model (Room.Loaded, minus what the save says was eaten), so the cross-room food search sees
    /// the whole region, not only rooms already visited.
    ///
    /// Computed on the last row of each 8-tick chunk (the row the reward reads) and on room
    /// changes; other rows repeat the last value. -1 = none / unknown.
    /// </summary>
    internal sealed class PathInfo
    {
        public int FoodPath = -1, FoodPoints, FoodBites, ShelterPath = -1;
        public int HeldPoints, HeldBites;
        public int FoodPathHere = -1;     // diagnostics: the in-room part before the cross-room search
        /// <summary>In a shelter: tiles to the nearest tile the game lets the slugcat sleep on (more than
        /// 6 tiles from the entrance, inside the shelter range; Player.Update). -1 elsewhere.</summary>
        public int SleepPath = -1;
        /// <summary>Tiles from the slugcat to every tile of its room (-1 unreachable), last chunk.</summary>
        public int[] HereDist = new int[0];
        public int HereW, HereH, HereRoom = -1, HereMax;

        private CreatureTemplate _tpl;
        private int[] _dist = new int[0];
        private readonly List<int> _heapK = new List<int>(), _heapV = new List<int>();
        private World _world;
        private readonly Dictionary<int, int[]> _exitToShelter = new Dictionary<int, int[]>();
        private bool _checked;

        // Cross-room food: per room, the first time the game has it realized, a Dijkstra distance
        // table from each exit's pipe entrance (cached for the life of the game process; rooms the
        // game never loaded have no placed objects in its world model).
        private sealed class RoomTable
        {
            public int W, H;
            public Dictionary<int, int[]> FromNode = new Dictionary<int, int[]>();
            public Dictionary<int, IntVector2> Entr = new Dictionary<int, IntVector2>();
        }
        // Survey: one room loading at a time; whether the survey loaded it (then it unloads it).
        private AbstractRoom _surveying;
        private bool _surveyLoaded;
        private int _surveyWait;
        private readonly HashSet<int> _surveyFailed = new HashSet<int>();
        private bool _solveDirty;
        /// <summary>Rooms with standing water (the game's own Room.water and a water level above the
        /// floor), learned as the survey loads each room; for the curriculum's dry/wet starts.</summary>
        public readonly List<string> WaterRooms = new List<string>();
        /// <summary>Set once, when every room has a table or failed to load; Lockstep then reports
        /// `water_rooms` to the agent.</summary>
        public bool SurveyComplete;
        private readonly Dictionary<int, RoomTable> _tables = new Dictionary<int, RoomTable>();
        // Food points and bites per object type, learned from realized edibles (abstract objects
        // in other rooms carry no FoodPoints of their own).
        private readonly Dictionary<string, int[]> _edibleTypes = new Dictionary<string, int[]>();
        // Region graph for the arrival search: state = (room, node).
        private readonly Dictionary<int, int> _nodeBase = new Dictionary<int, int>();
        private int _nStates;
        private int[] _arrive = new int[0];
        // Observed player moves: the AI map has no connection for a player's jumps (up ledges,
        // across gaps, off poles), so some exits have no path without them. Every recorded
        // trajectory (python -m rainbot.scripts.observed_moves -> RainBotData/moves-<region>.txt,
        // human sessions and the agent's records) adds, per room, an edge between consecutive tiles
        // the slugcat stood on, length = observed ticks x human travel speed (so length / speed in
        // the reward gives back the observed time). Directed: a drop is not a climb.
        private readonly Dictionary<string, List<int[]>> _moveRuns = new Dictionary<string, List<int[]>>();
        private float _moveSpeed;
        private readonly Dictionary<int, Dictionary<int, List<int>>> _extra = new Dictionary<int, Dictionary<int, List<int>>>();
        public int ExtraEdges;

        public void Update(Player self, Room room, AbstractRoom aroom, bool inPipe)
        {
            HeldPoints = HeldBites = 0;
            if (self.grasps != null)
                for (int i = 0; i < self.grasps.Length; i++)
                {
                    IPlayerEdible e = self.grasps[i] != null ? self.grasps[i].grabbed as IPlayerEdible : null;
                    if (e != null && e.Edible) { HeldPoints = e.FoodPoints; HeldBites = e.BitesLeft; break; }
                }
            if (aroom != null && aroom.shelter) ShelterPath = 0;
            if (_tpl == null) _tpl = StaticWorld.GetCreatureTemplate(CreatureTemplate.Type.Slugcat);
            if (_tpl == null || _tpl.PreBakedPathingIndex < 0) return;
            World w = self.abstractCreature != null ? self.abstractCreature.world : null;
            if (w != null)
            {
                EnsureWorld(w);
                SurveyStep(w, aroom);
                if (_solveDirty) { _solveDirty = false; Solve(w); }
            }
            if (room == null || room.aimap == null || inPipe || self.mainBodyChunk == null) return;   // hold

            EnsureWorld(room.world);
            if (!_tables.ContainsKey(aroom.index)) BuildTable(room, aroom);
            IntVector2 start;
            if (!StartTile(room, self, out start)) return;                                      // in the air: hold
            Dijkstra(room, start);
            // The slugcat's own distance field (tiles along the movement graph, jump links included),
            // for the frame's distance channel (FrameCapture, protocol v11).
            int nHere = room.TileWidth * room.TileHeight;
            if (HereDist.Length != nHere) HereDist = new int[nHere];
            Array.Copy(_dist, HereDist, nHere);
            PaintAir(room, HereDist);
            HereMax = 0;
            for (int i = 0; i < nHere; i++) if (HereDist[i] > HereMax) HereMax = HereDist[i];
            HereW = room.TileWidth; HereH = room.TileHeight; HereRoom = aroom.index;

            // nearest edible not in hand
            FoodPath = -1; FoodPoints = 0; FoodBites = 0;
            if (room.physicalObjects != null)
                for (int layer = 0; layer < room.physicalObjects.Length; layer++)
                    foreach (PhysicalObject o in room.physicalObjects[layer])
                    {
                        IPlayerEdible e = o as IPlayerEdible;
                        if (e == null || !e.Edible || o.firstChunk == null) continue;
                        if (o.grabbedBy != null && o.grabbedBy.Count > 0 && o.grabbedBy[0].grabber == self) continue;
                        string tn = o.abstractPhysicalObject.type.value;
                        if (!_edibleTypes.ContainsKey(tn)) _edibleTypes[tn] = new[] { e.FoodPoints, e.BitesLeft };
                        int d = Reach(room, room.GetTilePosition(o.firstChunk.pos));
                        if (d >= 0 && (FoodPath < 0 || d < FoodPath)) { FoodPath = d; FoodPoints = e.FoodPoints; FoodBites = e.BitesLeft; }
                    }

            // edibles in other rooms: path to an exit + region graph + that room's cached table
            int[] exitReach = new int[aroom.nodes.Length];
            for (int n = 0; n < exitReach.Length; n++)
            {
                IntVector2 et;
                exitReach[n] = Entrance(room, n, out et) ? Reach(room, et) : -1;
            }
            FoodPathHere = FoodPath;
            CrossRoomFood(room.world, aroom, exitReach);

            // nearest shelter: through each exit of this room
            SleepPath = -1;
            if (aroom.shelter)
            {
                ShelterPath = 0;
                try
                {
                    IntVector2 door = room.LocalCoordinateOfNode(0).Tile;
                    int H = room.TileHeight;
                    for (int x = 0; x < room.TileWidth; x++)
                        for (int y = 0; y < H; y++)
                        {
                            int d = _dist[x * H + y];
                            if (d < 0 || (SleepPath >= 0 && d >= SleepPath)) continue;
                            IntVector2 t = new IntVector2(x, y);
                            if (Custom.ManhattanDistance(t, door) > 6 && ShelterDoor.IsTileInsideShelterRange(aroom, t)) SleepPath = d;
                        }
                }
                catch (Exception) { }
                return;
            }
            int[] toShelter = ExitToShelter(room.world, aroom);
            int best = -1;
            for (int n = 0; n < aroom.nodes.Length && n < room.exitAndDenIndex.Length; n++)
            {
                if (toShelter[n] < 0) continue;
                IntVector2 et;
                if (!Entrance(room, n, out et)) continue;
                int d = Reach(room, et);
                if (d < 0) continue;
                if (best < 0 || d + toShelter[n] < best) best = d + toShelter[n];
            }
            ShelterPath = best;
            if (!_checked) { _checked = true; UnitCheck(room, aroom); }
        }

        /// <summary>The slugcat's tile, or the nearest accessible tile around or below it.</summary>
        private bool StartTile(Room room, Player self, out IntVector2 t)
        {
            IntVector2 c = room.GetTilePosition(self.mainBodyChunk.pos);
            IntVector2 b = self.bodyChunks.Length > 1 ? room.GetTilePosition(self.bodyChunks[1].pos) : c;
            return StandTile(room, c, b, out t);
        }

        private bool StandTile(Room room, IntVector2 c, IntVector2 b, out IntVector2 t)
        {
            foreach (IntVector2 p in new[] { c, b, c + new IntVector2(0, -1), b + new IntVector2(0, -1),
                                             c + new IntVector2(-1, 0), c + new IntVector2(1, 0), c + new IntVector2(0, -2) })
                if (Inside(room, p) && room.aimap.TileAccessibleToCreature(p, _tpl)) { t = p; return true; }
            t = c;
            return false;
        }

        /// <summary>Where the slugcat enters node n's pipe: the shortcut's in-room start tile
        /// (`exitAndDenIndex` is the pipe's other end, at the room edge inside the wall).</summary>
        private static bool Entrance(Room room, int n, out IntVector2 t)
        {
            t = new IntVector2(-1, -1);
            try
            {
                // Room.ShortcutLeadingToNode falls back to shortcuts[0] (with a warning) when no
                // shortcut leads to n, so search directly.
                for (int i = 0; i < room.shortcuts.Length; i++)
                    if (room.shortcuts[i].destNode == n) { t = room.shortcuts[i].StartTile; return true; }
                return false;
            }
            catch (Exception) { return false; }
        }

        private static bool Inside(Room room, IntVector2 p)
        {
            return p.x >= 0 && p.y >= 0 && p.x < room.TileWidth && p.y < room.TileHeight;
        }

        /// <summary>Shortest reach of a target tile: the tile itself or one next to/under it
        /// (a fruit hangs in the air; the slugcat grabs it from an adjacent tile).</summary>
        private int Reach(Room room, IntVector2 t)
        {
            int best = -1;
            for (int dx = -1; dx <= 1; dx++)
                for (int dy = -2; dy <= 1; dy++)
                {
                    IntVector2 p = new IntVector2(t.x + dx, t.y + dy);
                    if (!Inside(room, p)) continue;
                    int d = _dist[p.x * room.TileHeight + p.y];
                    if (d >= 0)
                    {
                        d += Math.Max(Math.Abs(dx), Math.Abs(dy));
                        if (best < 0 || d < best) best = d;
                    }
                }
            return best;
        }

        private void Dijkstra(Room room, IntVector2 start)
        {
            int W = room.TileWidth, H = room.TileHeight;
            if (_dist.Length < W * H) _dist = new int[W * H];
            for (int i = 0; i < W * H; i++) _dist[i] = -1;
            _heapK.Clear(); _heapV.Clear();
            int s = start.x * H + start.y;
            _dist[s] = 0;
            Push(0, s);
            AImap map = room.aimap;
            int ri = room.abstractRoom.index;
            Dictionary<int, List<int>> extra = Extras(room);
            while (_heapK.Count > 0)
            {
                int d, u;
                Pop(out d, out u);
                if (d > _dist[u]) continue;
                AItile tile = map.getAItile(u / H, u % H);
                if (tile == null || tile.outgoingPaths == null) continue;
                foreach (MovementConnection c in tile.outgoingPaths)
                {
                    if (c.destinationCoord.room != ri || !c.destinationCoord.TileDefined) continue;
                    IntVector2 dt = c.DestTile;
                    if (!Inside(room, dt)) continue;
                    if (!map.IsConnectionAllowedForCreature(c, _tpl)) continue;
                    int v = dt.x * H + dt.y;
                    int nd = d + Math.Max(1, c.distance);
                    if (_dist[v] < 0 || nd < _dist[v]) { _dist[v] = nd; Push(nd, v); }
                }
                List<int> ex;
                if (extra != null && extra.TryGetValue(u, out ex))
                    for (int i = 0; i + 1 < ex.Count; i += 2)
                    {
                        int v = ex[i], nd = d + ex[i + 1];
                        if (_dist[v] < 0 || nd < _dist[v]) { _dist[v] = nd; Push(nd, v); }
                    }
            }
        }

        /// <summary>The instance the observation uses (one per game), for the reset's start selection.
        /// The reset must measure paths on the same graph as shelter_path: on the AI map alone some
        /// shelter routes that need player jumps do not exist.</summary>
        public static PathInfo Current;
        public PathInfo() { Current = this; }

        /// <summary>The extra edges (observed moves, jump links) of a room: tile -> [dest, len, ...].</summary>
        public Dictionary<int, List<int>> ExtraEdgesOf(Room room) { return Extras(room); }

        /// <summary>The observed-move and jump edges of a room (built on first use, from its AI map):
        /// tile index -> [dest, length, dest, length, ...], the shortest observation per pair.</summary>
        private Dictionary<int, List<int>> Extras(Room room)
        {
            AbstractRoom a = room.abstractRoom;
            Dictionary<int, List<int>> ex;
            if (_extra.TryGetValue(a.index, out ex)) return ex;
            ex = null;
            List<int[]> visits;
            int H = room.TileHeight;
            Dictionary<long, int> best = new Dictionary<long, int>();
            if (_moveSpeed > 0f && _moveRuns.TryGetValue(a.name, out visits))
            {
                foreach (int[] runs in visits)
                {
                    int prev = -1, prevEnd = 0;
                    for (int i = 0; i + 5 < runs.Length; i += 6)
                    {
                        IntVector2 t;
                        if (!StandTile(room, new IntVector2(runs[i + 2], runs[i + 3]), new IntVector2(runs[i + 4], runs[i + 5]), out t)) continue;
                        int u = t.x * H + t.y;
                        if (prev >= 0 && u != prev)
                        {
                            int len = Math.Max(1, (int)Math.Ceiling((runs[i] - prevEnd) * _moveSpeed));
                            long key = (long)prev * (1L << 32) + u;
                            int old;
                            if (!best.TryGetValue(key, out old) || len < old) best[key] = len;
                        }
                        prev = u; prevEnd = runs[i + 1];
                    }
                }
            }
            if (_moveSpeed > 0f && _jumps.Count > 0) AddJumps(room, best);
            if (best.Count > 0)
            {
                ex = new Dictionary<int, List<int>>();
                foreach (KeyValuePair<long, int> kv in best)
                {
                    int u = (int)(kv.Key >> 32), v = (int)(kv.Key & 0xffffffffL);
                    List<int> l;
                    if (!ex.TryGetValue(u, out l)) ex[u] = l = new List<int>();
                    l.Add(v); l.Add(kv.Value);
                }
                ExtraEdges += best.Count;
            }
            _extra[a.index] = ex;
            return ex;
        }

        // Player jumps the AI map lacks: envelopes per (from, to) tile kind, measured from every
        // human jump (rainbot.scripts.missing_moves -> RainBotData/jumps-<region>.txt): horizontal
        // reach, height range, median air time. Tile kinds: floor, pole, wall.
        private readonly List<int[]> _jumps = new List<int[]>();      // fromKind, toKind, dxMax, dyMin, dyMax, airTicks
        private static int Kind(string k) { return k == "floor" ? 1 : k == "pole" ? 2 : k == "wall" ? 3 : 0; }

        private static int TileKind(Room room, int x, int y)
        {
            AItile t = room.aimap.getAItile(x, y);
            if (t == null) return 0;
            if (t.acc == AItile.Accessibility.Floor) return 1;
            if (t.acc == AItile.Accessibility.Climb) return 2;
            if (t.acc == AItile.Accessibility.Wall) return 3;
            return 0;
        }

        /// <summary>Links every floor / pole / wall tile to every tile of a kind humans jumped to, within
        /// that jump's measured reach, when the straight line between them crosses no solid tile. Length
        /// = the jump's air time x the human travel speed (as for the observed moves).</summary>
        private void AddJumps(Room room, Dictionary<long, int> best)
        {
            if (room.aimap == null) return;
            int W = room.TileWidth, H = room.TileHeight;
            int[] kind = new int[W * H];
            for (int x = 0; x < W; x++) for (int y = 0; y < H; y++) kind[x * H + y] = TileKind(room, x, y);
            int added = 0;
            for (int x = 0; x < W; x++)
                for (int y = 0; y < H; y++)
                {
                    int k0 = kind[x * H + y];
                    if (k0 == 0) continue;
                    foreach (int[] j in _jumps)
                    {
                        if (j[0] != k0) continue;
                        int len = Math.Max(1, (int)Math.Ceiling(j[5] * _moveSpeed));
                        for (int dx = -j[2]; dx <= j[2]; dx++)
                            for (int dy = j[3]; dy <= j[4]; dy++)
                            {
                                if (dx == 0 && dy == 0) continue;
                                int tx = x + dx, ty = y + dy;
                                if (tx < 0 || ty < 0 || tx >= W || ty >= H || kind[tx * H + ty] != j[1]) continue;
                                if (!ClearLine(room, x, y, tx, ty)) continue;
                                long key = (long)(x * H + y) * (1L << 32) + (tx * H + ty);
                                int old;
                                if (!best.TryGetValue(key, out old) || len < old) { best[key] = len; added++; }
                            }
                    }
                }
            JumpEdges += added;
        }

        private static bool ClearLine(Room room, int x0, int y0, int x1, int y1)
        {
            int n = Math.Max(Math.Abs(x1 - x0), Math.Abs(y1 - y0));
            for (int i = 1; i < n; i++)
            {
                int x = x0 + (int)Math.Round((x1 - x0) * (double)i / n), y = y0 + (int)Math.Round((y1 - y0) * (double)i / n);
                if (room.GetTile(x, y).Solid) return false;
            }
            return true;
        }

        public int JumpEdges;

        /// <summary>Arrival-time flood into the air. The graph only has tiles the slugcat stands or
        /// holds on, and jumps and drops are the only ways through the air between them. Every move
        /// out of a reached tile (the game's connections, observed moves, jump links)
        /// paints the tiles on its straight line with the time it gets there: start time + the move's
        /// length x the fraction covered. A tile keeps its earliest time; air no move crosses stays -1.</summary>
        private void PaintAir(Room room, int[] dist)
        {
            AImap map = room.aimap;
            int H = room.TileHeight, n = room.TileWidth * H, ri = room.abstractRoom.index;
            Dictionary<int, List<int>> extra = Extras(room);
            for (int u = 0; u < n; u++)
            {
                int d = _dist[u];
                if (d < 0) continue;
                int ux = u / H, uy = u % H;
                AItile tile = map.getAItile(ux, uy);
                if (tile != null && tile.outgoingPaths != null)
                    foreach (MovementConnection c in tile.outgoingPaths)
                    {
                        if (c.destinationCoord.room != ri || !c.destinationCoord.TileDefined) continue;
                        if (!map.IsConnectionAllowedForCreature(c, _tpl)) continue;
                        IntVector2 dt = c.DestTile;
                        if (!Inside(room, dt)) continue;
                        Segment(room, dist, ux, uy, dt.x, dt.y, d, Math.Max(1, c.distance));
                    }
                List<int> ex;
                if (extra != null && extra.TryGetValue(u, out ex))
                    for (int i = 0; i + 1 < ex.Count; i += 2)
                        Segment(room, dist, ux, uy, ex[i] / H, ex[i] % H, d, ex[i + 1]);
            }
        }

        private static void Segment(Room room, int[] dist, int x0, int y0, int x1, int y1, int d, int len)
        {
            int m = Math.Max(Math.Abs(x1 - x0), Math.Abs(y1 - y0)), H = room.TileHeight;
            for (int i = 1; i < m; i++)
            {
                int x = x0 + (int)Math.Round((x1 - x0) * (double)i / m), y = y0 + (int)Math.Round((y1 - y0) * (double)i / m);
                if (room.GetTile(x, y).Solid) continue;
                int t = d + (int)Math.Ceiling(len * (double)i / m), k = x * H + y;
                if (dist[k] < 0 || t < dist[k]) dist[k] = t;
            }
        }

        /// <summary>Reads RainBotData/jumps-&lt;region&gt;.txt and moves-&lt;region&gt;.txt (see Extras);
        /// a missing file means no jumps or no observed moves.</summary>
        private void LoadMoves(World world)
        {
            _moveRuns.Clear(); _extra.Clear(); _moveSpeed = 0f; ExtraEdges = 0; JumpEdges = 0; _jumps.Clear();
            try
            {
                string jp = System.IO.Path.Combine(System.IO.Path.Combine(BepInEx.Paths.GameRootPath, "RainBotData"), "jumps-" + world.name + ".txt");
                if (System.IO.File.Exists(jp))
                    foreach (string line in System.IO.File.ReadAllLines(jp))
                    {
                        if (line.StartsWith("#")) continue;
                        string[] f = line.Split(' ');
                        if (f.Length < 6) continue;
                        int k0 = Kind(f[0]), k1 = Kind(f[1]);
                        if (k0 == 0 || k1 == 0) continue;
                        _jumps.Add(new[] { k0, k1, int.Parse(f[2]), int.Parse(f[3]), int.Parse(f[4]), int.Parse(f[5]) });
                    }
                Plugin.Log.LogMessage("paths: " + _jumps.Count + " jump envelopes");
            }
            catch (Exception e) { Plugin.Log.LogWarning("paths: jumps not read: " + e.Message); _jumps.Clear(); }
            try
            {
                string path = System.IO.Path.Combine(System.IO.Path.Combine(BepInEx.Paths.GameRootPath, "RainBotData"), "moves-" + world.name + ".txt");
                if (!System.IO.File.Exists(path)) { Plugin.Log.LogWarning("paths: no observed moves at " + path); return; }
                foreach (string line in System.IO.File.ReadAllLines(path))
                {
                    int sp = line.IndexOf(' ');
                    if (sp <= 0) continue;
                    string head = line.Substring(0, sp), rest = line.Substring(sp + 1);
                    if (head == "speed") { _moveSpeed = float.Parse(rest, System.Globalization.CultureInfo.InvariantCulture); continue; }
                    string[] runs = rest.Split(';');
                    int[] flat = new int[runs.Length * 6];
                    int n = 0;
                    foreach (string r in runs)
                    {
                        string[] f = r.Split(',');
                        if (f.Length != 6) continue;
                        for (int k = 0; k < 6; k++) flat[n++] = int.Parse(f[k]);
                    }
                    if (n < flat.Length) Array.Resize(ref flat, n);
                    List<int[]> l;
                    if (!_moveRuns.TryGetValue(head, out l)) _moveRuns[head] = l = new List<int[]>();
                    l.Add(flat);
                }
                Plugin.Log.LogMessage("paths: observed moves for " + _moveRuns.Count + " rooms, speed " + _moveSpeed);
            }
            catch (Exception e) { Plugin.Log.LogWarning("paths: observed moves not read: " + e.Message); _moveRuns.Clear(); }
        }

        private void Push(int k, int v)
        {
            _heapK.Add(k); _heapV.Add(v);
            int i = _heapK.Count - 1;
            while (i > 0)
            {
                int p = (i - 1) / 2;
                if (_heapK[p] <= _heapK[i]) break;
                Swap(i, p); i = p;
            }
        }

        private void Pop(out int k, out int v)
        {
            k = _heapK[0]; v = _heapV[0];
            int last = _heapK.Count - 1;
            _heapK[0] = _heapK[last]; _heapV[0] = _heapV[last];
            _heapK.RemoveAt(last); _heapV.RemoveAt(last);
            int i = 0, n = _heapK.Count;
            while (true)
            {
                int l = 2 * i + 1, r = l + 1, m = i;
                if (l < n && _heapK[l] < _heapK[m]) m = l;
                if (r < n && _heapK[r] < _heapK[m]) m = r;
                if (m == i) break;
                Swap(i, m); i = m;
            }
        }

        private void Swap(int a, int b)
        {
            int k = _heapK[a]; _heapK[a] = _heapK[b]; _heapK[b] = k;
            int v = _heapV[a]; _heapV[a] = _heapV[b]; _heapV[b] = v;
        }

        /// <summary>Per node of a room: tiles from entering that exit to arriving in a shelter
        /// room (-1 = no route). Solved once per world by relaxation over the room graph.</summary>
        private int[] ExitToShelter(World world, AbstractRoom aroom)
        {
            EnsureWorld(world);
            int[] r;
            if (_exitToShelter.TryGetValue(aroom.index, out r)) return r;
            r = new int[aroom.nodes.Length];
            for (int i = 0; i < r.Length; i++) r[i] = -1;
            return r;
        }

        private void Solve(World world)
        {
            AbstractRoom[] rooms = world.abstractRooms;
            foreach (AbstractRoom a in rooms)
            {
                int[] r = new int[a.nodes.Length];
                for (int i = 0; i < r.Length; i++) r[i] = -1;
                _exitToShelter[a.index] = r;
            }
            // arrive[t][m]: tiles from arriving at node m of room t to arriving in a shelter.
            bool changed = true;
            for (int iter = 0; iter < 200 && changed; iter++)
            {
                changed = false;
                foreach (AbstractRoom a in rooms)
                {
                    int[] r = _exitToShelter[a.index];
                    for (int n = 0; n < a.nodes.Length && n < a.connections.Length; n++)
                    {
                        if (a.nodes[n].type != AbstractRoomNode.Type.Exit) continue;
                        int ti = a.connections[n];
                        if (ti < 0) continue;
                        AbstractRoom t = world.GetAbstractRoom(ti);
                        if (t == null) continue;
                        int m = t.ExitIndex(a.index);
                        int pipe = Math.Max(0, a.nodes[n].shortCutLength);
                        int arrive = -1;
                        if (t.shelter) arrive = 0;
                        else if (m >= 0)
                        {
                            int[] tr = _exitToShelter[t.index];
                            for (int k = 0; k < t.nodes.Length && k < tr.Length; k++)
                            {
                                if (k == m || tr[k] < 0) continue;
                                int len = Len(t, m, k);
                                if (len < 0) continue;
                                if (arrive < 0 || len + tr[k] < arrive) arrive = len + tr[k];
                            }
                        }
                        if (arrive < 0) continue;
                        int v = pipe + arrive;
                        if (r[n] < 0 || v < r[n]) { r[n] = v; changed = true; }
                    }
                }
            }
        }

        /// <summary>Everything cached here is per world (region): rebuild on a world change.</summary>
        private void EnsureWorld(World world)
        {
            if (ReferenceEquals(world, _world)) return;
            _world = world;
            _exitToShelter.Clear();
            _tables.Clear();
            _nodeBase.Clear(); _nStates = 0;
            _surveying = null; _surveyFailed.Clear(); _solveDirty = false;
            WaterRooms.Clear(); SurveyComplete = false;
            foreach (AbstractRoom a in world.abstractRooms) { _nodeBase[a.index] = _nStates; _nStates += a.nodes.Length; }
            LoadMoves(world);
            Solve(world);
        }

        private void BuildTable(Room room, AbstractRoom aroom)
        {
            RoomTable t = new RoomTable { W = room.TileWidth, H = room.TileHeight };
            for (int n = 0; n < aroom.nodes.Length; n++)
            {
                IntVector2 et;
                if (aroom.nodes[n].type != AbstractRoomNode.Type.Exit || !Entrance(room, n, out et) || !Inside(room, et)) continue;
                Dijkstra(room, et);
                int[] copy = new int[t.W * t.H];
                Array.Copy(_dist, copy, copy.Length);
                t.FromNode[n] = copy;
                t.Entr[n] = et;
            }
            _tables[aroom.index] = t;
            _solveDirty = true;
            if (room.water && room.defaultWaterLevel > 0 && !WaterRooms.Contains(aroom.name)) WaterRooms.Add(aroom.name);
            if (room.physicalObjects != null)
                for (int layer = 0; layer < room.physicalObjects.Length; layer++)
                    foreach (PhysicalObject o in room.physicalObjects[layer])
                    {
                        IPlayerEdible e = o as IPlayerEdible;
                        if (e == null || o.abstractPhysicalObject == null) continue;
                        string tn = o.abstractPhysicalObject.type.value;
                        if (!_edibleTypes.ContainsKey(tn)) _edibleTypes[tn] = new[] { e.FoodPoints, e.BitesLeft };
                    }
        }

        /// <summary>Tiles from entering room a at node m to entering its node k: this module's own
        /// search where the room is surveyed, else the game's prebaked length (-1 = no path).</summary>
        private int Len(AbstractRoom a, int m, int k)
        {
            RoomTable t;
            if (_tables.TryGetValue(a.index, out t))
            {
                int[] dm; IntVector2 ek;
                if (t.FromNode.TryGetValue(m, out dm) && t.Entr.TryGetValue(k, out ek))
                {
                    int d = ReachIn(dm, t.W, t.H, ek);
                    if (d >= 0) return d;
                }
            }
            return a.ConnectionLength(m, k, _tpl);
        }

        /// <summary>One step of the region survey: load the next room without a table (the game's
        /// own World.ActivateRoom), wait until its AI map is ready, build its table, and unload it
        /// again if the survey was what loaded it. One room at a time; a room that does not get
        /// ready is skipped.</summary>
        private void SurveyStep(World world, AbstractRoom here)
        {
            try
            {
                if (_surveying != null)
                {
                    AbstractRoom a = _surveying;
                    Room r = a.realizedRoom;
                    if (r == null || ++_surveyWait > 600) { _surveyFailed.Add(a.index); _surveying = null; return; }
                    if (!r.readyForAI || r.aimap == null) return;
                    if (!_tables.ContainsKey(a.index)) BuildTable(r, a);
                    if (_surveyLoaded && a != here && (here == null || a.index != here.index)) a.Abstractize();
                    _surveying = null;
                    return;
                }
                foreach (AbstractRoom a in world.abstractRooms)
                {
                    if (a == null || a.offScreenDen || _tables.ContainsKey(a.index) || _surveyFailed.Contains(a.index)) continue;
                    if (a.realizedRoom != null && a.realizedRoom.readyForAI && a.realizedRoom.aimap != null)
                    {
                        BuildTable(a.realizedRoom, a);       // already loaded by the game
                        return;
                    }
                    _surveyLoaded = a.realizedRoom == null;
                    if (_surveyLoaded) world.ActivateRoom(a);
                    _surveying = a; _surveyWait = 0;
                    return;
                }
                SurveyComplete = true;
            }
            catch (Exception e)
            {
                if (_surveying != null) _surveyFailed.Add(_surveying.index);
                _surveying = null;
                Plugin.Log.LogWarning("paths survey: " + e.Message);
            }
        }

        private static int ReachIn(int[] dist, int W, int H, IntVector2 t)
        {
            int best = -1;
            for (int dx = -1; dx <= 1; dx++)
                for (int dy = -2; dy <= 1; dy++)
                {
                    int x = t.x + dx, y = t.y + dy;
                    if (x < 0 || y < 0 || x >= W || y >= H) continue;
                    int d = dist[x * H + y];
                    if (d >= 0) { d += Math.Max(Math.Abs(dx), Math.Abs(dy)); if (best < 0 || d < best) best = d; }
                }
            return best;
        }

        private int State(int room, int node)
        {
            int b;
            return _nodeBase.TryGetValue(room, out b) ? b + node : -1;
        }

        /// <summary>Dijkstra over (room, node) from the slugcat's exits, then the nearest edible in any
        /// room with a cached table; replaces FoodPath if shorter.</summary>
        private void CrossRoomFood(World world, AbstractRoom here, int[] exitReach)
        {
            EnsureWorld(world);
            if (_arrive.Length < _nStates) _arrive = new int[_nStates];
            for (int i = 0; i < _nStates; i++) _arrive[i] = -1;
            _heapK.Clear(); _heapV.Clear();
            // leaving `here` through exit n lands at node m of the next room after the pipe
            for (int n = 0; n < exitReach.Length && n < here.connections.Length; n++)
            {
                if (exitReach[n] < 0 || here.nodes[n].type != AbstractRoomNode.Type.Exit || here.connections[n] < 0) continue;
                AbstractRoom t = world.GetAbstractRoom(here.connections[n]);
                if (t == null) continue;
                int m = t.ExitIndex(here.index), sid = State(t.index, m);
                if (m < 0 || sid < 0) continue;
                int c = exitReach[n] + Math.Max(0, here.nodes[n].shortCutLength);
                if (_arrive[sid] < 0 || c < _arrive[sid]) { _arrive[sid] = c; Push(c, sid); }
            }
            while (_heapK.Count > 0)
            {
                int d, u;
                Pop(out d, out u);
                if (d > _arrive[u]) continue;
                AbstractRoom a = null; int m = -1;
                foreach (KeyValuePair<int, int> kv in _nodeBase)            // state -> (room, node)
                {
                    AbstractRoom r = world.GetAbstractRoom(kv.Key);
                    if (r != null && u >= kv.Value && u < kv.Value + r.nodes.Length) { a = r; m = u - kv.Value; break; }
                }
                if (a == null || a.index == here.index) continue;
                for (int k = 0; k < a.nodes.Length && k < a.connections.Length; k++)
                {
                    if (k == m || a.nodes[k].type != AbstractRoomNode.Type.Exit || a.connections[k] < 0) continue;
                    int len = Len(a, m, k);
                    if (len < 0) continue;
                    AbstractRoom t = world.GetAbstractRoom(a.connections[k]);
                    if (t == null || t.index == here.index) continue;
                    int mm = t.ExitIndex(a.index), sid = State(t.index, mm);
                    if (mm < 0 || sid < 0) continue;
                    int c = d + len + Math.Max(0, a.nodes[k].shortCutLength);
                    if (_arrive[sid] < 0 || c < _arrive[sid]) { _arrive[sid] = c; Push(c, sid); }
                }
            }
            foreach (KeyValuePair<int, RoomTable> kv in _tables)
            {
                if (kv.Key == here.index) continue;
                AbstractRoom a = world.GetAbstractRoom(kv.Key);
                if (a == null) continue;
                RoomTable tab = kv.Value;
                foreach (AbstractWorldEntity ent in a.entities)
                {
                    AbstractPhysicalObject o = ent as AbstractPhysicalObject;
                    if (o == null || o is AbstractCreature || !o.pos.TileDefined) continue;
                    int[] pb;
                    if (!_edibleTypes.TryGetValue(o.type.value, out pb) || pb[0] <= 0) continue;
                    if (o.stuckObjects != null && o.stuckObjects.Count > 0) continue;       // held by something
                    foreach (KeyValuePair<int, int[]> nd in tab.FromNode)
                    {
                        int sid = State(a.index, nd.Key);
                        if (sid < 0 || _arrive[sid] < 0) continue;
                        int r = ReachIn(nd.Value, tab.W, tab.H, o.pos.Tile);
                        if (r < 0) continue;
                        int total = _arrive[sid] + r;
                        if (FoodPath < 0 || total < FoodPath) { FoodPath = total; FoodPoints = pb[0]; FoodBites = pb[1]; }
                    }
                }
            }
        }

        /// <summary>The `pathdebug` command: every node of the current room, its tile, whether it is
        /// reachable, and its exit-to-shelter length.</summary>
        public string Debug(Player self, Room room)
        {
            System.Text.StringBuilder sb = new System.Text.StringBuilder("pathdebug");
            if (room == null) return sb.Append(" no room").ToString();
            AbstractRoom a = room.abstractRoom;
            if (_tpl == null) _tpl = StaticWorld.GetCreatureTemplate(CreatureTemplate.Type.Slugcat);
            IntVector2 st;
            bool ok = StartTile(room, self, out st);
            if (ok) Dijkstra(room, st);
            int[] ts = ExitToShelter(room.world, a);
            int nValid = 0;
            foreach (int[] v in _exitToShelter.Values) foreach (int x in v) if (x >= 0) nValid++;
            sb.Append(' ').Append(a.name).Append(" start=").Append(ok ? st.ToString() : "none")
              .Append(" nodes=").Append(a.nodes.Length).Append(" exitAndDen=").Append(room.exitAndDenIndex.Length)
              .Append(" worldValidExits=").Append(nValid).Append(" rooms=").Append(room.world.abstractRooms.Length)
              .Append(" tables=").Append(_tables.Count).Append(" movedRooms=").Append(_moveRuns.Count).Append(" extraEdges=").Append(ExtraEdges).Append(" jumpEdges=").Append(JumpEdges).Append(" surveyFailed=").Append(_surveyFailed.Count).Append(" edibleTypes=").Append(string.Join(",", new List<string>(_edibleTypes.Keys).ToArray()))
              .Append(" foodPath=").Append(FoodPath).Append(" foodPathHere=").Append(FoodPathHere).Append(" shelterPath=").Append(ShelterPath);
            for (int n = 0; n < a.nodes.Length; n++)
            {
                IntVector2 t;
                Entrance(room, n, out t);
                sb.Append(" | n").Append(n).Append(' ').Append(a.nodes[n].type.value)
                  .Append(" conn=").Append(n < a.connections.Length ? a.connections[n] : -9)
                  .Append(" tile=").Append(t.ToString())
                  .Append(" acc=").Append(Inside(room, t) && room.aimap.TileAccessibleToCreature(t, _tpl))
                  .Append(" reach=").Append(ok && Inside(room, t) ? Reach(room, t) : -9)
                  .Append(" toShelter=").Append(n < ts.Length ? ts[n] : -9);
                if (n < a.connections.Length && a.connections[n] >= 0)
                {
                    AbstractRoom tr = room.world.GetAbstractRoom(a.connections[n]);
                    if (tr != null)
                    {
                        int m = tr.ExitIndex(a.index);
                        sb.Append(" ->").Append(tr.name).Append(" m=").Append(m).Append(" shelter=").Append(tr.shelter);
                        if (m >= 0)
                            for (int k = 0; k < tr.nodes.Length; k++)
                                if (k != m) sb.Append(" len").Append(k).Append('=').Append(Len(tr, m, k))
                                              .Append('/').Append(tr.ConnectionLength(m, k, _tpl));
                    }
                }
            }
            return sb.ToString();
        }

        /// <summary>Once: compare the game's prebaked node-to-node lengths with this room's own
        /// Dijkstra between the same exits, to confirm both are in tiles.</summary>
        private void UnitCheck(Room room, AbstractRoom aroom)
        {
            try
            {
                for (int m = 0; m < aroom.nodes.Length && m < room.exitAndDenIndex.Length; m++)
                {
                    if (aroom.nodes[m].type != AbstractRoomNode.Type.Exit) continue;
                    IntVector2 st;
                    if (!Entrance(room, m, out st) || !Inside(room, st)) continue;
                    Dijkstra(room, st);
                    for (int k = 0; k < aroom.nodes.Length && k < room.exitAndDenIndex.Length; k++)
                    {
                        if (k == m || aroom.nodes[k].type != AbstractRoomNode.Type.Exit) continue;
                        IntVector2 kt;
                        if (!Entrance(room, k, out kt)) continue;
                        Plugin.Log.LogMessage("paths unitcheck " + aroom.name + " " + m + "->" + k + ": prebaked="
                            + aroom.ConnectionLength(m, k, _tpl) + " dijkstra=" + Reach(room, kt));
                    }
                    return;
                }
            }
            catch (Exception e) { Plugin.Log.LogWarning("paths unitcheck failed: " + e.Message); }
        }
    }
}
