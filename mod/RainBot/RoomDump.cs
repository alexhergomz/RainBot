using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace RainBot
{
    /// <summary>
    /// One-time export of a room's static geometry: camera rects and the tile grid.
    ///
    /// Dumped once per room, not logged per tick, for two reasons.
    ///
    /// First, it is static. Any tile patch around the player can be rebuilt offline from this
    /// dump plus the logged player position, with no loss.
    ///
    /// Second, the session log records which camera is active but not where it is. A room spans
    /// several camera positions and the camera snaps between them, so logged positions can only
    /// be placed relative to a camera's world rect with this dump.
    /// </summary>
    internal static class RoomDump
    {
        // Tile byte encoding, low bits first.
        private const int TerrainMask = 0x07;
        private const int VerticalBeam = 1 << 3;
        private const int HorizontalBeam = 1 << 4;
        private const int Shortcut = 1 << 5;

        public static void Dump(Room room, string dir)
        {
            if (room == null || room.Tiles == null || room.abstractRoom == null) return;

            string region = (room.world != null && !string.IsNullOrEmpty(room.world.name))
                ? room.world.name : "unknown";
            string roomsDir = Path.Combine(dir, "rooms");
            string path = Path.Combine(roomsDir, region + "_" + Safe(room.abstractRoom.name) + ".json");
            if (File.Exists(path)) return;

            int w = room.Tiles.GetLength(0), h = room.Tiles.GetLength(1);
            byte[] tiles = new byte[w * h];
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    Room.Tile t = room.Tiles[x, y];
                    int v = ((int)t.Terrain) & TerrainMask;
                    if (t.verticalBeam) v |= VerticalBeam;
                    if (t.horizontalBeam) v |= HorizontalBeam;
                    if (t.shortCut != 0) v |= Shortcut;
                    tiles[y * w + x] = (byte)v;      // row-major, y-up as the game stores it
                }

            StringBuilder sb = new StringBuilder(w * h * 2 + 2048);
            sb.Append("{\"region\":\"").Append(Esc(region)).Append('"');
            sb.Append(",\"name\":\"").Append(Esc(room.abstractRoom.name)).Append('"');
            sb.Append(",\"index\":").Append(room.abstractRoom.index);
            sb.Append(",\"width\":").Append(w).Append(",\"height\":").Append(h);
            sb.Append(",\"tile_size\":20");     // Rain World tiles are 20 world units square
            sb.Append(",\"terrain_names\":");
            AppendNames(sb, Enum.GetNames(typeof(Room.Tile.TerrainType)));
            sb.Append(",\"tile_bits\":{\"terrain\":\"0-2\",\"vertical_beam\":3,\"horizontal_beam\":4,\"shortcut\":5}");

            sb.Append(",\"camera_positions\":[");
            if (room.cameraPositions != null)
                for (int i = 0; i < room.cameraPositions.Length; i++)
                {
                    if (i > 0) sb.Append(',');
                    sb.Append('[').Append(F(room.cameraPositions[i].x)).Append(',')
                      .Append(F(room.cameraPositions[i].y)).Append(']');
                }
            sb.Append(']');
            // The camera's size in world units.
            sb.Append(",\"camera_size\":[1400,800]");
            // Each shortcut entrance and where it leads. destNode indexes AbstractRoom.connections,
            // so the destination room of each entrance is known offline. Without it a route can
            // only aim for the nearest hole, which after leaving a room is the one just used.
            sb.Append(",\"shortcuts\":[");
            if (room.shortcuts != null)
                for (int i = 0; i < room.shortcuts.Length; i++)
                {
                    ShortcutData sc = room.shortcuts[i];
                    if (i > 0) sb.Append(',');
                    sb.Append("{\"start\":[").Append(sc.StartTile.x).Append(',').Append(sc.StartTile.y)
                      .Append("],\"dest_node\":").Append(sc.destNode)
                      .Append(",\"type\":\"").Append(Esc(sc.shortCutType != null ? sc.shortCutType.ToString() : "?"))
                      .Append("\"}");
                }
            sb.Append(']');

            // Node positions, so an exit node can be located even when its shortcut entrance is
            // not the tile the agent must reach.
            sb.Append(",\"exit_nodes\":[");
            if (room.abstractRoom != null && room.abstractRoom.nodes != null)
                for (int i = 0; i < room.abstractRoom.nodes.Length; i++)
                {
                    if (i > 0) sb.Append(',');
                    sb.Append("\"").Append(Esc(room.abstractRoom.nodes[i].type != null
                        ? room.abstractRoom.nodes[i].type.ToString() : "?")).Append("\"");
                }
            sb.Append(']');

            // The game's own navigation view of the room. Raw terrain only says whether a tile is
            // rock; the AI map says whether a creature can be there and how far it is above a
            // floor. Paths on raw terrain treat open air as walkable and plan routes through the air.
            sb.Append(",\"aimap_acc_names\":");
            AppendNames(sb, Enum.GetNames(typeof(AItile.Accessibility)));

            if (room.aimap != null)
            {
                byte[] acc = new byte[w * h];
                byte[] walk = new byte[w * h];
                byte[] alt = new byte[w * h];
                for (int y = 0; y < h; y++)
                    for (int x = 0; x < w; x++)
                    {
                        AItile ai = room.aimap.getAItile(new RWCustom.IntVector2(x, y));
                        int i = y * w + x;
                        if (ai == null) { acc[i] = (byte)AItile.Accessibility.Solid; continue; }
                        acc[i] = (byte)(int)ai.acc;
                        walk[i] = ai.walkable ? (byte)1 : (byte)0;
                        // Clamped: unreachable tiles carry a sentinel of 100000.
                        int fa = ai.floorAltitude;
                        alt[i] = (byte)(fa < 0 ? 0 : (fa > 255 ? 255 : fa));
                    }
                sb.Append(",\"aimap_acc_b64\":\"").Append(Convert.ToBase64String(acc)).Append('"');
                sb.Append(",\"aimap_walkable_b64\":\"").Append(Convert.ToBase64String(walk)).Append('"');
                sb.Append(",\"aimap_flooralt_b64\":\"").Append(Convert.ToBase64String(alt)).Append('"');
            }

            // The game's own movement graph, filtered to what a slugcat can do. Passable tiles are
            // not enough: a grid search over occupancy routes up a pole onto a ledge two tiles
            // above its tip. The connection list encodes ReachUp, DoubleReachUp, DropToClimb and
            // the rest, so a move the slugcat cannot make is not an edge.
            sb.Append(",\"move_types\":");
            AppendNames(sb, Enum.GetNames(typeof(MovementConnection.MovementType)));

            CreatureTemplate slugTemplate = null;
            try
            {
                if (room.game != null && room.game.Players != null && room.game.Players.Count > 0)
                    slugTemplate = room.game.Players[0].creatureTemplate;
            }
            catch (Exception) { }

            List<byte> edges = new List<byte>(1 << 16);
            int nEdges = 0;
            if (room.aimap != null && slugTemplate != null)
            {
                for (int y = 0; y < h; y++)
                    for (int x = 0; x < w; x++)
                    {
                        AItile ai = room.aimap.getAItile(new RWCustom.IntVector2(x, y));
                        if (ai == null || ai.outgoingPaths == null) continue;
                        for (int i = 0; i < ai.outgoingPaths.Count; i++)
                        {
                            MovementConnection mc = ai.outgoingPaths[i];
                            bool ok;
                            try { ok = room.aimap.IsConnectionAllowedForCreature(mc, slugTemplate); }
                            catch (Exception) { ok = false; }
                            if (!ok) continue;
                            RWCustom.IntVector2 a2 = mc.StartTile, b2 = mc.DestTile;
                            if (a2.x < 0 || a2.y < 0 || b2.x < 0 || b2.y < 0) continue;
                            if (a2.x > 255 || a2.y > 255 || b2.x > 255 || b2.y > 255) continue;
                            edges.Add((byte)a2.x); edges.Add((byte)a2.y);
                            edges.Add((byte)b2.x); edges.Add((byte)b2.y);
                            edges.Add((byte)(int)mc.type);
                            nEdges++;
                        }
                    }
            }
            sb.Append(",\"n_moves\":").Append(nEdges);
            sb.Append(",\"moves_b64\":\"").Append(Convert.ToBase64String(edges.ToArray())).Append('"');

            sb.Append(",\"tiles_b64\":\"").Append(Convert.ToBase64String(tiles)).Append("\"}");

            try
            {
                Directory.CreateDirectory(roomsDir);
                File.WriteAllText(path, sb.ToString(), new UTF8Encoding(false));
                Plugin.Log.LogInfo("room geometry -> " + Path.GetFileName(path) +
                                   " (" + w + "x" + h + " tiles, " +
                                   (room.cameraPositions == null ? 0 : room.cameraPositions.Length) + " cameras)");
            }
            catch (Exception e)
            {
                Plugin.Log.LogWarning("room dump failed: " + e.Message);
            }
        }

        /// <summary>Appends a JSON array of escaped strings.</summary>
        private static void AppendNames(StringBuilder sb, string[] names)
        {
            sb.Append('[');
            for (int i = 0; i < names.Length; i++)
            {
                if (i > 0) sb.Append(',');
                sb.Append('"').Append(Esc(names[i])).Append('"');
            }
            sb.Append(']');
        }

        private static string F(float v) { return v.ToString("0.##", System.Globalization.CultureInfo.InvariantCulture); }
        private static string Esc(string s) { return s == null ? "" : s.Replace("\\", "\\\\").Replace("\"", "\\\""); }

        private static string Safe(string s)
        {
            if (string.IsNullOrEmpty(s)) return "unnamed";
            char[] bad = Path.GetInvalidFileNameChars();
            StringBuilder sb = new StringBuilder(s.Length);
            foreach (char c in s) sb.Append(Array.IndexOf(bad, c) >= 0 ? '_' : c);
            return sb.ToString();
        }
    }
}
