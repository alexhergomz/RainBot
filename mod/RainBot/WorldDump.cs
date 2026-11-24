using System;
using System.IO;
using System.Text;

namespace RainBot
{
    /// <summary>
    /// One-time export of the region's room graph.
    ///
    /// Rain World's map is static and hand-authored, so the graph can be read once. It is dumped
    /// from the live game, not parsed from the world files, because the game has already resolved
    /// gates, shelters and the region's room numbering; parsing the files would be a second
    /// implementation to keep in sync.
    ///
    /// Written once per region, on entry. Connections are stored raw, including the -1 entries
    /// the game uses for unconnected exits. The reader filters them; dropping them here would
    /// hide a room whose exits do not match its connection count.
    /// </summary>
    internal static class WorldDump
    {
        public static void Dump(World world, string dir)
        {
            if (world == null || world.abstractRooms == null) return;
            string name = string.IsNullOrEmpty(world.name) ? "unknown" : world.name;
            string path = Path.Combine(dir, "graph-" + name + ".json");

            StringBuilder sb = new StringBuilder(1 << 16);
            sb.Append("{\"region\":\"").Append(Esc(name)).Append('"');
            sb.Append(",\"first_room_index\":").Append(world.firstRoomIndex);
            sb.Append(",\"rooms\":[");
            for (int i = 0; i < world.abstractRooms.Length; i++)
            {
                AbstractRoom r = world.abstractRooms[i];
                if (i > 0) sb.Append(',');
                if (r == null) { sb.Append("null"); continue; }
                sb.Append("{\"index\":").Append(r.index);
                sb.Append(",\"name\":\"").Append(Esc(r.name)).Append('"');
                sb.Append(",\"shelter\":").Append(r.shelter ? "true" : "false");
                sb.Append(",\"gate\":").Append(r.gate ? "true" : "false");
                sb.Append(",\"connections\":[");
                if (r.connections != null)
                    for (int c = 0; c < r.connections.Length; c++)
                    {
                        if (c > 0) sb.Append(',');
                        sb.Append(r.connections[c]);
                    }
                sb.Append("]}");
            }
            sb.Append("]}");

            try
            {
                Directory.CreateDirectory(dir);
                File.WriteAllText(path, sb.ToString(), new UTF8Encoding(false));
                Plugin.Log.LogMessage("room graph -> " + path + " (" +
                                      world.abstractRooms.Length + " rooms)");
            }
            catch (Exception e)
            {
                Plugin.Log.LogWarning("room graph dump failed: " + e.Message);
            }
        }

        private static string Esc(string s)
        {
            return s == null ? "" : s.Replace("\\", "\\\\").Replace("\"", "\\\"");
        }
    }
}
