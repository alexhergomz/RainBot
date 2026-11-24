using System;
using System.Globalization;
using System.IO;
using System.Text;

namespace RainBot
{
    /// <summary>
    /// Append-only session logger for human play.
    ///
    /// Writes straight to disk, not through the bridge. Logging over the socket would need a
    /// Python process that answers every tick (lockstep), which a human cannot play with. A
    /// Python crash also cannot lose a recording session this way.
    ///
    /// Two files per session:
    ///   .obs    header, then fixed-stride observation records, so Python can mmap the tail
    ///           and reshape it with no per-record framing to parse.
    ///   .events JSONL, one line per episode boundary (death, shelter, karma, region).
    ///
    /// The schema is written twice: into the .obs header (field layout, fixed for the session)
    /// and into a .schema.json sidecar rewritten periodically and on close. The ExtEnum tables
    /// grow as new creatures appear: the header holds what was known at the first tick, the
    /// sidecar is complete.
    /// </summary>
    internal sealed class Recorder : IDisposable
    {
        private static readonly byte[] Magic = Encoding.ASCII.GetBytes("RBLOG\x02");
        // 10 s of play. A longer interval leaves a session's tail (such as a cycle ending in a
        // shelter) in the buffer while the video and event log already show it.
        private const int FlushEveryTicks = 40 * 10;

        private readonly string _basePath;
        private readonly Func<string> _schemaProvider;
        private FileStream _obs;
        private StreamWriter _events;
        private int _sinceFlush;

        public string BasePath { get { return _basePath; } }
        public long Records { get; private set; }

        public Recorder(string dir, Func<string> schemaProvider)
        {
            _schemaProvider = schemaProvider;
            Directory.CreateDirectory(dir);
            string stamp = DateTime.Now.ToString("yyyyMMdd-HHmmss", CultureInfo.InvariantCulture);
            _basePath = Path.Combine(dir, "session-" + stamp);

            _obs = new FileStream(_basePath + ".obs", FileMode.CreateNew, FileAccess.Write,
                                  FileShare.Read, 1 << 20);
            byte[] schema = Encoding.UTF8.GetBytes(_schemaProvider());
            _obs.Write(Magic, 0, Magic.Length);
            _obs.Write(BitConverter.GetBytes(schema.Length), 0, 4);
            _obs.Write(schema, 0, schema.Length);
            // Flush the header now: a session killed before the first periodic flush would leave
            // a zero-byte file, which reads as corrupt instead of empty.
            _obs.Flush();

            _events = new StreamWriter(_basePath + ".events.jsonl", false, new UTF8Encoding(false));
            _events.AutoFlush = true;
            WriteSchemaSidecar();
        }

        public void Write(byte[] obs, int len)
        {
            _obs.Write(obs, 0, len);
            Records++;
            if (++_sinceFlush >= FlushEveryTicks)
            {
                _sinceFlush = 0;
                _obs.Flush();
                // The ExtEnum tables grow as new creatures appear; an up-to-date sidecar keeps the
                // data readable after a crash.
                WriteSchemaSidecar();
            }
        }

        public void Event(string json)
        {
            _events.WriteLine(json);
            // Episode boundaries matter most and are rare, so flushing on each costs nothing.
            try { _obs.Flush(); _sinceFlush = 0; } catch (Exception) { }
        }

        public void WriteSchemaSidecar()
        {
            try
            {
                File.WriteAllText(_basePath + ".schema.json", _schemaProvider(),
                                  new UTF8Encoding(false));
            }
            catch (Exception e)
            {
                Plugin.Log.LogWarning("could not write schema sidecar: " + e.Message);
            }
        }

        public void Dispose()
        {
            try { if (_obs != null) { _obs.Flush(); _obs.Dispose(); _obs = null; } } catch (Exception) { }
            try { if (_events != null) { _events.Dispose(); _events = null; } } catch (Exception) { }
        }
    }
}
