using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;

namespace RainBot
{
    /// <summary>
    /// Localhost TCP link to the Python agent. It also implements lockstep.
    ///
    /// No semaphore is needed: the blocking socket read is the block. The Unity main thread
    /// writes an observation and then waits in Receive() until an action comes back, so the sim
    /// cannot advance while Python thinks. There is no cross-thread state, and no release can be
    /// lost or doubled.
    ///
    /// Only the accept loop runs on a background thread, so the game stays playable before
    /// anything connects.
    /// </summary>
    internal sealed class Bridge : IDisposable
    {
        public const byte MsgObs = 1, MsgEvent = 2, MsgLog = 3, MsgChunk = 4;   // mod -> python
        public const byte MsgAction = 1, MsgCmd = 2;                            // python -> mod
        private static readonly byte[] Magic = Encoding.ASCII.GetBytes("RAINBOT\x01");

        private readonly TcpListener _listener;
        private readonly Func<string> _schemaProvider;
        private readonly int _recvTimeoutMs;
        private volatile Socket _client;
        private Thread _acceptThread;
        private volatile bool _running;

        private readonly byte[] _header = new byte[5];
        private readonly byte[] _actionBuf = new byte[Observation.ActionBytes * Observation.DecisionsPerStep];
        private int _connectionId;
        /// <summary>Commands received on the action stream, drained by the plugin each tick.
        /// Plain text lines, not JSON: netstandard2.0 has no built-in JSON parser, and commands
        /// like "kill" and "speed 4" do not need one.</summary>
        private readonly Queue<string> _commands = new Queue<string>();

        /// <summary>
        /// The schema comes from a delegate, not a string, because the ExtEnum name tables it
        /// embeds are empty at plugin-load time (the game registers those types later). Built at
        /// accept time, it gives a connecting client complete tables.
        /// </summary>
        public Bridge(int port, Func<string> schemaProvider, int recvTimeoutMs)
        {
            _schemaProvider = schemaProvider;
            _recvTimeoutMs = recvTimeoutMs;
            _listener = new TcpListener(IPAddress.Loopback, port);
        }

        public bool Connected => _client != null;

        /// <summary>Connected, and the agent has not closed its end. A closed peer is only noticed on
        /// the next read or write, which never comes on a menu screen; this polls for it.</summary>
        public bool Alive()
        {
            Socket s = _client;
            if (s == null) return false;
            try
            {
                if (s.Poll(0, SelectMode.SelectRead) && s.Available == 0)
                {
                    Plugin.Log.LogWarning("agent link lost (closed by the agent)");
                    Disconnect();
                    return false;
                }
            }
            catch (Exception) { Disconnect(); return false; }
            return true;
        }

        /// <summary>Increments on every accepted connection, so the main thread can notice a new
        /// agent (and send it the world table) without touching game state from the accept thread.</summary>
        public int ConnectionId => _connectionId;

        public void Start()
        {
            // A hot reload (mod/Loader) rebinds the same port right after the old core let go of it.
            try { _listener.Server.SetSocketOption(SocketOptionLevel.Socket, SocketOptionName.ReuseAddress, true); } catch (Exception) { }
            _listener.Start();
            _running = true;
            _acceptThread = new Thread(AcceptLoop) { IsBackground = true, Name = "RainBot.Accept" };
            _acceptThread.Start();
        }

        private void AcceptLoop()
        {
            while (_running)
            {
                try
                {
                    Socket s = _listener.AcceptSocket();
                    s.NoDelay = true;                       // a 5-byte action must not wait on Nagle
                    s.ReceiveTimeout = _recvTimeoutMs;
                    s.SendTimeout = _recvTimeoutMs;

                    // Handshake: magic, then the length-prefixed schema that tells Python how to
                    // read every observation that follows.
                    byte[] schema = Encoding.UTF8.GetBytes(_schemaProvider());
                    s.Send(Magic);
                    s.Send(BitConverter.GetBytes(schema.Length));
                    s.Send(schema);

                    lock (_commands) _commands.Clear();
                    Interlocked.Increment(ref _connectionId);
                    Socket old = Interlocked.Exchange(ref _client, s);
                    if (old != null) SafeClose(old);
                    Plugin.Log.LogMessage("agent connected");
                }
                catch (Exception e)
                {
                    if (_running) Plugin.Log.LogWarning("accept failed: " + e.Message);
                    Thread.Sleep(200);
                }
            }
        }

        /// <summary>
        /// Publish a full chunk of `n` (= 8) rows and block until the agent answers with one
        /// MSG_ACTION of `DecisionsPerStep` decisions. Every MSG_CHUNK except a session's final
        /// one (<see cref="SendFinalChunk"/>) is answered by exactly one MSG_ACTION, so the two
        /// streams can never drift out of step.
        ///
        /// Returns false if no agent is connected or the exchange failed. The caller then falls
        /// back to normal play instead of freezing the game. Commands received while waiting are
        /// queued. `ping` only keeps the receive timeout from expiring while the learner is busy,
        /// and is dropped here.
        /// </summary>
        public bool ExchangeChunk(byte[] rows, int n, int rowBytes, ActionPackage[] decisions,
                                  byte[] frame = null, int frameLen = 0)
        {
            Socket s = _client;
            if (s == null) return false;

            try
            {
                SendChunk(s, rows, n, rowBytes, frame, frameLen);

                int want = Observation.ActionBytes * Observation.DecisionsPerStep;
                while (true)
                {
                    ReceiveAll(s, _header, 5);
                    byte type = _header[0];
                    int len = ReadInt32(_header, 1);

                    if (type == MsgAction)
                    {
                        if (len != want)
                            throw new LinkException("action frame is " + len + "B, expected " + want);
                        ReceiveAll(s, _actionBuf, len);
                        for (int i = 0; i < Observation.DecisionsPerStep; i++)
                            decisions[i] = ActionPackage.FromBytes(_actionBuf, i * Observation.ActionBytes);
                        return true;
                    }

                    byte[] scratch = new byte[len];
                    if (len > 0) ReceiveAll(s, scratch, len);
                    if (type == MsgCmd)
                        EnqueueCommand(scratch);
                    else
                        Plugin.Log.LogWarning("ignoring unknown frame type " + type);
                }
            }
            catch (Exception e)
            {
                Plugin.Log.LogWarning("agent link lost (" + e.Message + "), resuming free play");
                Disconnect();
                return false;
            }
        }

        /// <summary>Send the final rows of an ending session as a MSG_CHUNK that is not an exchange:
        /// nothing is waited for or consumed. Python recognises it as the MSG_CHUNK that follows a
        /// `session_end` event.</summary>
        public void SendFinalChunk(byte[] rows, int n, int rowBytes, byte[] frame = null, int frameLen = 0)
        {
            Socket s = _client;
            if (s == null || n <= 0) return;
            try { SendChunk(s, rows, n, rowBytes, frame, frameLen); }
            catch (Exception e)
            {
                Plugin.Log.LogWarning("agent link lost on final chunk (" + e.Message + ")");
                Disconnect();
            }
        }

        /// <summary>Read commands that are already waiting on the socket without blocking, so a
        /// `reset` sent while the game is between exchanges (menus, a reset in progress) does not
        /// wait in the kernel buffer for an exchange that may never come.</summary>
        public void PollCommands()
        {
            Socket s = _client;
            if (s == null) return;
            try
            {
                while (s.Available >= 5)
                {
                    // Peek the header: only consume whole command frames, never an action frame
                    // (which would belong to an exchange) and never a frame not yet fully arrived.
                    int got = s.Receive(_peek, 0, 5, SocketFlags.Peek);
                    if (got < 5) return;
                    byte type = _peek[0];
                    int len = ReadInt32(_peek, 1);
                    if (type != MsgCmd || s.Available < 5 + len) return;
                    ReceiveAll(s, _header, 5);
                    byte[] scratch = new byte[len];
                    if (len > 0) ReceiveAll(s, scratch, len);
                    EnqueueCommand(scratch);
                }
            }
            catch (Exception e)
            {
                Plugin.Log.LogWarning("command poll failed (" + e.Message + ")");
                Disconnect();
            }
        }

        private readonly byte[] _one = new byte[1];
        private readonly byte[] _peek = new byte[5];

        /// <summary>MSG_CHUNK: header, row count, `n` rows, then the latest captured frame if any.</summary>
        private void SendChunk(Socket s, byte[] rows, int n, int rowBytes, byte[] frame, int frameLen)
        {
            _header[0] = MsgChunk;
            WriteInt32(_header, 1, 1 + n * rowBytes + frameLen);
            s.Send(_header, 0, 5, SocketFlags.None);
            _one[0] = (byte)n;
            s.Send(_one, 0, 1, SocketFlags.None);
            if (n > 0) SendAll(s, rows, n * rowBytes);
            if (frameLen > 0) SendAll(s, frame, frameLen);
        }

        private void EnqueueCommand(byte[] payload)
        {
            string cmd = Encoding.UTF8.GetString(payload);
            if (cmd.Trim() != "ping") lock (_commands) _commands.Enqueue(cmd);
        }

        public void SendEvent(string json) { TrySendText(MsgEvent, json); }

        /// <summary>Take one queued command, or null.</summary>
        public string NextCommand()
        {
            lock (_commands) return _commands.Count > 0 ? _commands.Dequeue() : null;
        }

        private void TrySendText(byte type, string text)
        {
            Socket s = _client;
            if (s == null) return;
            try
            {
                byte[] payload = Encoding.UTF8.GetBytes(text);
                _header[0] = type;
                WriteInt32(_header, 1, payload.Length);
                s.Send(_header, 0, 5, SocketFlags.None);
                SendAll(s, payload, payload.Length);
            }
            catch (Exception) { Disconnect(); }
        }

        private void Disconnect()
        {
            Socket old = Interlocked.Exchange(ref _client, null);
            if (old != null) SafeClose(old);
        }

        private static void SendAll(Socket s, byte[] buf, int len)
        {
            int sent = 0;
            while (sent < len)
                sent += s.Send(buf, sent, len - sent, SocketFlags.None);
        }

        private static void ReceiveAll(Socket s, byte[] buf, int len)
        {
            int got = 0;
            while (got < len)
            {
                int k = s.Receive(buf, got, len - got, SocketFlags.None);
                if (k == 0) throw new LinkException("agent closed the connection");
                got += k;
            }
        }

        private static void WriteInt32(byte[] b, int off, int v)
        {
            b[off] = (byte)v; b[off + 1] = (byte)(v >> 8); b[off + 2] = (byte)(v >> 16); b[off + 3] = (byte)(v >> 24);
        }

        private static int ReadInt32(byte[] b, int off)
        {
            return b[off] | (b[off + 1] << 8) | (b[off + 2] << 16) | (b[off + 3] << 24);
        }

        private static void SafeClose(Socket s)
        {
            try { s.Shutdown(SocketShutdown.Both); } catch (Exception) { }
            try { s.Close(); } catch (Exception) { }
        }

        public void Dispose()
        {
            _running = false;
            Disconnect();
            try { _listener.Stop(); } catch (Exception) { }
        }

        /// <summary>A broken or malformed agent link.</summary>
        internal sealed class LinkException : Exception
        {
            public LinkException(string m) : base(m) { }
        }
    }
}
