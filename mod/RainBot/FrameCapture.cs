using System;
using System.Collections;
using UnityEngine;

namespace RainBot
{
    /// <summary>
    /// The rendered screen, downscaled on the GPU, as the agent's only visual input (protocol v8).
    ///
    /// Once per rendered frame, at end of frame: the back buffer is copied into a render texture
    /// (`ScreenCapture.CaptureScreenshotIntoRenderTexture`), halved by bilinear blits until it is
    /// within 2x of the target (a single blit from 1366x768 to 128x72 would skip most texels and
    /// alias), blitted to the target size, and read back. Only the small texture crosses to the
    /// CPU. The result is kept as a ready-to-send block:
    ///
    ///     int32 frame_tick, uint16 w, uint16 h, uint8 c(=4), w*h*4 bytes RGBD interleaved, row 0 = top
    ///
    /// D (v11): how far each pixel's tile is from the slugcat along its movement graph (the game's
    /// moves, the recorded moves and the human-jump links), scaled per room: 255 at the slugcat,
    /// 1 at the farthest reachable tile, 0 unreachable or another room. The policy sees path costs,
    /// so long routes look costly and dead ends unreachable. Filled by Plugin.FillDistance.
    ///
    /// `frame_tick` is the sim tick the frame was rendered after, so Python knows how stale it is
    /// (up to max_ticks_per_frame ticks, since several sim ticks can run per rendered frame).
    /// Rendering must be on for frames to advance (`render off` freezes the image).
    /// </summary>
    internal sealed class FrameCapture
    {
        public const int HeaderBytes = 4 + 2 + 2 + 1;
        public const int Channels = 4;
        /// <summary>Fills the D plane: (block, offset of pixel 0, W, H), pixel stride 4.</summary>
        public Action<byte[], int, int, int> FillDistance;
        public readonly int W, H;
        public bool Enabled = true;
        public byte[] Block { get; private set; }
        public int BlockLen { get { return HaveFrame ? Block.Length : 0; } }
        public bool HaveFrame { get; private set; }
        public int Captured { get; private set; }

        private RenderTexture _full, _small;
        private Texture2D _read;

        public FrameCapture(int w, int h)
        {
            W = w; H = h;
            Block = new byte[HeaderBytes + w * h * Channels];
            Observation.PutI16(Block, 4, w);
            Observation.PutI16(Block, 6, h);
            Block[8] = Channels;
        }

        public IEnumerator Loop(Func<int> currentTick)
        {
            WaitForEndOfFrame eof = new WaitForEndOfFrame();
            while (true)
            {
                yield return eof;
                if (!Enabled) continue;
                try { Capture(currentTick()); }
                catch (Exception e)
                {
                    Plugin.Log.LogWarning("frame capture failed, disabling: " + e.Message);
                    Enabled = false;
                }
            }
        }

        private void Capture(int tick)
        {
            int sw = Screen.width, sh = Screen.height;
            if (sw <= 0 || sh <= 0) return;
            if (_full == null || _full.width != sw || _full.height != sh)
            {
                if (_full != null) _full.Release();
                _full = new RenderTexture(sw, sh, 0, RenderTextureFormat.ARGB32);
            }
            if (_small == null)
            {
                _small = new RenderTexture(W, H, 0, RenderTextureFormat.ARGB32) { filterMode = FilterMode.Bilinear };
                _read = new Texture2D(W, H, TextureFormat.RGB24, false);
            }
            ScreenCapture.CaptureScreenshotIntoRenderTexture(_full);

            // Halve until within 2x of the target, then one last blit to the exact size.
            RenderTexture src = _full;
            RenderTexture tmp = null;
            int w = sw, h = sh;
            while (w / 2 >= W * 2 && h / 2 >= H * 2)
            {
                w /= 2; h /= 2;
                RenderTexture next = RenderTexture.GetTemporary(w, h, 0, RenderTextureFormat.ARGB32);
                next.filterMode = FilterMode.Bilinear;
                Graphics.Blit(src, next);
                if (tmp != null) RenderTexture.ReleaseTemporary(tmp);
                tmp = next;
                src = next;
            }
            Graphics.Blit(src, _small);
            if (tmp != null) RenderTexture.ReleaseTemporary(tmp);

            RenderTexture prev = RenderTexture.active;
            RenderTexture.active = _small;
            _read.ReadPixels(new Rect(0, 0, W, H), 0, 0, false);
            RenderTexture.active = prev;

            // Rows to top-down. Texture rows run bottom-up, but on APIs whose UV origin is the top
            // (D3D/Vulkan, so DXVK under Proton) CaptureScreenshotIntoRenderTexture already
            // returns the image upside down and the two cancel. Flip only when they do not.
            byte[] raw = _read.GetRawTextureData();
            int stride = W * 3;
            for (int y = 0; y < H; y++)
            {
                int src0 = (Flip ? (H - 1 - y) : y) * stride, dst0 = HeaderBytes + y * W * Channels;
                for (int x = 0; x < W; x++)
                {
                    Block[dst0 + x * 4] = raw[src0 + x * 3];
                    Block[dst0 + x * 4 + 1] = raw[src0 + x * 3 + 1];
                    Block[dst0 + x * 4 + 2] = raw[src0 + x * 3 + 2];
                    Block[dst0 + x * 4 + 3] = 0;
                }
            }
            if (FillDistance != null)
            {
                try { FillDistance(Block, HeaderBytes, W, H); }
                catch (Exception e) { Plugin.Log.LogWarning("distance channel failed, off: " + e.Message); FillDistance = null; }
            }
            Observation.PutI32(Block, 0, tick);
            HaveFrame = true;
            Captured++;
            if (ClipArmed && tick != _ringLastTick) ClipPush(tick);
        }

        private static readonly bool Flip = !SystemInfo.graphicsUVStartsAtTop;

        // ---- native-resolution clip ring (the `clip` command) -----------------------------------
        // While armed, Plugin renders one sim tick per frame (ClipTicksPerFrame) and every new
        // tick's full-resolution back buffer is copied into a ring of render textures on the GPU.
        // A clip is then what a person sees: native resolution, every tick, 40 per game second.
        // `clip dump PATH` reads the ring back once and writes it raw:
        //     int32 n, int32 w, int32 h, then n frames of w*h*3 RGB, row 0 = top, then n int32 ticks.
        public int RingFrames = 400;                  // set by `clip arm N`; 400 = 10 s of game time
        public bool ClipArmed;
        private RenderTexture[] _ring;
        private int[] _ringTick = new int[400];
        private int _ringHead, _ringCount, _ringLastTick = int.MinValue;

        public void ClipArm(int frames)
        {
            frames = Math.Max(40, Math.Min(frames, 1200));
            if (frames != RingFrames) { ReleaseRing(); RingFrames = frames; _ringTick = new int[frames]; }
            ClipArmed = true;
            _ringHead = _ringCount = 0;
            _ringLastTick = int.MinValue;
        }

        /// <summary>Stop capturing. ``release`` frees the ring's VRAM (~4 MB per frame at
        /// 1366x768); a session end keeps it so a hibernation clip can still be dumped.</summary>
        public void ClipStop(bool release = false)
        {
            ClipArmed = false;
            if (release) ReleaseRing();
        }

        /// <summary>Release every texture (the core is being unloaded).</summary>
        public void Dispose()
        {
            ReleaseRing();
            if (_full != null) { _full.Release(); _full = null; }
            if (_small != null) { _small.Release(); _small = null; }
            if (_read != null) { UnityEngine.Object.Destroy(_read); _read = null; }
        }

        private void ReleaseRing()
        {
            if (_ring != null) foreach (RenderTexture r in _ring) if (r != null) r.Release();
            _ring = null;
            _ringCount = 0;
        }

        private void ClipPush(int tick)
        {
            int sw = _full.width, sh = _full.height;
            if (_ring == null || _ring.Length != RingFrames || _ring[0] == null || _ring[0].width != sw || _ring[0].height != sh)
            {
                if (_ring != null) foreach (RenderTexture r in _ring) if (r != null) r.Release();
                _ring = new RenderTexture[RingFrames];
                for (int i = 0; i < RingFrames; i++) _ring[i] = new RenderTexture(sw, sh, 0, RenderTextureFormat.ARGB32);
            }
            Graphics.Blit(_full, _ring[_ringHead]);
            _ringTick[_ringHead] = tick;
            _ringHead = (_ringHead + 1) % RingFrames;
            _ringCount = Math.Min(_ringCount + 1, RingFrames);
            _ringLastTick = tick;
        }

        /// <summary>Write the ring, oldest first, and disarm. Returns frames written.</summary>
        public int ClipDump(string path)
        {
            ClipArmed = false;
            if (_ring == null || _ringCount == 0) return 0;
            int w = _ring[0].width, h = _ring[0].height, n = _ringCount;
            Texture2D tex = new Texture2D(w, h, TextureFormat.RGB24, false);
            int stride = w * 3;
            byte[] frame = new byte[stride * h];
            using (System.IO.FileStream fs = new System.IO.FileStream(path, System.IO.FileMode.Create))
            {
                byte[] hdr = new byte[12];
                Observation.PutI32(hdr, 0, n); Observation.PutI32(hdr, 4, w); Observation.PutI32(hdr, 8, h);
                fs.Write(hdr, 0, 12);
                RenderTexture prev = RenderTexture.active;
                int start = (_ringHead - n + RingFrames) % RingFrames;
                for (int k = 0; k < n; k++)
                {
                    RenderTexture.active = _ring[(start + k) % RingFrames];
                    tex.ReadPixels(new Rect(0, 0, w, h), 0, 0, false);
                    byte[] raw = tex.GetRawTextureData();
                    for (int y = 0; y < h; y++)
                        Buffer.BlockCopy(raw, (Flip ? (h - 1 - y) : y) * stride, frame, y * stride, stride);
                    fs.Write(frame, 0, frame.Length);
                }
                RenderTexture.active = prev;
                byte[] ticks = new byte[4 * n];
                for (int k = 0; k < n; k++) Observation.PutI32(ticks, 4 * k, _ringTick[(start + k) % RingFrames]);
                fs.Write(ticks, 0, ticks.Length);
            }
            UnityEngine.Object.Destroy(tex);
            ReleaseRing();
            return n;
        }
    }
}
