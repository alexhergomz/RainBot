using UnityEngine;

namespace RainBot
{
    /// <summary>
    /// Draws the current sim tick on screen as a binary strip, so a recorded video frame can be
    /// mapped to an exact tick.
    ///
    /// OBS records at 60 fps against a 40 Hz sim: three frames per two ticks, with no fixed
    /// mapping. The observation log's `wall_ms` per tick aligns to within a frame or two, but
    /// capture latency and dropped frames make that an estimate, and errors build up over a long
    /// session. One tick of misalignment corrupts anything learned about physics from video.
    /// The tick in the frame itself makes alignment exact and checkable, for one IMGUI draw per
    /// frame.
    ///
    /// IMGUI, not Futile sprites: IMGUI works in screen pixels, independent of the game's atlas
    /// and camera state, and draws above everything, so no HUD element can hide it.
    ///
    /// Strip layout, left to right, top row then bottom row:
    ///   cell 0  always white   } calibration: locates the strip and fixes the white/black levels
    ///   cell 1  always black   }
    ///   cells 2..25            24 bits of tick, least-significant first, white = 1
    ///
    /// 24 bits wrap at 16.7M ticks (about 116 hours of play), far longer than a session. The
    /// reader resolves any wrap with the log's own tick column.
    /// </summary>
    internal sealed class TickMarker
    {
        public const int DataBits = 24;
        public const int CalibCells = 2;
        public const int TotalCells = CalibCells + DataBits;   // 26
        public const int Columns = 13;                          // 13 x 2 grid

        private Texture2D _white;
        private Texture2D _black;

        public bool Enabled;
        public int CellPx = 12;

        private void EnsureTextures()
        {
            if (_white != null) return;
            _white = new Texture2D(1, 1, TextureFormat.RGB24, false);
            _white.SetPixel(0, 0, Color.white);
            _white.Apply();
            _black = new Texture2D(1, 1, TextureFormat.RGB24, false);
            _black.SetPixel(0, 0, Color.black);
            _black.Apply();
            // Point filtering keeps cell edges hard, so a decoder sampling cell centres sees no
            // interpolation from neighbouring cells.
            _white.filterMode = FilterMode.Point;
            _black.filterMode = FilterMode.Point;
        }

        /// <summary>Call from OnGUI on repaint events only.</summary>
        public void Draw(int tick)
        {
            if (!Enabled) return;
            EnsureTextures();

            int c = CellPx;
            for (int i = 0; i < TotalCells; i++)
            {
                bool on;
                if (i == 0) on = true;              // calibration white
                else if (i == 1) on = false;        // calibration black
                else on = ((tick >> (i - CalibCells)) & 1) != 0;

                int col = i % Columns, row = i / Columns;
                GUI.DrawTexture(new Rect(col * c, row * c, c, c), on ? _white : _black);
            }
        }
    }
}
