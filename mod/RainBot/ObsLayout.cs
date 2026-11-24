using System;
using System.Collections.Generic;
using System.Text;

namespace RainBot
{
    /// <summary>
    /// Declarative observation layout: fields are declared once, offsets are computed, and the
    /// same object emits both the JSON description Python reads and the offsets C# writes to.
    ///
    /// With hand-written offset constants, one stale constant shifts every field after it and
    /// corrupts the data without any error. Computed offsets remove that failure.
    ///
    /// Fields are packed with no alignment padding: numpy structured dtypes accept any offset,
    /// and every C# write here is byte-wise.
    /// </summary>
    internal sealed class ObsLayout
    {
        private struct Fld
        {
            public string Name, Dtype;
            public int Offset, Count;
        }

        private readonly List<Fld> _fields = new List<Fld>();
        private int _cursor;

        public int Size { get { return _cursor; } }

        private static int Width(string dtype)
        {
            switch (dtype)
            {
                case "i1": case "u1": return 1;
                case "i2": case "u2": return 2;
                case "i4": case "u4": case "f4": return 4;
                case "i8": case "u8": case "f8": return 8;
                default: throw new ArgumentException("unsupported dtype " + dtype);
            }
        }

        /// <summary>Declare a field and return its byte offset. Count > 1 declares an array.</summary>
        public int Add(string name, string dtype, int count = 1)
        {
            int off = _cursor;
            _fields.Add(new Fld { Name = name, Dtype = dtype, Offset = off, Count = count });
            _cursor += Width(dtype) * count;
            return off;
        }

        public string FieldsJson()
        {
            StringBuilder sb = new StringBuilder(4096);
            for (int i = 0; i < _fields.Count; i++)
            {
                Fld f = _fields[i];
                if (i > 0) sb.Append(',');
                sb.Append("{\"name\":\"").Append(f.Name).Append("\",\"dtype\":\"").Append(f.Dtype)
                  .Append("\",\"offset\":").Append(f.Offset);
                if (f.Count > 1) sb.Append(",\"shape\":[").Append(f.Count).Append(']');
                sb.Append('}');
            }
            return sb.ToString();
        }
    }
}
