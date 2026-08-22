"""A safetensors writer that never holds the whole artifact in memory.

`safetensors.numpy.save_file` takes a dict, so writing a 14 GB artifact with it means
14 GB resident before a byte reaches disk -- on a 31.8 GB host that is survivable but
it would put an hour of GPU work behind a swap event. This writer takes the shape of
every tensor first, computes the data region from the declarations, writes the header,
and then seeks to each tensor's offset as its bytes are produced.

The layout is byte-identical to what `save_file` produces, which is the point: the file
this writes must be readable by the same library that wrote the artifacts the reader
gate was proven against. Measured, not assumed -- the data region is ordered by dtype
size descending, then by name, and `test_matches_save_file` holds this writer to that.

Mila's `PretrainedModelReader` rebases from the header's offsets and does not care about
order, so this constraint exists for the Python side alone.
"""

import json
import struct

import numpy as np

# safetensors dtype spellings for the types an artifact carries. int64, float64, uint64
# and bool are deliberately absent: Mila has no wire code for them and rejects them
# loudly at construction, so an id or index tensor must be cast before it reaches here.
DTYPE_NAMES = {
    np.dtype(np.float32): "F32",
    np.dtype(np.float16): "F16",
    np.dtype(np.uint8): "U8",
    np.dtype(np.int32): "I32",
}

DTYPE_SIZES = {name: dtype.itemsize for dtype, name in DTYPE_NAMES.items()}

# BF16 has no numpy dtype. It travels as uint16 and is declared as BF16, which is what
# the converter does too -- the bytes are the same and only the label differs.
BFLOAT16 = "BF16"
DTYPE_SIZES[BFLOAT16] = 2


def dtype_name(array, bfloat16=False):
    if bfloat16:
        if array.dtype != np.uint16:
            raise ValueError(f"a BF16 tensor must arrive as uint16, got {array.dtype}")

        return BFLOAT16

    name = DTYPE_NAMES.get(array.dtype)

    if name is None:
        raise ValueError(f"{array.dtype} has no safetensors spelling Mila can read")

    return name


class StreamingSafetensorsWriter:
    """Declare every tensor, then write each one's bytes when they are produced.

    Declaration order is free: the file's own order is imposed here, so a caller may
    declare and write in whatever order its computation runs in.
    """

    def __init__(self, path, metadata=None):
        self.path = path
        self.metadata = dict(metadata or {})
        self._declared = {}
        self._header = None
        self._written = set()
        self._handle = None

    def declare(self, name, dtype, shape):
        """Record one tensor's dtype (a safetensors spelling) and shape."""
        if name in self._declared:
            raise ValueError(f"{name}: declared twice")

        if dtype not in DTYPE_SIZES:
            raise ValueError(f"{name}: unsupported dtype {dtype}")

        self._declared[name] = (dtype, tuple(int(extent) for extent in shape))

    def payload_bytes(self):
        return sum(self._tensor_bytes(dtype, shape)
                   for dtype, shape in self._declared.values())

    @staticmethod
    def _tensor_bytes(dtype, shape):
        count = 1

        for extent in shape:
            count *= extent

        return count * DTYPE_SIZES[dtype]

    def _build_header(self):
        """Offsets in the library's own order: dtype size descending, then name."""
        ordered = sorted(self._declared.items(),
                         key=lambda item: (-DTYPE_SIZES[item[1][0]], item[0]))

        # __metadata__ leads and the tensors follow in data order, which is the order
        # the library emits and therefore the order byte equality requires.
        header = {}

        if self.metadata:
            header["__metadata__"] = {key: str(value)
                                      for key, value in self.metadata.items()}

        offset = 0

        for name, (dtype, shape) in ordered:
            nbytes = self._tensor_bytes(dtype, shape)
            header[name] = {"dtype": dtype, "shape": list(shape),
                            "data_offsets": [offset, offset + nbytes]}
            offset += nbytes

        return header

    def __enter__(self):
        self._header = self._build_header()

        # The library pads the header to an 8-byte boundary with spaces, which json
        # ignores; matching it keeps the two writers byte-identical.
        encoded = json.dumps(self._header, separators=(",", ":")).encode("utf-8")
        encoded += b" " * ((-len(encoded)) % 8)

        self._handle = open(self.path, "wb")
        self._handle.write(struct.pack("<Q", len(encoded)))
        self._handle.write(encoded)
        self._data_start = 8 + len(encoded)

        return self

    def write(self, name, array, bfloat16=False):
        """Write one declared tensor's bytes at its offset."""
        entry = self._header.get(name)

        if entry is None:
            raise KeyError(f"{name}: written but never declared")

        declared_dtype, declared_shape = self._declared[name]
        actual_dtype = dtype_name(array, bfloat16)

        if actual_dtype != declared_dtype or tuple(array.shape) != declared_shape:
            raise ValueError(
                f"{name}: declared {declared_dtype} {list(declared_shape)}, "
                f"writing {actual_dtype} {list(array.shape)}")

        contiguous = np.ascontiguousarray(array)
        self._handle.seek(self._data_start + entry["data_offsets"][0])
        self._handle.write(contiguous.tobytes())
        self._written.add(name)

    def __exit__(self, exception_type, exception, traceback):
        # A tensor declared and never written leaves a hole the file does not admit
        # to -- the header says it is there and the bytes are whatever the filesystem
        # left. Only raised on a clean exit, so it never masks the real failure.
        missing = sorted(set(self._declared) - self._written)

        if self._handle is not None:
            self._handle.seek(self._data_start + self.payload_bytes())
            self._handle.truncate()
            self._handle.close()
            self._handle = None

        if exception_type is None and missing:
            raise RuntimeError(
                f"{self.path}: declared but never written: {missing[:8]}"
                + (f" and {len(missing) - 8} more" if len(missing) > 8 else ""))

        return False


def read_header(path):
    """(header_dict, data_start) for a safetensors file."""
    with open(path, "rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]

        return json.loads(handle.read(length)), 8 + length


def test_matches_save_file(tmp_directory):
    """Hold this writer to the library it has to interoperate with.

    The comparison is the DATA REGION byte-for-byte plus the header as a parsed
    object, not the whole file as bytes. Measured 2026-08-22: `save_file` does not
    emit `__metadata__` keys in a stable order -- two runs over the same dict
    produced different orders -- because the Rust side carries them in a hash map.
    So a whole-file comparison fails for a reason that means nothing, while the data
    region and the offsets are exactly what a reader depends on.
    """
    from pathlib import Path

    from safetensors.numpy import save_file, load_file

    tmp_directory = Path(tmp_directory)
    rng = np.random.default_rng(0)

    tensors = {
        "zz.weight": rng.integers(0, 255, (5, 7), dtype=np.uint8),
        "aa.weight_scale": rng.random((3, 4), dtype=np.float32).astype(np.float16),
        "mm.weight_codebook": rng.random(8, dtype=np.float32),
        "bb.weight": rng.integers(0, 255, (2, 3), dtype=np.uint8),
    }
    metadata = {"mila_quantization": "codebook", "count": "4"}

    expected_path = tmp_directory / "expected.safetensors"
    save_file(tensors, expected_path, metadata=metadata)

    streamed_path = tmp_directory / "streamed.safetensors"
    writer = StreamingSafetensorsWriter(streamed_path, metadata)

    for name, array in tensors.items():
        writer.declare(name, dtype_name(array), array.shape)

    with writer:
        # Deliberately out of file order: that a caller may write in its own order is
        # the property this writer exists for.
        for name in reversed(list(tensors)):
            writer.write(name, tensors[name])

    expected_header, expected_start = read_header(expected_path)
    streamed_header, streamed_start = read_header(streamed_path)

    if expected_header != streamed_header:
        raise AssertionError(
            f"header differs:\n  save_file {expected_header}\n  streamed  {streamed_header}")

    if expected_path.read_bytes()[expected_start:] != streamed_path.read_bytes()[streamed_start:]:
        raise AssertionError("data region differs from save_file byte-for-byte")

    reloaded = load_file(streamed_path)

    for name, array in tensors.items():
        if not np.array_equal(reloaded[name], array):
            raise AssertionError(f"{name}: reload mismatch")

    return len(tensors)
