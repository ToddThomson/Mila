"""Packed-layout implementation for the codebook policies, Python side.

The normative statement of this layout is Mila/Src/Dnn/Quantization/Weight/
CodebookPacking.ixx; this module must match it byte for byte, and the generated
fixture header lets the C++ tests prove that it does. artifact.py's emitter builds on
these functions.

Regenerate the fixture after any layout change:
  python packing.py --emit-fixture ../../Tests/Dnn/Quantization/CodebookOracle.Fixture.h
"""

import argparse

import numpy as np

FIXTURE_SEED = 20260816

# FP4 E2M1, matching Src/Dnn/Compute/Devices/Cuda/Operations/Linear/Kernels/
# Quantization/CudaFp4WeightQuantization.cu. Representable magnitudes are
# {0, .5, 1, 1.5, 2, 3, 4, 6}; these are the breakpoints between adjacent ones, and a
# magnitude is the count of breakpoints a value has reached or passed.
FP4_E2M1_MAX = 6.0
FP4_BREAKPOINTS = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=np.float32)


def quantize_fp4_e2m1(values, group_size=128):
    """values [rows, columns] -> (packed uint8 [rows, columns//2],
    scales float32 [rows, columns//group_size]).

    Data-free: the scale is the group absmax over 6, so this needs no calibration and
    is idempotent -- quantizing an already-quantized tensor reproduces its own codes.
    That is what makes it correct to pack these roles offline rather than at load.

    The arithmetic mirrors the kernel deliberately, including multiplying by the
    reciprocal rather than dividing: at a rounding breakpoint the two differ in the last
    bit, and this layout has to agree with the device byte for byte.
    """
    weight = np.ascontiguousarray(values, dtype=np.float32)
    rows, columns = weight.shape

    if columns % group_size != 0:
        raise ValueError(f"columns {columns} is not a multiple of group size {group_size}")

    if columns % 2 != 0:
        raise ValueError(f"columns {columns} must be even to pack two nibbles per byte")

    grouped = weight.reshape(rows, columns // group_size, group_size)
    absmax = np.abs(grouped).max(axis=2)

    # A group of all zeros has no scale to derive; the kernel substitutes 1.0 so the
    # nibbles come out zero rather than NaN.
    scales = np.where(absmax > 0.0, absmax / np.float32(FP4_E2M1_MAX), np.float32(1.0))
    scales = scales.astype(np.float32)

    normalized = grouped * (np.float32(1.0) / scales)[:, :, None]
    normalized = normalized.reshape(rows, columns)

    magnitude = np.searchsorted(FP4_BREAKPOINTS, np.abs(normalized), side="right")
    nibbles = magnitude.astype(np.uint8) | np.where(normalized < 0.0, np.uint8(8), np.uint8(0))

    packed = (nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)).astype(np.uint8)

    return np.ascontiguousarray(packed), np.ascontiguousarray(scales)


def dequantize_fp4_e2m1(packed, scales, columns, group_size=128):
    """Inverse of quantize_fp4_e2m1, for verifying a packed tensor without a device."""
    levels = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)

    rows = packed.shape[0]
    nibbles = np.zeros((rows, columns), dtype=np.uint8)
    nibbles[:, 0::2] = packed & 0x0F
    nibbles[:, 1::2] = packed >> 4

    magnitude = levels[nibbles & 0x07]
    signed = np.where((nibbles & 0x08) != 0, -magnitude, magnitude)

    column_group = np.arange(columns) // group_size

    return signed * scales[:, column_group]


def packed_row_bytes_two_bit(columns):
    return (columns + 3) // 4


def packed_row_bytes_one_bit(columns):
    return (columns + 7) // 8


def pack_two_bit_codes(codes):
    """codes [rows, columns] uint8 in 0..3 -> packed [rows, row_bytes] uint8.

    Code j of a row lives in byte j // 4 at bit offset (j % 4) * 2.
    """
    rows, columns = codes.shape
    packed = np.zeros((rows, packed_row_bytes_two_bit(columns)), dtype=np.uint8)
    for column in range(columns):
        packed[:, column >> 2] |= (codes[:, column] & 0x3) << ((column & 3) * 2)
    return packed


def unpack_two_bit_codes(packed, columns):
    """Inverse of pack_two_bit_codes for a known column count."""
    codes = np.zeros((packed.shape[0], columns), dtype=np.uint8)
    for column in range(columns):
        codes[:, column] = (packed[:, column >> 2] >> ((column & 3) * 2)) & 0x3
    return codes


def unpack_three_bit_codes(plane_two_bits, plane_one_bit, columns):
    """Inverse of pack_three_bit_codes for a known column count."""
    codes = unpack_two_bit_codes(plane_two_bits, columns)
    for column in range(columns):
        codes[:, column] |= ((plane_one_bit[:, column >> 3] >> (column & 7)) & 0x1) << 2
    return codes


def pack_three_bit_codes(codes):
    """codes [rows, columns] uint8 in 0..7 -> (plane_two_bits, plane_one_bit).

    Low two bits pack exactly as pack_two_bit_codes; the high bit packs one bit
    per code, code j in byte j // 8 at bit offset j % 8.
    """
    rows, columns = codes.shape
    plane_two_bits = pack_two_bit_codes(codes)
    plane_one_bit = np.zeros((rows, packed_row_bytes_one_bit(columns)), dtype=np.uint8)
    for column in range(columns):
        plane_one_bit[:, column >> 3] |= ((codes[:, column] >> 2) & 0x1) << (column & 7)
    return plane_two_bits, plane_one_bit


def half_bits(values):
    """FP32 -> IEEE half bit patterns (numpy rounds to nearest even, keeps subnormals)."""
    return values.astype(np.float16).view(np.uint16)


def encode_nearest(normalized, codebook):
    """Nearest codebook entry per element; ties take the lower index (argmin is
    first-occurrence), matching encodeNearestCode in CodebookPacking.ixx."""
    return np.abs(normalized[..., None] - codebook[None, None, :]).argmin(-1).astype(np.uint8)


def quantize_reference(weights, codebook, group_size):
    """Absmax-scale + encode + bit-exact expected dequantization for a tensor.

    Returns (codes, scale_bits, expected) where expected is computed exactly as
    the C++ reference: codebook[code] * float32(float16(scale)).
    """
    rows, columns = weights.shape
    groups = (columns + group_size - 1) // group_size
    pad = groups * group_size - columns
    padded = np.pad(weights, ((0, 0), (0, pad))).reshape(rows, groups, group_size)

    absmax = np.abs(padded).max(-1).astype(np.float32)
    absmax[absmax == 0] = 1e-8
    scale_bits = half_bits(absmax)
    stored_scale = scale_bits.view(np.float16).astype(np.float32)

    normalized = (padded / stored_scale[..., None]).reshape(rows, groups * group_size)
    normalized = normalized[:, :columns].astype(np.float32)
    codes = encode_nearest(normalized, codebook)

    column_group = np.arange(columns) // group_size
    expected = codebook[codes].astype(np.float32) * stored_scale[:, column_group]
    return codes, scale_bits, expected.astype(np.float32)


# ---------------------------------------------------------------------------
# Fixture emission: a generated C header the MilaTests oracle test compiles in,
# proving the two implementations of the layout agree byte for byte.
# ---------------------------------------------------------------------------

def _array_lines(name, ctype, values, fmt, per_line=12):
    lines = [f"inline constexpr {ctype} {name}[] = {{"]
    values = list(values)
    for start in range(0, len(values), per_line):
        chunk = ", ".join(fmt(v) for v in values[start:start + per_line])
        lines.append(f"    {chunk},")
    lines.append("};")
    return lines


def emit_fixture(path):
    rng = np.random.default_rng(FIXTURE_SEED)
    lines = [
        "// Generated by Mila/Tools/Quantization/packing.py --emit-fixture.",
        "// Do not edit; regenerate after any layout change (see packing.py docstring).",
        "#pragma once",
        "#include <cstdint>",
        "",
        "namespace CodebookOracleFixture",
        "{",
    ]

    cases = [
        # (name, entries, group_size, rows, columns) -- tail columns on the 2-bit
        # case exercise partial bytes in every packed structure.
        ("TwoBit", 4, 32, 3, 101),
        ("ThreeBit", 8, 64, 2, 96),
    ]

    for name, entries, group_size, rows, columns in cases:
        codebook = np.sort(rng.uniform(-1.0, 1.0, entries)).astype(np.float32)
        codebook[0], codebook[-1] = np.float32(-1.0), np.float32(1.0)
        weights = rng.normal(0.0, 0.02, (rows, columns)).astype(np.float32)

        codes, scale_bits, expected = quantize_reference(weights, codebook, group_size)
        if entries == 4:
            plane_two, plane_one = pack_two_bit_codes(codes), None
        else:
            plane_two, plane_one = pack_three_bit_codes(codes)

        prefix = f"k{name}"
        body = [
            f"inline constexpr int {prefix}Rows = {rows};",
            f"inline constexpr int {prefix}Columns = {columns};",
            f"inline constexpr int {prefix}GroupSize = {group_size};",
            f"inline constexpr int {prefix}Entries = {entries};",
        ]
        body += _array_lines(f"{prefix}Codes", "std::uint8_t", codes.flatten(),
                             lambda v: f"{v}")
        body += _array_lines(f"{prefix}PackedPlaneTwoBits", "std::uint8_t",
                             plane_two.flatten(), lambda v: f"0x{v:02X}")
        if plane_one is not None:
            body += _array_lines(f"{prefix}PackedPlaneOneBit", "std::uint8_t",
                                 plane_one.flatten(), lambda v: f"0x{v:02X}")
        body += _array_lines(f"{prefix}ScaleBits", "std::uint16_t",
                             scale_bits.flatten(), lambda v: f"0x{v:04X}")
        body += _array_lines(f"{prefix}CodebookBits", "std::uint32_t",
                             codebook.view(np.uint32), lambda v: f"0x{v:08X}")
        body += _array_lines(f"{prefix}ExpectedBits", "std::uint32_t",
                             expected.flatten().view(np.uint32), lambda v: f"0x{v:08X}")
        lines += ["    " + line for line in body]
        lines.append("")

    lines.append("}")

    with open(path, "w", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--emit-fixture", metavar="PATH",
                        help="write the C++ oracle fixture header to PATH")
    args = parser.parse_args()

    if args.emit_fixture:
        emit_fixture(args.emit_fixture)
    else:
        print("Nothing to do: pass --emit-fixture PATH. See --help.")


if __name__ == "__main__":
    main()
