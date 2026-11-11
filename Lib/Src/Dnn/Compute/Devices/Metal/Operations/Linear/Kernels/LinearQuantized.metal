// output = input * weight^T on quantized weights, on the paths of MetalBackend.md section 4.2: one row of Q4_0 or of six bits
// on MLX's one-row shape, other few rows on own lanes; for more, the weights expanded a block of outputs at a time for
// TensorOps, or for FP8, TensorOps on the bare codes with each output's scale after. Sums are 32 bits. Needs QuantizedCodes.metal.

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;

// A rectangle of one product: `rows` rows from `first_row`, outputs [first_output, last_output). Every output holds
// `row_bytes` bytes of codes and `groups` scales, one for each `group` inputs. Output rows are `output_stride` apart.
struct QuantizedLinearShape
{
    uint inputs;
    uint output_stride;
    uint first_row;
    uint rows;
    uint first_output;
    uint last_output;
    uint row_bytes;
    uint groups;
    uint group;
};

// Eight weights from one load of their codes, the `c`th eight of a row of `inputs`. Every format's group is at least 32, so
// eight never cross a scale.
struct Fp8Codes
{
    static void chunk( device const uchar* row, uint c, uint, thread float4& first, thread float4& second )
    {
        device const uint* words = reinterpret_cast<device const uint*>( row ) + c * 2;
        const uint4 a = uint4( words[0], words[0] >> 8, words[0] >> 16, words[0] >> 24 ) & 255u;
        const uint4 b = uint4( words[1], words[1] >> 8, words[1] >> 16, words[1] >> 24 ) & 255u;

        first = float4( kFp8E4M3Values[a.x], kFp8E4M3Values[a.y], kFp8E4M3Values[a.z], kFp8E4M3Values[a.w] );
        second = float4( kFp8E4M3Values[b.x], kFp8E4M3Values[b.y], kFp8E4M3Values[b.z], kFp8E4M3Values[b.w] );
    }
};

// Two codes to a byte, the low nibble the even input.
struct Fp4Codes
{
    static void chunk( device const uchar* row, uint c, uint, thread float4& first, thread float4& second )
    {
        const uint bits = reinterpret_cast<device const uint*>( row )[c];
        const uint4 a = uint4( bits, bits >> 4, bits >> 8, bits >> 12 ) & 15u;
        const uint4 b = uint4( bits >> 16, bits >> 20, bits >> 24, bits >> 28 ) & 15u;

        first = float4( kFp4E2M1Values[a.x], kFp4E2M1Values[a.y], kFp4E2M1Values[a.z], kFp4E2M1Values[a.w] );
        second = float4( kFp4E2M1Values[b.x], kFp4E2M1Values[b.y], kFp4E2M1Values[b.z], kFp4E2M1Values[b.w] );
    }
};

// Q4_0 (Int4Packing.ixx): Fp4Codes' nibble order, each code a signed step plus 8.
struct Int4Codes
{
    static void chunk( device const uchar* row, uint c, uint, thread float4& first, thread float4& second )
    {
        const uint bits = reinterpret_cast<device const uint*>( row )[c];
        const int4 a = int4( uint4( bits, bits >> 4, bits >> 8, bits >> 12 ) & 15u ) - 8;
        const int4 b = int4( uint4( bits >> 16, bits >> 20, bits >> 24, bits >> 28 ) & 15u ) - 8;

        first = float4( a );
        second = float4( b );
    }
};

// Six bits (Int6Packing.ixx): the low nibbles as Int4Codes', then from byte inputs / 2 each code's high two bits, four codes
// to a byte from bit 0, so eight codes' are one ushort. Each code is a signed step plus 32.
struct Int6Codes
{
    static void chunk( device const uchar* row, uint c, uint inputs, thread float4& first, thread float4& second )
    {
        const uint bits = reinterpret_cast<device const uint*>( row )[c];
        const uint high = reinterpret_cast<device const ushort*>( row + inputs / 2 )[c];
        const uint4 a = ( uint4( bits, bits >> 4, bits >> 8, bits >> 12 ) & 15u ) | ( ( uint4( high, high >> 2, high >> 4, high >> 6 ) & 3u ) << 4 );
        const uint4 b = ( uint4( bits >> 16, bits >> 20, bits >> 24, bits >> 28 ) & 15u )
            | ( ( uint4( high >> 8, high >> 10, high >> 12, high >> 14 ) & 3u ) << 4 );

        first = float4( int4( a ) - 32 );
        second = float4( int4( b ) - 32 );
    }
};

// One SIMD-group per TAcross outputs of up to TRows rows: its lanes walk the codes once, eight weights at a time, for every
// row. The grid is (groups of outputs, groups of rows). One row reads its activations once for TAcross outputs, which won
// at decode; several rows read each output's weights once for all of them, which won at 8 rows.
template<typename TCodes, typename TScale, typename TElement, typename TVector, uint TAcross, uint TRows>
kernel void linear_quantized_lanes(
    device const TVector* input [[buffer(0)]],
    device const uchar* codes [[buffer(1)]],
    device const TScale* scales [[buffer(2)]],
    device TElement* output [[buffer(3)]],
    constant QuantizedLinearShape& shape [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 threads [[threads_per_threadgroup]] )
{
    const uint vectors = shape.inputs / 4;
    const uint first_output = shape.first_output + group.x * TAcross;
    const uint first_row = shape.first_row + group.y * TRows;
    const uint last_row = shape.first_row + shape.rows;

    float sums[TRows][TAcross];

    for ( uint m = 0; m < TRows; ++m )
    {
        for ( uint a = 0; a < TAcross; ++a )
        {
            sums[m][a] = 0.0f;
        }
    }

    for ( uint c = lane; c < shape.inputs / 8; c += threads.x )
    {
        for ( uint a = 0; a < TAcross; ++a )
        {
            // An output past the last reads the last one's codes, and is never written.
            const uint o = min( first_output + a, shape.last_output - 1 );

            float4 first;
            float4 second;
            TCodes::chunk( codes + o * shape.row_bytes, c, shape.inputs, first, second );

            const float scale = float( scales[o * shape.groups + c * 8 / shape.group] );

            for ( uint m = 0; m < TRows; ++m )
            {
                const uint row = min( first_row + m, last_row - 1 );
                device const TVector* x = input + row * vectors + c * 2;

                sums[m][a] += scale * ( dot( float4( x[0] ), first ) + dot( float4( x[1] ), second ) );
            }
        }
    }

    for ( uint m = 0; m < TRows; ++m )
    {
        for ( uint a = 0; a < TAcross; ++a )
        {
            const float total = simd_sum( sums[m][a] );

            if ( lane == 0 && first_row + m < last_row && first_output + a < shape.last_output )
            {
                output[( first_row + m ) * shape.output_stride + first_output + a] = TElement( total );
            }
        }
    }
}

#define MILA_QUANTIZED_LANES( codes, scale, name, element, across, rows ) \
    template [[host_name( "linear_quantized_lanes_" name "_" #element "_" #across "_" #rows )]] \
    kernel void linear_quantized_lanes<codes, scale, element, element##4, across, rows>( \
        device const element##4*, device const uchar*, device const scale*, device element*, constant QuantizedLinearShape&, \
        uint3, uint, uint3 );

// FP8's and FP4's scales are FP32, Q4_0's and six bits' FP16, each as its policy stores it. One row of Q4_0 or of six bits
// takes linear_quantized_row below; two to eight take these lanes.
#define MILA_QUANTIZED_LANES_ALL( codes, scale, name ) \
    MILA_QUANTIZED_LANES( codes, scale, name, half, 4, 1 ) \
    MILA_QUANTIZED_LANES( codes, scale, name, half, 1, 8 ) \
    MILA_QUANTIZED_LANES( codes, scale, name, bfloat, 4, 1 ) \
    MILA_QUANTIZED_LANES( codes, scale, name, bfloat, 1, 8 )

MILA_QUANTIZED_LANES_ALL( Fp8Codes, float, "fp8" )
MILA_QUANTIZED_LANES_ALL( Fp4Codes, float, "fp4" )
MILA_QUANTIZED_LANES( Int6Codes, half, "int6", half, 1, 8 )
MILA_QUANTIZED_LANES( Int6Codes, half, "int6", bfloat, 1, 8 )
MILA_QUANTIZED_LANES( Int4Codes, half, "int4", half, 1, 8 )
MILA_QUANTIZED_LANES( Int4Codes, half, "int4", bfloat, 1, 8 )

// One row of Q4_0 or of six bits, as decode runs it, in the shape of MLX's one-row quantized product (`qmv_fast_impl` in
// mlx/backend/metal/kernels/quantized.h, ml-explore/mlx v0.32.4, MIT): two SIMD-groups a threadgroup, four outputs a
// SIMD-group, and each lane sixteen consecutive inputs of every 512, read once into registers for all four outputs. The
// codes are masked in place and never shifted: the activations are prescaled to match each code's place in its ushort, and
// a code's offset comes off once, through the activations' sum. Sixteen inputs never cross a scale, since both formats'
// group is 32; a row of inputs not a multiple of 512 ends in a pass the lanes past it leave. FP4 was slower in this shape
// than on the lanes above, and so were six bits while their codes were shifted (Notebooks/MetalMatrixPath.md 7.1, 7.2).
template<typename TElement>
static float loadPrescaled( device const TElement* x, thread float* values )
{
    device const vec<TElement, 4>* x4 = reinterpret_cast<device const vec<TElement, 4>*>( x );
    float sum = 0.0f;

    for ( uint i = 0; i < 4; ++i )
    {
        const float4 v = float4( x4[i] );
        sum += v.x + v.y + v.z + v.w;
        values[4 * i] = v.x;
        values[4 * i + 1] = v.y / 16.0f;
        values[4 * i + 2] = v.z / 256.0f;
        values[4 * i + 3] = v.w / 4096.0f;
    }

    return sum;
}

// Sixteen nibbles, the low first, against prescaled activations: the codes' sum before any offset.
static float maskedNibbles( device const ushort* codes, thread const float* values )
{
    float sum = 0.0f;

    for ( uint i = 0; i < 4; ++i )
    {
        const ushort b = codes[i];
        sum += values[4 * i] * float( b & 0x000f ) + values[4 * i + 1] * float( b & 0x00f0 )
            + values[4 * i + 2] * float( b & 0x0f00 ) + values[4 * i + 3] * float( b & 0xf000 );
    }

    return sum;
}

// Sixteen pairs of bits, eight to a ushort from bit 0, against activations prescaled for a pair's place.
static float maskedPairs( device const ushort* pairs, thread const float* values )
{
    float sum = 0.0f;

    for ( uint i = 0; i < 2; ++i )
    {
        const ushort b = pairs[i];
        sum += values[8 * i] * float( b & 0x0003 ) + values[8 * i + 1] * float( b & 0x000c )
            + values[8 * i + 2] * float( b & 0x0030 ) + values[8 * i + 3] * float( b & 0x00c0 )
            + values[8 * i + 4] * float( b & 0x0300 ) + values[8 * i + 5] * float( b & 0x0c00 )
            + values[8 * i + 6] * float( b & 0x3000 ) + values[8 * i + 7] * float( b & 0xc000 );
    }

    return sum;
}

struct Int4Row
{
    static constexpr constant uint kGroup = 32;
    static constexpr constant uint kRegisters = 16;

    template<typename TElement>
    static float load( device const TElement* x, thread float* values )
    {
        return loadPrescaled( x, values );
    }

    static float dot( device const uchar* row, uint first, uint, thread const float* values, float sum )
    {
        return maskedNibbles( reinterpret_cast<device const ushort*>( row + first / 2 ), values ) - 8.0f * sum;
    }
};

// Six bits (Int6Packing.ixx): the low nibbles are Q4_0's, masked against the same sixteen values. A code's high pair weighs 16
// and sits at bit 2j of its ushort, j the input's place in eight, so sixteen more values hold each activation times 16 / 4^j:
// from the value already divided for its nibble, the factors below, every one a power of two.
struct Int6Row
{
    static constexpr constant uint kGroup = 32;
    static constexpr constant uint kRegisters = 32;

    template<typename TElement>
    static float load( device const TElement* x, thread float* values )
    {
        const float sum = loadPrescaled( x, values );

        for ( uint i = 0; i < 2; ++i )
        {
            values[16 + 8 * i] = values[8 * i] * 16.0f;
            values[16 + 8 * i + 1] = values[8 * i + 1] * 64.0f;
            values[16 + 8 * i + 2] = values[8 * i + 2] * 256.0f;
            values[16 + 8 * i + 3] = values[8 * i + 3] * 1024.0f;
            values[16 + 8 * i + 4] = values[8 * i + 4] / 16.0f;
            values[16 + 8 * i + 5] = values[8 * i + 5] / 4.0f;
            values[16 + 8 * i + 6] = values[8 * i + 6];
            values[16 + 8 * i + 7] = values[8 * i + 7] * 4.0f;
        }

        return sum;
    }

    static float dot( device const uchar* row, uint first, uint inputs, thread const float* values, float sum )
    {
        return maskedNibbles( reinterpret_cast<device const ushort*>( row + first / 2 ), values )
            + maskedPairs( reinterpret_cast<device const ushort*>( row + inputs / 2 + first / 4 ), values + 16 ) - 32.0f * sum;
    }
};

template<typename TRow, typename TScale, typename TElement>
kernel void linear_quantized_row(
    device const TElement* input [[buffer(0)]],
    device const uchar* codes [[buffer(1)]],
    device const TScale* scales [[buffer(2)]],
    device TElement* output [[buffer(3)]],
    constant QuantizedLinearShape& shape [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]] )
{
    constexpr uint kOutputs = 4;
    constexpr uint kValues = 16;
    constexpr uint kBlock = kValues * 32;

    const uint first_output = shape.first_output + ( group.x * 2 + simdgroup ) * kOutputs;

    if ( first_output >= shape.last_output )
    {
        return;
    }

    // Outputs past the last read the last one's codes and are never written, so the loop needs no bounds.
    const uint last = min( kOutputs, shape.last_output - first_output ) - 1;
    const uint inputs = shape.inputs;
    device const TElement* x = input + shape.first_row * inputs;

    // Where each output's codes and scales start, found once: the loop below only adds to them, and the group is the format's
    // constant, so a pass divides by nothing it has to read.
    uint code_offset[kOutputs];
    uint scale_offset[kOutputs];
    float results[kOutputs];

    for ( uint a = 0; a < kOutputs; ++a )
    {
        const uint o = first_output + min( a, last );

        code_offset[a] = o * shape.row_bytes;
        scale_offset[a] = o * shape.groups;
        results[a] = 0.0f;
    }

    float values[TRow::kRegisters];

    for ( uint first = lane * kValues; first < inputs; first += kBlock )
    {
        const float sum = TRow::load( x + first, values );
        const uint scale = first / TRow::kGroup;

        for ( uint a = 0; a < kOutputs; ++a )
        {
            results[a] += float( scales[scale_offset[a] + scale] ) * TRow::dot( codes + code_offset[a], first, inputs, values, sum );
        }
    }

    for ( uint a = 0; a < kOutputs; ++a )
    {
        const float total = simd_sum( results[a] );

        if ( lane == 0 && a <= last )
        {
            output[shape.first_row * shape.output_stride + first_output + a] = TElement( total );
        }
    }
}

#define MILA_QUANTIZED_ROW( row, scale, name, element ) \
    template [[host_name( "linear_quantized_row_" name "_" #element )]] \
    kernel void linear_quantized_row<row, scale, element>( \
        device const element*, device const uchar*, device const scale*, device element*, constant QuantizedLinearShape&, \
        uint3, uint, uint );

MILA_QUANTIZED_ROW( Int4Row, half, "int4", half )
MILA_QUANTIZED_ROW( Int4Row, half, "int4", bfloat )
MILA_QUANTIZED_ROW( Int6Row, half, "int6", half )
MILA_QUANTIZED_ROW( Int6Row, half, "int6", bfloat )

// Expand outputs [first_output, last_output) to the compute type, a row of inputs each, for TensorOps to multiply. Eight
// weights to a thread; the grid is every eight inputs of every output in the block.
template<typename TCodes, typename TScale, typename TVector>
kernel void linear_quantized_expand(
    device const uchar* codes [[buffer(0)]],
    device const TScale* scales [[buffer(1)]],
    device TVector* expanded [[buffer(2)]],
    constant QuantizedLinearShape& shape [[buffer(3)]],
    uint index [[thread_position_in_grid]] )
{
    const uint chunks = shape.inputs / 8;
    const uint block_output = index / chunks;
    const uint c = index % chunks;
    const uint o = shape.first_output + block_output;

    float4 first;
    float4 second;
    TCodes::chunk( codes + o * shape.row_bytes, c, shape.inputs, first, second );

    const float scale = float( scales[o * shape.groups + c * 8 / shape.group] );
    device TVector* out = expanded + block_output * ( shape.inputs / 4 ) + c * 2;

    out[0] = TVector( first * scale );
    out[1] = TVector( second * scale );
}

#define MILA_QUANTIZED_EXPAND( codes, scale, name, element ) \
    template [[host_name( "linear_quantized_expand_" name "_" #element )]] \
    kernel void linear_quantized_expand<codes, scale, element##4>( \
        device const uchar*, device const scale*, device element##4*, constant QuantizedLinearShape&, uint );

MILA_QUANTIZED_EXPAND( Fp8Codes, float, "fp8", half )
MILA_QUANTIZED_EXPAND( Fp8Codes, float, "fp8", bfloat )
MILA_QUANTIZED_EXPAND( Fp4Codes, float, "fp4", half )
MILA_QUANTIZED_EXPAND( Fp4Codes, float, "fp4", bfloat )
MILA_QUANTIZED_EXPAND( Int4Codes, half, "int4", half )
MILA_QUANTIZED_EXPAND( Int4Codes, half, "int4", bfloat )
MILA_QUANTIZED_EXPAND( Int6Codes, half, "int6", half )
MILA_QUANTIZED_EXPAND( Int6Codes, half, "int6", bfloat )

// Two- and three-bit codebook codes (CodebookPacking.ixx): code j of a row in byte j / 4 from bit (j % 4) * 2, so eight codes'
// are one ushort; for three bits each code's high bit in a plane of its own, code j in byte j / 8 at bit j % 8, so eight codes'
// are one byte. A code indexes the tensor's FP32 table, held in registers. Each output's row of the plane is inputs / 8 bytes.
template<uint TEntries>
struct CodebookCodes
{
    float table[TEntries];
    device const uchar* high_plane;

    void chunk( device const uchar* row, uint o, uint c, uint inputs, thread float4& first, thread float4& second ) const
    {
        const uint low = reinterpret_cast<device const ushort*>( row )[c];
        uint4 a = uint4( low, low >> 2, low >> 4, low >> 6 ) & 3u;
        uint4 b = uint4( low >> 8, low >> 10, low >> 12, low >> 14 ) & 3u;

        if ( TEntries == 8 )
        {
            const uint high = high_plane[o * ( inputs / 8 ) + c];
            a |= ( uint4( high, high >> 1, high >> 2, high >> 3 ) & 1u ) << 2;
            b |= ( uint4( high >> 4, high >> 5, high >> 6, high >> 7 ) & 1u ) << 2;
        }

        first = float4( table[a.x], table[a.y], table[a.z], table[a.w] );
        second = float4( table[b.x], table[b.y], table[b.z], table[b.w] );
    }
};

template<uint TEntries>
static inline CodebookCodes<TEntries> codebookOf( device const float* codebook, device const uchar* high_plane )
{
    CodebookCodes<TEntries> codes;

    for ( uint e = 0; e < TEntries; ++e )
    {
        codes.table[e] = codebook[e];
    }

    codes.high_plane = high_plane;

    return codes;
}

// linear_quantized_lanes over codebook codes, with the table and the high plane bound after the output.
template<uint TEntries, typename TElement, typename TVector, uint TAcross, uint TRows>
kernel void linear_codebook_lanes(
    device const TVector* input [[buffer(0)]],
    device const uchar* codes [[buffer(1)]],
    device const half* scales [[buffer(2)]],
    device TElement* output [[buffer(3)]],
    device const float* codebook [[buffer(4)]],
    device const uchar* high_plane [[buffer(5)]],
    constant QuantizedLinearShape& shape [[buffer(6)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 threads [[threads_per_threadgroup]] )
{
    const CodebookCodes<TEntries> reader = codebookOf<TEntries>( codebook, high_plane );

    const uint vectors = shape.inputs / 4;
    const uint first_output = shape.first_output + group.x * TAcross;
    const uint first_row = shape.first_row + group.y * TRows;
    const uint last_row = shape.first_row + shape.rows;

    float sums[TRows][TAcross];

    for ( uint m = 0; m < TRows; ++m )
    {
        for ( uint a = 0; a < TAcross; ++a )
        {
            sums[m][a] = 0.0f;
        }
    }

    for ( uint c = lane; c < shape.inputs / 8; c += threads.x )
    {
        for ( uint a = 0; a < TAcross; ++a )
        {
            // An output past the last reads the last one's codes, and is never written.
            const uint o = min( first_output + a, shape.last_output - 1 );

            float4 first;
            float4 second;
            reader.chunk( codes + o * shape.row_bytes, o, c, shape.inputs, first, second );

            const float scale = float( scales[o * shape.groups + c * 8 / shape.group] );

            for ( uint m = 0; m < TRows; ++m )
            {
                const uint row = min( first_row + m, last_row - 1 );
                device const TVector* x = input + row * vectors + c * 2;

                sums[m][a] += scale * ( dot( float4( x[0] ), first ) + dot( float4( x[1] ), second ) );
            }
        }
    }

    for ( uint m = 0; m < TRows; ++m )
    {
        for ( uint a = 0; a < TAcross; ++a )
        {
            const float total = simd_sum( sums[m][a] );

            if ( lane == 0 && first_row + m < last_row && first_output + a < shape.last_output )
            {
                output[( first_row + m ) * shape.output_stride + first_output + a] = TElement( total );
            }
        }
    }
}

#define MILA_CODEBOOK_LANES( entries, name, element, across, rows ) \
    template [[host_name( "linear_codebook_lanes_" name "_" #element "_" #across "_" #rows )]] \
    kernel void linear_codebook_lanes<entries, element, element##4, across, rows>( \
        device const element##4*, device const uchar*, device const half*, device element*, device const float*, \
        device const uchar*, constant QuantizedLinearShape&, uint3, uint, uint3 );

#define MILA_CODEBOOK_LANES_ALL( entries, name ) \
    MILA_CODEBOOK_LANES( entries, name, half, 4, 1 ) \
    MILA_CODEBOOK_LANES( entries, name, half, 1, 8 ) \
    MILA_CODEBOOK_LANES( entries, name, bfloat, 4, 1 ) \
    MILA_CODEBOOK_LANES( entries, name, bfloat, 1, 8 )

MILA_CODEBOOK_LANES_ALL( 4, "codebook2" )
MILA_CODEBOOK_LANES_ALL( 8, "codebook3" )

// linear_quantized_expand over codebook codes.
template<uint TEntries, typename TVector>
kernel void linear_codebook_expand(
    device const uchar* codes [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device TVector* expanded [[buffer(2)]],
    device const float* codebook [[buffer(3)]],
    device const uchar* high_plane [[buffer(4)]],
    constant QuantizedLinearShape& shape [[buffer(5)]],
    uint index [[thread_position_in_grid]] )
{
    const CodebookCodes<TEntries> reader = codebookOf<TEntries>( codebook, high_plane );

    const uint chunks = shape.inputs / 8;
    const uint block_output = index / chunks;
    const uint c = index % chunks;
    const uint o = shape.first_output + block_output;

    float4 first;
    float4 second;
    reader.chunk( codes + o * shape.row_bytes, o, c, shape.inputs, first, second );

    const float scale = float( scales[o * shape.groups + c * 8 / shape.group] );
    device TVector* out = expanded + block_output * ( shape.inputs / 4 ) + c * 2;

    out[0] = TVector( first * scale );
    out[1] = TVector( second * scale );
}

#define MILA_CODEBOOK_EXPAND( entries, name, element ) \
    template [[host_name( "linear_codebook_expand_" name "_" #element )]] \
    kernel void linear_codebook_expand<entries, element##4>( \
        device const uchar*, device const half*, device element##4*, device const float*, device const uchar*, \
        constant QuantizedLinearShape&, uint );

MILA_CODEBOOK_EXPAND( 4, "codebook2", half )
MILA_CODEBOOK_EXPAND( 4, "codebook2", bfloat )
MILA_CODEBOOK_EXPAND( 8, "codebook3", half )
MILA_CODEBOOK_EXPAND( 8, "codebook3", bfloat )

// One tile of activations times FP8's bare codes, into 32-bit sums. Its scale belongs to one output, so it comes out of the
// sum and is applied after (linear_apply_output_scales). FP16 activations only: TensorOps refuses BF16 by FP8 codes.
template<typename TElement>
struct LinearFp8Matrices
{
    tensor<device TElement, dextents<int, 2>> input;
    tensor<device metal_fp8_e4m3_format, dextents<int, 2>> weight;
    tensor<device float, dextents<int, 2>> sums;
};

template<typename TElement, int TTileRows, int TTileColumns, int TSimdgroups>
kernel void linear_fp8_tiles(
    constant LinearFp8Matrices<TElement>& matrices [[buffer(0)]],
    uint2 tile [[threadgroup_position_in_grid]] )
{
    auto input = matrices.input;
    auto weight = matrices.weight;
    auto sums = matrices.sums;

    constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(
        TTileRows, TTileColumns, static_cast<int>( dynamic_extent ), false, true, false );

    mpp::tensor_ops::matmul2d<descriptor, execution_simdgroups<TSimdgroups>> multiply;

    auto a = input.slice( 0, tile.y * TTileRows );
    auto w = weight.slice( 0, tile.x * TTileColumns );
    auto c = sums.slice( tile.x * TTileColumns, tile.y * TTileRows );

    multiply.run( a, w, c );
}

template [[host_name( "linear_fp8_tiles_half_8_16_2" )]] kernel void linear_fp8_tiles<half, 8, 16, 2>(
    constant LinearFp8Matrices<half>&, uint2 );
template [[host_name( "linear_fp8_tiles_half_128_16_4" )]] kernel void linear_fp8_tiles<half, 128, 16, 4>(
    constant LinearFp8Matrices<half>&, uint2 );

// output = sums * the output's scale, over rows of `columns` sums. The grid is every sum.
struct OutputScaleShape
{
    uint columns;
    uint output_stride;
};

template<typename TElement>
kernel void linear_apply_output_scales(
    device const float* sums [[buffer(0)]],
    device const float* scales [[buffer(1)]],
    device TElement* output [[buffer(2)]],
    constant OutputScaleShape& shape [[buffer(3)]],
    uint index [[thread_position_in_grid]] )
{
    const uint row = index / shape.columns;
    const uint column = index % shape.columns;

    output[row * shape.output_stride + column] = TElement( sums[index] * scales[column] );
}

template [[host_name( "linear_apply_output_scales_half" )]] kernel void linear_apply_output_scales<half>(
    device const float*, device const float*, device half*, constant OutputScaleShape&, uint );
template [[host_name( "linear_apply_output_scales_bfloat" )]] kernel void linear_apply_output_scales<bfloat>(
    device const float*, device const float*, device bfloat*, constant OutputScaleShape&, uint );
