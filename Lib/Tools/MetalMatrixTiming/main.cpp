/**
 * @file main.cpp
 * @brief Times Linear's matrix multiply on Metal, by each path and tile, and checks every output against a reference.
 *
 * Raw metal-cpp, so that each arm is the Metal calls alone. MetalMatrixPath.md holds the gate and the results.
 */

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <format>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "WeightFormats.h"

namespace
{
    enum class Precision { Half, Bfloat, Float };

    struct PrecisionFacts
    {
        const char* name;
        const char* label;
        std::size_t bytes;
        MTL::TensorDataType tensor_type;
    };

    const PrecisionFacts& facts( Precision precision )
    {
        static const PrecisionFacts table[] = {
            { "half", "FP16", 2, MTL::TensorDataTypeFloat16 },
            { "bfloat", "BF16", 2, MTL::TensorDataTypeBFloat16 },
            { "float", "FP32", 4, MTL::TensorDataTypeFloat32 } };

        return table[static_cast<int>( precision )];
    }

    Precision parsePrecision( const std::string& text )
    {
        if ( text == "FP16" ) { return Precision::Half; }
        if ( text == "BF16" ) { return Precision::Bfloat; }
        if ( text == "FP32" ) { return Precision::Float; }

        throw std::invalid_argument( "unknown precision " + text );
    }

    float toFloat( const void* base, Precision precision, std::size_t index )
    {
        switch ( precision )
        {
            case Precision::Half: return static_cast<const __fp16*>( base )[index];
            case Precision::Bfloat: return static_cast<const __bf16*>( base )[index];
            default: return static_cast<const float*>( base )[index];
        }
    }

    struct Shape
    {
        std::uint32_t rows;
        std::uint32_t inputs;
        std::uint32_t outputs;
    };

    /**
     * @brief One way of running the multiply, from an argument such as "tile:FP16,spill=256,k=512".
     *
     * Families: `tensor` is Metal's matmul2d, `tile` is SIMD-group 8 x 8 tiles, `dot` is one thread for each output element,
     * `lanes` reads each row of weights once for all rows of activations, `expand` is TensorOps on weights expanded a block at a time,
     * and `direct` is TensorOps on FP8's bare codes with the scales applied after. `format=` names the weights.
     */
    struct Arm
    {
        std::string label;
        std::string family;
        Precision input{ Precision::Half };
        Precision output{ Precision::Half };

        // tensor
        int tile_rows{ 64 };
        int tile_columns{ 32 };
        int simdgroups{ 4 };
        bool weights_row_per_input{ false };

        // tile
        int patch_rows{ 1 };
        int patch_columns{ 1 };
        int spill{ 0 };
        std::uint32_t k_block{ 0 };
        std::uint32_t m_block{ 0 };
        std::uint32_t n_block{ 0 };
        bool swap_grid{ false };

        // quantized weights
        WeightFormats::Format format;
        std::uint32_t expand_block{ 0 };
        bool chunks{ false };
        int repeat{ 1 };
        int across{ 0 };
        bool use_table{ false };

        // dot
        int dot_form{ 1 };
        int lane_rows{ 1 };

        // The result of the arm.
        std::vector<double> times;
        double error{ -1.0 };
    };

    Arm parseArm( const std::string& text )
    {
        Arm arm;
        arm.label = text;

        const std::size_t colon = text.find( ':' );
        arm.family = text.substr( 0, colon );

        std::size_t begin = colon + 1;
        bool first = true;

        while ( begin <= text.size() )
        {
            const std::size_t end = std::min( text.find( ',', begin ), text.size() );
            const std::string part = text.substr( begin, end - begin );
            begin = end + 1;

            if ( first )
            {
                first = false;

                const std::size_t slash = part.find( '/' );
                arm.input = parsePrecision( part.substr( 0, slash ) );
                arm.output = slash == std::string::npos ? arm.input : parsePrecision( part.substr( slash + 1 ) );

                continue;
            }

            const std::size_t equals = part.find( '=' );
            const std::string key = part.substr( 0, equals );

            if ( key == "format" )
            {
                arm.format = WeightFormats::parseFormat( part.substr( equals + 1 ) );

                continue;
            }

            const int value = equals == std::string::npos ? 1 : std::stoi( part.substr( equals + 1 ) );

            if ( key == "rows" ) { arm.tile_rows = value; }
            else if ( key == "columns" ) { arm.tile_columns = value; }
            else if ( key == "groups" ) { arm.simdgroups = value; }
            else if ( key == "byinput" ) { arm.weights_row_per_input = value != 0; }
            else if ( key == "p" ) { arm.patch_rows = value; }
            else if ( key == "q" ) { arm.patch_columns = value; }
            else if ( key == "spill" ) { arm.spill = value; }
            else if ( key == "k" ) { arm.k_block = static_cast<std::uint32_t>( value ); }
            else if ( key == "m" ) { arm.m_block = static_cast<std::uint32_t>( value ); }
            else if ( key == "n" ) { arm.n_block = static_cast<std::uint32_t>( value ); }
            else if ( key == "swap" ) { arm.swap_grid = value != 0; }
            else if ( key == "form" ) { arm.dot_form = value; }
            else if ( key == "lanes" ) { arm.lane_rows = value; }
            else if ( key == "block" ) { arm.expand_block = static_cast<std::uint32_t>( value ); }
            else if ( key == "chunks" ) { arm.chunks = value != 0; }
            else if ( key == "repeat" ) { arm.repeat = value; }
            else if ( key == "across" ) { arm.across = value; }
            else if ( key == "table" ) { arm.use_table = value != 0; }
            else { throw std::invalid_argument( "unknown key " + key ); }
        }

        return arm;
    }

    // Own kernels. The weights are row-major, N outputs by K inputs, so a tile of them is loaded transposed.
    const char* const kOwnSource = R"metal(
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

struct Arguments
{
    uint inputs;
    uint outputs;
    uint first_row;
    uint first_column;
    uint first_input;
    uint last_input;
};

#if GRID_SWAPPED
#define GRID_ROW tg.x
#define GRID_COLUMN tg.y
#else
#define GRID_ROW tg.y
#define GRID_COLUMN tg.x
#endif

// PATCH_ROWS x PATCH_COLUMNS tiles at once, so that each step of K loads fewer matrices than it multiplies.
// SPILL moves the sums into 32 bits every so many inputs, through threadgroup memory. BLOCKED adds a block of K into a
// 32-bit sum in device memory, so that blocks can run as separate dispatches. With neither, the sums stay in TYPE.
kernel void tile(
    device const TYPE* A [[buffer(0)]], device const TYPE* W [[buffer(1)]],
#if BLOCKED
    device float* sums [[buffer(2)]],
#else
    device TYPE* C [[buffer(2)]],
#endif
    constant Arguments& arguments [[buffer(3)]],
#if SPILL || BLOCKED
    threadgroup TYPE* moved [[threadgroup(0)]],
#endif
#if SPILL
    threadgroup float* total [[threadgroup(1)]],
#endif
    uint3 tg [[threadgroup_position_in_grid]], uint thread_index [[thread_index_in_threadgroup]],
    uint3 threads [[threads_per_threadgroup]] )
{
    const uint row = arguments.first_row + GRID_ROW * 8 * PATCH_ROWS;
    const uint column = arguments.first_column + GRID_COLUMN * 8 * PATCH_COLUMNS;
    const uint each = 64 * PATCH_ROWS * PATCH_COLUMNS / threads.x;

    MATRIX a[PATCH_ROWS];
    MATRIX w[PATCH_COLUMNS];
    MATRIX result[PATCH_ROWS][PATCH_COLUMNS];

    for ( uint p = 0; p < PATCH_ROWS; ++p )
    {
        for ( uint q = 0; q < PATCH_COLUMNS; ++q )
        {
            result[p][q] = make_filled_simdgroup_matrix<TYPE, 8, 8>( TYPE( 0 ) );
        }
    }

#if SPILL
    for ( uint i = thread_index * each; i < ( thread_index + 1 ) * each; ++i )
    {
        total[i] = 0.0f;
    }
#endif

    for ( uint k = arguments.first_input; k < arguments.last_input; k += 8 )
    {
        for ( uint p = 0; p < PATCH_ROWS; ++p )
        {
            simdgroup_load( a[p], A, arguments.inputs, ulong2( k, row + p * 8 ) );
        }

        for ( uint q = 0; q < PATCH_COLUMNS; ++q )
        {
            simdgroup_load( w[q], W, arguments.inputs, ulong2( k, column + q * 8 ), true );
        }

        for ( uint p = 0; p < PATCH_ROWS; ++p )
        {
            for ( uint q = 0; q < PATCH_COLUMNS; ++q )
            {
                simdgroup_multiply_accumulate( result[p][q], a[p], w[q], result[p][q] );
            }
        }

#if SPILL
        if ( ( k + 8 ) % SPILL == 0 || k + 8 == arguments.last_input )
        {
            for ( uint p = 0; p < PATCH_ROWS; ++p )
            {
                for ( uint q = 0; q < PATCH_COLUMNS; ++q )
                {
                    simdgroup_store( result[p][q], moved + ( p * PATCH_COLUMNS + q ) * 64 );
                    result[p][q] = make_filled_simdgroup_matrix<TYPE, 8, 8>( TYPE( 0 ) );
                }
            }

            threadgroup_barrier( mem_flags::mem_threadgroup );

            for ( uint i = thread_index * each; i < ( thread_index + 1 ) * each; ++i )
            {
                total[i] += float( moved[i] );
            }

            threadgroup_barrier( mem_flags::mem_threadgroup );
        }
#endif
    }

#if SPILL || BLOCKED
#if BLOCKED
    for ( uint p = 0; p < PATCH_ROWS; ++p )
    {
        for ( uint q = 0; q < PATCH_COLUMNS; ++q )
        {
            simdgroup_store( result[p][q], moved + ( p * PATCH_COLUMNS + q ) * 64 );
        }
    }

    threadgroup_barrier( mem_flags::mem_threadgroup );
#endif

    for ( uint i = thread_index * each; i < ( thread_index + 1 ) * each; ++i )
    {
        const uint tile_index = i / 64;
        const uint inside = i % 64;
        const uint element = ( row + ( tile_index / PATCH_COLUMNS ) * 8 + inside / 8 ) * arguments.outputs
            + column + ( tile_index % PATCH_COLUMNS ) * 8 + inside % 8;
#if BLOCKED
        sums[element] = ( arguments.first_input == 0 ? 0.0f : sums[element] ) + float( moved[i] );
#else
        C[element] = TYPE( total[i] );
#endif
    }
#else
    for ( uint p = 0; p < PATCH_ROWS; ++p )
    {
        for ( uint q = 0; q < PATCH_COLUMNS; ++q )
        {
            simdgroup_store( result[p][q], C, arguments.outputs, ulong2( column + q * 8, row + p * 8 ) );
        }
    }
#endif
}

kernel void narrow( device const float* sums [[buffer(0)]], device TYPE* C [[buffer(1)]], uint i [[thread_position_in_grid]] )
{
    C[i] = TYPE( sums[i] );
}

// One output element per thread: the row of weights is contiguous, and the sum is 32 bits in a register.
kernel void dotScalar(
    device const TYPE* A [[buffer(0)]], device const TYPE* W [[buffer(1)]], device TYPE* C [[buffer(2)]],
    constant Arguments& arguments [[buffer(3)]], uint2 position [[thread_position_in_grid]] )
{
    device const TYPE* a = A + position.y * arguments.inputs;
    device const TYPE* w = W + position.x * arguments.inputs;
    float sum = 0.0f;

    for ( uint k = 0; k < arguments.inputs; ++k )
    {
        sum += float( a[k] ) * float( w[k] );
    }

    C[position.y * arguments.outputs + position.x] = TYPE( sum );
}

// As above, four inputs at a time, multiplied in the storage type.
kernel void dotVector(
    device const VECTOR* A [[buffer(0)]], device const VECTOR* W [[buffer(1)]], device TYPE* C [[buffer(2)]],
    constant Arguments& arguments [[buffer(3)]], uint2 position [[thread_position_in_grid]] )
{
    device const VECTOR* a = A + position.y * ( arguments.inputs / 4 );
    device const VECTOR* w = W + position.x * ( arguments.inputs / 4 );
    float sum = 0.0f;

    for ( uint k = 0; k < arguments.inputs / 4; ++k )
    {
#if FLOATING_DOT
        sum += float( dot( a[k], w[k] ) );
#else
        sum += dot( float4( a[k] ), float4( w[k] ) );
#endif
    }

    C[position.y * arguments.outputs + position.x] = TYPE( sum );
}

// As above, four inputs at a time, multiplied in 32 bits.
kernel void dotWide(
    device const VECTOR* A [[buffer(0)]], device const VECTOR* W [[buffer(1)]], device TYPE* C [[buffer(2)]],
    constant Arguments& arguments [[buffer(3)]], uint2 position [[thread_position_in_grid]] )
{
    device const VECTOR* a = A + position.y * ( arguments.inputs / 4 );
    device const VECTOR* w = W + position.x * ( arguments.inputs / 4 );
    float sum = 0.0f;

    for ( uint k = 0; k < arguments.inputs / 4; ++k )
    {
        sum += dot( float4( a[k] ), float4( w[k] ) );
    }

    C[position.y * arguments.outputs + position.x] = TYPE( sum );
}

// One SIMD-group per LANE_ROWS outputs: its lanes walk the same rows of weights side by side, in 32 bits, and are summed at the end.
kernel void dotLanes(
    device const VECTOR* A [[buffer(0)]], device const VECTOR* W [[buffer(1)]], device TYPE* C [[buffer(2)]],
    constant Arguments& arguments [[buffer(3)]], uint3 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint3 threads [[threads_per_threadgroup]] )
{
    const uint vectors = arguments.inputs / 4;
    const uint column = tg.x * LANE_ROWS;
    device const VECTOR* a = A + tg.y * vectors;
    device const VECTOR* w = W + column * vectors;
    float sums[LANE_ROWS];

    for ( uint r = 0; r < LANE_ROWS; ++r )
    {
        sums[r] = 0.0f;
    }

    for ( uint k = lane; k < vectors; k += threads.x )
    {
        const float4 x = float4( a[k] );

        for ( uint r = 0; r < LANE_ROWS; ++r )
        {
            sums[r] += dot( x, float4( w[r * vectors + k] ) );
        }
    }

    for ( uint r = 0; r < LANE_ROWS; ++r )
    {
        const float total = simd_sum( sums[r] );

        if ( lane == 0 )
        {
            C[tg.y * arguments.outputs + column + r] = TYPE( total );
        }
    }
}
)metal";

    // Every format's reading of one weight. Only the chosen format's lines are compiled.
    const char* const kQuantizedSource = R"metal(
#include <metal_stdlib>
using namespace metal;

struct QuantizedArguments
{
    uint inputs;
    uint outputs;
    uint rows;
    uint row_bytes;
    uint high_row_bytes;
    uint groups;
    uint group;
    uint first_output;
};

static inline float weightOf( device const uchar* codes, device const uchar* high, constant float* table, uint j, uint inputs )
{
#if KIND_FP8
    return table[codes[j]];
#elif KIND_FP4
    return table[( codes[j >> 1] >> ( ( j & 1 ) * 4 ) ) & 15];
#elif KIND_INT4
    return float( ( codes[j >> 1] >> ( ( j & 1 ) * 4 ) ) & 15 ) - 8.0f;
#elif KIND_INT6
    const uint low = ( codes[j >> 1] >> ( ( j & 1 ) * 4 ) ) & 15;
    const uint top = ( codes[inputs / 2 + ( j >> 2 )] >> ( ( j & 3 ) * 2 ) ) & 3;
    return float( low | ( top << 4 ) ) - 32.0f;
#elif KIND_CODEBOOK2
    return table[( codes[j >> 2] >> ( ( j & 3 ) * 2 ) ) & 3];
#elif KIND_CODEBOOK3
    const uint low = ( codes[j >> 2] >> ( ( j & 3 ) * 2 ) ) & 3;
    const uint top = ( high[j >> 3] >> ( j & 7 ) ) & 1;
    return table[low | ( top << 2 )];
#else
    return 0.0f;
#endif
}

// Eight weights at a time, from one load of their codes. Every group of every format is at least 32, so eight never cross a scale.
static inline void chunkOf(
    device const uchar* codes, device const uchar* high, constant float* table, uint c, uint inputs, thread float4& first, thread float4& second )
{
#if KIND_FP8
    device const uint* words = reinterpret_cast<device const uint*>( codes ) + c * 2;
    const uint w0 = words[0];
    const uint w1 = words[1];
    const uint4 a = uint4( w0, w0 >> 8, w0 >> 16, w0 >> 24 ) & 255;
    const uint4 b = uint4( w1, w1 >> 8, w1 >> 16, w1 >> 24 ) & 255;
#if USE_TABLE
    first = float4( table[a.x], table[a.y], table[a.z], table[a.w] );
    second = float4( table[b.x], table[b.y], table[b.z], table[b.w] );
#else
    const auto expand = []( uint4 code )
    {
        const uint4 exponent = ( code >> 3 ) & 15;
        const float4 mantissa = float4( code & 7 );
        const float4 normal = ( 1.0f + mantissa * 0.125f ) * as_type<float4>( ( exponent + 120 ) << 23 );
        const float4 magnitude = select( normal, mantissa * 0.001953125f, exponent == 0 );
        return select( magnitude, -magnitude, ( code & 128 ) != 0 );
    };
    first = expand( a );
    second = expand( b );
#endif
#elif KIND_FP4 || KIND_INT4 || KIND_INT6
    const uint bits = reinterpret_cast<device const uint*>( codes )[c];
    const uint4 a = uint4( bits, bits >> 4, bits >> 8, bits >> 12 ) & 15;
    const uint4 b = uint4( bits >> 16, bits >> 20, bits >> 24, bits >> 28 ) & 15;
#if KIND_FP4
#if USE_TABLE
    first = float4( table[a.x], table[a.y], table[a.z], table[a.w] );
    second = float4( table[b.x], table[b.y], table[b.z], table[b.w] );
#else
    const auto expand = []( uint4 code )
    {
        const uint4 index = code & 7;
        const float4 upper = ( 1.0f + 0.5f * float4( index & 1 ) ) * 0.5f * float4( uint4( 1 ) << ( index >> 1 ) );
        const float4 magnitude = select( upper, float4( index ) * 0.5f, index < 2 );
        return select( magnitude, -magnitude, ( code & 8 ) != 0 );
    };
    first = expand( a );
    second = expand( b );
#endif
#elif KIND_INT4
    first = float4( a ) - 8.0f;
    second = float4( b ) - 8.0f;
#else
    const uint top = reinterpret_cast<device const ushort*>( codes + inputs / 2 )[c];
    first = float4( a | ( ( uint4( top, top >> 2, top >> 4, top >> 6 ) & 3 ) << 4 ) ) - 32.0f;
    second = float4( b | ( ( uint4( top >> 8, top >> 10, top >> 12, top >> 14 ) & 3 ) << 4 ) ) - 32.0f;
#endif
#elif KIND_CODEBOOK2 || KIND_CODEBOOK3
    const uint bits = reinterpret_cast<device const ushort*>( codes )[c];
    uint4 a = uint4( bits, bits >> 2, bits >> 4, bits >> 6 ) & 3;
    uint4 b = uint4( bits >> 8, bits >> 10, bits >> 12, bits >> 14 ) & 3;
#if KIND_CODEBOOK3
    const uint top = high[c];
    a |= ( uint4( top, top >> 1, top >> 2, top >> 3 ) & 1 ) << 2;
    b |= ( uint4( top >> 4, top >> 5, top >> 6, top >> 7 ) & 1 ) << 2;
#endif
    first = float4( table[a.x], table[a.y], table[a.z], table[a.w] );
    second = float4( table[b.x], table[b.y], table[b.z], table[b.w] );
#endif
}

// As quantizedLanes, eight weights at a time. The sum is in 32 bits, and the activations are read four at a time.
kernel void quantizedChunks(
    device const TYPE4* A [[buffer(0)]], device const uchar* codes [[buffer(1)]], device TYPE* C [[buffer(2)]],
    constant QuantizedArguments& arguments [[buffer(3)]], device const SCALE* scales [[buffer(4)]],
    constant float* table [[buffer(5)]], device const uchar* high [[buffer(6)]],
    uint3 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint3 threads [[threads_per_threadgroup]] )
{
    const uint output = tg.x;
    const uint vectors = arguments.inputs / 4;
    device const uchar* row = codes + output * arguments.row_bytes;
    device const uchar* row_high = high + output * arguments.high_row_bytes;
    device const SCALE* row_scales = scales + output * arguments.groups;
    float sums[ROWS];

    for ( uint m = 0; m < ROWS; ++m )
    {
        sums[m] = 0.0f;
    }

    for ( uint c = lane; c < arguments.inputs / 8; c += threads.x )
    {
        float4 first, second;
        chunkOf( row, row_high, table, c, arguments.inputs, first, second );

        const float scale = float( row_scales[c * 8 / arguments.group] );
        first *= scale;
        second *= scale;

        for ( uint m = 0; m < ROWS; ++m )
        {
            device const TYPE4* x = A + m * vectors + c * 2;
            sums[m] += dot( float4( x[0] ), first ) + dot( float4( x[1] ), second );
        }
    }

    for ( uint m = 0; m < ROWS; ++m )
    {
        const float total = simd_sum( sums[m] );

        if ( lane == 0 )
        {
            C[m * arguments.outputs + output] = TYPE( total );
        }
    }
}

// M = 1, several outputs to a SIMD-group: the activations are read once for all of them.
kernel void quantizedChunksAcross(
    device const TYPE4* A [[buffer(0)]], device const uchar* codes [[buffer(1)]], device TYPE* C [[buffer(2)]],
    constant QuantizedArguments& arguments [[buffer(3)]], device const SCALE* scales [[buffer(4)]],
    constant float* table [[buffer(5)]], device const uchar* high [[buffer(6)]],
    uint3 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint3 threads [[threads_per_threadgroup]] )
{
    const uint output = tg.x * ACROSS;
    float sums[ACROSS];

    for ( uint r = 0; r < ACROSS; ++r )
    {
        sums[r] = 0.0f;
    }

    for ( uint c = lane; c < arguments.inputs / 8; c += threads.x )
    {
        const float4 x0 = float4( A[c * 2] );
        const float4 x1 = float4( A[c * 2 + 1] );

        for ( uint r = 0; r < ACROSS; ++r )
        {
            float4 first, second;
            chunkOf( codes + ( output + r ) * arguments.row_bytes, high + ( output + r ) * arguments.high_row_bytes, table, c, arguments.inputs, first, second );
            sums[r] += float( scales[( output + r ) * arguments.groups + c * 8 / arguments.group] ) * ( dot( x0, first ) + dot( x1, second ) );
        }
    }

    for ( uint r = 0; r < ACROSS; ++r )
    {
        const float total = simd_sum( sums[r] );

        if ( lane == 0 )
        {
            C[output + r] = TYPE( total );
        }
    }
}

// As expandBlock, eight weights to a thread.
kernel void expandChunks(
    device const uchar* codes [[buffer(0)]], device TYPE4* expanded [[buffer(1)]],
    constant QuantizedArguments& arguments [[buffer(3)]], device const SCALE* scales [[buffer(4)]],
    constant float* table [[buffer(5)]], device const uchar* high [[buffer(6)]],
    uint2 position [[thread_position_in_grid]] )
{
    const uint output = arguments.first_output + position.y;
    const uint c = position.x;
    float4 first, second;
    chunkOf( codes + output * arguments.row_bytes, high + output * arguments.high_row_bytes, table, c, arguments.inputs, first, second );

    const float scale = float( scales[output * arguments.groups + c * 8 / arguments.group] );
    device TYPE4* out = expanded + position.y * ( arguments.inputs / 4 ) + c * 2;
    out[0] = TYPE4( first * scale );
    out[1] = TYPE4( second * scale );
}

// One SIMD-group per output: its lanes walk the row of weights once, for all rows of activations, in 32 bits.
kernel void quantizedLanes(
    device const TYPE* A [[buffer(0)]], device const uchar* codes [[buffer(1)]], device TYPE* C [[buffer(2)]],
    constant QuantizedArguments& arguments [[buffer(3)]], device const SCALE* scales [[buffer(4)]],
    constant float* table [[buffer(5)]], device const uchar* high [[buffer(6)]],
    uint3 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint3 threads [[threads_per_threadgroup]] )
{
    const uint output = tg.x;
    device const uchar* row = codes + output * arguments.row_bytes;
    device const uchar* row_high = high + output * arguments.high_row_bytes;
    float sums[ROWS];

    for ( uint m = 0; m < ROWS; ++m )
    {
        sums[m] = 0.0f;
    }

    for ( uint g = 0; g < arguments.groups; ++g )
    {
        const float scale = float( scales[output * arguments.groups + g] );
        const uint first = g * arguments.group;

        for ( uint i = lane; i < arguments.group; i += threads.x )
        {
            const uint j = first + i;
            const float weight = weightOf( row, row_high, table, j, arguments.inputs ) * scale;

            for ( uint m = 0; m < ROWS; ++m )
            {
                sums[m] += float( A[m * arguments.inputs + j] ) * weight;
            }
        }
    }

    for ( uint m = 0; m < ROWS; ++m )
    {
        const float total = simd_sum( sums[m] );

        if ( lane == 0 )
        {
            C[m * arguments.outputs + output] = TYPE( total );
        }
    }
}

// Lanes on unquantized weights, read once for all rows.
kernel void plainLanes(
    device const TYPE* A [[buffer(0)]], device const TYPE* W [[buffer(1)]], device TYPE* C [[buffer(2)]],
    constant QuantizedArguments& arguments [[buffer(3)]],
    uint3 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint3 threads [[threads_per_threadgroup]] )
{
    device const TYPE* row = W + tg.x * arguments.inputs;
    float sums[ROWS];

    for ( uint m = 0; m < ROWS; ++m )
    {
        sums[m] = 0.0f;
    }

    for ( uint j = lane; j < arguments.inputs; j += threads.x )
    {
        const float weight = float( row[j] );

        for ( uint m = 0; m < ROWS; ++m )
        {
            sums[m] += float( A[m * arguments.inputs + j] ) * weight;
        }
    }

    for ( uint m = 0; m < ROWS; ++m )
    {
        const float total = simd_sum( sums[m] );

        if ( lane == 0 )
        {
            C[m * arguments.outputs + tg.x] = TYPE( total );
        }
    }
}

// Expand a block of outputs to a type TensorOps reads, one thread for each weight.
kernel void expandBlock(
    device const uchar* codes [[buffer(0)]], device TYPE* expanded [[buffer(1)]],
    constant QuantizedArguments& arguments [[buffer(3)]], device const SCALE* scales [[buffer(4)]],
    constant float* table [[buffer(5)]], device const uchar* high [[buffer(6)]],
    uint2 position [[thread_position_in_grid]] )
{
    const uint output = arguments.first_output + position.y;
    const uint j = position.x;
    device const uchar* row = codes + output * arguments.row_bytes;
    device const uchar* row_high = high + output * arguments.high_row_bytes;
    const float scale = float( scales[output * arguments.groups + j / arguments.group] );

    expanded[position.y * arguments.inputs + j] = TYPE( weightOf( row, row_high, table, j, arguments.inputs ) * scale );
}

// FP8's scale belongs to one output, so it can be applied after a multiply by the bare codes.
kernel void applyRowScales(
    device const float* sums [[buffer(0)]], device TYPE* C [[buffer(1)]],
    constant QuantizedArguments& arguments [[buffer(3)]], device const float* scales [[buffer(4)]],
    uint2 position [[thread_position_in_grid]] )
{
    const uint element = position.y * arguments.outputs + position.x;

    C[element] = TYPE( sums[element] * scales[position.x] );
}
)metal";

    // Multiply by one expanded block of outputs, or by FP8's bare codes.
    const char* const kBlockTensorSource = R"metal(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp;

struct Arguments
{
    tensor<device INPUT, dextents<int, 2>> A;
    tensor<device WEIGHT, dextents<int, 2>> W;
    tensor<device OUTPUT, dextents<int, 2>> C;
};

kernel void multiplyBlock(
    constant Arguments& arguments [[buffer(0)]], constant uint& first_output [[buffer(1)]], uint2 tg [[threadgroup_position_in_grid]] )
{
    auto A = arguments.A;
    auto W = arguments.W;
    auto C = arguments.C;
    constexpr auto descriptor = tensor_ops::matmul2d_descriptor(
        TILE_ROWS, TILE_COLUMNS, static_cast<int>( dynamic_extent ), false, true, false );
    tensor_ops::matmul2d<descriptor, execution_simdgroups<GROUPS>> operation;
    auto a = A.slice( 0, tg.y * TILE_ROWS );
    auto w = W.slice( 0, tg.x * TILE_COLUMNS );
    auto c = C.slice( first_output + tg.x * TILE_COLUMNS, tg.y * TILE_ROWS );
    operation.run( a, w, c );
}
)metal";

    const char* const kTensorSource = R"metal(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp;

struct Arguments
{
    tensor<device INPUT, dextents<int, 2>> A;
    tensor<device INPUT, dextents<int, 2>> W;
    tensor<device OUTPUT, dextents<int, 2>> C;
};

kernel void multiply( constant Arguments& arguments [[buffer(0)]], uint2 tg [[threadgroup_position_in_grid]] )
{
    auto A = arguments.A;
    auto W = arguments.W;
    auto C = arguments.C;
    constexpr auto descriptor = tensor_ops::matmul2d_descriptor(
        TILE_ROWS, TILE_COLUMNS, static_cast<int>( dynamic_extent ), false, TRANSPOSED, false );
    tensor_ops::matmul2d<descriptor, execution_simdgroups<GROUPS>> operation;
    auto a = A.slice( 0, tg.y * TILE_ROWS );
#if TRANSPOSED
    auto w = W.slice( 0, tg.x * TILE_COLUMNS );
#else
    auto w = W.slice( tg.x * TILE_COLUMNS, 0 );
#endif
    auto c = C.slice( tg.x * TILE_COLUMNS, tg.y * TILE_ROWS );
    operation.run( a, w, c );
}
)metal";

    struct QuantizedArguments
    {
        std::uint32_t inputs;
        std::uint32_t outputs;
        std::uint32_t rows;
        std::uint32_t row_bytes;
        std::uint32_t high_row_bytes;
        std::uint32_t groups;
        std::uint32_t group;
        std::uint32_t first_output;
    };

    struct OwnArguments
    {
        std::uint32_t inputs;
        std::uint32_t outputs;
        std::uint32_t first_row;
        std::uint32_t first_column;
        std::uint32_t first_input;
        std::uint32_t last_input;
    };

    class Runner
    {
    public:

        Runner( Shape shape, const std::string& data )
            : shape_( shape ), data_( data )
        {
            device_ = MTL::CreateSystemDefaultDevice();
            queue_ = device_->newCommandQueue();
        }

        const Shape& shape() const { return shape_; }

        /// Rows the arm really runs: matmul2d's tile has at least 8, so a smaller M is padded with zero rows.
        std::uint32_t rowsOf( const Arm& arm ) const
        {
            const bool tiled = arm.family == "tensor" || arm.family == "expand" || arm.family == "direct";

            return tiled ? std::max<std::uint32_t>( shape_.rows, arm.tile_rows ) : shape_.rows;
        }

        MTL::Buffer* output( Precision precision, std::uint32_t rows )
        {
            MTL::Buffer*& buffer = outputs_[{ precision, rows }];

            if ( !buffer )
            {
                buffer = shared( std::size_t{ rows } * shape_.outputs * facts( precision ).bytes );
            }

            return buffer;
        }

        /// So that an element the arm does not write cannot pass on what an earlier arm left there.
        void clearOutput( const Arm& arm )
        {
            MTL::Buffer* const buffer = output( arm.output, rowsOf( arm ) );
            std::memset( buffer->contents(), 0, buffer->length() );
        }

        /// Run the arm once, returning the GPU's time in milliseconds.
        double run( Arm& arm )
        {
            NS::AutoreleasePool* const pool = NS::AutoreleasePool::alloc()->init();
            MTL::CommandBuffer* const commands = queue_->commandBuffer();
            MTL::ComputeCommandEncoder* const encoder = commands->computeCommandEncoder();

            // Sub-millisecond work, alone, runs on a GPU that has not ramped up, and a decode step is many kernels in a row.
            for ( int repeat = 0; repeat < arm.repeat; ++repeat )
            {
                if ( arm.family == "tensor" ) { encodeTensor( encoder, arm ); }
                else if ( arm.family == "tile" ) { encodeTile( encoder, arm ); }
                else if ( arm.family == "dot" ) { encodeDot( encoder, arm ); }
                else if ( arm.family == "lanes" ) { encodeLanes( encoder, arm ); }
                else if ( arm.family == "expand" ) { encodeExpand( encoder, arm ); }
                else if ( arm.family == "direct" ) { encodeDirect( encoder, arm ); }
                else { throw std::invalid_argument( "unknown family " + arm.family ); }
            }

            encoder->endEncoding();
            commands->commit();
            commands->waitUntilCompleted();

            if ( commands->status() != MTL::CommandBufferStatusCompleted )
            {
                throw std::runtime_error( commands->error()->localizedDescription()->utf8String() );
            }

            const double milliseconds = ( commands->GPUEndTime() - commands->GPUStartTime() ) * 1000.0 / arm.repeat;
            pool->release();

            return milliseconds;
        }

        /// Relative L2 error of 16 sampled rows, against a reference from this arm's own rounded inputs.
        double check( Arm& arm, std::int64_t shift_elements = 0 )
        {
            const bool transposed = !arm.weights_row_per_input;
            const std::uint32_t sampled = std::min( shape_.rows, 16u );
            const std::uint32_t row_step = shape_.rows / sampled;
            const std::size_t count = std::size_t{ rowsOf( arm ) } * shape_.outputs;
            const void* const a = inputBuffer( arm.input, false, true, rowsOf( arm ) )->contents();
            const bool quantized = arm.format.kind != WeightFormats::Kind::None;
            const void* const w = quantized ? nullptr : inputBuffer( arm.input, true, transposed, rowsOf( arm ) )->contents();
            const WeightFormats::Encoded* const encoded = quantized ? &weights( arm ).encoded : nullptr;
            const void* const c = output( arm.output, rowsOf( arm ) )->contents();

            std::vector<double> errors( sampled, 0.0 ), norms( sampled, 0.0 );
            std::vector<std::thread> workers;

            for ( std::uint32_t worker = 0; worker < sampled; ++worker )
            {
                workers.emplace_back( [&, worker]
                {
                    const std::size_t row = std::size_t{ worker } * row_step;
                    std::vector<double> reference( shape_.outputs, 0.0 );

                    for ( std::uint32_t n = 0; n < shape_.outputs; ++n )
                    {
                        double sum = 0.0;

                        for ( std::uint32_t k = 0; k < shape_.inputs; ++k )
                        {
                            const std::size_t weight = transposed
                                ? std::size_t{ n } * shape_.inputs + k : std::size_t{ k } * shape_.outputs + n;
                            sum += static_cast<double>( toFloat( a, arm.input, row * shape_.inputs + k ) )
                                * ( quantized ? WeightFormats::decode( arm.format, *encoded, n, k, shape_.inputs ) : toFloat( w, arm.input, weight ) );
                        }

                        reference[n] = sum;
                    }

                    for ( std::uint32_t n = 0; n < shape_.outputs; ++n )
                    {
                        const std::size_t element = ( row * shape_.outputs + n + static_cast<std::size_t>( shift_elements ) ) % count;
                        const double difference = toFloat( c, arm.output, element ) - reference[n];
                        errors[worker] += difference * difference;
                        norms[worker] += reference[n] * reference[n];
                    }
                } );
            }

            for ( std::thread& worker : workers )
            {
                worker.join();
            }

            double error = 0.0, norm = 0.0;

            for ( std::uint32_t i = 0; i < sampled; ++i )
            {
                error += errors[i];
                norm += norms[i];
            }

            return std::sqrt( error / norm );
        }

    private:

        MTL::Buffer* shared( std::size_t bytes )
        {
            MTL::Buffer* const buffer = device_->newBuffer( bytes, MTL::ResourceStorageModeShared );

            if ( !buffer )
            {
                throw std::runtime_error( std::format( "allocation of {} bytes failed", bytes ) );
            }

            return buffer;
        }

        /// Values from one table, so that a billion of them cost no more than a second, rounded to the format on the host.
        void fill( MTL::Buffer* buffer, Precision precision, std::size_t count, double sigma, std::uint64_t seed )
        {
            static const std::vector<float> normal = []
            {
                std::mt19937_64 generator( 12345 );
                std::normal_distribution<float> distribution( 0.0f, 1.0f );
                std::vector<float> table( std::size_t{ 1 } << 20 );

                for ( float& value : table )
                {
                    value = distribution( generator );
                }

                return table;
            }();

            const bool uniform = data_ == "uniform";
            std::vector<std::thread> workers;
            const std::size_t block = count / 8 + 1;

            for ( int worker = 0; worker < 8; ++worker )
            {
                workers.emplace_back( [=, this]
                {
                    std::uint64_t x = ( seed + worker ) * 2685821657736338717ULL + 88172645463325252ULL;

                    for ( std::size_t i = block * worker; i < std::min( count, block * ( worker + 1 ) ); ++i )
                    {
                        x ^= x << 13;
                        x ^= x >> 7;
                        x ^= x << 17;

                        const float value = uniform
                            ? static_cast<float>( ( x >> 40 ) / 8388608.0 - 1.0 )
                            : static_cast<float>( normal[x >> 44] * sigma );

                        switch ( precision )
                        {
                            case Precision::Half: static_cast<__fp16*>( buffer->contents() )[i] = value; break;
                            case Precision::Bfloat: static_cast<__bf16*>( buffer->contents() )[i] = value; break;
                            default: static_cast<float*>( buffer->contents() )[i] = value; break;
                        }
                    }
                } );
            }

            for ( std::thread& worker : workers )
            {
                worker.join();
            }
        }

        MTL::Buffer* inputBuffer( Precision precision, bool weights, bool row_per_output, std::uint32_t rows )
        {
            MTL::Buffer*& buffer = inputs_[{ precision, weights, row_per_output, weights ? 0 : rows }];

            if ( !buffer )
            {
                const std::size_t count = weights
                    ? std::size_t{ shape_.outputs } * shape_.inputs : std::size_t{ shape_.rows } * shape_.inputs;
                const double sigma = weights
                    ? ( data_ == "small" ? 0.001 : 0.02 ) : ( data_ == "small" ? 0.01 : 1.0 );

                const std::size_t allocated = weights ? count : std::size_t{ rows } * shape_.inputs;
                buffer = shared( allocated * facts( precision ).bytes );
                std::memset( buffer->contents(), 0, buffer->length() );
                fill( buffer, precision, count, sigma, weights ? 23 : 11 );
            }

            return buffer;
        }

        MTL::ComputePipelineState* pipeline( const std::string& key, const std::string& source, const char* function )
        {
            MTL::ComputePipelineState*& state = pipelines_[key];

            if ( state )
            {
                return state;
            }

            NS::Error* error = nullptr;
            MTL::CompileOptions* const options = MTL::CompileOptions::alloc()->init();
            options->setLanguageVersion( MTL::LanguageVersion4_1 );

            MTL::Library* const library = device_->newLibrary(
                NS::String::string( source.c_str(), NS::UTF8StringEncoding ), options, &error );

            if ( !library )
            {
                throw std::runtime_error( std::string( "compile failed: " ) + error->localizedDescription()->utf8String() );
            }

            MTL::Function* const entry = library->newFunction( NS::String::string( function, NS::UTF8StringEncoding ) );
            state = device_->newComputePipelineState( entry, &error );

            if ( !state )
            {
                throw std::runtime_error( std::string( "pipeline failed: " ) + error->localizedDescription()->utf8String() );
            }

            return state;
        }

        std::string ownSource( const Arm& arm ) const
        {
            const std::string type = facts( arm.input ).name;

            return std::format(
                "#define TYPE {0}\n#define MATRIX simdgroup_{0}8x8\n#define VECTOR {0}4\n#define SPILL {1}\n#define GRID_SWAPPED {2}\n"
                "#define PATCH_ROWS {3}\n#define PATCH_COLUMNS {4}\n#define BLOCKED {5}\n#define LANE_ROWS {6}\n#define FLOATING_DOT {7}\n{8}",
                type, arm.spill, arm.swap_grid ? 1 : 0, arm.patch_rows, arm.patch_columns, arm.k_block != 0 ? 1 : 0, arm.lane_rows,
                arm.input == Precision::Bfloat ? 0 : 1, kOwnSource );
        }

        MTL::Tensor* tensorOver( MTL::Buffer* buffer, Precision precision, NS::Integer first, NS::Integer second )
        {
            NS::Integer dimensions[2] = { first, second };
            NS::Integer strides[2] = { 1, first };

            MTL::TensorDescriptor* const descriptor = MTL::TensorDescriptor::alloc()->init();
            descriptor->setDataType( facts( precision ).tensor_type );
            descriptor->setUsage( MTL::TensorUsageCompute );
            descriptor->setDimensions( MTL::TensorExtents::alloc()->init( 2, dimensions ) );
            descriptor->setStrides( MTL::TensorExtents::alloc()->init( 2, strides ) );

            NS::Error* error = nullptr;
            MTL::Tensor* const tensor = buffer->newTensor( descriptor, 0, &error );

            if ( !tensor )
            {
                throw std::runtime_error( std::string( "tensor failed: " ) + error->localizedDescription()->utf8String() );
            }

            return tensor;
        }

        void encodeTensor( MTL::ComputeCommandEncoder* encoder, const Arm& arm )
        {
            const bool transposed = !arm.weights_row_per_input;
            const std::string key = std::format( "tensor {} {} {} {} {} {}", facts( arm.input ).name, facts( arm.output ).name,
                arm.tile_rows, arm.tile_columns, arm.simdgroups, transposed );
            const std::string source = std::format(
                "#define INPUT {}\n#define OUTPUT {}\n#define TILE_ROWS {}\n#define TILE_COLUMNS {}\n#define GROUPS {}\n#define TRANSPOSED {}\n{}",
                facts( arm.input ).name, facts( arm.output ).name, arm.tile_rows, arm.tile_columns, arm.simdgroups,
                transposed ? "true" : "false", kTensorSource );

            MTL::ComputePipelineState* const state = pipeline( key, source, "multiply" );
            const std::uint32_t rows = rowsOf( arm );
            MTL::Buffer* const a = inputBuffer( arm.input, false, true, rows );
            MTL::Buffer* const w = inputBuffer( arm.input, true, transposed, rows );
            MTL::Buffer* const c = output( arm.output, rows );

            const auto make = [&]( MTL::Tensor*& slot, MTL::Buffer* buffer, Precision precision, NS::Integer first, NS::Integer second )
            {
                if ( !slot )
                {
                    slot = tensorOver( buffer, precision, first, second );
                }

                return slot;
            };

            MTL::Tensor* const ta = make( tensors_[{ arm.input, 0, false, rows }], a, arm.input, shape_.inputs, rows );
            MTL::Tensor* const tw = transposed
                ? make( tensors_[{ arm.input, 1, true, 0 }], w, arm.input, shape_.inputs, shape_.outputs )
                : make( tensors_[{ arm.input, 1, false, 0 }], w, arm.input, shape_.outputs, shape_.inputs );
            MTL::Tensor* const tc = make( tensors_[{ arm.output, 2, false, rows }], c, arm.output, shape_.outputs, rows );

            const MTL::ResourceID identifiers[3] = { ta->gpuResourceID(), tw->gpuResourceID(), tc->gpuResourceID() };

            encoder->setComputePipelineState( state );
            encoder->setBytes( identifiers, sizeof( identifiers ), 0 );
            encoder->useResource( ta, MTL::ResourceUsageRead );
            encoder->useResource( tw, MTL::ResourceUsageRead );
            encoder->useResource( tc, MTL::ResourceUsageWrite );
            encoder->useResource( a, MTL::ResourceUsageRead );
            encoder->useResource( w, MTL::ResourceUsageRead );
            encoder->useResource( c, MTL::ResourceUsageWrite );
            encoder->dispatchThreadgroups(
                MTL::Size( shape_.outputs / arm.tile_columns, rows / arm.tile_rows, 1 ),
                MTL::Size( state->threadExecutionWidth() * arm.simdgroups, 1, 1 ) );
        }

        void encodeTile( MTL::ComputeCommandEncoder* encoder, const Arm& arm )
        {
            const std::size_t bytes = facts( arm.input ).bytes;
            const bool block = arm.k_block != 0;

            if ( shape_.rows % ( 8 * arm.patch_rows ) != 0 || shape_.outputs % ( 8 * arm.patch_columns ) != 0 || shape_.inputs % 8 != 0 )
            {
                throw std::invalid_argument( "the shape is not a multiple of the patch: " + arm.label );
            }

            const std::size_t patch = std::size_t( arm.patch_rows ) * arm.patch_columns;
            const std::string key = std::format( "tile {} {} {} {} {} {}", facts( arm.input ).name, arm.spill, arm.swap_grid,
                arm.patch_rows, arm.patch_columns, block );
            MTL::ComputePipelineState* const state = pipeline( key, ownSource( arm ), "tile" );

            MTL::Buffer* const a = inputBuffer( arm.input, false, true, shape_.rows );
            MTL::Buffer* const w = inputBuffer( arm.input, true, true, shape_.rows );
            MTL::Buffer* const c = output( arm.output, shape_.rows );

            if ( block && !sums_ )
            {
                sums_ = shared( std::size_t{ shape_.rows } * shape_.outputs * 4 );
            }

            encoder->setComputePipelineState( state );
            encoder->setBuffer( a, 0, 0 );
            encoder->setBuffer( w, 0, 1 );
            encoder->setBuffer( block ? sums_ : c, 0, 2 );

            if ( block || arm.spill > 0 )
            {
                encoder->setThreadgroupMemoryLength( 64 * patch * bytes, 0 );
            }

            if ( arm.spill > 0 )
            {
                encoder->setThreadgroupMemoryLength( 64 * patch * 4, 1 );
            }

            const std::uint32_t k_block = block ? arm.k_block : shape_.inputs;
            const std::uint32_t m_block = arm.m_block ? arm.m_block : shape_.rows;
            const std::uint32_t n_block = arm.n_block ? arm.n_block : shape_.outputs;
            const NS::UInteger width = state->threadExecutionWidth();

            for ( std::uint32_t k = 0; k < shape_.inputs; k += k_block )
            {
                for ( std::uint32_t m = 0; m < shape_.rows; m += m_block )
                {
                    for ( std::uint32_t n = 0; n < shape_.outputs; n += n_block )
                    {
                        const OwnArguments arguments{ shape_.inputs, shape_.outputs, m, n, k, std::min( k + k_block, shape_.inputs ) };
                        const NS::UInteger columns = std::min( n_block, shape_.outputs - n ) / ( 8 * arm.patch_columns );
                        const NS::UInteger rows = std::min( m_block, shape_.rows - m ) / ( 8 * arm.patch_rows );

                        encoder->setBytes( &arguments, sizeof( arguments ), 3 );
                        encoder->dispatchThreadgroups(
                            arm.swap_grid ? MTL::Size( rows, columns, 1 ) : MTL::Size( columns, rows, 1 ),
                            MTL::Size( width, 1, 1 ) );
                    }
                }
            }

            if ( block )
            {
                MTL::ComputePipelineState* const narrow = pipeline( key + "narrow", ownSource( arm ), "narrow" );

                encoder->setComputePipelineState( narrow );
                encoder->setBuffer( sums_, 0, 0 );
                encoder->setBuffer( c, 0, 1 );
                encoder->dispatchThreads(
                    MTL::Size( std::size_t{ shape_.rows } * shape_.outputs, 1, 1 ), MTL::Size( narrow->threadExecutionWidth(), 1, 1 ) );
            }
        }

        struct QuantizedWeights
        {
            WeightFormats::Encoded encoded;
            MTL::Buffer* codes{ nullptr };
            MTL::Buffer* high{ nullptr };
            MTL::Buffer* scales{ nullptr };
            MTL::Buffer* table{ nullptr };
        };

        MTL::Buffer* copyOf( const void* bytes, std::size_t count )
        {
            MTL::Buffer* const buffer = shared( std::max<std::size_t>( count, 16 ) );

            if ( count > 0 )
            {
                std::memcpy( buffer->contents(), bytes, count );
            }

            return buffer;
        }

        const QuantizedWeights& weights( const Arm& arm )
        {
            std::unique_ptr<QuantizedWeights>& slot = quantized_[arm.format.name];

            if ( !slot )
            {
                slot = std::make_unique<QuantizedWeights>();
                slot->encoded = WeightFormats::encode( arm.format, shape_.outputs, shape_.inputs, data_ == "small" ? 0.001f : 0.02f, 23 );

                const WeightFormats::Encoded& e = slot->encoded;
                slot->codes = copyOf( e.codes.data(), e.codes.size() );
                slot->high = copyOf( e.high_bits.data(), e.high_bits.size() );
                slot->table = copyOf( e.table.data(), e.table.size() * sizeof( float ) );
                slot->scales = arm.format.floatScales()
                    ? copyOf( e.float_scales.data(), e.float_scales.size() * sizeof( float ) )
                    : copyOf( e.half_scales.data(), e.half_scales.size() * sizeof( std::uint16_t ) );
            }

            return *slot;
        }

        QuantizedArguments quantizedArguments( const Arm& arm, std::uint32_t first_output ) const
        {
            const WeightFormats::Encoded& e = quantized_.at( arm.format.name )->encoded;

            return QuantizedArguments{ shape_.inputs, shape_.outputs, shape_.rows, static_cast<std::uint32_t>( e.row_bytes ),
                static_cast<std::uint32_t>( e.high_row_bytes ), e.groups, arm.format.groupOf( shape_.inputs ), first_output };
        }

        std::string quantizedSource( const Arm& arm, const char* type ) const
        {
            using WeightFormats::Kind;
            const Kind kind = arm.format.kind;

            return std::format(
                "#define TYPE {0}\n#define TYPE4 {0}4\n#define SCALE {1}\n#define ROWS {2}\n#define KIND_FP8 {3}\n#define KIND_FP4 {4}\n#define KIND_INT4 {5}\n"
                "#define KIND_INT6 {6}\n#define KIND_CODEBOOK2 {7}\n#define KIND_CODEBOOK3 {8}\n#define USE_TABLE {9}\n#define ACROSS {10}\n{11}",
                type, arm.format.floatScales() ? "float" : "half", shape_.rows, kind == Kind::Fp8 ? 1 : 0, kind == Kind::Fp4 ? 1 : 0,
                kind == Kind::Int4 ? 1 : 0, kind == Kind::Int6 ? 1 : 0, kind == Kind::Codebook2 ? 1 : 0, kind == Kind::Codebook3 ? 1 : 0,
                arm.use_table ? 1 : 0, arm.across > 0 ? arm.across : 1, kQuantizedSource );
        }

        void encodeLanes( MTL::ComputeCommandEncoder* encoder, const Arm& arm )
        {
            const char* const type = facts( arm.input ).name;
            const bool plain = arm.format.kind == WeightFormats::Kind::None;
            const char* const function = plain ? "plainLanes"
                : ( arm.across > 0 ? "quantizedChunksAcross" : ( arm.chunks ? "quantizedChunks" : "quantizedLanes" ) );
            MTL::ComputePipelineState* const state = pipeline(
                std::format( "lanes {} {} {} {} {} {}", type, arm.format.name, shape_.rows, function, arm.use_table, arm.across ), quantizedSource( arm, type ), function );

            encoder->setComputePipelineState( state );
            encoder->setBuffer( inputBuffer( arm.input, false, true, shape_.rows ), 0, 0 );
            encoder->setBuffer( output( arm.output, shape_.rows ), 0, 2 );

            if ( plain )
            {
                const QuantizedArguments arguments{ shape_.inputs, shape_.outputs, shape_.rows, 0, 0, 0, 0, 0 };
                encoder->setBuffer( inputBuffer( arm.input, true, true, shape_.rows ), 0, 1 );
                encoder->setBytes( &arguments, sizeof( arguments ), 3 );
            } else {
                const QuantizedWeights& w = weights( arm );
                const QuantizedArguments arguments = quantizedArguments( arm, 0 );
                encoder->setBuffer( w.codes, 0, 1 );
                encoder->setBytes( &arguments, sizeof( arguments ), 3 );
                encoder->setBuffer( w.scales, 0, 4 );
                encoder->setBuffer( w.table, 0, 5 );
                encoder->setBuffer( w.high, 0, 6 );
            }

            encoder->dispatchThreadgroups(
                MTL::Size( arm.across > 0 ? shape_.outputs / arm.across : shape_.outputs, 1, 1 ), MTL::Size( state->threadExecutionWidth(), 1, 1 ) );
        }

        /// A tensor that shares a buffer, over a block of expanded weights or over bare FP8 codes.
        MTL::Tensor* blockTensor( const std::string& key, MTL::Buffer* buffer, MTL::TensorDataType type, NS::Integer first, NS::Integer second )
        {
            MTL::Tensor*& slot = block_tensors_[key];

            if ( slot )
            {
                return slot;
            }

            NS::Integer dimensions[2] = { first, second };
            NS::Integer strides[2] = { 1, first };
            MTL::TensorDescriptor* const descriptor = MTL::TensorDescriptor::alloc()->init();
            descriptor->setDataType( type );
            descriptor->setUsage( MTL::TensorUsageCompute );
            descriptor->setDimensions( MTL::TensorExtents::alloc()->init( 2, dimensions ) );
            descriptor->setStrides( MTL::TensorExtents::alloc()->init( 2, strides ) );

            NS::Error* error = nullptr;
            slot = buffer->newTensor( descriptor, 0, &error );

            if ( !slot )
            {
                throw std::runtime_error( std::string( "tensor failed: " ) + error->localizedDescription()->utf8String() );
            }

            return slot;
        }

        void encodeMultiplyBlock(
            MTL::ComputeCommandEncoder* encoder, const Arm& arm, const char* weight_type, MTL::Tensor* w, MTL::Buffer* w_buffer,
            MTL::Tensor* c, MTL::Buffer* c_buffer, const char* output_type, std::uint32_t first_output, std::uint32_t columns )
        {
            const std::string key = std::format( "block {} {} {} {} {} {}", facts( arm.input ).name, weight_type, output_type, arm.tile_rows, arm.tile_columns, arm.simdgroups );
            const std::string source = std::format(
                "#define INPUT {}\n#define WEIGHT {}\n#define OUTPUT {}\n#define TILE_ROWS {}\n#define TILE_COLUMNS {}\n#define GROUPS {}\n{}",
                facts( arm.input ).name, weight_type, output_type, arm.tile_rows, arm.tile_columns, arm.simdgroups, kBlockTensorSource );
            MTL::ComputePipelineState* const state = pipeline( key, source, "multiplyBlock" );

            const std::uint32_t rows = rowsOf( arm );
            MTL::Buffer* const a = inputBuffer( arm.input, false, true, rows );
            MTL::Tensor* const ta = blockTensor( std::format( "A {} {}", facts( arm.input ).name, rows ), a, facts( arm.input ).tensor_type, shape_.inputs, rows );
            const MTL::ResourceID identifiers[3] = { ta->gpuResourceID(), w->gpuResourceID(), c->gpuResourceID() };

            encoder->setComputePipelineState( state );
            encoder->setBytes( identifiers, sizeof( identifiers ), 0 );
            encoder->setBytes( &first_output, sizeof( first_output ), 1 );
            encoder->useResource( ta, MTL::ResourceUsageRead );
            encoder->useResource( w, MTL::ResourceUsageRead );
            encoder->useResource( c, MTL::ResourceUsageWrite );
            encoder->useResource( a, MTL::ResourceUsageRead );
            encoder->useResource( w_buffer, MTL::ResourceUsageRead );
            encoder->useResource( c_buffer, MTL::ResourceUsageWrite );
            encoder->dispatchThreadgroups(
                MTL::Size( columns / arm.tile_columns, rows / arm.tile_rows, 1 ),
                MTL::Size( state->threadExecutionWidth() * arm.simdgroups, 1, 1 ) );
        }

        void encodeExpand( MTL::ComputeCommandEncoder* encoder, const Arm& arm )
        {
            const char* const type = facts( arm.input ).name;
            const std::uint32_t block = arm.expand_block ? std::min( arm.expand_block, shape_.outputs ) : shape_.outputs;

            if ( shape_.outputs % block != 0 || block % arm.tile_columns != 0 )
            {
                throw std::invalid_argument( "the block does not divide the outputs and the tile: " + arm.label );
            }

            const QuantizedWeights& w = weights( arm );
            const char* const expand_function = arm.chunks ? "expandChunks" : "expandBlock";
            MTL::ComputePipelineState* const expand = pipeline(
                std::format( "expand {} {} {} {}", type, arm.format.name, expand_function, arm.use_table ), quantizedSource( arm, type ), expand_function );
            MTL::Buffer*& scratch = expand_scratch_[std::format( "{} {}", type, block )];

            if ( !scratch )
            {
                scratch = shared( std::size_t{ block } * shape_.inputs * facts( arm.input ).bytes );
            }

            MTL::Tensor* const tw = blockTensor(
                std::format( "scratch {} {}", type, block ), scratch, facts( arm.input ).tensor_type, shape_.inputs, block );
            const std::uint32_t rows = rowsOf( arm );
            MTL::Buffer* const c = output( arm.output, rows );
            MTL::Tensor* const tc = blockTensor(
                std::format( "C {} {}", facts( arm.output ).name, rows ), c, facts( arm.output ).tensor_type, shape_.outputs, rows );

            for ( std::uint32_t first = 0; first < shape_.outputs; first += block )
            {
                const QuantizedArguments arguments = quantizedArguments( arm, first );

                encoder->setComputePipelineState( expand );
                encoder->setBuffer( w.codes, 0, 0 );
                encoder->setBuffer( scratch, 0, 1 );
                encoder->setBytes( &arguments, sizeof( arguments ), 3 );
                encoder->setBuffer( w.scales, 0, 4 );
                encoder->setBuffer( w.table, 0, 5 );
                encoder->setBuffer( w.high, 0, 6 );
                encoder->dispatchThreads( MTL::Size( arm.chunks ? shape_.inputs / 8 : shape_.inputs, block, 1 ), MTL::Size( expand->threadExecutionWidth(), 1, 1 ) );
                encoder->memoryBarrier( MTL::BarrierScopeBuffers );

                encodeMultiplyBlock( encoder, arm, type, tw, scratch, tc, c, facts( arm.output ).name, first, block );
                encoder->memoryBarrier( MTL::BarrierScopeBuffers );
            }
        }

        /// FP8 only: TensorOps on the bare codes into 32 bits, then each output's scale.
        void encodeDirect( MTL::ComputeCommandEncoder* encoder, const Arm& arm )
        {
            const QuantizedWeights& w = weights( arm );
            const char* const type = facts( arm.input ).name;
            const std::uint32_t rows = rowsOf( arm );

            if ( arm.format.kind != WeightFormats::Kind::Fp8 )
            {
                throw std::invalid_argument( "TensorOps can carry only FP8's scales: " + arm.label );
            }

            if ( !sums_ )
            {
                sums_ = shared( std::size_t{ rows } * shape_.outputs * 4 );
            }

            MTL::Tensor* const tw = blockTensor( "fp8 codes", w.codes, MTL::TensorDataTypeMetalFloat8E4M3, shape_.inputs, shape_.outputs );
            MTL::Tensor* const tc = blockTensor( "sums", sums_, MTL::TensorDataTypeFloat32, shape_.outputs, rows );

            encodeMultiplyBlock( encoder, arm, "metal_fp8_e4m3_format", tw, w.codes, tc, sums_, "float", 0, shape_.outputs );
            encoder->memoryBarrier( MTL::BarrierScopeBuffers );

            MTL::ComputePipelineState* const apply = pipeline(
                std::format( "apply {} fp8", type ), quantizedSource( arm, type ), "applyRowScales" );
            const QuantizedArguments arguments = quantizedArguments( arm, 0 );

            encoder->setComputePipelineState( apply );
            encoder->setBuffer( sums_, 0, 0 );
            encoder->setBuffer( output( arm.output, rows ), 0, 1 );
            encoder->setBytes( &arguments, sizeof( arguments ), 3 );
            encoder->setBuffer( w.scales, 0, 4 );
            encoder->dispatchThreads( MTL::Size( shape_.outputs, rows, 1 ), MTL::Size( apply->threadExecutionWidth(), 1, 1 ) );
        }

        void encodeDot( MTL::ComputeCommandEncoder* encoder, const Arm& arm )
        {
            const char* const function = arm.dot_form == 1 ? "dotScalar"
                : ( arm.dot_form == 2 ? "dotVector" : ( arm.dot_form == 3 ? "dotWide" : "dotLanes" ) );
            MTL::ComputePipelineState* const state = pipeline(
                std::format( "dot {} {} {}", facts( arm.input ).name, function, arm.lane_rows ), ownSource( arm ), function );
            const OwnArguments arguments{ shape_.inputs, shape_.outputs, 0, 0, 0, shape_.inputs };

            encoder->setComputePipelineState( state );
            encoder->setBuffer( inputBuffer( arm.input, false, true, shape_.rows ), 0, 0 );
            encoder->setBuffer( inputBuffer( arm.input, true, true, shape_.rows ), 0, 1 );
            encoder->setBuffer( output( arm.output, shape_.rows ), 0, 2 );
            encoder->setBytes( &arguments, sizeof( arguments ), 3 );

            if ( arm.dot_form == 4 )
            {
                encoder->dispatchThreadgroups(
                    MTL::Size( shape_.outputs / arm.lane_rows, shape_.rows, 1 ), MTL::Size( state->threadExecutionWidth(), 1, 1 ) );

                return;
            }

            encoder->dispatchThreads( MTL::Size( shape_.outputs, shape_.rows, 1 ), MTL::Size( state->threadExecutionWidth(), 1, 1 ) );
        }

        Shape shape_;
        std::string data_;
        MTL::Device* device_{ nullptr };
        MTL::CommandQueue* queue_{ nullptr };
        MTL::Buffer* sums_{ nullptr };
        std::map<std::tuple<Precision, std::uint32_t>, MTL::Buffer*> outputs_;
        std::map<std::tuple<Precision, bool, bool, std::uint32_t>, MTL::Buffer*> inputs_;
        std::map<std::tuple<Precision, int, bool, std::uint32_t>, MTL::Tensor*> tensors_;
        std::map<std::string, MTL::ComputePipelineState*> pipelines_;
        std::map<std::string, std::unique_ptr<QuantizedWeights>> quantized_;
        std::map<std::string, MTL::Tensor*> block_tensors_;
        std::map<std::string, MTL::Buffer*> expand_scratch_;
    };

    double quantile( std::vector<double> values, double fraction )
    {
        std::ranges::sort( values );

        return values[static_cast<std::size_t>( fraction * static_cast<double>( values.size() - 1 ) + 0.5 )];
    }
}

/**
 * Usage: mila-metal-matrix-timing M,K,N data run rounds arm [arm ...]
 * `data` is normal, uniform or small. Each result line starts with R, and is one arm of one process run.
 */
int main( int argument_count, char** arguments )
{
    if ( argument_count < 6 )
    {
        std::cerr << "usage: mila-metal-matrix-timing M,K,N data run rounds arm [arm ...]\n";

        return 2;
    }

    const std::string shape_text = arguments[1];
    const std::size_t first_comma = shape_text.find( ',' );
    const std::size_t second_comma = shape_text.find( ',', first_comma + 1 );
    const Shape shape{
        static_cast<std::uint32_t>( std::stoul( shape_text.substr( 0, first_comma ) ) ),
        static_cast<std::uint32_t>( std::stoul( shape_text.substr( first_comma + 1, second_comma - first_comma - 1 ) ) ),
        static_cast<std::uint32_t>( std::stoul( shape_text.substr( second_comma + 1 ) ) ) };
    const std::string data = arguments[2];
    const int run = std::atoi( arguments[3] );
    const int rounds = std::atoi( arguments[4] );

    std::vector<Arm> arms;

    for ( int i = 5; i < argument_count; ++i )
    {
        arms.push_back( parseArm( arguments[i] ) );
    }

    try
    {
        Runner runner( shape, data );

        for ( Arm& arm : arms )
        {
            for ( int warm = 0; warm < 3; ++warm )
            {
                runner.run( arm );
            }
        }

        for ( int round = 0; round < rounds; ++round )
        {
            for ( std::size_t i = 0; i < arms.size(); ++i )
            {
                Arm& arm = arms[( i + static_cast<std::size_t>( round * 3 + run ) ) % arms.size()];
                arm.times.push_back( runner.run( arm ) );
            }
        }

        for ( std::size_t i = 0; i < arms.size(); ++i )
        {
            Arm& arm = arms[i];
            runner.clearOutput( arm );
            runner.run( arm );
            arm.error = runner.check( arm );

            if ( i == 0 && run == 0 )
            {
                const std::int64_t shift = shape.rows > 8 ? std::int64_t{ 8 } * shape.outputs : 8;
                std::cout << std::format( "check of the check: misaligned output scores {:.3f}, true output {:.3g}\n",
                    runner.check( arm, shift ), arm.error );
            }

            std::cout << std::format( "R {} {},{},{} {} {} {:.3f} {:.3f} {:.3f} {:.3f} {:.3g}\n", run, shape.rows, shape.inputs,
                shape.outputs, data, arm.label, quantile( arm.times, 0.5 ), quantile( arm.times, 0.25 ),
                quantile( arm.times, 0.75 ), quantile( arm.times, 0.0 ), arm.error );
        }
    } catch ( const std::exception& error ) {
        std::cerr << "failed: " << error.what() << "\n";

        return 1;
    }

    return 0;
}
