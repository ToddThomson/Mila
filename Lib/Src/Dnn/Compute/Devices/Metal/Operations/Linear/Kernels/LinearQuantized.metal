// output = input * weight^T on quantized weights, on the paths Phase 2 measured (MetalBackend.md section 4.2): own lanes
// for a few rows; for more, the weights expanded a block of outputs at a time for TensorOps, or for FP8, TensorOps on the
// bare codes with each output's scale after. Sums are 32 bits. Needs QuantizedCodes.metal.

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

// Eight weights from one load of their codes. Every format's group is at least 32, so eight never cross a scale.
struct Fp8Codes
{
    static void chunk( device const uchar* row, uint c, thread float4& first, thread float4& second )
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
    static void chunk( device const uchar* row, uint c, thread float4& first, thread float4& second )
    {
        const uint bits = reinterpret_cast<device const uint*>( row )[c];
        const uint4 a = uint4( bits, bits >> 4, bits >> 8, bits >> 12 ) & 15u;
        const uint4 b = uint4( bits >> 16, bits >> 20, bits >> 24, bits >> 28 ) & 15u;

        first = float4( kFp4E2M1Values[a.x], kFp4E2M1Values[a.y], kFp4E2M1Values[a.z], kFp4E2M1Values[a.w] );
        second = float4( kFp4E2M1Values[b.x], kFp4E2M1Values[b.y], kFp4E2M1Values[b.z], kFp4E2M1Values[b.w] );
    }
};

// One SIMD-group per TAcross outputs of up to TRows rows: its lanes walk the codes once, eight weights at a time, for every
// row. The grid is (groups of outputs, groups of rows). One row reads its activations once for TAcross outputs, which won
// at decode; several rows read each output's weights once for all of them, which won at 8 rows.
template<typename TCodes, typename TElement, typename TVector, uint TAcross, uint TRows>
kernel void linear_quantized_lanes(
    device const TVector* input [[buffer(0)]],
    device const uchar* codes [[buffer(1)]],
    device const float* scales [[buffer(2)]],
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
            TCodes::chunk( codes + o * shape.row_bytes, c, first, second );

            const float scale = scales[o * shape.groups + c * 8 / shape.group];

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

#define MILA_QUANTIZED_LANES( codes, name, element, across, rows ) \
    template [[host_name( "linear_quantized_lanes_" name "_" #element "_" #across "_" #rows )]] \
    kernel void linear_quantized_lanes<codes, element, element##4, across, rows>( \
        device const element##4*, device const uchar*, device const float*, device element*, constant QuantizedLinearShape&, \
        uint3, uint, uint3 );

MILA_QUANTIZED_LANES( Fp8Codes, "fp8", half, 4, 1 )
MILA_QUANTIZED_LANES( Fp8Codes, "fp8", half, 1, 8 )
MILA_QUANTIZED_LANES( Fp8Codes, "fp8", bfloat, 4, 1 )
MILA_QUANTIZED_LANES( Fp8Codes, "fp8", bfloat, 1, 8 )
MILA_QUANTIZED_LANES( Fp4Codes, "fp4", half, 4, 1 )
MILA_QUANTIZED_LANES( Fp4Codes, "fp4", half, 1, 8 )
MILA_QUANTIZED_LANES( Fp4Codes, "fp4", bfloat, 4, 1 )
MILA_QUANTIZED_LANES( Fp4Codes, "fp4", bfloat, 1, 8 )

// Expand outputs [first_output, last_output) to the compute type, a row of inputs each, for TensorOps to multiply. Eight
// weights to a thread; the grid is every eight inputs of every output in the block.
template<typename TCodes, typename TVector>
kernel void linear_quantized_expand(
    device const uchar* codes [[buffer(0)]],
    device const float* scales [[buffer(1)]],
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
    TCodes::chunk( codes + o * shape.row_bytes, c, first, second );

    const float scale = scales[o * shape.groups + c * 8 / shape.group];
    device TVector* out = expanded + block_output * ( shape.inputs / 4 ) + c * 2;

    out[0] = TVector( first * scale );
    out[1] = TVector( second * scale );
}

template [[host_name( "linear_quantized_expand_fp8_half" )]] kernel void linear_quantized_expand<Fp8Codes, half4>(
    device const uchar*, device const float*, device half4*, constant QuantizedLinearShape&, uint );
template [[host_name( "linear_quantized_expand_fp8_bfloat" )]] kernel void linear_quantized_expand<Fp8Codes, bfloat4>(
    device const uchar*, device const float*, device bfloat4*, constant QuantizedLinearShape&, uint );
template [[host_name( "linear_quantized_expand_fp4_half" )]] kernel void linear_quantized_expand<Fp4Codes, half4>(
    device const uchar*, device const float*, device half4*, constant QuantizedLinearShape&, uint );
template [[host_name( "linear_quantized_expand_fp4_bfloat" )]] kernel void linear_quantized_expand<Fp4Codes, bfloat4>(
    device const uchar*, device const float*, device bfloat4*, constant QuantizedLinearShape&, uint );

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
