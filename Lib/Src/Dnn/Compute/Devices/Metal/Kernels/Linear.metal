// output = input * weight^T, for a Linear. TensorOps runs what fills whole tiles, and the own kernels run a decode's one row
// and whatever a tile leaves over (MetalBackend.md section 4.2). Own kernels sum in 32 bits, which needs no storage.

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;

// A rectangle of the output, [first_row, last_row) by [first_output, last_output).
struct LinearRectangle
{
    uint inputs;
    uint outputs;
    uint first_row;
    uint first_output;
    uint last_output;
};

constant uint kLinearLaneOutputs = 4;

// One SIMD-group per kLinearLaneOutputs outputs of one row: its lanes walk the same rows of weights side by side, and are summed
// at the end. The grid is (groups of outputs, rows). Needs `inputs` a multiple of 4, since it reads the rows as vectors.
template<typename TElement, typename TVector>
kernel void linear_lanes(
    device const TVector* input [[buffer(0)]],
    device const TVector* weight [[buffer(1)]],
    device TElement* output [[buffer(2)]],
    constant LinearRectangle& rectangle [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 threads [[threads_per_threadgroup]] )
{
    const uint vectors = rectangle.inputs / 4;
    const uint first = rectangle.first_output + group.x * kLinearLaneOutputs;
    const uint outputs = min( kLinearLaneOutputs, rectangle.last_output - first );
    const uint row = rectangle.first_row + group.y;

    device const TVector* a = input + row * vectors;
    device const TVector* w = weight + first * vectors;

    float sums[kLinearLaneOutputs] = { 0.0f, 0.0f, 0.0f, 0.0f };

    for ( uint k = lane; k < vectors; k += threads.x )
    {
        const float4 x = float4( a[k] );

        for ( uint r = 0; r < outputs; ++r )
        {
            sums[r] += dot( x, float4( w[r * vectors + k] ) );
        }
    }

    for ( uint r = 0; r < outputs; ++r )
    {
        const float total = simd_sum( sums[r] );

        if ( lane == 0 )
        {
            output[row * rectangle.outputs + first + r] = TElement( total );
        }
    }
}

template [[host_name("linear_lanes_half")]] kernel void linear_lanes<half, half4>(
    device const half4*, device const half4*, device half*, constant LinearRectangle&, uint3, uint, uint3 );
template [[host_name("linear_lanes_bfloat")]] kernel void linear_lanes<bfloat, bfloat4>(
    device const bfloat4*, device const bfloat4*, device bfloat*, constant LinearRectangle&, uint3, uint, uint3 );

// One output element per thread, for inputs that are not a multiple of 4. The grid is (outputs, rows), in `elements` order.
struct LinearElementRectangle
{
    uint inputs;
    uint outputs;
    uint first_row;
    uint first_output;
    uint outputs_in_rectangle;
};

template<typename TElement>
kernel void linear_elements(
    device const TElement* input [[buffer(0)]],
    device const TElement* weight [[buffer(1)]],
    device TElement* output [[buffer(2)]],
    constant LinearElementRectangle& rectangle [[buffer(3)]],
    uint index [[thread_position_in_grid]] )
{
    const uint row = rectangle.first_row + index / rectangle.outputs_in_rectangle;
    const uint column = rectangle.first_output + index % rectangle.outputs_in_rectangle;

    device const TElement* a = input + row * rectangle.inputs;
    device const TElement* w = weight + column * rectangle.inputs;
    float sum = 0.0f;

    for ( uint k = 0; k < rectangle.inputs; ++k )
    {
        sum += float( a[k] ) * float( w[k] );
    }

    output[row * rectangle.outputs + column] = TElement( sum );
}

template [[host_name("linear_elements_half")]] kernel void linear_elements<half>(
    device const half*, device const half*, device half*, constant LinearElementRectangle&, uint );
template [[host_name("linear_elements_bfloat")]] kernel void linear_elements<bfloat>(
    device const bfloat*, device const bfloat*, device bfloat*, constant LinearElementRectangle&, uint );

// output += bias, on every row. The grid is every output element.
struct LinearBiasShape
{
    uint outputs;
};

template<typename TElement>
kernel void linear_add_bias(
    device const TElement* bias [[buffer(0)]],
    device TElement* output [[buffer(1)]],
    constant LinearBiasShape& shape [[buffer(2)]],
    uint index [[thread_position_in_grid]] )
{
    output[index] = TElement( float( output[index] ) + float( bias[index % shape.outputs] ) );
}

template [[host_name("linear_add_bias_half")]] kernel void linear_add_bias<half>(
    device const half*, device half*, constant LinearBiasShape&, uint );
template [[host_name("linear_add_bias_bfloat")]] kernel void linear_add_bias<bfloat>(
    device const bfloat*, device bfloat*, constant LinearBiasShape&, uint );

// One tile of the product per threadgroup. The weights are stored a row per output, so they multiply transposed. The struct of
// resource ids is the binding that works (section 4.2); a tensor bound alone wrote zeros.
template<typename TElement>
struct LinearMatrices
{
    tensor<device TElement, dextents<int, 2>> input;
    tensor<device TElement, dextents<int, 2>> weight;
    tensor<device TElement, dextents<int, 2>> output;
};

template<typename TElement, int TTileRows, int TTileColumns, int TSimdgroups>
kernel void linear_tiles(
    constant LinearMatrices<TElement>& matrices [[buffer(0)]],
    uint2 tile [[threadgroup_position_in_grid]] )
{
    auto input = matrices.input;
    auto weight = matrices.weight;
    auto output = matrices.output;

    constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(
        TTileRows, TTileColumns, static_cast<int>( dynamic_extent ), false, true, false );

    mpp::tensor_ops::matmul2d<descriptor, execution_simdgroups<TSimdgroups>> multiply;

    auto a = input.slice( 0, tile.y * TTileRows );
    auto w = weight.slice( 0, tile.x * TTileColumns );
    auto c = output.slice( tile.x * TTileColumns, tile.y * TTileRows );

    multiply.run( a, w, c );
}

template [[host_name("linear_tiles_half_8_16_2")]] kernel void linear_tiles<half, 8, 16, 2>(
    constant LinearMatrices<half>&, uint2 );
template [[host_name("linear_tiles_half_128_16_4")]] kernel void linear_tiles<half, 128, 16, 4>(
    constant LinearMatrices<half>&, uint2 );
template [[host_name("linear_tiles_bfloat_8_16_2")]] kernel void linear_tiles<bfloat, 8, 16, 2>(
    constant LinearMatrices<bfloat>&, uint2 );
template [[host_name("linear_tiles_bfloat_128_16_4")]] kernel void linear_tiles<bfloat, 128, 16, 4>(
    constant LinearMatrices<bfloat>&, uint2 );
