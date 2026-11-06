// BF16 weights to FP8 per output channel and FP4 per group, at load, writing the bytes Mila's CUDA quantize writes: the
// same scale, its reciprocal by IEEE division, and the same rounding of each weight. Needs QuantizedCodes.metal.
//
// A weight that is not finite is counted at `not_finite`, which the host reads to refuse the tensor by name.
//
// Each kernel's body is compiled in safe math: Metal's default fast math may turn w * (1 / scale) into w / scale, which rounds
// differently at a code's boundary. Scoped to the body, since the library's sources are compiled as one.

#include <metal_stdlib>

using namespace metal;

struct WeightQuantizationShape
{
    uint inputs;
};

constant uint kFp8QuantizationThreads = 256;

// One threadgroup of 256 per output channel: scale = max |w| / 448, or 1 for a row of zeros, and code = E4M3( w * (1 / scale) ).
kernel void linear_quantize_fp8_per_channel(
    device const ushort* source [[buffer(0)]],
    device uchar* codes [[buffer(1)]],
    device float* scales [[buffer(2)]],
    device atomic_uint* not_finite [[buffer(3)]],
    constant WeightQuantizationShape& shape [[buffer(4)]],
    uint channel [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]] )
{
#pragma METAL fp math_mode(safe)
    threadgroup float largest_of_simdgroup[kFp8QuantizationThreads / 32];

    device const ushort* row = source + channel * shape.inputs;
    device uchar* row_codes = codes + channel * shape.inputs;

    float largest = 0.0f;
    uint not_finite_here = 0;

    for ( uint i = thread_index; i < shape.inputs; i += kFp8QuantizationThreads )
    {
        not_finite_here += bf16IsNotFinite( row[i] ) ? 1u : 0u;
        largest = fmax( largest, fabs( bf16Value( row[i] ) ) );
    }

    largest = simd_max( largest );

    if ( lane == 0 )
    {
        largest_of_simdgroup[simdgroup] = largest;
    }

    threadgroup_barrier( mem_flags::mem_threadgroup );

    largest = 0.0f;

    for ( uint s = 0; s < kFp8QuantizationThreads / 32; ++s )
    {
        largest = fmax( largest, largest_of_simdgroup[s] );
    }

    const float scale = largest > 0.0f ? precise::divide( largest, kFp8E4M3Largest ) : 1.0f;
    const float inverse = precise::divide( 1.0f, scale );

    if ( thread_index == 0 )
    {
        scales[channel] = scale;
    }

    for ( uint i = thread_index; i < shape.inputs; i += kFp8QuantizationThreads )
    {
        row_codes[i] = fp8E4M3Code( bf16Value( row[i] ) * inverse );
    }

    if ( not_finite_here > 0 )
    {
        atomic_fetch_add_explicit( not_finite, not_finite_here, memory_order_relaxed );
    }
}

// One threadgroup per group of TGroup inputs of one output, a thread for each input: scale = max |w| / 6, or 1 for a group of
// zeros, and code = E2M1( w * (1 / scale) ), two to a byte with the even input low. The grid is (groups, outputs).
template<uint TGroup>
kernel void linear_quantize_fp4_per_group(
    device const ushort* source [[buffer(0)]],
    device uchar* codes [[buffer(1)]],
    device float* scales [[buffer(2)]],
    device atomic_uint* not_finite [[buffer(3)]],
    constant WeightQuantizationShape& shape [[buffer(4)]],
    uint2 position [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]] )
{
#pragma METAL fp math_mode(safe)
    threadgroup float largest_of_simdgroup[TGroup / 32];

    const uint group = position.x;
    const uint channel = position.y;
    const uint groups = shape.inputs / TGroup;
    const uint input = group * TGroup + thread_index;

    const ushort bits = source[channel * shape.inputs + input];
    const float value = bf16Value( bits );

    const float largest_here = simd_max( fabs( value ) );

    if ( lane == 0 )
    {
        largest_of_simdgroup[simdgroup] = largest_here;
    }

    threadgroup_barrier( mem_flags::mem_threadgroup );

    float largest = 0.0f;

    for ( uint s = 0; s < TGroup / 32; ++s )
    {
        largest = fmax( largest, largest_of_simdgroup[s] );
    }

    const float scale = largest > 0.0f ? precise::divide( largest, kFp4E2M1Largest ) : 1.0f;
    const float inverse = precise::divide( 1.0f, scale );

    if ( thread_index == 0 )
    {
        scales[channel * groups + group] = scale;
    }

    const uchar code = fp4E2M1Code( value * inverse );
    const uchar odd = simd_shuffle_down( code, 1 );

    if ( ( thread_index & 1u ) == 0u )
    {
        codes[( channel * shape.inputs + input ) / 2] = code | uchar( odd << 4 );
    }

    if ( bf16IsNotFinite( bits ) )
    {
        atomic_fetch_add_explicit( not_finite, 1u, memory_order_relaxed );
    }
}

template [[host_name( "linear_quantize_fp4_per_group_64" )]] kernel void linear_quantize_fp4_per_group<64>(
    device const ushort*, device uchar*, device float*, device atomic_uint*, constant WeightQuantizationShape&, uint2, uint, uint, uint );
template [[host_name( "linear_quantize_fp4_per_group_128" )]] kernel void linear_quantize_fp4_per_group<128>(
    device const ushort*, device uchar*, device float*, device atomic_uint*, constant WeightQuantizationShape&, uint2, uint, uint, uint );
