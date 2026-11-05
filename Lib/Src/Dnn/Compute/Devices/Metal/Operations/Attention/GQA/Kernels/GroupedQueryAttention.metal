// Grouped-query attention over a KV cache held [B, NKV, capacity, HS].
//
// Prefill runs Q * K^T through the Linear product, then a causal softmax, then attention * V through TensorOps below. A decode step
// runs own kernels only: it reads its position from memory, which TensorOps' fixed shapes could not do, and its grid never changes, so
// that the step stays a pure function of device memory (CLAUDE.md, DecodeGraph.md). All arithmetic is in 32 bits. A lane with no keys starts its
// largest score at -FLT_MAX and not at -infinity, since Metal compiles with fast math on, which assumes no infinities.

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;

struct GqaGeometry
{
    uint batches;
    uint tokens;
    uint heads;
    uint kv_heads;
    uint head_size;
    uint capacity;
    uint window;
    uint position_offset;
    float attention_scale;
};

// ---------------------------------------------------------------------------------------------------------------------
// The cache write. Element `index` of K and V's [B, tokens, NKV, HS] goes to the cache at its absolute position.
// ---------------------------------------------------------------------------------------------------------------------

template<typename TElement>
inline void gqa_write_element(
    device const TElement* key, device const TElement* value, device TElement* key_cache, device TElement* value_cache,
    constant GqaGeometry& geometry, uint first_position, uint index )
{
    const uint channel = index % geometry.head_size;
    const uint rest = index / geometry.head_size;
    const uint kv_head = rest % geometry.kv_heads;
    const uint token_of_batch = rest / geometry.kv_heads;
    const uint token = token_of_batch % geometry.tokens;
    const uint batch = token_of_batch / geometry.tokens;

    const uint cached = ( ( batch * geometry.kv_heads + kv_head ) * geometry.capacity + first_position + token ) * geometry.head_size + channel;

    key_cache[cached] = key[index];
    value_cache[cached] = value[index];
}

template<typename TElement>
kernel void gqa_write_prefill(
    device const TElement* key [[buffer(0)]],
    device const TElement* value [[buffer(1)]],
    device TElement* key_cache [[buffer(2)]],
    device TElement* value_cache [[buffer(3)]],
    constant GqaGeometry& geometry [[buffer(4)]],
    uint index [[thread_position_in_grid]] )
{
    gqa_write_element<TElement>( key, value, key_cache, value_cache, geometry, geometry.position_offset, index );
}

template<typename TElement>
kernel void gqa_write_decode(
    device const TElement* key [[buffer(0)]],
    device const TElement* value [[buffer(1)]],
    device TElement* key_cache [[buffer(2)]],
    device TElement* value_cache [[buffer(3)]],
    device const int* decode_position [[buffer(4)]],
    constant GqaGeometry& geometry [[buffer(5)]],
    uint index [[thread_position_in_grid]] )
{
    gqa_write_element<TElement>( key, value, key_cache, value_cache, geometry, uint( *decode_position ), index );
}

template [[host_name("gqa_write_prefill_half")]] kernel void gqa_write_prefill<half>(
    device const half*, device const half*, device half*, device half*, constant GqaGeometry&, uint );
template [[host_name("gqa_write_prefill_bfloat")]] kernel void gqa_write_prefill<bfloat>(
    device const bfloat*, device const bfloat*, device bfloat*, device bfloat*, constant GqaGeometry&, uint );
template [[host_name("gqa_write_decode_half")]] kernel void gqa_write_decode<half>(
    device const half*, device const half*, device half*, device half*, device const int*, constant GqaGeometry&, uint );
template [[host_name("gqa_write_decode_bfloat")]] kernel void gqa_write_decode<bfloat>(
    device const bfloat*, device const bfloat*, device bfloat*, device bfloat*, device const int*, constant GqaGeometry&, uint );

// ---------------------------------------------------------------------------------------------------------------------
// Prefill. Q [B, tokens, NH, HS] goes to [B, NH, tokens, HS], scaled here because TensorOps has no scale of its own. Scaling first
// also keeps the scores in FP16's range, where an unscaled product could pass 65504.
// ---------------------------------------------------------------------------------------------------------------------

template<typename TElement>
kernel void gqa_permute_query(
    device const TElement* query [[buffer(0)]],
    device TElement* permuted [[buffer(1)]],
    constant GqaGeometry& geometry [[buffer(2)]],
    uint index [[thread_position_in_grid]] )
{
    const uint channel = index % geometry.head_size;
    const uint rest = index / geometry.head_size;
    const uint token = rest % geometry.tokens;
    const uint head_of_batch = rest / geometry.tokens;
    const uint head = head_of_batch % geometry.heads;
    const uint batch = head_of_batch / geometry.heads;

    const uint source = ( ( batch * geometry.tokens + token ) * geometry.heads + head ) * geometry.head_size + channel;

    permuted[index] = TElement( float( query[source] ) * geometry.attention_scale );
}

template [[host_name("gqa_permute_query_half")]] kernel void gqa_permute_query<half>(
    device const half*, device half*, constant GqaGeometry&, uint );
template [[host_name("gqa_permute_query_bfloat")]] kernel void gqa_permute_query<bfloat>(
    device const bfloat*, device bfloat*, constant GqaGeometry&, uint );

// A row of [B, NH, tokens, capacity] per SIMD-group. Everything outside the row's band is zero, for the product with V that follows.
template<typename TElement>
kernel void gqa_prefill_softmax(
    device const TElement* scores [[buffer(0)]],
    device TElement* attention [[buffer(1)]],
    constant GqaGeometry& geometry [[buffer(2)]],
    constant uint& attended [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 threads [[threads_per_threadgroup]] )
{
    const uint row = group.x;
    const uint position = geometry.position_offset + row % geometry.tokens;
    const uint first_key = geometry.window > 0 && position + 1 > geometry.window ? position + 1 - geometry.window : 0;
    const uint start = row * geometry.capacity;

    float largest = -FLT_MAX;

    for ( uint j = lane; j < attended; j += threads.x )
    {
        if ( j >= first_key && j <= position )
        {
            largest = max( largest, float( scores[start + j] ) );
        }
    }

    largest = simd_max( largest );

    float sum = 0.0f;

    for ( uint j = lane; j < attended; j += threads.x )
    {
        if ( j >= first_key && j <= position )
        {
            sum += exp( float( scores[start + j] ) - largest );
        }
    }

    sum = simd_sum( sum );

    for ( uint j = lane; j < attended; j += threads.x )
    {
        attention[start + j] = j >= first_key && j <= position
            ? TElement( exp( float( scores[start + j] ) - largest ) / sum )
            : TElement( 0.0f );
    }
}

template [[host_name("gqa_prefill_softmax_half")]] kernel void gqa_prefill_softmax<half>(
    device const half*, device half*, constant GqaGeometry&, constant uint&, uint3, uint, uint3 );
template [[host_name("gqa_prefill_softmax_bfloat")]] kernel void gqa_prefill_softmax<bfloat>(
    device const bfloat*, device bfloat*, constant GqaGeometry&, constant uint&, uint3, uint, uint3 );

// attention * V, one tile per threadgroup. V's rows are the keys, so unlike a Linear's weights it multiplies as stored.
template<typename TElement>
struct GqaValueMatrices
{
    tensor<device TElement, dextents<int, 2>> attention;
    tensor<device TElement, dextents<int, 2>> values;
    tensor<device TElement, dextents<int, 2>> output;
};

template<typename TElement, int TTileRows, int TTileColumns, int TSimdgroups>
kernel void gqa_value_tiles(
    constant GqaValueMatrices<TElement>& matrices [[buffer(0)]],
    uint2 tile [[threadgroup_position_in_grid]] )
{
    auto attention = matrices.attention;
    auto values = matrices.values;
    auto output = matrices.output;

    constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(
        TTileRows, TTileColumns, static_cast<int>( dynamic_extent ), false, false, false );

    mpp::tensor_ops::matmul2d<descriptor, execution_simdgroups<TSimdgroups>> multiply;

    auto a = attention.slice( 0, tile.y * TTileRows );
    auto v = values.slice( tile.x * TTileColumns, 0 );
    auto c = output.slice( tile.x * TTileColumns, tile.y * TTileRows );

    multiply.run( a, v, c );
}

template [[host_name("gqa_value_tiles_half_8_16_2")]] kernel void gqa_value_tiles<half, 8, 16, 2>(
    constant GqaValueMatrices<half>&, uint2 );
template [[host_name("gqa_value_tiles_half_128_16_4")]] kernel void gqa_value_tiles<half, 128, 16, 4>(
    constant GqaValueMatrices<half>&, uint2 );
template [[host_name("gqa_value_tiles_bfloat_8_16_2")]] kernel void gqa_value_tiles<bfloat, 8, 16, 2>(
    constant GqaValueMatrices<bfloat>&, uint2 );
template [[host_name("gqa_value_tiles_bfloat_128_16_4")]] kernel void gqa_value_tiles<bfloat, 128, 16, 4>(
    constant GqaValueMatrices<bfloat>&, uint2 );

// What a tile leaves over: one output element per thread, over [first_row, rows) by [first_column, head_size). Grid order is (rows, columns).
struct GqaValueRectangle
{
    uint attended;
    uint attention_stride;
    uint head_size;
    uint first_row;
    uint first_column;
    uint columns;
};

template<typename TElement>
kernel void gqa_value_elements(
    device const TElement* attention [[buffer(0)]],
    device const TElement* values [[buffer(1)]],
    device TElement* output [[buffer(2)]],
    constant GqaValueRectangle& rectangle [[buffer(3)]],
    uint index [[thread_position_in_grid]] )
{
    const uint row = rectangle.first_row + index / rectangle.columns;
    const uint column = rectangle.first_column + index % rectangle.columns;

    device const TElement* a = attention + row * rectangle.attention_stride;
    float sum = 0.0f;

    for ( uint j = 0; j < rectangle.attended; ++j )
    {
        sum += float( a[j] ) * float( values[j * rectangle.head_size + column] );
    }

    output[row * rectangle.head_size + column] = TElement( sum );
}

template [[host_name("gqa_value_elements_half")]] kernel void gqa_value_elements<half>(
    device const half*, device const half*, device half*, constant GqaValueRectangle&, uint );
template [[host_name("gqa_value_elements_bfloat")]] kernel void gqa_value_elements<bfloat>(
    device const bfloat*, device const bfloat*, device bfloat*, constant GqaValueRectangle&, uint );

// [B, NH, tokens, HS] back to the output's [B, tokens, NH * HS].
template<typename TElement>
kernel void gqa_unpermute_output(
    device const TElement* permuted [[buffer(0)]],
    device TElement* output [[buffer(1)]],
    constant GqaGeometry& geometry [[buffer(2)]],
    uint index [[thread_position_in_grid]] )
{
    const uint channel = index % geometry.head_size;
    const uint rest = index / geometry.head_size;
    const uint head = rest % geometry.heads;
    const uint token_of_batch = rest / geometry.heads;
    const uint token = token_of_batch % geometry.tokens;
    const uint batch = token_of_batch / geometry.tokens;

    const uint source = ( ( ( batch * geometry.heads + head ) * geometry.tokens ) + token ) * geometry.head_size + channel;

    output[index] = permuted[source];
}

template [[host_name("gqa_unpermute_output_half")]] kernel void gqa_unpermute_output<half>(
    device const half*, device half*, constant GqaGeometry&, uint );
template [[host_name("gqa_unpermute_output_bfloat")]] kernel void gqa_unpermute_output<bfloat>(
    device const bfloat*, device bfloat*, constant GqaGeometry&, uint );

// ---------------------------------------------------------------------------------------------------------------------
// Decode, one token. A SIMD-group per query head of each batch, its lanes walking the keys up to the position.
// ---------------------------------------------------------------------------------------------------------------------

template<typename TElement>
kernel void gqa_decode_softmax(
    device const TElement* query [[buffer(0)]],
    device const TElement* key_cache [[buffer(1)]],
    device TElement* scores [[buffer(2)]],
    device TElement* attention [[buffer(3)]],
    device const int* decode_position [[buffer(4)]],
    constant GqaGeometry& geometry [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 threads [[threads_per_threadgroup]] )
{
    const uint head_of_batch = group.x;
    const uint head = head_of_batch % geometry.heads;
    const uint batch = head_of_batch / geometry.heads;
    const uint kv_head = head / ( geometry.heads / geometry.kv_heads );

    const uint position = uint( *decode_position );
    const uint first_key = geometry.window > 0 && position + 1 > geometry.window ? position + 1 - geometry.window : 0;

    device const TElement* q = query + ( batch * geometry.heads + head ) * geometry.head_size;
    device const TElement* keys = key_cache + ( batch * geometry.kv_heads + kv_head ) * geometry.capacity * geometry.head_size;
    device TElement* row_scores = scores + head_of_batch * geometry.capacity;
    device TElement* row_attention = attention + head_of_batch * geometry.capacity;

    float largest = -FLT_MAX;

    // Lanes stay at their own residue of the band, so that every key has exactly one lane.
    for ( uint j = first_key + lane; j <= position; j += threads.x )
    {
        device const TElement* key = keys + j * geometry.head_size;
        float sum = 0.0f;

        for ( uint s = 0; s < geometry.head_size; ++s )
        {
            sum += float( q[s] ) * float( key[s] );
        }

        const float score = sum * geometry.attention_scale;

        row_scores[j] = TElement( score );
        largest = max( largest, float( row_scores[j] ) );
    }

    largest = simd_max( largest );

    float total = 0.0f;

    for ( uint j = first_key + lane; j <= position; j += threads.x )
    {
        total += exp( float( row_scores[j] ) - largest );
    }

    total = simd_sum( total );

    for ( uint j = first_key + lane; j <= position; j += threads.x )
    {
        row_attention[j] = TElement( exp( float( row_scores[j] ) - largest ) / total );
    }
}

template [[host_name("gqa_decode_softmax_half")]] kernel void gqa_decode_softmax<half>(
    device const half*, device const half*, device half*, device half*, device const int*, constant GqaGeometry&, uint3, uint, uint3 );
template [[host_name("gqa_decode_softmax_bfloat")]] kernel void gqa_decode_softmax<bfloat>(
    device const bfloat*, device const bfloat*, device bfloat*, device bfloat*, device const int*, constant GqaGeometry&, uint3, uint, uint3 );

constant uint kGqaDecodeLaneOutputs = 4;

// A SIMD-group per few output channels of each head. The grid is (groups of channels, heads of every batch).
template<typename TElement>
kernel void gqa_decode_values(
    device const TElement* attention [[buffer(0)]],
    device const TElement* value_cache [[buffer(1)]],
    device TElement* output [[buffer(2)]],
    device const int* decode_position [[buffer(3)]],
    constant GqaGeometry& geometry [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 threads [[threads_per_threadgroup]] )
{
    const uint first_channel = group.x * kGqaDecodeLaneOutputs;
    const uint channels = min( kGqaDecodeLaneOutputs, geometry.head_size - first_channel );
    const uint head_of_batch = group.y;
    const uint head = head_of_batch % geometry.heads;
    const uint batch = head_of_batch / geometry.heads;
    const uint kv_head = head / ( geometry.heads / geometry.kv_heads );

    const uint position = uint( *decode_position );
    const uint first_key = geometry.window > 0 && position + 1 > geometry.window ? position + 1 - geometry.window : 0;

    device const TElement* row_attention = attention + head_of_batch * geometry.capacity;
    device const TElement* values = value_cache + ( batch * geometry.kv_heads + kv_head ) * geometry.capacity * geometry.head_size;

    float sums[kGqaDecodeLaneOutputs] = { 0.0f, 0.0f, 0.0f, 0.0f };

    for ( uint j = first_key + lane; j <= position; j += threads.x )
    {
        const float weight = float( row_attention[j] );

        for ( uint r = 0; r < channels; ++r )
        {
            sums[r] += weight * float( values[j * geometry.head_size + first_channel + r] );
        }
    }

    for ( uint r = 0; r < channels; ++r )
    {
        const float total = simd_sum( sums[r] );

        if ( lane == 0 )
        {
            output[head_of_batch * geometry.head_size + first_channel + r] = TElement( total );
        }
    }
}

template [[host_name("gqa_decode_values_half")]] kernel void gqa_decode_values<half>(
    device const half*, device const half*, device half*, device const int*, constant GqaGeometry&, uint3, uint, uint3 );
template [[host_name("gqa_decode_values_bfloat")]] kernel void gqa_decode_values<bfloat>(
    device const bfloat*, device const bfloat*, device bfloat*, device const int*, constant GqaGeometry&, uint3, uint, uint3 );
