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

    // How many blocks a decode step splits the keys into (Notebooks/MetalDecodeAttention.md).
    uint decode_blocks;
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
// The FP8 cache (PerTokenKvFp8): each row, one KV head at one token, as E4M3 codes with one FP32 scale, max |x| / 448, which
// Mila's CUDA write uses too. A row of zeros stores scale 0 and zero codes. One SIMD-group per row of K and V; needs
// QuantizedCodes.metal.
// ---------------------------------------------------------------------------------------------------------------------

template<typename TElement>
inline void gqa_write_fp8_row(
    device const TElement* source, device uchar* codes, device float* scale_slot, uint head_size, uint lane )
{
    float largest = 0.0f;

    for ( uint c = lane; c < head_size; c += 32 )
    {
        largest = fmax( largest, fabs( float( source[c] ) ) );
    }

    largest = simd_max( largest );

    const float scale = precise::divide( largest, kFp8E4M3Largest );
    const float inverse = largest > 0.0f ? precise::divide( 1.0f, scale ) : 0.0f;

    if ( lane == 0 )
    {
        *scale_slot = scale;
    }

    for ( uint c = lane; c < head_size; c += 32 )
    {
        codes[c] = fp8E4M3Code( float( source[c] ) * inverse );
    }
}

// Row `row` of K and V's [B, tokens, NKV] rows goes to the cache at its absolute position.
template<typename TElement>
inline void gqa_write_fp8_rows(
    device const TElement* key, device const TElement* value, device uchar* key_codes, device uchar* value_codes,
    device float* key_scales, device float* value_scales, constant GqaGeometry& geometry, uint first_position, uint row, uint lane )
{
    const uint kv_head = row % geometry.kv_heads;
    const uint token_of_batch = row / geometry.kv_heads;
    const uint token = token_of_batch % geometry.tokens;
    const uint batch = token_of_batch / geometry.tokens;

    const uint cached = ( batch * geometry.kv_heads + kv_head ) * geometry.capacity + first_position + token;

    gqa_write_fp8_row( key + row * geometry.head_size, key_codes + cached * geometry.head_size, key_scales + cached, geometry.head_size, lane );
    gqa_write_fp8_row( value + row * geometry.head_size, value_codes + cached * geometry.head_size, value_scales + cached, geometry.head_size, lane );
}

template<typename TElement>
kernel void gqa_write_fp8_prefill(
    device const TElement* key [[buffer(0)]],
    device const TElement* value [[buffer(1)]],
    device uchar* key_codes [[buffer(2)]],
    device uchar* value_codes [[buffer(3)]],
    device float* key_scales [[buffer(4)]],
    device float* value_scales [[buffer(5)]],
    constant GqaGeometry& geometry [[buffer(6)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]] )
{
    gqa_write_fp8_rows<TElement>( key, value, key_codes, value_codes, key_scales, value_scales, geometry, geometry.position_offset, group.x, lane );
}

template<typename TElement>
kernel void gqa_write_fp8_decode(
    device const TElement* key [[buffer(0)]],
    device const TElement* value [[buffer(1)]],
    device uchar* key_codes [[buffer(2)]],
    device uchar* value_codes [[buffer(3)]],
    device float* key_scales [[buffer(4)]],
    device float* value_scales [[buffer(5)]],
    device const int* decode_position [[buffer(6)]],
    constant GqaGeometry& geometry [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]] )
{
    gqa_write_fp8_rows<TElement>( key, value, key_codes, value_codes, key_scales, value_scales, geometry, uint( *decode_position ), group.x, lane );
}

template [[host_name("gqa_write_fp8_prefill_half")]] kernel void gqa_write_fp8_prefill<half>(
    device const half*, device const half*, device uchar*, device uchar*, device float*, device float*, constant GqaGeometry&, uint3, uint );
template [[host_name("gqa_write_fp8_prefill_bfloat")]] kernel void gqa_write_fp8_prefill<bfloat>(
    device const bfloat*, device const bfloat*, device uchar*, device uchar*, device float*, device float*, constant GqaGeometry&, uint3, uint );
template [[host_name("gqa_write_fp8_decode_half")]] kernel void gqa_write_fp8_decode<half>(
    device const half*, device const half*, device uchar*, device uchar*, device float*, device float*, device const int*, constant GqaGeometry&, uint3, uint );
template [[host_name("gqa_write_fp8_decode_bfloat")]] kernel void gqa_write_fp8_decode<bfloat>(
    device const bfloat*, device const bfloat*, device uchar*, device uchar*, device float*, device float*, device const int*, constant GqaGeometry&, uint3, uint );

// One KV head's first `rows` cached rows of K or V, as the compute type, for prefill's products. The grid is every element.
template<typename TElement>
kernel void gqa_expand_fp8_rows(
    device const uchar* codes [[buffer(0)]],
    device const float* scales [[buffer(1)]],
    device TElement* expanded [[buffer(2)]],
    constant GqaGeometry& geometry [[buffer(3)]],
    uint index [[thread_position_in_grid]] )
{
    expanded[index] = TElement( kFp8E4M3Values[codes[index]] * scales[index / geometry.head_size] );
}

template [[host_name("gqa_expand_fp8_rows_half")]] kernel void gqa_expand_fp8_rows<half>(
    device const uchar*, device const float*, device half*, constant GqaGeometry&, uint );
template [[host_name("gqa_expand_fp8_rows_bfloat")]] kernel void gqa_expand_fp8_rows<bfloat>(
    device const uchar*, device const float*, device bfloat*, constant GqaGeometry&, uint );

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
// Decode, one token, in two passes (Notebooks/MetalDecodeAttention.md). The partial pass splits the keys over a fixed grid of blocks,
// a threadgroup taking one KV head's query heads, a SIMD-group each, so that they read the same rows together; block b takes keys
// b, b + blocks, ... up to the position, with the softmax kept online. The merge rescales each block's output to the largest score of
// all. The shape is MLX's two-pass decode (ml-explore/mlx, sdpa_vector.h) and the split with its merge Flash-Decoding's (Dao,
// Haziza, Massa, Sizov, 2023). The partials are FP32 in the context's scratch: output [B * NH, blocks, HS], then largest and sum
// [B * NH, blocks].
// ---------------------------------------------------------------------------------------------------------------------

// Head sizes the models use: a lane's channels are a contiguous run of HS / 32, which is what reaches the floor.
template<uint HS>
struct GqaContiguousChannels
{
    uint lane;

    uint channel( uint e ) const
    {
        return lane * ( HS / 32 ) + e;
    }

    bool present( uint ) const
    {
        return true;
    }
};

// Any other head size up to 512: lane l takes channels l, l + 32, ... About three times slower than a contiguous run at the same size
// (the notebook's section 3), and used by no model Mila runs.
struct GqaInterleavedChannels
{
    uint lane;
    uint head_size;

    uint channel( uint e ) const
    {
        return e * 32 + lane;
    }

    bool present( uint e ) const
    {
        return e * 32 + lane < head_size;
    }
};

// A cache in the compute type: each row as stored.
template<typename TElement>
struct GqaCacheRows
{
    device const TElement* keys;
    device const TElement* values;

    GqaCacheRows atKvHead( uint kv_head_of_batch, uint rows_per_head, uint head_size ) const
    {
        return GqaCacheRows{ keys + kv_head_of_batch * rows_per_head * head_size, values + kv_head_of_batch * rows_per_head * head_size };
    }

    float key( uint row, uint head_size, uint channel ) const { return float( keys[row * head_size + channel] ); }
    float value( uint row, uint head_size, uint channel ) const { return float( values[row * head_size + channel] ); }
    float keyScale( uint ) const { return 1.0f; }
    float valueScale( uint ) const { return 1.0f; }
};

// An FP8 cache: each row's codes, and its scale, which comes out of the row's score and multiplies its weight.
struct GqaFp8CacheRows
{
    device const uchar* keys;
    device const uchar* values;
    device const float* key_scales;
    device const float* value_scales;

    GqaFp8CacheRows atKvHead( uint kv_head_of_batch, uint rows_per_head, uint head_size ) const
    {
        const uint first_row = kv_head_of_batch * rows_per_head;

        return GqaFp8CacheRows{ keys + first_row * head_size, values + first_row * head_size, key_scales + first_row, value_scales + first_row };
    }

    float key( uint row, uint head_size, uint channel ) const { return kFp8E4M3Values[keys[row * head_size + channel]]; }
    float value( uint row, uint head_size, uint channel ) const { return kFp8E4M3Values[values[row * head_size + channel]]; }
    float keyScale( uint row ) const { return key_scales[row]; }
    float valueScale( uint row ) const { return value_scales[row]; }
};

// One block's keys for one query head, the softmax online: the largest score so far, the sum, the output rescaled as it grows.
template<uint kPerLane, typename TCache, typename TElement, typename TChannels>
inline void gqa_decode_block(
    device const TElement* query, TCache cache,
    device float* partial_output, device float* partial_largest, device float* partial_sum, device const int* decode_position,
    constant GqaGeometry& geometry, uint2 group, uint simd, uint lane, TChannels channels )
{
    const uint kv_head_of_batch = group.y;
    const uint block = group.x;
    const uint batch = kv_head_of_batch / geometry.kv_heads;
    const uint kv_head = kv_head_of_batch % geometry.kv_heads;
    const uint head_of_batch = batch * geometry.heads + kv_head * ( geometry.heads / geometry.kv_heads ) + simd;
    const uint head_size = geometry.head_size;

    const TCache rows = cache.atKvHead( kv_head_of_batch, geometry.capacity, head_size );

    const uint position = uint( *decode_position );
    const uint first_key = geometry.window > 0 && position + 1 > geometry.window ? position + 1 - geometry.window : 0;

    float q[kPerLane];
    float out[kPerLane];

    for ( uint e = 0; e < kPerLane; ++e )
    {
        q[e] = channels.present( e ) ? float( query[head_of_batch * head_size + channels.channel( e )] ) * geometry.attention_scale : 0.0f;
        out[e] = 0.0f;
    }

    float largest = -FLT_MAX;
    float sum = 0.0f;

    for ( uint j = first_key + block; j <= position; j += geometry.decode_blocks )
    {
        float score = 0.0f;

        for ( uint e = 0; e < kPerLane; ++e )
        {
            score += channels.present( e ) ? q[e] * rows.key( j, head_size, channels.channel( e ) ) : 0.0f;
        }

        score = simd_sum( score ) * rows.keyScale( j );

        const float next = max( largest, score );
        const float rescale = exp( largest - next );
        const float weight = exp( score - next );

        sum = sum * rescale + weight;

        const float value_weight = weight * rows.valueScale( j );

        for ( uint e = 0; e < kPerLane; ++e )
        {
            out[e] = out[e] * rescale + ( channels.present( e ) ? value_weight * rows.value( j, head_size, channels.channel( e ) ) : 0.0f );
        }

        largest = next;
    }

    // A block with no keys writes a sum of zero, and drops out of the merge.
    const uint slot = head_of_batch * geometry.decode_blocks + block;

    for ( uint e = 0; e < kPerLane; ++e )
    {
        if ( channels.present( e ) )
        {
            partial_output[slot * head_size + channels.channel( e )] = out[e];
        }
    }

    if ( lane == 0 )
    {
        partial_largest[slot] = largest;
        partial_sum[slot] = sum;
    }
}

// The grid is (blocks, KV heads of every batch); a threadgroup holds one SIMD-group per query head of its KV head.
template<typename TElement, uint HS>
kernel void gqa_decode_partial(
    device const TElement* query [[buffer(0)]],
    device const TElement* key_cache [[buffer(1)]],
    device const TElement* value_cache [[buffer(2)]],
    device float* partial_output [[buffer(3)]],
    device float* partial_largest [[buffer(4)]],
    device float* partial_sum [[buffer(5)]],
    device const int* decode_position [[buffer(6)]],
    constant GqaGeometry& geometry [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]] )
{
    gqa_decode_block<HS / 32>( query, GqaCacheRows<TElement>{ key_cache, value_cache }, partial_output, partial_largest, partial_sum,
        decode_position, geometry, group, simd, lane, GqaContiguousChannels<HS>{ lane } );
}

template<typename TElement>
kernel void gqa_decode_partial_any(
    device const TElement* query [[buffer(0)]],
    device const TElement* key_cache [[buffer(1)]],
    device const TElement* value_cache [[buffer(2)]],
    device float* partial_output [[buffer(3)]],
    device float* partial_largest [[buffer(4)]],
    device float* partial_sum [[buffer(5)]],
    device const int* decode_position [[buffer(6)]],
    constant GqaGeometry& geometry [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]] )
{
    gqa_decode_block<16>( query, GqaCacheRows<TElement>{ key_cache, value_cache }, partial_output, partial_largest, partial_sum,
        decode_position, geometry, group, simd, lane, GqaInterleavedChannels{ lane, geometry.head_size } );
}

// The same pass over an FP8 cache: codes and each row's scale, for K and for V.
template<typename TElement, uint HS>
kernel void gqa_decode_partial_fp8(
    device const TElement* query [[buffer(0)]],
    device const uchar* key_codes [[buffer(1)]],
    device const uchar* value_codes [[buffer(2)]],
    device const float* key_scales [[buffer(3)]],
    device const float* value_scales [[buffer(4)]],
    device float* partial_output [[buffer(5)]],
    device float* partial_largest [[buffer(6)]],
    device float* partial_sum [[buffer(7)]],
    device const int* decode_position [[buffer(8)]],
    constant GqaGeometry& geometry [[buffer(9)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]] )
{
    gqa_decode_block<HS / 32>( query, GqaFp8CacheRows{ key_codes, value_codes, key_scales, value_scales }, partial_output, partial_largest,
        partial_sum, decode_position, geometry, group, simd, lane, GqaContiguousChannels<HS>{ lane } );
}

template<typename TElement>
kernel void gqa_decode_partial_fp8_any(
    device const TElement* query [[buffer(0)]],
    device const uchar* key_codes [[buffer(1)]],
    device const uchar* value_codes [[buffer(2)]],
    device const float* key_scales [[buffer(3)]],
    device const float* value_scales [[buffer(4)]],
    device float* partial_output [[buffer(5)]],
    device float* partial_largest [[buffer(6)]],
    device float* partial_sum [[buffer(7)]],
    device const int* decode_position [[buffer(8)]],
    constant GqaGeometry& geometry [[buffer(9)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]] )
{
    gqa_decode_block<16>( query, GqaFp8CacheRows{ key_codes, value_codes, key_scales, value_scales }, partial_output, partial_largest,
        partial_sum, decode_position, geometry, group, simd, lane, GqaInterleavedChannels{ lane, geometry.head_size } );
}

#define MILA_GQA_DECODE_PARTIAL_FP8( element, size ) \
    template [[host_name( "gqa_decode_partial_fp8_" #element "_" #size )]] kernel void gqa_decode_partial_fp8<element, size>( \
        device const element*, device const uchar*, device const uchar*, device const float*, device const float*, device float*, \
        device float*, device float*, device const int*, constant GqaGeometry&, uint2, uint, uint );

MILA_GQA_DECODE_PARTIAL_FP8( half, 64 )
MILA_GQA_DECODE_PARTIAL_FP8( half, 128 )
MILA_GQA_DECODE_PARTIAL_FP8( half, 256 )
MILA_GQA_DECODE_PARTIAL_FP8( half, 512 )
MILA_GQA_DECODE_PARTIAL_FP8( bfloat, 64 )
MILA_GQA_DECODE_PARTIAL_FP8( bfloat, 128 )
MILA_GQA_DECODE_PARTIAL_FP8( bfloat, 256 )
MILA_GQA_DECODE_PARTIAL_FP8( bfloat, 512 )

template [[host_name( "gqa_decode_partial_fp8_half_any" )]] kernel void gqa_decode_partial_fp8_any<half>(
    device const half*, device const uchar*, device const uchar*, device const float*, device const float*, device float*, device float*,
    device float*, device const int*, constant GqaGeometry&, uint2, uint, uint );
template [[host_name( "gqa_decode_partial_fp8_bfloat_any" )]] kernel void gqa_decode_partial_fp8_any<bfloat>(
    device const bfloat*, device const uchar*, device const uchar*, device const float*, device const float*, device float*, device float*,
    device float*, device const int*, constant GqaGeometry&, uint2, uint, uint );

// One SIMD-group per query head of each batch: the largest score of the blocks, then each block's output and sum rescaled to it.
template<typename TElement, uint kPerLane, typename TChannels>
inline void gqa_decode_merge_head(
    device const float* partial_output, device const float* partial_largest, device const float* partial_sum,
    device TElement* output, constant GqaGeometry& geometry, uint head_of_batch, uint lane, TChannels channels )
{
    const uint first_slot = head_of_batch * geometry.decode_blocks;

    float largest = -FLT_MAX;

    for ( uint b = lane; b < geometry.decode_blocks; b += 32 )
    {
        largest = max( largest, partial_largest[first_slot + b] );
    }

    largest = simd_max( largest );

    float total = 0.0f;
    float out[kPerLane];

    for ( uint e = 0; e < kPerLane; ++e )
    {
        out[e] = 0.0f;
    }

    for ( uint b = 0; b < geometry.decode_blocks; ++b )
    {
        const float scale = exp( partial_largest[first_slot + b] - largest );

        total += scale * partial_sum[first_slot + b];

        for ( uint e = 0; e < kPerLane; ++e )
        {
            out[e] += channels.present( e ) ? scale * partial_output[( first_slot + b ) * geometry.head_size + channels.channel( e )] : 0.0f;
        }
    }

    for ( uint e = 0; e < kPerLane; ++e )
    {
        if ( channels.present( e ) )
        {
            output[head_of_batch * geometry.head_size + channels.channel( e )] = TElement( out[e] / total );
        }
    }
}

template<typename TElement, uint HS>
kernel void gqa_decode_merge(
    device const float* partial_output [[buffer(0)]],
    device const float* partial_largest [[buffer(1)]],
    device const float* partial_sum [[buffer(2)]],
    device TElement* output [[buffer(3)]],
    constant GqaGeometry& geometry [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]] )
{
    gqa_decode_merge_head<TElement, HS / 32>( partial_output, partial_largest, partial_sum, output, geometry, group.x, lane,
        GqaContiguousChannels<HS>{ lane } );
}

template<typename TElement>
kernel void gqa_decode_merge_any(
    device const float* partial_output [[buffer(0)]],
    device const float* partial_largest [[buffer(1)]],
    device const float* partial_sum [[buffer(2)]],
    device TElement* output [[buffer(3)]],
    constant GqaGeometry& geometry [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]] )
{
    gqa_decode_merge_head<TElement, 16>( partial_output, partial_largest, partial_sum, output, geometry, group.x, lane,
        GqaInterleavedChannels{ lane, geometry.head_size } );
}

template [[host_name("gqa_decode_partial_half_64")]] kernel void gqa_decode_partial<half, 64>(
    device const half*, device const half*, device const half*, device float*, device float*, device float*, device const int*, constant GqaGeometry&, uint2, uint, uint );
template [[host_name("gqa_decode_partial_half_128")]] kernel void gqa_decode_partial<half, 128>(
    device const half*, device const half*, device const half*, device float*, device float*, device float*, device const int*, constant GqaGeometry&, uint2, uint, uint );
template [[host_name("gqa_decode_partial_half_256")]] kernel void gqa_decode_partial<half, 256>(
    device const half*, device const half*, device const half*, device float*, device float*, device float*, device const int*, constant GqaGeometry&, uint2, uint, uint );
template [[host_name("gqa_decode_partial_half_512")]] kernel void gqa_decode_partial<half, 512>(
    device const half*, device const half*, device const half*, device float*, device float*, device float*, device const int*, constant GqaGeometry&, uint2, uint, uint );
template [[host_name("gqa_decode_partial_bfloat_64")]] kernel void gqa_decode_partial<bfloat, 64>(
    device const bfloat*, device const bfloat*, device const bfloat*, device float*, device float*, device float*, device const int*, constant GqaGeometry&, uint2, uint, uint );
template [[host_name("gqa_decode_partial_bfloat_128")]] kernel void gqa_decode_partial<bfloat, 128>(
    device const bfloat*, device const bfloat*, device const bfloat*, device float*, device float*, device float*, device const int*, constant GqaGeometry&, uint2, uint, uint );
template [[host_name("gqa_decode_partial_bfloat_256")]] kernel void gqa_decode_partial<bfloat, 256>(
    device const bfloat*, device const bfloat*, device const bfloat*, device float*, device float*, device float*, device const int*, constant GqaGeometry&, uint2, uint, uint );
template [[host_name("gqa_decode_partial_bfloat_512")]] kernel void gqa_decode_partial<bfloat, 512>(
    device const bfloat*, device const bfloat*, device const bfloat*, device float*, device float*, device float*, device const int*, constant GqaGeometry&, uint2, uint, uint );
template [[host_name("gqa_decode_partial_half_any")]] kernel void gqa_decode_partial_any<half>(
    device const half*, device const half*, device const half*, device float*, device float*, device float*, device const int*, constant GqaGeometry&, uint2, uint, uint );
template [[host_name("gqa_decode_partial_bfloat_any")]] kernel void gqa_decode_partial_any<bfloat>(
    device const bfloat*, device const bfloat*, device const bfloat*, device float*, device float*, device float*, device const int*, constant GqaGeometry&, uint2, uint, uint );

template [[host_name("gqa_decode_merge_half_64")]] kernel void gqa_decode_merge<half, 64>(
    device const float*, device const float*, device const float*, device half*, constant GqaGeometry&, uint3, uint );
template [[host_name("gqa_decode_merge_half_128")]] kernel void gqa_decode_merge<half, 128>(
    device const float*, device const float*, device const float*, device half*, constant GqaGeometry&, uint3, uint );
template [[host_name("gqa_decode_merge_half_256")]] kernel void gqa_decode_merge<half, 256>(
    device const float*, device const float*, device const float*, device half*, constant GqaGeometry&, uint3, uint );
template [[host_name("gqa_decode_merge_half_512")]] kernel void gqa_decode_merge<half, 512>(
    device const float*, device const float*, device const float*, device half*, constant GqaGeometry&, uint3, uint );
template [[host_name("gqa_decode_merge_bfloat_64")]] kernel void gqa_decode_merge<bfloat, 64>(
    device const float*, device const float*, device const float*, device bfloat*, constant GqaGeometry&, uint3, uint );
template [[host_name("gqa_decode_merge_bfloat_128")]] kernel void gqa_decode_merge<bfloat, 128>(
    device const float*, device const float*, device const float*, device bfloat*, constant GqaGeometry&, uint3, uint );
template [[host_name("gqa_decode_merge_bfloat_256")]] kernel void gqa_decode_merge<bfloat, 256>(
    device const float*, device const float*, device const float*, device bfloat*, constant GqaGeometry&, uint3, uint );
template [[host_name("gqa_decode_merge_bfloat_512")]] kernel void gqa_decode_merge<bfloat, 512>(
    device const float*, device const float*, device const float*, device bfloat*, constant GqaGeometry&, uint3, uint );
template [[host_name("gqa_decode_merge_half_any")]] kernel void gqa_decode_merge_any<half>(
    device const float*, device const float*, device const float*, device half*, constant GqaGeometry&, uint3, uint );
template [[host_name("gqa_decode_merge_bfloat_any")]] kernel void gqa_decode_merge_any<bfloat>(
    device const float*, device const float*, device const float*, device bfloat*, constant GqaGeometry&, uint3, uint );
