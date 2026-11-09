// Fused prefill attention (MetalBackend.md section 4.5): Q x K^T, the causal softmax and P x V in one kernel, the scores
// never leaving the threadgroup. Follows GroupedQueryAttention.metal, whose GqaGeometry and gqa_position_of_row it uses.
//
// A threadgroup takes one KV head and a block of query positions; each SIMD-group takes one query head of that KV head's
// group over TRows of those positions, so every head of the group reads the same keys and values together. Per key tile,
// a SIMD-group runs Q x K^T as a TensorOp into FP32, scales and masks it, keeps the softmax online (FlashAttention-2,
// Dao, arXiv 2307.08691) in base 2 with the scale carrying log2(e), as MLX's steel_attention.h (ml-explore/mlx v0.32.4)
// does, and adds P x V into an FP32 output. P is rounded to the compute type for that product. Only TensorOps and the
// cooperative tensor's documented accessors are used: the Metal Shading Language leaves the SIMD-group matrix's layout
// unspecified.
//
// Every product is 128 channels wide or less: Q x K^T sums a head over 128-channel slices, and P x V writes 128 output
// columns a product. Measured on the M1 Pro: one TensorOp at execution_simdgroup over a 256- or 512-channel head returned
// wrong values with no error, where 128 returned the reference's (MetalBackend.md section 4.5).
//
// A tile always reads TKeys whole cache rows: the last tile of a range shorter than that starts earlier and masks the rows
// already taken, so a read never passes the cache. Rows not yet written are zero, since the operation clears its cache
// when built, and the masked rows multiply as zeros.

struct GqaFlashConstants
{
    GqaGeometry geometry;

    // Elements between one KV head's cache rows and the next's in this dispatch's K and V; zero when it takes one head.
    uint kv_stride;

    // batch * kv_heads + kv_head of the grid's first row.
    uint first_kv_head;

    // The query positions a threadgroup takes.
    uint positions;

    // The query heads of the group a threadgroup takes, and how many threadgroups share a group: a threadgroup holds every head of a
    // small group, and a part of a large one where the kernel's threads would pass what the device gives a threadgroup.
    uint heads_per_threadgroup;
    uint head_splits;
};

// The widest product, in channels.
constant constexpr int kGqaFlashSlice = 128;

// A key no query may attend, after scaling. Fast math assumes no infinities.
constant constexpr float kGqaFlashMasked = -FLT_MAX;

// Each SIMD-group's part of threadgroup memory; a threadgroup holds at most TSimdgroups of them, which the host keeps to. Keys and
// values are read from the cache by each SIMD-group: copying a tile to threadgroup memory once for all of them was measured slower
// on the M1 Pro, where the tile then has to be smaller (MetalBackend.md section 4.5).
template<typename TElement, int TRows, int TKeys, int TSimdgroups>
struct GqaFlashShared
{
    TElement probabilities[TSimdgroups][TRows * TKeys];
    float largest[TSimdgroups][TRows];
    float sum[TSimdgroups][TRows];
    float tile_row[TSimdgroups][TRows];
    float factor[TSimdgroups][TRows];
};

template<typename TCooperative>
inline void gqa_flash_clear( thread TCooperative& tensor )
{
    for ( uint16_t i = 0; i < tensor.get_capacity(); ++i )
    {
        if ( tensor.is_valid_element( i ) )
        {
            tensor[i] = 0.0f;
        }
    }
}

// Each element of an output slice times its row's factor.
template<typename TCooperative>
inline void gqa_flash_rescale( thread TCooperative& tensor, threadgroup const float* factor )
{
    for ( uint16_t i = 0; i < tensor.get_capacity(); ++i )
    {
        if ( tensor.is_valid_element( i ) )
        {
            tensor[i] *= factor[tensor.get_multidimensional_index( i )[1]];
        }
    }
}

// An output slice's rows that hold queries, divided by their sums, at `first_column` of each output row.
template<typename TElement, typename TCooperative>
inline void gqa_flash_store( thread TCooperative& tensor, threadgroup const float* sum, device TElement* output, uint row_stride,
    uint rows, uint first_column )
{
    for ( uint16_t i = 0; i < tensor.get_capacity(); ++i )
    {
        if ( !tensor.is_valid_element( i ) )
        {
            continue;
        }

        const auto index = tensor.get_multidimensional_index( i );
        const uint row = uint( index[1] );

        if ( row < rows )
        {
            output[row * row_stride + first_column + uint( index[0] )] = TElement( tensor[i] / sum[row] );
        }
    }
}

template<typename TElement, int TRows, int TKeys, int THeadSize, int TSimdgroups>
kernel void gqa_flash_prefill(
    device const TElement* query [[buffer(0)]],
    device const TElement* keys [[buffer(1)]],
    device const TElement* values [[buffer(2)]],
    device TElement* output [[buffer(3)]],
    constant GqaFlashConstants& constants [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]] )
{
    static_assert( THeadSize % kGqaFlashSlice == 0 && THeadSize / kGqaFlashSlice <= 4, "a head of 128, 256, 384 or 512 channels" );

    constexpr int kSlices = THeadSize / kGqaFlashSlice;

    threadgroup GqaFlashShared<TElement, TRows, TKeys, TSimdgroups> shared;

    constant GqaGeometry& geometry = constants.geometry;

    const uint group_size = geometry.heads / geometry.kv_heads;
    const uint kv_head_of_batch = constants.first_kv_head + group.y;
    const uint batch = kv_head_of_batch / geometry.kv_heads;
    const uint split = group.x % constants.head_splits;
    const uint head = ( kv_head_of_batch % geometry.kv_heads ) * group_size + split * constants.heads_per_threadgroup
        + simdgroup % constants.heads_per_threadgroup;

    const uint group_first_token = ( group.x / constants.head_splits ) * constants.positions;
    const uint first_token = group_first_token + ( simdgroup / constants.heads_per_threadgroup ) * TRows;

    if ( first_token >= geometry.tokens )
    {
        return;
    }

    const uint rows = min( uint( TRows ), geometry.tokens - first_token );
    const uint group_last_token = min( group_first_token + constants.positions, geometry.tokens ) - 1;

    // The cache rows any query of the threadgroup attends: a contiguous run while no row has been reused, the whole ring after.
    const uint attended = geometry.position_offset + geometry.tokens;
    const uint last = attended - 1;
    const bool wrapped = attended > geometry.capacity;
    const uint first_position = geometry.position_offset + group_first_token;
    const uint window_start = geometry.window > 0 && first_position + 1 > geometry.window ? first_position + 1 - geometry.window : 0;
    const uint low = wrapped ? 0 : window_start;
    const uint high = wrapped ? geometry.capacity : geometry.position_offset + group_last_token + 1;

    const float scale = geometry.attention_scale * M_LOG2E_F;
    const uint query_stride = geometry.heads * THeadSize;

    device TElement* const query_rows = const_cast<device TElement*>( query )
        + ( ( batch * geometry.tokens + first_token ) * geometry.heads + head ) * THeadSize;
    device TElement* const key_rows = const_cast<device TElement*>( keys ) + group.y * constants.kv_stride;
    device TElement* const value_rows = const_cast<device TElement*>( values ) + group.y * constants.kv_stride;

    using SliceTensor = tensor<device TElement, dextents<int, 2>, tensor_inline>;

    threadgroup TElement* const stage = shared.probabilities[simdgroup];

    tensor<threadgroup TElement, dextents<int, 2>, tensor_inline> p(
        stage, dextents<int, 2>( TKeys, TRows ), array<int, 2>{ 1, TKeys } );

    constexpr auto score_descriptor = mpp::tensor_ops::matmul2d_descriptor( TRows, TKeys, kGqaFlashSlice, false, true, false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate );
    constexpr auto value_descriptor = mpp::tensor_ops::matmul2d_descriptor( TRows, kGqaFlashSlice, TKeys, false, false, false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate );

    mpp::tensor_ops::matmul2d<score_descriptor, execution_simdgroup> score_product;
    mpp::tensor_ops::matmul2d<value_descriptor, execution_simdgroup> value_product;

    // One output slice of 128 columns each; a cooperative tensor cannot be held in an array.
    auto out0 = value_product.template get_destination_cooperative_tensor<decltype( p ), SliceTensor, float>();
    auto out1 = value_product.template get_destination_cooperative_tensor<decltype( p ), SliceTensor, float>();
    auto out2 = value_product.template get_destination_cooperative_tensor<decltype( p ), SliceTensor, float>();
    auto out3 = value_product.template get_destination_cooperative_tensor<decltype( p ), SliceTensor, float>();

    gqa_flash_clear( out0 );

    if constexpr ( kSlices > 1 ) gqa_flash_clear( out1 );
    if constexpr ( kSlices > 2 ) gqa_flash_clear( out2 );
    if constexpr ( kSlices > 3 ) gqa_flash_clear( out3 );

    if ( lane < uint( TRows ) )
    {
        shared.largest[simdgroup][lane] = kGqaFlashMasked;
        shared.sum[simdgroup][lane] = 0.0f;
    }

    simdgroup_barrier( mem_flags::mem_threadgroup );

    for ( uint tile = ( low / TKeys ) * TKeys; tile < high; tile += TKeys )
    {
        const uint read = tile + TKeys > high ? ( high >= uint( TKeys ) ? high - TKeys : 0 ) : tile;

        device TElement* const tile_keys = key_rows + read * THeadSize;
        device TElement* const tile_values = value_rows + read * THeadSize;

        auto scores = score_product.template get_destination_cooperative_tensor<SliceTensor, SliceTensor, float>();

        gqa_flash_clear( scores );

        for ( int s = 0; s < kSlices; ++s )
        {
            SliceTensor q( query_rows + s * kGqaFlashSlice, dextents<int, 2>( kGqaFlashSlice, int( rows ) ), array<int, 2>{ 1, int( query_stride ) } );
            SliceTensor k( tile_keys + s * kGqaFlashSlice, dextents<int, 2>( kGqaFlashSlice, TKeys ), array<int, 2>{ 1, THeadSize } );

            score_product.run( q, k, scores );
        }

        // A tile every row of this SIMD-group attends whole needs no mask: no row of the ring reused, no row taken twice, every key at or
        // before the first row's position and inside the last row's window. After a held prompt, that is nearly every tile.
        const uint first_row_position = geometry.position_offset + first_token;
        const uint last_row_position = first_row_position + rows - 1;
        const uint last_row_first_key = geometry.window > 0 && last_row_position + 1 > geometry.window ? last_row_position + 1 - geometry.window : 0;
        const bool whole = !wrapped && read == tile && rows == uint( TRows ) && tile + TKeys - 1 <= first_row_position
            && tile >= last_row_first_key;

        if ( !whole )
        {
            for ( uint16_t i = 0; i < scores.get_capacity(); ++i )
            {
                if ( !scores.is_valid_element( i ) )
                {
                    continue;
                }

                const auto index = scores.get_multidimensional_index( i );
                const uint column = read + uint( index[0] );
                const uint row = uint( index[1] );
                const uint position = first_row_position + row;
                const uint first_key = geometry.window > 0 && position + 1 > geometry.window ? position + 1 - geometry.window : 0;
                const uint key = wrapped ? gqa_position_of_row( column, last, geometry.capacity ) : column;

                const bool attends = row < rows && column >= tile && column < high && key <= position && key >= first_key;

                if ( !attends )
                {
                    scores[i] = kGqaFlashMasked;
                }
            }
        }

        // The largest is taken before scaling, which is monotone, so the scale joins the exponent below.
        auto tile_largest = score_product.template get_row_reduction_destination_cooperative_tensor<SliceTensor, SliceTensor, float>();

        reduce_rows( scores, tile_largest, mpp::tensor_ops::reduction_operation::max, kGqaFlashMasked );

        for ( uint16_t i = 0; i < tile_largest.get_capacity(); ++i )
        {
            if ( tile_largest.is_valid_element( i ) )
            {
                shared.tile_row[simdgroup][tile_largest.get_multidimensional_index( i )[0]] = tile_largest[i];
            }
        }

        simdgroup_barrier( mem_flags::mem_threadgroup );

        bool rescaled = false;

        if ( lane < uint( TRows ) )
        {
            const float before = shared.largest[simdgroup][lane];
            const float tile_row = shared.tile_row[simdgroup][lane];
            const float largest = max( before, tile_row == kGqaFlashMasked ? kGqaFlashMasked : tile_row * scale );

            // Nothing held yet scales by zero, and a difference of two masked values is never formed.
            const float factor = before == kGqaFlashMasked ? 0.0f : exp2( before - largest );

            shared.factor[simdgroup][lane] = factor;
            shared.largest[simdgroup][lane] = largest;
            rescaled = factor != 1.0f;
        }

        // The output is rescaled only when some row's largest moved, which grows rare as the keys go by.
        const bool any_rescaled = simd_any( rescaled );

        simdgroup_barrier( mem_flags::mem_threadgroup );

        for ( uint16_t i = 0; i < scores.get_capacity(); ++i )
        {
            if ( !scores.is_valid_element( i ) )
            {
                continue;
            }

            const auto index = scores.get_multidimensional_index( i );
            const uint row = uint( index[1] );
            const float weight = scores[i] == kGqaFlashMasked ? 0.0f : exp2( fma( scores[i], scale, -shared.largest[simdgroup][row] ) );

            scores[i] = weight;
            stage[row * TKeys + uint( index[0] )] = TElement( weight );
        }

        auto tile_sum = score_product.template get_row_reduction_destination_cooperative_tensor<SliceTensor, SliceTensor, float>();

        reduce_rows( scores, tile_sum, mpp::tensor_ops::reduction_operation::sum, 0.0f );

        for ( uint16_t i = 0; i < tile_sum.get_capacity(); ++i )
        {
            if ( tile_sum.is_valid_element( i ) )
            {
                shared.tile_row[simdgroup][tile_sum.get_multidimensional_index( i )[0]] = tile_sum[i];
            }
        }

        simdgroup_barrier( mem_flags::mem_threadgroup );

        if ( lane < uint( TRows ) )
        {
            shared.sum[simdgroup][lane] = shared.sum[simdgroup][lane] * shared.factor[simdgroup][lane] + shared.tile_row[simdgroup][lane];
        }

        threadgroup const float* const factor = shared.factor[simdgroup];

        SliceTensor v0( tile_values, dextents<int, 2>( kGqaFlashSlice, TKeys ), array<int, 2>{ 1, THeadSize } );

        if ( any_rescaled ) gqa_flash_rescale( out0, factor );
        value_product.run( p, v0, out0 );

        if constexpr ( kSlices > 1 )
        {
            SliceTensor v1( tile_values + kGqaFlashSlice, dextents<int, 2>( kGqaFlashSlice, TKeys ), array<int, 2>{ 1, THeadSize } );

            if ( any_rescaled ) gqa_flash_rescale( out1, factor );
            value_product.run( p, v1, out1 );
        }

        if constexpr ( kSlices > 2 )
        {
            SliceTensor v2( tile_values + 2 * kGqaFlashSlice, dextents<int, 2>( kGqaFlashSlice, TKeys ), array<int, 2>{ 1, THeadSize } );

            if ( any_rescaled ) gqa_flash_rescale( out2, factor );
            value_product.run( p, v2, out2 );
        }

        if constexpr ( kSlices > 3 )
        {
            SliceTensor v3( tile_values + 3 * kGqaFlashSlice, dextents<int, 2>( kGqaFlashSlice, TKeys ), array<int, 2>{ 1, THeadSize } );

            if ( any_rescaled ) gqa_flash_rescale( out3, factor );
            value_product.run( p, v3, out3 );
        }

        simdgroup_barrier( mem_flags::mem_threadgroup );
    }

    threadgroup const float* const sum = shared.sum[simdgroup];
    device TElement* const output_rows = output + ( ( batch * geometry.tokens + first_token ) * geometry.heads + head ) * THeadSize;

    gqa_flash_store( out0, sum, output_rows, query_stride, rows, 0 );

    if constexpr ( kSlices > 1 ) gqa_flash_store( out1, sum, output_rows, query_stride, rows, uint( kGqaFlashSlice ) );
    if constexpr ( kSlices > 2 ) gqa_flash_store( out2, sum, output_rows, query_stride, rows, uint( 2 * kGqaFlashSlice ) );
    if constexpr ( kSlices > 3 ) gqa_flash_store( out3, sum, output_rows, query_stride, rows, uint( 3 * kGqaFlashSlice ) );
}

// Query rows a SIMD-group takes, cache rows a tile reads, and the most SIMD-groups a threadgroup holds, by head size: MetalGqaOp's
// flashTile() states the same.
#define MILA_GQA_FLASH_PREFILL( element, rows, keys, size, simdgroups )                                                           \
    template [[host_name( "gqa_flash_prefill_" #element "_" #size )]] kernel void                                                \
    gqa_flash_prefill<element, rows, keys, size, simdgroups>(                                                                    \
        device const element*, device const element*, device const element*, device element*, constant GqaFlashConstants&,    \
        uint2, uint, uint );

MILA_GQA_FLASH_PREFILL( half, 16, 64, 128, 8 )
MILA_GQA_FLASH_PREFILL( half, 16, 64, 256, 8 )
MILA_GQA_FLASH_PREFILL( half, 8, 64, 512, 8 )
MILA_GQA_FLASH_PREFILL( bfloat, 16, 64, 128, 8 )
MILA_GQA_FLASH_PREFILL( bfloat, 16, 64, 256, 8 )
MILA_GQA_FLASH_PREFILL( bfloat, 8, 64, 512, 8 )
