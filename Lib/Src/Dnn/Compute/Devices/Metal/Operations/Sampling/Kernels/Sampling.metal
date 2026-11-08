// Token sampling: greedy, and the multinomial with top-k and top-p, in the stages of CUDA's multi-block pipeline (Sampling.cu).
//
// A row is spread over many threadgroups of kSamplingThreads, a grid's y index naming the row, since one SIMD-group alone waits on
// memory through every pass of a 262K vocabulary. Each truncation finds its cut by radix selection on the scaled logits' bits: a float
// maps to a key whose order is the float's, and three histogram rounds take its top 12 bits, the next 12 and the last 8. Membership is
// a key's bits, so the cut is the exact k-th value or nucleus edge, with no bin edge to round. A suppressed token holds -FLT_MAX and not
// -infinity, since Metal compiles with fast math on, which assumes no infinities.

#include <metal_stdlib>

using namespace metal;

// The most rows one call draws: a speculative verify's drafts and the token before them, as CUDA's kMaxSampleRows.
constant constexpr int kSamplingMaximumRows = 8;

// The pipeline's shape, which MetalSamplingOp mirrors to size its scratch and its grids.
constant constexpr int kSamplingThreads = 256;
constant constexpr int kSamplingSimdgroups = 8;
constant constexpr int kSamplingBins = 4096;
constant constexpr int kSamplingMaximumChunks = 1024;

// The radix rounds: each takes `bits` of the key from `shift` up, within the prefix the rounds before it chose.
constant constexpr int kSamplingRounds = 3;
constant constexpr int kRoundShift[kSamplingRounds] = { 20, 8, 0 };
constant constexpr int kRoundBits[kSamplingRounds] = { 12, 12, 8 };

// A row's float scratch: header slots, then the scale stage's partials, the top-p bins, and the chunk sums.
constant constexpr int kSlotLargest = 0;
constant constexpr int kSlotTopKThreshold = 1;
constant constexpr int kSlotTopPThreshold = 2;
constant constexpr int kSlotMassAbove = 3;
constant constexpr int kSlotTotalMass = 4;
constant constexpr int kSamplingHeader = 16;
constant constexpr int kOffsetLargestPartials = kSamplingHeader;
constant constexpr int kOffsetMassBins = kOffsetLargestPartials + kSamplingMaximumChunks;
constant constexpr int kOffsetChunkPartials = kOffsetMassBins + kSamplingBins;
constant constexpr int kSamplingFloatScratch = kOffsetChunkPartials + kSamplingMaximumChunks;

// A row's count scratch: the count above the prefix, the prefix, then the top-k bins.
constant constexpr int kSlotCountAbove = 0;
constant constexpr int kSlotPrefix = 1;
constant constexpr int kOffsetCountBins = 16;
constant constexpr int kSamplingCountScratch = kOffsetCountBins + kSamplingBins;

struct SamplingParameters
{
    int vocab;
    int top_k;
    float top_p;
    float temperature;
    float softcap;
    float presence_penalty;
    uint use_suppressed;
    uint use_produced;
    uint use_round;
    float uniforms[kSamplingMaximumRows];
    int round;
    int partial_count;
    int chunk_size;
    int chunk_count;
};

// Whether token i counts as produced for row `row`: the reply's tokens, and with a round, the drafts 1..row before the row.
inline bool sampling_produced( int i, int row, device const uchar* produced, device const int* round, constant SamplingParameters& parameters )
{
    if ( produced[i] != 0 )
    {
        return true;
    }

    if ( parameters.use_round != 0 )
    {
        for ( int j = 1; j <= row; ++j )
        {
            if ( round[j] == i )
            {
                return true;
            }
        }
    }

    return false;
}

// A logit softcapped and then penalized, before temperature (TokenSampling.md 3.3).
inline float sampling_adjusted( float x, int i, int row, device const uchar* produced, device const int* round, constant SamplingParameters& parameters )
{
    if ( parameters.softcap > 0.0f )
    {
        x = parameters.softcap * tanh( x / parameters.softcap );
    }

    if ( parameters.use_produced != 0 && sampling_produced( i, row, produced, round, parameters ) )
    {
        x -= parameters.presence_penalty;
    }

    return x;
}

// A float's bits as a key whose unsigned order is the float's order, and back.
inline uint sampling_key( float x )
{
    const uint bits = as_type<uint>( x );

    return ( bits & 0x80000000u ) != 0u ? ~bits : bits | 0x80000000u;
}

inline float sampling_value( uint key )
{
    return as_type<float>( ( key & 0x80000000u ) != 0u ? key & 0x7fffffffu : ~key );
}

// A cut's value from its key. A walk that ran past every survivor, as fewer survivors than top_k leave it, ends on a key below any
// value a token holds, and keeps them all.
inline float sampling_threshold( uint key )
{
    return key <= sampling_key( -FLT_MAX ) ? -FLT_MAX : sampling_value( key );
}

// The bits a round's prefix fixes: none on the first round.
inline uint sampling_prefix_mask( int round )
{
    return round == 0 ? 0u : 0xffffffffu << uint( kRoundShift[round] + kRoundBits[round] );
}

// Threadgroup-wide reductions: each SIMD-group's result through `shared`, then every SIMD-group reduces those. Every thread gets the
// result, and `shared` is free again on return.
inline float sampling_group_max( float value, threadgroup float* shared, uint lane, uint simd )
{
    value = simd_max( value );

    if ( lane == 0 )
    {
        shared[simd] = value;
    }

    threadgroup_barrier( mem_flags::mem_threadgroup );

    const float result = simd_max( lane < uint( kSamplingSimdgroups ) ? shared[lane] : -FLT_MAX );

    threadgroup_barrier( mem_flags::mem_threadgroup );

    return result;
}

inline float sampling_group_sum( float value, threadgroup float* shared, uint lane, uint simd )
{
    value = simd_sum( value );

    if ( lane == 0 )
    {
        shared[simd] = value;
    }

    threadgroup_barrier( mem_flags::mem_threadgroup );

    const float result = simd_sum( lane < uint( kSamplingSimdgroups ) ? shared[lane] : 0.0f );

    threadgroup_barrier( mem_flags::mem_threadgroup );

    return result;
}

// The largest logit, ties to the lowest index as std::max_element has it, one threadgroup per row. The softcap is monotonic, so it is
// applied only where the penalty needs it.
template<typename TElement>
kernel void sampling_greedy(
    device const TElement* all_logits [[buffer(0)]],
    device int* tokens [[buffer(1)]],
    device const uchar* suppressed [[buffer(2)]],
    device const uchar* produced [[buffer(3)]],
    device const int* round [[buffer(4)]],
    constant SamplingParameters& parameters [[buffer(5)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]] )
{
    threadgroup float shared_values[kSamplingSimdgroups];
    threadgroup int shared_indices[kSamplingSimdgroups];

    const int row = int( group.y );
    device const TElement* logits = all_logits + size_t( row ) * size_t( parameters.vocab );

    float best = -FLT_MAX;
    int best_index = INT_MAX;

    for ( int i = int( thread_index ); i < parameters.vocab; i += kSamplingThreads )
    {
        if ( parameters.use_suppressed != 0 && suppressed[i] != 0 )
        {
            continue;
        }

        const float value = parameters.use_produced != 0
            ? sampling_adjusted( float( logits[i] ), i, row, produced, round, parameters )
            : float( logits[i] );

        if ( value > best )
        {
            best = value;
            best_index = i;
        }
    }

    const float simd_largest = simd_max( best );
    const int simd_winner = simd_min( best == simd_largest ? best_index : INT_MAX );

    if ( lane == 0 )
    {
        shared_values[simd] = simd_largest;
        shared_indices[simd] = simd_winner;
    }

    threadgroup_barrier( mem_flags::mem_threadgroup );

    if ( simd == 0 )
    {
        const bool holds = lane < uint( kSamplingSimdgroups );
        const float value = holds ? shared_values[lane] : -FLT_MAX;
        const int index = holds ? shared_indices[lane] : INT_MAX;
        const float largest = simd_max( value );
        const int winner = simd_min( value == largest ? index : INT_MAX );

        if ( lane == 0 )
        {
            tokens[row] = winner == INT_MAX ? 0 : winner;
        }
    }
}

template [[host_name("sampling_greedy_half")]] kernel void sampling_greedy<half>(
    device const half*, device int*, device const uchar*, device const uchar*, device const int*, constant SamplingParameters&, uint, uint,
    uint, uint2 );
template [[host_name("sampling_greedy_bfloat")]] kernel void sampling_greedy<bfloat>(
    device const bfloat*, device int*, device const uchar*, device const uchar*, device const int*, constant SamplingParameters&, uint, uint,
    uint, uint2 );

// The multinomial's stages share one binding: 0 logits, 1 tokens, 2 scratch (the row's scaled logits, then its probabilities), 3 the
// row's float scratch, 4 its count scratch, 5 suppressed, 6 produced, 7 round, 8 the parameters.

// Stage 1: scaled logits into scratch, with each threadgroup's largest. A suppressed token is left out of it.
template<typename TElement>
kernel void sampling_scale(
    device const TElement* all_logits [[buffer(0)]],
    device float* all_scratch [[buffer(2)]],
    device float* all_reduction [[buffer(3)]],
    device const uchar* suppressed [[buffer(5)]],
    device const uchar* produced [[buffer(6)]],
    device const int* round [[buffer(7)]],
    constant SamplingParameters& parameters [[buffer(8)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 groups [[threadgroups_per_grid]] )
{
    threadgroup float shared[kSamplingSimdgroups];

    const int vocab = parameters.vocab;
    const int row = int( group.y );

    device const TElement* logits = all_logits + size_t( row ) * size_t( vocab );
    device float* scratch = all_scratch + size_t( row ) * size_t( vocab );
    device float* reduction = all_reduction + size_t( row ) * size_t( kSamplingFloatScratch );

    float local_largest = -FLT_MAX;

    for ( int i = int( group.x ) * kSamplingThreads + int( thread_index ); i < vocab; i += int( groups.x ) * kSamplingThreads )
    {
        if ( parameters.use_suppressed != 0 && suppressed[i] != 0 )
        {
            scratch[i] = -FLT_MAX;

            continue;
        }

        const float x = sampling_adjusted( float( logits[i] ), i, row, produced, round, parameters ) / parameters.temperature;

        scratch[i] = x;
        local_largest = max( local_largest, x );
    }

    const float largest = sampling_group_max( local_largest, shared, lane, simd );

    if ( thread_index == 0 )
    {
        reduction[kOffsetLargestPartials + int( group.x )] = largest;
    }
}

template [[host_name("sampling_scale_half")]] kernel void sampling_scale<half>(
    device const half*, device float*, device float*, device const uchar*, device const uchar*, device const int*,
    constant SamplingParameters&, uint, uint, uint, uint2, uint2 );
template [[host_name("sampling_scale_bfloat")]] kernel void sampling_scale<bfloat>(
    device const bfloat*, device float*, device float*, device const uchar*, device const uchar*, device const int*,
    constant SamplingParameters&, uint, uint, uint, uint2, uint2 );

// Stage 2, one threadgroup a row: the row's largest, no cut yet, and every bin cleared.
kernel void sampling_prepare(
    device float* all_reduction [[buffer(3)]],
    device uint* all_counts [[buffer(4)]],
    constant SamplingParameters& parameters [[buffer(8)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]] )
{
    threadgroup float shared[kSamplingSimdgroups];

    device float* reduction = all_reduction + size_t( group.y ) * size_t( kSamplingFloatScratch );
    device uint* counts = all_counts + size_t( group.y ) * size_t( kSamplingCountScratch );

    float local_largest = -FLT_MAX;

    for ( int i = int( thread_index ); i < parameters.partial_count; i += kSamplingThreads )
    {
        local_largest = max( local_largest, reduction[kOffsetLargestPartials + i] );
    }

    const float largest = sampling_group_max( local_largest, shared, lane, simd );

    if ( thread_index == 0 )
    {
        reduction[kSlotLargest] = largest;
        reduction[kSlotTopKThreshold] = -FLT_MAX;
        reduction[kSlotTopPThreshold] = -FLT_MAX;
        reduction[kSlotMassAbove] = 0.0f;
        reduction[kSlotTotalMass] = 0.0f;
        counts[kSlotCountAbove] = 0u;
        counts[kSlotPrefix] = 0u;
    }

    for ( int j = int( thread_index ); j < kSamplingBins; j += kSamplingThreads )
    {
        counts[kOffsetCountBins + j] = 0u;
        reduction[kOffsetMassBins + j] = 0.0f;
    }
}

// Top-k, count pass: each token whose key carries the round's prefix counted in the bin of its next bits, and those whose key is above
// the prefix counted above.
kernel void sampling_top_k_count(
    device const float* all_scratch [[buffer(2)]],
    device atomic_uint* all_counts [[buffer(4)]],
    constant SamplingParameters& parameters [[buffer(8)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 groups [[threadgroups_per_grid]] )
{
    threadgroup atomic_uint bins[kSamplingBins];
    threadgroup atomic_uint above;

    const int vocab = parameters.vocab;
    const int round = parameters.round;
    const int bin_count = 1 << kRoundBits[round];

    device const float* scratch = all_scratch + size_t( group.y ) * size_t( vocab );
    device atomic_uint* counts = all_counts + size_t( group.y ) * size_t( kSamplingCountScratch );

    for ( int j = int( thread_index ); j < bin_count; j += kSamplingThreads )
    {
        atomic_store_explicit( &bins[j], 0u, memory_order_relaxed );
    }

    if ( thread_index == 0 )
    {
        atomic_store_explicit( &above, 0u, memory_order_relaxed );
    }

    threadgroup_barrier( mem_flags::mem_threadgroup );

    const uint mask = sampling_prefix_mask( round );
    const uint prefix = atomic_load_explicit( &counts[kSlotPrefix], memory_order_relaxed );
    const uint shift = uint( kRoundShift[round] );
    const uint bin_mask = uint( bin_count - 1 );

    for ( int i = int( group.x ) * kSamplingThreads + int( thread_index ); i < vocab; i += int( groups.x ) * kSamplingThreads )
    {
        const uint key = sampling_key( scratch[i] );
        const uint fixed = key & mask;

        if ( fixed > prefix )
        {
            atomic_fetch_add_explicit( &above, 1u, memory_order_relaxed );
        } else if ( fixed == prefix )
        {
            atomic_fetch_add_explicit( &bins[( key >> shift ) & bin_mask], 1u, memory_order_relaxed );
        }
    }

    threadgroup_barrier( mem_flags::mem_threadgroup );

    if ( thread_index == 0 )
    {
        const uint count = atomic_load_explicit( &above, memory_order_relaxed );

        if ( count > 0u )
        {
            atomic_fetch_add_explicit( &counts[kSlotCountAbove], count, memory_order_relaxed );
        }
    }

    for ( int j = int( thread_index ); j < bin_count; j += kSamplingThreads )
    {
        const uint count = atomic_load_explicit( &bins[j], memory_order_relaxed );

        if ( count > 0u )
        {
            atomic_fetch_add_explicit( &counts[kOffsetCountBins + j], count, memory_order_relaxed );
        }
    }
}

// The bin, walking down from the top of `bin_count`, at which a running total from `start` first reaches `target`: each thread sums its
// own run of bins, and one walks the runs and then the run that holds it. Zero when none does. Called by every thread of the threadgroup;
// the answer is thread 0's.
template<typename TValue, typename TBins>
inline int sampling_boundary_bin( TBins bins, int bin_count, TValue start, TValue target, threadgroup TValue* runs, uint thread_index )
{
    const int per_thread = max( bin_count / kSamplingThreads, 1 );
    const int owners = bin_count / per_thread;

    TValue run = TValue( 0 );

    if ( int( thread_index ) < owners )
    {
        for ( int b = 0; b < per_thread; ++b )
        {
            run += bins[int( thread_index ) * per_thread + b];
        }

        runs[thread_index] = run;
    }

    threadgroup_barrier( mem_flags::mem_threadgroup );

    int boundary = 0;

    if ( thread_index == 0 )
    {
        TValue cumulative = start;
        bool found = false;

        for ( int t = owners - 1; t >= 0 && !found; --t )
        {
            if ( cumulative + runs[t] < target )
            {
                cumulative += runs[t];

                continue;
            }

            for ( int b = ( t + 1 ) * per_thread - 1; b >= t * per_thread; --b )
            {
                if ( cumulative + bins[b] >= target )
                {
                    boundary = b;
                    found = true;

                    break;
                }

                cumulative += bins[b];
            }
        }
    }

    return boundary;
}

// Top-k, select pass: the bin holding the k-th key from the top extends the prefix; after the last round the prefix is the k-th value,
// and every token tied with it survives, as HuggingFace's TopKLogitsWarper keeps them. Fewer survivors than k leave bin 0, which keeps
// all. The bins and the count above are cleared for the next round, and the prefix for top-p's rounds.
kernel void sampling_top_k_select(
    device float* all_reduction [[buffer(3)]],
    device uint* all_counts [[buffer(4)]],
    constant SamplingParameters& parameters [[buffer(8)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]] )
{
    threadgroup uint runs[kSamplingThreads];

    device float* reduction = all_reduction + size_t( group.y ) * size_t( kSamplingFloatScratch );
    device uint* counts = all_counts + size_t( group.y ) * size_t( kSamplingCountScratch );

    const int round = parameters.round;
    const int bin_count = 1 << kRoundBits[round];
    const int boundary = sampling_boundary_bin<uint>(
        counts + kOffsetCountBins, bin_count, counts[kSlotCountAbove], uint( parameters.top_k ), runs, thread_index );

    if ( thread_index == 0 )
    {
        const uint prefix = counts[kSlotPrefix] | ( uint( boundary ) << uint( kRoundShift[round] ) );

        if ( round == kSamplingRounds - 1 )
        {
            reduction[kSlotTopKThreshold] = sampling_threshold( prefix );
            counts[kSlotPrefix] = 0u;
        } else {
            counts[kSlotPrefix] = prefix;
        }

        counts[kSlotCountAbove] = 0u;
    }

    threadgroup_barrier( mem_flags::mem_device );

    for ( int j = int( thread_index ); j < bin_count; j += kSamplingThreads )
    {
        counts[kOffsetCountBins + j] = 0u;
    }
}

// Top-p, mass pass: as the count pass over top-k's survivors, with exp( x - largest ) in place of one. Masses meet in device float
// atomics, so their order, and a boundary's last bit, may differ from run to run, as on CUDA.
kernel void sampling_top_p_mass(
    device const float* all_scratch [[buffer(2)]],
    device atomic_float* all_reduction [[buffer(3)]],
    device const uint* all_counts [[buffer(4)]],
    constant SamplingParameters& parameters [[buffer(8)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 groups [[threadgroups_per_grid]] )
{
    const int vocab = parameters.vocab;
    const int round = parameters.round;

    device const float* scratch = all_scratch + size_t( group.y ) * size_t( vocab );
    device atomic_float* reduction = all_reduction + size_t( group.y ) * size_t( kSamplingFloatScratch );
    device const uint* counts = all_counts + size_t( group.y ) * size_t( kSamplingCountScratch );

    const float floor_value = atomic_load_explicit( &reduction[kSlotTopKThreshold], memory_order_relaxed );
    const float largest = atomic_load_explicit( &reduction[kSlotLargest], memory_order_relaxed );
    const uint mask = sampling_prefix_mask( round );
    const uint prefix = counts[kSlotPrefix];
    const uint shift = uint( kRoundShift[round] );
    const uint bin_mask = uint( ( 1 << kRoundBits[round] ) - 1 );

    for ( int i = int( group.x ) * kSamplingThreads + int( thread_index ); i < vocab; i += int( groups.x ) * kSamplingThreads )
    {
        const float x = scratch[i];

        if ( !( x > -FLT_MAX && x >= floor_value ) )
        {
            continue;
        }

        const uint key = sampling_key( x );
        const uint fixed = key & mask;

        if ( fixed > prefix )
        {
            atomic_fetch_add_explicit( &reduction[kSlotMassAbove], exp( x - largest ), memory_order_relaxed );
        } else if ( fixed == prefix )
        {
            atomic_fetch_add_explicit( &reduction[kOffsetMassBins + int( ( key >> shift ) & bin_mask )], exp( x - largest ), memory_order_relaxed );
        }
    }
}

// Top-p, select pass: the nucleus is the smallest top set whose mass reaches top_p of the total, so the boundary is kept, and after the
// last round its value is the threshold. The total is the first round's, over every survivor of top-k.
kernel void sampling_top_p_select(
    device float* all_reduction [[buffer(3)]],
    device uint* all_counts [[buffer(4)]],
    constant SamplingParameters& parameters [[buffer(8)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]] )
{
    threadgroup float shared[kSamplingSimdgroups];
    threadgroup float runs[kSamplingThreads];

    device float* reduction = all_reduction + size_t( group.y ) * size_t( kSamplingFloatScratch );
    device uint* counts = all_counts + size_t( group.y ) * size_t( kSamplingCountScratch );
    device float* bins = reduction + kOffsetMassBins;

    const int round = parameters.round;
    const int bin_count = 1 << kRoundBits[round];

    if ( round == 0 )
    {
        float local = 0.0f;

        for ( int j = int( thread_index ); j < bin_count; j += kSamplingThreads )
        {
            local += bins[j];
        }

        const float total = sampling_group_sum( local, shared, lane, simd ) + reduction[kSlotMassAbove];

        if ( thread_index == 0 )
        {
            reduction[kSlotTotalMass] = total;
        }

        threadgroup_barrier( mem_flags::mem_device );
    }

    const float target = parameters.top_p * reduction[kSlotTotalMass];
    const int boundary = sampling_boundary_bin<float>( bins, bin_count, reduction[kSlotMassAbove], target, runs, thread_index );

    if ( thread_index == 0 )
    {
        const uint prefix = counts[kSlotPrefix] | ( uint( boundary ) << uint( kRoundShift[round] ) );

        if ( round == kSamplingRounds - 1 )
        {
            reduction[kSlotTopPThreshold] = sampling_threshold( prefix );
        } else {
            counts[kSlotPrefix] = prefix;
        }

        reduction[kSlotMassAbove] = 0.0f;
    }

    threadgroup_barrier( mem_flags::mem_device );

    for ( int j = int( thread_index ); j < bin_count; j += kSamplingThreads )
    {
        bins[j] = 0.0f;
    }
}

// Stage 5: each survivor's exp( x - largest ), the rest 0, into scratch, with each contiguous chunk's sum, so that the draw finds its
// chunk without walking the row. With neither truncation the threshold is -FLT_MAX and every token not suppressed survives.
kernel void sampling_probability(
    device float* all_scratch [[buffer(2)]],
    device float* all_reduction [[buffer(3)]],
    constant SamplingParameters& parameters [[buffer(8)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]] )
{
    threadgroup float shared[kSamplingSimdgroups];

    const int vocab = parameters.vocab;

    device float* scratch = all_scratch + size_t( group.y ) * size_t( vocab );
    device float* reduction = all_reduction + size_t( group.y ) * size_t( kSamplingFloatScratch );

    const int begin = int( group.x ) * parameters.chunk_size;
    const int end = min( begin + parameters.chunk_size, vocab );
    const float threshold = max( reduction[kSlotTopKThreshold], reduction[kSlotTopPThreshold] );
    const float largest = reduction[kSlotLargest];

    float local = 0.0f;

    for ( int i = begin + int( thread_index ); i < end; i += kSamplingThreads )
    {
        const float x = scratch[i];
        const float e = x > -FLT_MAX && x >= threshold ? exp( x - largest ) : 0.0f;

        scratch[i] = e;
        local += e;
    }

    const float chunk = sampling_group_sum( local, shared, lane, simd );

    if ( thread_index == 0 )
    {
        reduction[kOffsetChunkPartials + int( group.x )] = chunk;
    }
}

// Stage 6, one threadgroup a row: the inverse CDF in token order. The chunk sums find the first chunk whose running total reaches
// r times the total; its values are staged a tile at a time, and one thread walks them. A rounding that leaves the target past every
// survivor takes the last survivor walked, and a row with none, the configuration suppressing everything, gives vocab - 1.
kernel void sampling_draw(
    device int* tokens [[buffer(1)]],
    device const float* all_scratch [[buffer(2)]],
    device const float* all_reduction [[buffer(3)]],
    constant SamplingParameters& parameters [[buffer(8)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]] )
{
    threadgroup float chunk_sums[kSamplingMaximumChunks];
    threadgroup float tile[kSamplingThreads];
    threadgroup float shared_target;
    threadgroup float shared_base;
    threadgroup int shared_start;
    threadgroup int shared_done;
    threadgroup int shared_result;

    const int vocab = parameters.vocab;
    const int row = int( group.y );
    const int chunk_count = parameters.chunk_count;

    device const float* scratch = all_scratch + size_t( row ) * size_t( vocab );
    device const float* reduction = all_reduction + size_t( row ) * size_t( kSamplingFloatScratch );

    for ( int c = int( thread_index ); c < chunk_count; c += kSamplingThreads )
    {
        chunk_sums[c] = reduction[kOffsetChunkPartials + c];
    }

    threadgroup_barrier( mem_flags::mem_threadgroup );

    if ( thread_index == 0 )
    {
        float total = 0.0f;

        for ( int c = 0; c < chunk_count; ++c )
        {
            total += chunk_sums[c];
        }

        const float target = parameters.uniforms[row] * total;

        float prefix = 0.0f;
        int start = chunk_count - 1;
        float base = total - chunk_sums[chunk_count - 1];

        for ( int c = 0; c < chunk_count; ++c )
        {
            if ( prefix + chunk_sums[c] >= target )
            {
                start = c;
                base = prefix;

                break;
            }

            prefix += chunk_sums[c];
        }

        shared_target = target;
        shared_base = base;
        shared_start = start;
        shared_done = 0;
        shared_result = -1;
    }

    threadgroup_barrier( mem_flags::mem_threadgroup );

    float cumulative = shared_base;

    for ( int c = shared_start; c < chunk_count; ++c )
    {
        const int begin = c * parameters.chunk_size;
        const int end = min( begin + parameters.chunk_size, vocab );

        for ( int first = begin; first < end; first += kSamplingThreads )
        {
            const int index = first + int( thread_index );

            tile[thread_index] = index < end ? scratch[index] : 0.0f;

            threadgroup_barrier( mem_flags::mem_threadgroup );

            if ( thread_index == 0 )
            {
                const int count = min( kSamplingThreads, end - first );

                for ( int k = 0; k < count; ++k )
                {
                    const float value = tile[k];

                    if ( value <= 0.0f )
                    {
                        continue;
                    }

                    cumulative += value;
                    shared_result = first + k;

                    if ( cumulative >= shared_target )
                    {
                        shared_done = 1;

                        break;
                    }
                }
            }

            threadgroup_barrier( mem_flags::mem_threadgroup );

            if ( shared_done != 0 )
            {
                break;
            }
        }

        if ( shared_done != 0 )
        {
            break;
        }
    }

    if ( thread_index == 0 )
    {
        tokens[row] = shared_result >= 0 ? shared_result : vocab - 1;
    }
}

kernel void sampling_mark_produced(
    device const int* token [[buffer(0)]],
    device uchar* produced [[buffer(1)]] )
{
    produced[token[0]] = 1;
}

struct SamplingAcceptParameters
{
    int rows;
    uint use_produced;
};

// A verified round's acceptance, one thread: a draft is kept while the row before it chose it. Writes [m, d_1 .. d_m, next], and the next
// token into slot 0 of `tokens`, after every read of it (Gemma4Mtp.md 4.8).
kernel void sampling_accept(
    device int* tokens [[buffer(0)]],
    device const int* chosen [[buffer(1)]],
    device int* result [[buffer(2)]],
    device uchar* produced [[buffer(3)]],
    constant SamplingAcceptParameters& parameters [[buffer(4)]] )
{
    int accepted = 0;

    while ( accepted < parameters.rows - 1 && tokens[accepted + 1] == chosen[accepted] )
    {
        ++accepted;
    }

    const int next = chosen[accepted];

    result[0] = accepted;

    for ( int i = 1; i <= accepted; ++i )
    {
        result[i] = tokens[i];
    }

    result[accepted + 1] = next;

    if ( parameters.use_produced != 0 )
    {
        for ( int i = 1; i <= accepted; ++i )
        {
            produced[tokens[i]] = 1;
        }

        produced[next] = 1;
    }

    tokens[0] = next;
}
