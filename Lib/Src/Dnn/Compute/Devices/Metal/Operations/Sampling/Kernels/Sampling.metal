// Token sampling: greedy, and the multinomial with top-k and top-p, following CUDA's reference kernel step for step.
//
// One SIMD-group per row does everything, its lanes striding the vocabulary and agreeing through simd_sum and simd_max; a draw of one
// row is a grid of one. A suppressed token holds -FLT_MAX and not -infinity, since Metal compiles with fast math on, which assumes no
// infinities.

#include <metal_stdlib>

using namespace metal;

// The most rows one call draws: a speculative verify's drafts and the token before them, as CUDA's kMaxSampleRows.
constant constexpr int kSamplingMaximumRows = 8;

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

// The largest logit, ties to the lowest index as std::max_element has it. The softcap is monotonic, so it is applied only where the penalty needs it.
template<typename TElement>
kernel void sampling_greedy(
    device const TElement* all_logits [[buffer(0)]],
    device int* tokens [[buffer(1)]],
    device const uchar* suppressed [[buffer(2)]],
    device const uchar* produced [[buffer(3)]],
    device const int* round [[buffer(4)]],
    constant SamplingParameters& parameters [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 threads [[threads_per_threadgroup]] )
{
    const int row = int( group.x );
    device const TElement* logits = all_logits + size_t( row ) * size_t( parameters.vocab );

    float best = -FLT_MAX;
    int best_index = INT_MAX;

    for ( int i = int( lane ); i < parameters.vocab; i += int( threads.x ) )
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

    const float largest = simd_max( best );
    const int winner = simd_min( best == largest ? best_index : INT_MAX );

    if ( lane == 0 )
    {
        tokens[row] = winner == INT_MAX ? 0 : winner;
    }
}

template [[host_name("sampling_greedy_half")]] kernel void sampling_greedy<half>(
    device const half*, device int*, device const uchar*, device const uchar*, device const int*, constant SamplingParameters&, uint, uint3,
    uint3 );
template [[host_name("sampling_greedy_bfloat")]] kernel void sampling_greedy<bfloat>(
    device const bfloat*, device int*, device const uchar*, device const uchar*, device const int*, constant SamplingParameters&, uint,
    uint3, uint3 );

template<typename TElement>
kernel void sampling_draw(
    device const TElement* all_logits [[buffer(0)]],
    device int* tokens [[buffer(1)]],
    device float* all_scratch [[buffer(2)]],
    device const uchar* suppressed [[buffer(3)]],
    device const uchar* produced [[buffer(4)]],
    device const int* round [[buffer(5)]],
    constant SamplingParameters& parameters [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 threads [[threads_per_threadgroup]] )
{
    const int vocab = parameters.vocab;
    const int width = int( threads.x );
    const int row = int( group.x );

    device const TElement* logits = all_logits + size_t( row ) * size_t( vocab );
    device float* scratch = all_scratch + size_t( row ) * size_t( vocab );
    device int* token = tokens + row;

    float local_largest = -FLT_MAX;
    float local_smallest = FLT_MAX;

    for ( int i = int( lane ); i < vocab; i += width )
    {
        if ( parameters.use_suppressed != 0 && suppressed[i] != 0 )
        {
            scratch[i] = -FLT_MAX;

            continue;
        }

        const float x = sampling_adjusted( float( logits[i] ), i, row, produced, round, parameters ) / parameters.temperature;

        scratch[i] = x;
        local_largest = max( local_largest, x );
        local_smallest = min( local_smallest, x );
    }

    const float largest = simd_max( local_largest );
    const float smallest = simd_min( local_smallest );

    // Top-k: `low` keeps count >= top_k and `high` count < top_k, so `low` converges on the k-th value from below, keeping exactly top_k
    // for distinct logits, and every token tied with the k-th where they tie.
    if ( parameters.top_k > 0 && parameters.top_k < vocab )
    {
        float low = smallest;
        float high = largest;

        for ( int iteration = 0; iteration < 40; ++iteration )
        {
            const float middle = 0.5f * ( low + high );

            int count = 0;

            for ( int i = int( lane ); i < vocab; i += width )
            {
                count += scratch[i] >= middle ? 1 : 0;
            }

            if ( simd_sum( count ) >= parameters.top_k )
            {
                low = middle;
            } else {
                high = middle;
            }
        }

        for ( int i = int( lane ); i < vocab; i += width )
        {
            if ( scratch[i] < low )
            {
                scratch[i] = -FLT_MAX;
            }
        }
    }

    // exp( scaled - largest ), with what was removed at exactly 0.
    float local_total = 0.0f;

    for ( int i = int( lane ); i < vocab; i += width )
    {
        const float e = scratch[i] > -FLT_MAX ? exp( scratch[i] - largest ) : 0.0f;

        scratch[i] = e;
        local_total += e;
    }

    float total = simd_sum( local_total );

    // Top-p: a threshold on the unnormalized probability, keeping the smallest set whose mass reaches top_p of the total.
    if ( parameters.top_p < 1.0f )
    {
        const float target = parameters.top_p * total;

        float low = 0.0f;
        float high = 1.0f;

        for ( int iteration = 0; iteration < 40; ++iteration )
        {
            const float middle = 0.5f * ( low + high );

            float mass = 0.0f;

            for ( int i = int( lane ); i < vocab; i += width )
            {
                mass += scratch[i] >= middle ? scratch[i] : 0.0f;
            }

            if ( simd_sum( mass ) > target )
            {
                low = middle;
            } else {
                high = middle;
            }
        }

        local_total = 0.0f;

        for ( int i = int( lane ); i < vocab; i += width )
        {
            if ( scratch[i] < low )
            {
                scratch[i] = 0.0f;
            }

            local_total += scratch[i];
        }

        total = simd_sum( local_total );
    }

    // The inverse CDF in token order. Each lane holds a contiguous run of tokens, so that lane order is token order for the prefix sum, and
    // the first lane the target falls inside walks its run. A rounding that leaves the target past every lane takes the last survivor.
    const int first = int( lane ) * vocab / width;
    const int last = ( int( lane ) + 1 ) * vocab / width;

    float mass = 0.0f;

    for ( int i = first; i < last; ++i )
    {
        mass += scratch[i];
    }

    const float before = simd_prefix_exclusive_sum( mass );
    const float target = parameters.uniforms[row] * total;
    const bool holds_target = mass > 0.0f && before + mass >= target;
    const uint first_holder = simd_min( holds_target ? lane : UINT_MAX );
    const int last_with_mass = simd_max( mass > 0.0f ? int( lane ) : -1 );

    const bool takes_target = lane == first_holder;
    const bool takes_fallback = first_holder == UINT_MAX && int( lane ) == last_with_mass;

    if ( takes_target || takes_fallback )
    {
        float cumulative = before;
        int result = vocab - 1;

        for ( int i = first; i < last; ++i )
        {
            cumulative += scratch[i];

            if ( scratch[i] > 0.0f )
            {
                result = i;

                if ( takes_target && cumulative >= target )
                {
                    break;
                }
            }
        }

        token[0] = result;
    }

    // Nothing survived at all, which a configuration suppressing the whole vocabulary would mean.
    if ( lane == 0 && last_with_mass < 0 )
    {
        token[0] = vocab - 1;
    }
}

template [[host_name("sampling_draw_half")]] kernel void sampling_draw<half>(
    device const half*, device int*, device float*, device const uchar*, device const uchar*, device const int*, constant SamplingParameters&,
    uint, uint3, uint3 );
template [[host_name("sampling_draw_bfloat")]] kernel void sampling_draw<bfloat>(
    device const bfloat*, device int*, device float*, device const uchar*, device const uchar*, device const int*,
    constant SamplingParameters&, uint, uint3, uint3 );

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
