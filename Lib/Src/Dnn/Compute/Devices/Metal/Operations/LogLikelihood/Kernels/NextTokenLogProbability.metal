// Each row's log-probability of the token that followed it, with the row's largest subtracted before exponentiating, as
// Dnn::nextTokenLogProbability has it. Metal has no 64-bit float, so the exponentials sum in 32 bits, the one difference from that reference.

#include <metal_stdlib>

using namespace metal;

struct NextTokenParameters
{
    int vocab;
    float softcap;
};

inline float next_token_capped( float logit, float softcap )
{
    return softcap > 0.0f ? softcap * tanh( logit / softcap ) : logit;
}

template<typename TElement>
kernel void next_token_log_probability(
    device const TElement* logits [[buffer(0)]],
    device const int* tokens [[buffer(1)]],
    device float* log_probabilities [[buffer(2)]],
    constant NextTokenParameters& parameters [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 threads [[threads_per_threadgroup]] )
{
    const int row = int( group.x );
    device const TElement* row_logits = logits + row * parameters.vocab;

    float local_largest = -FLT_MAX;

    for ( int v = int( lane ); v < parameters.vocab; v += int( threads.x ) )
    {
        local_largest = max( local_largest, next_token_capped( float( row_logits[v] ), parameters.softcap ) );
    }

    const float largest = simd_max( local_largest );

    float sum = 0.0f;

    for ( int v = int( lane ); v < parameters.vocab; v += int( threads.x ) )
    {
        sum += exp( next_token_capped( float( row_logits[v] ), parameters.softcap ) - largest );
    }

    const float total = simd_sum( sum );

    if ( lane == 0 )
    {
        const int target = tokens[row + 1];

        log_probabilities[row] = target < 0 || target >= parameters.vocab
            ? NAN
            : next_token_capped( float( row_logits[target] ), parameters.softcap ) - largest - log( total );
    }
}

template [[host_name("next_token_log_probability_half")]] kernel void next_token_log_probability<half>(
    device const half*, device const int*, device float*, constant NextTokenParameters&, uint3, uint, uint3 );
template [[host_name("next_token_log_probability_bfloat")]] kernel void next_token_log_probability<bfloat>(
    device const bfloat*, device const int*, device float*, constant NextTokenParameters&, uint3, uint, uint3 );
