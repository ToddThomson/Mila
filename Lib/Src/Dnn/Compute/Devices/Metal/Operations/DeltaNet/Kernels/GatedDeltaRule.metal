// The gated delta rule in its recurrent form, as CUDA's (GatedDeltaRule.cuh). Per value head the state S is
// [head_key_dim, head_value_dim], FP32, and each step is
//
//   S <- S * exp(g);  kv <- k^T S;  delta <- (v - kv) * beta;  S <- S + k (x) delta;  out <- q^T S
//
// with q and k L2-normalized (eps 1e-6), q also scaled by 1 / sqrt(head_key_dim), beta = sigmoid(b) and
// g = -exp(A_log) * softplus(a + dt_bias). Value head h reads key head h / (value heads / key heads).
//
// A threadgroup holds one value head's block of 32 columns, and four adjacent lanes of a SIMD-group hold one column: lane p
// of the four holds key rows p, p + 4, p + 8, ..., in registers for the whole chunk. Their partial dot products meet by two
// shuffles, so no step needs threadgroup memory but q and k. The arithmetic is 32 bits throughout.

#include <metal_stdlib>

using namespace metal;

struct GatedDeltaRuleShape
{
    uint batch;
    uint steps;
    uint key_heads;
    uint value_heads;
    uint head_key_dim;
    uint head_value_dim;
};

constant uint kDeltaRuleKeySplit = 4;
constant uint kDeltaRuleColumnsPerGroup = 32;
constant uint kDeltaRuleMostHeadKeyDim = 256;

// log(1 + y) for y >= 0, accurate where y is small, which a decaying gate's softplus reaches.
static inline float logOnePlus( float y )
{
    return y < 1e-4f ? y * ( 1.0f - y * ( 0.5f - y * ( 1.0f / 3.0f ) ) ) : precise::log( 1.0f + y );
}

// torch.nn.functional.softplus, with its cutover at 20.
static inline float softplusOf( float x )
{
    return x > 20.0f ? x : logOnePlus( precise::exp( x ) );
}

static inline float sumOverKeyParts( float value )
{
    value += simd_shuffle_xor( value, 1 );

    return value + simd_shuffle_xor( value, 2 );
}

template<typename TElement, uint TSlots>
kernel void gated_delta_rule(
    device const TElement* q [[buffer(0)]],
    device const TElement* k [[buffer(1)]],
    device const TElement* v [[buffer(2)]],
    device const TElement* a [[buffer(3)]],
    device const TElement* b [[buffer(4)]],
    device const TElement* a_log [[buffer(5)]],
    device const TElement* dt_bias [[buffer(6)]],
    device float* state [[buffer(7)]],
    device TElement* output [[buffer(8)]],
    constant GatedDeltaRuleShape& shape [[buffer(9)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threads [[threads_per_threadgroup]] )
{
    threadgroup float q_shared[kDeltaRuleMostHeadKeyDim];
    threadgroup float k_shared[kDeltaRuleMostHeadKeyDim];

    const uint batch_index = group.x / shape.value_heads;
    const uint head = group.x % shape.value_heads;
    const uint key_head = head / ( shape.value_heads / shape.key_heads );

    const uint part = thread_index % kDeltaRuleKeySplit;
    const uint column = group.y * kDeltaRuleColumnsPerGroup + thread_index / kDeltaRuleKeySplit;
    const bool active = column < shape.head_value_dim;

    const uint key_width = shape.key_heads * shape.head_key_dim;
    const uint value_width = shape.value_heads * shape.head_value_dim;
    const uint state_base = ( batch_index * shape.value_heads + head ) * shape.head_key_dim * shape.head_value_dim;

    float column_state[TSlots];

    for ( uint s = 0; s < TSlots; ++s )
    {
        const uint d = s * kDeltaRuleKeySplit + part;
        column_state[s] = d < shape.head_key_dim && active ? state[state_base + d * shape.head_value_dim + column] : 0.0f;
    }

    const float scale = precise::rsqrt( float( shape.head_key_dim ) );
    const float decay_rate = -precise::exp( float( a_log[head] ) );
    const float head_dt_bias = float( dt_bias[head] );

    for ( uint t = 0; t < shape.steps; ++t )
    {
        const uint qk_offset = ( batch_index * shape.steps + t ) * key_width + key_head * shape.head_key_dim;

        for ( uint d = thread_index; d < shape.head_key_dim; d += threads.x )
        {
            q_shared[d] = float( q[qk_offset + d] );
            k_shared[d] = float( k[qk_offset + d] );
        }

        threadgroup_barrier( mem_flags::mem_threadgroup );

        float q_squares = 0.0f;
        float k_squares = 0.0f;

        for ( uint s = 0; s < TSlots; ++s )
        {
            const uint d = s * kDeltaRuleKeySplit + part;

            if ( d < shape.head_key_dim )
            {
                q_squares += q_shared[d] * q_shared[d];
                k_squares += k_shared[d] * k_shared[d];
            }
        }

        const float q_inverse_norm = precise::rsqrt( sumOverKeyParts( q_squares ) + 1e-6f ) * scale;
        const float k_inverse_norm = precise::rsqrt( sumOverKeyParts( k_squares ) + 1e-6f );

        const uint scalar_offset = ( batch_index * shape.steps + t ) * shape.value_heads + head;
        const float beta = 1.0f / ( 1.0f + precise::exp( -float( b[scalar_offset] ) ) );
        const float decay = precise::exp( decay_rate * softplusOf( float( a[scalar_offset] ) + head_dt_bias ) );

        const uint v_offset = ( batch_index * shape.steps + t ) * value_width + head * shape.head_value_dim + column;
        const float v_t = active ? float( v[v_offset] ) : 0.0f;

        float memory = 0.0f;

        for ( uint s = 0; s < TSlots; ++s )
        {
            const uint d = s * kDeltaRuleKeySplit + part;

            if ( d < shape.head_key_dim )
            {
                column_state[s] *= decay;
                memory += column_state[s] * ( k_shared[d] * k_inverse_norm );
            }
        }

        const float delta = ( v_t - sumOverKeyParts( memory ) ) * beta;

        float result = 0.0f;

        for ( uint s = 0; s < TSlots; ++s )
        {
            const uint d = s * kDeltaRuleKeySplit + part;

            if ( d < shape.head_key_dim )
            {
                column_state[s] += ( k_shared[d] * k_inverse_norm ) * delta;
                result += column_state[s] * ( q_shared[d] * q_inverse_norm );
            }
        }

        result = sumOverKeyParts( result );

        if ( active && part == 0 )
        {
            output[v_offset] = TElement( result );
        }

        // The next step overwrites q and k.
        threadgroup_barrier( mem_flags::mem_threadgroup );
    }

    for ( uint s = 0; s < TSlots; ++s )
    {
        const uint d = s * kDeltaRuleKeySplit + part;

        if ( d < shape.head_key_dim && active )
        {
            state[state_base + d * shape.head_value_dim + column] = column_state[s];
        }
    }
}

// Slots per lane for head_key_dim up to 32, 128 and 256.
#define MILA_GATED_DELTA_RULE( element, slots ) \
    template [[host_name( "gated_delta_rule_" #element "_" #slots )]] kernel void gated_delta_rule<element, slots>( \
        device const element*, device const element*, device const element*, device const element*, device const element*, \
        device const element*, device const element*, device float*, device element*, constant GatedDeltaRuleShape&, \
        uint3, uint, uint3 );

MILA_GATED_DELTA_RULE( half, 8 )
MILA_GATED_DELTA_RULE( half, 32 )
MILA_GATED_DELTA_RULE( half, 64 )
MILA_GATED_DELTA_RULE( bfloat, 8 )
MILA_GATED_DELTA_RULE( bfloat, 32 )
MILA_GATED_DELTA_RULE( bfloat, 64 )
