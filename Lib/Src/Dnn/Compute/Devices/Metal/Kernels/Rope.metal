// Rotary positions: each (token, pair) calculates its angle once and rotates that pair in every Q and K head of the token.
// Input and output may alias, since each pair is written only by the thread that read it. A decode step reads its position from
// memory, never from an argument, so that its work is a pure function of device memory (CLAUDE.md, DecodeGraph.md).

#include <metal_stdlib>

using namespace metal;

struct RopeParameters
{
    float base;
    int rope_pairs;
    int frequency_denominator;
    float scaling_factor;
    float scaling_low_frequency_factor;
    float scaling_high_frequency_factor;
    int scaling_original_context_length;
    int tokens_in_chunk;
    int heads;
    int kv_heads;
    int head_dim;
    int pair_half;
    int position_offset;
};

// Llama 3's rescaling of one inverse frequency, as HuggingFace's _compute_llama3_parameters has it. 0 leaves it unchanged.
inline float rope_scaled_frequency( float inverse_frequency, constant RopeParameters& parameters )
{
    if ( parameters.scaling_original_context_length <= 0 )
    {
        return inverse_frequency;
    }

    const float original = float( parameters.scaling_original_context_length );
    const float wavelength = 2.0f * M_PI_F / inverse_frequency;

    if ( wavelength > original / parameters.scaling_low_frequency_factor )
    {
        return inverse_frequency / parameters.scaling_factor;
    }

    if ( wavelength < original / parameters.scaling_high_frequency_factor )
    {
        return inverse_frequency;
    }

    const float smooth = ( original / wavelength - parameters.scaling_low_frequency_factor )
        / ( parameters.scaling_high_frequency_factor - parameters.scaling_low_frequency_factor );

    return ( 1.0f - smooth ) * inverse_frequency / parameters.scaling_factor + smooth * inverse_frequency;
}

template<typename TElement>
inline void rope_rotate_pair(
    device TElement* out_q, device const TElement* in_q, device TElement* out_k, device const TElement* in_k,
    constant RopeParameters& parameters, int first_position, uint index )
{
    const int token = int( index ) / parameters.pair_half;
    const int pair = int( index ) % parameters.pair_half;
    const int position = first_position + token % parameters.tokens_in_chunk;

    float cosine = 1.0f;
    float sine = 0.0f;

    // A pair past rope_pairs carries zero frequency, so its rotation is the identity.
    if ( pair < parameters.rope_pairs )
    {
        const float exponent = -2.0f * float( pair ) / float( parameters.frequency_denominator );
        const float angle = float( position ) * rope_scaled_frequency( pow( parameters.base, exponent ), parameters );

        cosine = cos( angle );
        sine = sin( angle );
    }

    for ( int h = 0; h < parameters.heads + parameters.kv_heads; ++h )
    {
        const bool is_query = h < parameters.heads;
        const int first = is_query
            ? ( token * parameters.heads + h ) * parameters.head_dim
            : ( token * parameters.kv_heads + ( h - parameters.heads ) ) * parameters.head_dim;

        const int a = first + pair;
        const int b = a + parameters.pair_half;

        const float x0 = float( is_query ? in_q[a] : in_k[a] );
        const float x1 = float( is_query ? in_q[b] : in_k[b] );

        const TElement r0 = TElement( x0 * cosine - x1 * sine );
        const TElement r1 = TElement( x0 * sine + x1 * cosine );

        if ( is_query )
        {
            out_q[a] = r0;
            out_q[b] = r1;
        } else {
            out_k[a] = r0;
            out_k[b] = r1;
        }
    }
}

template<typename TElement>
kernel void rope_prefill(
    device TElement* out_q [[buffer(0)]],
    device const TElement* in_q [[buffer(1)]],
    device TElement* out_k [[buffer(2)]],
    device const TElement* in_k [[buffer(3)]],
    constant RopeParameters& parameters [[buffer(4)]],
    uint index [[thread_position_in_grid]] )
{
    rope_rotate_pair<TElement>( out_q, in_q, out_k, in_k, parameters, parameters.position_offset, index );
}

template<typename TElement>
kernel void rope_decode(
    device TElement* out_q [[buffer(0)]],
    device const TElement* in_q [[buffer(1)]],
    device TElement* out_k [[buffer(2)]],
    device const TElement* in_k [[buffer(3)]],
    device const int* decode_position [[buffer(4)]],
    constant RopeParameters& parameters [[buffer(5)]],
    uint index [[thread_position_in_grid]] )
{
    rope_rotate_pair<TElement>( out_q, in_q, out_k, in_k, parameters, *decode_position, index );
}

template [[host_name("rope_prefill_half")]] kernel void rope_prefill<half>(
    device half*, device const half*, device half*, device const half*, constant RopeParameters&, uint );
template [[host_name("rope_prefill_bfloat")]] kernel void rope_prefill<bfloat>(
    device bfloat*, device const bfloat*, device bfloat*, device const bfloat*, constant RopeParameters&, uint );
template [[host_name("rope_decode_half")]] kernel void rope_decode<half>(
    device half*, device const half*, device half*, device const half*, device const int*, constant RopeParameters&, uint );
template [[host_name("rope_decode_bfloat")]] kernel void rope_decode<bfloat>(
    device bfloat*, device const bfloat*, device bfloat*, device const bfloat*, device const int*, constant RopeParameters&, uint );
