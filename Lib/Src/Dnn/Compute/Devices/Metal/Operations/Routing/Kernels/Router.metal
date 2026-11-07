// Mixture-of-experts selection: router logits to the top-k experts and their combine weights, CUDA's router step for step.
//
// One SIMD-group per row. Each of top_k rounds takes every lane's best unselected logit and agrees on the row's across the group, the
// greater logit winning and the lower index on a tie. The softmax denominator cancels in the renormalization, so only the selected
// experts' exponentials are taken, one per lane, in 32 bits where CUDA takes them in 64: Metal has no double.

#include <metal_stdlib>

using namespace metal;

// Largest top_k one lane's selection holds, CUDA's kMaximumTopK.
constant constexpr int kRouterMaximumTopK = 16;

struct RouterShape
{
    int rows;
    int experts;
    int top_k;
};

inline bool router_ranks_above( float value, int expert, float other_value, int other_expert )
{
    return value > other_value || ( value == other_value && expert < other_expert );
}

template<typename TElement>
kernel void router_select(
    device const TElement* logits [[buffer(0)]],
    device const TElement* per_expert_scale [[buffer(1)]],
    device TElement* weights [[buffer(2)]],
    device int* indices [[buffer(3)]],
    constant RouterShape& shape [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 group [[threadgroup_position_in_grid]] )
{
    const int row = int( group.x );

    if ( row >= shape.rows )
    {
        return;
    }

    device const TElement* row_logits = logits + size_t( row ) * size_t( shape.experts );

    // Every lane holds the whole selection; lane s keeps slot s's logit for the weights below.
    int selected_expert[kRouterMaximumTopK];
    float lane_logit = 0.0f;

    for ( int slot = 0; slot < shape.top_k; ++slot )
    {
        float best_value = -FLT_MAX;
        int best_expert = shape.experts;

        for ( int expert = int( lane ); expert < shape.experts; expert += 32 )
        {
            bool taken = false;

            for ( int previous = 0; previous < slot; ++previous )
            {
                taken |= selected_expert[previous] == expert;
            }

            const float value = float( row_logits[expert] );

            if ( !taken && router_ranks_above( value, expert, best_value, best_expert ) )
            {
                best_value = value;
                best_expert = expert;
            }
        }

        for ( ushort offset = 16; offset > 0; offset >>= 1 )
        {
            const float other_value = simd_shuffle_xor( best_value, offset );
            const int other_expert = simd_shuffle_xor( best_expert, offset );

            if ( router_ranks_above( other_value, other_expert, best_value, best_expert ) )
            {
                best_value = other_value;
                best_expert = other_expert;
            }
        }

        selected_expert[slot] = best_expert;

        if ( int( lane ) == slot )
        {
            lane_logit = best_value;
        }
    }

    // Slot 0 holds the largest logit; subtracting it keeps every exponential at most one.
    const float largest = simd_broadcast_first( lane_logit );
    const float exponential = int( lane ) < shape.top_k ? precise::exp( lane_logit - largest ) : 0.0f;
    const float selected_mass = simd_sum( exponential );

    if ( int( lane ) < shape.top_k )
    {
        const int expert = selected_expert[lane];
        const float scale = expert < shape.experts ? float( per_expert_scale[expert] ) : NAN;
        const size_t slot_index = size_t( row ) * size_t( shape.top_k ) + lane;

        weights[slot_index] = TElement( exponential / selected_mass * scale );
        indices[slot_index] = expert;
    }
}

template [[host_name("router_select_half")]] kernel void router_select<half>(
    device const half*, device const half*, device half*, device int*, constant RouterShape&, uint, uint3 );
template [[host_name("router_select_bfloat")]] kernel void router_select<bfloat>(
    device const bfloat*, device const bfloat*, device bfloat*, device int*, constant RouterShape&, uint, uint3 );
