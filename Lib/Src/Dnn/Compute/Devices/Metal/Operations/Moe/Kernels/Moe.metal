// Mixture-of-experts bank, CUDA's two passes over the stacked expert tensors in place.
//
// Pass 1, gated[token, slot, i] = gate( g ) * u in 32 bits, g and u rows i and I + i of gate_up[ expert ] applied to the token. Pass 2,
// output[token, j] = the sum over slots of weight * ( down[ expert ][ j ] . gated ). One SIMD-group per output cell, its lanes striding
// the row, so a cell depends on its own token alone: a call of several tokens gives each the bits a call of one gives it. An expert
// index out of range writes not-a-number for its cell. Packed rows read the codes as LinearQuantized.metal's Fp4Codes and Int4Codes
// do, eight at a time, each eight's sum scaled by its group's scale.

#include <metal_stdlib>

using namespace metal;

constant constexpr uint kMoeSimdgroups = 4;

struct MoeShape
{
    uint tokens;
    uint hidden;
    uint intermediate;
    uint experts;
    uint top_k;
    uint group;
};

// GELU's tanh form, Gemma's gate, as Geglu.metal computes it.
struct MoeGeluTanh
{
    static float apply( float x )
    {
        return 0.5f * x * ( 1.0f + precise::tanh( 0.7978845608f * ( x + 0.044715f * x * x * x ) ) );
    }
};

struct MoeSilu
{
    static float apply( float x )
    {
        return x / ( 1.0f + precise::exp( -x ) );
    }
};

// The cell this SIMD-group computes, of `width` across the grid's first axis; false past the last.
inline bool moe_cell( uint3 group, uint simdgroup, uint width, thread uint& cell )
{
    cell = group.x * kMoeSimdgroups + simdgroup;

    return cell < width;
}

template<typename TElement, typename TActivation>
kernel void moe_gated(
    device const TElement* input [[buffer(0)]],
    device const TElement* gate_up [[buffer(1)]],
    device const int* indices [[buffer(2)]],
    device float* gated [[buffer(3)]],
    constant MoeShape& shape [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]] )
{
    uint cell;

    if ( !moe_cell( group, simdgroup, shape.top_k * shape.intermediate, cell ) )
    {
        return;
    }

    const uint token = group.y;
    const uint slot = cell / shape.intermediate;
    const uint unit = cell % shape.intermediate;
    const int expert = indices[token * shape.top_k + slot];
    const size_t destination = size_t( token ) * shape.top_k * shape.intermediate + cell;

    if ( expert < 0 || uint( expert ) >= shape.experts )
    {
        if ( lane == 0 )
        {
            gated[destination] = NAN;
        }

        return;
    }

    const size_t expert_base = size_t( expert ) * 2 * shape.intermediate;
    device const TElement* gate_row = gate_up + ( expert_base + unit ) * shape.hidden;
    device const TElement* up_row = gate_up + ( expert_base + shape.intermediate + unit ) * shape.hidden;
    device const TElement* x = input + size_t( token ) * shape.hidden;

    float gate = 0.0f;
    float up = 0.0f;

    for ( uint column = lane; column < shape.hidden; column += 32 )
    {
        const float value = float( x[column] );

        gate += float( gate_row[column] ) * value;
        up += float( up_row[column] ) * value;
    }

    gate = simd_sum( gate );
    up = simd_sum( up );

    if ( lane == 0 )
    {
        gated[destination] = TActivation::apply( gate ) * up;
    }
}

template<typename TElement>
kernel void moe_combine(
    device const float* gated [[buffer(0)]],
    device const TElement* down [[buffer(1)]],
    device const TElement* weights [[buffer(2)]],
    device const int* indices [[buffer(3)]],
    device TElement* output [[buffer(4)]],
    constant MoeShape& shape [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]] )
{
    uint column;

    if ( !moe_cell( group, simdgroup, shape.hidden, column ) )
    {
        return;
    }

    const uint token = group.y;
    device const float* token_gated = gated + size_t( token ) * shape.top_k * shape.intermediate;

    float sum = 0.0f;

    for ( uint slot = 0; slot < shape.top_k; ++slot )
    {
        const int expert = indices[token * shape.top_k + slot];

        if ( expert < 0 || uint( expert ) >= shape.experts )
        {
            sum = NAN;
            break;
        }

        device const TElement* down_row = down + ( size_t( expert ) * shape.hidden + column ) * shape.intermediate;
        device const float* slot_gated = token_gated + slot * shape.intermediate;

        float projected = 0.0f;

        for ( uint unit = lane; unit < shape.intermediate; unit += 32 )
        {
            projected += float( down_row[unit] ) * slot_gated[unit];
        }

        sum += float( weights[token * shape.top_k + slot] ) * simd_sum( projected );
    }

    if ( lane == 0 )
    {
        output[size_t( token ) * shape.hidden + column] = TElement( sum );
    }
}

// The dot of a packed row of `columns` codes with 32-bit values, each eight codes' sum scaled by its group's scale.
template<typename TCodes, typename TScale, typename TValue>
inline float moe_packed_dot( device const uchar* row, device const TScale* scales, device const TValue* values, uint columns, uint group, uint lane )
{
    float sum = 0.0f;

    for ( uint c = lane; c < columns / 8; c += 32 )
    {
        float4 first;
        float4 second;

        TCodes::chunk( row, c, columns, first, second );

        const uint at = c * 8;
        const float4 low = float4( float( values[at] ), float( values[at + 1] ), float( values[at + 2] ), float( values[at + 3] ) );
        const float4 high = float4( float( values[at + 4] ), float( values[at + 5] ), float( values[at + 6] ), float( values[at + 7] ) );

        sum += ( dot( first, low ) + dot( second, high ) ) * float( scales[at / group] );
    }

    return simd_sum( sum );
}

template<typename TCodes, typename TScale, typename TElement, typename TActivation>
kernel void moe_gated_packed(
    device const TElement* input [[buffer(0)]],
    device const uchar* gate_up [[buffer(1)]],
    device const TScale* gate_up_scales [[buffer(2)]],
    device const int* indices [[buffer(3)]],
    device float* gated [[buffer(4)]],
    constant MoeShape& shape [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]] )
{
    uint cell;

    if ( !moe_cell( group, simdgroup, shape.top_k * shape.intermediate, cell ) )
    {
        return;
    }

    const uint token = group.y;
    const uint slot = cell / shape.intermediate;
    const uint unit = cell % shape.intermediate;
    const int expert = indices[token * shape.top_k + slot];
    const size_t destination = size_t( token ) * shape.top_k * shape.intermediate + cell;

    if ( expert < 0 || uint( expert ) >= shape.experts )
    {
        if ( lane == 0 )
        {
            gated[destination] = NAN;
        }

        return;
    }

    const size_t expert_base = size_t( expert ) * 2 * shape.intermediate;
    const size_t gate_index = expert_base + unit;
    const size_t up_index = expert_base + shape.intermediate + unit;
    const uint row_bytes = shape.hidden / 2;
    const uint groups = shape.hidden / shape.group;
    device const TElement* x = input + size_t( token ) * shape.hidden;

    const float gate = moe_packed_dot<TCodes>( gate_up + gate_index * row_bytes, gate_up_scales + gate_index * groups, x, shape.hidden, shape.group, lane );
    const float up = moe_packed_dot<TCodes>( gate_up + up_index * row_bytes, gate_up_scales + up_index * groups, x, shape.hidden, shape.group, lane );

    if ( lane == 0 )
    {
        gated[destination] = TActivation::apply( gate ) * up;
    }
}

template<typename TCodes, typename TScale, typename TElement>
kernel void moe_combine_packed(
    device const float* gated [[buffer(0)]],
    device const uchar* down [[buffer(1)]],
    device const TScale* down_scales [[buffer(2)]],
    device const TElement* weights [[buffer(3)]],
    device const int* indices [[buffer(4)]],
    device TElement* output [[buffer(5)]],
    constant MoeShape& shape [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]] )
{
    uint column;

    if ( !moe_cell( group, simdgroup, shape.hidden, column ) )
    {
        return;
    }

    const uint token = group.y;
    const uint row_bytes = shape.intermediate / 2;
    const uint groups = shape.intermediate / shape.group;
    device const float* token_gated = gated + size_t( token ) * shape.top_k * shape.intermediate;

    float sum = 0.0f;

    for ( uint slot = 0; slot < shape.top_k; ++slot )
    {
        const int expert = indices[token * shape.top_k + slot];

        if ( expert < 0 || uint( expert ) >= shape.experts )
        {
            sum = NAN;
            break;
        }

        const size_t row = size_t( expert ) * shape.hidden + column;
        const float projected = moe_packed_dot<TCodes>(
            down + row * row_bytes, down_scales + row * groups, token_gated + slot * shape.intermediate, shape.intermediate, shape.group, lane );

        sum += float( weights[token * shape.top_k + slot] ) * projected;
    }

    if ( lane == 0 )
    {
        output[size_t( token ) * shape.hidden + column] = TElement( sum );
    }
}

#define MILA_MOE_GATED( ELEMENT, ACTIVATION, NAME ) \
    template [[host_name( "moe_gated_" NAME )]] kernel void moe_gated<ELEMENT, ACTIVATION>( \
        device const ELEMENT*, device const ELEMENT*, device const int*, device float*, constant MoeShape&, uint, uint, uint3 );

MILA_MOE_GATED( half, MoeGeluTanh, "gelu_half" )
MILA_MOE_GATED( bfloat, MoeGeluTanh, "gelu_bfloat" )
MILA_MOE_GATED( half, MoeSilu, "silu_half" )
MILA_MOE_GATED( bfloat, MoeSilu, "silu_bfloat" )

#define MILA_MOE_COMBINE( ELEMENT, NAME ) \
    template [[host_name( "moe_combine_" NAME )]] kernel void moe_combine<ELEMENT>( \
        device const float*, device const ELEMENT*, device const ELEMENT*, device const int*, device ELEMENT*, constant MoeShape&, uint, uint, uint3 );

MILA_MOE_COMBINE( half, "half" )
MILA_MOE_COMBINE( bfloat, "bfloat" )

#define MILA_MOE_GATED_PACKED( CODES, SCALE, ELEMENT, ACTIVATION, NAME ) \
    template [[host_name( "moe_gated_" NAME )]] kernel void moe_gated_packed<CODES, SCALE, ELEMENT, ACTIVATION>( \
        device const ELEMENT*, device const uchar*, device const SCALE*, device const int*, device float*, constant MoeShape&, uint, uint, uint3 );

#define MILA_MOE_COMBINE_PACKED( CODES, SCALE, ELEMENT, NAME ) \
    template [[host_name( "moe_combine_" NAME )]] kernel void moe_combine_packed<CODES, SCALE, ELEMENT>( \
        device const float*, device const uchar*, device const SCALE*, device const ELEMENT*, device const int*, device ELEMENT*, \
        constant MoeShape&, uint, uint, uint3 );

#define MILA_MOE_PACKED( CODES, SCALE, FORMAT ) \
    MILA_MOE_GATED_PACKED( CODES, SCALE, half, MoeGeluTanh, FORMAT "_gelu_half" ) \
    MILA_MOE_GATED_PACKED( CODES, SCALE, bfloat, MoeGeluTanh, FORMAT "_gelu_bfloat" ) \
    MILA_MOE_GATED_PACKED( CODES, SCALE, half, MoeSilu, FORMAT "_silu_half" ) \
    MILA_MOE_GATED_PACKED( CODES, SCALE, bfloat, MoeSilu, FORMAT "_silu_bfloat" ) \
    MILA_MOE_COMBINE_PACKED( CODES, SCALE, half, FORMAT "_half" ) \
    MILA_MOE_COMBINE_PACKED( CODES, SCALE, bfloat, FORMAT "_bfloat" )

MILA_MOE_PACKED( Fp4Codes, float, "fp4" )
MILA_MOE_PACKED( Int4Codes, half, "int4" )

#undef MILA_MOE_PACKED
#undef MILA_MOE_COMBINE_PACKED
#undef MILA_MOE_GATED_PACKED
#undef MILA_MOE_COMBINE
#undef MILA_MOE_GATED
