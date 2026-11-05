// One run of columns of each input row, written densely. A split calls this once per output.

#include <metal_stdlib>

using namespace metal;

struct SplitColumns
{
    uint input_columns;
    uint first_column;
    uint columns;
};

template<typename TElement>
kernel void split_columns(
    device const TElement* input [[buffer(0)]],
    device TElement* output [[buffer(1)]],
    constant SplitColumns& shape [[buffer(2)]],
    uint index [[thread_position_in_grid]] )
{
    const uint row = index / shape.columns;

    output[index] = input[row * shape.input_columns + shape.first_column + index % shape.columns];
}

template [[host_name("split_columns_half")]] kernel void split_columns<half>( device const half*, device half*, constant SplitColumns&, uint );
template [[host_name("split_columns_bfloat")]] kernel void split_columns<bfloat>( device const bfloat*, device bfloat*, constant SplitColumns&, uint );
