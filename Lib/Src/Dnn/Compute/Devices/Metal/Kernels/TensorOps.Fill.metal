// A scalar written to every element, converted from the host's representation by Metal's own conversion.

#include <metal_stdlib>

using namespace metal;

template<typename TElement, typename THostValue>
kernel void fill(
    device TElement* destination [[buffer(0)]],
    constant THostValue& value [[buffer(1)]],
    uint index [[thread_position_in_grid]] )
{
    destination[index] = static_cast<TElement>( value );
}

template [[host_name("fill_float")]] kernel void fill<float, float>( device float*, constant float&, uint );
template [[host_name("fill_half")]] kernel void fill<half, float>( device half*, constant float&, uint );
template [[host_name("fill_bfloat")]] kernel void fill<bfloat, float>( device bfloat*, constant float&, uint );
template [[host_name("fill_int")]] kernel void fill<int, int>( device int*, constant int&, uint );
template [[host_name("fill_uint")]] kernel void fill<uint, int>( device uint*, constant int&, uint );
template [[host_name("fill_short")]] kernel void fill<short, int>( device short*, constant int&, uint );
template [[host_name("fill_ushort")]] kernel void fill<ushort, int>( device ushort*, constant int&, uint );
template [[host_name("fill_char")]] kernel void fill<char, int>( device char*, constant int&, uint );
template [[host_name("fill_uchar")]] kernel void fill<uchar, int>( device uchar*, constant int&, uint );
