// output[token, :] = table[ids[token], :]. Unquantized, the elements move as bits; a six-bit table is decoded as it is read.

#include <metal_stdlib>

using namespace metal;

struct TokenEmbeddingShape
{
    uint embedding_dim;
};

template<typename TElement>
kernel void token_embedding(
    device const int* ids [[buffer(0)]],
    device const TElement* table [[buffer(1)]],
    device TElement* output [[buffer(2)]],
    constant TokenEmbeddingShape& shape [[buffer(3)]],
    uint index [[thread_position_in_grid]] )
{
    const uint token = index / shape.embedding_dim;
    const uint channel = index % shape.embedding_dim;

    output[index] = table[uint( ids[token] ) * shape.embedding_dim + channel];
}

template [[host_name("token_embedding_half")]] kernel void token_embedding<half>(
    device const int*, device const half*, device half*, constant TokenEmbeddingShape&, uint );
template [[host_name("token_embedding_bfloat")]] kernel void token_embedding<bfloat>(
    device const int*, device const bfloat*, device bfloat*, constant TokenEmbeddingShape&, uint );

// A six-bit table (Int6Packing.ixx): each row its codes' low nibbles, the even channel low, then from byte embedding_dim / 2
// their high two bits, four channels to a byte from bit 0; an FP16 scale for each 32 channels. ( code - 32 ) * scale is exact in
// 32 bits, so each output rounds once, as CUDA's gather does.
template<typename TElement>
kernel void token_embedding_int6(
    device const int* ids [[buffer(0)]],
    device const uchar* table [[buffer(1)]],
    device const half* scales [[buffer(2)]],
    device TElement* output [[buffer(3)]],
    constant TokenEmbeddingShape& shape [[buffer(4)]],
    uint index [[thread_position_in_grid]] )
{
    const uint token = index / shape.embedding_dim;
    const uint channel = index % shape.embedding_dim;
    const uint id = uint( ids[token] );

    device const uchar* row = table + id * ( shape.embedding_dim / 2 + shape.embedding_dim / 4 );

    const uint low = ( row[channel >> 1] >> ( ( channel & 1 ) * 4 ) ) & 15u;
    const uint high = ( row[shape.embedding_dim / 2 + ( channel >> 2 )] >> ( ( channel & 3 ) * 2 ) ) & 3u;
    const float scale = float( scales[id * ( shape.embedding_dim / 32 ) + channel / 32] );

    output[index] = TElement( float( int( low | ( high << 4 ) ) - 32 ) * scale );
}

template [[host_name("token_embedding_int6_half")]] kernel void token_embedding_int6<half>(
    device const int*, device const uchar*, device const half*, device half*, constant TokenEmbeddingShape&, uint );
template [[host_name("token_embedding_int6_bfloat")]] kernel void token_embedding_int6<bfloat>(
    device const int*, device const uchar*, device const half*, device bfloat*, constant TokenEmbeddingShape&, uint );
