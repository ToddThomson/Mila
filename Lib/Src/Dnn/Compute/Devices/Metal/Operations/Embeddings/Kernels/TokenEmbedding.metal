// output[token, :] = table[ids[token], :]. No arithmetic, so the elements move as bits.

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
