// Device-side token sampling kernels. Phase A: greedy argmax over the logits row.

#include "Sampling.cuh"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cfloat>
#include <cstdint>

namespace Mila::Dnn::Compute::Cuda::Sampling
{
    namespace
    {
        __device__ inline float to_float( float v ) { return v; }
        __device__ inline float to_float( __nv_bfloat16 v ) { return __bfloat162float( v ); }

        // Single-block argmax: each thread grid-strides the vocab keeping its local
        // (max value, lowest index), then a shared-memory tree reduction picks the
        // global winner. Ties resolve to the lowest index to match std::max_element.
        template <typename TNative, int kBlock>
        __global__ void argmax_kernel( const TNative* logits, int32_t* token_out, int vocab )
        {
            __shared__ float s_val[kBlock];
            __shared__ int s_idx[kBlock];

            const int tid = threadIdx.x;
            float best = -FLT_MAX;
            int best_idx = 0;

            for ( int i = tid; i < vocab; i += kBlock )
            {
                const float v = to_float( logits[i] );

                if ( v > best )
                {
                    best = v;
                    best_idx = i;
                }
            }

            s_val[tid] = best;
            s_idx[tid] = best_idx;
            __syncthreads();

            for ( int stride = kBlock / 2; stride > 0; stride >>= 1 )
            {
                if ( tid < stride )
                {
                    const float other = s_val[tid + stride];

                    if ( other > s_val[tid] || ( other == s_val[tid] && s_idx[tid + stride] < s_idx[tid] ) )
                    {
                        s_val[tid] = other;
                        s_idx[tid] = s_idx[tid + stride];
                    }
                }

                __syncthreads();
            }

            if ( tid == 0 )
                token_out[0] = s_idx[0];
        }

        template <typename TNative>
        inline void launch_argmax( const TNative* logits, int32_t* token_out, int vocab, cudaStream_t stream )
        {
            constexpr int kBlock = 256;
            argmax_kernel<TNative, kBlock><<<1, kBlock, 0, stream>>>( logits, token_out, vocab );
        }
    }

    void cuda_sample_argmax_fp32( const float* logits, int32_t* token_out, int vocab, cudaStream_t stream )
    {
        launch_argmax( logits, token_out, vocab, stream );
    }

    void cuda_sample_argmax_bf16( const __nv_bfloat16* logits, int32_t* token_out, int vocab, cudaStream_t stream )
    {
        launch_argmax( logits, token_out, vocab, stream );
    }
}
