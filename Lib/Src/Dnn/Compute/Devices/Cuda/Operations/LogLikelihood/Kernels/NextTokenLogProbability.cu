// Device-side next-token log-probability: one block per row of logits. The host reference is
// Dnn::nextTokenLogProbability (SequenceLogLikelihood.ixx), which this follows step for step.

#include "NextTokenLogProbability.cuh"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cfloat>
#include <cmath>
#include <cstdint>

namespace Mila::Dnn::Compute::Cuda::LogLikelihood
{
    namespace
    {
        constexpr int kBlock = 256;

        __device__ inline float to_float( float v ) { return v; }
        __device__ inline float to_float( __nv_bfloat16 v ) { return __bfloat162float( v ); }

        __device__ inline float capped( float logit, float softcap )
        {
            return softcap > 0.0f ? softcap * tanhf( logit / softcap ) : logit;
        }

        template <typename TNative>
        __global__ void next_token_log_probability_kernel(
            const TNative* logits, const int32_t* tokens, float* log_probabilities, int vocab, float softcap )
        {
            __shared__ float s_max[ kBlock ];
            __shared__ double s_sum[ kBlock ];

            const int row = blockIdx.x;
            const int tid = threadIdx.x;
            const TNative* row_logits = logits + static_cast<int64_t>( row ) * vocab;

            float local_max = -FLT_MAX;

            for ( int v = tid; v < vocab; v += kBlock )
            {
                local_max = fmaxf( local_max, capped( to_float( row_logits[ v ] ), softcap ) );
            }

            s_max[ tid ] = local_max;
            __syncthreads();

            for ( int stride = kBlock / 2; stride > 0; stride >>= 1 )
            {
                if ( tid < stride )
                {
                    s_max[ tid ] = fmaxf( s_max[ tid ], s_max[ tid + stride ] );
                }

                __syncthreads();
            }

            const float row_max = s_max[ 0 ];
            double local_sum = 0.0;

            for ( int v = tid; v < vocab; v += kBlock )
            {
                local_sum += exp( static_cast<double>( capped( to_float( row_logits[ v ] ), softcap ) - row_max ) );
            }

            s_sum[ tid ] = local_sum;
            __syncthreads();

            for ( int stride = kBlock / 2; stride > 0; stride >>= 1 )
            {
                if ( tid < stride )
                {
                    s_sum[ tid ] += s_sum[ tid + stride ];
                }

                __syncthreads();
            }

            if ( tid == 0 )
            {
                const int32_t target = tokens[ row + 1 ];

                if ( target < 0 || target >= vocab )
                {
                    log_probabilities[ row ] = __int_as_float( 0x7fc00000 );
                }
                else
                {
                    const float target_logit = capped( to_float( row_logits[ target ] ), softcap );

                    log_probabilities[ row ] = static_cast<float>(
                        static_cast<double>( target_logit - row_max ) - log( s_sum[ 0 ] ) );
                }
            }
        }

        template <typename TNative>
        void launch( const TNative* logits, const int32_t* tokens, float* log_probabilities,
            int rows, int vocab, float softcap, cudaStream_t stream )
        {
            if ( rows <= 0 )
                return;

            next_token_log_probability_kernel<TNative><<<rows, kBlock, 0, stream>>>(
                logits, tokens, log_probabilities, vocab, softcap );
        }
    }

    void cuda_next_token_log_probability_fp32(
        const float* logits, const int32_t* tokens, float* log_probabilities,
        int rows, int vocab, float softcap, cudaStream_t stream )
    {
        launch( logits, tokens, log_probabilities, rows, vocab, softcap, stream );
    }

    void cuda_next_token_log_probability_bf16(
        const __nv_bfloat16* logits, const int32_t* tokens, float* log_probabilities,
        int rows, int vocab, float softcap, cudaStream_t stream )
    {
        launch( logits, tokens, log_probabilities, rows, vocab, softcap, stream );
    }
}
