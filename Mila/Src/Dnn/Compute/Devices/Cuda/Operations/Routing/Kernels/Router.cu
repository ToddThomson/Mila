// Mixture-of-experts selection: router logits to top-k experts and combine weights. One thread per
// row with nothing shared -- see Router.cuh.

#include <cmath>
#include <cfloat>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "device_launch_parameters.h"
#include "CudaUtils.h"
#include "Router.cuh"

namespace Mila::Dnn::Compute::Cuda::Routing
{
    namespace
    {
        __device__ inline float to_float( float value ) { return value; }
        __device__ inline float to_float( __nv_bfloat16 value ) { return __bfloat162float( value ); }

        __device__ inline void store( float* destination, float value ) { *destination = value; }
        __device__ inline void store( __nv_bfloat16* destination, float value ) { *destination = __float2bfloat16( value ); }

        template <typename TNative>
        __global__ void router_select_kernel(
            const TNative* logits, const TNative* per_expert_scale,
            TNative* weights, int32_t* indices,
            int rows, int experts, int top_k )
        {
            const int row = blockIdx.x * blockDim.x + threadIdx.x;

            if ( row >= rows )
            {
                return;
            }

            const TNative* row_logits = logits + row * experts;

            float max_logit = -FLT_MAX;

            for ( int expert = 0; expert < experts; ++expert )
            {
                const float value = to_float( row_logits[ expert ] );

                if ( value > max_logit )
                {
                    max_logit = value;
                }
            }

            double total = 0.0;

            for ( int expert = 0; expert < experts; ++expert )
            {
                total += exp( static_cast<double>( to_float( row_logits[ expert ] ) ) - max_logit );
            }

            // Descending by logit. Scanning experts in ascending order and displacing only on a
            // strictly greater logit keeps the lower index ahead on a tie, as the CPU op does.
            float selected_logit[ kMaximumTopK ];
            int selected_expert[ kMaximumTopK ];
            int count = 0;

            for ( int expert = 0; expert < experts; ++expert )
            {
                const float value = to_float( row_logits[ expert ] );

                if ( count == top_k && !( value > selected_logit[ top_k - 1 ] ) )
                {
                    continue;
                }

                int position = count < top_k ? count : top_k - 1;

                while ( position > 0 && value > selected_logit[ position - 1 ] )
                {
                    selected_logit[ position ] = selected_logit[ position - 1 ];
                    selected_expert[ position ] = selected_expert[ position - 1 ];
                    --position;
                }

                selected_logit[ position ] = value;
                selected_expert[ position ] = expert;

                if ( count < top_k )
                {
                    ++count;
                }
            }

            float selected_mass = 0.0f;

            for ( int slot = 0; slot < top_k; ++slot )
            {
                selected_mass += static_cast<float>( exp( static_cast<double>( selected_logit[ slot ] ) - max_logit ) / total );
            }

            for ( int slot = 0; slot < top_k; ++slot )
            {
                const int expert = selected_expert[ slot ];
                const float probability = static_cast<float>( exp( static_cast<double>( selected_logit[ slot ] ) - max_logit ) / total );

                store( weights + row * top_k + slot, probability / selected_mass * to_float( per_expert_scale[ expert ] ) );
                indices[ row * top_k + slot ] = expert;
            }
        }

        template <typename TNative>
        inline void launch_router_select(
            const TNative* logits, const TNative* per_expert_scale,
            TNative* weights, int32_t* indices,
            int rows, int experts, int top_k, cudaStream_t stream )
        {
            constexpr int block_size = 256;
            const int grid_size = ceil_div( rows, block_size );

            router_select_kernel<TNative><<<grid_size, block_size, 0, stream>>>(
                logits, per_expert_scale, weights, indices, rows, experts, top_k );

            cudaCheck( cudaGetLastError() );
        }
    }

    void cuda_router_select_fp32(
        const float* logits, const float* per_expert_scale,
        float* weights, int32_t* indices,
        int rows, int experts, int top_k, cudaStream_t stream )
    {
        launch_router_select( logits, per_expert_scale, weights, indices, rows, experts, top_k, stream );
    }

    void cuda_router_select_bf16(
        const __nv_bfloat16* logits, const __nv_bfloat16* per_expert_scale,
        __nv_bfloat16* weights, int32_t* indices,
        int rows, int experts, int top_k, cudaStream_t stream )
    {
        launch_router_select( logits, per_expert_scale, weights, indices, rows, experts, top_k, stream );
    }
}
