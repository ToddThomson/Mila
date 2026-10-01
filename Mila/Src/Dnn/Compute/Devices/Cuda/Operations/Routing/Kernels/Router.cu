// Mixture-of-experts selection: router logits to top-k experts and combine weights. One warp per
// row -- see Router.cuh.

#include <cmath>
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
        constexpr int kWarpsPerBlock = 8;

        __device__ inline float to_float( float value ) { return value; }
        __device__ inline float to_float( __nv_bfloat16 value ) { return __bfloat162float( value ); }

        __device__ inline void store( float* destination, float value ) { *destination = value; }
        __device__ inline void store( __nv_bfloat16* destination, float value ) { *destination = __float2bfloat16( value ); }

        // The greater logit, and on a tie the lower index, as the CPU op orders them. NaN never wins.
        __device__ inline bool ranksAbove( float value, int expert, float other_value, int other_expert )
        {
            return value > other_value || ( value == other_value && expert < other_expert );
        }

        template <typename TNative>
        __global__ void __launch_bounds__( 32 * kWarpsPerBlock ) router_select_kernel(
            const TNative* logits, const TNative* per_expert_scale,
            TNative* weights, int32_t* indices,
            int rows, int experts, int top_k )
        {
            const int row = blockIdx.x * kWarpsPerBlock + threadIdx.y;
            const int lane = threadIdx.x;

            if ( row >= rows )
            {
                return;
            }

            const TNative* row_logits = logits + static_cast<int64_t>( row ) * experts;

            // Every lane holds the whole selection; lane s keeps slot s's logit for the weights below.
            int selected_expert[ kMaximumTopK ];
            float lane_logit = 0.0f;

            for ( int slot = 0; slot < top_k; ++slot )
            {
                float best_value = -INFINITY;
                int best_expert = experts;

                for ( int expert = lane; expert < experts; expert += 32 )
                {
                    bool taken = false;

                    for ( int previous = 0; previous < slot; ++previous )
                    {
                        taken |= selected_expert[ previous ] == expert;
                    }

                    const float value = to_float( row_logits[ expert ] );

                    if ( !taken && ranksAbove( value, expert, best_value, best_expert ) )
                    {
                        best_value = value;
                        best_expert = expert;
                    }
                }

#pragma unroll
                for ( int offset = 16; offset > 0; offset >>= 1 )
                {
                    const float other_value = __shfl_xor_sync( 0xffffffff, best_value, offset );
                    const int other_expert = __shfl_xor_sync( 0xffffffff, best_expert, offset );

                    if ( ranksAbove( other_value, other_expert, best_value, best_expert ) )
                    {
                        best_value = other_value;
                        best_expert = other_expert;
                    }
                }

                selected_expert[ slot ] = best_expert;

                if ( lane == slot )
                {
                    lane_logit = best_value;
                }
            }

            // Slot 0 holds the largest logit; subtracting it keeps every exponential at most one.
            const float max_logit = __shfl_sync( 0xffffffff, lane_logit, 0 );
            const double exponential = lane < top_k ? exp( static_cast<double>( lane_logit ) - max_logit ) : 0.0;
            double selected_mass = exponential;

#pragma unroll
            for ( int offset = 16; offset > 0; offset >>= 1 )
            {
                selected_mass += __shfl_xor_sync( 0xffffffff, selected_mass, offset );
            }

            if ( lane < top_k )
            {
                int expert = experts;

                for ( int slot = 0; slot < top_k; ++slot )
                {
                    expert = slot == lane ? selected_expert[ slot ] : expert;
                }

                const float scale = expert < experts ? to_float( per_expert_scale[ expert ] ) : NAN;

                store( weights + static_cast<int64_t>( row ) * top_k + lane, static_cast<float>( exponential / selected_mass ) * scale );
                indices[ static_cast<int64_t>( row ) * top_k + lane ] = expert;
            }
        }

        template <typename TNative>
        inline void launch_router_select(
            const TNative* logits, const TNative* per_expert_scale,
            TNative* weights, int32_t* indices,
            int rows, int experts, int top_k, cudaStream_t stream )
        {
            if ( rows == 0 )
            {
                return;
            }

            const dim3 block( 32, kWarpsPerBlock );
            const int grid_size = ceil_div( rows, kWarpsPerBlock );

            router_select_kernel<TNative><<<grid_size, block, 0, stream>>>(
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
