// Mixture-of-experts bank kernels: the gated pass and the combine pass. See Moe.cuh.

#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "device_launch_parameters.h"
#include "CudaUtils.h"
#include "Moe.cuh"
#include "Fp4E2M1.h"
// Shared activation math source, at the depth-relative path the elementwise kernel uses.
#include "../../../../../../Components/Activations/Activation/Kernels/ElementwiseActivation.h"

namespace Mila::Dnn::Compute::Cuda::Moe
{
    namespace
    {
        __device__ inline float to_float( float value ) { return value; }
        __device__ inline float to_float( __nv_bfloat16 value ) { return __bfloat162float( value ); }

        __device__ inline void store( float* destination, float value ) { *destination = value; }
        __device__ inline void store( __nv_bfloat16* destination, float value ) { *destination = __float2bfloat16( value ); }

        template<typename TNative, typename TFunctor>
        __global__ void moe_gated_kernel(
            const TNative* input, const TNative* gate_up, const int32_t* indices, float* gated,
            int tokens, int hidden, int intermediate, int experts, int top_k, TFunctor functor )
        {
            const int64_t cell = static_cast<int64_t>( blockIdx.x ) * blockDim.x + threadIdx.x;
            const int64_t per_token = static_cast<int64_t>( top_k ) * intermediate;

            if ( cell >= tokens * per_token )
            {
                return;
            }

            const int64_t token = cell / per_token;
            const int64_t slot = ( cell / intermediate ) % top_k;
            const int64_t unit = cell % intermediate;
            const int32_t expert = indices[ token * top_k + slot ];

            if ( expert < 0 || expert >= experts )
            {
                gated[ cell ] = NAN;
                return;
            }

            const int64_t expert_base = static_cast<int64_t>( expert ) * 2 * intermediate;
            const int64_t gate_offset = ( expert_base + unit ) * hidden;
            const int64_t up_offset = ( expert_base + intermediate + unit ) * hidden;
            const int64_t input_offset = token * hidden;

            float gate = 0.0f;
            float up = 0.0f;

            for ( int column = 0; column < hidden; ++column )
            {
                const float x = to_float( input[ input_offset + column ] );

                gate += to_float( gate_up[ gate_offset + column ] ) * x;
                up += to_float( gate_up[ up_offset + column ] ) * x;
            }

            gated[ cell ] = functor.fwd( gate ) * up;
        }

        template<typename TNative>
        __global__ void moe_combine_kernel(
            const float* gated, const TNative* down, const TNative* weights, const int32_t* indices, TNative* output,
            int tokens, int hidden, int intermediate, int experts, int top_k )
        {
            const int64_t cell = static_cast<int64_t>( blockIdx.x ) * blockDim.x + threadIdx.x;

            if ( cell >= static_cast<int64_t>( tokens ) * hidden )
            {
                return;
            }

            const int64_t token = cell / hidden;
            const int64_t column = cell % hidden;
            const int64_t gated_base = token * top_k * intermediate;

            float sum = 0.0f;

            for ( int slot = 0; slot < top_k; ++slot )
            {
                const int32_t expert = indices[ token * top_k + slot ];

                if ( expert < 0 || expert >= experts )
                {
                    sum = NAN;
                    break;
                }

                const int64_t down_offset = ( static_cast<int64_t>( expert ) * hidden + column ) * intermediate;
                const int64_t slot_offset = gated_base + static_cast<int64_t>( slot ) * intermediate;

                float projected = 0.0f;

                for ( int unit = 0; unit < intermediate; ++unit )
                {
                    projected += to_float( down[ down_offset + unit ] ) * gated[ slot_offset + unit ];
                }

                sum += to_float( weights[ token * top_k + slot ] ) * projected;
            }

            store( output + cell, sum );
        }

        template<typename TFunctor>
        __global__ void moe_gated_fp4_kernel(
            const __nv_bfloat16* input, const uint8_t* gate_up, const float* gate_up_scales, const int32_t* indices,
            float* gated, int tokens, int hidden, int intermediate, int experts, int top_k, int group_size,
            TFunctor functor )
        {
            const int64_t cell = static_cast<int64_t>( blockIdx.x ) * blockDim.x + threadIdx.x;
            const int64_t per_token = static_cast<int64_t>( top_k ) * intermediate;

            if ( cell >= tokens * per_token )
            {
                return;
            }

            const int64_t token = cell / per_token;
            const int64_t slot = ( cell / intermediate ) % top_k;
            const int64_t unit = cell % intermediate;
            const int32_t expert = indices[ token * top_k + slot ];

            if ( expert < 0 || expert >= experts )
            {
                gated[ cell ] = NAN;
                return;
            }

            const int64_t expert_base = static_cast<int64_t>( expert ) * 2 * intermediate;
            const int64_t packed_columns = hidden / 2;
            const int64_t groups = hidden / group_size;
            const uint8_t* gate_row = gate_up + ( expert_base + unit ) * packed_columns;
            const uint8_t* up_row = gate_up + ( expert_base + intermediate + unit ) * packed_columns;
            const float* gate_scales = gate_up_scales + ( expert_base + unit ) * groups;
            const float* up_scales = gate_up_scales + ( expert_base + intermediate + unit ) * groups;
            const int64_t input_offset = token * hidden;

            float gate = 0.0f;
            float up = 0.0f;

            for ( int column = 0; column < hidden; ++column )
            {
                const float x = __bfloat162float( input[ input_offset + column ] );
                const int shift = ( column & 1 ) * 4;
                const float gate_weight =
                    fp4_e2m1_decode( ( gate_row[ column / 2 ] >> shift ) & 0xFu ) * gate_scales[ column / group_size ];
                const float up_weight =
                    fp4_e2m1_decode( ( up_row[ column / 2 ] >> shift ) & 0xFu ) * up_scales[ column / group_size ];

                gate += gate_weight * x;
                up += up_weight * x;
            }

            gated[ cell ] = functor.fwd( gate ) * up;
        }

        __global__ void moe_combine_fp4_kernel(
            const float* gated, const uint8_t* down, const float* down_scales, const __nv_bfloat16* weights,
            const int32_t* indices, __nv_bfloat16* output,
            int tokens, int hidden, int intermediate, int experts, int top_k, int group_size )
        {
            const int64_t cell = static_cast<int64_t>( blockIdx.x ) * blockDim.x + threadIdx.x;

            if ( cell >= static_cast<int64_t>( tokens ) * hidden )
            {
                return;
            }

            const int64_t token = cell / hidden;
            const int64_t column = cell % hidden;
            const int64_t gated_base = token * top_k * intermediate;
            const int64_t packed_units = intermediate / 2;
            const int64_t groups = intermediate / group_size;

            float sum = 0.0f;

            for ( int slot = 0; slot < top_k; ++slot )
            {
                const int32_t expert = indices[ token * top_k + slot ];

                if ( expert < 0 || expert >= experts )
                {
                    sum = NAN;
                    break;
                }

                const int64_t row = static_cast<int64_t>( expert ) * hidden + column;
                const uint8_t* down_row = down + row * packed_units;
                const float* row_scales = down_scales + row * groups;
                const int64_t slot_offset = gated_base + static_cast<int64_t>( slot ) * intermediate;

                float projected = 0.0f;

                for ( int unit = 0; unit < intermediate; ++unit )
                {
                    const int shift = ( unit & 1 ) * 4;
                    const float weight =
                        fp4_e2m1_decode( ( down_row[ unit / 2 ] >> shift ) & 0xFu ) * row_scales[ unit / group_size ];

                    projected += weight * gated[ slot_offset + unit ];
                }

                sum += __bfloat162float( weights[ token * top_k + slot ] ) * projected;
            }

            output[ cell ] = __float2bfloat16( sum );
        }
    }

    template<typename TNative, typename TFunctor>
    void launch_moe_gated_forward(
        const TNative* input, const TNative* gate_up, const int32_t* indices, float* gated,
        int tokens, int hidden, int intermediate, int experts, int top_k,
        TFunctor functor, cudaStream_t stream )
    {
        const int64_t cells = static_cast<int64_t>( tokens ) * top_k * intermediate;

        if ( cells == 0 )
        {
            return;
        }

        constexpr int block_size = 512;
        const int grid_size = static_cast<int>( ( cells + block_size - 1 ) / block_size );

        moe_gated_kernel<TNative, TFunctor><<<grid_size, block_size, 0, stream>>>(
            input, gate_up, indices, gated, tokens, hidden, intermediate, experts, top_k, functor );

        cudaCheck( cudaGetLastError() );
    }

    template<typename TNative>
    void launch_moe_combine_forward(
        const float* gated, const TNative* down, const TNative* weights, const int32_t* indices, TNative* output,
        int tokens, int hidden, int intermediate, int experts, int top_k,
        cudaStream_t stream )
    {
        const int64_t cells = static_cast<int64_t>( tokens ) * hidden;

        if ( cells == 0 )
        {
            return;
        }

        constexpr int block_size = 512;
        const int grid_size = static_cast<int>( ( cells + block_size - 1 ) / block_size );

        moe_combine_kernel<TNative><<<grid_size, block_size, 0, stream>>>(
            gated, down, weights, indices, output, tokens, hidden, intermediate, experts, top_k );

        cudaCheck( cudaGetLastError() );
    }

    template<typename TFunctor>
    void launch_moe_gated_forward_fp4(
        const __nv_bfloat16* input, const uint8_t* gate_up, const float* gate_up_scales, const int32_t* indices,
        float* gated, int tokens, int hidden, int intermediate, int experts, int top_k, int group_size,
        TFunctor functor, cudaStream_t stream )
    {
        const int64_t cells = static_cast<int64_t>( tokens ) * top_k * intermediate;

        if ( cells == 0 )
        {
            return;
        }

        constexpr int block_size = 512;
        const int grid_size = static_cast<int>( ( cells + block_size - 1 ) / block_size );

        moe_gated_fp4_kernel<TFunctor><<<grid_size, block_size, 0, stream>>>(
            input, gate_up, gate_up_scales, indices, gated, tokens, hidden, intermediate, experts, top_k,
            group_size, functor );

        cudaCheck( cudaGetLastError() );
    }

    void launch_moe_combine_forward_fp4(
        const float* gated, const uint8_t* down, const float* down_scales, const __nv_bfloat16* weights,
        const int32_t* indices, __nv_bfloat16* output,
        int tokens, int hidden, int intermediate, int experts, int top_k, int group_size,
        cudaStream_t stream )
    {
        const int64_t cells = static_cast<int64_t>( tokens ) * hidden;

        if ( cells == 0 )
        {
            return;
        }

        constexpr int block_size = 512;
        const int grid_size = static_cast<int>( ( cells + block_size - 1 ) / block_size );

        moe_combine_fp4_kernel<<<grid_size, block_size, 0, stream>>>(
            gated, down, down_scales, weights, indices, output, tokens, hidden, intermediate, experts, top_k,
            group_size );

        cudaCheck( cudaGetLastError() );
    }

    template void launch_moe_gated_forward_fp4<Mila::Dnn::Activations::GeluTanh>(
        const __nv_bfloat16*, const uint8_t*, const float*, const int32_t*, float*, int, int, int, int, int, int,
        Mila::Dnn::Activations::GeluTanh, cudaStream_t );
    template void launch_moe_gated_forward_fp4<Mila::Dnn::Activations::Silu>(
        const __nv_bfloat16*, const uint8_t*, const float*, const int32_t*, float*, int, int, int, int, int, int,
        Mila::Dnn::Activations::Silu, cudaStream_t );

#define MILA_INSTANTIATE_MOE_GATED( NATIVE, FUNCTOR ) \
    template void launch_moe_gated_forward<NATIVE, FUNCTOR>( \
        const NATIVE*, const NATIVE*, const int32_t*, float*, int, int, int, int, int, FUNCTOR, cudaStream_t );

    MILA_INSTANTIATE_MOE_GATED( float, Mila::Dnn::Activations::GeluTanh )
    MILA_INSTANTIATE_MOE_GATED( float, Mila::Dnn::Activations::Silu )
    MILA_INSTANTIATE_MOE_GATED( __nv_bfloat16, Mila::Dnn::Activations::GeluTanh )
    MILA_INSTANTIATE_MOE_GATED( __nv_bfloat16, Mila::Dnn::Activations::Silu )

#undef MILA_INSTANTIATE_MOE_GATED

    template void launch_moe_combine_forward<float>(
        const float*, const float*, const float*, const int32_t*, float*, int, int, int, int, int, cudaStream_t );
    template void launch_moe_combine_forward<__nv_bfloat16>(
        const float*, const __nv_bfloat16*, const __nv_bfloat16*, const int32_t*, __nv_bfloat16*, int, int, int, int, int, cudaStream_t );
}
