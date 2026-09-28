/**
 * @file CudaInt4Dequantize.cu
 * @brief INT4 weights -> BF16 staging expansion: one block per output channel, a 32-bit word (8 codes) per
 * thread step.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <format>
#include <stdexcept>
#include "CudaInt4Dequantize.cuh"

namespace Mila::Dnn::Compute::Cuda::Linear
{
    namespace
    {
        constexpr int kGroupSize = 32;
        constexpr int kCodesPerWord = 8;

        __global__ void dequantize_int4_to_bf16_kernel(
            __nv_bfloat16* __restrict__ output,
            const uint8_t* __restrict__ weights_packed,
            const __half* __restrict__  scales,
            int                         in_features )
        {
            const int64_t row = blockIdx.x;
            const int words = in_features / kCodesPerWord;
            const int groups = in_features / kGroupSize;

            const uint32_t* row_codes = reinterpret_cast<const uint32_t*>( weights_packed + row * ( in_features / 2 ) );
            const __half* row_scales = scales + row * groups;
            __nv_bfloat162* row_output = reinterpret_cast<__nv_bfloat162*>( output + row * in_features );

            for ( int word = static_cast<int>( threadIdx.x ); word < words; word += static_cast<int>( blockDim.x ) )
            {
                const uint32_t codes = row_codes[ word ];
                const float scale = __half2float( row_scales[ word * kCodesPerWord / kGroupSize ] );

#pragma unroll
                for ( int pair = 0; pair < kCodesPerWord / 2; ++pair )
                {
                    const float low = static_cast<float>( static_cast<int>( ( codes >> ( 8 * pair ) ) & 0xFu ) - 8 );
                    const float high = static_cast<float>( static_cast<int>( ( codes >> ( 8 * pair + 4 ) ) & 0xFu ) - 8 );

                    row_output[ word * ( kCodesPerWord / 2 ) + pair ] = __floats2bfloat162_rn( low * scale, high * scale );
                }
            }
        }
    }

    void cuda_int4_dequantize_to_bf16(
        __nv_bfloat16* output,
        const uint8_t* weights_packed,
        const __half*  scales,
        int            out_features,
        int            in_features,
        int            group_size,
        cudaStream_t   stream )
    {
        if ( group_size != kGroupSize || in_features % kGroupSize != 0 )
        {
            throw std::invalid_argument( std::format(
                "cuda_int4_dequantize_to_bf16: group_size {} with in_features {} is unsupported; the group must "
                "be {} and divide in_features", group_size, in_features, kGroupSize ) );
        }

        constexpr int kBlockSize = 256;

        dequantize_int4_to_bf16_kernel<<<static_cast<unsigned>( out_features ), kBlockSize, 0, stream>>>(
            output, weights_packed, scales, in_features );
    }

} // namespace Mila::Dnn::Compute::Cuda::Linear
