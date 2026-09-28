/**
 * @file CudaInt4Dequantize.cuh
 * @brief INT4 weights -> BF16 staging expansion for the staged prefill.
 */

#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace Mila::Dnn::Compute::Cuda::Linear
{
    /**
     * @brief Expand packed INT4 codes into BF16: output[n, k] = bf16( ( code[n, k] - 8 ) * scale[n, k / group_size] ).
     *
     * The product is exact in FP32 and rounds once, to nearest even, into BF16.
     *
     * @param output         Device BF16 staging buffer [out_features x in_features].
     * @param weights_packed Device uint8 codes [out_features x in_features/2], layout of Int4Packing.ixx.
     * @param scales         Device IEEE half scales [out_features x in_features/group_size].
     * @param group_size     Must be 32, dividing in_features.
     *
     * @throws std::invalid_argument on an unsupported group size.
     */
    void cuda_int4_dequantize_to_bf16(
        __nv_bfloat16* output,
        const uint8_t* weights_packed,
        const __half*  scales,
        int            out_features,
        int            in_features,
        int            group_size,
        cudaStream_t   stream );

} // namespace Mila::Dnn::Compute::Cuda::Linear
