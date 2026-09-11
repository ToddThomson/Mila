/**
 * @file CudaLinearOp.Dispatch.ixx
 * @brief CUDA implementation of Linear operation with two-phase cuBLASLt optimization.
 */

module;
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <vector>
#include <memory>
#include <string>
#include <format>
#include <stdexcept>
#include <exception>
#include <cstdint>
#include <type_traits>
#include <sstream>
#include <cassert>
#include <algorithm>
#include "Kernels/Linear.cuh"

export module Compute.CudaLinearOp:Dispatch;

import Dnn.Components.LinearConfig;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.ComponentConfig;
import Compute.OperationBase;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.IExecutionContext;
import Compute.OperationType;
import Compute.MemoryResource;
import Compute.CudaDeviceMemoryResource;
import Compute.CudaDevice;
import Compute.CudaTensorDataType;
import CublasLt.Error;
import Logging.Logger;

// DEBUG:
import Dnn.TensorOps;
import Dnn.TensorHelpers;

namespace Mila::Dnn::Compute::Cuda::Linear
{
    namespace Detail
    {
        /**
         * @brief CUDA kernel dispatcher for matrix-vector multiply (M=1 decode path).
         *
         * Specialized for float, bfloat16 native CUDA types.
         * Used exclusively for the decode path where batch_size=1 and T=1.
         */
        template <typename TComputeType, typename TWeightType = TComputeType>
            requires std::is_same_v<TComputeType, float> || std::is_same_v<TComputeType, nv_bfloat16>
        struct cuda_matvec_impl;

        template <>
        struct cuda_matvec_impl<float,float>
        {
            /**
             * @brief Dispatches M=1 matrix-vector decode pass for unquantized weights.
             */
            static void decode(
                float* y,
                const float* x,
                const float* weight,
                const float* /*scales*/, // Unused for FP32 path, present for API consistency with quantized decode
                const float* bias,
                int C,
                int OC,
                cudaStream_t stream )
            {
                cuda_matvec_decode_fp32( y, x, weight, bias, C, OC, stream );
            }
        };

        template <>
        struct cuda_matvec_impl<nv_bfloat16,nv_bfloat16>
        {
            /**
             * @brief Dispatches M=1 matrix-vector decode pass for unquantized weights.
             */
            static void decode(
                nv_bfloat16* y,
                const nv_bfloat16* x,
                const nv_bfloat16* weight,
                const float* /*scales*/, // Unused for FP32 path, present for API consistency with quantized decode
                const nv_bfloat16* bias,
                int C,
                int OC,
                cudaStream_t stream )
            {
                cuda_matvec_decode_bf16( y, x, weight, bias, C, OC, stream );
            }
        };

        // BF16 activation with FP8 weight -- Alpha.5 quantized decode path
        template<>
        struct cuda_matvec_impl<nv_bfloat16, __nv_fp8_e4m3>
        {
            static void decode(
                nv_bfloat16* y, const nv_bfloat16* x,
                const __nv_fp8_e4m3* weight, const float* scales,
                const nv_bfloat16* bias,
                int C, int OC, cudaStream_t stream )
            {
                cuda_matvec_decode_bf16_qfp8( y, x, weight, scales, bias, C, OC, stream );
            }
        };

        /**
         * @brief CUDA kernel dispatcher for Linear operations.
         *
         * Specialized for float, bfloat16 native CUDA types.
         */
        template <typename TNative>
            requires std::is_same_v<TNative, float> || std::is_same_v<TNative, nv_bfloat16>
        struct cuda_matmul_impl;

        template <>
        struct cuda_matmul_impl<float>
        {
            cuda_matmul_impl() = default;

            static inline void forward(
                float* output,
                const float* input,
                const float* weight,
                const float* bias,
                int batch_size,
                int in_features,
                int out_features,
                cudaStream_t stream )
            {
                cuda_matmul_forward_fp32( output, input, weight, bias, batch_size, in_features, out_features, stream );
            }

            static inline void backward(
                float* input_grad,
                float* weight_grad,
                float* bias_grad,
                const float* output_grad,
                const float* input,
                const float* weight,
                int batch_size,
                int in_features,
                int out_features,
                cudaStream_t stream )
            {
                cuda_matmul_backward_fp32( input_grad, weight_grad, bias_grad,
                    output_grad, input, weight,
                    batch_size, in_features, out_features, stream );
            }
        };
    }
}
