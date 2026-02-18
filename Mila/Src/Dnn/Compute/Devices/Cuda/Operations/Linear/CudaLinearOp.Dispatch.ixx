/**
 * @file CudaLinearOp.ixx
 * @brief CUDA implementation of Linear operation with two-phase cuBLASLt optimization.
 */

module;
#include <cublasLt.h>
#include <cuda_fp16.h>
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

import Dnn.Components.Linear;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.ComponentConfig;
import Compute.OperationBase;
import Compute.UnaryOperation;
import Compute.Precision;
import Compute.OperationRegistry;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.IExecutionContext;
import Compute.ExecutionContextTemplate;
import Compute.OperationType;
import Compute.MemoryResource;
import Compute.CudaDeviceMemoryResource;
import Compute.CudaDevice;
import Compute.CudaTensorDataType;
import CublasLt.Error;
import Utils.Logger;

// DEBUG:
import Dnn.TensorOps;
import Dnn.TensorHelpers;

namespace Mila::Dnn::Compute::Cuda::Linear
{

    namespace Detail
    {
        /**
         * @brief CUDA kernel dispatcher for Linear operations.
         *
         * Specialized for float, half, bfloat16, and FP8 native CUDA types.
         */
        template <typename TNative>
            requires std::is_same_v<TNative, float> ||
        std::is_same_v<TNative, half> ||
            std::is_same_v<TNative, nv_bfloat16> ||
            std::is_same_v<TNative, __nv_fp8_e4m3> ||
            std::is_same_v<TNative, __nv_fp8_e5m2>
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

        template <>
        struct cuda_matmul_impl<half>
        {
            cuda_matmul_impl() = default;

            static inline void forward(
                half* output,
                const half* input,
                const half* weight,
                const half* bias,
                int batch_size,
                int in_features,
                int out_features,
                cudaStream_t stream )
            {
                cuda_matmul_forward_fp16( output, input, weight, bias, batch_size, in_features, out_features, stream );
            }

            static inline void backward(
                half* input_grad,
                half* weight_grad,
                half* bias_grad,
                const half* output_grad,
                const half* input,
                const half* weight,
                int batch_size,
                int in_features,
                int out_features,
                cudaStream_t stream )
            {
                throw std::logic_error( "CudaLinearOp::backward for FP16 is not yet implemented" );
            }
        };

        template <>
        struct cuda_matmul_impl<nv_bfloat16>
        {
            cuda_matmul_impl() = default;

            static inline void forward(
                nv_bfloat16* output,
                const nv_bfloat16* input,
                const nv_bfloat16* weight,
                const nv_bfloat16* bias,
                int batch_size,
                int in_features,
                int out_features,
                cudaStream_t stream )
            {
                throw std::logic_error( "CudaLinearOp::forward for BF16 is not yet implemented" );
            }

            static inline void backward(
                nv_bfloat16* input_grad,
                nv_bfloat16* weight_grad,
                nv_bfloat16* bias_grad,
                const nv_bfloat16* output_grad,
                const nv_bfloat16* input,
                const nv_bfloat16* weight,
                int batch_size,
                int in_features,
                int out_features,
                cudaStream_t stream )
            {
                throw std::logic_error( "CudaLinearOp::backward for BF16 is not yet implemented" );
            }
        };

        template <>
        struct cuda_matmul_impl<__nv_fp8_e4m3>
        {
            cuda_matmul_impl() = default;

            static inline void forward(
                __nv_fp8_e4m3* output,
                const __nv_fp8_e4m3* input,
                const __nv_fp8_e4m3* weight,
                const __nv_fp8_e4m3* bias,
                int batch_size,
                int in_features,
                int out_features,
                cudaStream_t stream )
            {
                throw std::logic_error( "CudaLinearOp::forward for FP8_E4M3 is not yet implemented" );
            }

            static inline void backward(
                __nv_fp8_e4m3* input_grad,
                __nv_fp8_e4m3* weight_grad,
                __nv_fp8_e4m3* bias_grad,
                const __nv_fp8_e4m3* output_grad,
                const __nv_fp8_e4m3* input,
                const __nv_fp8_e4m3* weight,
                int batch_size,
                int in_features,
                int out_features,
                cudaStream_t stream )
            {
                throw std::logic_error( "CudaLinearOp::backward for FP8_E4M3 is not yet implemented" );
            }
        };

        template <>
        struct cuda_matmul_impl<__nv_fp8_e5m2>
        {
            cuda_matmul_impl() = default;

            static inline void forward(
                __nv_fp8_e5m2* output,
                const __nv_fp8_e5m2* input,
                const __nv_fp8_e5m2* weight,
                const __nv_fp8_e5m2* bias,
                int batch_size,
                int in_features,
                int out_features,
                cudaStream_t stream )
            {
                throw std::logic_error( "CudaLinearOp::forward for FP8_E5M2 is not yet implemented" );
            }

            static inline void backward(
                __nv_fp8_e5m2* input_grad,
                __nv_fp8_e5m2* weight_grad,
                __nv_fp8_e5m2* bias_grad,
                const __nv_fp8_e5m2* output_grad,
                const __nv_fp8_e5m2* input,
                const __nv_fp8_e5m2* weight,
                int batch_size,
                int in_features,
                int out_features,
                cudaStream_t stream )
            {
                throw std::logic_error( "CudaLinearOp::backward for FP8_E5M2 is not yet implemented" );
            }
        };
    }
}
