/**
 * @file CudaLinearOp.ixx
 * @brief CUDA implementation of the Linear (Fully Connected) operation using
 *        the abstract TensorDataType + ExecutionContext UnaryOperation API.
 *
 * Ported to follow the CudaGeluOp pattern: uses ExecutionContext<DeviceType::Cuda>,
 * TensorDataType template, and the new OperationRegistry registration model.
 */

module;
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <vector>
#include <memory>
#include <string>
#include <stdexcept>
#include <exception>
#include "Kernels/CudaOps.h"
#include <type_traits>

export module Compute.CudaLinearOp;

import Dnn.Modules.Linear;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.ConfigurationBase;
import Compute.OperationBase;
import Compute.UnaryOperation;
import Compute.Precision;
import Compute.OperationRegistry;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.CudaExecutionContext;
import Compute.OperationType;
import Compute.OperationAttributes;
import Compute.MemoryResource;
import Compute.CudaDeviceMemoryResource;
import Compute.CudaDevice;
import Compute.CudaTensorDataType;
import Utils.Logger;

namespace Mila::Dnn::Compute
{
    using namespace Mila::Dnn;

    namespace Detail
    {
        template <typename TNative>
        struct cuda_matmul_impl;

        template <>
        struct cuda_matmul_impl<float>
        {
            static inline void forward(
                float* Y, const float* X,
                const float* weight, const float* bias,
                int outer_size, int C, int OC,
                cudaStream_t stream )
            {
                cuda_matmul_forward_fp32( Y, X, weight, bias, outer_size, C, OC, stream );
            }

            static inline void backward(
                float* dX, float* dW, float* dB,
                const float* dY, const float* X, const float* W,
                int outer_size, int C, int OC,
                cudaStream_t stream )
            {
                // FIXME: cuda_matmul_backward_input_fp32( dX, dY, W, outer_size, C, OC, stream );
                //cuda_matmul_backward_weight_fp32( dW, X, dY, outer_size, C, OC, stream );
                //if (dB) cuda_matmul_backward_bias_fp32( dB, dY, outer_size, OC, stream );
            }
        };

        template <>
        struct cuda_matmul_impl<half>
        {
            static inline void forward(
                half* Y, const half* X,
                const half* weight, const half* bias,
                int outer_size, int C, int OC,
                cudaStream_t stream )
            {
                cuda_matmul_forward_fp16( Y, X, weight, bias, outer_size, C, OC, stream );
            }

            static inline void backward(
                half* dX, half* dW, half* dB,
                const half* dY, const half* X, const half* W,
                int outer_size, int C, int OC,
                cudaStream_t stream )
            {
                // FIXME: cuda_matmul_backward_input_fp16( dX, dY, W, outer_size, C, OC, stream );
                //cuda_matmul_backward_weight_fp16( dW, X, dY, outer_size, C, OC, stream );
                //if (dB) cuda_matmul_backward_bias_fp16( dB, dY, outer_size, OC, stream );
            }
        };

        template <>
        struct cuda_matmul_impl<nv_bfloat16>
        {
            static inline void forward(
                nv_bfloat16* Y, const nv_bfloat16* X,
                const nv_bfloat16* weight, const nv_bfloat16* bias,
                int outer_size, int C, int OC,
                cudaStream_t stream )
            {
                // Fallback to FP16 kernel or dedicated BF16 kernel if available
                // cuda_matmul_forward_bf16( Y, X, weight, bias, outer_size, C, OC, stream );
            }

            static inline void backward(
                nv_bfloat16* dX, nv_bfloat16* dW, nv_bfloat16* dB,
                const nv_bfloat16* dY, const nv_bfloat16* X, const nv_bfloat16* W,
                int outer_size, int C, int OC,
                cudaStream_t stream )
            {
                // Implement when BF16 backward kernels exist
            }
        };

        template <>
        struct cuda_matmul_impl<__nv_fp8_e4m3>
        {
            static inline void forward(
                __nv_fp8_e4m3* Y, const __nv_fp8_e4m3* X,
                const __nv_fp8_e4m3* weight, const __nv_fp8_e4m3* bias,
                int outer_size, int C, int OC,
                cudaStream_t stream )
            {
                // Implement FP8 forward kernel wrapper when available
            }

            static inline void backward(
                __nv_fp8_e4m3* dX, __nv_fp8_e4m3* dW, __nv_fp8_e4m3* dB,
                const __nv_fp8_e4m3* dY, const __nv_fp8_e4m3* X, const __nv_fp8_e4m3* W,
                int outer_size, int C, int OC,
                cudaStream_t stream )
            {
                // Implement FP8 backward kernel wrapper when available
            }
        };

        template <>
        struct cuda_matmul_impl<__nv_fp8_e5m2>
        {
            static inline void forward(
                __nv_fp8_e5m2* Y, const __nv_fp8_e5m2* X,
                const __nv_fp8_e5m2* weight, const __nv_fp8_e5m2* bias,
                int outer_size, int C, int OC,
                cudaStream_t stream )
            {
                // Implement FP8 forward kernel wrapper when available
            }

            static inline void backward(
                __nv_fp8_e5m2* dX, __nv_fp8_e5m2* dW, __nv_fp8_e5m2* dB,
                const __nv_fp8_e5m2* dY, const __nv_fp8_e5m2* X, const __nv_fp8_e5m2* W,
                int outer_size, int C, int OC,
                cudaStream_t stream )
            {
                // Implement FP8 backward kernel wrapper when available
            }
        };
    } // namespace Detail

    export template<TensorDataType TPrecision>
        requires ValidFloatTensorDataType<TPrecision>
    class CudaLinearOp : public UnaryOperation<DeviceType::Cuda, TPrecision>
    {
    public:
        using MR = CudaDeviceMemoryResource;
        using UnaryOperationBase = UnaryOperation<DeviceType::Cuda, TPrecision>;
        using TensorType = Tensor<TPrecision, MR>;
        using Parameters = std::vector<std::shared_ptr<TensorType>>;
        using OutputState = std::vector<std::shared_ptr<TensorType>>;
        using NativeType = typename Mila::Dnn::Compute::Cuda::TensorDataTypeMap<TPrecision>::native_type;
        using CudaExecutionContext = ExecutionContext<DeviceType::Cuda>;

        CudaLinearOp( std::shared_ptr<CudaExecutionContext> context, const LinearConfig& config )
            : context_( context ), config_( config )
        {
            if (!context_)
            {
                throw std::invalid_argument( "CudaExecutionContext cannot be null." );
            }
            config_.validate();
        }

        void forward(
            const ITensor& input,
            const Parameters& parameters,
            ITensor& output,
            OutputState& output_state ) const override
        {
            if (input.getDeviceType() != DeviceType::Cuda || output.getDeviceType() != DeviceType::Cuda)
            {
                throw std::invalid_argument( "CudaLinearOp: Input and output tensors must be on CUDA device." );
            }

            const NativeType* X = static_cast<const NativeType*>(input.rawData());
            NativeType* Y = static_cast<NativeType*>(output.rawData());

            if (!X || !Y)
            {
                throw std::runtime_error( "CudaLinearOp::forward - null tensor data pointer" );
            }

            if (parameters.empty() || !parameters[0])
            {
                throw std::invalid_argument( "CudaLinearOp::forward requires weight parameter" );
            }

            const NativeType* weight = static_cast<const NativeType*>(parameters[0]->rawData());
            const NativeType* bias = nullptr;
            if (parameters.size() > 1 && parameters[1])
            {
                bias = static_cast<const NativeType*>(parameters[1]->rawData());
            }

            const auto& shape = input.shape();
            if (shape.size() < 2)
            {
                throw std::runtime_error( "CudaLinearOp::forward - expected input rank >= 2" );
            }

            int C = static_cast<int>(shape.back());
            int outer_size = 1;
            for (size_t i = 0; i + 1 < shape.size(); ++i) outer_size *= static_cast<int>(shape[i]);

            const auto& out_shape = output.shape();
            if (out_shape.size() < 2)
            {
                throw std::runtime_error( "CudaLinearOp::forward - expected output rank >= 2" );
            }
            int OC = static_cast<int>(out_shape.back());

            cudaStream_t stream = context_->getStream();
            auto precision_policy = config_.getPrecisionPolicy();

            // Try using cuBLASLt if available
            try
            {
                cublasLtHandle_t cublasLtHandle = context_->getCublasLtHandle();
                if (cublasLtHandle)
                {
                    // Optional: user may implement cublaslt path here (commented earlier)
                    // cublaslt_matmul_forward<NativeType>( Y, X, weight, bias, outer_size, C, OC, stream, cublasLtHandle, precision_policy );
                    return;
                }
            }
            catch (const std::exception& e)
            {
                Utils::Logger::warning( std::string( "cuBLASLt path failed, falling back to custom kernel: " ) + e.what() );
            }

            // Fallback to custom CUDA kernel wrapper
            Detail::cuda_matmul_impl<NativeType>::forward( Y, X, weight, bias, outer_size, C, OC, stream );
        }

        void backward(
            const ITensor& grad_output,
            const ITensor& input,
            const Parameters& parameters,
            const OutputState& output_state,
            ITensor& grad_input,
            Parameters& grad_parameters ) const override
        {
            if (input.getDeviceType() != DeviceType::Cuda || grad_output.getDeviceType() != DeviceType::Cuda || grad_input.getDeviceType() != DeviceType::Cuda)
            {
                throw std::invalid_argument( "CudaLinearOp::backward: tensors must be on CUDA device." );
            }

            const NativeType* X = static_cast<const NativeType*>(input.rawData());
            const NativeType* dY = static_cast<const NativeType*>(grad_output.rawData());
            NativeType* dX = static_cast<NativeType*>(grad_input.rawData());

            if (!X || !dY || !dX)
            {
                throw std::runtime_error( "CudaLinearOp::backward - null tensor data pointer" );
            }

            if (parameters.empty() || !parameters[0])
            {
                throw std::invalid_argument( "CudaLinearOp::backward requires weight parameter" );
            }

            const NativeType* W = static_cast<const NativeType*>(parameters[0]->rawData());

            NativeType* dW = nullptr;
            NativeType* dB = nullptr;
            if (grad_parameters.size() > 0 && grad_parameters[0])
            {
                dW = static_cast<NativeType*>(grad_parameters[0]->rawData());
            }
            if (grad_parameters.size() > 1 && grad_parameters[1])
            {
                dB = static_cast<NativeType*>(grad_parameters[1]->rawData());
            }

            const auto& shape = input.shape();
            int C = static_cast<int>(shape.back());
            int outer_size = 1;
            for (size_t i = 0; i + 1 < shape.size(); ++i) outer_size *= static_cast<int>(shape[i]);

            const auto& out_shape = grad_output.shape();
            int OC = static_cast<int>(out_shape.back());

            cudaStream_t stream = context_->getStream();

            Detail::cuda_matmul_impl<NativeType>::backward(
                dX, dW, dB,
                dY, X, W,
                outer_size, C, OC,
                stream );
        }

        OperationType getOperationType() const override
        {
            return OperationType::LinearOp;
        }

        std::string getName() const override
        {
            return "Cuda::LinearOp";
        }

        const LinearConfig& getConfig() const
        {
            return config_;
        }

    private:
        LinearConfig config_;
        std::shared_ptr<CudaExecutionContext> context_;
    };

    // Registrar: register supported precisions
    export class CudaLinearOpRegistrar
    {
    public:
        static void registerOperations()
        {
            const std::string opName = "LinearOp";

            OperationRegistry::instance().registerUnaryOperation<DeviceType::Cuda, TensorDataType::FP32>(
                opName,
                []( std::shared_ptr<ExecutionContext<DeviceType::Cuda>> context,
                    const ConfigurationBase& config ) -> std::shared_ptr<UnaryOperation<DeviceType::Cuda, TensorDataType::FP32>>
                {
                    const auto& linearConfig = static_cast<const LinearConfig&>(config);
                    return std::make_shared<CudaLinearOp<TensorDataType::FP32>>( context, linearConfig );
                }
            );

            OperationRegistry::instance().registerUnaryOperation<DeviceType::Cuda, TensorDataType::FP16>(
                opName,
                []( std::shared_ptr<ExecutionContext<DeviceType::Cuda>> context,
                    const ConfigurationBase& config ) -> std::shared_ptr<UnaryOperation<DeviceType::Cuda, TensorDataType::FP16>>
                {
                    const auto& linearConfig = static_cast<const LinearConfig&>(config);
                    return std::make_shared<CudaLinearOp<TensorDataType::FP16>>( context, linearConfig );
                }
            );

            OperationRegistry::instance().registerUnaryOperation<DeviceType::Cuda, TensorDataType::BF16>(
                opName,
                []( std::shared_ptr<ExecutionContext<DeviceType::Cuda>> context,
                    const ConfigurationBase& config ) -> std::shared_ptr<UnaryOperation<DeviceType::Cuda, TensorDataType::BF16>>
                {
                    const auto& linearConfig = static_cast<const LinearConfig&>(config);
                    return std::make_shared<CudaLinearOp<TensorDataType::BF16>>( context, linearConfig );
                }
            );

            OperationRegistry::instance().registerUnaryOperation<DeviceType::Cuda, TensorDataType::FP8_E4M3>(
                opName,
                []( std::shared_ptr<ExecutionContext<DeviceType::Cuda>> context,
                    const ConfigurationBase& config ) -> std::shared_ptr<UnaryOperation<DeviceType::Cuda, TensorDataType::FP8_E4M3>>
                {
                    const auto& linearConfig = static_cast<const LinearConfig&>(config);
                    return std::make_shared<CudaLinearOp<TensorDataType::FP8_E4M3>>( context, linearConfig );
                }
            );

            OperationRegistry::instance().registerUnaryOperation<DeviceType::Cuda, TensorDataType::FP8_E5M2>(
                opName,
                []( std::shared_ptr<ExecutionContext<DeviceType::Cuda>> context,
                    const ConfigurationBase& config ) -> std::shared_ptr<UnaryOperation<DeviceType::Cuda, TensorDataType::FP8_E5M2>>
                {
                    const auto& linearConfig = static_cast<const LinearConfig&>(config);
                    return std::make_shared<CudaLinearOp<TensorDataType::FP8_E5M2>>( context, linearConfig );
                }
            );
        }

        static inline bool isRegistered = []() {
            registerOperations();
            return true;
            }();
    };
}