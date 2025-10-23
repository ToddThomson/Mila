/**
 * @file CudaLayerNormOp.ixx
 * @brief CUDA implementation of Layer Normalization (TensorDataType-based).
 *
 * Ported to the ExecutionContext / TensorDataType UnaryOperation interface
 * following the pattern used by CudaGeluOp.
 */

module;
#include <vector>
#include <memory>
#include <string>
#include <stdexcept>
#include <cuda_fp16.h>
#include "Kernels/CudaOps.h"

export module Compute.CudaLayerNormOp;

import Dnn.Modules.LayerNorm;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.ConfigurationBase;
import Compute.OperationBase;
import Compute.UnaryOperation;
import Compute.OperationRegistry;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.CudaExecutionContext;
import Compute.CudaDeviceResources;
import Compute.OperationType;
import Compute.OperationAttributes;
import Compute.MemoryResource;
import Compute.CudaDeviceMemoryResource;
import Compute.CudaTensorDataType;
import Compute.CudaDevice;

namespace Mila::Dnn::Compute
{
    namespace Detail
    {
        // Primary template - will cause a compile error if no specialization exists
        template <typename TNative>
            requires std::is_same_v<TNative, float> || std::is_same_v<TNative, half>
        struct cuda_layernorm_impl;

        template <>
        struct cuda_layernorm_impl<float>
        {
            cuda_layernorm_impl() = default;

            static inline void forward(
                float* Y, const float* X,
                const float* weight, const float* bias,
                float* mean, float* rstd,
                int B, int T, int C, float epsilon,
                cudaStream_t stream )
            {
                cuda_layernorm_forward_fp32( Y, mean, rstd, X, weight, bias, B, T, C, epsilon, stream );
            }

            static inline void backward(
                float* dX, float* dweight, float* dbias,
                const float* dY, const float* X, const float* weight,
                const float* mean, const float* rstd,
                int B, int T, int C, cudaStream_t stream )
            {
                // FIXME: cuda_layernorm_backward_fp32( dX, dweight, dbias, dY, X, weight, mean, rstd, B, T, C, stream );
            }
        };

        template <>
        struct cuda_layernorm_impl<half>
        {
            cuda_layernorm_impl() = default;

            static inline void forward(
                half* Y, const half* X,
                const half* weight, const half* bias,
                half* mean, half* rstd,
                int B, int T, int C, float epsilon,
                cudaStream_t stream )
            {
                cuda_layernorm_forward_fp16( Y, mean, rstd, X, weight, bias, B, T, C, epsilon, stream );
            }

            static inline void backward(
                half* dX, half* dweight, half* dbias,
                const half* dY, const half* X, const half* weight,
                const half* mean, const half* rstd,
                int B, int T, int C, cudaStream_t stream )
            {
                // FIXME: cuda_layernorm_backward_fp16( dX, dweight, dbias, dY, X, weight, mean, rstd, B, T, C, stream );
            }
        };
    }

    using namespace Mila::Dnn;

    export template<TensorDataType TPrecision>
        requires ValidFloatTensorDataType<TPrecision>
    class CudaLayerNormOp : public UnaryOperation<DeviceType::Cuda, TPrecision>
    {
    public:
        using MR = CudaDeviceMemoryResource;
        using UnaryOperationBase = UnaryOperation<DeviceType::Cuda, TPrecision>;
        using TensorType = Tensor<TPrecision, MR>;
        using Parameters = std::vector<std::shared_ptr<TensorType>>;
        using OutputState = std::vector<std::shared_ptr<TensorType>>;
        using NativeType = typename Mila::Dnn::Compute::Cuda::TensorDataTypeMap<TPrecision>::native_type;
        using CudaExecutionContext = ExecutionContext<DeviceType::Cuda>;

        CudaLayerNormOp( std::shared_ptr<CudaExecutionContext> context, const LayerNormConfig& config )
            : context_( context ), config_( config ), impl_()
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
                throw std::invalid_argument( "CudaLayerNormOp: Input and output tensors must be on CUDA device." );
            }

            // Validate shapes and state
            const NativeType* X = static_cast<const NativeType*>(input.rawData());
            NativeType* Y = static_cast<NativeType*>(output.rawData());

            if (!X || !Y)
            {
                throw std::runtime_error( "CudaLayerNormOp::forward - null tensor data pointer" );
            }

            if (parameters.empty() || !parameters[0])
            {
                throw std::invalid_argument( "CudaLayerNormOp::forward requires weight parameter" );
            }

            const NativeType* weight = static_cast<const NativeType*>(parameters[0]->rawData());
            const NativeType* bias = (parameters.size() > 1 && parameters[1]) ? static_cast<const NativeType*>(parameters[1]->rawData()) : nullptr;

            if (output_state.size() < 2 || !output_state[0] || !output_state[1])
            {
                throw std::invalid_argument( "CudaLayerNormOp::forward requires output_state[0]=mean and output_state[1]=rstd tensors." );
            }

            NativeType* mean = static_cast<NativeType*>(output_state[0]->rawData());
            NativeType* rstd = static_cast<NativeType*>(output_state[1]->rawData());

            const auto& shape = input.shape();
            if (shape.size() < 3)
            {
                throw std::runtime_error( "CudaLayerNormOp::forward - expected input rank >= 3 [B,T,C]" );
            }

            int B = static_cast<int>(shape[0]);
            int T = static_cast<int>(shape[1]);
            int C = static_cast<int>(shape[2]);
            float epsilon = config_.getEpsilon();

            cudaStream_t stream = context_->getStream();

            Detail::cuda_layernorm_impl<NativeType>::forward(
                reinterpret_cast<NativeType*>(Y),
                reinterpret_cast<const NativeType*>(X),
                reinterpret_cast<const NativeType*>(weight),
                reinterpret_cast<const NativeType*>(bias),
                reinterpret_cast<NativeType*>(mean),
                reinterpret_cast<NativeType*>(rstd),
                B, T, C, epsilon,
                stream );
        }

        void backward(
            const ITensor& output_gradient,
            const ITensor& input,
            const Parameters& parameters,
            const OutputState& output_state,
            ITensor& input_gradient,
            Parameters& parameter_gradients ) const override
        {
            if (input.getDeviceType() != DeviceType::Cuda || output_gradient.getDeviceType() != DeviceType::Cuda || input_gradient.getDeviceType() != DeviceType::Cuda)
            {
                throw std::invalid_argument( "CudaLayerNormOp::backward: tensors must be on CUDA device." );
            }

            const auto& shape = input.shape();
            if (shape.size() < 3)
            {
                throw std::runtime_error( "CudaLayerNormOp::backward - expected input rank >= 3 [B,T,C]" );
            }

            int B = static_cast<int>(shape[0]);
            int T = static_cast<int>(shape[1]);
            int C = static_cast<int>(shape[2]);

            const NativeType* X = static_cast<const NativeType*>(input.rawData());
            const NativeType* dY = static_cast<const NativeType*>(output_gradient.rawData());
            NativeType* dX = static_cast<NativeType*>(input_gradient.rawData());

            if (!X || !dY || !dX)
            {
                throw std::runtime_error( "CudaLayerNormOp::backward - null tensor data pointer" );
            }

            if (parameters.empty() || !parameters[0])
            {
                throw std::invalid_argument( "CudaLayerNormOp::backward requires weight parameter" );
            }

            const NativeType* weight = static_cast<const NativeType*>(parameters[0]->rawData());

            NativeType* dweight = nullptr;
            NativeType* dbias = nullptr;
            if (parameter_gradients.size() > 0 && parameter_gradients[0])
            {
                dweight = static_cast<NativeType*>(parameter_gradients[0]->rawData());
            }
            if (parameter_gradients.size() > 1 && parameter_gradients[1])
            {
                dbias = static_cast<NativeType*>(parameter_gradients[1]->rawData());
            }

            if (output_state.size() < 2 || !output_state[0] || !output_state[1])
            {
                throw std::invalid_argument( "CudaLayerNormOp::backward requires output_state[0]=mean and output_state[1]=rstd tensors." );
            }

            const NativeType* mean = static_cast<const NativeType*>(output_state[0]->rawData());
            const NativeType* rstd = static_cast<const NativeType*>(output_state[1]->rawData());

            cudaStream_t stream = context_->getStream();

            Detail::cuda_layernorm_impl<NativeType>::backward(
                reinterpret_cast<NativeType*>(dX),
                reinterpret_cast<NativeType*>(dweight),
                reinterpret_cast<NativeType*>(dbias),
                reinterpret_cast<const NativeType*>(dY),
                reinterpret_cast<const NativeType*>(X),
                reinterpret_cast<const NativeType*>(weight),
                reinterpret_cast<const NativeType*>(mean),
                reinterpret_cast<const NativeType*>(rstd),
                B, T, C, stream );
        }

        OperationType getOperationType() const override
        {
            return OperationType::LayerNormOp;
        }

        std::string getName() const override
        {
            return "Cuda::LayerNormOp";
        }

        const LayerNormConfig& getConfig() const
        {
            return config_;
        }

    private:
        LayerNormConfig config_;
        std::shared_ptr<CudaExecutionContext> context_;
        Detail::cuda_layernorm_impl<NativeType> impl_;
    };

    export class CudaLayerNormOpRegistrar
    {
    public:
        static void registerOperations()
        {
            const std::string opName = "LayerNormOp";

            OperationRegistry::instance().registerUnaryOperation<DeviceType::Cuda, TensorDataType::FP32>(
                opName,
                []( std::shared_ptr<ExecutionContext<DeviceType::Cuda>> context,
                    const ConfigurationBase& config ) -> std::shared_ptr<UnaryOperation<DeviceType::Cuda, TensorDataType::FP32>>
                {
                    const auto& lnConfig = static_cast<const LayerNormConfig&>(config);
                    return std::make_shared<CudaLayerNormOp<TensorDataType::FP32>>( context, lnConfig );
                }
            );

            OperationRegistry::instance().registerUnaryOperation<DeviceType::Cuda, TensorDataType::FP16>(
                opName,
                []( std::shared_ptr<ExecutionContext<DeviceType::Cuda>> context,
                    const ConfigurationBase& config ) -> std::shared_ptr<UnaryOperation<DeviceType::Cuda, TensorDataType::FP16>>
                {
                    const auto& lnConfig = static_cast<const LayerNormConfig&>(config);
                    return std::make_shared<CudaLayerNormOp<TensorDataType::FP16>>( context, lnConfig );
                }
            );
        }

        static inline bool isRegistered = []() {
            registerOperations();
            return true;
            }();
    };
}