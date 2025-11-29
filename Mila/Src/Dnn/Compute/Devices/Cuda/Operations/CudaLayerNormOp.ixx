/**
 * @file CudaLayerNormOp.ixx
 * @brief CUDA implementation of Layer Normalization operation (TensorDataType-based).
 *
 * Ported to the ExecutionContext / TensorDataType UnaryOperation interface
 * following the pattern used by CpuLayerNormOp and CudaGeluOp.
 */

module;
#include <vector>
#include <memory>
#include <string>
#include <stdexcept>
#include <cstdint>
#include <cuda_fp16.h>
#include "Kernels/CudaOps.h"

export module Compute.CudaLayerNormOp;

import Dnn.Components.LayerNorm;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.TensorHostTypeMap;
import Dnn.TensorPartitioning;
import Dnn.ComponentConfig;
import Compute.OperationBase;
import Compute.UnaryOperation;
import Compute.OperationRegistry;
import Compute.DeviceType;
import Compute.IExecutionContext;
import Compute.ExecutionContext;
//import Compute.CudaExecutionContext;
import Compute.CudaDeviceResources;
import Compute.OperationType;
import Compute.MemoryResource;
import Compute.CudaDeviceMemoryResource;
import Compute.CudaTensorDataType;
import Compute.CudaDevice;
import Compute.Precision;

namespace Mila::Dnn::Compute
{
    namespace Detail
    {
        /**
         * @brief CUDA kernel dispatcher for LayerNorm operations.
         *
         * Specialized for float (FP32) and half (FP16) native CUDA types.
         */
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

    /**
     * @brief CUDA implementation of Layer Normalization using abstract TensorDataType API.
     *
     * Template parameter TPrecision selects the abstract tensor precision (e.g. FP32, FP16).
     * NativeType is the corresponding CUDA device representation for that precision.
     *
     * Design philosophy:
     * - Two-phase initialization: build() does all setup, forward()/backward() are pure dispatch
     * - Module owns weight/bias parameters and binds them via setParameters()
     * - Operation allocates backend-owned mean/rstd device storage during build()
     * - All dimension computation and validation happens once in build()
     * - Forward/backward are hot-path methods with minimal overhead
     */
    export template<TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, DeviceType::Cuda>
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
            : config_( config ), context_( context ), impl_()
        {
            if (!context_)
            {
                throw std::runtime_error( "CudaLayerNormOp requires a CUDA execution context" );
            }

            config_.validate();
        }

        // ====================================================================
		// Parameters and Gradients
        // ====================================================================

        /**
         * @brief Set parameter tensor references (module remains owner).
         *
         * The operation caches native device pointers for hot-path access. The
         * weight tensor is required; bias is bound only when the LayerNorm
         * config indicates a bias is present.
         *
         * Note: build() requires parameters to be bound before it is called.
         */
        void setParameters( ITensor* weight, ITensor* bias ) override
        {
            if (!weight)
            {
                throw std::invalid_argument( "CudaLayerNormOp::setParameters - weight parameter is required" );
            }

            if (weight->getDeviceType() != DeviceType::Cuda)
            {
                throw std::invalid_argument( "CudaLayerNormOp::setParameters - weight must be a CUDA tensor" );
            }

            weight_ = static_cast<NativeType*>(weight->rawData());

            if (config_.hasBias())
            {
                if (!bias)
                {
                    throw std::invalid_argument( "CudaLayerNormOp::setParameters - bias parameter expected but null was provided" );
                }

                if (bias->getDeviceType() != DeviceType::Cuda)
                {
                    throw std::invalid_argument( "CudaLayerNormOp::setParameters - bias must be a CUDA tensor" );
                }

                bias_ = static_cast<NativeType*>(bias->rawData());
            }
            else
            {
                bias_ = nullptr;
            }
        }

        /**
         * @brief Set parameter gradient tensor references for training.
         *
         * The operation caches native device gradient pointers for hot-path write access
         * during backward(). Weight gradient is required; bias gradient is bound
         * only when the LayerNorm config indicates a bias is present.
         *
         * @param weight_grad Gradient tensor for weight parameter
         * @param bias_grad Gradient tensor for bias parameter (optional based on config)
         *
         * @throws std::invalid_argument If weight_grad is null
         * @throws std::invalid_argument If weight_grad is not a CUDA tensor
         * @throws std::invalid_argument If bias_grad is null when config requires bias
         * @throws std::invalid_argument If bias_grad is not a CUDA tensor when required
         */
        void setGradients( ITensor* weight_grad, ITensor* bias_grad ) override
        {
            if (!weight_grad)
            {
                throw std::invalid_argument( "CudaLayerNormOp::setParameterGradients - weight gradient is required" );
            }

            if (weight_grad->getDeviceType() != DeviceType::Cuda)
            {
                throw std::invalid_argument( "CudaLayerNormOp::setParameterGradients - weight gradient must be a CUDA tensor" );
            }

            weight_grad_ = static_cast<NativeType*>(weight_grad->rawData());

            if (config_.hasBias())
            {
                if (!bias_grad)
                {
                    throw std::invalid_argument( "CudaLayerNormOp::setParameterGradients - bias gradient expected but null was provided" );
                }

                if (bias_grad->getDeviceType() != DeviceType::Cuda)
                {
                    throw std::invalid_argument( "CudaLayerNormOp::setParameterGradients - bias gradient must be a CUDA tensor" );
                }

                bias_grad_ = static_cast<NativeType*>(bias_grad->rawData());
            }
            else
            {
                bias_grad_ = nullptr;
            }
        }

        // ====================================================================
        // Lifecycle
        // ====================================================================

        /**
         * @brief Build the operation for a concrete input shape.
         *
         * This is the COLD PATH where all setup, validation, and computation happens ONCE.
         * After build() completes, forward() and backward() become pure dispatch methods.
         *
         * Responsibilities:
         *  1. Validate parameters are bound via setParameters()
         *  2. Validate input shape compatibility with configuration
         *  3. Compute and cache normalization axis
         *  4. Compute and cache kernel dispatch dimensions [B, T, C]
         *  5. Allocate backend-owned device storage for mean/rstd
         *  6. Cache all device pointers for hot-path access
         *
         * After build(), the operation is ready for zero-overhead forward/backward dispatch.
         */
        void build( const shape_t& input_shape ) override
        {
            if (weight_ == nullptr)
            {
                throw std::runtime_error( "CudaLayerNormOp::build requires parameters bound via setParameters() before build()." );
            }

            if (config_.hasBias() && bias_ == nullptr)
            {
                throw std::runtime_error( "CudaLayerNormOp::build - bias expected by config but not bound via setParameters()." );
            }

            if (!config_.getNormalizedShape().empty())
            {
                if (input_shape.size() < config_.getNormalizedShape().size())
                {
                    throw std::invalid_argument( "CudaLayerNormOp::build - input rank is less than normalized_shape rank" );
                }

                size_t offset = input_shape.size() - config_.getNormalizedShape().size();
                for (size_t i = 0; i < config_.getNormalizedShape().size(); ++i)
                {
                    if (input_shape[offset + i] != config_.getNormalizedShape()[i])
                    {
                        throw std::invalid_argument( "CudaLayerNormOp::build - input trailing dimensions don't match normalized_shape" );
                    }
                }
            }
            else if (!config_.getAxis().has_value())
            {
                throw std::invalid_argument( "CudaLayerNormOp::build - configuration must specify normalized_shape or axis before build()" );
            }

            const auto& shape = input_shape;
            const int64_t ndim = static_cast<int64_t>(shape.size());

            int64_t axis = -1;
            if (config_.getAxis().has_value())
            {
                axis = config_.getAxis().value();
            }
            else
            {
                axis = static_cast<int64_t>(shape.size()) - static_cast<int64_t>(config_.getNormalizedShape().size());
            }

            if (axis < 0)
                axis += ndim;

            if (axis < 0 || axis >= ndim)
            {
                throw std::invalid_argument( "CudaLayerNormOp::build - computed axis out of range" );
            }

            cached_axis_ = axis;

            int64_t outer_size = 1;
            for (int64_t i = 0; i < axis; ++i)
                outer_size *= static_cast<int64_t>( shape[i] );

            int64_t inner_size = 1;
            for (int64_t i = axis + 1; i < ndim; ++i)
                inner_size *= static_cast<int64_t>( shape[i] );

            const int64_t dim_size = static_cast<int64_t>( shape[axis] );
            const int64_t expected_slices = outer_size * inner_size;

            cached_B_ = static_cast<int>( outer_size );
            cached_T_ = static_cast<int>( inner_size );
            cached_C_ = static_cast<int>( dim_size );

            auto device = context_->getDevice();

            mean_storage_ = std::make_shared<TensorType>( device, shape_t{ expected_slices } );
            mean_storage_->setName( "CudaLayerNormOp.mean" );
            mean_ = static_cast<NativeType*>(mean_storage_->rawData());

            rstd_storage_ = std::make_shared<TensorType>( device, shape_t{ expected_slices } );
            rstd_storage_->setName( "CudaLayerNormOp.rstd" );
            rstd_ = static_cast<NativeType*>(rstd_storage_->rawData());

            UnaryOperationBase::build( input_shape );
        }

        /**
         * @brief Forward pass - HOT PATH, pure dispatch to CUDA kernel.
         *
         * All setup, validation, and dimension computation was done in build().
         * This method extracts raw pointers and dispatches directly to the kernel
         * using pre-computed cached dimensions.
         *
         * Zero redundant work - maximum performance.
         */
        void forward( const ITensor& input, ITensor& output ) const override
        {
            const NativeType* X = static_cast<const NativeType*>(input.rawData());
            NativeType* Y = static_cast<NativeType*>(output.rawData());

            cudaStream_t stream = context_->getStream();

            Detail::cuda_layernorm_impl<NativeType>::forward(
                Y, X,
                weight_, bias_,
                mean_, rstd_,
                cached_B_, cached_T_, cached_C_,
                config_.getEpsilon(),
                stream
            );
        }

        /**
         * @brief Backward pass - HOT PATH, pure dispatch to CUDA kernel.
         *
         * Similar to forward(), this method does minimal work and dispatches
         * directly to the backward kernel using cached dimensions from build().
         */
        void backward(
            const ITensor& input,
            const ITensor& output_grad,
            ITensor& input_grad ) const override
        {
            const NativeType* X = static_cast<const NativeType*>(input.rawData());
            const NativeType* dY = static_cast<const NativeType*>(output_grad.rawData());
            NativeType* dX = static_cast<NativeType*>(input_grad.rawData());

            NativeType* dweight = weight_grad_;
            NativeType* dbias = bias_grad_;

            cudaStream_t stream = context_->getStream();

            Detail::cuda_layernorm_impl<NativeType>::backward(
                dX, dweight, dbias,
                dY, X, weight_,
                mean_, rstd_,
                cached_B_, cached_T_, cached_C_,
                stream
            );
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

        // Cached native device parameter pointers (module owns underlying tensors)
        NativeType* weight_{ nullptr };
        NativeType* bias_{ nullptr };

        // Cached native device parameter gradient pointers (module owns underlying tensors)
        NativeType* weight_grad_{ nullptr };
        NativeType* bias_grad_{ nullptr };

        // Backend-owned device runtime statistics storage
        std::shared_ptr<TensorType> mean_storage_;
        std::shared_ptr<TensorType> rstd_storage_;
        NativeType* mean_{ nullptr };
        NativeType* rstd_{ nullptr };

        // Cached dimension values computed once in build() for hot-path dispatch
        int64_t cached_axis_{ -1 };
        int cached_B_{ 0 };  // outer_size: batch and leading dimensions
        int cached_T_{ 0 };  // inner_size: trailing dimensions after normalized axis
        int cached_C_{ 0 };  // dim_size: size of the normalized axis dimension
    };

    export class CudaLayerNormOpRegistrar
    {
    public:
        static void registerOperations()
        {
            const std::string opName = "LayerNormOp";

            OperationRegistry::instance().registerUnaryOperation<DeviceType::Cuda, TensorDataType::FP32, TensorDataType::FP32>(
                opName,
                []( std::shared_ptr<ExecutionContext<DeviceType::Cuda>> context,
                    const ComponentConfig& config ) -> std::shared_ptr<UnaryOperation<DeviceType::Cuda, TensorDataType::FP32>>
                {
                    const auto& lnConfig = static_cast<const LayerNormConfig&>(config);
                    return std::make_shared<CudaLayerNormOp<TensorDataType::FP32>>( context, lnConfig );
                }
            );

            OperationRegistry::instance().registerUnaryOperation<DeviceType::Cuda, TensorDataType::FP16, TensorDataType::FP16>(
                opName,
                []( std::shared_ptr<ExecutionContext<DeviceType::Cuda>> context,
                    const ComponentConfig& config ) -> std::shared_ptr<UnaryOperation<DeviceType::Cuda, TensorDataType::FP16>>
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