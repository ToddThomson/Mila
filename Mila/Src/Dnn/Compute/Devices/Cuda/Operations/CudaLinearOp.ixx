/**
 * @file CudaLinearOp.ixx
 * @brief CUDA implementation of Linear (fully connected) operation (TensorDataType-based).
 *
 * Ported to the ExecutionContext / TensorDataType UnaryOperation interface
 * following the two-phase initialization pattern used by CudaLayerNormOp.
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
#include <cstdint>
#include <type_traits>
#include "Kernels/CudaOps.h"
#include <sstream>

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
import Compute.CublasLtMatMulBias;
import Utils.Logger;

namespace Mila::Dnn::Compute
{
    using namespace Mila::Dnn;

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
                // cuda_matmul_backward_weight_fp32( dW, X, dY, outer_size, C, OC, stream );
                // if (dB) cuda_matmul_backward_bias_fp32( dB, dY, outer_size, OC, stream );
            }
        };

        template <>
        struct cuda_matmul_impl<half>
        {
            cuda_matmul_impl() = default;

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
                // cuda_matmul_backward_weight_fp16( dW, X, dY, outer_size, C, OC, stream );
                // if (dB) cuda_matmul_backward_bias_fp16( dB, dY, outer_size, OC, stream );
            }
        };

        template <>
        struct cuda_matmul_impl<nv_bfloat16>
        {
            cuda_matmul_impl() = default;

            static inline void forward(
                nv_bfloat16* Y, const nv_bfloat16* X,
                const nv_bfloat16* weight, const nv_bfloat16* bias,
                int outer_size, int C, int OC,
                cudaStream_t stream )
            {
                // FIXME: cuda_matmul_forward_bf16( Y, X, weight, bias, outer_size, C, OC, stream );
            }

            static inline void backward(
                nv_bfloat16* dX, nv_bfloat16* dW, nv_bfloat16* dB,
                const nv_bfloat16* dY, const nv_bfloat16* X, const nv_bfloat16* W,
                int outer_size, int C, int OC,
                cudaStream_t stream )
            {
                // FIXME: Implement when BF16 backward kernels exist
            }
        };

        template <>
        struct cuda_matmul_impl<__nv_fp8_e4m3>
        {
            cuda_matmul_impl() = default;

            static inline void forward(
                __nv_fp8_e4m3* Y, const __nv_fp8_e4m3* X,
                const __nv_fp8_e4m3* weight, const __nv_fp8_e4m3* bias,
                int outer_size, int C, int OC,
                cudaStream_t stream )
            {
                // FIXME: Implement FP8 forward kernel wrapper when available
            }

            static inline void backward(
                __nv_fp8_e4m3* dX, __nv_fp8_e4m3* dW, __nv_fp8_e4m3* dB,
                const __nv_fp8_e4m3* dY, const __nv_fp8_e4m3* X, const __nv_fp8_e4m3* W,
                int outer_size, int C, int OC,
                cudaStream_t stream )
            {
                // FIXME: Implement FP8 backward kernel wrapper when available
            }
        };

        template <>
        struct cuda_matmul_impl<__nv_fp8_e5m2>
        {
            cuda_matmul_impl() = default;

            static inline void forward(
                __nv_fp8_e5m2* Y, const __nv_fp8_e5m2* X,
                const __nv_fp8_e5m2* weight, const __nv_fp8_e5m2* bias,
                int outer_size, int C, int OC,
                cudaStream_t stream )
            {
                // FIXME: Implement FP8 forward kernel wrapper when available
            }

            static inline void backward(
                __nv_fp8_e5m2* dX, __nv_fp8_e5m2* dW, __nv_fp8_e5m2* dB,
                const __nv_fp8_e5m2* dY, const __nv_fp8_e5m2* X, const __nv_fp8_e5m2* W,
                int outer_size, int C, int OC,
                cudaStream_t stream )
            {
                // FIXME: Implement FP8 backward kernel wrapper when available
            }
        };
    }

    /**
     * @brief CUDA implementation of Linear operation using abstract TensorDataType API.
     *
     * Template parameter TPrecision selects the abstract tensor precision (e.g. FP32, FP16, BF16).
     * NativeType is the corresponding CUDA device representation for that precision.
     *
     * Design philosophy:
     * - Two-phase initialization: build() does all setup, forward()/backward() are pure dispatch
     * - Module owns weight/bias parameters and binds them via setParameters()
     * - All dimension computation happens once in build()
     * - Forward/backward are hot-path methods with minimal overhead
     * - Implements: y = x * W^T + b where W is (out_features, in_features)
     * - Supports cuBLASLt for optimized GEMM when available
     *
     * Forward: Y = X * W^T + b
     * Backward:
     *  - dX += dY * W
     *  - dW += dY^T * X
     *  - db += sum(dY)
     */
    export template<TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, DeviceType::Cuda>
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
            : context_( context ), config_( config ), impl_()
        {
            if (!context_)
            {
                throw std::runtime_error( "CudaLinearOp requires a CUDA execution context" );
            }

            config_.validate();
        }

        // ====================================================================
        // Parameters
        // ====================================================================

        /**
         * @brief Set parameter tensor references (module remains owner).
         *
         * The operation caches native device pointers for hot-path access. The
         * weight tensor is required; bias is bound only when the Linear
         * config indicates a bias is present.
         *
         * Note: build() requires parameters to be bound before it is called.
         */
        void setParameters( ITensor* weight, ITensor* bias ) override
        {
            if (!weight)
            {
                throw std::invalid_argument( "CudaLinearOp::setParameters - weight parameter is required" );
            }

            if (weight->getDeviceType() != DeviceType::Cuda)
            {
                throw std::invalid_argument( "CudaLinearOp::setParameters - weight must be a CUDA tensor" );
            }

            weight_ = static_cast<const NativeType*>(weight->rawData());

            // Validate weight is 2D
            const auto& weight_shape = weight->shape();
            if (weight_shape.size() != 2)
            {
                throw std::invalid_argument( "CudaLinearOp::setParameters - weight must be 2D tensor" );
            }

            // Store weight dimensions for validation
            weight_out_features_ = weight_shape[0];
            weight_in_features_ = weight_shape[1];

            if (config_.hasBias())
            {
                if (!bias)
                {
                    throw std::invalid_argument( "CudaLinearOp::setParameters - bias parameter expected but null was provided" );
                }

                if (bias->getDeviceType() != DeviceType::Cuda)
                {
                    throw std::invalid_argument( "CudaLinearOp::setParameters - bias must be a CUDA tensor" );
                }

                bias_ = static_cast<const NativeType*>(bias->rawData());
            }
            else
            {
                bias_ = nullptr;
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
         *  1. Validate parameters were bound via setParameters()
         *  2. Validate input shape compatibility with weight dimensions
         *  3. Compute and cache batch size and feature dimensions
         *  4. Cache cuBLASLt handle availability
         *  5. Cache precision policy for mixed-precision operations
         *
         * After build(), the operation is ready for zero-overhead forward/backward dispatch.
         */
        void build( const shape_t& input_shape ) override
        {
            if (weight_ == nullptr)
            {
                throw std::runtime_error( "CudaLinearOp::build requires parameters bound via setParameters() before build()." );
            }

            if (config_.hasBias() && bias_ == nullptr)
            {
                throw std::runtime_error( "CudaLinearOp::build - bias expected by config but not bound via setParameters()." );
            }

            if (input_shape.empty())
            {
                throw std::invalid_argument( "CudaLinearOp::build - input shape cannot be empty" );
            }

            // Extract dimensions: input is (..., in_features)
            cached_in_features_ = static_cast<int>(input_shape.back());

            // Validate weight dimensions match configuration
            if (weight_out_features_ != config_.getOutputFeatures())
            {
                std::ostringstream oss;
                oss << "CudaLinearOp::build - weight output features mismatch. Expected "
                    << config_.getOutputFeatures() << ", got " << weight_out_features_;
                throw std::invalid_argument( oss.str() );
            }

            if (weight_in_features_ != cached_in_features_)
            {
                std::ostringstream oss;
                oss << "CudaLinearOp::build - weight input features mismatch. Expected "
                    << cached_in_features_ << ", got " << weight_in_features_;
                throw std::invalid_argument( oss.str() );
            }

            // Compute batch size (flatten all dimensions except last)
            cached_batch_size_ = 1;
            for (size_t i = 0; i + 1 < input_shape.size(); ++i)
            {
                cached_batch_size_ *= static_cast<int>(input_shape[i]);
            }

            cached_out_features_ = static_cast<int>(config_.getOutputFeatures());

            // Cache cuBLASLt availability for optimized path
            cached_cublaslt_handle_ = context_->getCublasLtHandle();
            use_cublaslt_ = (cached_cublaslt_handle_ != nullptr) && supportsCuBLASLt();

            // Cache precision policy for mixed-precision operations
            cached_precision_policy_ = config_.getPrecisionPolicy();

            UnaryOperationBase::build( input_shape );
        }

        // ====================================================================
        // Computation
        // ====================================================================

        /**
         * @brief Forward pass - HOT PATH, pure dispatch to CUDA kernel.
         *
         * All setup, validation, and dimension computation was done in build().
         * This method extracts raw pointers and dispatches to either cuBLASLt
         * or custom CUDA kernels using pre-computed cached dimensions.
         *
         * Algorithm: Y = X * W^T + b
         * Zero redundant work - maximum performance.
         */
        void forward( const ITensor& input, ITensor& output ) const override
        {
            const NativeType* X = static_cast<const NativeType*>(input.rawData());
            NativeType* Y = static_cast<NativeType*>(output.rawData());

            cudaStream_t stream = context_->getStream();

            // Try cuBLASLt optimized path if available
            if (use_cublaslt_)
            {
                try
                {
                    cublaslt_matmul_forward<NativeType>(
                        Y, X,
                        weight_, bias_,
                        cached_batch_size_,
                        cached_in_features_, cached_out_features_,
                        stream, cached_cublaslt_handle_,
                        cached_precision_policy_ );

                    return;
                }
                catch (const std::exception& e)
                {
                    Utils::Logger::warning(
                        std::string( "cuBLASLt path failed, falling back to custom kernel: " ) + e.what() );
                }
            }

            // Fallback to custom CUDA kernel
            Detail::cuda_matmul_impl<NativeType>::forward(
                Y, X,
                weight_, bias_,
                cached_batch_size_,
                cached_in_features_, cached_out_features_,
                stream );
        }

        /**
         * @brief Backward pass - HOT PATH, pure dispatch to CUDA kernel.
         *
         * Similar to forward(), this method does minimal work and dispatches
         * directly to the backward kernel using cached dimensions from build().
         *
         * Algorithm:
         *  - dX += dY * W
         *  - dW += dY^T * X
         *  - db += sum(dY)
         */
        void backward(
            const ITensor& input,
            const ITensor& output_grad,
            ITensor& input_grad,
            Parameters& parameter_grads ) const override
        {
            const NativeType* X = static_cast<const NativeType*>(input.rawData());
            const NativeType* dY = static_cast<const NativeType*>(output_grad.rawData());
            NativeType* dX = static_cast<NativeType*>(input_grad.rawData());

            const NativeType* W = weight_;

            NativeType* dW = nullptr;
            NativeType* dB = nullptr;

            if (parameter_grads.size() > 0 && parameter_grads[0])
            {
                dW = static_cast<NativeType*>(parameter_grads[0]->rawData());
            }

            if (parameter_grads.size() > 1 && parameter_grads[1])
            {
                dB = static_cast<NativeType*>(parameter_grads[1]->rawData());
            }

            cudaStream_t stream = context_->getStream();

            Detail::cuda_matmul_impl<NativeType>::backward(
                dX, dW, dB,
                dY, X, W,
                cached_batch_size_,
                cached_in_features_, cached_out_features_,
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
        Detail::cuda_matmul_impl<NativeType> impl_;

        // Cached native device parameter pointers (module owns underlying tensors)
        const NativeType* weight_{ nullptr };
        const NativeType* bias_{ nullptr };

        // Weight dimensions for validation
        int64_t weight_out_features_{ 0 };
        int64_t weight_in_features_{ 0 };

        // Cached dimension values computed once in build() for hot-path dispatch
        int cached_batch_size_{ 0 };
        int cached_in_features_{ 0 };
        int cached_out_features_{ 0 };

        // Cached cuBLASLt resources and flags
        cublasLtHandle_t cached_cublaslt_handle_{ nullptr };
        bool use_cublaslt_{ false };
        ComputePrecision::Policy cached_precision_policy_;

        /**
         * @brief Check if cuBLASLt supports the current precision type.
         *
         * cuBLASLt supports float, half, and bfloat16, but not FP8 types yet.
         */
        constexpr bool supportsCuBLASLt() const
        {
            return std::is_same_v<NativeType, float> ||
                std::is_same_v<NativeType, half> ||
                std::is_same_v<NativeType, nv_bfloat16>;
        }
    };

    /**
     * @brief Registrar for CUDA Linear operations.
     *
     * Registers FP32, FP16, and BF16 precision variants.
     * FP8 variants are commented out pending kernel implementation.
     */
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

            // FP8 support pending kernel implementation
            /*OperationRegistry::instance().registerUnaryOperation<DeviceType::Cuda, TensorDataType::FP8_E4M3>(
                opName, ... );
            OperationRegistry::instance().registerUnaryOperation<DeviceType::Cuda, TensorDataType::FP8_E5M2>(
                opName, ... );*/
        }

        static inline bool isRegistered = []() {
            registerOperations();
            return true;
            }();
    };
}