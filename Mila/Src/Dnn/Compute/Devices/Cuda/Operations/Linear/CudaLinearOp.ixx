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

export module Compute.CudaLinearOp;
import :Plans;
import :Dispatch;

import Dnn.Components.LinearConfig;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.ComponentConfig;
import Compute.OperationBase;
import Compute.Precision;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.IExecutionContext;
import Compute.ExecutionContextTemplate;
import Compute.OperationType;
import Compute.MemoryResource;
import Compute.CudaDeviceMemoryResource;
import Compute.CudaDevice;
import Compute.CudaTensorDataType;
import Compute.CublasLtPlan;
import Compute.CublasLtPlanCache;
import CublasLt.Error;
import Utils.Logger;

import Dnn.TensorOps;
import Dnn.TensorHelpers;

namespace Mila::Dnn::Compute::Cuda::Linear
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute::Cuda;

    /**
     * @brief CUDA implementation of Linear operation using two-phase cuBLASLt optimization.
     *
     * Derives from Operation<> only — forward/backward are non-virtual typed methods.
     * The concrete type is selected at compile time by CudaLinearOpTraits; no vtable
     * dispatch is used for the hot path.
     *
     * Forward:  output = input * weight^T + bias
     * Backward:
     *   input_grad  = output_grad * weight
     *   weight_grad = output_grad^T * input  (accumulated)
     *   bias_grad   = sum(output_grad, dim=0)
     */
    export template<TensorDataType TComputePrecision, TensorDataType TParameterPrecision = TComputePrecision>
        requires PrecisionSupportedOnDevice<TComputePrecision, DeviceType::Cuda>
    class CudaLinearOp : public Operation<DeviceType::Cuda, TComputePrecision>
    {
    public:
        using MR = CudaDeviceMemoryResource;
        using TensorType = Tensor<TComputePrecision, MR>;
        using ActivationType = typename Mila::Dnn::Compute::Cuda::TensorDataTypeMap<TComputePrecision>::device_type;
        using ParameterType = typename Mila::Dnn::Compute::Cuda::TensorDataTypeMap<TParameterPrecision>::device_type;
        using CudaExecutionContext = ExecutionContext<DeviceType::Cuda>;

        static constexpr bool kIsQuantized = (TParameterPrecision != TComputePrecision);

        CudaLinearOp( IExecutionContext* context, const LinearConfig& config )
            : context_( validateExecutionContext_<DeviceType::Cuda>( context, "CudaLinearOp" ) ), config_( config )
        {
            config_.validate();
        }

        void setParameters( ITensor* weight, ITensor* bias ) override
        {
            if ( !weight )
            {
                throw std::invalid_argument( "CudaLinearOp::setParameters - weight parameter is required" );
            }

            if ( weight->getDeviceType() != DeviceType::Cuda )
            {
                throw std::invalid_argument( "CudaLinearOp::setParameters - weight must be a CUDA tensor" );
            }

            weight_ = static_cast<const ActivationType*>(weight->rawData());

            const auto& weight_shape = weight->shape();
            if ( weight_shape.size() != 2 )
            {
                throw std::invalid_argument( "CudaLinearOp::setParameters - weight must be 2D tensor" );
            }

            weight_out_features_ = weight_shape[ 0 ];
            weight_in_features_ = weight_shape[ 1 ];

            if ( config_.hasBias() )
            {
                if ( !bias )
                {
                    throw std::invalid_argument( "CudaLinearOp::setParameters - bias expected but null was provided" );
                }

                if ( bias->getDeviceType() != DeviceType::Cuda )
                {
                    throw std::invalid_argument( "CudaLinearOp::setParameters - bias must be a CUDA tensor" );
                }

                bias_ = static_cast<const ActivationType*>(bias->rawData());
            }
            else
            {
                bias_ = nullptr;
            }
        }

        void setGradients( ITensor* weight_grad, ITensor* bias_grad ) override
        {
            if ( !weight_grad )
            {
                throw std::invalid_argument( "CudaLinearOp::setGradients - weight gradient is required" );
            }

            if ( weight_grad->getDeviceType() != DeviceType::Cuda )
            {
                throw std::invalid_argument( "CudaLinearOp::setGradients - weight gradient must be a CUDA tensor" );
            }

            weight_grad_ = static_cast<ActivationType*>(weight_grad->rawData());

            if ( config_.hasBias() )
            {
                if ( !bias_grad )
                {
                    throw std::invalid_argument( "CudaLinearOp::setGradients - bias gradient expected but null was provided" );
                }

                if ( bias_grad->getDeviceType() != DeviceType::Cuda )
                {
                    throw std::invalid_argument( "CudaLinearOp::setGradients - bias gradient must be a CUDA tensor" );
                }

                bias_grad_ = static_cast<ActivationType*>(bias_grad->rawData());
            }
            else
            {
                bias_grad_ = nullptr;
            }
        }

        void build( const BuildContext& build_config ) override
        {
            const auto& input_shape = build_config.inputShape();

            if ( weight_ == nullptr )
            {
                throw std::runtime_error( "CudaLinearOp::build requires parameters bound via setParameters() before build()." );
            }

            if ( config_.hasBias() && bias_ == nullptr )
            {
                throw std::runtime_error( "CudaLinearOp::build - bias expected by config but not bound." );
            }

            if ( input_shape.empty() )
            {
                throw std::invalid_argument( "CudaLinearOp::build - input shape cannot be empty" );
            }

            cached_in_features_ = static_cast<int>(input_shape.back());

            if ( weight_out_features_ != config_.getOutputFeatures() )
            {
                throw std::invalid_argument( std::format(
                    "CudaLinearOp::build - weight output features mismatch. Expected {}, got {}",
                    config_.getOutputFeatures(), weight_out_features_ ) );
            }

            if ( weight_in_features_ != cached_in_features_ )
            {
                throw std::invalid_argument( std::format(
                    "CudaLinearOp::build - weight input features mismatch. Expected {}, got {}",
                    cached_in_features_, weight_in_features_ ) );
            }

            cached_outer_size_ = 1;
            for ( size_t i = 0; i + 1 < input_shape.size(); ++i )
            {
                cached_outer_size_ *= static_cast<int>(input_shape[ i ]);
            }

            cached_out_features_ = static_cast<int>(config_.getOutputFeatures());

            cached_cublaslt_handle_ = context_->getCublasLtHandle();
            use_cublaslt_ = (cached_cublaslt_handle_ != nullptr) && supportsCuBLASLt();

            cached_precision_policy_ = config_.getPrecisionPolicy();

            if ( use_cublaslt_ )
            {
                try
                {
                    buildCublasLtPlans();
                }
                catch ( const std::exception& e )
                {
                    Utils::Logger::warning(
                        std::string( "Failed to build cuBLASLt plans, falling back to custom kernels: " ) + e.what() );

                    use_cublaslt_ = false;
                }
            }

            Operation<DeviceType::Cuda, TComputePrecision>::build( build_config );
        }

        /**
         * @brief Forward pass: output = input * weight^T + bias
         *
         * Single-token decode path (outer_size == 1) dispatches to the fused matvec kernel.
         * Multi-token path uses a cached cuBLASLt plan.
         *
         * @param input  Input tensor [outer_size, in_features]
         * @param output Output tensor [outer_size, out_features]
         */
        void forward( const TensorType& input, TensorType& output ) const
        {
            const int outer_size = static_cast<int>(input.size()) / cached_in_features_;

            const ActivationType* input_ptr = static_cast<const ActivationType*>(input.rawData());
            ActivationType* output_ptr = static_cast<ActivationType*>(output.rawData());

            cudaStream_t stream = context_->getStream();

            if ( outer_size == 1 )
            {
                Detail::cuda_matvec_impl<ActivationType>::decode(
                    output_ptr, input_ptr,
                    weight_, bias_,
                    cached_in_features_, cached_out_features_,
                    stream );

                return;
            }

            if ( use_cublaslt_ )
            {
                const float alpha = 1.0f;
                const float beta = 0.0f;

                execute_plan<ActivationType>(
                    cached_cublaslt_handle_,
                    forward_plan_cache_.get( static_cast<int>(outer_size) ),
                    &alpha,
                    input_ptr, weight_,
                    &beta,
                    output_ptr,
                    bias_,
                    stream,
                    context_->getCublasLtWorkspace(),
                    context_->getCublasLtWorkspaceSize() );

                // DEBUG: remove after stabilization
                // this->context_->synchronize();

                return;
            }

            throw std::runtime_error( "CudaLinearOp: no valid execution path available for forward." );
        }

        /**
         * @brief Backward pass: compute input, weight, and bias gradients.
         *
         * Weight gradient accumulates (beta=1) across the full batch. Bias gradient
         * is reduced via a separate kernel. Input gradient uses a separate cached plan.
         *
         * @param input       Saved forward input tensor.
         * @param output_grad Upstream gradient tensor.
         * @param input_grad  Output: gradient with respect to forward input.
         */
        void backward( const TensorType& input, const TensorType& output_grad, TensorType& input_grad ) const
        {
            if ( this->isEvalMode() )
            {
                throw std::runtime_error( "CudaLinearOp::backward: not available in eval mode" );
            }

            const int outer_size = static_cast<int>(output_grad.size()) / cached_out_features_;

            const ActivationType* input_ptr = static_cast<const ActivationType*>(input.rawData());
            const ActivationType* output_grad_ptr = static_cast<const ActivationType*>(output_grad.rawData());
            ActivationType* input_grad_ptr = static_cast<ActivationType*>(input_grad.rawData());

            cudaStream_t stream = context_->getStream();

            if ( use_cublaslt_ )
            {
                const float alpha = 1.0f;
                const float beta = 0.0f;
                const float beta_accum = 1.0f;

                // dX[batch, in] = dY[batch, out] @ weight[out, in]
                execute_plan<ActivationType>(
                    cached_cublaslt_handle_,
                    backward_input_plan_cache_.get( static_cast<int>(outer_size) ),
                    &alpha,
                    output_grad_ptr, weight_,
                    &beta,
                    input_grad_ptr,
                    nullptr,
                    stream,
                    context_->getCublasLtWorkspace(),
                    context_->getCublasLtWorkspaceSize() );

                // dW[out, in] = dY^T @ X  (accumulated across full batch)
                execute_plan<ActivationType>(
                    cached_cublaslt_handle_,
                    backward_weight_plan_,
                    &alpha,
                    output_grad_ptr, input_ptr,
                    &beta_accum,
                    weight_grad_,
                    nullptr,
                    stream,
                    context_->getCublasLtWorkspace(),
                    context_->getCublasLtWorkspaceSize() );

                if ( bias_grad_ != nullptr )
                {
                    Detail::compute_bias_gradient(
                        bias_grad_,
                        output_grad_ptr,
                        static_cast<int>(outer_size),
                        cached_out_features_,
                        stream );
                }

                return;
            }

            throw std::runtime_error( "CudaLinearOp: no valid execution path available for backward." );
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
        CudaExecutionContext* context_;

        const ActivationType* weight_{ nullptr };
        const ActivationType* bias_{ nullptr };

        ActivationType* weight_grad_{ nullptr };
        ActivationType* bias_grad_{ nullptr };

        int64_t weight_out_features_{ 0 };
        int64_t weight_in_features_{ 0 };

        int cached_outer_size_{ 0 };
        int cached_in_features_{ 0 };
        int cached_out_features_{ 0 };

        cublasLtHandle_t cached_cublaslt_handle_{ nullptr };
        bool use_cublaslt_{ false };
        ComputePrecision::Policy cached_precision_policy_;

        CublasLtPlanCache<CublasLtMatMulPlan<ActivationType>> forward_plan_cache_;
        CublasLtPlanCache<CublasLtMatMulPlan<ActivationType>> backward_input_plan_cache_;
        CublasLtMatMulPlan<ActivationType> backward_weight_plan_;

        cudaDataType_t cuda_data_type_{};
        cublasComputeType_t compute_type_{};
        cudaDataType_t scale_type_{};

        constexpr bool supportsCuBLASLt() const
        {
            return std::is_same_v<ActivationType, float> ||
                std::is_same_v<ActivationType, half> ||
                std::is_same_v<ActivationType, nv_bfloat16>;
        }

        void buildCublasLtPlans()
        {
            cudaDataType_t cuda_data_type = getCudaDataType();
            cublasComputeType_t compute_type;
            cudaDataType_t scale_type;

            getComputeTypes( compute_type, scale_type );

            cuda_data_type_ = cuda_data_type;
            compute_type_ = compute_type;
            scale_type_ = scale_type;

            forward_plan_cache_ = CublasLtPlanCache<CublasLtMatMulPlan<ActivationType>>(
                cached_outer_size_,
                [&]( int bucket )
                {
                    return Detail::build_forward_plan<ActivationType>(
                        cached_cublaslt_handle_,
                        bucket,
                        cached_in_features_,
                        cached_out_features_,
                        config_.hasBias(),
                        cuda_data_type_,
                        compute_type_,
                        scale_type_ );
                } );

            backward_input_plan_cache_ = CublasLtPlanCache<CublasLtMatMulPlan<ActivationType>>(
                cached_outer_size_,
                [&]( int bucket )
                {
                    return Detail::build_backward_input_plan<ActivationType>(
                        cached_cublaslt_handle_,
                        bucket,
                        cached_in_features_,
                        cached_out_features_,
                        cuda_data_type_,
                        compute_type_,
                        scale_type_ );
                } );

            backward_weight_plan_ = Detail::build_backward_weight_plan<ActivationType>(
                cached_cublaslt_handle_,
                cached_outer_size_,
                cached_in_features_,
                cached_out_features_,
                cuda_data_type_,
                compute_type_,
                scale_type_ );
        }

        // REVIEW: delegate to TensorDataTypeMap<NativeType>::cuda_data_type when confirmed available
        cudaDataType_t getCudaDataType() const
        {
            if constexpr ( std::is_same_v<ActivationType, float> )
                return CUDA_R_32F;
            else if constexpr ( std::is_same_v<ActivationType, half> )
                return CUDA_R_16F;
            else if constexpr ( std::is_same_v<ActivationType, nv_bfloat16> )
                return CUDA_R_16BF;
            else if constexpr ( std::is_same_v<ActivationType, __nv_fp8_e4m3> )
                return CUDA_R_8F_E4M3;
            else if constexpr ( std::is_same_v<ActivationType, __nv_fp8_e5m2> )
                return CUDA_R_8F_E5M2;
        }

        void getComputeTypes( cublasComputeType_t& compute_type, cudaDataType_t& scale_type ) const
        {
            scale_type = CUDA_R_32F;

            switch ( cached_precision_policy_ )
            {
                case ComputePrecision::Policy::Native:
                case ComputePrecision::Policy::Accuracy:
                    if constexpr ( std::is_same_v<ActivationType, half> )
                        compute_type = CUBLAS_COMPUTE_16F;
                    else
                        compute_type = CUBLAS_COMPUTE_32F;
                    break;

                case ComputePrecision::Policy::Performance:
                case ComputePrecision::Policy::Auto:
                default:
                    if constexpr ( std::is_same_v<ActivationType, half> )
                        compute_type = CUBLAS_COMPUTE_32F_FAST_16F;
                    else if constexpr ( std::is_same_v<ActivationType, nv_bfloat16> )
                        compute_type = CUBLAS_COMPUTE_32F_FAST_16BF;
                    else
                        compute_type = CUBLAS_COMPUTE_32F;
                    break;
            }
        }
    };

    export class CudaLinearOpRegistrar
    {
    public:
        static void registerOperations()
        {
            const std::string opName = "LinearOp";

            /*OperationRegistry::instance().registerUnaryOperation<DeviceType::Cuda, TensorDataType::FP32, TensorDataType::FP32>(
                opName,
                []( IExecutionContext* context,
                    const ComponentConfig& config ) -> std::shared_ptr<UnaryOperation<DeviceType::Cuda, TensorDataType::FP32>>
                {
                    const auto& linearConfig = static_cast<const LinearConfig&>(config);
                    return std::make_shared<CudaLinearOp<TensorDataType::FP32>>( context, linearConfig );
                }
            );

            OperationRegistry::instance().registerUnaryOperation<DeviceType::Cuda, TensorDataType::BF16, TensorDataType::BF16>(
                opName,
                []( IExecutionContext* context,
                    const ComponentConfig& config ) -> std::shared_ptr<UnaryOperation<DeviceType::Cuda, TensorDataType::BF16>>
                {
                    const auto& linearConfig = static_cast<const LinearConfig&>(config);
                    return std::make_shared<CudaLinearOp<TensorDataType::BF16>>( context, linearConfig );
                }
            );*/
        }

        static inline bool isRegistered = []() {
            registerOperations();
            return true;
            }();
    };
}