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
import Compute.CublasLtPlan;
import Compute.CublasLtPlanCache;
import Compute.IDecode;
import CublasLt.Error;
import Utils.Logger;

// DEBUG:
import Dnn.TensorOps;
import Dnn.TensorHelpers;

namespace Mila::Dnn::Compute::Cuda::Linear
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute::Cuda;

    /**
     * @brief CUDA implementation of Linear operation using two-phase cuBLASLt optimization.
     *
     * Design philosophy:
     * - Two-phase initialization: build() creates cuBLASLt plans, forward()/backward() execute them
     * - Module owns weight/bias parameters and binds them via setParameters()
     * - All dimension computation and algorithm selection happens once in build()
     * - Forward/backward are hot-path methods with zero setup overhead
     * - cuBLASLt plans cache descriptors, layouts, and optimal algorithms
     *
     * Forward: output = input * weight^T + bias
     * Backward:
     *  - input_grad = output_grad * weight
     *  - weight_grad = output_grad^T * input
     *  - bias_grad = sum(output_grad)
     */
    export template<TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, DeviceType::Cuda>
    class CudaLinearOp : public UnaryOperation<DeviceType::Cuda, TPrecision>, public IDecode
    {
    public:
        using MR = CudaDeviceMemoryResource;
        using UnaryOperationBase = UnaryOperation<DeviceType::Cuda, TPrecision>;
        using TensorType = Tensor<TPrecision, MR>;
        using NativeType = typename Mila::Dnn::Compute::Cuda::TensorDataTypeMap<TPrecision>::native_type;
        using CudaExecutionContext = ExecutionContext<DeviceType::Cuda>;

        CudaLinearOp( IExecutionContext* context, const LinearConfig& config )
            : context_( validateExecutionContext_<DeviceType::Cuda>( context, "CudaLinearOp" ) ), config_( config ), impl_()
        {
            config_.validate();
        }

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

            const auto& weight_shape = weight->shape();
            if (weight_shape.size() != 2)
            {
                throw std::invalid_argument( "CudaLinearOp::setParameters - weight must be 2D tensor" );
            }

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

        void setGradients( ITensor* weight_grad, ITensor* bias_grad ) override
        {
            if (!weight_grad)
            {
                throw std::invalid_argument( "CudaLinearOp::setParameterGradients - weight gradient is required" );
            }

            if (weight_grad->getDeviceType() != DeviceType::Cuda)
            {
                throw std::invalid_argument( "CudaLinearOp::setParameterGradients - weight gradient must be a CUDA tensor" );
            }

            weight_grad_ = static_cast<NativeType*>(weight_grad->rawData());

            if (config_.hasBias())
            {
                if (!bias_grad)
                {
                    throw std::invalid_argument( "CudaLinearOp::setParameterGradients - bias gradient expected but null was provided" );
                }

                if (bias_grad->getDeviceType() != DeviceType::Cuda)
                {
                    throw std::invalid_argument( "CudaLinearOp::setParameterGradients - bias gradient must be a CUDA tensor" );
                }

                bias_grad_ = static_cast<NativeType*>(bias_grad->rawData());
            }
            else
            {
                bias_grad_ = nullptr;
            }
        }

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

            cached_in_features_ = static_cast<int>(input_shape.back());

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

			// TJT: Better here is outer_size_ . The use of batch_size_ is misleading.
            cached_batch_size_ = 1;
            for (size_t i = 0; i + 1 < input_shape.size(); ++i)
            {
                cached_batch_size_ *= static_cast<int>(input_shape[i]);
            }

            cached_out_features_ = static_cast<int>(config_.getOutputFeatures());

            cached_cublaslt_handle_ = context_->getCublasLtHandle();
            use_cublaslt_ = (cached_cublaslt_handle_ != nullptr) && supportsCuBLASLt();

            cached_precision_policy_ = config_.getPrecisionPolicy();

            if (use_cublaslt_)
            {
                try
                {
                    buildCublasLtPlans();
                }
                catch (const std::exception& e)
                {
                    Utils::Logger::warning(
                        std::string( "Failed to build cuBLASLt plans, falling back to custom kernels: " ) + e.what() );
                    
                    use_cublaslt_ = false;
                }
            }

            UnaryOperationBase::build( input_shape );
        }

        void forward( const ITensor& input, ITensor& output ) const override
        {
            const auto& input_shape = input.shape();

            // Compute leading dim size (all dims except last)
            int64_t batch_size = 1;
            for ( size_t i = 0; i < input_shape.size() - 1; ++i )
            {
                batch_size *= input_shape[ i ];
            }

            // Last dimension should match in_features (already validated at build)
            // int64_t actual_in_features = input_shape.back();  // Should equal cached_in_features_

            const NativeType* input_ptr = static_cast<const NativeType*>(input.rawData());
            NativeType* output_ptr = static_cast<NativeType*>(output.rawData());

            cudaStream_t stream = context_->getStream();

            if (use_cublaslt_)
            {
                const float alpha = 1.0f;
                const float beta = 0.0f;

                execute_plan<NativeType>(
                    cached_cublaslt_handle_,
                    forward_plan_cache_.get( static_cast<int>(batch_size) ),
                    &alpha,
                    input_ptr, weight_,
                    &beta,
                    output_ptr,
                    bias_,
                    stream,
                    context_->getCublasLtWorkspace(),
                    context_->getCublasLtWorkspaceSize() );

                // DEBUG: To imediately catch CUDA errors
                this->context_->synchronize();

                return;
            }

            // REVIEW: Requires testing. Focus is currently on CublasLt plan caching and implementation.
            // We need to revisit this code block

            // Fallback to custom non-cublasLt kernel
            Detail::cuda_matmul_impl<NativeType>::forward(
                output_ptr, input_ptr,
                weight_, bias_,
                static_cast<int>(batch_size),
                cached_in_features_, cached_out_features_,
                stream );
        }

        void backward(
            const ITensor& input,
            const ITensor& output_grad,
            ITensor& input_grad ) const override
        {
            if ( !this->isTraining() )
            {
                throw std::runtime_error( "CudaLinearOp::backward called in inference mode" );
			}

            const auto& grad_shape = output_grad.shape();
            int64_t batch_size = 1;
            for ( size_t i = 0; i + 1 < grad_shape.size(); ++i )
                batch_size *= grad_shape[ i ];

            const NativeType* input_ptr = static_cast<const NativeType*>(input.rawData());
            const NativeType* output_grad_ptr = static_cast<const NativeType*>(output_grad.rawData());
            NativeType* input_grad_ptr = static_cast<NativeType*>(input_grad.rawData());

            cudaStream_t stream = context_->getStream();

            if (use_cublaslt_)
            {
                const float alpha = 1.0f;
                const float beta = 0.0f;
                const float beta_accum = 1.0f; // Accumulate into weight grad

                // dX[batch, in] = dY[batch, out] @ weight[out, in]
                execute_plan<NativeType>(
                    cached_cublaslt_handle_,
                    backward_input_plan_cache_.get( static_cast<int>(batch_size) ),
                    &alpha,
                    output_grad_ptr, weight_,
                    &beta,
                    input_grad_ptr,
                    /* bias */ nullptr,
                    stream,
                    context_->getCublasLtWorkspace(),
                    context_->getCublasLtWorkspaceSize() );

                // dW[out, in] = dY^T @ X  (always full batch)
                // NOTE: This plan is not cached as batch size does not change during training.
                execute_plan<NativeType>(
                    cached_cublaslt_handle_,
                    backward_weight_plan_,
                    &alpha,
                    output_grad_ptr, input_ptr,
                    &beta_accum,
                    weight_grad_,
                    /* bias */ nullptr,
                    stream,
                    context_->getCublasLtWorkspace(),
                    context_->getCublasLtWorkspaceSize() );

                // dB[out] = sum(dY, dim=0)
                if ( bias_grad_ != nullptr )
                {
                    Detail::compute_bias_gradient(
                        bias_grad_,
                        output_grad_ptr,
                        static_cast<int>(batch_size),
                        cached_out_features_,
                        stream );
                }

                return;
            }

            // Fallback to custom non-cublasLt kernels
            Detail::cuda_matmul_impl<NativeType>::backward(
                input_grad_ptr, weight_grad_, bias_grad_,
                output_grad_ptr, input_ptr, weight_,
                cached_batch_size_,
                cached_in_features_, cached_out_features_,
                stream );
        }

        void decode( const ITensor& input, ITensor& output ) const override
        {
            const NativeType* input_ptr = static_cast<const NativeType*>(input.rawData());
            NativeType* output_ptr = static_cast<NativeType*>(output.rawData());
            cudaStream_t stream = context_->getStream();

            Detail::cuda_matvec_impl<NativeType>::decode(
                output_ptr,
                input_ptr,
                weight_,
                bias_,
                cached_in_features_,
                cached_out_features_,
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
        CudaExecutionContext* context_;
        Detail::cuda_matmul_impl<NativeType> impl_;

        const NativeType* weight_{ nullptr };
        const NativeType* bias_{ nullptr };

        NativeType* weight_grad_{ nullptr };
        NativeType* bias_grad_{ nullptr };

        int64_t weight_out_features_{ 0 };
        int64_t weight_in_features_{ 0 };

        int cached_batch_size_{ 0 };
        int cached_in_features_{ 0 };
        int cached_out_features_{ 0 };

        cublasLtHandle_t cached_cublaslt_handle_{ nullptr };
        bool use_cublaslt_{ false };
        ComputePrecision::Policy cached_precision_policy_;

        CublasLtPlanCache<CublasLtMatMulPlan<NativeType>> forward_plan_cache_;
        CublasLtPlanCache<CublasLtMatMulPlan<NativeType>> backward_input_plan_cache_;
        CublasLtMatMulPlan<NativeType> backward_weight_plan_;

        cudaDataType_t cuda_data_type_{};
        cublasComputeType_t compute_type_{};
        cudaDataType_t scale_type_{};

        //Detail::CublasLtMatMulPlan<NativeType> forward_plan_;
        //Detail::CublasLtMatMulPlan<NativeType> backward_input_plan_;
        //Detail::CublasLtMatMulPlan<NativeType> backward_weight_plan_;

        constexpr bool supportsCuBLASLt() const
        {
            return std::is_same_v<NativeType, float> ||
                std::is_same_v<NativeType, half> ||
                std::is_same_v<NativeType, nv_bfloat16>;
        }

        void buildCublasLtPlans()
        {
            cudaDataType_t cuda_data_type = getCudaDataType();
            cublasComputeType_t compute_type;
            cudaDataType_t scale_type;

            getComputeTypes( compute_type, scale_type );

            // Store for use by cache builders
            cuda_data_type_ = cuda_data_type;
            compute_type_ = compute_type;
            scale_type_ = scale_type;

            forward_plan_cache_ = CublasLtPlanCache<CublasLtMatMulPlan<NativeType>>(
                cached_batch_size_,
                [&]( int bucket )
                {
                    return Detail::build_forward_plan<NativeType>(
                        cached_cublaslt_handle_,
                        bucket,
                        cached_in_features_,
                        cached_out_features_,
                        config_.hasBias(),
                        cuda_data_type_,
                        compute_type_,
                        scale_type_ );
                } );

            backward_input_plan_cache_ = CublasLtPlanCache<CublasLtMatMulPlan<NativeType>>(
                cached_batch_size_,
                [&]( int bucket )
                {
                    return Detail::build_backward_input_plan<NativeType>(
                        cached_cublaslt_handle_,
                        bucket,
                        cached_in_features_,
                        cached_out_features_,
                        cuda_data_type_,
                        compute_type_,
                        scale_type_ );
                } );

            backward_weight_plan_ = Detail::build_backward_weight_plan<NativeType>(
                cached_cublaslt_handle_,
                cached_batch_size_,
                cached_in_features_,
                cached_out_features_,
                cuda_data_type_,
                compute_type_,
                scale_type_ );
        }

        cudaDataType_t getCudaDataType() const
        {
            if constexpr (std::is_same_v<NativeType, float>)
            {
                return CUDA_R_32F;
            }
            else if constexpr (std::is_same_v<NativeType, half>)
            {
                return CUDA_R_16F;
            }
            else if constexpr (std::is_same_v<NativeType, nv_bfloat16>)
            {
                return CUDA_R_16BF;
            }
            else if constexpr (std::is_same_v<NativeType, __nv_fp8_e4m3>)
            {
                return CUDA_R_8F_E4M3;
            }
            else if constexpr (std::is_same_v<NativeType, __nv_fp8_e5m2>)
            {
                return CUDA_R_8F_E5M2;
            }
        }

        void getComputeTypes( cublasComputeType_t& compute_type, cudaDataType_t& scale_type ) const
        {
            scale_type = CUDA_R_32F;

            switch (cached_precision_policy_)
            {
                case ComputePrecision::Policy::Native:
                case ComputePrecision::Policy::Accuracy:
                    if constexpr (std::is_same_v<NativeType, half>)
                    {
                        compute_type = CUBLAS_COMPUTE_16F;
                    }
                    else if constexpr (std::is_same_v<NativeType, nv_bfloat16>)
                    {
                        compute_type = CUBLAS_COMPUTE_32F;
                    }
                    else
                    {
                        compute_type = CUBLAS_COMPUTE_32F;
                    }
                    break;

                case ComputePrecision::Policy::Performance:
                case ComputePrecision::Policy::Auto:
                default:
                    if constexpr (std::is_same_v<NativeType, half>)
                    {
                        compute_type = CUBLAS_COMPUTE_32F_FAST_16F;
                    }
                    else if constexpr (std::is_same_v<NativeType, nv_bfloat16>)
                    {
                        compute_type = CUBLAS_COMPUTE_32F_FAST_16BF;
                    }
                    else
                    {
                        compute_type = CUBLAS_COMPUTE_32F;
                    }
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

            OperationRegistry::instance().registerUnaryOperation<DeviceType::Cuda, TensorDataType::FP32, TensorDataType::FP32>(
                opName,
                []( IExecutionContext* context,
                    const ComponentConfig& config ) -> std::shared_ptr<UnaryOperation<DeviceType::Cuda, TensorDataType::FP32>>
                {
                    const auto& linearConfig = static_cast<const LinearConfig&>(config);
                    return std::make_shared<CudaLinearOp<TensorDataType::FP32>>( context, linearConfig );
                }
            );

            /*OperationRegistry::instance().registerUnaryOperation<DeviceType::Cuda, TensorDataType::FP16>(
                opName,
                []( std::shared_ptr<ExecutionContext<DeviceType::Cuda>> context,
                    const ModuleConfig& config ) -> std::shared_ptr<UnaryOperation<DeviceType::Cuda, TensorDataType::FP16>>
                {
                    const auto& linearConfig = static_cast<const LinearConfig&>(config);
                    return std::make_shared<CudaLinearOp<TensorDataType::FP16>>( context, linearConfig );
                }
            );

            OperationRegistry::instance().registerUnaryOperation<DeviceType::Cuda, TensorDataType::BF16>(
                opName,
                []( std::shared_ptr<ExecutionContext<DeviceType::Cuda>> context,
                    const ModuleConfig& config ) -> std::shared_ptr<UnaryOperation<DeviceType::Cuda, TensorDataType::BF16>>
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