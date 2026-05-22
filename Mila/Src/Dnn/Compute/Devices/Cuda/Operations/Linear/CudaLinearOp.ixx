/**
 * @file CudaLinearOp.ixx
 * @brief CUDA Linear operation with compile-time weight quantization policy dispatch.
 *
 * TWeightQuant = NoWeightQuant selects the standard BF16/FP32 cuBLASLt path.
 * TWeightQuant = PerChannelFp8<> selects the FP8 weight + BF16 activation mixed-precision
 * path. quantize() and setWeightScales() are only callable on the PerChannelFp8<> 
 * instantiation (enforced via requires). All other operations are unaware they exist.
 */

module;
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
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
#include <cmath>
#include "Kernels/Linear.cuh"

export module Compute.CudaLinearOp;
import :Plans;
import :Dispatch;
import :Quantize;

import Dnn.Components.LinearConfig;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.ComponentConfig;
import Dnn.Quantization.Weight.Policies;
import Compute.OperationBase;
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
import Serialization.Tensor;
import CublasLt.Error;
import Logging.Logger;

import Dnn.TensorOps;
import Dnn.TensorHelpers;

namespace Mila::Dnn::Compute::Cuda::Linear
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute::Cuda;
    using namespace Mila::Dnn::Quant::Weight;
    using namespace Mila::Dnn::Serialization;

    /**
     * @brief CUDA Linear operation with compile-time weight quantization policy dispatch.
     *
     * Forward:  output = input * weight^T + bias
     * Backward (NoWeightQuant only):
     *   input_grad  = output_grad * weight
     *   weight_grad = output_grad^T * input  (accumulated)
     *   bias_grad   = sum(output_grad, dim=0)
     *
     * When TWeightQuant = PerChannelFp8<>, weights are stored as FP8_E4M3 with one
     * float32 scale per output channel. quantize() performs the one-time host-side
     * BF16->FP8 conversion at load time. cuBLASLt executes the mixed-precision GEMM
     * (BF16 activation x FP8 weight -> BF16 output) natively on the forward path.
     * Backward is not supported on the quantized path (inference only).
     *
     * @tparam TComputePrecision  Activation and accumulation precision.
     * @tparam TWeightQuant       Weight quantization policy. Defaults to NoWeightQuant.
     */
    export template<TensorDataType TComputePrecision, WeightQuantPolicy TWeightQuant = NoWeightQuant>
        requires PrecisionSupportedOnDevice<TComputePrecision, DeviceType::Cuda>
    class CudaLinearOp : public Operation<DeviceType::Cuda, TComputePrecision>
    {
    public:
        using MR = CudaDeviceMemoryResource;
        using TensorType = Tensor<TComputePrecision, MR>;
        using ComputeType = typename TensorDataTypeMap<TComputePrecision>::device_type;

        static constexpr bool kIsQuantized = TWeightQuant::kIsQuantized;

        static constexpr TensorDataType kWeightDtype = kIsQuantized
            ? TWeightQuant::kStorageDtype : TComputePrecision;

        using WeightType = typename TensorDataTypeMap<kWeightDtype>::device_type;

        using CudaExecutionContext = ExecutionContext<DeviceType::Cuda>;

        CudaLinearOp( IExecutionContext* context, const LinearConfig& config )
            : context_( validateExecutionContext_<DeviceType::Cuda>( context, "CudaLinearOp" ) ),
              config_( config )
        {
            config_.validate();
        }

        void setParameters( ITensor* weight, ITensor* bias ) override
        {
            if ( !weight )
            {
                throw std::invalid_argument( "CudaLinearOp::setParameters - weight is required" );
            }

            if ( weight->getDeviceType() != DeviceType::Cuda )
            {
                throw std::invalid_argument( "CudaLinearOp::setParameters - weight must be a CUDA tensor" );
            }

            const auto& weight_shape = weight->shape();

            if ( weight_shape.size() != 2 )
            {
                throw std::invalid_argument( "CudaLinearOp::setParameters - weight must be 2D" );
            }

            weight_ = static_cast<const WeightType*>(weight->rawData());
            weight_out_features_ = weight_shape[ 0 ];
            weight_in_features_ = weight_shape[ 1 ];

            if ( config_.hasBias() )
            {
                if ( !bias )
                {
                    throw std::invalid_argument( "CudaLinearOp::setParameters - bias expected but null" );
                }

                if ( bias->getDeviceType() != DeviceType::Cuda )
                {
                    throw std::invalid_argument( "CudaLinearOp::setParameters - bias must be a CUDA tensor" );
                }

                bias_ = static_cast<const ComputeType*>(bias->rawData());
            }
            else
            {
                bias_ = nullptr;
            }
        }

        /**
         * @brief Bind the per-channel FP32 weight scale tensor produced by quantize().
         *
         * Must be called after quantize() and before the first forward(). The scale
         * pointer is stored and passed to the cuBLASLt FP8 matmul descriptor.
         *
         * @param scales Device tensor of shape [output_features], dtype Float32.
         */
        void setWeightScales( ITensor* scales ) requires kIsQuantized
        {
            if ( !scales )
            {
                throw std::invalid_argument( "CudaLinearOp::setWeightScales - scales tensor is required" );
            }

            if ( scales->getDeviceType() != DeviceType::Cuda )
            {
                throw std::invalid_argument( "CudaLinearOp::setWeightScales - scales must be a CUDA tensor" );
            }

            weight_scales_ = static_cast<const float*>(scales->rawData());
        }

        /**
         * @brief Quantize a BF16 host blob to FP8_E4M3 with per-channel FP32 scales.
         *
         * Runs once at model load time. Delegates to Detail::quantize_fp8_per_channel()
         * (pre-compiled by NVCC in the :Quantize partition), which performs per-channel
         * absmax scaling and uploads both the FP8 weight tensor and the FP32 scale tensor
         * to device. The BF16 source blob is never retained on device.
         *
         * @param blob           Host BF16 weight blob from the model archive.
         * @param weight_out     Device FP8_E4M3 tensor [out_features, in_features].
         * @param scales_out     Device Float32 tensor [out_features].
         * @param expected_shape Expected weight shape for validation.
         */
        void quantize(
            const ITensorBlob& blob,
            ITensor& weight_out,
            ITensor& scales_out,
            const shape_t& expected_shape ) requires kIsQuantized
        {
            Detail::quantize_fp8_per_channel( blob, weight_out, scales_out, expected_shape );
        }

        void setGradients( ITensor* weight_grad, ITensor* bias_grad ) override
        {
            if constexpr ( kIsQuantized )
            {
                throw std::logic_error( "CudaLinearOp: gradient computation is not supported on quantized paths" );
            }

            if ( !weight_grad )
            {
                throw std::invalid_argument( "CudaLinearOp::setGradients - weight gradient is required" );
            }

            if ( weight_grad->getDeviceType() != DeviceType::Cuda )
            {
                throw std::invalid_argument( "CudaLinearOp::setGradients - weight gradient must be a CUDA tensor" );
            }

            weight_grad_ = static_cast<ComputeType*>(weight_grad->rawData());

            if ( config_.hasBias() )
            {
                if ( !bias_grad )
                {
                    throw std::invalid_argument( "CudaLinearOp::setGradients - bias gradient expected but null" );
                }

                if ( bias_grad->getDeviceType() != DeviceType::Cuda )
                {
                    throw std::invalid_argument( "CudaLinearOp::setGradients - bias gradient must be a CUDA tensor" );
                }

                bias_grad_ = static_cast<ComputeType*>(bias_grad->rawData());
            }
            else
            {
                bias_grad_ = nullptr;
            }
        }

        void build( const BuildContext& build_context ) override
        {
            const auto& input_shape = build_context.inputShape();

            if ( weight_ == nullptr )
            {
                throw std::runtime_error( "CudaLinearOp::build - setParameters() must be called before build()" );
            }

            if ( config_.hasBias() && bias_ == nullptr )
            {
                throw std::runtime_error( "CudaLinearOp::build - bias expected by config but not bound" );
            }

            if ( input_shape.empty() )
            {
                throw std::invalid_argument( "CudaLinearOp::build - input shape cannot be empty" );
            }

            cached_in_features_ = static_cast<int>(input_shape.back());

            if ( weight_out_features_ != config_.getOutputFeatures() )
            {
                throw std::invalid_argument( std::format(
                    "CudaLinearOp::build - weight output features mismatch: expected {}, got {}",
                    config_.getOutputFeatures(), weight_out_features_ ) );
            }

            if ( weight_in_features_ != cached_in_features_ )
            {
                throw std::invalid_argument( std::format(
                    "CudaLinearOp::build - weight input features mismatch: expected {}, got {}",
                    cached_in_features_, weight_in_features_ ) );
            }

            cached_outer_size_ = 1;

            for ( size_t i = 0; i + 1 < input_shape.size(); ++i )
            {
                cached_outer_size_ *= static_cast<int>(input_shape[ i ]);
            }

            out_features_ = static_cast<int>(config_.getOutputFeatures());
            cached_cublaslt_handle_ = context_->getCublasLtHandle();
            use_cublaslt_ = (cached_cublaslt_handle_ != nullptr) && supportsCuBLASLt();

            if ( use_cublaslt_ )
            {
                try
                {
                    buildCublasLtPlans();
                }
                catch ( const std::exception& e )
                {
                    Logging::Logger::warning(
                        std::string( "CudaLinearOp: failed to build cuBLASLt plans, falling back: " ) + e.what() );
                    use_cublaslt_ = false;
                }
            }

            Operation<DeviceType::Cuda, TComputePrecision>::build( build_context );
        }

        /**
         * @brief Forward pass: output = input * weight^T + bias
         *
         * Dispatch priority:
         *   1. outer_size == 1 (decode): fused matvec kernel — both FP32/BF16 and FP8 paths.
         *   2. outer_size > 1, kIsQuantized (FP8 prefill, Phase 1): token-by-token matvec loop.
         *      TODO(Phase 2): Replace with cuBLASLt FP8 mixed-precision GEMM once Plans.ixx
         *      supports separate dataTypeA / dataTypeB descriptors.
         *   3. outer_size > 1, !kIsQuantized: cuBLASLt GEMM (BF16/FP32).
         */
        void forward( const TensorType& input, TensorType& output ) const
        {
            const int outer_size = static_cast<int>(input.size()) / cached_in_features_;

            const ComputeType* input_ptr = static_cast<const ComputeType*>(input.rawData());
            ComputeType* output_ptr = static_cast<ComputeType*>(output.rawData());
            cudaStream_t stream = context_->getStream();

            if ( outer_size == 1 )
            {
                Detail::cuda_matvec_impl<ComputeType, WeightType>::decode(
                    output_ptr, input_ptr,
                    weight_, weight_scales_,
                    bias_,
                    cached_in_features_, out_features_,
                    stream );

                return;
            }

            // FP8 prefill fallback: no cuBLASLt FP8 mixed-precision plan yet.
            // Dispatch the single-token matvec kernel once per input token.
            // Each launch is async on the same stream so the GPU stays busy; the
            // kernel launch overhead (~3-5 µs each) is acceptable for Phase 1 validation.
            if constexpr ( kIsQuantized )
            {
                for ( int t = 0; t < outer_size; ++t )
                {
                    Detail::cuda_matvec_impl<ComputeType, WeightType>::decode(
                        output_ptr + static_cast<ptrdiff_t>(t) * out_features_,
                        input_ptr  + static_cast<ptrdiff_t>(t) * cached_in_features_,
                        weight_, weight_scales_,
                        bias_,
                        cached_in_features_, out_features_,
                        stream );
                }

                return;
            }
            else
            {
                if ( use_cublaslt_ )
                {
                    const float alpha = 1.0f;
                    const float beta = 0.0f;

                    execute_plan<ComputeType>(
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

                    return;
                }

                throw std::runtime_error( "CudaLinearOp: no valid forward execution path available" );
            }
        }

        /**
         * @brief Backward pass. Not supported on the quantized path.
         *
         * @param input       Saved forward input.
         * @param output_grad Upstream gradient.
         * @param input_grad  Output: gradient with respect to forward input.
         */
        void backward( const TensorType& input, const TensorType& output_grad, TensorType& input_grad ) const
        {
            if constexpr ( kIsQuantized )
            {
                throw std::logic_error( "CudaLinearOp: backward is not supported on the FP8 quantized path" );
            }
            else
            {
                if ( this->isEvalMode() )
                {
                    throw std::runtime_error( "CudaLinearOp::backward: not available in eval mode" );
                }

                const int outer_size = static_cast<int>(output_grad.size()) / out_features_;

                const ComputeType* input_ptr = static_cast<const ComputeType*>(input.rawData());
                const ComputeType* output_grad_ptr = static_cast<const ComputeType*>(output_grad.rawData());
                ComputeType* input_grad_ptr = static_cast<ComputeType*>(input_grad.rawData());
                cudaStream_t stream = context_->getStream();

                if ( use_cublaslt_ )
                {
                    const float alpha = 1.0f;
                    const float beta = 0.0f;
                    const float beta_accum = 1.0f;

                    execute_plan<ComputeType>(
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

                    execute_plan<ComputeType>(
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
                            out_features_,
                            stream );
                    }

                    return;
                }

                throw std::runtime_error( "CudaLinearOp: no valid backward execution path available" );
            }
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

        // Weight pointer typed to WeightType — differs from ComputeType on the FP8 path.
        const WeightType* weight_{ nullptr };

        // Per-channel FP32 scales [out_features] — non-null on kIsQuantized path only.
        const float* weight_scales_{ nullptr };

        const ComputeType* bias_{ nullptr };

        ComputeType* weight_grad_{ nullptr };
        ComputeType* bias_grad_{ nullptr };

        int64_t weight_out_features_{ 0 };
        int64_t weight_in_features_{ 0 };

        int cached_outer_size_{ 0 };
        int cached_in_features_{ 0 };
        int out_features_{ 0 };

        cublasLtHandle_t cached_cublaslt_handle_{ nullptr };
        bool use_cublaslt_{ false };
        
        CublasLtPlanCache<CublasLtMatMulPlan<ComputeType>> forward_plan_cache_;
        CublasLtPlanCache<CublasLtMatMulPlan<ComputeType>> backward_input_plan_cache_;
        CublasLtMatMulPlan<ComputeType> backward_weight_plan_;

        cudaDataType_t cuda_data_type_{};
        cudaDataType_t cuda_weight_data_type_{};
        cublasComputeType_t compute_type_{};
        cudaDataType_t scale_type_{};

        /**
         * @brief Returns true if a cuBLASLt plan can be built for this instantiation.
         *
         * For FP8 paths, requires SM >= 8.9 (Ada Lovelace) for native FP8 tensor core support.
         */
        bool supportsCuBLASLt() const
        {
            if constexpr ( kIsQuantized )
            {
                int device = 0;
                int major = 0, minor = 0;
                cudaGetDevice( &device );
                cudaDeviceGetAttribute( &major, cudaDevAttrComputeCapabilityMajor, device );
                cudaDeviceGetAttribute( &minor, cudaDevAttrComputeCapabilityMinor, device );

                return (major > 8) || (major == 8 && minor >= 9);
            }
            else
            {
                return std::is_same_v<ComputeType, float> ||
                    std::is_same_v<ComputeType, half> ||
                    std::is_same_v<ComputeType, nv_bfloat16>;
            }
        }

        cudaDataType_t getActivationCudaDataType() const
        {
            if constexpr ( std::is_same_v<ComputeType, float> )
                return CUDA_R_32F;
            else if constexpr ( std::is_same_v<ComputeType, half> )
                return CUDA_R_16F;
            else if constexpr ( std::is_same_v<ComputeType, nv_bfloat16> )
                return CUDA_R_16BF;
            else
                static_assert( !sizeof( ComputeType ), "Unsupported ComputeType for cuBLASLt" );
        }

        cudaDataType_t getWeightCudaDataType() const
        {
            if constexpr ( kIsQuantized )
                return CUDA_R_8F_E4M3;
            else
                return getActivationCudaDataType();
        }

        void getComputeTypes( cublasComputeType_t& compute_type, cudaDataType_t& scale_type ) const
        {
            scale_type = CUDA_R_32F;

            if constexpr ( kIsQuantized )
            {
                // Mixed-precision FP8 x BF16 -> BF16: accumulate in FP32.
                compute_type = CUBLAS_COMPUTE_32F;
                return;
            }

            // REVIEW: we need only support bf16 for CUDA

            if constexpr ( std::is_same_v<ComputeType, half> )
                compute_type = CUBLAS_COMPUTE_32F_FAST_16F;
            else if constexpr ( std::is_same_v<ComputeType, nv_bfloat16> )
                compute_type = CUBLAS_COMPUTE_32F_FAST_16BF;
            else
                compute_type = CUBLAS_COMPUTE_32F;
        }

        void buildCublasLtPlans()
        {
            cuda_data_type_ = getActivationCudaDataType();
            cuda_weight_data_type_ = getWeightCudaDataType();

            getComputeTypes( compute_type_, scale_type_ );

            // TODO(Alpha.5): build_forward_plan / build_strided_plan require separate
            // data_type_A (activation) and data_type_B (weight) parameters to support
            // the mixed-precision FP8 descriptor. Plans.ixx must be updated before
            // the FP8 prefill cuBLASLt path is activated.
            if constexpr ( kIsQuantized )
            {
                Logging::Logger::warning(
                    "CudaLinearOp: FP8 cuBLASLt prefill plan pending Plans.ixx mixed-precision support" );
                use_cublaslt_ = false;
                return;
            }

            forward_plan_cache_ = CublasLtPlanCache<CublasLtMatMulPlan<ComputeType>>(
                cached_outer_size_,
                [&]( int bucket )
                {
                    return Detail::build_forward_plan<ComputeType>(
                        cached_cublaslt_handle_,
                        bucket,
                        cached_in_features_,
                        out_features_,
                        config_.hasBias(),
                        cuda_data_type_,
                        compute_type_,
                        scale_type_ );
                } );

            backward_input_plan_cache_ = CublasLtPlanCache<CublasLtMatMulPlan<ComputeType>>(
                cached_outer_size_,
                [&]( int bucket )
                {
                    return Detail::build_backward_input_plan<ComputeType>(
                        cached_cublaslt_handle_,
                        bucket,
                        cached_in_features_,
                        out_features_,
                        cuda_data_type_,
                        compute_type_,
                        scale_type_ );
                } );

            backward_weight_plan_ = Detail::build_backward_weight_plan<ComputeType>(
                cached_cublaslt_handle_,
                cached_outer_size_,
                cached_in_features_,
                out_features_,
                cuda_data_type_,
                compute_type_,
                scale_type_ );
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