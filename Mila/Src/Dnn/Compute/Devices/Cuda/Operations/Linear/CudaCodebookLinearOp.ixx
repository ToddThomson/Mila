/**
 * @file CudaCodebookLinearOp.ixx
 * @brief Linear operation for the sub-4-bit codebook weight policies (W2A16 / W3A16).
 * Experimental: outside the v0.20 surface. See Specifications/Qwen3.8.md sections 5 and 8.
 */

module;
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <format>
#include <stdexcept>
#include <string>
#include "Kernels/Codebook/CodebookDequantize.cuh"
#include "Kernels/Codebook/CodebookGemv.cuh"

export module Compute.CudaCodebookLinearOp;

import Dnn.Component;
import Dnn.Components.LinearConfig;
import Dnn.ITensor;
import Dnn.Tensor;
import Dnn.TensorDataType;
import Dnn.TensorTypes;
import Dnn.Quantization.Weight.Policies;
import Compute.CublasLtPlanCache;
import Compute.Cuda.CublasLtLinearPlan;
import Compute.CudaDeviceMemoryResource;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.IExecutionContext;
import Compute.OperationBase;
import Compute.OperationType;
import Dnn.Quantization.Weight.CodebookPacking;

namespace Mila::Dnn::Compute::Cuda::Linear
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Quant::Weight;

    /**
     * @brief A weight policy whose codes index a per-tensor codebook.
     *
     * Refines the production concept with the discriminator the codebook policies add.
     * Everything the production Linear dispatch requires is already satisfied, which is
     * the property the research track is testing.
     */
    export template<typename TPolicy>
    concept CodebookWeightPolicy =
        Mila::Dnn::Quant::Weight::WeightQuantPolicy<TPolicy> && TPolicy::kIsCodebook;

    /**
     * @brief CUDA Linear operation over pre-quantized codebook weights.
     *
     * Unlike every other quantized path in the tree, this operation never quantizes.
     * Phase 0 measured that data-free round-to-nearest at these bit widths destroys the
     * model, so the codes are produced offline by the converter with calibration and GPTQ
     * compensation and arrive already packed. quantize() is therefore absent rather than
     * throwing: Linear only calls it when the source blob is at compute precision, and a
     * codebook artifact never is.
     *
     * Every buffer is BORROWED. The component owns the packed codes, the scales, the
     * codebook and the high-bit plane; this operation caches raw pointers and frees
     * nothing, exactly as the production ops treat setParameters().
     *
     * @tparam TComputePrecision    Activation precision. BF16 only -- the kernels are
     *                              BF16 in, BF16 out, so no other row is advertised.
     * @tparam TWeightQuantization  PerGroupCodebook2<> or PerGroupCodebook3<>.
     */
    export template<TensorDataType TComputePrecision, CodebookWeightPolicy TWeightQuantization>
        requires (TComputePrecision == TensorDataType::BF16)
    class CudaCodebookLinearOp : public Operation<DeviceType::Cuda, TComputePrecision>
    {
    public:
        using MR = CudaDeviceMemoryResource;
        using TensorType = Tensor<TComputePrecision, MR>;
        using CudaExecutionContext = ExecutionContext<DeviceType::Cuda>;

        static constexpr int kCodeBits = TWeightQuantization::kCodeBits;
        static constexpr int kCodebookEntries = TWeightQuantization::kCodebookEntries;
        static constexpr int kGroupSize = TWeightQuantization::kQuantizationGroupSize;
        static constexpr bool kHasOneBitPlane = TWeightQuantization::kHasHighBitPlane;

        CudaCodebookLinearOp( IExecutionContext* context, const LinearConfig& config )
            : context_( validateExecutionContext_<DeviceType::Cuda>( context, "CudaCodebookLinearOp" ) ),
              config_( config )
        {
            config_.validate();
        }

        CudaCodebookLinearOp( const CudaCodebookLinearOp& ) = delete;
        CudaCodebookLinearOp& operator=( const CudaCodebookLinearOp& ) = delete;

        /**
         * @brief Bind the module-owned packed codes and bias.
         *
         * The weight tensor holds the primary code plane: two bits per element, four per
         * byte, so its physical extent is [out, in/4].
         */
        void setParameters( ITensor* weight, ITensor* bias ) override
        {
            if ( weight == nullptr )
            {
                throw std::invalid_argument(
                    "CudaCodebookLinearOp::setParameters - packed weight is required" );
            }

            if ( weight->getDeviceType() != DeviceType::Cuda )
            {
                throw std::invalid_argument(
                    "CudaCodebookLinearOp::setParameters - weight must be a CUDA tensor" );
            }

            if ( config_.hasBias() && bias == nullptr )
            {
                throw std::invalid_argument(
                    "CudaCodebookLinearOp::setParameters - bias expected by config but not provided" );
            }

            plane_two_bits_ = static_cast<const std::uint8_t*>( weight->rawData() );
            bias_ = bias;
        }

        /// Bind the module-owned per-group scales: IEEE half, [out, in / group_size].
        void setWeightScales( ITensor* scales )
        {
            if ( scales == nullptr )
            {
                throw std::invalid_argument(
                    "CudaCodebookLinearOp::setWeightScales - scales are required" );
            }

            scales_ = static_cast<const __half*>( scales->rawData() );
        }

        /**
         * @brief Bind the per-tensor codebook and, for a 3-bit format, the high-bit plane.
         *
         * @param codebook  FP32, kCodebookEntries entries.
         * @param highPlane UINT8, [out, in/8]; null for a 2-bit policy.
         */
        void setCodebookTensors( ITensor* codebook, ITensor* highPlane )
        {
            if ( codebook == nullptr )
            {
                throw std::invalid_argument(
                    "CudaCodebookLinearOp::setCodebookTensors - codebook is required" );
            }

            if ( kHasOneBitPlane != (highPlane != nullptr) )
            {
                throw std::invalid_argument( std::format(
                    "CudaCodebookLinearOp::setCodebookTensors - {}-bit policy {} a high-bit plane",
                    kCodeBits, kHasOneBitPlane ? "requires" : "does not take" ) );
            }

            codebook_ = static_cast<const float*>( codebook->rawData() );
            plane_one_bit_ = highPlane != nullptr
                ? static_cast<const std::uint8_t*>( highPlane->rawData() )
                : nullptr;
        }

        /**
         * @brief Post-load hook; nothing is derived from these weights.
         *
         * The production FP4 path uses this to compute the FP8 activation-prefill scalar
         * from the group scales. A codebook carries no such derived quantity -- every
         * value it needs was fitted offline against calibration data -- so this exists to
         * satisfy the component contract and deliberately does nothing.
         */
        void onQuantizedWeightsLoaded()
        {
        }

        void build( const BuildContext& build_context ) override
        {
            if ( plane_two_bits_ == nullptr || scales_ == nullptr || codebook_ == nullptr )
            {
                throw std::runtime_error(
                    "CudaCodebookLinearOp::build - packed codes, scales and codebook must all "
                    "be bound before build()" );
            }

            const auto& input_shape = build_context.inputShape();

            if ( input_shape.empty() )
            {
                throw std::invalid_argument( "CudaCodebookLinearOp::build - input shape cannot be empty" );
            }

            weight_columns_ = config_.getInputFeatures();
            weight_rows_ = config_.getOutputFeatures();

            if ( input_shape.back() != weight_columns_ )
            {
                throw std::invalid_argument( std::format(
                    "CudaCodebookLinearOp::build - input features ({}) do not match the "
                    "configured input features ({})", input_shape.back(), weight_columns_ ) );
            }

            // The GEMV loads codes as uint32 words and one scale per chunk of 16, so a row
            // that is not a whole number of both would read past its end on the last word.
            if ( weight_columns_ % 16 != 0 || weight_columns_ % kGroupSize != 0 )
            {
                throw std::invalid_argument( std::format(
                    "CudaCodebookLinearOp::build - input features ({}) must be a multiple of "
                    "16 and of the group size ({})", weight_columns_, kGroupSize ) );
            }

            dim_t outer_size = 1;

            for ( std::size_t axis = 0; axis + 1 < input_shape.size(); ++axis )
                outer_size *= input_shape[axis];

            built_outer_size_ = narrowToKernelIndex( outer_size );
            cublaslt_handle_ = context_->getCublasLtHandle();

            if ( cublaslt_handle_ != nullptr )
            {
                // Staged BF16 x BF16: both operands are the compute precision by the time
                // the GEMM runs, so this is the UNQUANTIZED plan shape and the native bias
                // epilogue is available -- unlike the FP4 path, which disables it because
                // its plan is genuinely mixed-precision.
                forward_plan_cache_ = CublasLtPlanCache<CublasLtLinearPlan<TComputePrecision>>(
                    built_outer_size_,
                    [&]( int bucket )
                    {
                        return build_linear_plan<TComputePrecision>(
                            cublaslt_handle_,
                            bucket,
                            narrowToKernelIndex( weight_columns_ ),
                            narrowToKernelIndex( weight_rows_ ),
                            config_.hasBias(),
                            CUBLAS_COMPUTE_32F_FAST_16BF,
                            CUDA_R_32F );
                    } );
            }

            this->is_built_ = true;
        }

        void forward( const TensorType& input, TensorType& output ) const
        {
            const dim_t outer_size = input.size() / weight_columns_;

            const auto* input_pointer = static_cast<const __nv_bfloat16*>(input.rawData());
            auto* output_pointer = static_cast<__nv_bfloat16*>(output.rawData());
            const auto* bias_pointer = bias_ != nullptr
                ? static_cast<const __nv_bfloat16*>(bias_->rawData())
                : nullptr;

            const int columns = narrowToKernelIndex( weight_columns_ );
            const int rows = narrowToKernelIndex( weight_rows_ );
            cudaStream_t stream = context_->getStream();

            if ( outer_size == 1 )
            {
                if constexpr ( kHasOneBitPlane )
                {
                    launch_codebook3_matvec_decode(
                        stream, output_pointer, input_pointer,
                        plane_two_bits_, plane_one_bit_, scales_, codebook_,
                        bias_pointer, columns, rows, kGroupSize );
                }
                else
                {
                    launch_codebook2_matvec_decode(
                        stream, output_pointer, input_pointer,
                        plane_two_bits_, scales_, codebook_,
                        bias_pointer, columns, rows, kGroupSize );
                }

                return;
            }

            if ( cublaslt_handle_ == nullptr )
            {
                throw std::runtime_error(
                    "CudaCodebookLinearOp::forward - batched forward needs cuBLASLt, which the "
                    "execution context did not provide" );
            }

            // Two-phase prefill, the same structure as the proven FP4 baseline:
            //   1. expand the packed weights into the shared BF16 staging buffer
            //   2. standard BF16 cuBLASLt NT row-major GEMM, bias in the epilogue
            //
            // Unlike the W4A8-FP8 path, nothing here is sized by the plan's row bucket:
            // the staging buffer holds the WEIGHT matrix, whose extent is independent of
            // outer_size. The out-of-bounds hazard that path documents does not arise.
            //
            // This does not scale. One Qwen3.8 FFN matrix expands to 170 MiB per forward,
            // which is why Section 5 requires the fused tile-load GEMM before the 27B
            // chassis runs. It exists so Phase 1 can reach end-to-end at 3B scale.
            const std::size_t staging_bytes = static_cast<std::size_t>( rows )
                * static_cast<std::size_t>( columns ) * sizeof( __nv_bfloat16 );

            // Fetched per forward, never cached: the scratch buffer may move on grow.
            auto* staging = static_cast<__nv_bfloat16*>(
                context_->getDeviceScratchBuffer( staging_bytes ) );

            if constexpr ( kHasOneBitPlane )
            {
                launch_codebook3_dequantize_to_bf16(
                    stream, staging, plane_two_bits_, plane_one_bit_,
                    scales_, codebook_, columns, rows, kGroupSize );
            }
            else
            {
                launch_codebook2_dequantize_to_bf16(
                    stream, staging, plane_two_bits_,
                    scales_, codebook_, columns, rows, kGroupSize );
            }

            const float alpha = 1.0f;
            const float beta = 0.0f;

            execute_linear_plan<TComputePrecision>(
                cublaslt_handle_,
                forward_plan_cache_.get( narrowToKernelIndex( outer_size ) ),
                &alpha,
                input_pointer,
                staging,
                &beta,
                output_pointer,
                bias_pointer,
                nullptr,
                stream,
                context_->getCublasLtWorkspace(),
                context_->getCublasLtWorkspaceSize() );
        }

        /**
         * @brief Present to satisfy LinearOpConcept; the codebook path is inference only.
         *
         * The codes are a lossy non-differentiable encoding produced offline, so there is
         * no gradient to propagate through them and no plausible training use.
         */
        void backward( const TensorType&, const TensorType&, TensorType& ) const
        {
            throw std::logic_error(
                "CudaCodebookLinearOp: backward is not supported on the codebook path" );
        }

        OperationType getOperationType() const override
        {
            return OperationType::LinearOp;
        }

        std::string getName() const override
        {
            return std::format( "Cuda::CodebookLinearOp<W{}A16,g{}>", kCodeBits, kGroupSize );
        }

        const LinearConfig& getConfig() const
        {
            return config_;
        }

    private:
        CudaExecutionContext* context_{ nullptr };
        LinearConfig config_;
        ITensor* bias_{ nullptr };

        // Borrowed device pointers into component-owned tensors. Never freed here.
        const std::uint8_t* plane_two_bits_{ nullptr };
        const std::uint8_t* plane_one_bit_{ nullptr };
        const __half* scales_{ nullptr };
        const float* codebook_{ nullptr };

        cublasLtHandle_t cublaslt_handle_{ nullptr };
        CublasLtPlanCache<CublasLtLinearPlan<TComputePrecision>> forward_plan_cache_;
        int built_outer_size_{ 0 };

        dim_t weight_rows_{ 0 };
        dim_t weight_columns_{ 0 };
    };
}
