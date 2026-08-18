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
#include <algorithm>
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

        /**
         * @brief Ceiling on the BF16 staging buffer, in bytes.
         *
         * Section 5 of Specifications/Qwen3.8.md allots 461 MiB to all activations and
         * scratch, and expanding one Qwen3.8 FFN matrix whole would take 170 MiB of it. The
         * prefill pass therefore walks the output channels in strips sized to fit this cap
         * rather than expanding the matrix in one go.
         *
         * 32 MiB was picked from the measured trade at the Qwen FFN shape and prefill chunk
         * 512: it yields six strips at 28 MiB of staging for about 1.2x the monolithic
         * prefill time. Halving the buffer costs 1.03x and eighth-ing it 1.23x, so the curve
         * is shallow and this sits on the flat part of it. The fused tile-load GEMM that
         * Section 5 originally required for this was measured at 2.0x instead -- striping is
         * both cheaper to run and cheaper to build.
         */
        static constexpr std::size_t kMaxStagingBytes = 32ull * 1024 * 1024;

        /// Strip widths are tensor-core tile aligned along N so cuBLASLt is not handed a ragged shape.
        static constexpr dim_t kStripAlignment = 16;

        /**
         * @param max_staging_bytes Ceiling on the prefill staging buffer. Defaults to
         *        kMaxStagingBytes. It is a constructor parameter rather than a constant
         *        because a real budget belongs to whoever owns the device's memory, not to
         *        this operation, and because it is the only way to exercise the striped path
         *        at a shape small enough for a test.
         */
        CudaCodebookLinearOp( IExecutionContext* context, const LinearConfig& config,
            std::size_t max_staging_bytes = kMaxStagingBytes )
            : context_( validateExecutionContext_<DeviceType::Cuda>( context, "CudaCodebookLinearOp" ) ),
              config_( config ),
              max_staging_bytes_( max_staging_bytes )
        {
            if ( max_staging_bytes_ == 0 )
            {
                throw std::invalid_argument(
                    "CudaCodebookLinearOp - max_staging_bytes must be non-zero" );
            }

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

            planStripWidths();

            if ( cublaslt_handle_ != nullptr )
            {
                // Staged BF16 x BF16: both operands are the compute precision by the time
                // the GEMM runs, so this is the UNQUANTIZED plan shape and the native bias
                // epilogue is available -- unlike the FP4 path, which disables it because
                // its plan is genuinely mixed-precision.
                //
                // Each strip's C is a column slice of the full output, so every plan carries
                // the full row stride and writes its own channels in place. The trailing
                // strip is narrower whenever the row count is not a whole number of strips,
                // and gets its own cache; nothing else about the two differs.
                const auto makeCache = [&]( dim_t strip_rows )
                {
                    return CublasLtPlanCache<CublasLtLinearPlan<TComputePrecision>>(
                        built_outer_size_,
                        [&]( int bucket )
                        {
                            return build_linear_plan<TComputePrecision>(
                                cublaslt_handle_,
                                bucket,
                                narrowToKernelIndex( weight_columns_ ),
                                narrowToKernelIndex( strip_rows ),
                                config_.hasBias(),
                                CUBLAS_COMPUTE_32F_FAST_16BF,
                                CUDA_R_32F,
                                nullptr,
                                narrowToKernelIndex( weight_rows_ ) );
                        } );
                };

                forward_plan_cache_ = makeCache( strip_rows_ );

                if ( trailing_strip_rows_ != strip_rows_ )
                    trailing_plan_cache_ = makeCache( trailing_strip_rows_ );
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

            // Two-phase prefill, the same structure as the proven FP4 baseline, but STRIPED
            // along the output channels:
            //   for each strip of output rows:
            //     1. expand that strip's packed weights into the shared BF16 staging buffer
            //     2. standard BF16 cuBLASLt NT row-major GEMM, bias in the epilogue,
            //        writing the strip's own columns of the output in place
            //
            // Expanding the whole matrix would take 170 MiB at the Qwen3.8 FFN shape, which
            // is the Section 5 constraint kMaxStagingBytes exists to hold. Total dequantize
            // work is unchanged -- every code is still decoded exactly once -- and so is the
            // arithmetic: each output is one dot product over one strip, never a sum across
            // strips, so results are bit-identical to the unstriped pass. That is why the
            // split is along the output dimension and not along the contraction, where
            // partial sums would round to BF16 between strips.
            //
            // Unlike the W4A8-FP8 path, nothing here is sized by the plan's row bucket:
            // the staging buffer holds WEIGHTS, whose extent is independent of outer_size.
            // The out-of-bounds hazard that path documents does not arise.
            const std::size_t staging_bytes = static_cast<std::size_t>( strip_rows_ )
                * static_cast<std::size_t>( columns ) * sizeof( __nv_bfloat16 );

            // Fetched per forward, never cached: the scratch buffer may move on grow.
            auto* staging = static_cast<__nv_bfloat16*>(
                context_->getDeviceScratchBuffer( staging_bytes ) );

            const float alpha = 1.0f;
            const float beta = 0.0f;
            const int outer = narrowToKernelIndex( outer_size );

            for ( dim_t begin = 0; begin < weight_rows_; begin += strip_rows_ )
            {
                const dim_t count = std::min( strip_rows_, weight_rows_ - begin );
                const int strip_rows = narrowToKernelIndex( count );

                // The dequantize kernel addresses every plane relative to row 0 of the
                // pointers it is handed, so a strip needs no kernel change -- only the row
                // offset folded into each pointer and its own row count.
                const std::uint8_t* strip_two_bits = plane_two_bits_
                    + static_cast<std::size_t>( begin ) * ( columns / 4 );
                const __half* strip_scales = scales_
                    + static_cast<std::size_t>( begin ) * ( columns / kGroupSize );

                if constexpr ( kHasOneBitPlane )
                {
                    launch_codebook3_dequantize_to_bf16(
                        stream, staging, strip_two_bits,
                        plane_one_bit_ + static_cast<std::size_t>( begin ) * ( columns / 8 ),
                        strip_scales, codebook_, columns, strip_rows, kGroupSize );
                }
                else
                {
                    launch_codebook2_dequantize_to_bf16(
                        stream, staging, strip_two_bits,
                        strip_scales, codebook_, columns, strip_rows, kGroupSize );
                }

                const bool is_trailing = ( count != strip_rows_ );
                const auto& cache = is_trailing ? trailing_plan_cache_ : forward_plan_cache_;

                execute_linear_plan<TComputePrecision>(
                    cublaslt_handle_,
                    cache.get( outer ),
                    &alpha,
                    input_pointer,
                    staging,
                    &beta,
                    output_pointer + begin,
                    bias_pointer != nullptr ? bias_pointer + begin : nullptr,
                    nullptr,
                    stream,
                    context_->getCublasLtWorkspace(),
                    context_->getCublasLtWorkspaceSize() );
            }
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

        /// Output rows per prefill strip, and the width of the trailing one. Valid after build().
        dim_t getStripRows() const
        {
            return strip_rows_;
        }

        dim_t getTrailingStripRows() const
        {
            return trailing_strip_rows_;
        }

    private:
        /**
         * @brief Choose the strip width that holds staging under kMaxStagingBytes.
         *
         * Derives the strip count from the cap first and the width from the count, so the
         * strips come out balanced: at most two distinct widths, and the trailing one as
         * close to the others as alignment allows. Deriving the width first instead leaves a
         * trailing strip that can be a single row wide, which is a wasted cuBLASLt call.
         */
        void planStripWidths()
        {
            const std::size_t full_bytes = static_cast<std::size_t>( weight_rows_ )
                * static_cast<std::size_t>( weight_columns_ ) * sizeof( __nv_bfloat16 );

            const std::size_t strip_count =
                ( full_bytes + max_staging_bytes_ - 1 ) / max_staging_bytes_;

            if ( strip_count <= 1 )
            {
                strip_rows_ = weight_rows_;
                trailing_strip_rows_ = weight_rows_;
                return;
            }

            const dim_t even = ( weight_rows_ + static_cast<dim_t>( strip_count ) - 1 )
                / static_cast<dim_t>( strip_count );

            strip_rows_ = std::min( weight_rows_,
                ( ( even + kStripAlignment - 1 ) / kStripAlignment ) * kStripAlignment );

            const dim_t remainder = weight_rows_ % strip_rows_;
            trailing_strip_rows_ = ( remainder == 0 ) ? strip_rows_ : remainder;
        }

        CudaExecutionContext* context_{ nullptr };
        LinearConfig config_;
        std::size_t max_staging_bytes_{ kMaxStagingBytes };
        ITensor* bias_{ nullptr };

        // Borrowed device pointers into component-owned tensors. Never freed here.
        const std::uint8_t* plane_two_bits_{ nullptr };
        const std::uint8_t* plane_one_bit_{ nullptr };
        const __half* scales_{ nullptr };
        const float* codebook_{ nullptr };

        cublasLtHandle_t cublaslt_handle_{ nullptr };
        CublasLtPlanCache<CublasLtLinearPlan<TComputePrecision>> forward_plan_cache_;
        // Only populated when the row count is not a whole number of strips.
        CublasLtPlanCache<CublasLtLinearPlan<TComputePrecision>> trailing_plan_cache_;
        int built_outer_size_{ 0 };

        dim_t weight_rows_{ 0 };
        dim_t weight_columns_{ 0 };
        dim_t strip_rows_{ 0 };
        dim_t trailing_strip_rows_{ 0 };
    };
}
