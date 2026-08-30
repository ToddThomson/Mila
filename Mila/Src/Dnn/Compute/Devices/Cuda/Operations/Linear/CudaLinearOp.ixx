/**
 * @file CudaLinearOp.ixx
 * @brief CUDA Linear operation with compile-time weight quantization policy dispatch.
 *
 * One operation serves every weight format. The policy selects the storage type, the
 * scale type, the decode kernel and the prefill strategy at compile time; a member that
 * a format has no meaning for is constrained away rather than present and throwing.
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
#include <variant>
#include "Kernels/Linear.cuh"
#include "Kernels/Fp8Prefill/CudaFp8Prefill.cuh"
#include "Kernels/W8A16Gemm/CudaW8A16Gemm.cuh"
#include "Kernels/W4A16Gemm/CudaW4A16Gemm.cuh"
#include "Kernels/W4A16Gemm/CudaW4A16Gemm.Wmma.cuh"
#include "Kernels/Codebook/CodebookDequantize.cuh"
#include "Kernels/Codebook/CodebookGemv.cuh"

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
// MSVC WORKAROUND, not a design decision: a consumer instantiating this operation must
// complete ExecutionContext<Cuda>, and MSVC 14.51 demands the type be VISIBLE where the
// standard -- and Clang 19+ -- accept it being merely reachable. Restore the plain import
// when that is fixed; nothing else here depends on the export.
export import Compute.ExecutionContext;
import Compute.IExecutionContext;
import Compute.ExecutionContextTemplate;
import Compute.OperationType;
import Dnn.Component;
import Compute.MemoryResource;
import Compute.CudaDeviceMemoryResource;
import Compute.CudaDevice;
import Compute.CudaTensorDataType;
import Compute.CublasLtPlan;
import Compute.CublasLtPlanCache;
import Compute.Cuda.CublasLtLinearPlan;
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
     * BF16->FP8 conversion at load time.
     *
     * Forward dispatch on the quantized path:
     *   Single vector (outer_size == 1): fused matvec applies FP8 per-channel dequantization
     *     inline -- optimal for memory-bandwidth-bound single-vector compute.
     *   Batch (outer_size > 1): two paths selected by kUseW8A16Gemm:
     *     kUseW8A16Gemm=true  -- fused W8A16 GEMM reads FP8 once, dequantizes per-channel
     *       inline in shared memory, writes BF16 output directly (no staging buffer).
     *     kUseW8A16Gemm=false -- 2-phase: dequantize FP8 -> BF16 staging buffer, then
     *       standard BF16 cuBLASLt NT GEMM, then cuda_add_bias post-pass.
     *   PerGroupFp4 batches mirror the same structure via kUseFusedFp4Gemm (fused
     *   WMMA/tiled kernels vs the default 2-phase dequant-staging + cuBLASLt); FP4
     *   decode (outer_size == 1) stays on the dedicated fused matvec.
     *   PerGroupCodebook2/3 batches take the same 2-phase structure with the codebook
     *   expansion kernel, and decode goes to the codebook GEMV.
     *
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

        // Dispatch discriminators derived from the weight quantization policy.
        //   kIsPerChannelQuantized: FP8_E4M3 per-channel path  (PerChannelFp8, kPerChannel=true)
        //   kIsPerGroupQuantized:   INT4 per-group path         (PerGroupInt4,  kPerChannel=false)
        static constexpr bool kIsPerChannelQuantized = kIsQuantized && TWeightQuant::kPerChannel;
        static constexpr bool kIsPerGroupQuantized   = kIsQuantized && !TWeightQuant::kPerChannel;

        // The codes index a per-tensor table fitted offline rather than a step ladder, so
        // this policy family binds two tensors the others do not and never quantizes.
        // Detection is STRUCTURAL: the concepts ask the policy what it declares, so a
        // format defined outside the tree resolves here without this file naming it.
        static constexpr bool kIsCodebookWeight = HasCodebookTable<TWeightQuant>;
        static constexpr bool kHasHighBitPlane  = HasHighBitPlane<TWeightQuant>;

        // Toggle between the fused W8A16 GEMM and the baseline 2-phase path for A/B testing.
        //   true  -- cuda_w8a16_gemm: reads FP8 once, dequantizes inline, no staging buffer.
        //   false -- cuda_fp8_dequantize_to_bf16 -> cuBLASLt BF16 GEMM -> cuda_add_bias (proven path).
        static constexpr bool kUseW8A16Gemm = false;

        // Toggle between the fused FP4 GEMM kernels and the 2-phase dequant path for A/B testing.
        //   true  -- cuda_fp4a16_gemm_wmma / cuda_fp4a16_gemm: reads packed FP4 once, dequantizes
        //            inline per tile, no staging buffer. Stage 1 tiled rewrite (64x64/4-warp) was
        //            CORRECT but ~5-7x prefill regression vs 2-phase -- Long-Scoreboard bound
        //            (synchronous loads), ~6.8 TFLOP/s. Stage 2 (cp.async double-buffered software
        //            pipeline) IMPLEMENTED 2026-07-12: correct + chunk-INDEPENDENT (~510-527 tok/s
        //            flat vs 2-phase 574@48K/1032@24K), but still below cuBLASLt at every chunk
        //            (1.13x slower @48K, 1.96x @24K) -- cp.async killed the Stage 1 latency stall,
        //            the kernel is now compute-bound below cuBLASLt's BF16 GEMM. Needs the rest of
        //            the ladder (swizzle/ldmatrix/bigger tiles) to cross it; stays FALSE until then.
        //            See project_w4a16_prefill_gemm.
        //   false -- cuda_fp4_dequantize_to_bf16 -> cuBLASLt BF16 GEMM -> cuda_add_bias, the same
        //            2-phase structure as the proven FP8 baseline path (the "P0" prefill fix).
        static constexpr bool kUseFusedFp4Gemm = false;

        // Toggle the W4A8-FP8 prefill path for the FP4 weight policy. When true, batched
        // (prefill) forwards upcast the FP4 weights transiently to FP8_E4M3, quantize the
        // BF16 activations to FP8, and run a native FP8xFP8 cuBLASLt GEMM (~2x BF16 on Ada)
        // instead of the 2-phase FP4->BF16 staging + BF16 GEMM. Weights STAY FP4 in VRAM;
        // only the transient staging buffer is FP8 (half the bytes of the BF16 staging).
        // Decode (outer_size == 1) is untouched -- it stays on the FP4 matvec.
        // NUMERICS HISTORY: +98 shipped ON and generated incoherently; +99 reverted to OFF blaming
        // the per-TENSOR activation scale. The actual root cause (found 2026-07-13): the static
        // FP8 weight scale sB was computed at build() time from the FP4 group-scales buffer, which
        // loadParameter()/quantize() only fills LATER -- an uninitialized-memory read. sB cancels
        // in the GEMM (A_SCALE = sB, weights staged as W/sB), so benign junk generated correctly
        // and zeroed pages saturated every weight -- run-to-run luck, which is why +98 validated
        // coherent one day and produced garbage the next. sB is now computed in quantize() where
        // the group scales are produced. Activation scales are per-TOKEN (row absmax), applied
        // exactly by a post-GEMM epilogue because Ada cuBLASLt accepts only per-tensor scale
        // pointers (see Fp8ActivationPrefill.md section 5.2). Gate for keeping this ON: Linear FP8
        // oracle AND Gemma token-for-token parity vs the BF16 path AND a coherent chat -- a
        // per-layer oracle alone is NOT sufficient.
        static constexpr bool kUseFp8ActivationPrefill = true;

        static constexpr TensorDataType kWeightDtype = kIsQuantized
            ? TWeightQuant::kStorageDtype : TComputePrecision;

        // True only for the FP4 E2M1 weight policy -- guards TWeightQuant::kIsFp4E2M1,
        // which is not a member of NoWeightQuant / the FP8 per-channel policy.
        static constexpr bool kIsFp4Weight = []
        {
            if constexpr ( kIsPerGroupQuantized )
                return TWeightQuant::kIsFp4E2M1;
            else
                return false;
        }();

        // Compile-time predicate for the active W4A8-FP8 prefill path: FP4 weights AND the
        // toggle on. Gates the FP8 plan cache member type and every FP8-specific branch.
        static constexpr bool kUseFp8ActivationPrefillPath = kIsFp4Weight && kUseFp8ActivationPrefill;

        // Every batched path that expands packed weights into a BF16 staging buffer and
        // then runs one standard BF16 cuBLASLt GEMM. FP8 per-channel, FP4 E2M1 and the
        // codebook formats differ only in which expansion kernel fills the buffer, so they
        // share one implementation -- runStagedPrefill() -- and one plan-cache shape.
        static constexpr bool kUsesStagedPrefill =
            ( kIsPerChannelQuantized && !kUseW8A16Gemm )
            || kIsCodebookWeight
            || ( kIsFp4Weight && !kUseFp8ActivationPrefillPath && !kUseFusedFp4Gemm );

        using WeightType = typename TensorDataTypeMap<kWeightDtype>::device_type;

        // The scale element type is the policy's, not a fixed float: the codebook formats
        // carry IEEE half scales because at 2-3 bits the scale is a third of the payload.
        using ScaleType = typename TensorDataTypeMap<TWeightQuant::kScaleDtype>::device_type;

        /// Elements of the logical weight matrix packed into one byte of the primary tensor.
        /// One for every format at or above byte width, so it is a valid divisor everywhere.
        static constexpr int64_t kElementsPerStorageByte =
            TWeightQuant::kStorageBitsPerElement < 8 ? 8 / TWeightQuant::kStorageBitsPerElement : 1;

        /**
         * @brief Ceiling on the BF16 staging buffer, in bytes.
         *
         * Expanding a whole matrix costs out_features * in_features * 2 bytes, which at the
         * Qwen3.8 FFN shape is 170 MiB against the 461 MiB that Specifications/Qwen3.8.md
         * section 5 allots to all activations and scratch. Above this cap the prefill walks
         * the output channels in strips instead, which is bit-identical (see
         * runStagedPrefill) and costs about 1.2x at six strips.
         *
         * FP8 and FP4 were left UNBOUNDED on the reasoning that capping them was an
         * unmeasured VRAM-for-throughput trade. That reasoning held only because nobody had
         * run the one matrix big enough to break it. **The language-model head is that
         * matrix**: at Qwen 3.8's 248320 x 5120 an unstriped expansion is 1212.5 MiB of FP8
         * staging -- asked for whether the caller wants 8 output rows or 512 -- which does not
         * fit beside the model and aborts. It went unseen because prefill evaluates the head
         * at exactly one position, which takes the decode matvec and never reaches here;
         * teacher-forced scoring was the first caller to ask for more (Qwen3.8.md section 8).
         *
         * So every packed format is capped now. 256 MiB is chosen to be a no-op for every
         * shape already measured -- the 27B's feed-forward matrices are 178 MiB and still
         * expand in one pass, bit-identical to before -- while forcing the head to walk its
         * output channels in strips. A cap that changes no working shape and fixes the broken
         * one is not a trade.
         *
         * The budget is accounted in BF16 bytes whatever the staging dtype actually is, so
         * the FP8 paths come in at half their allowance. Conservative in the safe direction.
         */
        static constexpr std::size_t kMaxStagingBytes =
            kIsCodebookWeight ? 32ull * 1024 * 1024 : 256ull * 1024 * 1024;

        /// Strip widths are tensor-core tile aligned along N so cuBLASLt is not handed a ragged shape.
        static constexpr int kStripAlignment = 16;

        using CudaExecutionContext = ExecutionContext<DeviceType::Cuda>;

        /**
         * @param max_staging_bytes Ceiling on the prefill staging buffer. Defaults to
         *        kMaxStagingBytes. It is a constructor parameter rather than a constant
         *        because a real budget belongs to whoever owns the device's memory, not to
         *        this operation, and because it is the only way to exercise the striped path
         *        at a shape small enough for a test.
         */
        CudaLinearOp( IExecutionContext* context, const LinearConfig& config,
            std::size_t max_staging_bytes = kMaxStagingBytes )
            : context_( validateExecutionContext_<DeviceType::Cuda>( context, "CudaLinearOp" ) ),
              config_( config ),
              max_staging_bytes_( max_staging_bytes )
        {
            if ( max_staging_bytes_ == 0 )
            {
                throw std::invalid_argument( "CudaLinearOp - max_staging_bytes must be non-zero" );
            }

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

            // A packed weight tensor is allocated at its PHYSICAL extent {N, K / elements
            // per byte}, so the logical K has to be recovered before build() can validate
            // it against the input shape. The multiplier is the fact the policy states --
            // 2 for a nibble format, 4 for a 2-bit code plane -- and not an inference from
            // kPerChannel, which only ever meant "nibble-packed" because every per-group
            // policy happened to be 4-bit until the codebook formats arrived.
            if constexpr ( kIsQuantized && kElementsPerStorageByte > 1 )
                weight_in_features_ = weight_shape[ 1 ] * kElementsPerStorageByte;
            else
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
         * @brief Bind the weight scale tensor: produced by quantize(), or loaded with a
         * pre-quantized artifact.
         *
         * Must be bound before the first forward(). The element type and the extent are
         * the policy's: FP32 per output channel for PerChannelFp8, FP32 per group for the
         * FP4/INT4 formats, IEEE half per group for the codebook formats.
         *
         * @param scales Device tensor, dtype TWeightQuant::kScaleDtype.
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

            weight_scales_ = static_cast<const ScaleType*>(scales->rawData());
        }

        /**
         * @brief Bind the per-tensor codebook and, for a 3-bit format, the high-bit plane.
         *
         * Constrained rather than present-and-throwing: a format that decodes
         * arithmetically has no table, and calling this on one is a compile error.
         *
         * @param codebook  FP32, TWeightQuant::kCodebookEntries entries.
         * @param highPlane UINT8, [out, in/8]; null for a format with no high-bit plane.
         */
        void setCodebookTensors( ITensor* codebook, ITensor* highPlane )
            requires kIsCodebookWeight
        {
            if ( codebook == nullptr )
            {
                throw std::invalid_argument(
                    "CudaLinearOp::setCodebookTensors - codebook is required" );
            }

            if ( kHasHighBitPlane != (highPlane != nullptr) )
            {
                throw std::invalid_argument( std::format(
                    "CudaLinearOp::setCodebookTensors - {}-bit policy {} a high-bit plane",
                    TWeightQuant::kCodeBits, kHasHighBitPlane ? "requires" : "does not take" ) );
            }

            weight_codebook_ = static_cast<const float*>( codebook->rawData() );
            weight_high_plane_ = highPlane != nullptr
                ? static_cast<const std::uint8_t*>( highPlane->rawData() )
                : nullptr;
        }

        /**
         * @brief Bind the packed INT4 zero-point tensor for per-group asymmetric quantization.
         *
         * Optional -- only required for asymmetric INT4 quantization. Pass nullptr (or omit)
         * for symmetric quantization (implicit zero = 8). The tensor layout must match
         * the kernel expectation: [out_features, in_features / (group_size * 2)], dtype UINT8,
         * with two packed INT4 zero values per byte.
         *
         * @param zero_points Device UINT8 tensor, or nullptr for symmetric.
         */
        void setWeightZeroPoints( ITensor* zero_points ) requires kIsPerGroupQuantized
        {
            if ( zero_points == nullptr )
            {
                weight_zero_points_ = nullptr;
                return;
            }

            if ( zero_points->getDeviceType() != DeviceType::Cuda )
            {
                throw std::invalid_argument(
                    "CudaLinearOp::setWeightZeroPoints - zero_points must be a CUDA tensor" );
            }

            weight_zero_points_ = static_cast<const uint8_t*>(zero_points->rawData());
        }

        /**
         * @brief Recompute scales derived from the per-group scales, after a direct upload.
         *
         * A pre-quantized artifact carries the weights and their per-group scales, so
         * quantize() is skipped -- and with it the sB reduction below, which nothing else
         * performs. ensureFp8ScaleScalarsAllocated() only allocates weight_fp8_scale_; it
         * never writes it. Leaving it uninitialized reproduces the +98/+99 incident exactly:
         * the FP4->FP8 dequant divides by garbage and every activation becomes NaN.
         *
         * Must be called after BOTH the weights and the scales have landed.
         */
        void onQuantizedWeightsLoaded() requires kIsQuantized
        {
            if constexpr ( kUseFp8ActivationPrefillPath )
            {
                ensureFp8ScaleScalarsAllocated();

                const int64_t num_scales =
                    ( static_cast<int64_t>( out_features_ ) * static_cast<int64_t>( weight_in_features_ ) )
                    / TWeightQuant::kQuantizationGroupSize;

                cuda_compute_fp8_weight_scale(
                    weight_fp8_scale_, weight_scales_, num_scales, context_->getStream() );
            }
        }

        /**
         * @brief Quantize a compute-precision host blob into the policy's storage format.
         *
         * Runs once at model load time. Delegates to the Detail:: entries pre-compiled by
         * NVCC in the :Quantize partition, which do the absmax reduction and upload both
         * the packed weight tensor and its scales. The source blob is never retained on
         * device.
         *
         * Absent, not throwing, for a codebook policy. Its codes come from an offline GPTQ
         * fit against calibration data, so there is nothing to quantize on the way in;
         * Linear only reaches this when the source blob is at compute precision, and a
         * codebook artifact never is. Making it uncallable is what keeps that a compile
         * error instead of a runtime one.
         *
         * @param blob           Host weight blob from the model archive.
         * @param weight_out     Device tensor at TWeightQuant::kStorageDtype.
         * @param scales_out     Device scale tensor at TWeightQuant::kScaleDtype.
         * @param expected_shape Logical weight shape, for validation.
         */
        /**
         * @brief Ceiling on the BF16 staging buffer a quantize-on-load pass may take.
         *
         * The shared scratch is grow-on-demand and never shrinks, so whatever this pass asks
         * for is paid for the life of the process. A vocabulary-sized output axis makes the
         * whole-tensor request gigabytes; row blocks under this ceiling cost the same total
         * bandwidth and a bounded footprint. Mirrors CudaTokenEmbeddingOp's identical cap.
         */
        static constexpr size_t kQuantizeStagingLimitBytes = size_t{ 256 } * 1024 * 1024;

        void quantize(
            const ITensorBlob& blob,
            ITensor& weight_out,
            ITensor& scales_out,
            const shape_t& expected_shape ) requires ( kIsQuantized && !kIsCodebookWeight )
        {
            const int64_t out_features = static_cast<int64_t>( expected_shape[ 0 ] );
            const int64_t in_features  = static_cast<int64_t>( expected_shape[ 1 ] );
            const size_t  src_bytes    = static_cast<size_t>( out_features * in_features )
                                         * sizeof( __nv_bfloat16 );

            cudaStream_t stream = context_->getStream();

            if constexpr ( kIsPerChannelQuantized )
            {
                // FP8 per-channel: scale[o] = max(|W[o,:]|) / 448.0f
                // per_tensor needs 4 extra bytes for the atomicMax scratch -- allocate the
                // larger size so the same scratch buffer covers both variants.
                void* staging = context_->getDeviceScratchBuffer( src_bytes + sizeof( unsigned int ) );
                Detail::quantize_fp8_per_channel( blob, weight_out, scales_out, expected_shape,
                                                  staging, stream );
            }
            else if constexpr ( kIsPerGroupQuantized && TWeightQuant::kIsFp4E2M1 )
            {
                // FP4 E2M1 per-group: scale[n,g] = max(|W[n,g*gs..(g+1)*gs)|) / 6.0f
                //
                // Staged in row blocks under a ceiling rather than whole. The scratch buffer
                // is grow-only, so a request sized to the tensor is not a transient at all --
                // it raises steady-state VRAM for the life of the process. Qwen 3.8's lm_head
                // is 2.54 GiB of BF16 source, which put a 27B load 300 MiB over a 12 GiB card
                // and killed it. Same ceiling and same reasoning as the FP8 table path in
                // CudaTokenEmbeddingOp.
                const size_t staging_bytes = std::min( src_bytes, kQuantizeStagingLimitBytes );
                void* staging = context_->getDeviceScratchBuffer( staging_bytes );
                Detail::quantize_fp4_per_group(
                    blob, weight_out, scales_out, expected_shape,
                    TWeightQuant::kQuantizationGroupSize,
                    staging, staging_bytes, stream );

                if constexpr ( kUseFp8ActivationPrefillPath )
                {
                    // The static FP8 weight scale sB is a function of the group scales,
                    // so it is computed HERE, where those scales are produced -- NOT at
                    // build() time. build() runs before loadParameter() in every flow, so
                    // a build-time computation reads the pre-allocated but not-yet-filled
                    // scales buffer: uninitialized memory. sB cancels algebraically in
                    // the GEMM (A_SCALE = sB, weights staged as W/sB), which made that
                    // bug LUCK-DEPENDENT -- benign junk generated correctly, zeroed pages
                    // saturated every weight to +-448 and produced incoherent generation
                    // (the +98/+99 incident). Same stream as the scale upload above, so
                    // the reduction reads the freshly written values.
                    ensureFp8ScaleScalarsAllocated();

                    const int64_t num_scales = ( out_features * in_features )
                        / TWeightQuant::kQuantizationGroupSize;
                    cuda_compute_fp8_weight_scale(
                        weight_fp8_scale_,
                        static_cast<const float*>( scales_out.rawData() ),
                        num_scales, stream );
                }
            }
            else
            {
                // INT4 per-group: weights must come from a pre-quantized GPTQ checkpoint.
                // On-the-fly BF16->INT4 quantization is not supported here.
                static_assert( !sizeof( TWeightQuant ),
                    "CudaLinearOp::quantize() is not implemented for PerGroupInt4." );
            }
        }

        void setGradients( ITensor* weight_grad, ITensor* bias_grad ) override
        {
            // The remainder sits in the else branch so it is discarded rather than
            // merely unreachable in the quantized instantiation.
            if constexpr ( kIsQuantized )
            {
                throw std::logic_error( "CudaLinearOp: gradient computation is not supported on quantized paths" );
            }
            else
            {
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

            if constexpr ( kIsCodebookWeight )
            {
                if ( weight_scales_ == nullptr || weight_codebook_ == nullptr )
                {
                    throw std::runtime_error(
                        "CudaLinearOp::build - packed codes, scales and codebook must all be "
                        "bound before build()" );
                }
            }

            if ( input_shape.empty() )
            {
                throw std::invalid_argument( "CudaLinearOp::build - input shape cannot be empty" );
            }

            cached_in_features_ = static_cast<int>(input_shape.back());

            if constexpr ( kIsCodebookWeight )
            {
                // The GEMV loads codes as uint32 words and one scale per chunk of 16, so a
                // row that is not a whole number of both would read past its end on the
                // last word.
                if ( cached_in_features_ % 16 != 0
                    || cached_in_features_ % TWeightQuant::kQuantizationGroupSize != 0 )
                {
                    throw std::invalid_argument( std::format(
                        "CudaLinearOp::build - input features ({}) must be a multiple of 16 "
                        "and of the group size ({})",
                        cached_in_features_, TWeightQuant::kQuantizationGroupSize ) );
                }
            }

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

            // Capture the compile-time group size for the INT4 path so it is available
            // as a runtime int when calling cuda_w4a16_gemm.
            if constexpr ( kIsPerGroupQuantized )
            {
                weight_group_size_ = TWeightQuant::kQuantizationGroupSize;
            }

            if constexpr ( kUsesStagedPrefill || kUseFp8ActivationPrefillPath )
            {
                planStripWidths();
            }

            cached_cublaslt_handle_ = context_->getCublasLtHandle();
            use_cublaslt_ = (cached_cublaslt_handle_ != nullptr) && supportsCuBLASLt();

            if constexpr ( kIsPerGroupQuantized )
            {
                if constexpr ( TWeightQuant::kIsFp4E2M1 )
                {
                    int device = 0, major = 0;
                    cudaGetDevice( &device );
                    cudaDeviceGetAttribute( &major, cudaDevAttrComputeCapabilityMajor, device );
                    use_wmma_fp4_gemm_ = ( major >= 8 ); // BF16 tensor-core WMMA requires SM 8.0+
                }
            }

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
         *   1. outer_size == 1:
         *      FP8/non-quantized: fused matvec via cuda_matvec_impl.
         *      INT4: M=1 tiled W4A16 GEMM (no dedicated decode matvec yet).
         *   2. outer_size > 1, use_cublaslt_:
         *      kIsPerChannelQuantized: fused W8A16 GEMM -- reads FP8 weights once,
         *        dequantizes per-channel inline in shared memory, bias added in-kernel.
         *      kIsPerGroupQuantized:   fused W4A16 GEMM -- inline per-group INT4 dequant.
         *      !kIsQuantized:          NT row-major BF16 cuBLASLt GEMM; bias via epilogue.
         *   3. outer_size > 1, quantized, no cuBLASLt: per-row fallback loop (SM < 8.0
         *      SM < 8.0 or plan build failure).
         *   4. outer_size > 1, !kIsQuantized, no cuBLASLt: error -- non-quantized batch
         *      compute always requires cuBLASLt.
         */
        void forward( const TensorType& input, TensorType& output ) const
        {
            const int outer_size = narrowToKernelIndex( input.size() / cached_in_features_ );

            const ComputeType* input_ptr = static_cast<const ComputeType*>(input.rawData());
            ComputeType* output_ptr = static_cast<ComputeType*>(output.rawData());
            cudaStream_t stream = context_->getStream();

            if ( outer_size == 1 )
            {
                if constexpr ( kIsCodebookWeight )
                {
                    launchCodebookDecode( output_ptr, input_ptr, weight_, weight_scales_, stream );
                }
                else if constexpr ( kIsPerGroupQuantized )
                {
                    if constexpr ( TWeightQuant::kIsFp4E2M1 )
                    {
                        // Dedicated FP4 E2M1 decode matvec: all threads useful, warp shuffle
                        // reduction, one per-group scale per 8-element chunk. ~6x faster than
                        // the M=1 tiled GEMM for this path.
                        cuda_matvec_decode_bf16_qfp4(
                            output_ptr, input_ptr,
                            weight_, weight_scales_, bias_,
                            cached_in_features_, out_features_,
                            weight_group_size_, stream );
                    }
                    else
                    {
                        cuda_w4a16_gemm(
                            output_ptr, input_ptr,
                            weight_, weight_scales_, weight_zero_points_, bias_,
                            1, cached_in_features_, out_features_,
                            weight_group_size_, stream );
                    }
                }
                else
                {
                    // FP8 and non-quantized decode: fused matvec via cuda_matvec_impl.
                    // Handles BF16/FP32 weights and FP8_E4M3 weights transparently.
                    Detail::cuda_matvec_impl<ComputeType, WeightType>::decode(
                        output_ptr, input_ptr,
                        weight_, weight_scales_,
                        bias_,
                        cached_in_features_, out_features_,
                        stream );
                }

                return;
            }

            if ( use_cublaslt_ )
            {
                if constexpr ( kUsesStagedPrefill )
                {
                    runStagedPrefill( input_ptr, output_ptr, outer_size, stream );
                }
                else if constexpr ( kIsPerChannelQuantized )
                {
                    // Fused W8A16 single-kernel path: reads FP8 weights once from VRAM,
                    // dequantizes per-channel inline in shared memory, and accumulates
                    // directly into BF16 output. No staging buffer required.
                    cuda_w8a16_gemm(
                        output_ptr, input_ptr, weight_, weight_scales_, bias_,
                        outer_size, cached_in_features_, out_features_, stream );
                }
                else if constexpr ( kIsPerGroupQuantized )
                {
                    if constexpr ( kUseFp8ActivationPrefillPath )
                    {
                        // W4A8-FP8 prefill: transient FP4->FP8 weight upcast + dynamic per-token
                        // BF16->FP8 activation quantize, then a native FP8xFP8 cuBLASLt GEMM
                        // (~2x BF16 on Ada). Weights stay FP4 in VRAM; only this staging buffer
                        // is FP8. The GEMM runs with a unit activation scale (Ada cuBLASLt
                        // accepts only per-tensor scale pointers); the true per-token scales are
                        // applied exactly by the post-GEMM epilogue below, which also folds the
                        // bias. All three regions (FP8 weight, FP8 activation, per-token scales)
                        // share one scratch allocation, each 16-byte aligned for cuBLASLt.
                        // Fetched per-forward, never cached: the scratch buffer may be
                        // reallocated on grow.
                        // THE GEMM READS THE PLAN'S M, NOT outer_size. get() rounds up to a bucket,
                        // so a prompt whose length is not itself a bucket runs a kernel that reads
                        // (bucket - outer_size) * in_features bytes beyond what outer_size would
                        // stage. Sizing the staging region by outer_size therefore reads out of
                        // bounds -- silently while the grow-only scratch happens to have that much
                        // slack, and as cudaErrorIllegalAddress when it does not. Measured
                        // 2026-08-15: a 300-token prompt at prefill chunk 512 takes bucket 512 and
                        // overran by 212 * 3072 bytes. The rows between outer_size and the bucket
                        // are read as garbage and their outputs discarded, which is what the plan
                        // already assumed; only the allocation was short.
                        const int plan_rows =
                            fp8_forward_plan_cache_.bucketFor( static_cast<int>( outer_size ) );

                        // ONE STRIP of output channels, not the whole matrix. Unstriped, the
                        // head asks for 1212.5 MiB here regardless of how many rows the
                        // caller wants -- see kMaxStagingBytes.
                        const size_t weight_fp8_bytes = static_cast<size_t>( strip_rows_ )
                            * static_cast<size_t>( cached_in_features_ );
                        const size_t weight_fp8_bytes_aligned =
                            ( weight_fp8_bytes + 15u ) & ~static_cast<size_t>( 15u );
                        const size_t activation_fp8_bytes = static_cast<size_t>( plan_rows )
                            * static_cast<size_t>( cached_in_features_ );
                        const size_t activation_fp8_bytes_aligned =
                            ( activation_fp8_bytes + 15u ) & ~static_cast<size_t>( 15u );
                        const size_t token_scale_bytes = static_cast<size_t>( plan_rows )
                            * sizeof( float );

                        auto* scratch = static_cast<char*>( context_->getDeviceScratchBuffer(
                            weight_fp8_bytes_aligned + activation_fp8_bytes_aligned + token_scale_bytes ) );
                        auto* weight_fp8 = reinterpret_cast<__nv_fp8_e4m3*>( scratch );
                        auto* activation_fp8 = reinterpret_cast<__nv_fp8_e4m3*>( scratch + weight_fp8_bytes_aligned );
                        auto* activation_token_scales = reinterpret_cast<float*>(
                            scratch + weight_fp8_bytes_aligned + activation_fp8_bytes_aligned );

                        // Once for the pass: the activation quantization does not depend on
                        // which output channels are being computed.
                        cuda_quantize_bf16_to_fp8_per_token(
                            activation_fp8,
                            activation_token_scales,
                            input_ptr,
                            outer_size, cached_in_features_,
                            stream );

                        const float alpha = 1.0f;
                        const float beta  = 0.0f;

                        for ( int begin = 0; begin < out_features_; begin += strip_rows_ )
                        {
                            const int rows = std::min( strip_rows_, out_features_ - begin );

                            // The expansion kernel addresses its planes relative to row 0 of
                            // the pointers it is handed, so a strip needs only the row offset
                            // folded in and its own row count -- the same property
                            // dequantizeStrip relies on for the BF16 formats.
                            cuda_fp4_dequantize_to_fp8(
                                weight_fp8,
                                weight_ + static_cast<ptrdiff_t>( begin )
                                    * ( cached_in_features_ / kElementsPerStorageByte ),
                                weight_scales_ + static_cast<ptrdiff_t>( begin )
                                    * ( cached_in_features_ / weight_group_size_ ),
                                weight_fp8_scale_,
                                rows, cached_in_features_,
                                weight_group_size_,
                                stream );

                            const auto& cache = ( rows != strip_rows_ )
                                ? fp8_trailing_plan_cache_ : fp8_forward_plan_cache_;

                            // output_ptr + begin with the plan's ldc set to out_features_:
                            // column-major C with a leading dimension writes this strip's
                            // channels into their columns of the full row-major output.
                            execute_fp8_prefill_plan<TComputePrecision>(
                                cached_cublaslt_handle_,
                                cache.get( outer_size ),
                                &alpha,
                                weight_fp8,
                                activation_fp8,
                                &beta,
                                output_ptr + begin,
                                stream,
                                context_->getCublasLtWorkspace(),
                                context_->getCublasLtWorkspaceSize() );
                        }

                        // After every strip: the per-token scales and the bias apply to the
                        // whole output row, not to one strip of it.
                        cuda_fp8_apply_per_token_scales(
                            output_ptr,
                            activation_token_scales,
                            bias_,
                            outer_size, out_features_,
                            stream );
                    }
                    else if constexpr ( TWeightQuant::kIsFp4E2M1 )
                    {
                        if ( use_wmma_fp4_gemm_ )
                        {
                            cuda_fp4a16_gemm_wmma(
                                output_ptr, input_ptr, weight_, weight_scales_, bias_,
                                outer_size, cached_in_features_, out_features_,
                                weight_group_size_, stream );
                        }
                        else
                        {
                            cuda_fp4a16_gemm(
                                output_ptr, input_ptr, weight_, weight_scales_, bias_,
                                outer_size, cached_in_features_, out_features_,
                                weight_group_size_, stream );
                        }
                    }
                    else
                    {
                        // INT4 W4A16 fused GEMM: packed INT4 weights dequantized per-group
                        // inline. Optional asymmetric zero-points via weight_zero_points_.
                        cuda_w4a16_gemm(
                            output_ptr,
                            input_ptr,
                            weight_,
                            weight_scales_,
                            weight_zero_points_,
                            bias_,
                            outer_size, cached_in_features_, out_features_,
                            weight_group_size_,
                            stream );
                    }
                }
                else
                {
                    const float alpha = 1.0f;
                    const float beta  = 0.0f;

                    // Bias intentionally omitted from the plan (built has_bias=false) and
                    // added post-GEMM below -- see buildCublasLtPlans for why the FP32
                    // bias epilogue is unsupported.
                    execute_linear_plan<TComputePrecision>(
                        cached_cublaslt_handle_,
                        forward_plan_cache_.get( outer_size ),
                        &alpha,
                        input_ptr,
                        weight_,
                        &beta,
                        output_ptr,
                        nullptr,
                        nullptr,
                        stream,
                        context_->getCublasLtWorkspace(),
                        context_->getCublasLtWorkspaceSize() );

                    if ( bias_ != nullptr )
                    {
                        cuda_add_bias( output_ptr, bias_, outer_size, out_features_, stream );
                    }
                }

                return;
            }

            // No cuBLASLt plan -- fallback paths, every one of them the decode kernel driven
            // a row at a time.
            // FP8:      the FP8 decode matvec.
            // FP4:      the FP4 decode matvec.
            // Codebook: the codebook GEMV.
            // INT4:     the M=1 tiled W4A16 GEMM -- less optimal than a dedicated matvec.
            // Non-quantized: no fallback for batch compute.
            if constexpr ( kIsPerChannelQuantized )
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
            else if constexpr ( kIsPerGroupQuantized )
            {
                // Fallback: drive the matvec / tiled GEMM one row at a time.
                for ( int t = 0; t < outer_size; ++t )
                {
                    const auto* in_row  = input_ptr  + static_cast<ptrdiff_t>(t) * cached_in_features_;
                    auto*       out_row = output_ptr + static_cast<ptrdiff_t>(t) * out_features_;

                    if constexpr ( kIsCodebookWeight )
                    {
                        launchCodebookDecode( out_row, in_row, weight_, weight_scales_, stream );
                    }
                    else if constexpr ( TWeightQuant::kIsFp4E2M1 )
                    {
                        cuda_matvec_decode_bf16_qfp4(
                            out_row, in_row,
                            weight_, weight_scales_, bias_,
                            cached_in_features_, out_features_,
                            weight_group_size_, stream );
                    }
                    else
                    {
                        cuda_w4a16_gemm(
                            out_row, in_row,
                            weight_, weight_scales_, weight_zero_points_, bias_,
                            1, cached_in_features_, out_features_,
                            weight_group_size_, stream );
                    }
                }

                return;
            }
            else
            {
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

        ~CudaLinearOp()
        {
            // Persistent FP8 scale scalars owned by the op (W4A8-FP8 path only; nullptr otherwise).
            if ( activation_fp8_unit_scale_ != nullptr ) cudaFree( activation_fp8_unit_scale_ );
            if ( weight_fp8_scale_ != nullptr ) cudaFree( weight_fp8_scale_ );
        }

        OperationType getOperationType() const override
        {
            return OperationType::LinearOp;
        }

        std::string getName() const override
        {
            if constexpr ( kIsCodebookWeight )
                return std::format( "Cuda::LinearOp<W{}A16,g{}>",
                    TWeightQuant::kCodeBits, TWeightQuant::kQuantizationGroupSize );
            else
                return "Cuda::LinearOp";
        }

        const LinearConfig& getConfig() const
        {
            return config_;
        }

        /// Output rows per prefill strip, and the width of the trailing one. Valid after build().
        int getStripRows() const
        {
            return strip_rows_;
        }

        int getTrailingStripRows() const
        {
            return trailing_strip_rows_;
        }

    private:

        LinearConfig config_;
        CudaExecutionContext* context_;
        std::size_t max_staging_bytes_{ kMaxStagingBytes };

        // Weight pointer typed to WeightType -- differs from ComputeType on the FP8 path.
        const WeightType* weight_{ nullptr };

        // Scales at the policy's element type. Per-channel FP32 [out_features] on the FP8
        // path; per-group [out_features x in_features/group_size] on every per-group path,
        // FP32 for FP4/INT4 and IEEE half for the codebook formats.
        const ScaleType* weight_scales_{ nullptr };

        // Codebook formats only. Borrowed exactly as the weight is: the component owns the
        // table and the high-bit plane, this operation caches pointers and frees nothing.
        const float* weight_codebook_{ nullptr };
        const std::uint8_t* weight_high_plane_{ nullptr };

        // Packed INT4 zero points [out_features x in_features/(group_size*2)] -- kIsPerGroupQuantized path only.
        // nullptr when symmetric quantization is used (implicit zero = 8).
        const uint8_t* weight_zero_points_{ nullptr };

        // INT4 quantization group size along K -- set from TWeightQuant::kQuantizationGroupSize at build.
        int weight_group_size_{ 128 };


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
        bool use_wmma_fp4_gemm_{ false };
        
        // cuBLASLt plan cache -- forward path.
        // kUsesStagedPrefill: BF16xBF16 NT plan fed by the staging buffer, sized to one strip.
        // Non-quantized: BF16xBF16 (or FP32xFP32) plan fed directly by the weight tensor.
        CublasLtPlanCache<CublasLtLinearPlan<TComputePrecision>> forward_plan_cache_;

        // Only populated when the row count is not a whole number of strips.
        CublasLtPlanCache<CublasLtLinearPlan<TComputePrecision>> trailing_plan_cache_;

        // Staged prefill strip geometry, planned at build(). Both equal out_features_ when
        // the whole matrix fits under the cap, which is the unstriped single-pass case.
        int strip_rows_{ 0 };
        int trailing_strip_rows_{ 0 };

        // W4A8-FP8 prefill path only. Persistent device scalars baked into the FP8 plan's
        // A_SCALE/B_SCALE, allocated by ensureFp8ScaleScalarsAllocated() and freed in the
        // destructor. The weight scale VALUE is written by quantize() at loadParameter()
        // time, where the FP4 group scales it derives from are produced (build() runs
        // before the scales exist). The activation scale is a constant 1.0f: Ada cuBLASLt
        // accepts only per-tensor scale pointers, so the true per-token activation scales
        // live in scratch and are applied post-GEMM (exact factorization).
        float* activation_fp8_unit_scale_{ nullptr };
        float* weight_fp8_scale_{ nullptr };

        // FP8xFP8 forward plan cache -- present only on the active W4A8-FP8 path (std::monostate
        // otherwise, so no CublasLtLinearPlan<*, FP8_E4M3> is instantiated for other policies).
        using Fp8LinearPlan = CublasLtLinearPlan<TComputePrecision, TensorDataType::FP8_E4M3>;
        std::conditional_t<kUseFp8ActivationPrefillPath,
            CublasLtPlanCache<Fp8LinearPlan>, std::monostate> fp8_forward_plan_cache_;

        // The narrower last strip, when out_features_ is not a whole number of them. Mirrors
        // trailing_plan_cache_ on the BF16-staged path; a copy of the forward cache when the
        // strips come out even, so the forward loop can select without a special case.
        std::conditional_t<kUseFp8ActivationPrefillPath,
            CublasLtPlanCache<Fp8LinearPlan>, std::monostate> fp8_trailing_plan_cache_;

        CublasLtPlanCache<CublasLtMatMulPlan<ComputeType>> backward_input_plan_cache_;
        CublasLtMatMulPlan<ComputeType> backward_weight_plan_;

        cudaDataType_t cuda_data_type_{};
        cudaDataType_t cuda_weight_data_type_{};
        cublasComputeType_t compute_type_{};
        cudaDataType_t scale_type_{};

        /**
         * @brief Choose the strip width that holds staging under max_staging_bytes_.
         *
         * Derives the strip count from the cap first and the width from the count, so the
         * strips come out balanced: at most two distinct widths, and the trailing one as
         * close to the others as alignment allows. Deriving the width first instead leaves a
         * trailing strip that can be a single row wide, which is a wasted cuBLASLt call.
         *
         * A cap that already covers the whole matrix yields one strip, which is the
         * unstriped single-pass prefill and the default for every format but the codebook.
         */
        void planStripWidths()
        {
            const std::size_t full_bytes = static_cast<std::size_t>( out_features_ )
                * static_cast<std::size_t>( cached_in_features_ ) * sizeof( __nv_bfloat16 );

            if ( max_staging_bytes_ >= full_bytes )
            {
                strip_rows_ = out_features_;
                trailing_strip_rows_ = out_features_;
                return;
            }

            const std::size_t strip_count =
                ( full_bytes + max_staging_bytes_ - 1 ) / max_staging_bytes_;

            const int even = ( out_features_ + static_cast<int>( strip_count ) - 1 )
                / static_cast<int>( strip_count );

            strip_rows_ = std::min( out_features_,
                ( ( even + kStripAlignment - 1 ) / kStripAlignment ) * kStripAlignment );

            const int remainder = out_features_ % strip_rows_;
            trailing_strip_rows_ = ( remainder == 0 ) ? strip_rows_ : remainder;
        }

        /**
         * @brief Expand one strip of output channels into the BF16 staging buffer.
         *
         * Every expansion kernel addresses its planes relative to row 0 of the pointers it
         * is handed, so a strip needs no kernel change -- only the row offset folded into
         * each pointer and its own row count. The policy picks the kernel; the striping
         * around it is the same for all three formats.
         */
        void dequantizeStrip( __nv_bfloat16* staging, int begin, int rows, cudaStream_t stream ) const
        {
            const auto* strip_weight = weight_ + static_cast<ptrdiff_t>( begin )
                * ( cached_in_features_ / kElementsPerStorageByte );

            if constexpr ( kIsCodebookWeight )
            {
                const auto* strip_scales = weight_scales_ + static_cast<ptrdiff_t>( begin )
                    * ( cached_in_features_ / weight_group_size_ );

                if constexpr ( kHasHighBitPlane )
                {
                    launch_codebook3_dequantize_to_bf16(
                        stream, staging, strip_weight,
                        weight_high_plane_ + static_cast<ptrdiff_t>( begin ) * ( cached_in_features_ / 8 ),
                        strip_scales, weight_codebook_,
                        cached_in_features_, rows, weight_group_size_ );
                }
                else
                {
                    launch_codebook2_dequantize_to_bf16(
                        stream, staging, strip_weight,
                        strip_scales, weight_codebook_,
                        cached_in_features_, rows, weight_group_size_ );
                }
            }
            else if constexpr ( kIsPerChannelQuantized )
            {
                // One scale per output channel, so the strip offset is the channel index.
                cuda_fp8_dequantize_to_bf16(
                    staging, strip_weight, weight_scales_ + begin,
                    rows, cached_in_features_, stream );
            }
            else
            {
                cuda_fp4_dequantize_to_bf16(
                    staging, strip_weight,
                    weight_scales_ + static_cast<ptrdiff_t>( begin )
                        * ( cached_in_features_ / weight_group_size_ ),
                    rows, cached_in_features_, weight_group_size_, stream );
            }
        }

        /**
         * @brief Two-phase batched forward, shared by every packed weight format.
         *
         *   for each strip of output rows:
         *     1. expand that strip's packed weights into the shared BF16 staging buffer
         *     2. standard BF16 cuBLASLt NT row-major GEMM, writing the strip's own columns
         *        of the output in place
         *   then add bias once over the whole output.
         *
         * The fused kernels this replaces are compute-bound well below cuBLASLt's BF16
         * GEMM at prefill-shaped M, so the staging traffic -- write and read one BF16 weight
         * matrix per forward -- is the cheaper side of the trade.
         *
         * Striping is along the OUTPUT dimension, never the contraction: each output is one
         * dot product over one strip and never a sum across strips, so the result is
         * bit-identical to the unstriped pass. Splitting along K would need beta=1
         * accumulation, which rounds partial sums to BF16 between strips.
         *
         * The staging buffer is fetched from the context on every forward rather than cached
         * at build time. getDeviceScratchBuffer() is O(1) when the buffer is already large
         * enough, and the buffer grows as later layers build their plans, so any pointer
         * captured at build time may point at freed memory by the time forward() runs.
         *
         * Nothing here is sized by the plan's row bucket: the staging buffer holds WEIGHTS,
         * whose extent is independent of outer_size. The out-of-bounds hazard the W4A8-FP8
         * path documents does not arise.
         */
        void runStagedPrefill( const ComputeType* input_ptr, ComputeType* output_ptr,
            int outer_size, cudaStream_t stream ) const
        {
            const size_t staging_bytes = static_cast<size_t>( strip_rows_ )
                * static_cast<size_t>( cached_in_features_ )
                * sizeof( __nv_bfloat16 );

            auto* staging = static_cast<__nv_bfloat16*>(
                context_->getDeviceScratchBuffer( staging_bytes ) );

            const float alpha = 1.0f;
            const float beta  = 0.0f;

            for ( int begin = 0; begin < out_features_; begin += strip_rows_ )
            {
                const int rows = std::min( strip_rows_, out_features_ - begin );

                dequantizeStrip( staging, begin, rows, stream );

                const auto& cache =
                    ( rows != strip_rows_ ) ? trailing_plan_cache_ : forward_plan_cache_;

                execute_linear_plan<TComputePrecision>(
                    cached_cublaslt_handle_,
                    cache.get( outer_size ),
                    &alpha,
                    input_ptr,
                    staging,
                    &beta,
                    output_ptr + begin,
                    nullptr,
                    nullptr,
                    stream,
                    context_->getCublasLtWorkspace(),
                    context_->getCublasLtWorkspaceSize() );
            }

            if ( bias_ != nullptr )
            {
                cuda_add_bias( output_ptr, bias_, outer_size, out_features_, stream );
            }
        }

        /**
         * @brief Decode (outer_size == 1) through the codebook GEMV.
         *
         * Kept a separate entry because the fallback loop drives it per row as well.
         */
        void launchCodebookDecode( ComputeType* output_row, const ComputeType* input_row,
            const WeightType* codes, const ScaleType* scales, cudaStream_t stream ) const
        {
            static_assert( TComputePrecision == TensorDataType::BF16,
                "The codebook GEMV kernels are BF16 in, BF16 out" );

            if constexpr ( kHasHighBitPlane )
            {
                launch_codebook3_matvec_decode(
                    stream, output_row, input_row, codes, weight_high_plane_,
                    scales, weight_codebook_, bias_,
                    cached_in_features_, out_features_, weight_group_size_ );
            }
            else
            {
                launch_codebook2_matvec_decode(
                    stream, output_row, input_row, codes, scales, weight_codebook_, bias_,
                    cached_in_features_, out_features_, weight_group_size_ );
            }
        }

        /**
         * @brief Returns true if an optimised batch compute path is available.
         *
         * FP8 (kIsPerChannelQuantized): SM >= 8.0 (Ampere+) for both the fused W8A16 GEMM
         *   and the 2-phase dequant + cuBLASLt BF16 GEMM baseline.
         * INT4 (kIsPerGroupQuantized):  SM >= 8.0 (Ampere+) for BF16. The fused
         *   W4A16 kernel reads packed INT4 and dequantizes per-group inline.
         * Non-quantized: requires a cuBLASLt-supported compute type (FP32/FP16/BF16).
         */
        bool supportsCuBLASLt() const
        {
            if constexpr ( kIsQuantized )
            {
                int device = 0, major = 0;
                cudaGetDevice( &device );
                cudaDeviceGetAttribute( &major, cudaDevAttrComputeCapabilityMajor, device );
                return (major >= 8);
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

            // The quantized dequant prefill path executes a standard BF16 cuBLASLt GEMM
            // (FP8 weights are expanded to BF16 before the GEMM), so it uses the same
            // compute type as the non-quantized path.  The dead TN FP8xBF16 branch
            // that used CUBLAS_COMPUTE_32F has been removed; all active paths go here.

            // REVIEW: we need only support bf16 for CUDA

            if constexpr ( std::is_same_v<ComputeType, half> )
                compute_type = CUBLAS_COMPUTE_32F_FAST_16F;
            else if constexpr ( std::is_same_v<ComputeType, nv_bfloat16> )
                compute_type = CUBLAS_COMPUTE_32F_FAST_16BF;
            else
                compute_type = CUBLAS_COMPUTE_32F;
        }

        /**
         * @brief Allocate the two persistent FP8 scale scalars on first use (idempotent).
         *
         * Called from both quantize() (which writes the sB value into weight_fp8_scale_)
         * and buildCublasLtPlans() (which bakes both pointers into the FP8 plan
         * descriptor). Neither caller may assume the other ran first: build() precedes
         * loadParameter() in the model flows, but the scalars must exist wherever the
         * first of the two arrives. The unit activation scale is a constant, so its
         * value is final at allocation.
         */
        void ensureFp8ScaleScalarsAllocated() requires kUseFp8ActivationPrefillPath
        {
            if ( weight_fp8_scale_ != nullptr )
                return;

            cudaMalloc( reinterpret_cast<void**>( &activation_fp8_unit_scale_ ), sizeof( float ) );
            cudaMalloc( reinterpret_cast<void**>( &weight_fp8_scale_ ), sizeof( float ) );

            const float unit_scale = 1.0f;
            cudaMemcpy( activation_fp8_unit_scale_, &unit_scale, sizeof( float ),
                cudaMemcpyHostToDevice );
        }

        /**
         * @brief Build the BF16 plan cache(s) that runStagedPrefill() executes.
         *
         * Each strip's C is a column slice of the full output, so every plan carries the
         * full row stride and writes its own channels in place. The trailing strip is
         * narrower whenever the row count is not a whole number of strips and gets its own
         * cache; nothing else about the two differs. Unstriped, there is one cache built at
         * the full row count, which is exactly the plan shape this path always had.
         *
         * has_bias=false: bias is applied post-GEMM by cuda_add_bias. cuBLASLt's heuristic
         * returns CUBLAS_STATUS_NOT_SUPPORTED for CUBLAS_COMPUTE_32F with
         * CUBLASLT_EPILOGUE_BIAS, and the bias epilogue carries the Ada multi-row
         * INVALID_VALUE constraint.
         */
        void buildStagedPrefillPlans()
        {
            const auto makeCache = [&]( int strip_rows )
            {
                return CublasLtPlanCache<CublasLtLinearPlan<TComputePrecision>>(
                    cached_outer_size_,
                    [&]( int bucket )
                    {
                        return build_linear_plan<TComputePrecision>(
                            cached_cublaslt_handle_,
                            bucket,
                            cached_in_features_,
                            strip_rows,
                            false,
                            compute_type_,
                            scale_type_,
                            nullptr,
                            out_features_ );
                    } );
            };

            forward_plan_cache_ = makeCache( strip_rows_ );

            if ( trailing_strip_rows_ != strip_rows_ )
            {
                trailing_plan_cache_ = makeCache( trailing_strip_rows_ );
            }
        }

        void buildCublasLtPlans()
        {
            cuda_data_type_ = getActivationCudaDataType();
            cuda_weight_data_type_ = getWeightCudaDataType();

            getComputeTypes( compute_type_, scale_type_ );

            if constexpr ( kUsesStagedPrefill )
            {
                buildStagedPrefillPlans();

                Logging::Logger::info( std::format(
                    "CudaLinearOp: staged dequant + BF16 cuBLASLt GEMM -- {} in -> {} out "
                    "({} rows per strip)",
                    cached_in_features_, out_features_, strip_rows_ ) );

                return;
            }

            if constexpr ( kIsPerChannelQuantized )
            {
                // W8A16 fused path: no staging buffer or cuBLASLt plan needed.
                // cuda_w8a16_gemm reads FP8 weights once and dequantizes per-channel
                // inline in shared memory -- SM >= 8.0 guaranteed by supportsCuBLASLt().
                Logging::Logger::info( std::format(
                    "CudaLinearOp: W8A16 fused GEMM ready -- {} in -> {} out",
                    cached_in_features_, out_features_ ) );

                return;
            }

            if constexpr ( kIsPerGroupQuantized )
            {
                if constexpr ( kUseFp8ActivationPrefillPath )
                {
                    // W4A8-FP8 prefill path. The persistent scale scalars must exist here so
                    // their pointers can be baked into the FP8 plan descriptor below -- but
                    // ONLY the pointers. The weight scale VALUE is written by quantize() at
                    // loadParameter() time, where the FP4 group scales it derives from are
                    // produced; computing it here read the pre-allocated, not-yet-filled
                    // scales buffer (the +98/+99 incoherence root cause). The activation
                    // B_SCALE is a constant 1.0f: the true per-token activation scales are
                    // applied post-GEMM (Ada cuBLASLt accepts only per-tensor scale pointers).
                    ensureFp8ScaleScalarsAllocated();

                    // One plan per strip width. The output_row_stride argument is what makes a
                    // strip addressable: without it every plan would write its channels to
                    // column 0 of the output.
                    auto makeFp8Cache = [&]( int strip_rows )
                        {
                            return CublasLtPlanCache<Fp8LinearPlan>(
                                cached_outer_size_,
                                [&, strip_rows]( int bucket )
                                {
                                    return build_fp8_prefill_plan<TComputePrecision>(
                                        cached_cublaslt_handle_,
                                        bucket,
                                        cached_in_features_,
                                        strip_rows,
                                        activation_fp8_unit_scale_,
                                        weight_fp8_scale_,
                                        out_features_ );
                                } );
                        };

                    fp8_forward_plan_cache_ = makeFp8Cache( strip_rows_ );
                    fp8_trailing_plan_cache_ = ( trailing_strip_rows_ != strip_rows_ )
                        ? makeFp8Cache( trailing_strip_rows_ )
                        : makeFp8Cache( strip_rows_ );

                    Logging::Logger::info( std::format(
                        "CudaLinearOp: FP4->FP8 upcast + FP8xFP8 cuBLASLt GEMM (W4A8) -- {} in -> {} out "
                        "(group_size={}, strip={}{})",
                        cached_in_features_, out_features_, weight_group_size_, strip_rows_,
                        ( strip_rows_ == out_features_ ) ? ", single pass" : "" ) );

                    return;
                }

                // Fused W4A16 / FP4 batch path: the fused kernels read packed weights and
                // dequantize per-group inline -- no staging buffer or cuBLASLt plan needed.
                // SM >= 8.0 is already guaranteed by supportsCuBLASLt() gating this path.
                return;
            }

            // has_bias=false: bias is added post-GEMM by cuda_add_bias. cuBLASLt's
            // heuristic returns CUBLAS_STATUS_NOT_SUPPORTED for CUBLAS_COMPUTE_32F with
            // CUBLASLT_EPILOGUE_BIAS (and the bias epilogue carries the Ada multi-row
            // INVALID_VALUE constraint), so the non-quantized path mirrors the FP8 path
            // and keeps bias out of the plan. GPT-2 (biased Linears) is the first model
            // to exercise this; bias-free models (MNIST, Llama) never hit the epilogue.
            forward_plan_cache_ = CublasLtPlanCache<CublasLtLinearPlan<TComputePrecision>>(
                cached_outer_size_,
                [&]( int bucket )
                {
                    return build_linear_plan<TComputePrecision>(
                        cached_cublaslt_handle_,
                        bucket,
                        cached_in_features_,
                        out_features_,
                        false,
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

}
