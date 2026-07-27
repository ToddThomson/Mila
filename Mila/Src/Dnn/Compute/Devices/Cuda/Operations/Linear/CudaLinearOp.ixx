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
#include <variant>
#include "Kernels/Linear.cuh"
#include "Kernels/Fp8Prefill/CudaFp8Prefill.cuh"
#include "Kernels/W8A16Gemm/CudaW8A16Gemm.cuh"
#include "Kernels/W4A16Gemm/CudaW4A16Gemm.cuh"
#include "Kernels/W4A16Gemm/CudaW4A16Gemm.Wmma.cuh"

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

            // Per-group quantized weights (INT4, FP4) are packed: 2 nibbles per byte.
            // initializeParameters() allocates shape {N, K/2} for these types, so
            // weight_shape[1] == K/2. Multiply by 2 to recover the logical K so that
            // build()'s validation against cached_in_features_ (= logical K) passes.
            if constexpr ( kIsPerGroupQuantized )
                weight_in_features_ = weight_shape[ 1 ] * 2;
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
                void* staging = context_->getDeviceScratchBuffer( src_bytes );
                Detail::quantize_fp4_per_group(
                    blob, weight_out, scales_out, expected_shape,
                    TWeightQuant::kQuantizationGroupSize,
                    staging, stream );

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

            // Capture the compile-time group size for the INT4 path so it is available
            // as a runtime int when calling cuda_w4a16_gemm.
            if constexpr ( kIsPerGroupQuantized )
            {
                weight_group_size_ = TWeightQuant::kQuantizationGroupSize;
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
                if constexpr ( kIsPerGroupQuantized )
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
                if constexpr ( kIsPerChannelQuantized )
                {
                    if constexpr ( kUseW8A16Gemm )
                    {
                        // Fused W8A16 single-kernel path: reads FP8 weights once from VRAM,
                        // dequantizes per-channel inline in shared memory, and accumulates
                        // directly into BF16 output. No staging buffer required.
                        cuda_w8a16_gemm(
                            output_ptr, input_ptr, weight_, weight_scales_, bias_,
                            outer_size, cached_in_features_, out_features_, stream );
                    }
                    else
                    {
                        // 2-phase baseline path:
                        //   Phase 1 -- dequantize FP8 weights to the shared BF16 staging buffer.
                        //   Phase 2 -- standard BF16 cuBLASLt NT row-major GEMM using the staging buffer.
                        //   Phase 3 -- add bias post-GEMM (plan built with has_bias=false to avoid
                        //              the Ada multi-row epilogue INVALID_VALUE constraint).
                        //
                        // The staging buffer is fetched from the context on every forward call rather
                        // than cached at build time. getDeviceScratchBuffer() is O(1) when the buffer
                        // is already large enough. Fetching here avoids stale pointers: the buffer
                        // grows during buildCublasLtPlans() as larger layers build their plans, so
                        // any pointer captured at build time may point to freed memory by the time
                        // forward() is called.
                        const size_t staging_bytes = static_cast<size_t>( out_features_ )
                            * static_cast<size_t>( cached_in_features_ )
                            * sizeof( __nv_bfloat16 );

                        auto* staging = static_cast<__nv_bfloat16*>(
                            context_->getDeviceScratchBuffer( staging_bytes ) );

                        cuda_fp8_dequantize_to_bf16(
                            staging,
                            weight_,
                            weight_scales_,
                            out_features_, cached_in_features_,
                            stream );

                        const float alpha = 1.0f;
                        const float beta  = 0.0f;

                        execute_linear_plan<TComputePrecision>(
                            cached_cublaslt_handle_,
                            forward_plan_cache_.get( outer_size ),
                            &alpha,
                            input_ptr,
                            staging,
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
                        const size_t weight_fp8_bytes = static_cast<size_t>( out_features_ )
                            * static_cast<size_t>( cached_in_features_ );
                        const size_t weight_fp8_bytes_aligned =
                            ( weight_fp8_bytes + 15u ) & ~static_cast<size_t>( 15u );
                        const size_t activation_fp8_bytes = static_cast<size_t>( outer_size )
                            * static_cast<size_t>( cached_in_features_ );
                        const size_t activation_fp8_bytes_aligned =
                            ( activation_fp8_bytes + 15u ) & ~static_cast<size_t>( 15u );
                        const size_t token_scale_bytes = static_cast<size_t>( outer_size )
                            * sizeof( float );

                        auto* scratch = static_cast<char*>( context_->getDeviceScratchBuffer(
                            weight_fp8_bytes_aligned + activation_fp8_bytes_aligned + token_scale_bytes ) );
                        auto* weight_fp8 = reinterpret_cast<__nv_fp8_e4m3*>( scratch );
                        auto* activation_fp8 = reinterpret_cast<__nv_fp8_e4m3*>( scratch + weight_fp8_bytes_aligned );
                        auto* activation_token_scales = reinterpret_cast<float*>(
                            scratch + weight_fp8_bytes_aligned + activation_fp8_bytes_aligned );

                        cuda_fp4_dequantize_to_fp8(
                            weight_fp8,
                            weight_,
                            weight_scales_,
                            weight_fp8_scale_,
                            out_features_, cached_in_features_,
                            weight_group_size_,
                            stream );

                        cuda_quantize_bf16_to_fp8_per_token(
                            activation_fp8,
                            activation_token_scales,
                            input_ptr,
                            outer_size, cached_in_features_,
                            stream );

                        const float alpha = 1.0f;
                        const float beta  = 0.0f;

                        execute_fp8_prefill_plan<TComputePrecision>(
                            cached_cublaslt_handle_,
                            fp8_forward_plan_cache_.get( outer_size ),
                            &alpha,
                            weight_fp8,
                            activation_fp8,
                            &beta,
                            output_ptr,
                            stream,
                            context_->getCublasLtWorkspace(),
                            context_->getCublasLtWorkspaceSize() );

                        cuda_fp8_apply_per_token_scales(
                            output_ptr,
                            activation_token_scales,
                            bias_,
                            outer_size, out_features_,
                            stream );
                    }
                    else if constexpr ( TWeightQuant::kIsFp4E2M1 && !kUseFusedFp4Gemm )
                    {
                        // 2-phase FP4 prefill path (mirrors the FP8 baseline above):
                        //   Phase 1 -- expand packed FP4 weights into the shared BF16 staging buffer.
                        //   Phase 2 -- standard BF16 cuBLASLt NT row-major GEMM using the staging buffer.
                        //   Phase 3 -- add bias post-GEMM (plan built with has_bias=false).
                        //
                        // The fused WMMA kernel this replaces is compute-bound at ~2.5 TFLOPS on
                        // prefill-shaped M while the staged cuBLASLt GEMM runs at tensor-core rates;
                        // the staging traffic (write + read one BF16 weight matrix per forward) is
                        // the cheaper side of that trade by ~3x at chunk 32. Fetched per-forward,
                        // never cached: the scratch buffer may be reallocated on grow.
                        const size_t staging_bytes = static_cast<size_t>( out_features_ )
                            * static_cast<size_t>( cached_in_features_ )
                            * sizeof( __nv_bfloat16 );

                        auto* staging = static_cast<__nv_bfloat16*>(
                            context_->getDeviceScratchBuffer( staging_bytes ) );

                        cuda_fp4_dequantize_to_bf16(
                            staging,
                            weight_,
                            weight_scales_,
                            out_features_, cached_in_features_,
                            weight_group_size_,
                            stream );

                        const float alpha = 1.0f;
                        const float beta  = 0.0f;

                        execute_linear_plan<TComputePrecision>(
                            cached_cublaslt_handle_,
                            forward_plan_cache_.get( outer_size ),
                            &alpha,
                            input_ptr,
                            staging,
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

            // No cuBLASLt plan -- fallback paths.
            // FP8: per-row matvec loop using the existing FP8 decode matvec.
            // INT4: per-row W4A16 GEMM (uses M=1 tiled GEMM -- less optimal than a dedicated matvec).
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

                    if constexpr ( TWeightQuant::kIsFp4E2M1 )
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
            return "Cuda::LinearOp";
        }

        const LinearConfig& getConfig() const
        {
            return config_;
        }

    private:

        LinearConfig config_;
        CudaExecutionContext* context_;

        // Weight pointer typed to WeightType -- differs from ComputeType on the FP8 path.
        const WeightType* weight_{ nullptr };

        // Per-channel FP32 scales [out_features] -- non-null on kIsPerChannelQuantized path.
        // Per-group  FP32 scales [out_features x in_features/group_size] -- non-null on kIsPerGroupQuantized path.
        const float* weight_scales_{ nullptr };

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
        // kIsPerChannelQuantized + !kUseW8A16Gemm: BF16xBF16 NT plan fed by the FP8->BF16 staging buffer.
        // Non-quantized: BF16xBF16 (or FP32xFP32) plan fed directly by the weight tensor.
        CublasLtPlanCache<CublasLtLinearPlan<TComputePrecision>> forward_plan_cache_;

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

        CublasLtPlanCache<CublasLtMatMulPlan<ComputeType>> backward_input_plan_cache_;
        CublasLtMatMulPlan<ComputeType> backward_weight_plan_;

        cudaDataType_t cuda_data_type_{};
        cudaDataType_t cuda_weight_data_type_{};
        cublasComputeType_t compute_type_{};
        cudaDataType_t scale_type_{};

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

        void buildCublasLtPlans()
        {
            cuda_data_type_ = getActivationCudaDataType();
            cuda_weight_data_type_ = getWeightCudaDataType();

            getComputeTypes( compute_type_, scale_type_ );

            if constexpr ( kIsPerChannelQuantized )
            {
                if constexpr ( kUseW8A16Gemm )
                {
                    // W8A16 fused path: no staging buffer or cuBLASLt plan needed.
                    // cuda_w8a16_gemm reads FP8 weights once and dequantizes per-channel
                    // inline in shared memory -- SM >= 8.0 guaranteed by supportsCuBLASLt().
                    Logging::Logger::info( std::format(
                        "CudaLinearOp: W8A16 fused GEMM ready -- {} in -> {} out",
                        cached_in_features_, out_features_ ) );
                }
                else
                {
                    // 2-phase baseline path:
                    //   Allocate BF16 staging buffer -- filled per-forward by cuda_fp8_dequantize_to_bf16
                    //   then fed into a standard BF16xBF16 NT row-major cuBLASLt plan.
                    // has_bias=false: bias applied post-GEMM by cuda_add_bias (Ada epilogue constraint).
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

                    Logging::Logger::info( std::format(
                        "CudaLinearOp: FP8 dequant + BF16 cuBLASLt GEMM -- {} in -> {} out",
                        cached_in_features_, out_features_ ) );
                }

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

                    fp8_forward_plan_cache_ = CublasLtPlanCache<Fp8LinearPlan>(
                        cached_outer_size_,
                        [&]( int bucket )
                        {
                            return build_fp8_prefill_plan<TComputePrecision>(
                                cached_cublaslt_handle_,
                                bucket,
                                cached_in_features_,
                                out_features_,
                                activation_fp8_unit_scale_,
                                weight_fp8_scale_ );
                        } );

                    Logging::Logger::info( std::format(
                        "CudaLinearOp: FP4->FP8 upcast + FP8xFP8 cuBLASLt GEMM (W4A8) -- {} in -> {} out (group_size={})",
                        cached_in_features_, out_features_, weight_group_size_ ) );

                    return;
                }

                if constexpr ( TWeightQuant::kIsFp4E2M1 && !kUseFusedFp4Gemm )
                {
                    // 2-phase FP4 prefill path: packed FP4 weights are expanded into the
                    // shared BF16 staging buffer per forward, then fed into a standard
                    // BF16 cuBLASLt plan -- identical structure to the FP8 baseline above.
                    // has_bias=false: bias applied post-GEMM by cuda_add_bias.
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

                    Logging::Logger::info( std::format(
                        "CudaLinearOp: FP4 dequant + BF16 cuBLASLt GEMM -- {} in -> {} out (group_size={})",
                        cached_in_features_, out_features_, weight_group_size_ ) );

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
