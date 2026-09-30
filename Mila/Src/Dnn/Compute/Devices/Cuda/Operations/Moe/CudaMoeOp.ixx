/**
 * @file CudaMoeOp.ixx
 * @brief CUDA mixture-of-experts bank: the two-pass kernel over the stacked expert tensors.
 *
 * Same contract as CpuMoeOp, gated against HuggingFace through MixtureOfExperts (Gemma4MoE.md Phase 6).
 * Under a per-group FP4 or Q4_0 policy the passes read packed codes and their group scales in place.
 */

module;
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <format>
#include <stdexcept>
#include <string>
#include <cuda_runtime_api.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "Kernels/Moe.cuh"
#include "../Linear/Kernels/Quantization/CudaFp4WeightQuantization.cuh"
#include "../Linear/Kernels/Quantization/CudaInt4WeightQuantization.cuh"

export module Compute.CudaMoeOp;

import Dnn.Components.MixtureOfExpertsConfig;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.Component;
import Dnn.Quantization.Weight.Policies;
import Compute.OperationBase;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.OperationType;
import Compute.CudaTensorDataType;
import Serialization.Tensor;

namespace Mila::Dnn::Compute::Cuda::Moe
{
    /**
     * @tparam TPrecision          Activation and weight precision (FP32 or BF16).
     * @tparam TFunctor            POD gate functor from Mila::Dnn::Activations exposing fwd(x).
     * @tparam TWeightQuantization NoWeightQuant, or PerGroupFp4<g> or PerGroupInt4<32> at BF16; anything else is
     *                             refused.
     */
    export template<TensorDataType TPrecision, typename TFunctor,
        typename TWeightQuantization = Mila::Dnn::Quant::Weight::NoWeightQuant>
    class CudaMoeOp : public Operation<DeviceType::Cuda, TPrecision>
    {
    public:
        using OperationBaseType = Operation<DeviceType::Cuda, TPrecision>;
        using NativeType = typename Mila::Dnn::Compute::Cuda::TensorDataTypeMap<TPrecision>::device_type;
        using CudaExecutionContext = ExecutionContext<DeviceType::Cuda>;

        static constexpr bool kIsFp4 = Mila::Dnn::Quant::Weight::HasFp4E2M1Codes<TWeightQuantization>;

        // Q4_0 only: the INT4 quantizer takes a group of 32.
        static constexpr bool kIsInt4 = []() constexpr
        {
            if constexpr ( Mila::Dnn::Quant::Weight::HasInt4Codes<TWeightQuantization> )
            {
                return TWeightQuantization::kQuantizationGroupSize == 32;
            }
            else
            {
                return false;
            }
        }();

        static constexpr bool kIsPacked = kIsFp4 || kIsInt4;

        CudaMoeOp( IExecutionContext* context, const MixtureOfExpertsConfig& config )
            : context_( validateExecutionContext_<DeviceType::Cuda>( context, "CudaMoeOp" ) ),
              config_( config )
        {
            config_.validate();

            if ( TWeightQuantization::kIsQuantized && !kIsPacked )
            {
                throw std::invalid_argument(
                    "CudaMoeOp: an expert bank stores unquantized, per-group FP4 or Q4_0 weights; this policy is not implemented" );
            }

            if ( kIsPacked && TPrecision != TensorDataType::BF16 )
            {
                throw std::invalid_argument( "CudaMoeOp: packed expert weights require BF16 compute precision" );
            }
        }

        /**
         * @brief Bind the stacked expert projections: gate_up [E, 2I, H] and down [E, H, I], packed two codes a byte
         *        under a per-group policy.
         */
        void setParameters( ITensor* gate_up_projection, ITensor* down_projection ) override
        {
            gate_up_projection_ = gate_up_projection;
            down_projection_ = down_projection;
        }

        /**
         * @brief Bind the per-group scales, of the policy's scale type: gate_up [E, 2I, H/g] and down [E, H, I/g].
         */
        void setWeightScales( ITensor* gate_up_scales, ITensor* down_scales )
        {
            gate_up_scales_ = gate_up_scales;
            down_scales_ = down_scales;
        }

        /**
         * @brief Check the expert widths against the quantization group; the op allocates nothing.
         */
        void build( const BuildContext& build_context ) override
        {
            if constexpr ( kIsPacked )
            {
                const dim_t group = TWeightQuantization::kQuantizationGroupSize;

                for ( const dim_t columns : { config_.getHiddenSize(), config_.getExpertIntermediateSize() } )
                {
                    if ( columns % group != 0 )
                    {
                        throw std::invalid_argument( std::format(
                            "CudaMoeOp::build - expert projection input width ({}) must be a multiple of the group size ({})",
                            columns, group ) );
                    }
                }
            }

            OperationBaseType::build( build_context );
        }

        /**
         * @brief Quantize a BF16 stacked projection into packed codes and their per-group scales.
         *
         * Every expert row is a Linear weight row of the same policy, so the stack is `rows` output channels of
         * Linear's per-group quantizer for that policy.
         *
         * @param rows    Rows across the whole stack: experts x rows per expert.
         * @param columns Input width of every row.
         */
        void quantize( const Serialization::ITensorBlob& blob, ITensor& packed_out, ITensor& scales_out,
            dim_t rows, dim_t columns ) requires kIsPacked
        {
            const auto& metadata = blob.getMetadata();
            const std::size_t source_bytes = static_cast<std::size_t>( rows * columns ) * sizeof( __nv_bfloat16 );

            if ( metadata.dtype != TensorDataType::BF16 || blob.sizeBytes() != source_bytes )
            {
                throw std::invalid_argument( std::format(
                    "CudaMoeOp::quantize: expected {} bytes of BF16 for {} x {}, got {} bytes of {}",
                    source_bytes, rows, columns, blob.sizeBytes(), tensorDataTypeToString( metadata.dtype ) ) );
            }

            const std::size_t staging_bytes = std::min( source_bytes, context_->getLoadStagingLimitBytes() );
            void* staging = context_->getLoadStagingBuffer( staging_bytes );

            if constexpr ( kIsFp4 )
            {
                Linear::cuda_quantize_fp4_per_group(
                    blob.data(), packed_out.rawData(), static_cast<float*>( scales_out.rawData() ),
                    static_cast<int64_t>( rows ), static_cast<int64_t>( columns ),
                    TWeightQuantization::kQuantizationGroupSize, staging, staging_bytes, context_->getStream() );
            }
            else
            {
                Linear::cuda_quantize_int4_per_group(
                    blob.data(), packed_out.rawData(), scales_out.rawData(),
                    static_cast<int64_t>( rows ), static_cast<int64_t>( columns ),
                    TWeightQuantization::kQuantizationGroupSize, staging, staging_bytes, context_->getStream() );
            }
        }

        /**
         * @brief Route every token of @p input through its experts into @p output.
         *
         * @param input    [..., H] at the op's precision.
         * @param weights  [..., top_k] combine weights at the op's precision.
         * @param indices  INT32 [..., top_k] expert indices. An out-of-range index yields NaN.
         * @param gated    FP32 scratch of at least tokens x top_k x intermediate elements, overwritten.
         * @param output   [..., H] at the op's precision, overwritten.
         */
        void forward( const ITensor& input, const ITensor& weights, const ITensor& indices, ITensor& gated,
            ITensor& output ) const
        {
            const dim_t hidden = config_.getHiddenSize();
            const dim_t intermediate = config_.getExpertIntermediateSize();
            const dim_t experts = config_.getNumExperts();
            const dim_t top_k = config_.getTopK();

            if ( gate_up_projection_ == nullptr || down_projection_ == nullptr )
            {
                throw std::logic_error( "CudaMoeOp::forward: expert projections were never bound" );
            }

            if constexpr ( kIsPacked )
            {
                const dim_t group = TWeightQuantization::kQuantizationGroupSize;

                if ( gate_up_scales_ == nullptr || down_scales_ == nullptr
                     || gate_up_scales_->getDataType() != TWeightQuantization::kScaleDtype
                     || down_scales_->getDataType() != TWeightQuantization::kScaleDtype
                     || gate_up_projection_->size() != experts * 2 * intermediate * ( hidden / 2 )
                     || down_projection_->size() != experts * hidden * ( intermediate / 2 )
                     || gate_up_scales_->size() != experts * 2 * intermediate * ( hidden / group )
                     || down_scales_->size() != experts * hidden * ( intermediate / group ) )
                {
                    throw std::invalid_argument( "CudaMoeOp::forward: packed expert projections or their scales do not match the configured geometry" );
                }
            }
            else
            {
                if ( gate_up_projection_->size() != experts * 2 * intermediate * hidden
                     || down_projection_->size() != experts * hidden * intermediate )
                {
                    throw std::invalid_argument( "CudaMoeOp::forward: expert projections do not match the configured geometry" );
                }
            }

            if ( input.size() % hidden != 0 )
            {
                throw std::invalid_argument( std::format(
                    "CudaMoeOp::forward: {} inputs is not a whole number of {}-wide tokens", input.size(), hidden ) );
            }

            const dim_t tokens = input.size() / hidden;

            if ( output.size() != input.size() || weights.size() != tokens * top_k || indices.size() != tokens * top_k )
            {
                throw std::invalid_argument( std::format(
                    "CudaMoeOp::forward: {} tokens need an output of {} and routing of {} elements; got {}, {} weights, {} indices",
                    tokens, input.size(), tokens * top_k, output.size(), weights.size(), indices.size() ) );
            }

            if ( gated.getDataType() != TensorDataType::FP32 || gated.size() < tokens * top_k * intermediate )
            {
                throw std::invalid_argument( std::format(
                    "CudaMoeOp::forward: {} tokens need {} FP32 gated elements; got {} of {}",
                    tokens, tokens * top_k * intermediate, gated.size(), tensorDataTypeToString( gated.getDataType() ) ) );
            }

            const cudaStream_t stream = context_->getStream();
            auto* gated_data = static_cast<float*>( gated.rawData() );
            const auto* index_data = static_cast<const int32_t*>( indices.rawData() );

            if constexpr ( kIsFp4 )
            {
                const int group = TWeightQuantization::kQuantizationGroupSize;

                launch_moe_gated_forward_fp4<TFunctor>(
                    static_cast<const __nv_bfloat16*>( input.rawData() ),
                    static_cast<const uint8_t*>( gate_up_projection_->rawData() ),
                    static_cast<const float*>( gate_up_scales_->rawData() ),
                    index_data, gated_data,
                    narrowToKernelIndex( tokens ), narrowToKernelIndex( hidden ), narrowToKernelIndex( intermediate ),
                    narrowToKernelIndex( experts ), narrowToKernelIndex( top_k ), group,
                    functor_, stream );

                launch_moe_combine_forward_fp4(
                    gated_data,
                    static_cast<const uint8_t*>( down_projection_->rawData() ),
                    static_cast<const float*>( down_scales_->rawData() ),
                    static_cast<const __nv_bfloat16*>( weights.rawData() ),
                    index_data,
                    static_cast<__nv_bfloat16*>( output.rawData() ),
                    narrowToKernelIndex( tokens ), narrowToKernelIndex( hidden ), narrowToKernelIndex( intermediate ),
                    narrowToKernelIndex( experts ), narrowToKernelIndex( top_k ), group,
                    stream );
            }
            else if constexpr ( kIsInt4 )
            {
                const int group = TWeightQuantization::kQuantizationGroupSize;

                launch_moe_gated_forward_int4<TFunctor>(
                    static_cast<const __nv_bfloat16*>( input.rawData() ),
                    static_cast<const uint8_t*>( gate_up_projection_->rawData() ),
                    static_cast<const __half*>( gate_up_scales_->rawData() ),
                    index_data, gated_data,
                    narrowToKernelIndex( tokens ), narrowToKernelIndex( hidden ), narrowToKernelIndex( intermediate ),
                    narrowToKernelIndex( experts ), narrowToKernelIndex( top_k ), group,
                    functor_, stream );

                launch_moe_combine_forward_int4(
                    gated_data,
                    static_cast<const uint8_t*>( down_projection_->rawData() ),
                    static_cast<const __half*>( down_scales_->rawData() ),
                    static_cast<const __nv_bfloat16*>( weights.rawData() ),
                    index_data,
                    static_cast<__nv_bfloat16*>( output.rawData() ),
                    narrowToKernelIndex( tokens ), narrowToKernelIndex( hidden ), narrowToKernelIndex( intermediate ),
                    narrowToKernelIndex( experts ), narrowToKernelIndex( top_k ), group,
                    stream );
            }
            else
            {
                launch_moe_gated_forward<NativeType, TFunctor>(
                    static_cast<const NativeType*>( input.rawData() ),
                    static_cast<const NativeType*>( gate_up_projection_->rawData() ),
                    index_data, gated_data,
                    narrowToKernelIndex( tokens ), narrowToKernelIndex( hidden ), narrowToKernelIndex( intermediate ),
                    narrowToKernelIndex( experts ), narrowToKernelIndex( top_k ),
                    functor_, stream );

                launch_moe_combine_forward<NativeType>(
                    gated_data,
                    static_cast<const NativeType*>( down_projection_->rawData() ),
                    static_cast<const NativeType*>( weights.rawData() ),
                    index_data,
                    static_cast<NativeType*>( output.rawData() ),
                    narrowToKernelIndex( tokens ), narrowToKernelIndex( hidden ), narrowToKernelIndex( intermediate ),
                    narrowToKernelIndex( experts ), narrowToKernelIndex( top_k ),
                    stream );
            }
        }

        OperationType getOperationType() const override
        {
            return OperationType::MoeOp;
        }

        std::string getName() const override
        {
            return "Cuda::MoeOp";
        }

    private:
        CudaExecutionContext* context_{ nullptr };
        MixtureOfExpertsConfig config_;
        TFunctor functor_{};

        ITensor* gate_up_projection_{ nullptr };
        ITensor* down_projection_{ nullptr };
        ITensor* gate_up_scales_{ nullptr };
        ITensor* down_scales_{ nullptr };
    };
}
