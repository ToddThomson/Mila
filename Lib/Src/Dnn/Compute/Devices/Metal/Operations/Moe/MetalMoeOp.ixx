/**
 * @file MetalMoeOp.ixx
 * @brief A mixture-of-experts bank on Metal: CUDA's two passes over the stacked expert tensors, unquantized, per-group FP4 or Q4_0.
 */

module;
#include "../../../../../Components/Activations/Activation/Kernels/ElementwiseActivation.h"

export module Compute.MetalMoeOp;

import std;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.IExecutionContext;
import Compute.MetalExecutionContext;
import Compute.MetalKernelLibrary;
import Compute.MetalLinearOp;
import Compute.MetalTensorDataTypeMap;
import Compute.OperationBase;
import Compute.OperationType;
import Dnn.Component;
import Dnn.Components.LinearConfig;
import Dnn.Components.MixtureOfExpertsConfig;
import Dnn.ITensor;
import Dnn.Quantization.Weight.Policies;
import Dnn.TensorDataType;
import Dnn.TensorTypes;
import Serialization.Tensor;

namespace Mila::Dnn::Compute::Metal::Moe
{
    using namespace Mila::Dnn::Quant::Weight;

    /**
     * @brief Routes each token through its selected experts and combines them by the routing weights.
     *
     * Each output depends on its own token alone, so `forward` over several tokens gives each the bits a one-token call gives it, and
     * `decode` is `forward`. Packed weights are two codes a byte, the low nibble the even column, with a scale per group of the row:
     * FP32 for FP4 and FP16 for Q4_0, as each policy stores them.
     *
     * @tparam TPrecision          FP16 or BF16.
     * @tparam TFunctor            The gate, GeluTanh or Silu from Mila::Dnn::Activations.
     * @tparam TWeightQuantization NoWeightQuant, PerGroupFp4<g> or PerGroupInt4<32>.
     */
    export template<TensorDataType TPrecision, typename TFunctor, typename TWeightQuantization = NoWeightQuant>
    class MetalMoeOp : public Operation<DeviceType::Metal, TPrecision>
    {
    public:

        static constexpr bool kIsFp4 = HasFp4E2M1Codes<TWeightQuantization>;

        static constexpr bool kIsInt4 = []() constexpr {
            if constexpr ( HasInt4Codes<TWeightQuantization> )
            {
                return TWeightQuantization::kQuantizationGroupSize == 32;
            } else {
                return false;
            }
        }();

        static constexpr bool kIsPacked = kIsFp4 || kIsInt4;

        static_assert( !TWeightQuantization::kIsQuantized || kIsPacked, "Metal's expert bank carries unquantized, per-group FP4 and Q4_0 weights" );
        static_assert( std::is_same_v<TFunctor, Activations::GeluTanh> || std::is_same_v<TFunctor, Activations::Silu>,
            "Metal's expert bank gates with GELU's tanh form or SiLU" );

        /// Most tokens `decode` takes in one call, as CUDA's: a speculative verify's draft and the token before it.
        static constexpr dim_t kMaximumDecodeTokens = 8;

        MetalMoeOp( IExecutionContext* context, const MixtureOfExpertsConfig& config )
            : context_( validateExecutionContext_<DeviceType::Metal>( context, "MetalMoeOp" ) ),
              config_( config ),
              gated_pipeline_( context_->device().kernels().pipeline( gatedKernelName() ) ),
              combine_pipeline_( context_->device().kernels().pipeline( combineKernelName() ) )
        {
            config_.validate();
        }

        /// Bind gate_up [E, 2I, H] and down [E, H, I], packed two codes a byte under a per-group policy.
        void setParameters( ITensor* gate_up_projection, ITensor* down_projection ) override
        {
            gate_up_projection_ = gate_up_projection;
            down_projection_ = down_projection;
        }

        /// Bind the per-group scales: gate_up [E, 2I, H/g] and down [E, H, I/g].
        void setWeightScales( ITensor* gate_up_scales, ITensor* down_scales )
        {
            gate_up_scales_ = gate_up_scales;
            down_scales_ = down_scales;
        }

        /// @throws std::invalid_argument When a packed bank's widths are not whole groups of eight codes.
        void build( const BuildContext& build_context ) override
        {
            if constexpr ( kIsPacked )
            {
                const dim_t multiple = std::max<dim_t>( 8, TWeightQuantization::kQuantizationGroupSize );

                for ( const dim_t columns : { config_.getHiddenSize(), config_.getExpertIntermediateSize() } )
                {
                    if ( columns % multiple != 0 )
                    {
                        throw std::invalid_argument( std::format(
                            "MetalMoeOp::build - an expert projection's input width ({}) must be a multiple of {}", columns, multiple ) );
                    }
                }
            }

            Operation<DeviceType::Metal, TPrecision>::build( build_context );
        }

        /**
         * @brief Quantize a stacked BF16 projection, `rows` rows of `columns` across every expert, into FP4 codes and their scales,
         *        each row as Linear quantizes one.
         *
         * @throws std::invalid_argument For a Q4_0 bank, which Metal loads only as stored, for weights that are not BF16 of that size,
         *         or for any weight that is not finite.
         */
        void quantize( const Serialization::ITensorBlob& blob, ITensor& packed_out, ITensor& scales_out, dim_t rows, dim_t columns )
            requires kIsPacked
        {
            if constexpr ( kIsInt4 )
            {
                throw std::invalid_argument( std::format(
                    "MetalMoeOp: '{}' holds full-precision weights, and Metal loads int4 weights only as stored in that format",
                    packed_out.getName() ) );
            } else {
                const std::size_t source_bytes = static_cast<std::size_t>( rows * columns ) * sizeof( std::uint16_t );

                if ( blob.getMetadata().dtype != TensorDataType::BF16 || blob.sizeBytes() != source_bytes )
                {
                    throw std::invalid_argument( std::format(
                        "MetalMoeOp::quantize: expected {} bytes of BF16 for {} x {}, got {} bytes of {}",
                        source_bytes, rows, columns, blob.sizeBytes(), tensorDataTypeToString( blob.getMetadata().dtype ) ) );
                }

                Linear::MetalLinearOp<TPrecision, TWeightQuantization> rows_as_linear( context_, LinearConfig( columns, rows ).withBias( false ) );

                rows_as_linear.quantizeRows( static_cast<const std::byte*>( blob.data() ), static_cast<std::size_t>( rows ),
                    static_cast<std::size_t>( columns ), packed_out, scales_out );
            }
        }

        /**
         * @brief Route every token of `input` [..., H] through its experts into `output` [..., H].
         *
         * @param weights [..., top_k] combine weights at the op's precision.
         * @param indices INT32 [..., top_k] expert indices. An index out of range writes not-a-number for its token.
         * @param gated   FP32 of at least tokens x top_k x I elements, overwritten.
         * @throws std::logic_error When the projections were never bound.
         * @throws std::invalid_argument When the tensors do not match the configured geometry.
         */
        void forward( const ITensor& input, const ITensor& weights, const ITensor& indices, ITensor& gated, ITensor& output ) const
        {
            const dim_t tokens = checkedTokens( "forward", input, weights, indices, gated, output );

            if ( tokens == 0 )
            {
                return;
            }

            const dim_t hidden = config_.getHiddenSize();
            const dim_t intermediate = config_.getExpertIntermediateSize();
            const dim_t top_k = config_.getTopK();

            std::uint32_t group = 0;

            if constexpr ( kIsPacked )
            {
                group = static_cast<std::uint32_t>( TWeightQuantization::kQuantizationGroupSize );
            }

            const Shape shape{
                static_cast<std::uint32_t>( narrowToKernelIndex( tokens ) ), static_cast<std::uint32_t>( narrowToKernelIndex( hidden ) ),
                static_cast<std::uint32_t>( narrowToKernelIndex( intermediate ) ),
                static_cast<std::uint32_t>( narrowToKernelIndex( config_.getNumExperts() ) ),
                static_cast<std::uint32_t>( narrowToKernelIndex( top_k ) ), group };

            const auto constants = std::as_bytes( std::span( &shape, 1 ) );
            const std::size_t token_rows = static_cast<std::size_t>( tokens );
            const std::size_t gated_groups = groupsOf( top_k * intermediate );
            const std::size_t combine_groups = groupsOf( hidden );

            if constexpr ( kIsPacked )
            {
                const std::array<const void*, 5> gated_buffers{
                    input.rawData(), gate_up_projection_->rawData(), gate_up_scales_->rawData(), indices.rawData(), gated.rawData() };

                context_->dispatchThreadgroups(
                    gated_pipeline_, gated_buffers, constants, gated_groups, token_rows, kSimdgroups );

                const std::array<const void*, 6> combine_buffers{
                    gated.rawData(), down_projection_->rawData(), down_scales_->rawData(), weights.rawData(), indices.rawData(), output.rawData() };

                context_->dispatchThreadgroups(
                    combine_pipeline_, combine_buffers, constants, combine_groups, token_rows, kSimdgroups );
            } else {
                const std::array<const void*, 4> gated_buffers{ input.rawData(), gate_up_projection_->rawData(), indices.rawData(), gated.rawData() };

                context_->dispatchThreadgroups(
                    gated_pipeline_, gated_buffers, constants, gated_groups, token_rows, kSimdgroups );

                const std::array<const void*, 5> combine_buffers{
                    gated.rawData(), down_projection_->rawData(), weights.rawData(), indices.rawData(), output.rawData() };

                context_->dispatchThreadgroups(
                    combine_pipeline_, combine_buffers, constants, combine_groups, token_rows, kSimdgroups );
            }
        }

        /**
         * @brief A multi-token decode of up to kMaximumDecodeTokens tokens (Gemma4Mtp.md 4.7): `forward`, whose tokens are each
         *        computed as one alone. Arguments as `forward`.
         *
         * @throws std::invalid_argument For more than kMaximumDecodeTokens tokens.
         */
        void decode( const ITensor& input, const ITensor& weights, const ITensor& indices, ITensor& gated, ITensor& output ) const
        {
            const dim_t tokens = checkedTokens( "decode", input, weights, indices, gated, output );

            if ( tokens > kMaximumDecodeTokens )
            {
                throw std::invalid_argument( std::format(
                    "MetalMoeOp::decode: {} tokens; a multi-token decode takes at most {}", tokens, kMaximumDecodeTokens ) );
            }

            forward( input, weights, indices, gated, output );
        }

        [[nodiscard]] OperationType getOperationType() const override
        {
            return OperationType::MoeOp;
        }

        [[nodiscard]] std::string getName() const override
        {
            return "Metal.Moe";
        }

    private:

        // The kernels' MoeShape.
        struct Shape
        {
            std::uint32_t tokens;
            std::uint32_t hidden;
            std::uint32_t intermediate;
            std::uint32_t experts;
            std::uint32_t top_k;
            std::uint32_t group;
        };

        // SIMD-groups to a threadgroup, Moe.metal's kMoeSimdgroups.
        static constexpr std::size_t kSimdgroups = 4;

        static constexpr std::string_view kFormatName = kIsInt4 ? "int4" : "fp4";
        static constexpr std::string_view kGateName = std::is_same_v<TFunctor, Activations::GeluTanh> ? "gelu" : "silu";

        [[nodiscard]] static std::string gatedKernelName()
        {
            const std::string_view type = MetalTensorDataTypeMap<TPrecision>::kernel_type;

            return kIsPacked ? std::format( "moe_gated_{}_{}_{}", kFormatName, kGateName, type ) : std::format( "moe_gated_{}_{}", kGateName, type );
        }

        [[nodiscard]] static std::string combineKernelName()
        {
            const std::string_view type = MetalTensorDataTypeMap<TPrecision>::kernel_type;

            return kIsPacked ? std::format( "moe_combine_{}_{}", kFormatName, type ) : std::format( "moe_combine_{}", type );
        }

        static std::size_t groupsOf( dim_t cells )
        {
            return ( static_cast<std::size_t>( narrowToKernelIndex( cells ) ) + kSimdgroups - 1 ) / kSimdgroups;
        }

        // The bound projections and the call's tensors against the configured geometry; the call's token count.
        dim_t checkedTokens( const char* caller, const ITensor& input, const ITensor& weights, const ITensor& indices, const ITensor& gated,
            const ITensor& output ) const
        {
            const dim_t hidden = config_.getHiddenSize();
            const dim_t intermediate = config_.getExpertIntermediateSize();
            const dim_t experts = config_.getNumExperts();
            const dim_t top_k = config_.getTopK();

            if ( gate_up_projection_ == nullptr || down_projection_ == nullptr )
            {
                throw std::logic_error( std::format( "MetalMoeOp::{}: expert projections were never bound", caller ) );
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
                    throw std::invalid_argument( std::format(
                        "MetalMoeOp::{}: packed expert projections or their scales do not match the configured geometry", caller ) );
                }
            } else {
                if ( gate_up_projection_->size() != experts * 2 * intermediate * hidden || down_projection_->size() != experts * hidden * intermediate )
                {
                    throw std::invalid_argument( std::format( "MetalMoeOp::{}: expert projections do not match the configured geometry", caller ) );
                }
            }

            if ( input.size() % hidden != 0 )
            {
                throw std::invalid_argument( std::format(
                    "MetalMoeOp::{}: {} inputs is not a whole number of {}-wide tokens", caller, input.size(), hidden ) );
            }

            const dim_t tokens = input.size() / hidden;

            if ( output.size() != input.size() || weights.size() != tokens * top_k || indices.size() != tokens * top_k )
            {
                throw std::invalid_argument( std::format(
                    "MetalMoeOp::{}: {} tokens need an output of {} and routing of {} elements; got {}, {} weights, {} indices",
                    caller, tokens, input.size(), tokens * top_k, output.size(), weights.size(), indices.size() ) );
            }

            if ( gated.getDataType() != TensorDataType::FP32 || gated.size() < tokens * top_k * intermediate )
            {
                throw std::invalid_argument( std::format( "MetalMoeOp::{}: {} tokens need {} FP32 gated elements; got {} of {}",
                    caller, tokens, tokens * top_k * intermediate, gated.size(), tensorDataTypeToString( gated.getDataType() ) ) );
            }

            return tokens;
        }

        MetalExecutionContext* context_;
        MixtureOfExpertsConfig config_;
        MetalPipeline gated_pipeline_;
        MetalPipeline combine_pipeline_;

        ITensor* gate_up_projection_{ nullptr };
        ITensor* down_projection_{ nullptr };
        ITensor* gate_up_scales_{ nullptr };
        ITensor* down_scales_{ nullptr };
    };
}
