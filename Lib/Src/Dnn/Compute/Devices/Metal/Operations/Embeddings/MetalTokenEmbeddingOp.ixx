/**
 * @file MetalTokenEmbeddingOp.ixx
 * @brief The token embedding lookup on Metal, on an unquantized table or a six-bit one.
 */

export module Compute.MetalTokenEmbeddingOp;

import std;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.IExecutionContext;
import Compute.MetalExecutionContext;
import Compute.MetalKernelLibrary;
import Compute.MetalTensorDataTypeMap;
import Compute.OperationBase;
import Compute.OperationType;
import Dnn.Component;
import Dnn.Components.TokenEmbeddingConfig;
import Dnn.ITensor;
import Dnn.Quantization.Weight.Policies;
import Dnn.TensorDataType;
import Dnn.TensorTypes;
import Serialization.Tensor;

namespace Mila::Dnn::Compute::Metal::TokenEmbedding
{
    using namespace Mila::Dnn::Quant::Weight;

    /**
     * @brief output[b, t, :] = table[ids[b, t], :]. A token id past the table is the caller's to prevent, as on CUDA.
     *
     * A six-bit table is Gemma's (`Quantization.md`): codes and an FP16 scale per 32 channels, decoded as they are gathered. The
     * tied head reads the same bytes through Linear.
     *
     * @tparam TPrecision FP16 or BF16.
     * @tparam TTableQuantization NoWeightQuant or PerGroupInt6<32>.
     */
    export template<TensorDataType TPrecision, WeightQuantPolicy TTableQuantization = NoWeightQuant>
    class MetalTokenEmbeddingOp : public Operation<DeviceType::Metal, TPrecision>
    {
    public:

        static constexpr bool kIsQuantized = TTableQuantization::kIsQuantized;

        static_assert( !kIsQuantized || std::is_same_v<TTableQuantization, PerGroupInt6<32>>, "Metal's quantized table is six bits per 32" );

        MetalTokenEmbeddingOp( IExecutionContext* context, const TokenEmbeddingConfig& config )
            : context_( validateExecutionContext_<DeviceType::Metal>( context, "MetalTokenEmbeddingOp" ) ),
              config_( config )
        {
            config_.validate();
        }

        void setParameters( ITensor* table, ITensor* ) override
        {
            table_ = table;
        }

        /// The table's FP16 scales, one for each 32 channels of a row.
        void setTableScales( ITensor* scales ) requires kIsQuantized
        {
            if ( scales == nullptr || scales->getDeviceType() != DeviceType::Metal )
            {
                throw std::invalid_argument( "MetalTokenEmbeddingOp::setTableScales - the scales must be a Metal tensor" );
            }

            table_scales_ = scales;
        }

        /// @throws std::invalid_argument Always: Metal loads a six-bit table only as stored in that format.
        void quantize( const Serialization::ITensorBlob&, ITensor& table_out, ITensor&, const shape_t& ) requires kIsQuantized
        {
            throw std::invalid_argument( std::format(
                "MetalTokenEmbeddingOp: '{}' holds a full-precision table, and Metal loads a six-bit table only as stored in that format",
                table_out.getName() ) );
        }

        void build( const BuildContext& build_context ) override
        {
            if constexpr ( kIsQuantized )
            {
                if ( table_scales_ == nullptr )
                {
                    throw std::runtime_error( "MetalTokenEmbeddingOp::build - setTableScales() must be called before build()" );
                }

                if ( config_.getEmbeddingDim() % 32 != 0 )
                {
                    throw std::invalid_argument( std::format(
                        "MetalTokenEmbeddingOp::build - a six-bit table needs a width a multiple of 32, and this is {}", config_.getEmbeddingDim() ) );
                }
            }

            Operation<DeviceType::Metal, TPrecision>::build( build_context );
        }

        void forward( const ITensor& input, ITensor& output ) const
        {
            const std::uint32_t embedding_dim = static_cast<std::uint32_t>( narrowToKernelIndex( config_.getEmbeddingDim() ) );
            const std::size_t elements = static_cast<std::size_t>( narrowToKernelIndex( input.size() * config_.getEmbeddingDim() ) );
            const auto type = MetalTensorDataTypeMap<TPrecision>::kernel_type;

            if constexpr ( kIsQuantized )
            {
                const std::array<const void*, 4> buffers{ input.rawData(), table_->rawData(), table_scales_->rawData(), output.rawData() };

                context_->dispatch( context_->device().kernels().pipeline( std::format( "token_embedding_int6_{}", type ) ),
                    buffers, std::as_bytes( std::span( &embedding_dim, 1 ) ), elements );
            } else {
                const std::array<const void*, 3> buffers{ input.rawData(), table_->rawData(), output.rawData() };

                context_->dispatch( context_->device().kernels().pipeline( std::format( "token_embedding_{}", type ) ),
                    buffers, std::as_bytes( std::span( &embedding_dim, 1 ) ), elements );
            }
        }

        [[nodiscard]] OperationType getOperationType() const override
        {
            return OperationType::TokenEmbeddingOp;
        }

        [[nodiscard]] std::string getName() const override
        {
            return "Metal.TokenEmbedding";
        }

    private:

        MetalExecutionContext* context_;
        TokenEmbeddingConfig config_;
        ITensor* table_{ nullptr };
        ITensor* table_scales_{ nullptr };
    };
}
