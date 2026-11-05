/**
 * @file MetalTokenEmbeddingOp.ixx
 * @brief The token embedding lookup on Metal, unquantized.
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
import Dnn.TensorDataType;
import Dnn.TensorTypes;

namespace Mila::Dnn::Compute::Metal::TokenEmbedding
{
    /// output[b, t, :] = table[ids[b, t], :]. A token id past the table is the caller's to prevent, as on CUDA.
    export template<TensorDataType TPrecision>
    class MetalTokenEmbeddingOp : public Operation<DeviceType::Metal, TPrecision>
    {
    public:

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

        void forward( const ITensor& input, ITensor& output ) const
        {
            const std::array<const void*, 3> buffers{ input.rawData(), table_->rawData(), output.rawData() };
            const std::uint32_t embedding_dim = static_cast<std::uint32_t>( narrowToKernelIndex( config_.getEmbeddingDim() ) );

            context_->dispatch(
                context_->device().kernels().pipeline( std::format( "token_embedding_{}", MetalTensorDataTypeMap<TPrecision>::kernel_type ) ),
                buffers, std::as_bytes( std::span( &embedding_dim, 1 ) ),
                static_cast<std::size_t>( narrowToKernelIndex( input.size() * config_.getEmbeddingDim() ) ) );
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
    };
}
