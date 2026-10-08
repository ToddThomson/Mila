/**
 * @file NgramEmbedding.ixx
 * @brief Hashed n-gram embedding: each token's recent n-grams hashed into one shared table and gathered.
 *
 * Qwen 4's per-layer embedding reads its features from one of these (Specifications/Qwen4.md section 3.4). Like
 * CausalConv1d it carries a rolling window -- the last ngram_size - 1 token ids -- so chunked prefill and
 * decode compose with one pass over the whole sequence.
 */

module;
#include <cstdint>
#include <format>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

export module Dnn.Components.NgramEmbedding;

export import Dnn.Components.NgramEmbeddingConfig;

import Dnn.Component;
import Dnn.ComponentType;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.TensorOps;
import Compute.DeviceAllocation;
import Compute.DeviceId;
import Compute.DeviceType;
import Compute.DeviceTypeTraits;
import Compute.IExecutionContext;
import Compute.ExecutionContext;
import Compute.ExecutionContextFactory;
import Compute.OperationTraits;
import Compute.CpuMemoryResource;
import Compute.Observation;
import Serialization.Tensor;
import Serialization.SafeTensors;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    /**
     * @brief Token ids [B, T] to n-gram features [B, T, embedding_dim].
     *
     * One parameter, `weight`, the table [table_rows, embedding_dim / heads]. The hash's constants are
     * configuration, not parameters. Inference-only.
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class NgramEmbedding : public Component<TDeviceType, TPrecision>
    {
    public:
        using ComponentBase = Component<TDeviceType, TPrecision>;
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using TensorType = Tensor<TPrecision, MR>;
        using IndexTensorType = Tensor<TensorDataType::INT32, MR>;

        explicit NgramEmbedding( const std::string& name, const NgramEmbeddingConfig& config,
            std::optional<DeviceId> device_id = std::nullopt )
            : ComponentBase( name ), config_( config )
        {
            config_.validate();

            if ( device_id.has_value() )
            {
                if ( device_id->type != TDeviceType )
                {
                    throw std::invalid_argument( "NgramEmbedding: device type mismatch" );
                }

                owned_exec_context_ = createExecutionContext( device_id.value() );
                this->setExecutionContext( owned_exec_context_.get() );
            }
        }

        ~NgramEmbedding() override = default;

        /**
         * @brief Embed a chunk starting at @p position_offset.
         *
         * At offset 0 the sequence starts here and the history is all eos; past that the retained ids supply it.
         */
        TensorType& prefill( const IndexTensorType& token_ids, dim_t position_offset )
        {
            if ( position_offset == 0 )
            {
                resetState();
            }

            auto& output = run( token_ids, position_offset > 0 );

            this->publish( ComputePass::Prefill, "output", output );

            return output;
        }

        /// Embed one token against the retained ids.
        TensorType& decode( const IndexTensorType& token_ids, dim_t /*position*/ )
        {
            auto& output = run( token_ids, true );

            this->publish( ComputePass::Decode, "output", output );

            return output;
        }

        /// The table rows of the last call, [B, T, heads].
        const IndexTensorType& ngramIds() const
        {
            requireBuilt( "ngramIds" );

            return *ngram_ids_view_;
        }

        /// The next prefill starts a fresh sequence.
        void resetState()
        {
            requireBuilt( "resetState" );

            fill( *history_, static_cast<int32_t>( config_.getEosTokenId() ), this->getExecutionContext() );
            state_primed_ = false;
        }

        std::vector<std::string> getParameterNames() const override
        {
            return { "weight" };
        }

        std::vector<ITensor*> getParameters() const override
        {
            if ( !table_ )
            {
                return {};
            }

            return { table_.get() };
        }

        std::vector<ITensor*> getGradients() const override
        {
            return {};
        }

        dim_t parameterCount() const override
        {
            return table_ ? table_->size() : 0;
        }

        void loadParameter( const std::string& name, const ITensorBlob& blob ) override
        {
            if ( name != "weight" )
            {
                throw std::invalid_argument( std::format( "NgramEmbedding '{}': no parameter '{}'", this->getName(), name ) );
            }

            this->loadParameterFromBlob( "weight", blob, *table_, table_->shape() );
        }

        void saveFlatTensors( SafeTensorsWriter& writer, const std::string& prefix, TensorSavePass pass ) const override
        {
            if ( table_ )
            {
                this->saveParameterToWriter( writer, prefix + ".weight", *table_, pass );
            }
        }

        DeviceId getDeviceId() const override
        {
            return this->getExecutionContext()->getDeviceId();
        }

        void synchronize() override
        {
            this->getExecutionContext()->synchronize();
        }

        const ComponentType getType() const override
        {
            return ComponentType::NgramEmbedding;
        }

        std::vector<const ITensor*> getOutputs() const override
        {
            if ( !output_ )
            {
                return {};
            }

            return { output_.get() };
        }

        std::vector<ObservableStage> getObservableStages() const override
        {
            return { { "output", ComputePassMask{ ComputePass::Prefill, ComputePass::Decode } } };
        }

        MemoryStats getMemoryStats() const override
        {
            MemoryStats stats;

            if ( table_ )
            {
                stats.device_parameter_bytes += occupiedTensorBytes( *table_ );
            }

            for ( const ITensor* tensor : { static_cast<const ITensor*>( history_.get() ), static_cast<const ITensor*>( ngram_ids_.get() ),
                                            static_cast<const ITensor*>( output_.get() ) } )
            {
                if ( tensor )
                {
                    stats.device_state_bytes += occupiedTensorBytes( *tensor );
                }
            }

            return stats;
        }

        MemoryStats getRequiredMemory( const BuildContext& context ) const override
        {
            const auto& input_shape = context.inputShape();

            validateInputShape( input_shape );

            const std::size_t granularity = context.getAllocationGranularity();
            const dim_t tokens = input_shape[ 0 ] * input_shape[ 1 ];

            MemoryStats stats;

            if ( !table_ )
            {
                stats.device_parameter_bytes += occupiedDeviceBytes(
                    storageBytes<TPrecision>( config_.getTableRows() * config_.getRowWidth() ), granularity );
            }

            stats.device_state_bytes += occupiedDeviceBytes(
                storageBytes<TensorDataType::INT32>( input_shape[ 0 ] * config_.getHistoryLength() ), granularity );
            stats.device_state_bytes += occupiedDeviceBytes(
                storageBytes<TensorDataType::INT32>( tokens * config_.getHeads() ), granularity );
            stats.device_state_bytes += occupiedDeviceBytes(
                storageBytes<TPrecision>( tokens * config_.getEmbeddingDim() ), granularity );

            return stats;
        }

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "NgramEmbedding: " << this->getName() << std::endl;
            oss << "Orders 2.." << config_.getNgramSize() << ", " << config_.getHeads() << " heads, table "
                << config_.getTableRows() << " x " << config_.getRowWidth() << std::endl;

            return oss.str();
        }

    protected:

        void onExecutionContextSet() override
        {
            operation_ = std::make_shared<OpType>( this->getExecutionContext(), config_ );
        }

        void onBuilding( const BuildContext& context ) override
        {
            const auto& input_shape = context.inputShape();

            validateInputShape( input_shape );

            auto device = this->getExecutionContext()->getDeviceId();
            const std::string n = this->getName();

            if ( !table_ )
            {
                table_ = std::make_shared<TensorType>( device, shape_t{ config_.getTableRows(), config_.getRowWidth() }, n + ".weight" );
            }

            if ( context.shouldInitializeParameters() )
            {
                fill_uniform( *table_, -0.02f, 0.02f, this->getExecutionContext() );
            }

            operation_->setParameters( table_.get(), nullptr );
            operation_->build( context );

            batch_ = input_shape[ 0 ];

            history_ = std::make_shared<IndexTensorType>( device, shape_t{ batch_, config_.getHistoryLength() }, n + ".history" );
            ngram_ids_ = std::make_shared<IndexTensorType>( device, shape_t{ input_shape[ 0 ], input_shape[ 1 ], config_.getHeads() }, n + ".ngram_ids" );
            output_ = std::make_shared<TensorType>( device, shape_t{ input_shape[ 0 ], input_shape[ 1 ], config_.getEmbeddingDim() }, n + ".output" );

            built_tokens_ = input_shape[ 0 ] * input_shape[ 1 ];
            view_shape_ = shape_t{};

            fill( *history_, static_cast<int32_t>( config_.getEosTokenId() ), this->getExecutionContext() );
            state_primed_ = false;
        }

        void onTrainingModeChanging( TrainingMode /*training_mode*/ ) override
        {
        }

    private:
        using OpType = typename OperationTraits<OperationType::NgramEmbeddingOp, TDeviceType, TPrecision>::type;

        NgramEmbeddingConfig config_;

        std::unique_ptr<IExecutionContext> owned_exec_context_{ nullptr };
        std::shared_ptr<OpType> operation_{ nullptr };

        std::shared_ptr<TensorType> table_{ nullptr };

        // The last ngram_size - 1 token ids, [B, ngram_size - 1]: the hash's whole memory.
        std::shared_ptr<IndexTensorType> history_{ nullptr };
        bool state_primed_{ false };
        dim_t batch_{ 0 };

        std::shared_ptr<IndexTensorType> ngram_ids_{ nullptr };
        std::shared_ptr<TensorType> output_{ nullptr };
        std::optional<IndexTensorType> ngram_ids_view_;
        std::optional<TensorType> output_view_;
        dim_t built_tokens_{ 0 };
        shape_t view_shape_{};

        TensorType& run( const IndexTensorType& token_ids, bool use_state )
        {
            requireBuilt( "run" );

            const auto& input_shape = token_ids.shape();

            validateInputShape( input_shape );

            if ( input_shape[ 0 ] != batch_ || input_shape[ 0 ] * input_shape[ 1 ] > built_tokens_ )
            {
                throw std::invalid_argument( std::format(
                    "NgramEmbedding '{}': input exceeds the built shape", this->getName() ) );
            }

            // A continuation before any chunk would hash against an eos history that stands for nothing.
            if ( use_state && !state_primed_ )
            {
                throw std::logic_error( std::format(
                    "NgramEmbedding '{}': continuation requested before any chunk was seen -- call prefill() at "
                    "position 0 first", this->getName() ) );
            }

            if ( input_shape != view_shape_ )
            {
                ngram_ids_view_.emplace( ngram_ids_->view( shape_t{ input_shape[ 0 ], input_shape[ 1 ], config_.getHeads() } ) );
                output_view_.emplace( output_->view( shape_t{ input_shape[ 0 ], input_shape[ 1 ], config_.getEmbeddingDim() } ) );
                view_shape_ = input_shape;
            }

            operation_->hash( token_ids, *history_, *ngram_ids_view_ );
            operation_->gather( *ngram_ids_view_, *output_view_ );

            state_primed_ = true;

            return *output_view_;
        }

        void requireBuilt( const char* caller ) const
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( std::format( "NgramEmbedding '{}': {} before build", this->getName(), caller ) );
            }
        }

        void validateInputShape( const shape_t& input_shape ) const
        {
            if ( input_shape.size() != 2 )
            {
                throw std::invalid_argument( std::format(
                    "NgramEmbedding '{}': token ids must be rank 2 [B, T], got rank {}", this->getName(), input_shape.size() ) );
            }
        }
    };
}
