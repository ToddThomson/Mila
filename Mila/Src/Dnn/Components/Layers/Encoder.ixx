/**
 * @file Encoder.ixx
 * @brief Device-templated Encoder module for token and positional embeddings.
 *
 * Delegates compute to a UnaryOperation backend. Module owns token (wte) and
 * positional (wpe) embedding parameters and exposes them to callers.
 */

module;
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <stdexcept>
#include <cstdint>
#include <optional>

export module Dnn.Components.Encoder;
export import :Config;

import Dnn.Component;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Compute.Precision;
import Compute.Device;
import Compute.DeviceId;
import Compute.DeviceType;
import Compute.DeviceTypeTraits;
import Compute.ExecutionContext;
import Compute.ExecutionContextFactory;
import Compute.UnaryOperation;
import Compute.OperationRegistry;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.CudaDeviceMemoryResource;
import Serialization.ModelArchive;
import Serialization.Mode;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    /**
     * @brief Encoder module for token and positional embeddings (device-templated).
     *
     * Delegates computation to a device-specific UnaryOperation implementation
     * registered in the OperationRegistry.
     *
     * The Encoder transforms input token IDs into continuous vector representations:
     * 1. Looks up token embeddings from vocabulary table (wte)
     * 2. Adds positional embeddings (wpe) based on sequence position
     *
     * Module owns trainable parameters (wte, wpe) and exposes them via accessors.
     * The operation implements embedding lookup and position encoding addition.
     *
     * Construction modes:
     * - Standalone: provide a DeviceId to create and own an ExecutionContext.
     * - Deferred/shared: omit DeviceId and caller must call setExecutionContext() before build().
     *
     * @tparam TDeviceType Device type (DeviceType::Cpu or DeviceType::Cuda)
     * @tparam TIndex Data type for token indices (typically INT32)
     * @tparam TPrecision Abstract tensor precision (TensorDataType) for embeddings
     */
    export template<DeviceType TDeviceType, TensorDataType TIndex = dtype_t::INT32, TensorDataType TPrecision = dtype_t::FP32>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class Encoder : public Component<TDeviceType, TPrecision>
    {
    public:
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using EmbeddingsTensorType = Tensor<TPrecision, MR>;
        using TokenIndexType = Tensor<TIndex, MR>;
        using ComponentBase = Component<TDeviceType, TPrecision>;
    
        /**
         * @brief Construct Encoder component.
         *
         * Two modes:
         * - Standalone mode: provide DeviceId and component will create and own an ExecutionContext.
         * - Child mode: omit DeviceId and parent must call setExecutionContext() before build().
         *
         * @param name Component name identifier (mandatory)
         * @param config Encoder configuration
         * @param device_id Optional DeviceId to create owned ExecutionContext (standalone mode)
         */
        explicit Encoder( const std::string& name, const EncoderConfig& config, std::optional<DeviceId> device_id = std::nullopt )
            : ComponentBase( name ), config_( config )
        {
            config_.validate();

            if ( device_id.has_value() )
            {
                if ( device_id->type != TDeviceType )
                {
                    throw std::invalid_argument( "Encoder: device type mismatch" );
                }

                owned_exec_context_ = createExecutionContext( device_id.value() );

                this->setExecutionContext( owned_exec_context_.get() );
            }
        }

        ~Encoder() override = default;

        // ====================================================================
        // Compute operation dispatch (new API)
        // ====================================================================

        /**
         * @brief Forward pass - returns component-owned embeddings tensor.
         *
         * @param input Input token indices tensor [B, T]
         * @return Reference to component-owned embeddings tensor [B, T, C]
         *
         * @throws std::runtime_error if component is not built or backend not initialized.
         */
        EmbeddingsTensorType& forward( const TokenIndexType& input )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "Encoder module must be built before calling forward." );
            }

            validateInputShape( input );

            if ( !operation_ )
            {
                throw std::runtime_error( "Encoder: operation backend not initialized" );
            }

            if ( !owned_output_ )
            {
                throw std::runtime_error( "Encoder: owned output buffer not allocated" );
            }

            operation_->forward( input, *owned_output_ );

            return *owned_output_;
        }

        /**
         * @brief Backward pass - compute parameter gradients and return owned input-grad.
         *
         * Token indices are discrete and not differentiable; the backend may still
         * expect an input-gradient tensor. The component owns a token-index-typed
         * input-gradient buffer that is passed to the backend and returned.
         *
         * @param input Input token indices tensor used during forward.
         * @param output_grad Gradient w.r.t. embeddings [B, T, C].
         * @return Reference to component-owned token-index-typed input-grad tensor.
         *
         * @throws std::runtime_error if component is not built, not in training mode,
         *         or backend/buffers are not initialized.
         */
        TokenIndexType& backward( const TokenIndexType& input, const EmbeddingsTensorType& output_grad )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "Encoder module must be built before calling backward." );
            }

            if ( !this->isTraining() )
            {
                throw std::runtime_error( "Encoder module must be in training mode to call backward. Call setTraining(true) first." );
            }

            if ( !wte_grad_ || !wpe_grad_ )
            {
                throw std::runtime_error( "Encoder module gradients not initialized. This is a bug." );
            }

            if ( !operation_ )
            {
                throw std::runtime_error( "Encoder: operation backend not initialized" );
            }

            if ( !owned_input_grad_ )
            {
                throw std::runtime_error( "Encoder: owned input-grad buffer not allocated" );
            }

            operation_->backward( input, output_grad, *owned_input_grad_ );

            return *owned_input_grad_;
        }

        void zeroGradients() override
        {
            if ( wte_grad_ )
            {
                zero( *wte_grad_, this->getExecutionContext() );
            }

            if ( wpe_grad_ )
            {
                zero( *wpe_grad_, this->getExecutionContext() );
            }
        }

        // ====================================================================
        // Serialization
        // ====================================================================

        void save_( ModelArchive& archive, SerializationMode mode ) const override
        {
            // Persist parameters if present
            (void)archive;
            (void)mode;
        }

        // ====================================================================
        // Parameters and Gradients
        // ====================================================================
        
        size_t parameterCount() const override
        {
            size_t count = 0;

            if ( wte_ )
                count += wte_->size();

            if ( wpe_ )
                count += wpe_->size();

            return count;
        }

        std::vector<ITensor*> getParameters() const override
        {
            std::vector<ITensor*> params;

            if ( wte_ )
                params.push_back( wte_.get() );

            if ( wpe_ )
                params.push_back( wpe_.get() );

            return params;
        }

        std::vector<ITensor*> getGradients() const override
        {
            if ( !this->isTraining() )
            {
                throw std::runtime_error( "Encoder: getGradients called when not in training mode" );
            }

            std::vector<ITensor*> grads;

            if ( wte_grad_ )
                grads.push_back( wte_grad_.get() );

            if ( wpe_grad_ )
                grads.push_back( wpe_grad_.get() );

            return grads;
        }

        EmbeddingsTensorType* getWteGrad() const noexcept
        {
            return wte_grad_.get();
        }

        EmbeddingsTensorType* getWpeGrad() const noexcept
        {
            return wpe_grad_.get();
        }

        // ====================================================================
        // Component interface
        // ====================================================================

        DeviceId getDeviceId() const override
        {
            return this->getExecutionContext()->getDeviceId();
        }

        void synchronize() override
        {
            this->getExecutionContext()->synchronize();
        }

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "--------------------" << std::endl;
            oss << "Encoder: " << this->getName() << std::endl;
            oss << "Vocabulary: " << config_.getVocabularyLength() << " tokens" << std::endl;
            oss << "Max sequence length: " << config_.getMaxSequenceLength() << std::endl;
            oss << "Embedding dimension: " << config_.getChannels() << std::endl;
            oss << "Device: " << deviceTypeToString( this->getDeviceType() ) << std::endl;
            oss << "Parameter count: " << parameterCount() << std::endl;

            return oss.str();
        }

        int64_t getVocabularyLength() const noexcept
        {
            return config_.getVocabularyLength();
        }

        int64_t getMaxSequenceLength() const noexcept
        {
            return config_.getMaxSequenceLength();
        }

        int64_t getChannels() const noexcept
        {
            return config_.getChannels();
        }

    protected:

        // ====================================================================
        // Lifecycle
        // ====================================================================

        /**
         * @brief Called after ExecutionContext is set on the base Component.
         *
         * Initialize device-bound parameters and create the backend operation.
         */
        void onExecutionContextSet() override
        {
            initializeParameters();

            createOperation();
        }

        void onBuilding( const shape_t& input_shape ) override
        {
            validateInputShape( input_shape );

            operation_->setParameters( wte_.get(), wpe_.get() );

            if ( this->isTraining() )
            {
                initializeParameterGradients();
                operation_->setGradients( wte_grad_.get(), wpe_grad_.get() );
            }

            operation_->build( input_shape );

            // Allocate and cache component-owned output and input-grad tensors.
            auto device = this->getExecutionContext()->getDeviceId();

            shape_t out_shape = input_shape;
            out_shape.push_back( config_.getChannels() );
            out_shape[ 2 ] = config_.getChannels();

            owned_output_ = std::make_unique<EmbeddingsTensorType>( device, out_shape );
            owned_output_->setName( this->getName() + ".output" );

            owned_input_grad_ = std::make_unique<TokenIndexType>( device, input_shape );
            owned_input_grad_->setName( this->getName() + ".input.grad" );
        }

        void onTrainingChanging( bool is_training ) override
        {
            operation_->setTraining( is_training );

            if ( is_training )
            {
                if ( this->isBuilt() )
                {
                    initializeParameterGradients();
                    operation_->setGradients( wte_grad_.get(), wpe_grad_.get() );
                }
            }
            else
            {
                operation_->clearGradients();

                if ( wte_grad_ )
                    zeros( *wte_grad_ );
                
                if ( wpe_grad_ )
                    zeros( *wpe_grad_ );
            }
        }

    private:
        EncoderConfig config_;

        std::unique_ptr<EmbeddingsTensorType> wte_{ nullptr };  // Token embeddings (V, C)
        std::unique_ptr<EmbeddingsTensorType> wpe_{ nullptr };  // Position embeddings (maxT, C)

        std::unique_ptr<EmbeddingsTensorType> wte_grad_{ nullptr };
        std::unique_ptr<EmbeddingsTensorType> wpe_grad_{ nullptr };

        std::unique_ptr<TokenIndexType> owned_input_grad_{ nullptr };
        std::unique_ptr<EmbeddingsTensorType> owned_output_{ nullptr };

        std::shared_ptr<UnaryOperation<TDeviceType, TIndex, TPrecision>> operation_{ nullptr };
        std::unique_ptr<IExecutionContext> owned_exec_context_{ nullptr };

        void validateInputShape( const TokenIndexType& input ) const
        {
            const auto& input_shape = input.shape();
            validateInputShape( input_shape );
        }

        void validateInputShape( const shape_t& input_shape ) const
        {
            if ( input_shape.size() != 2 )
            {
                throw std::invalid_argument( "Encoder: input must have rank 2 (batch_size, sequence_length)" );
            }

            int64_t seq_length = input_shape[1];

            if ( seq_length > config_.getMaxSequenceLength() )
            {
                std::ostringstream oss;
                oss << "Encoder: sequence length " << seq_length
                    << " exceeds maximum " << config_.getMaxSequenceLength();
                throw std::invalid_argument( oss.str() );
            }
        }

        void initializeParameterGradients()
        {
            auto device = this->getExecutionContext()->getDeviceId();

            if ( !wte_grad_ )
            {
                wte_grad_ = std::make_unique<EmbeddingsTensorType>( device, wte_->shape() );
                wte_grad_->setName( this->getName() + ".wte.grad" );
                zero( *wte_grad_ );
            }

            if ( !wpe_grad_ )
            {
                wpe_grad_ = std::make_unique<EmbeddingsTensorType>( device, wpe_->shape() );
                wpe_grad_->setName( this->getName() + ".wpe.grad" );
                zero( *wpe_grad_ );
            }
        }

        void initializeParameters()
        {
            int64_t vocab_size = config_.getVocabularyLength();
            int64_t max_seq_len = config_.getMaxSequenceLength();
            int64_t embedding_dim = config_.getChannels();

            // Standard deviation = 1/sqrt(embedding_dim)
            float std = 1.0f / std::sqrt( static_cast<float>(embedding_dim) );

            auto device_id = this->getExecutionContext()->getDeviceId();

            wte_ = std::make_unique<EmbeddingsTensorType>( device_id, shape_t{ vocab_size, embedding_dim } );
            wte_->setName( this->getName() + ".wte" );
            normal<TPrecision, MR>( *wte_, 0.0f, std );

            wpe_ = std::make_unique<EmbeddingsTensorType>( device_id, shape_t{ max_seq_len, embedding_dim } );
            wpe_->setName( this->getName() + ".wpe" );
            normal<TPrecision, MR>( *wpe_, 0.0f, std );
        }

        void createOperation()
        {
            operation_ = OperationRegistry::instance()
                .createUnaryOperation<TDeviceType, TIndex, TPrecision>(
                    "EncoderOp",
                    this->getExecutionContext(),
                    config_ );

            if ( !operation_ )
            {
                throw std::runtime_error( "Failed to create Encoder compute backend operation." );
            }
        }
    };
}