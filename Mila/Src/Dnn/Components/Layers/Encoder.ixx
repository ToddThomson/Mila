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
     * @tparam TPrecision Abstract tensor precision (TensorDataType) for embeddings
     * @tparam TTargets Data type for token indices (typically INT32)
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
        // Compute operation dispatch
        // ====================================================================

        /**
         * @brief Forward pass - delegates to backend operation.
         *
         * Transforms input token IDs into embeddings:
         * 1. Looks up token embeddings from wte table
         * 2. Adds positional embeddings from wpe table
         *
         * @param input Input tensor containing token IDs [B, T]
         * @param output Output tensor for embeddings [B, T, C]
         */
        void forward( const ITensor& input, ITensor& output )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "Encoder module must be built before calling forward." );
            }

            validateInputShape( input );

            operation_->forward( input, output );
        }

        /**
         * @brief Backward pass - delegates to backend operation.
         *
         * Computes gradients with respect to embedding parameters.
         * Token indices are discrete (non-differentiable), so no input gradients.
         */
        void backward( const ITensor& input, const ITensor& output_grad )
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

            auto device = this->getExecutionContext()->getDeviceId();
            auto input_shape = input.shape();
            TokenIndexType input_grad_dummy( device, input_shape );

            operation_->backward( input, output_grad, input_grad_dummy );
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

        std::shared_ptr<EmbeddingsTensorType> getWteGrad() const noexcept
        {
            return wte_grad_;
        }

        std::shared_ptr<EmbeddingsTensorType> getWpeGrad() const noexcept
        {
            return wpe_grad_;
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

        // ====================================================================
        // Parameter accessors
        // ====================================================================

        /*std::shared_ptr<EmbeddingsTensorType> getTokenEmbedding() const noexcept
        {
            return wte_;
        }

        std::shared_ptr<EmbeddingsTensorType> getPositionalEmbedding() const noexcept
        {
            return wpe_;
        }*/

        /*const EncoderConfig& getConfig() const noexcept
        {
            return config_;
        }*/

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

            //operation_->setTraining( this->isTraining() );

            operation_->setParameters( wte_.get(), wpe_.get() );

            // BUG: isTraining() is always false here during build
            //if ( this->isTraining() )
            //{
            //    initializeParameterGradients();
            //    operation_->setGradients( wte_grad_.get(), wpe_grad_.get() );
            //}

            operation_->build( input_shape );
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

        std::shared_ptr<EmbeddingsTensorType> wte_{ nullptr };  // Token embeddings (V, C)
        std::shared_ptr<EmbeddingsTensorType> wpe_{ nullptr };  // Position embeddings (maxT, C)

        std::shared_ptr<EmbeddingsTensorType> wte_grad_{ nullptr };
        std::shared_ptr<EmbeddingsTensorType> wpe_grad_{ nullptr };

        std::shared_ptr<UnaryOperation<TDeviceType, TIndex, TPrecision>> operation_{ nullptr };
        std::unique_ptr<IExecutionContext> owned_exec_context_{ nullptr };

        void validateInputShape( const ITensor& input ) const
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
                wte_grad_ = std::make_shared<EmbeddingsTensorType>( device, wte_->shape() );
                wte_grad_->setName( this->getName() + ".wte.grad" );
                zero( *wte_grad_ );
            }

            if ( !wpe_grad_ )
            {
                wpe_grad_ = std::make_shared<EmbeddingsTensorType>( device, wpe_->shape() );
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

            wte_ = std::make_shared<EmbeddingsTensorType>( device_id, shape_t{ vocab_size, embedding_dim } );
            wte_->setName( this->getName() + ".wte" );
            normal<TPrecision, MR>( *wte_, 0.0f, std );

            wpe_ = std::make_shared<EmbeddingsTensorType>( device_id, shape_t{ max_seq_len, embedding_dim } );
            wpe_->setName( this->getName() + ".wpe" );
            normal<TPrecision, MR>( *wte_, 0.0f, std );
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