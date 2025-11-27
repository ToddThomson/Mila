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

export module Dnn.Components.Encoder;
export import :Config;

import Dnn.Component;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Compute.Precision;
import Compute.ComputeDevice;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.UnaryOperation;
import Compute.OperationRegistry;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.CudaDeviceMemoryResource;
import Serialization.ModelArchive;

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
     * @tparam TDeviceType Device type (DeviceType::Cpu or DeviceType::Cuda)
     * @tparam TPrecision Abstract tensor precision (TensorDataType) for embeddings
     * @tparam TTargets Data type for token indices (typically INT32)
     */
    export template<DeviceType TDeviceType, TensorDataType TIndex = dtype_t::INT32, TensorDataType TPrecision = dtype_t::FP32>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class Encoder : public Component<TDeviceType, TPrecision>
    {
    public:
        using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaDeviceMemoryResource, CpuMemoryResource>;
        using ExecutionContextType = ExecutionContext<TDeviceType>;
        using EmbeddingsTensorType = Tensor<TPrecision, MR>;
        using TokenIndexType = Tensor<TIndex, MR>;

        /**
         * @brief Construct with an existing execution context.
         *
         * @param exec_context Shared execution context for device resources.
         * @param config Encoder configuration.
         */
        explicit Encoder( std::shared_ptr<ExecutionContextType> exec_context, const EncoderConfig& config )
            : exec_context_( exec_context ), config_( config )
        {
            if (!exec_context_)
            {
                throw std::invalid_argument( "ExecutionContext cannot be null." );
            }

            config_.validate();

            initializeParameters();
            createOperation();
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
            if (!this->isBuilt())
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

            // Ensure gradients are initialized (defensive check)
            if ( !wte_grad_ || !wpe_grad_ )
            {
                throw std::runtime_error( "Encoder module gradients not initialized. This is a bug." );
            }

            // Create dummy input gradient (token IDs are non-differentiable)
            auto device = exec_context_->getDevice();
            auto input_shape = input.shape();
            TokenIndexType input_grad_dummy( device, input_shape );

            operation_->backward( input, output_grad, input_grad_dummy );
        }

        // ====================================================================
        // Serialization
        // ====================================================================

        void save_( ModelArchive& archive, SerializationMode mode ) const override
        {
            // Persist parameters if present
            if (wte_)
            {
                // archive.saveTensor( this->getName() + ".wte", *wte_ );
            }

            if (wpe_)
            {
                // archive.saveTensor( this->getName() + ".wpe", *wpe_ );
            }
        }

        // ====================================================================
        // Parameters and Gradients
        // ====================================================================
        
        size_t parameterCount() const override
        {
            size_t count = 0;

            if (wte_)
                count += wte_->size();

            if (wpe_)
                count += wpe_->size();

            return count;
        }

        std::vector<ITensor*> getParameters() const override
        {
            std::vector<ITensor*> params;

            if (wte_)
                params.push_back( wte_.get() );

            if (wpe_)
                params.push_back( wpe_.get() );

            return params;
        }

        std::vector<ITensor*> getGradients() const override
        {
            if (!this->isTraining())
            {
                throw std::runtime_error( "Encoder: getGradients called when not in training mode" );
            }

            std::vector<ITensor*> grads;

            if (wte_grad_)
                grads.push_back( wte_grad_.get() );

            if (wpe_grad_)
                grads.push_back( wpe_grad_.get() );

            return grads;
        }

        /**
         * @brief Get token embedding gradient tensor.
         *
         * @return Shared pointer to wte gradient, or nullptr if not in training mode
         */
        std::shared_ptr<EmbeddingsTensorType> getWteGrad() const noexcept
        {
            return wte_grad_;
        }

        /**
         * @brief Get positional embedding gradient tensor.
         *
         * @return Shared pointer to wpe gradient, or nullptr if not in training mode
         */
        std::shared_ptr<EmbeddingsTensorType> getWpeGrad() const noexcept
        {
            return wpe_grad_;
        }

        // ====================================================================
        // Module interface
        // ====================================================================

        std::string getName() const override
        {
            return config_.getName();
        }

        std::shared_ptr<ComputeDevice> getDevice() const override
        {
            return exec_context_->getDevice();
        }

        void synchronize() override
        {
            exec_context_->synchronize();
        }

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "--------------------" << std::endl;
            oss << "Encoder: " << getName() << std::endl;
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

        /**
         * @brief Return shared ownership of the token embedding tensor (wte).
         *
         * @returns Shared pointer to the wte tensor.
         */
        std::shared_ptr<EmbeddingsTensorType> getTokenEmbedding() const noexcept
        {
            return wte_;
        }

        /**
         * @brief Return shared ownership of the positional embedding tensor (wpe).
         *
         * @returns Shared pointer to the wpe tensor.
         */
        std::shared_ptr<EmbeddingsTensorType> getPositionalEmbedding() const noexcept
        {
            return wpe_;
        }

        /**
         * @brief Get the configuration.
         *
         * @returns Reference to the EncoderConfig.
         */
        const EncoderConfig& getConfig() const noexcept
        {
            return config_;
        }

        /**
         * @brief Gets the vocabulary length.
         *
         * @return int64_t The vocabulary length (V).
         */
        int64_t getVocabularyLength() const noexcept
        {
            return config_.getVocabularyLength();
        }

        /**
         * @brief Gets the maximum sequence length.
         *
         * @return int64_t The maximum sequence length (maxT).
         */
        int64_t getMaxSequenceLength() const noexcept
        {
            return config_.getMaxSequenceLength();
        }

        /**
         * @brief Gets the embedding dimension (channels).
         *
         * @return int64_t The number of channels (C).
         */
        int64_t getChannels() const noexcept
        {
            return config_.getChannels();
        }

    protected:

        // ====================================================================
// Lifecycle
// ====================================================================


        /**
         * @brief Build the module using an input shape.
         *
         * Encoder parameters are eagerly created in the constructor based on
         * configuration (vocab_size, max_seq_len, embedding_dim). This method
         * binds parameters to the backend operation and triggers backend-specific setup.
         *
         * If in training mode, also initializes gradient tensors and binds them
         * to the operation.
         *
         * @param input_shape Expected shape: (batch_size, sequence_length)
         */
        void onBuilding( const shape_t& input_shape ) override
        {
            validateInputShape( input_shape );

            operation_->setTraining( this->isTraining() );

            // Bind forward parameters to operation
            operation_->setParameters( wte_.get(), wpe_.get() );

            // If training mode, initialize gradients and bind to operation
            if ( this->isTraining() )
            {
                initializeParameterGradients();
                operation_->setGradients( wte_grad_.get(), wpe_grad_.get() );
            }

            operation_->build( input_shape );
        }

        /**
         * @brief Hook invoked when training mode is about to change.
         *
         * Propagate training mode to the backend operation. When enabling
         * training after the module is built, allocate and bind gradient
         * tensors. When disabling training, unbind gradients and free buffers
         * to release memory.
         *
         * Called with Module's training mutex held; do not call setTraining() here.
         */
        void onTrainingChanging( bool is_training ) override
        {
            operation_->setTraining( is_training );

            if ( is_training )
            {
                // Entering training: if already built ensure gradients allocated and bound
                if (this->isBuilt())
                {
                    initializeParameterGradients();
                    operation_->setGradients( wte_grad_.get(), wpe_grad_.get() );
                }
            }
            else
            {
                // Leaving training: unbind and free gradients
                operation_->clearGradients();

                wte_grad_.reset();
                wpe_grad_.reset();
            }
        }

    private:
        EncoderConfig config_;

        std::shared_ptr<EmbeddingsTensorType> wte_{ nullptr };  // Token embeddings (V, C)
        std::shared_ptr<EmbeddingsTensorType> wpe_{ nullptr };  // Position embeddings (maxT, C)

        std::shared_ptr<EmbeddingsTensorType> wte_grad_{ nullptr };
        std::shared_ptr<EmbeddingsTensorType> wpe_grad_{ nullptr };

        std::shared_ptr<UnaryOperation<TDeviceType, TIndex, TPrecision>> operation_{ nullptr };
        std::shared_ptr<ExecutionContextType> exec_context_;

        /**
         * @brief Validate input shape for encoder operation.
         *
         * Ensures input is rank-2 (batch_size, sequence_length) and sequence
         * length doesn't exceed configured maximum.
         */
        void validateInputShape( const ITensor& input ) const
        {
            const auto& input_shape = input.shape();
            validateInputShape( input_shape );
        }

        /**
         * @brief Validate input shape for encoder operation.
         */
        void validateInputShape( const shape_t& input_shape ) const
        {
            if (input_shape.size() != 2)
            {
                throw std::invalid_argument( "Encoder: input must have rank 2 (batch_size, sequence_length)" );
            }

            int64_t seq_length = input_shape[1];

            if (seq_length > config_.getMaxSequenceLength())
            {
                std::ostringstream oss;
                oss << "Encoder: sequence length " << seq_length
                    << " exceeds maximum " << config_.getMaxSequenceLength();
                throw std::invalid_argument( oss.str() );
            }
        }

        /**
         * @brief Ensure gradient tensors are allocated with correct shapes.
         */
        void initializeParameterGradients()
        {
            auto device = exec_context_->getDevice();

            if (!wte_grad_)
            {
                wte_grad_ = std::make_shared<EmbeddingsTensorType>(
                    device,
                    wte_->shape() );
                wte_grad_->setName( this->getName() + ".wte.grad" );
                zeros( *wte_grad_ );
            }

            if (!wpe_grad_)
            {
                wpe_grad_ = std::make_shared<EmbeddingsTensorType>(
                    device,
                    wpe_->shape() );
                wpe_grad_->setName( this->getName() + ".wpe.grad" );
                zeros( *wpe_grad_ );
            }
        }

        /**
         * @brief Allocate and initialize token and positional embedding tensors.
         *
         * Tensors are created on the execution context device and initialized
         * using Xavier initialization for both wte and wpe.
         */
        void initializeParameters()
        {
            int64_t vocab_size = config_.getVocabularyLength();
            int64_t max_seq_len = config_.getMaxSequenceLength();
            int64_t embedding_dim = config_.getChannels();

            auto device = exec_context_->getDevice();

            // Token embeddings: (vocab_size, embedding_dim)
            wte_ = std::make_shared<EmbeddingsTensorType>( device, shape_t{ vocab_size, embedding_dim } );
            wte_->setName( this->getName() + ".wte" );
            xavier<TPrecision, MR>( *wte_, vocab_size, embedding_dim );

            // Positional embeddings: (max_seq_len, embedding_dim)
            wpe_ = std::make_shared<EmbeddingsTensorType>( device, shape_t{ max_seq_len, embedding_dim } );
            wpe_->setName( this->getName() + ".wpe" );
            xavier<TPrecision, MR>( *wpe_, max_seq_len, embedding_dim );
        }

        /**
         * @brief Create the backend compute operation.
         *
         * Looks up the appropriate device-specific operation from the registry
         * and creates an instance bound to this module's execution context.
         */
        void createOperation()
        {
            operation_ = OperationRegistry::instance()
                .createUnaryOperation<TDeviceType, TIndex, TPrecision>(
                    "EncoderOp",
                    exec_context_,
                    config_ );

            if (!operation_)
            {
                throw std::runtime_error( "Failed to create Encoder compute backend operation." );
            }
        }
    };

    // Convenience aliases for common usages
    /*export template<TensorDataType TPrecision, TensorDataType TTargets = dtype_t::INT32>
        using CpuEncoder = Encoder<DeviceType::Cpu, TPrecision, TTargets>;

    export template<TensorDataType TPrecision, TensorDataType TTargets = dtype_t::INT32>
        using CudaEncoder = Encoder<DeviceType::Cuda, TPrecision, TTargets>;*/
}