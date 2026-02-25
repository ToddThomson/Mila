/**
 * @file Gpt2Encoder.ixx
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
#include <numeric>
#include <algorithm>

export module Dnn.Components.Lpe;
export import :Config;

import Dnn.Component;
import Dnn.ComponentType;
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

// DEBUG:
import Dnn.TensorOps;
import Dnn.TensorHelpers;
import Utils.Logger;

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
    class Lpe : public Component<TDeviceType, TPrecision>
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
        explicit Lpe( const std::string& name, const LpeConfig& config, std::optional<DeviceId> device_id = std::nullopt )
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

        ~Lpe() override = default;

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

            // Get dimensions from input
            auto actual_shape = input.shape();
            int64_t B = actual_shape[ 0 ];
            int64_t T = actual_shape[ 1 ];

            // REVIEW: Validation() only in backend operation?
            // Validate actual dimensions fit within max
            if ( B > max_batch_size_ || T > max_seq_len_ )
            {
                throw std::runtime_error( std::format(
                    "Lpe: input shape [{}, {}] exceeds built max [{}, {}]",
                    B, T, max_batch_size_, max_seq_len_ ) );
            }

            // DEBUG: Check input range
            auto host_input = toHost<TensorDataType::FP32>( input );
            auto host_input_ptr = host_input.data();
            const size_t n = host_input.size();
            auto [min_in, max_in] = std::minmax_element( host_input_ptr, host_input_ptr + n );
            Utils::Logger::debug( std::format( "Lpe {} in:[{:.3f}, {:.3f}] with shape:{}",
                this->getName(), *min_in, *max_in, shapeToString( input.shape() ) ) );
            // END DEBUG:

            operation_->forward( input, *output_ );

            // Return view with actual output shape
            shape_t actual_out_shape = { B, T, static_cast<dim_t>( config_.getEmbeddingDim() ) };
            current_output_view_ = std::make_unique<EmbeddingsTensorType>( output_->view( actual_out_shape ) );

            // DEBUG: Check output range
            this->synchronize();

            auto host_output = toHost<TensorDataType::FP32>( *current_output_view_ );
            auto host_output_ptr = host_output.data();
            const size_t output_n = host_output.size();
            auto [min_out, max_out] = std::minmax_element( host_output_ptr, host_output_ptr + output_n );

            Utils::Logger::debug( std::format( "Lpe {} out:[{:.3f}, {:.3f}] with shape:{}",
                this->getName(), *min_out, *max_out, shapeToString( host_output.shape() ) ) );
            // DEBUG END

            return *current_output_view_;
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

            if ( !input_grad_ )
            {
                throw std::runtime_error( "Encoder: owned input-grad buffer not allocated" );
            }

            // Zero input gradient buffer before backward pass. No exeptions.
            // Backend ops use accumulation (atomicAdd/+=) which requires pre-zeroed buffers
            // to prevent gradient buildup across calls. Without this, gradients grow linearly
            // with each call -> explosion.
            zero( *input_grad_ /*, this->getExecutionContext() */);

            operation_->backward( input, output_grad, *input_grad_ );

            return *input_grad_;
        }

        void zeroGradients() override
        {
            if ( wte_grad_ )
            {
                zero( *wte_grad_ /*, this->getExecutionContext() */);
            }

            if ( wpe_grad_ )
            {
                zero( *wpe_grad_ /*, this->getExecutionContext() */);
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

        void loadParameter( const std::string& name, const TensorBlob& blob ) override
        {
            if ( name == "wte" )
            {
                this->loadParameterFromBlob( "wte", blob, *wte_, wte_->shape() );
            }
            else if ( name == "wpe" )
            {
                this->loadParameterFromBlob( "wpe", blob, *wpe_, wpe_->shape() );
            }
            else
            {
                // Throw by default for unknown parameter names
                this->loadParameter( name, blob ); 
            }

            // DEBUG: Diagnostics

            if ( name == "wte" )
            {
                // DIAGNOSTIC: Check weight statistics
                auto host_wte = toHost<TensorDataType::FP32>( *wte_ );

                const float* ptr = host_wte.data();
                const size_t n = host_wte.size();

                if ( n > 0 )
                {
                    float min_w = *std::min_element( ptr, ptr + n );
                    float max_w = *std::max_element( ptr, ptr + n );
                    float mean_w = std::accumulate( ptr, ptr + n, 0.0f ) / static_cast<float>(n);

                    Utils::Logger::info( std::format(
                        "Lpe wte stats: min={:.6f} max={:.6f} mean={:.6f}",
                        min_w, max_w, mean_w ) );
                }

            }

            if ( name == "wpe" )
            {
                auto host_wpe = toHost<TensorDataType::FP32>( *wpe_ );
                const float* ptr = host_wpe.data();
                const size_t n = host_wpe.size();

                if ( n > 0 )
                {
                    float min_b = *std::min_element( ptr, ptr + n );
                    float max_b = *std::max_element( ptr, ptr + n );
                    float mean_w = std::accumulate( ptr, ptr + n, 0.0f ) / static_cast<float>(n);

                    Utils::Logger::info( std::format(
                        "Lpe wpe stats: min={:.6f} max={:.6f} mean={:.6f}",
                        min_b, max_b, mean_w ) );
                }
            }

            // END DEBUG:
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

        const ComponentType getType() const override
        {
            return ComponentType::Lpe;
        }

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
            oss << "Embedding dimension: " << config_.getEmbeddingDim() << std::endl;
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

        int64_t getEmbeddingDim() const noexcept
        {
            return config_.getEmbeddingDim();
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

            // Store MAX dimensions for dynamic input validation in forward/backward 
            // (batch size can vary, but sequence length must be <= max)
            max_batch_size_ = input_shape[ 0 ];
            max_seq_len_ = input_shape[ 1 ];

            operation_->setParameters( wte_.get(), wpe_.get() );

            if ( this->isTraining() )
            {
                initializeParameterGradients();
                operation_->setGradients( wte_grad_.get(), wpe_grad_.get() );
            }

            operation_->build( input_shape );

            // Allocate and cache component-owned output and input-grad tensors.
            auto device = this->getExecutionContext()->getDeviceId();
            shape_t max_out_shape = { max_batch_size_, max_seq_len_, static_cast<dim_t>( config_.getEmbeddingDim() ) };

            output_ = std::make_unique<EmbeddingsTensorType>( device, max_out_shape );
            output_->setName( this->getName() + ".output" );

            input_grad_ = std::make_unique<TokenIndexType>( device, input_shape );
            input_grad_->setName( this->getName() + ".input.grad" );
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
                    zero( *wte_grad_ );
                
                if ( wpe_grad_ )
                    zero( *wpe_grad_ );
            }
        }

    private:
        LpeConfig config_;

        int64_t max_batch_size_{ 0 };
        int64_t max_seq_len_{ 0 };

        std::unique_ptr<EmbeddingsTensorType> wte_{ nullptr };  // Token embeddings (V, C)
        std::unique_ptr<EmbeddingsTensorType> wpe_{ nullptr };  // Position embeddings (maxT, C)

        std::unique_ptr<EmbeddingsTensorType> wte_grad_{ nullptr };
        std::unique_ptr<EmbeddingsTensorType> wpe_grad_{ nullptr };

        std::unique_ptr<TokenIndexType> input_grad_{ nullptr };
        std::unique_ptr<EmbeddingsTensorType> output_{ nullptr };
        std::unique_ptr<EmbeddingsTensorType> current_output_view_{ nullptr };

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
            int64_t embedding_dim = config_.getEmbeddingDim();

            // Standard deviation = 1/sqrt(embedding_dim)
            float std_dev = 1.0f / std::sqrt( static_cast<float>(embedding_dim) );

            auto device_id = this->getExecutionContext()->getDeviceId();

            wte_ = std::make_unique<EmbeddingsTensorType>( device_id, shape_t{ vocab_size, embedding_dim } );
            wte_->setName( this->getName() + ".wte" );
            normal( *wte_, std_dev );

            wpe_ = std::make_unique<EmbeddingsTensorType>( device_id, shape_t{ max_seq_len, embedding_dim } );
            wpe_->setName( this->getName() + ".wpe" );
            normal( *wpe_, std_dev );
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