/**
 * @file Lpe.ixx
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
#include <cmath>
#include <numeric>
#include <algorithm>

export module Dnn.Components.Lpe;
export import Dnn.Components.LpeConfig;

import Dnn.Component;
import Dnn.ComponentType;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Compute.Device;
import Compute.DeviceId;
import Compute.DeviceType;
import Compute.DeviceTypeTraits;
import Compute.ExecutionContext;
import Compute.ExecutionContextFactory;
import Compute.OperationTraits;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.IPositionalDecode;
import Compute.Observation;
import Serialization.ModelArchive;
import Serialization.Metadata;
import Serialization.Mode;
import Serialization.Tensor;
import Serialization.SafeTensors;

// DEBUG:
import Dnn.TensorOps;
import Dnn.TensorHelpers;
import Logging.Logger;

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
                    throw std::invalid_argument( "Lpe: device type mismatch" );
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

            operation_->forward( input, *output_ );

            // Return view with actual output shape
            shape_t actual_out_shape = { B, T, config_.getEmbeddingDim() };
            current_output_view_ = std::make_unique<EmbeddingsTensorType>( output_->view( actual_out_shape ) );

            this->publish( ComputePass::Forward, "output", *current_output_view_ );

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

            if ( !this->isTrainingMode() )
            {
                throw std::runtime_error( "Encoder module must be in training mode to call backward." );
            }

            // REVIEW: The following checks are not required. If built and in training mode,
            // these buffers should always be initialized in onBuilding. If not, it's a bug.

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

            // Zero the input gradient buffer -- and here that is not hygiene before an
            // accumulating op, it is the only thing that gives the buffer a value. Lpe's
            // input is token indices, which are non-differentiable: CudaLpeOp::backward
            // documents input_grad as "Unused" and never writes it. Without this zero the
            // buffer returned below would hold uninitialized memory.
            //
            // The atomicAdd accumulation in the Lpe kernels targets wte_grad_/wpe_grad_,
            // the PARAMETER gradients, which is a separate contract: those accumulate
            // across backward calls and are cleared by zeroGradients() between optimizer
            // steps.
            zero( *input_grad_ /*, this->getExecutionContext() */);

            operation_->backward( input, output_grad, *input_grad_ );

            return *input_grad_;
        }

        /**
         * @brief Decode pass - single token embedding at a specific sequence position.
         *
         * Unlike forward() which processes a full sequence [B, T] and uses positions
         * 0..T-1, decode() processes a single token and uses the caller-supplied
         * position for the positional embedding lookup. This is critical for
         * correctness in KV cache autoregressive generation -- without the correct
         * position, wpe[0] would be used for every generated token, corrupting
         * all subsequent attention computations.
         *
         * @param input   Single token index tensor [1, 1]
         * @param position Actual sequence position (prefill_len + decode_step)
         * @return Reference to component-owned embedding tensor [1, 1, C]
         *
         * @throws std::runtime_error if component is not built.
         */
        EmbeddingsTensorType& decode( const TokenIndexType& input, dim_t position )
        {
            if ( !this->isBuilt() )
                throw std::runtime_error( "Lpe must be built before calling decode()." );

            // Resolved IPositionalDecode from onBuilding
            if ( !decode_path_ )
                throw std::runtime_error( "Lpe: backend operation does not support decode() -- IPositionalDecode not implemented" );

            decode_path_->decode( input, *output_, position );

            // Single token output shape [1, 1, C]
            shape_t decode_out_shape = { 1, 1, config_.getEmbeddingDim() };
            current_output_view_ = std::make_unique<EmbeddingsTensorType>(
                output_->view( decode_out_shape ) );

            this->publish( ComputePass::Decode, "output", *current_output_view_ );

            return *current_output_view_;
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

        std::vector<std::string> getParameterNames() const override
        {
            return { "wte", "wpe" };
        }

        void saveFlatTensors(
            Serialization::SafeTensorsWriter& writer,
            const std::string& prefix,
            Serialization::TensorSavePass pass ) const override
        {
            if ( wte_ )
            {
                this->saveParameterToWriter( writer, prefix + ".wte", *wte_, pass );
            }

            if ( wpe_ )
            {
                this->saveParameterToWriter( writer, prefix + ".wpe", *wpe_, pass );
            }
        }

        void save_( ModelArchive& archive, SerializationMode mode ) const override
        {
            (void)mode;

            SerializationMetadata meta;
            meta.set( "type", "Lpe" )
                .set( "version", int64_t( 1 ) )
                .set( "name", this->getName() );

            archive.writeMetadata( "meta.json", meta );

            if ( wte_ )
            {
                this->saveParameterToArchive( archive, "wte", *wte_ );
            }

            if ( wpe_ )
            {
                this->saveParameterToArchive( archive, "wpe", *wpe_ );
            }
        }

        // ====================================================================
        // Parameters and Gradients
        // ====================================================================
        
        dim_t parameterCount() const override
        {
            dim_t count = 0;

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

        void loadParameter( const std::string& name, const ITensorBlob& blob ) override
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
        }

        std::vector<ITensor*> getGradients() const override
        {

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

        std::vector<const ITensor*> getOutputs() const override
        {
            if ( output_ == nullptr )
            {
                return {};
            }

            return { output_.get() };
        }

        std::vector<ObservableStage> getObservableStages() const override
        {
            return { { "output", ComputePassMask{ ComputePass::Forward, ComputePass::Decode } } };
        }

        MemoryStats getMemoryStats() const override
        {
            MemoryStats stats;

            if ( wte_ != nullptr )
            {
                stats.device_parameter_bytes += wte_->getStorageSize();
            }

            if ( wpe_ != nullptr )
            {
                stats.device_parameter_bytes += wpe_->getStorageSize();
            }

            if ( output_ != nullptr )
            {
                stats.device_state_bytes += output_->getStorageSize();
            }

            if ( wte_grad_ != nullptr )
            {
                stats.device_gradient_bytes += wte_grad_->getStorageSize();
            }

            if ( wpe_grad_ != nullptr )
            {
                stats.device_gradient_bytes += wpe_grad_->getStorageSize();
            }

            if ( input_grad_ != nullptr )
            {
                stats.device_gradient_bytes += input_grad_->getStorageSize();
            }

            return stats;
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

        void onBuilding( const BuildContext& build_config ) override
        {
            const auto& input_shape = build_config.inputShape();
            validateInputShape( input_shape );

            // Resolve IPositionalDecode once at build time. May be nullptr for some backends
            decode_path_ = dynamic_cast<IPositionalDecode*>(operation_.get());

            // Store MAX dimensions for dynamic input validation in forward/backward 
            // (batch size can vary, but sequence length must be <= max)
            max_batch_size_ = input_shape[ 0 ];
            max_seq_len_ = input_shape[ 1 ];

            operation_->setParameters( wte_.get(), wpe_.get() );
            operation_->build( build_config );

            // Positional encodings are initialized only for train-from-scratch; the
            // pretrained load path overwrites wte/wpe immediately after build().
            if ( build_config.shouldInitializeParameters() )
            {
                const float std_dev = 1.0f / std::sqrt( static_cast<float>( config_.getEmbeddingDim() ) );
                fill_normal( *wte_, 0.0f, std_dev, this->getExecutionContext() );
                fill_normal( *wpe_, 0.0f, std_dev, this->getExecutionContext() );
            }

            // Allocate and cache component-owned output and input-grad tensors.
            auto device = this->getExecutionContext()->getDeviceId();
            shape_t max_out_shape = { max_batch_size_, max_seq_len_, config_.getEmbeddingDim() };

            output_ = std::make_unique<EmbeddingsTensorType>( device, max_out_shape, this->getName() + ".output" );

            if ( build_config.isTrainingMode() )
            {
                initializeParameterGradients();
                operation_->setGradients( wte_grad_.get(), wpe_grad_.get() );

                input_grad_ = std::make_unique<TokenIndexType>( device, input_shape, this->getName() + ".input.grad" );
            }
        }

        void onTrainingModeChanging( TrainingMode training_mode ) override
        {
            operation_->setTrainingMode( training_mode );

            if ( training_mode == TrainingMode::Normal )
            {
                // REVIEW: Must already be built! Impossible to not be built already.
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

        using OpType = typename OperationTraits<OperationType::LpeOp, TDeviceType, TPrecision>::type;

        std::shared_ptr<OpType> operation_{ nullptr };
        IPositionalDecode* decode_path_{ nullptr };  // non-owning, resolved at build time.
        
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

            auto device_id = this->getExecutionContext()->getDeviceId();

            wte_ = std::make_unique<EmbeddingsTensorType>( device_id, shape_t{ vocab_size, embedding_dim } );
            wte_->setName( this->getName() + ".wte" );

            wpe_ = std::make_unique<EmbeddingsTensorType>( device_id, shape_t{ max_seq_len, embedding_dim } );
            wpe_->setName( this->getName() + ".wpe" );
        }

        void createOperation()
        {
            operation_ = std::make_shared<OpType>( this->getExecutionContext(), config_ );

            if ( !operation_ )
            {
                throw std::runtime_error( "Failed to create Lpe compute backend operation." );
            }
        }
    };
}
