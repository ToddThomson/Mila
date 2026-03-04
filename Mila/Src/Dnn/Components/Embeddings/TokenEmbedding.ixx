/**
 * @file TokenEmbedding.ixx
 * @brief Device-templated TokenEmbedding component.
 *
 * Pure vocabulary lookup: maps token indices [B, T] to dense vectors [B, T, C].
 * Owns the wte parameter and its gradient. No positional encoding — that is
 * handled downstream by a dedicated encoding component (RoPE, ALiBi, or Learned).
 *
 * Derived from Lpe with all wpe / IPositionalDecode / decode() concerns removed.
 */

module;
#include <memory>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include <optional>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <format>

export module Dnn.Components.TokenEmbedding;
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

import Dnn.TensorOps;
import Dnn.TensorHelpers;
import Utils.Logger;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    /**
     * @brief Pure token embedding component (device-templated).
     *
     * Transforms input token indices into continuous vector representations
     * by looking up each index in the vocabulary embedding table (wte).
     * No positional information is added here.
     *
     * Construction modes:
     * - Standalone: provide a DeviceId to create and own an ExecutionContext.
     * - Child/deferred: omit DeviceId; caller must call setExecutionContext()
     *   before build().
     *
     * @tparam TDeviceType Device type (DeviceType::Cpu or DeviceType::Cuda).
     * @tparam TIndex      Data type for token indices (typically INT32).
     * @tparam TPrecision  Tensor precision for embeddings (FP32 or FP16).
     */
    export template<DeviceType TDeviceType, TensorDataType TIndex = dtype_t::INT32, TensorDataType TPrecision = dtype_t::FP32>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class TokenEmbedding : public Component<TDeviceType, TPrecision>
    {
    public:
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using EmbeddingTensorType = Tensor<TPrecision, MR>;
        using TokenIndexType = Tensor<TIndex, MR>;
        using ComponentBase = Component<TDeviceType, TPrecision>;

        /**
         * @brief Construct a TokenEmbedding component.
         *
         * @param name      Component name identifier.
         * @param config    TokenEmbedding configuration.
         * @param device_id Optional DeviceId for standalone (owned context) mode.
         */
        explicit TokenEmbedding(
            const std::string& name,
            const TokenEmbeddingConfig& config,
            std::optional<DeviceId>     device_id = std::nullopt )
            : ComponentBase( name ), config_( config )
        {
            config_.validate();

            if ( device_id.has_value() )
            {
                if ( device_id->type != TDeviceType )
                    throw std::invalid_argument( "TokenEmbedding: device type mismatch" );

                owned_exec_context_ = createExecutionContext( device_id.value() );
                this->setExecutionContext( owned_exec_context_.get() );
            }
        }

        ~TokenEmbedding() override = default;

        // ====================================================================
        // Forward
        // ====================================================================

        /**
         * @brief Forward pass — returns component-owned embeddings tensor.
         *
         * output[b, t, :] = wte[ X[b, t], : ]
         *
         * Accepts any sequence length T <= max built T, including T=1 for
         * single-token autoregressive steps.
         *
         * @param input Token indices [B, T].
         * @return Reference to component-owned embeddings [B, T, C].
         *
         * @throws std::runtime_error if the component is not built.
         */
        EmbeddingTensorType& forward( const TokenIndexType& input )
        {
            if ( !this->isBuilt() )
                throw std::runtime_error( "TokenEmbedding must be built before calling forward()." );

            const auto& shape = input.shape();
            int64_t B = shape[ 0 ];
            int64_t T = shape[ 1 ];

            if ( B > max_batch_size_ || T > max_seq_len_ )
                throw std::runtime_error( std::format(
                    "TokenEmbedding: input shape [{}, {}] exceeds built max [{}, {}]",
                    B, T, max_batch_size_, max_seq_len_ ) );

            operation_->forward( input, *output_ );

            shape_t actual_out_shape = { B, T, static_cast<dim_t>(config_.getEmbeddingDim()) };
            current_output_view_ = std::make_unique<EmbeddingTensorType>(
                output_->view( actual_out_shape ) );

            return *current_output_view_;
        }

        // ====================================================================
        // Backward
        // ====================================================================

        /**
         * @brief Backward pass — accumulates gradients into wte.
         *
         * Token indices are discrete and non-differentiable; the returned
         * input_grad tensor exists for interface consistency but carries no
         * meaningful gradient.
         *
         * wte_grad buffers use atomicAdd accumulation and must be zeroed before
         * each backward call, which zeroGradients() handles.
         *
         * @param input       Token indices used in forward [B, T].
         * @param output_grad Upstream gradient w.r.t. embeddings [B, T, C].
         * @return Reference to component-owned (unused) input gradient tensor.
         *
         * @throws std::runtime_error if not built, not in training mode, or
         *         buffers are not initialized.
         */
        TokenIndexType& backward(
            const TokenIndexType& input,
            const EmbeddingTensorType& output_grad )
        {
            if ( !this->isBuilt() )
                throw std::runtime_error( "TokenEmbedding must be built before calling backward()." );

            if ( !this->isTraining() )
                throw std::runtime_error( "TokenEmbedding must be in training mode to call backward(). "
                    "Call setTraining(true) first." );

            if ( !wte_grad_ )
                throw std::runtime_error( "TokenEmbedding: wte_grad not initialized. This is a bug." );

            if ( !input_grad_ )
                throw std::runtime_error( "TokenEmbedding: input_grad buffer not allocated. This is a bug." );

            zero( *input_grad_ );

            operation_->backward( input, output_grad, *input_grad_ );

            return *input_grad_;
        }

        // ====================================================================
        // Gradient management
        // ====================================================================

        void zeroGradients() override
        {
            if ( wte_grad_ )
                zero( *wte_grad_ );
        }

        // ====================================================================
        // Serialization
        // ====================================================================

        void save_( ModelArchive& archive, SerializationMode mode ) const override
        {
            (void)archive;
            (void)mode;
        }

        // ====================================================================
        // Parameters and Gradients
        // ====================================================================

        size_t parameterCount() const override
        {
            return wte_ ? wte_->size() : 0;
        }

        std::vector<ITensor*> getParameters() const override
        {
            std::vector<ITensor*> params;
            if ( wte_ ) params.push_back( wte_.get() );
            return params;
        }

        void loadParameter( const std::string& name, const TensorBlob& blob ) override
        {
            if ( name == "wte" )
            {
                this->loadParameterFromBlob( "wte", blob, *wte_, wte_->shape() );

                // Diagnostics
                auto host_wte = toHost<TensorDataType::FP32>( *wte_ );
                const float* ptr = host_wte.data();
                const size_t n = host_wte.size();

                if ( n > 0 )
                {
                    float min_w = *std::min_element( ptr, ptr + n );
                    float max_w = *std::max_element( ptr, ptr + n );
                    float mean_w = std::accumulate( ptr, ptr + n, 0.0f ) / static_cast<float>(n);

                    Utils::Logger::info( std::format(
                        "TokenEmbedding wte stats: min={:.6f} max={:.6f} mean={:.6f}",
                        min_w, max_w, mean_w ) );
                }
            }
            else
            {
                this->loadParameter( name, blob );
            }
        }

        std::vector<ITensor*> getGradients() const override
        {
            if ( !this->isTraining() )
                throw std::runtime_error( "TokenEmbedding: getGradients() called when not in training mode" );

            std::vector<ITensor*> grads;
            if ( wte_grad_ ) grads.push_back( wte_grad_.get() );
            return grads;
        }

        EmbeddingTensorType* getWteGrad() const noexcept
        {
            return wte_grad_.get();
        }

        // ====================================================================
        // Component interface
        // ====================================================================

        const ComponentType getType() const override
        {
            return ComponentType::TokenEmbedding;
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
            oss << "--------------------\n";
            oss << "TokenEmbedding: " << this->getName() << "\n";
            oss << "Vocabulary size: " << config_.getVocabSize() << " tokens\n";
            oss << "Embedding dim:   " << config_.getEmbeddingDim() << "\n";
            oss << "Device: " << deviceTypeToString( this->getDeviceType() ) << "\n";
            oss << "Parameter count: " << parameterCount() << "\n";
            return oss.str();
        }

        int64_t getVocabSize()    const noexcept
        {
            return static_cast<int64_t>(config_.getVocabSize());
        }
        int64_t getEmbeddingDim() const noexcept
        {
            return static_cast<int64_t>(config_.getEmbeddingDim());
        }

    protected:

        // ====================================================================
        // Lifecycle hooks
        // ====================================================================

        void onExecutionContextSet() override
        {
            initializeParameters();
            createOperation();
        }

        void onBuilding( const shape_t& input_shape ) override
        {
            validateInputShape( input_shape );

            max_batch_size_ = input_shape[ 0 ];
            max_seq_len_ = input_shape[ 1 ];

            // REVIEW: API needs work
            operation_->setParameters( wte_.get(), nullptr );

            if ( this->isTraining() )
            {
                initializeParameterGradients();
                operation_->setGradients( wte_grad_.get(), nullptr );
            }

            operation_->build( input_shape );

            auto device = this->getExecutionContext()->getDeviceId();

            shape_t max_out_shape = {
                max_batch_size_,
                max_seq_len_,
                static_cast<dim_t>(config_.getEmbeddingDim()) };

            output_ = std::make_unique<EmbeddingTensorType>( device, max_out_shape );
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
                    operation_->setGradients( wte_grad_.get(), nullptr );
                }
            }
            else
            {
                operation_->clearGradients();

                if ( wte_grad_ )
                    zero( *wte_grad_ );
            }
        }

    private:
        TokenEmbeddingConfig config_;

        int64_t max_batch_size_{ 0 };
        int64_t max_seq_len_{ 0 };

        std::unique_ptr<EmbeddingTensorType> wte_{ nullptr };
        std::unique_ptr<EmbeddingTensorType> wte_grad_{ nullptr };

        std::unique_ptr<TokenIndexType>      input_grad_{ nullptr };
        std::unique_ptr<EmbeddingTensorType> output_{ nullptr };
        std::unique_ptr<EmbeddingTensorType> current_output_view_{ nullptr };

        std::shared_ptr<UnaryOperation<TDeviceType, TIndex, TPrecision>> operation_{ nullptr };

        std::unique_ptr<IExecutionContext> owned_exec_context_{ nullptr };

        void validateInputShape( const shape_t& shape ) const
        {
            if ( shape.size() != 2 )
                throw std::invalid_argument(
                    "TokenEmbedding: input must be rank-2 [B, T]" );
        }

        void initializeParameters()
        {
            const float std_dev = 1.0f / std::sqrt( static_cast<float>(config_.getEmbeddingDim()) );
            auto device_id = this->getExecutionContext()->getDeviceId();

            wte_ = std::make_unique<EmbeddingTensorType>(
                device_id,
                shape_t{ static_cast<dim_t>(config_.getVocabSize()),
                         static_cast<dim_t>(config_.getEmbeddingDim()) } );
            wte_->setName( this->getName() + ".wte" );
            normal( *wte_, std_dev );
        }

        void initializeParameterGradients()
        {
            if ( wte_grad_ ) return;

            auto device_id = this->getExecutionContext()->getDeviceId();

            wte_grad_ = std::make_unique<EmbeddingTensorType>( device_id, wte_->shape() );
            wte_grad_->setName( this->getName() + ".wte.grad" );
            zero( *wte_grad_ );
        }

        void createOperation()
        {
            operation_ = OperationRegistry::instance()
                .createUnaryOperation<TDeviceType, TIndex, TPrecision>(
                    "TokenEmbeddingOp",
                    this->getExecutionContext(),
                    config_ );

            if ( !operation_ )
                throw std::runtime_error( "TokenEmbedding: failed to create compute backend operation." );
        }
    };
}