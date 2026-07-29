/**
 * @file TokenEmbedding.ixx
 * @brief Device-templated TokenEmbedding component.
 *
 * Pure vocabulary lookup: maps token indices [B, T] to dense vectors [B, T, C].
 * Owns the wte parameter and its gradient. No positional encoding -- that is
 * handled downstream by a dedicated encoding component (RoPE, ALiBi, or Learned).
 *
 * Derived from Lpe with all wpe / IPositionalDecode / decode() concerns removed.
 */

module;
#include <memory>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <optional>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <format>

export module Dnn.Components.TokenEmbedding;
export import Dnn.Components.TokenEmbeddingConfig;

import Dnn.Component;
import Dnn.ComponentType;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.TensorOps;
import Dnn.Quantization.Weight.Policies;
import Compute.Device;
import Compute.DeviceId;
import Compute.DeviceType;
import Compute.DeviceTypeTraits;
import Compute.ExecutionContext;
import Compute.ExecutionContextFactory;
import Compute.OperationTraits;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Serialization.ModelArchive;
import Serialization.Metadata;
import Serialization.Mode;
import Serialization.Tensor;
import Serialization.SafeTensors;

import Dnn.TensorOps;
import Dnn.TensorHelpers;
import Logging.Logger;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;
    using namespace Mila::Dnn::Quant::Weight;

    /**
     * @brief Pure token embedding component (device-templated).
     *
     * Transforms input token indices into continuous vector representations
     * by looking up each index in the vocabulary embedding table (wte).
     * No positional information is added here.
     *
     * TTableQuantization = PerChannelFp8<> stores the table as FP8_E4M3 with one
     * float32 absmax scale per vocabulary row, quantized at loadParameter() time
     * (D4 Design B). The vocab-row scale axis coincides with a tied lm_head's
     * per-output-channel scale axis, so getWeightTensorShared() plus
     * getWeightScalesTensorShared() feed Linear::installSharedWeight directly.
     * The quantized path is inference-only.
     *
     * Construction modes:
     * - Standalone: provide a DeviceId to create and own an ExecutionContext.
     * - Child/deferred: omit DeviceId; caller must call setExecutionContext()
     *   before build().
     *
     * @tparam TDeviceType        Device type (DeviceType::Cpu or DeviceType::Cuda).
     * @tparam TIndex             Data type for token indices (typically INT32).
     * @tparam TPrecision         Tensor precision for embeddings (FP32 or BF16).
     * @tparam TTableQuantization Table quantization policy. Must satisfy
     *                            WeightQuantPolicy; defaults to NoWeightQuant.
     */
    export template<DeviceType TDeviceType, TensorDataType TIndex = dtype_t::INT32, TensorDataType TPrecision = dtype_t::FP32,
        WeightQuantPolicy TTableQuantization = NoWeightQuant>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class TokenEmbedding : public Component<TDeviceType, TPrecision>
    {
    public:
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using EmbeddingTensorType = Tensor<TPrecision, MR>;
        using TokenIndexType = Tensor<TIndex, MR>;
        using ComponentBase = Component<TDeviceType, TPrecision>;

        static constexpr bool kIsQuantized = TTableQuantization::kIsQuantized;

        // Per-group scales sit on the gather (input) axis and do not transfer to a
        // row lookup -- only per-vocab-row (per-channel) quantization is meaningful.
        static_assert( !kIsQuantized || TTableQuantization::kPerChannel,
            "TokenEmbedding: table quantization must be per-channel (per vocabulary row)" );

        static constexpr TensorDataType kTableDtype = kIsQuantized
            ? TTableQuantization::kStorageDtype : TPrecision;

        using TableTensorType = Tensor<kTableDtype, MR>;
        using TableScaleTensorType = Tensor<TTableQuantization::kScaleDtype, MR>;

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
            std::optional<DeviceId> device_id = std::nullopt )
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
         * @brief Forward pass -- returns component-owned embeddings tensor.
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

            shape_t actual_out_shape = { B, T, config_.getEmbeddingDim() };
            current_output_view_ = std::make_unique<EmbeddingTensorType>(
                output_->view( actual_out_shape ) );

            // Gemma stores the embedding table raw and shares it with a tied lm_head;
            // the sqrt(embedding_dim) scale is applied here instead of being folded into
            // the table (WeightTying.md D5). Scaling the [B, T, C] view (not the full
            // max-shape buffer) keeps the decode-step cost at one token. In-place is valid.
            if ( config_.getEmbeddingScale() != 1.0f )
                scale( *current_output_view_, config_.getEmbeddingScale(), *current_output_view_,
                    this->getExecutionContext() );

            return *current_output_view_;
        }

        // ====================================================================
        // Backward
        // ====================================================================

        /**
         * @brief Backward pass -- accumulates gradients into wte.
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
            if constexpr ( kIsQuantized )
            {
                throw std::logic_error( "TokenEmbedding: backward is not supported with a quantized table (inference only)." );
            }
            else
            {
                if ( !this->isBuilt() )
                    throw std::runtime_error( "TokenEmbedding must be built before calling backward()." );

                if ( !this->isTrainingMode() )
                    throw std::runtime_error( "TokenEmbedding must be in training mode to call backward()." );

                // REVIEW: If built and in training mode, these buffers should always be initialized in onBuilding. If not, it's a bug.
                if ( !wte_grad_ )
                    throw std::runtime_error( "TokenEmbedding: wte_grad not initialized. This is a bug." );

                if ( !input_grad_ )
                    throw std::runtime_error( "TokenEmbedding: input_grad buffer not allocated. This is a bug." );

                zero( *input_grad_ );

                operation_->backward( input, output_grad, *input_grad_ );

                return *input_grad_;
            }
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

        std::vector<std::string> getParameterNames() const override
        {
            return { "wte" };
        }

        /**
         * @brief Emit the table, and its per-row scales when the table is quantized.
         *
         * The scales ride as a sibling name rather than joining getParameterNames(), for the
         * same reason as Linear: that vector is the archive's save/load join. Underscore, not
         * dot -- parseParameterPath() splits on the last dot, so "wte.scales" would resolve to
         * a component named "<prefix>.wte" and the artifact could never be read back.
         */
        void saveFlatTensors(
            Serialization::SafeTensorsWriter& writer,
            const std::string& prefix,
            Serialization::TensorSavePass pass ) const override
        {
            if ( wte_ )
            {
                this->saveParameterToWriter( writer, prefix + ".wte", *wte_, pass );
            }

            if constexpr ( kIsQuantized )
            {
                if ( wte_scales_ )
                {
                    this->saveParameterToWriter(
                        writer, prefix + ".wte_scale", *wte_scales_, pass );
                }
            }
        }

        void save_( ModelArchive& archive, SerializationMode mode ) const override
        {
            (void)mode;

            // A quantized table is packed storage plus a per-row scale companion, and the
            // archive has no representation for that pairing. Quantization is applied on
            // load for inference; a checkpoint is written from the unquantized path.
            if constexpr ( kIsQuantized )
            {
                throw std::runtime_error(
                    std::format( "TokenEmbedding '{}': cannot serialize a quantized table ({}); "
                        "checkpoints are written from the unquantized path",
                        this->getName(), tensorDataTypeToString( kTableDtype ) ) );
            }
            else
            {
                SerializationMetadata meta;
                meta.set( "type", "TokenEmbedding" )
                    .set( "version", int64_t( 1 ) )
                    .set( "name", this->getName() );

                archive.writeMetadata( "meta.json", meta );

                if ( wte_ )
                {
                    this->saveParameterToArchive( archive, "wte", *wte_ );
                }
            }
        }

        // ====================================================================
        // Parameters and Gradients
        // ====================================================================

        dim_t parameterCount() const override
        {
            return wte_ ? wte_->size() : 0;
        }

        std::vector<ITensor*> getParameters() const override
        {
            std::vector<ITensor*> params;
            if ( wte_ ) params.push_back( wte_.get() );
            return params;
        }

        // Shared ownership of the embedding table, for installing into a tied lm_head
        // after load (WeightTying.md D3). The returned tensor shares the same device
        // buffer; both owners keep it alive regardless of teardown order. On the
        // quantized path the table is FP8_E4M3 and the tied head also needs
        // getWeightScalesTensorShared().
        std::shared_ptr<TableTensorType> getWeightTensorShared() const noexcept
        {
            return wte_;
        }

        // Shared ownership of the per-vocab-row FP32 dequantization scales -- the
        // same axis as a tied lm_head's per-output-channel scales, so one tensor
        // serves both consumers (D4 Design B).
        std::shared_ptr<TableScaleTensorType> getWeightScalesTensorShared() const noexcept requires kIsQuantized
        {
            return wte_scales_;
        }

        void loadParameter( const std::string& name, const ITensorBlob& blob ) override
        {
            if ( name == "wte" )
            {
                if constexpr ( kIsQuantized )
                {
                    // The blob's dtype says which source this is: storage dtype means a
                    // pre-quantized artifact whose scales arrive separately, compute
                    // precision means quantize-on-load, where the BF16 blob never lands on
                    // device at full precision -- absmax scales and FP8 rows are produced in
                    // one pass. Same discrimination as Linear.
                    if ( blob.getMetadata().dtype == kTableDtype )
                    {
                        this->loadParameterFromBlob( "wte", blob, *wte_, wte_->shape() );
                    }
                    else
                    {
                        operation_->quantize( blob, *wte_, *wte_scales_, wte_->shape() );
                    }
                }
                else
                {
                    this->loadParameterFromBlob( "wte", blob, *wte_, wte_->shape() );
                }
            }
            else if ( name == "wte_scale" )
            {
                if constexpr ( kIsQuantized )
                {
                    this->loadParameterFromBlob(
                        "wte_scale", blob, *wte_scales_, wte_scales_->shape() );
                }
                else
                {
                    throw std::invalid_argument( std::format(
                        "TokenEmbedding '{}': received 'wte_scale' but this build is "
                        "unquantized; the artifact was written with weight quantization",
                        this->getName() ) );
                }
            }
            else
            {
                ComponentBase::loadParameter( name, blob );
            }
        }

        std::vector<ITensor*> getGradients() const override
        {
            // Gradient buffers exist only when built for training; inference
            // yields an empty vector (see Component::getGradients contract).
            std::vector<ITensor*> grads;

            if ( wte_grad_ )
                grads.push_back( wte_grad_.get() );

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
            return config_.getVocabSize();
        }
        int64_t getEmbeddingDim() const noexcept
        {
            return config_.getEmbeddingDim();
        }

        MemoryStats getMemoryStats() const override
        {
            MemoryStats stats;

            if ( wte_ != nullptr )
            {
                stats.device_parameter_bytes += wte_->getStorageSize();
            }

            if ( wte_scales_ != nullptr )
            {
                stats.device_parameter_bytes += wte_scales_->getStorageSize();
            }

            if ( output_ != nullptr )
            {
                stats.device_state_bytes += output_->getStorageSize();
            }

            if ( wte_grad_ != nullptr )
            {
                stats.device_gradient_bytes += wte_grad_->getStorageSize();
            }

            if ( input_grad_ != nullptr )
            {
                stats.device_gradient_bytes += input_grad_->getStorageSize();
            }

            return stats;
        }

    protected:

        // ====================================================================
        // Lifecycle hooks
        // ====================================================================

        void onExecutionContextSet() override
        {
            createOperation();
        }

        void onBuilding( const BuildContext& build_context ) override
        {
            validateBuildContext( build_context );
            const auto& input_shape = build_context.inputShape();

            max_batch_size_ = input_shape[ 0 ];
            max_seq_len_ = input_shape[ 1 ];

            initializeParameters( build_context );

            // REVIEW: API needs work. setParameters() wants Weight and Bias
            operation_->setParameters( wte_.get(), nullptr );

            if constexpr ( kIsQuantized )
            {
                operation_->setTableScales( wte_scales_.get() );
            }

            operation_->build( build_context );

            auto device = this->getExecutionContext()->getDeviceId();

            // Output buffer -- allocated at full input shape with embedding_dim as the trailing dimension.
            shape_t output_shape = {
                max_batch_size_,
                max_seq_len_,
                config_.getEmbeddingDim()
            };

            output_ = std::make_unique<EmbeddingTensorType>( device, output_shape, this->getName() + ".output" );
            
            if ( build_context.isTrainingMode() )
            {
                if constexpr ( kIsQuantized )
                {
                    throw std::logic_error(
                        "TokenEmbedding: a quantized table cannot be built in training mode (inference only)." );
                }
                else
                {
                    initializeParameterGradients();

                    operation_->setGradients( wte_grad_.get(), nullptr );

                    input_grad_ = std::make_unique<TokenIndexType>( device, shape_t{ max_batch_size_, max_seq_len_ },
                        this->getName() + ".input.grad" );
                }
            }
        }

        void onTrainingModeChanging( TrainingMode training_mode ) override
        {
            operation_->setTrainingMode( training_mode );

            auto is_eval_mode = ( training_mode == TrainingMode::Eval );
            
            if ( is_eval_mode )
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

        // shared_ptr so a tied lm_head can share ownership of this table after load
        // (WeightTying.md D3). wte_grad_ stays unique_ptr: tying is inference-only.
        std::shared_ptr<TableTensorType> wte_{ nullptr };

        // Per-vocab-row FP32 absmax scales [vocab]. Non-null only when kIsQuantized.
        // shared_ptr for the same tying reason as wte_ (D4 Design B).
        std::shared_ptr<TableScaleTensorType> wte_scales_{ nullptr };

        std::unique_ptr<EmbeddingTensorType> wte_grad_{ nullptr };

        std::unique_ptr<TokenIndexType>      input_grad_{ nullptr };
        std::unique_ptr<EmbeddingTensorType> output_{ nullptr };
        std::unique_ptr<EmbeddingTensorType> current_output_view_{ nullptr };

        using OpType = typename OperationTraits<OperationType::TokenEmbeddingOp, TDeviceType, TPrecision, TTableQuantization>::type;

        std::shared_ptr<OpType> operation_{ nullptr };

        std::unique_ptr<IExecutionContext> owned_exec_context_{ nullptr };

        void validateBuildContext( const BuildContext& context ) const
        {
            const auto& input_shape = context.inputShape();

            if ( input_shape.size() != 2 )
            {
                throw std::invalid_argument(
                    "TokenEmbedding: input must be rank-2 [B, T]" );
            }
        }

        void initializeParameters( const BuildContext& build_context )
        {
            auto device_id = this->getExecutionContext()->getDeviceId();

            auto wte_shape = shape_t{ config_.getVocabSize(), config_.getEmbeddingDim() };

            wte_ = std::make_shared<TableTensorType>( device_id, wte_shape, this->getName() + ".wte" );

            if constexpr ( kIsQuantized )
            {
                // One scale per vocabulary row. Filled by operation_->quantize() at
                // load time; the quantized path has no random-initialization mode.
                wte_scales_ = std::make_shared<TableScaleTensorType>(
                    device_id, shape_t{ config_.getVocabSize() }, this->getName() + ".wte.scales" );
            }
            else
            {
                if ( build_context.shouldInitializeParameters() )
                {
                    const float std_dev = 1.0f / std::sqrt( static_cast<float>(config_.getEmbeddingDim()) );
                    fill_normal( *wte_, 0.0f, std_dev, this->getExecutionContext() );
                }
            }
        }

        void initializeParameterGradients()
        {
            if ( wte_grad_ ) 
                return;

            auto device_id = this->getExecutionContext()->getDeviceId();

            wte_grad_ = std::make_unique<EmbeddingTensorType>( device_id, wte_->shape(), this->getName() + ".wte.grad" );
            zero( *wte_grad_ );
        }

        void createOperation()
        {
            operation_ = std::make_shared<OpType>( this->getExecutionContext(), config_ );

            if ( !operation_ )
                throw std::runtime_error( "TokenEmbedding: failed to create compute backend operation." );
        }
    };
}
