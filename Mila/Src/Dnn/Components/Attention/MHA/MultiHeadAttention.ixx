/**
 * @file MultiHeadAttention.ixx
 * @brief Multi-Head Attention module (concatenated QKV input).
 */

module;
#include <memory>
#include <vector>
#include <string>
#include <sstream>
#include <type_traits>
#include <stdexcept>
#include <cstdint>
#include <optional>

export module Dnn.Components.MultiHeadAttention;
export import Dnn.Components.MultiHeadAttentionConfig;

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
import Compute.IPackedKvInference;
import Compute.IKvCacheLifecycle;
import Serialization.ModelArchive;
import Serialization.Mode;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    /**
     * @brief Multi-Head Attention module that accepts concatenated QKV input.
     *
     * The module requires a single input tensor in model-layout containing
     * concatenated Q, K and V along the feature axis:
     *   input shape == [B, T, 3 * embedding_dim]
     *
     * The backend compute implementation (registered as "MultiHeadAttentionOp") must
     * accept the concatenated QKV input and produce an output of shape:
     *   output shape == [B, T, embedding_dim]
     *
     * KV-cache inference is an optional backend capability. After build(),
     * supportsKVCache() indicates whether the underlying operation implements
     * both IPositionalUnaryOp (prefill/decode dispatch) and IKVCacheLifecycle
     * (cache init/reset). Both pointers are resolved once at build time.
     *
     * The KV cache lifecycle (initializeKVCache / resetKVCache) is intended
     * to be driven exclusively by the owning transformer's generate() method.
     * forward() is the sole entry point for prefill; decode() handles
     * autoregressive single-token generation.
     *
     * REVIEW: initializeKVCache() and resetKVCache() are currently public.
     * When TransformerBase<> is introduced as the common base for GptTransformer,
     * LlamaTransformer, MistralTransformer etc., revisit whether these should
     * become private with 'friend class TransformerBase<TDeviceType, TPrecision>'
     * to enforce that only the generate() orchestration path may manage the
     * KV cache lifecycle.
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class MultiHeadAttention : public Component<TDeviceType, TPrecision>
    {
    public:
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using TensorType = Tensor<TPrecision, MR>;
        using ComponentBase = Component<TDeviceType, TPrecision>;

        /**
         * @brief Construct MultiHeadAttention component.
         *
         * @param name      Component name identifier (mandatory).
         * @param config    MultiHeadAttention configuration.
         * @param device_id Optional DeviceId to create owned ExecutionContext (standalone mode).
         */
        explicit MultiHeadAttention( const std::string& name, const MultiHeadAttentionConfig& config, std::optional<DeviceId> device_id = std::nullopt )
            : ComponentBase( name ), config_( config )
        {
            config_.validate();

            if ( device_id.has_value() )
            {
                if ( device_id->type != TDeviceType )
                {
                    throw std::invalid_argument( "MultiHeadAttention: device type mismatch" );
                }

                context_ = createExecutionContext( device_id.value() );
                this->setExecutionContext( context_.get() );
            }
        }

        ~MultiHeadAttention() override = default;

        // ====================================================================
        // Forward / Backward
        // ====================================================================

        /**
         * @brief Standard forward pass.
         *
         * Always available regardless of backend. When the backend supports
         * KV caching, the first forward() call initializes and populates the
         * cache (prefill with position_offset=0). When called again after
         * decode() steps, it automatically resets the cache and begins a new
         * prefill session — no explicit session management required by callers.
         *
         * @param input Concatenated QKV input [B, T, 3 * embedding_dim].
         * @return Reference to component-owned output tensor.
         */
        TensorType& forward( const TensorType& input )
        {
            if ( !this->isBuilt() )
                throw std::runtime_error(
                    "MultiHeadAttention must be built before calling forward()." );

            validateConcatenatedQKVShape( input.shape() );

            if ( kv_cache_op_ && positional_op_ )
            {
                // Called after decode steps — reset for new session
                if ( decode_active_ )
                {
                    kv_cache_op_->resetKvCache();
                    cache_initialized_ = false;
                    decode_active_ = false;
                }

                // Initialize cache on first forward() if not yet done
                if ( !cache_initialized_ )
                {
                    kv_cache_op_->initializeKvCache(
                        static_cast<int>(max_input_shape_[ 0 ]),
                        static_cast<int>(max_input_shape_[ 1 ]) );
                    cache_initialized_ = true;
                }

                // Prefill — populates cache as side effect
                positional_op_->prefill( input, *owned_output_ );

                return resolveOutputView( input.shape() );
            }

            operation_->forward( input, *owned_output_ );

            return resolveOutputView( input.shape() );
        }

        /**
         * @brief Run backward pass and return component-owned input-gradient tensor.
         *
         * @param input       Concatenated QKV input tensor used in forward.
         * @param output_grad Gradient w.r.t. the module output.
         * @return Reference to component-owned TensorType containing the input gradient.
         */
        TensorType& backward( const TensorType& input, const TensorType& output_grad )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "MultiHeadAttention must be built before calling backward." );
            }

            if ( this->isInferenceMode() )
            {
                throw std::runtime_error( "MultiHeadAttention must be in training mode to call backward." );
            }

            validateConcatenatedQKVShape( input.shape() );

            zero( *owned_input_grad_ );
            operation_->backward( input, output_grad, *owned_input_grad_ );

            return *owned_input_grad_;
        }

        // ====================================================================
        // Decode path / KV Cache
        // ====================================================================

        /**
         * @brief Inference-only single-token decode pass.
         *
         * When the backend implements IPositionalUnaryOp and the cache has been
         * populated by a prior forward() call, uses the fast O(n) KV cache
         * path. When the backend does not support positional dispatch
         * (CpuMultiHeadAttentionOp), falls back to forward(). The caller never
         * needs to know which path was taken.
         *
         * Precondition: forward() must have been called at least once to
         * populate the KV cache before decode() is called.
         *
         * @param input    Single-token QKV input [B, 1, 3 * embedding_dim].
         * @param position Current sequence position (0-based).
         * @return Reference to component-owned single-token output tensor.
         */
        TensorType& decode( const TensorType& input, int position )
        {
            if ( !this->isBuilt() )
                throw std::runtime_error(
                    "MultiHeadAttention must be built before calling decode()." );

            validateConcatenatedQKVShape( input.shape() );

            if ( positional_op_ && cache_initialized_ )
            {
                positional_op_->decode( input, *owned_decode_output_, position );
                decode_active_ = true;

                return *owned_decode_output_;
            }

            // Fallback — CpuMultiHeadAttentionOp or cache not yet initialized.
            operation_->forward( input, *owned_output_ );

            return resolveOutputView( input.shape() );
        }

        /**
         * @brief Returns true when the underlying operation implements both
         * IPositionalUnaryOp and IKVCacheLifecycle.
         *
         * Resolved once at build time. CPU backends return false; CUDA backends
         * return true when CudaMultiHeadAttentionOp is in use. Safe to query before
         * calling generate() to determine which forward path is available.
         */
        bool supportsKVCache() const noexcept
        {
            return kv_cache_op_ != nullptr && positional_op_ != nullptr;
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

        std::vector<ITensor*> getParameters() const override
        {
            return {};
        }
        std::vector<ITensor*> getGradients() const override
        {
            return {};
        }

        // ====================================================================
        // Component interface
        // ====================================================================

        const ComponentType getType() const override
        {
            return ComponentType::MultiHeadAttention;
        }

        DeviceId getDeviceId() const override
        {
            return this->getExecutionContext()->getDeviceId();
        }

        void synchronize() override
        {
            this->getExecutionContext()->synchronize();
        }

        size_t parameterCount() const override
        {
            return 0;
        }

        MemoryStats getMemoryStats() const override
        {
            MemoryStats stats;

            if ( owned_output_ != nullptr )
            {
                stats.device_state_bytes += owned_output_->getStorageSize();
            }

            if ( owned_decode_output_ != nullptr )
            {
                stats.device_state_bytes += owned_decode_output_->getStorageSize();
            }

            if ( owned_input_grad_ != nullptr )
            {
                stats.device_gradient_bytes += owned_input_grad_->getStorageSize();
            }

            return stats;
        }

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "--------------------\n";
            oss << "MultiHeadAttention: " << this->getName() << "\n";
            oss << "Device Id: " << this->getExecutionContext()->getDeviceId().toString() << "\n";
            oss << "Model dimension: " << config_.getModelDim() << "\n";
            oss << "Number of heads: " << config_.getNumHeads() << "\n";
            oss << "Head size: " << (config_.getModelDim() / config_.getNumHeads()) << "\n";
            oss << "Decode path: " << (supportsKVCache() ? "KV cache (fast)" : "fallback (forward)") << "\n";
            oss << "Parameter count: " << parameterCount() << "\n";

            return oss.str();
        }

        int64_t getModelDim()  const noexcept
        {
            return config_.getModelDim();
        }
        int64_t getNumHeads()  const noexcept
        {
            return config_.getNumHeads();
        }
        const MultiHeadAttentionConfig& getConfig() const noexcept
        {
            return config_;
        }

    protected:

        void onExecutionContextSet() override
        {
            createOperation();
        }

        void onBuilding( const BuildContext& build_config ) override
        {
            const auto& input_shape = build_config.inputShape();

            validateConcatenatedQKVShape( input_shape );

            operation_->setParameters( nullptr, nullptr );
            operation_->build( build_config );

            // Resolve capability interfaces once at build time.
            // Null for CPU backends that lack KV cache / positional dispatch.
            kv_cache_op_ = dynamic_cast<IKvCacheLifecycle*>( operation_.get() );
            positional_op_ = dynamic_cast<IPackedKvInference*>( operation_.get() );

            max_input_shape_ = input_shape;

            auto device = this->getExecutionContext()->getDeviceId();

            shape_t out_shape = max_input_shape_;
            out_shape.back() = config_.getModelDim();

            owned_output_ = std::make_unique<TensorType>( device, out_shape, this->getName() + ".output" );
            owned_input_grad_ = std::make_unique<TensorType>( device, max_input_shape_, this->getName() + ".input.grad" );

            // Decode output is a single-token slice: [B, 1, model_dim]
            shape_t decode_output_shape = { max_input_shape_[ 0 ], 1, config_.getModelDim() };
            owned_decode_output_ = std::make_unique<TensorType>( device, decode_output_shape, this->getName() + ".output.decode" );
        }

        void onTrainingModeChanging( TrainingMode training_mode ) override
        {
            operation_->setTrainingMode( training_mode );

            auto is_training = (training_mode == TrainingMode::Normal);

            // Entering training mode resets any active decode session
            if ( is_training && kv_cache_op_ && cache_initialized_ )
            {
                kv_cache_op_->resetKvCache();
                cache_initialized_ = false;
                decode_active_ = false;
            }
        }

    private:
        using OpType = typename OperationTraits<OperationType::MultiHeadAttentionOp, TDeviceType, TPrecision>::type;

        MultiHeadAttentionConfig config_;
        shape_t max_input_shape_;

        std::shared_ptr<OpType> operation_{ nullptr };
        std::unique_ptr<IExecutionContext> context_{ nullptr };

        // Non-owning capability interface pointers. Lifetime tied to operation_.
        // Resolved once in onBuilding(). Null for backends that do not implement
        // the corresponding interface (e.g. CPU).
        IKvCacheLifecycle* kv_cache_op_{ nullptr };
        IPackedKvInference* positional_op_{ nullptr };

        // KV cache session state
        bool cache_initialized_{ false };
        bool decode_active_{ false };

        std::unique_ptr<TensorType> owned_output_{ nullptr };
        std::unique_ptr<TensorType> output_view_{ nullptr };
        std::unique_ptr<TensorType> owned_input_grad_{ nullptr };
        std::unique_ptr<TensorType> owned_decode_output_{ nullptr };

        TensorType& resolveOutputView( const shape_t& input_shape )
        {
            if ( input_shape == max_input_shape_ )
            {
                return *owned_output_;
            }

            auto output_shape = input_shape;
            output_shape.back() = config_.getModelDim();
            output_view_ = std::make_unique<TensorType>( owned_output_->view( output_shape ) );

            return *output_view_;
        }

        void validateConcatenatedQKVShape( const shape_t& shape ) const
        {
            if ( shape.size() != 3 )
            {
                throw std::invalid_argument( "MultiHeadAttention: expected 3D model-layout shape" );
            }

            const int64_t trailing = shape.back();
            const int64_t expected = config_.getModelDim() * 3;

            if ( trailing != expected )
            {
                std::ostringstream oss;
                oss << "MultiHeadAttention: expected concatenated QKV trailing dimension " << expected
                    << " (3 * embedding_dim), got " << trailing;
                throw std::invalid_argument( oss.str() );
            }
        }

        void createOperation()
        {
            operation_ = std::make_shared<OpType>( this->getExecutionContext(), config_ );

            if ( !operation_ )
            {
                throw std::runtime_error( "Failed to create MultiHeadAttention compute backend operation." );
            }
        }
    };
}
