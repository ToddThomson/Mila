/**
 * @file MultiHeadAttention.ixx
 * @brief Multi-Head MultiHeadAttention module (concatenated QKV input).
 *
 * Module delegates compute to a device-specific UnaryOperation implementation
 * that expects a concatenated QKV input.
 *
 * KV-cache inference is an optional backend capability surfaced via
 * supportsKVCache(). When supported, forward() dispatches to the appropriate
 * IKVCacheable path based on the MultiHeadAttentionForwardContext passed by the owning
 * transformer's generate() method. External callers use the default context
 * (Mode::Standard) and are unaffected by the KV cache machinery.
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
import Compute.KVCacheable;
import Serialization.ModelArchive;
import Serialization.Mode;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    /**
     * @brief Multi-Head MultiHeadAttention module that accepts concatenated QKV input.
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
     * IKVCacheable. The cached pointer is resolved once at build time.
     *
     * The KV cache lifecycle (initializeKVCache / resetKVCache) is intended
     * to be driven exclusively by the owning transformer's generate() method.
     * The ForwardContext-based dispatch in forward() is the sole entry point
     * for prefill and decode paths; forwardPrefill / forwardDecode do not
     * exist as public methods.
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
        // Forward / Backward / Decode
        // ====================================================================

        /**
         * @brief Standard forward pass.
         *
         * Always available regardless of backend. When the backend supports
         * KV caching, the first forward() call initializes and populates the
         * cache (prefill). When called again after decode() steps, it
         * automatically resets the cache and begins a new prefill session —
         * no explicit session management required by callers.
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

            if ( kv_cacheable_ )
            {
                // Called after decode steps — reset for new session
                if ( decode_active_ )
                {
                    kv_cacheable_->resetKVCache();
                    cache_initialized_ = false;
                    decode_active_ = false;
                }

                // Initialize cache on first forward() if not yet done
                if ( !cache_initialized_ )
                {
                    kv_cacheable_->initializeKVCache(
                        static_cast<int>(max_input_shape_[ 0 ]),
                        static_cast<int>(max_input_shape_[ 1 ]) );
                    cache_initialized_ = true;
                }

                // Prefill — populates cache as side effect
                kv_cacheable_->forwardPrefill( input, *owned_output_ );
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

            if ( !this->isTraining() )
            {
                throw std::runtime_error( "MultiHeadAttention must be in training mode to call backward. Call setTraining(true) first." );
            }

            validateConcatenatedQKVShape( input.shape() );

            zero( *owned_input_grad_ );
            operation_->backward( input, output_grad, *owned_input_grad_ );

            return *owned_input_grad_;
        }

        // ====================================================================
        // Decode path/KV Cache
        // ====================================================================

        /**
         * @brief Inference-only single-token decode pass.
         *
         * When the backend implements IKVCacheable and the cache has been
         * populated by a prior forward() call, uses the fast O(n) KV cache
         * path. When the backend does not support KV caching (CpuMultiHeadAttentionOp),
         * falls back to forward(). The caller never needs to know which path
         * was taken.
         *
         * Any future component with an optimized inference-only path follows
         * this same pattern — expose decode(), decide path internally.
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

            if ( kv_cacheable_ && cache_initialized_ )
            {
                // Fast path — O(n) MultiHeadAttention using cached KV state
                kv_cacheable_->forwardDecode( input, *owned_decode_output_, position );
                
                decode_active_ = true;
                return *owned_decode_output_;
            }

            // Fallback — CpuMultiHeadAttentionOp or cache not yet initialized.
            // Correct but not optimized — acceptable for CPU inference.
            operation_->forward( input, *owned_output_ );
            
            return resolveOutputView( input.shape() );
        }

        /**
         * @brief Returns true when the underlying operation implements IKVCacheable.
         *
         * Resolved once at build time. CPU backends return false; CUDA backends
         * return true when CudaMultiHeadAttentionOp is in use. Safe to query before
         * calling generate() to determine which forward path is available.
         */
        bool supportsKVCache() const noexcept
        {
            return kv_cacheable_ != nullptr;
        }

        // REVIEW: KV cache is handled by the operation_ when supported, so MultiHeadAttention itself has no direct
        // state to manage beyond the cache_initialized_ and decode_active_ flags.

        ///**
        // * @brief Allocate KV cache buffers for inference.
        // *
        // * Intended to be called exclusively by the owning transformer's generate()
        // * during session setup. Throws if the backend does not support KV caching.
        // *
        // * @param max_seq_len Maximum sequence length the cache must accommodate.
        // *
        // * REVIEW: Consider making private with friend TransformerBase<> once
        // * that base class is introduced. See class-level REVIEW note.
        // */
        //void initializeKVCache( int64_t max_seq_len )
        //{
        //    if ( !this->isBuilt() )
        //    {
        //        throw std::runtime_error( "MultiHeadAttention must be built before initializeKVCache()." );
        //    }

        //    if ( !kv_cacheable_ )
        //    {
        //        throw std::runtime_error( "MultiHeadAttention: KV cache is not supported by this backend." );
        //    }

        //    kv_cacheable_->initializeKVCache(
        //        static_cast<int>(max_input_shape_[ 0 ]),
        //        static_cast<int>(max_seq_len) );
        //}

        ///**
        // * @brief Reset KV cache state between generation sessions.
        // *
        // * Intended to be called exclusively by the owning transformer's generate()
        // * between independent generation requests. Throws if the backend does not
        // * support KV caching.
        // *
        // * REVIEW: Consider making private with friend TransformerBase<> once
        // * that base class is introduced. See class-level REVIEW note.
        // */
        //void resetKVCache()
        //{
        //    if ( !kv_cacheable_ )
        //    {
        //        throw std::runtime_error( "MultiHeadAttention: KV cache is not supported by this backend." );
        //    }

        //    kv_cacheable_->resetKVCache();
        //}

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

        /**
         * @brief Return memory allocation breakdown.
         */
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
            oss << "Decode path: " << (kv_cacheable_ ? "KV cache (fast)" : "fallback (forward)") << "\n";
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

        void onBuilding( const shape_t& input_shape ) override
        {
            validateConcatenatedQKVShape( input_shape );

            operation_->setTraining( this->isTraining() );
            operation_->setParameters( nullptr, nullptr );
            operation_->build( input_shape );

            // Resolve IKVCacheable once at build time. Null for CPU backends.
            kv_cacheable_ = dynamic_cast<IKVCacheable*>(operation_.get());

            max_input_shape_ = input_shape;

            auto device = this->getExecutionContext()->getDeviceId();

            shape_t out_shape = max_input_shape_;
            out_shape.back() = config_.getModelDim();

            owned_output_ = std::make_unique<TensorType>( device, out_shape );
            owned_output_->setName( this->getName() + ".output" );

            owned_input_grad_ = std::make_unique<TensorType>( device, max_input_shape_ );
            owned_input_grad_->setName( this->getName() + ".input.grad" );

            // Decode output is a single-token slice: [B, 1, model_dim]
            shape_t decode_output_shape = { max_input_shape_[ 0 ], 1, config_.getModelDim() };
            owned_decode_output_ = std::make_unique<TensorType>( device, decode_output_shape );
            owned_decode_output_->setName( this->getName() + ".output.decode" );
        }

        void onTrainingChanging( bool is_training ) override
        {
            if ( operation_ )
                operation_->setTraining( is_training );

            // Entering training mode resets any active decode session
            if ( is_training && kv_cacheable_ && cache_initialized_ )
            {
                kv_cacheable_->resetKVCache();
                cache_initialized_ = false;
                decode_active_ = false;
            }
        }

    private:
        MultiHeadAttentionConfig config_;
        shape_t max_input_shape_;

        std::shared_ptr<UnaryOperation<TDeviceType, TPrecision>> operation_{ nullptr };
        std::unique_ptr<IExecutionContext> context_{ nullptr };

        // Non-owning; lifetime tied to operation_. Null for when compute backend does not
        // implement IKVCacheable (e.g. CPU). Resolved once in onBuilding(). decode() uses this to select fast path.
        IKVCacheable* kv_cacheable_{ nullptr };

        // KV cache session state
        bool cache_initialized_{ false };
        bool decode_active_{ false };

        std::unique_ptr<TensorType> owned_output_{ nullptr };
        std::unique_ptr<TensorType> output_view_{ nullptr };
        std::unique_ptr<TensorType> owned_input_grad_{ nullptr };
        std::unique_ptr<TensorType> owned_decode_output_{ nullptr };

        /**
         * @brief Return a view of owned_output_ trimmed to the actual input shape.
         *
         * When the input is smaller than max_input_shape_ (e.g. final batch),
         * we return a view with the correct leading dimensions rather than the
         * full pre-allocated buffer.
         */
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
            operation_ = OperationRegistry::instance()
                .createUnaryOperation<TDeviceType, TPrecision>(
                    "MultiHeadAttentionOp",
                    this->getExecutionContext(),
                    config_ );

            if ( !operation_ )
            {
                throw std::runtime_error( "Failed to create MultiHeadAttention compute backend operation." );
            }
        }
    };
}
