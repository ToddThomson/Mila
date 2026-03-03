/**
 * @file GroupedQueryAttention.ixx
 * @brief Grouped-Query Attention module (concatenated QKV input).
 *
 * Grouped-Query Attention (GQA) generalises Multi-Head Attention by allowing
 * the number of K/V heads (num_kv_heads) to differ from the number of Q heads
 * (num_heads).  Each K/V head is shared by a group of (num_heads / num_kv_heads)
 * Q heads, reducing memory bandwidth and KV cache size during inference while
 * retaining most of the representational power of full MHA.
 *
 * Special cases:
 *   num_kv_heads == num_heads  →  standard Multi-Head Attention
 *   num_kv_heads == 1          →  Multi-Query Attention (MQA)
 *
 * Module delegates compute to a device-specific UnaryOperation implementation
 * registered as "GroupedQueryAttentionOp".  The operation receives a
 * concatenated QKV input whose trailing dimension is:
 *
 *   (num_heads + 2 * num_kv_heads) * head_dim
 *
 * where head_dim = model_dim / num_heads.
 *
 * KV-cache inference is an optional backend capability surfaced via
 * supportsKVCache().  When supported, forward() dispatches to the appropriate
 * IKVCacheable path (prefill / decode) based on internal session state.
 * External callers use the public forward() / decode() API and are unaffected
 * by the KV cache machinery.
 *
 * REVIEW: initializeKVCache() and resetKVCache() are currently public.
 * When TransformerBase<> is introduced as the common base for GptTransformer,
 * LlamaTransformer, MistralTransformer etc., revisit whether these should
 * become private with 'friend class TransformerBase<TDeviceType, TPrecision>'
 * to enforce that only the generate() orchestration path may manage the
 * KV cache lifecycle.
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

export module Dnn.Components.GroupedQueryAttention;
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
     * @brief Grouped-Query Attention module that accepts concatenated QKV input.
     *
     * The module requires a single input tensor in model-layout containing
     * concatenated Q, K and V along the feature axis.  Unlike MHA, Q and KV
     * have different numbers of heads, so the trailing dimension is:
     *
     *   input shape  == [B, T, (num_heads + 2 * num_kv_heads) * head_dim]
     *   output shape == [B, T, model_dim]                       (model_dim = num_heads * head_dim)
     *
     * The backend compute implementation (registered as "GroupedQueryAttentionOp")
     * must accept this layout and produce the output above.
     *
     * KV-cache inference is an optional backend capability.  After build(),
     * supportsKVCache() indicates whether the underlying operation implements
     * IKVCacheable.  The cached pointer is resolved once at build time.
     *
     * The KV cache lifecycle (initializeKVCache / resetKVCache) is intended to
     * be driven exclusively by the owning transformer's generate() method.
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class GroupedQueryAttention : public Component<TDeviceType, TPrecision>
    {
    public:
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using TensorType = Tensor<TPrecision, MR>;
        using ComponentBase = Component<TDeviceType, TPrecision>;

        /**
         * @brief Construct a GroupedQueryAttention component.
         *
         * @param name      Component name identifier (mandatory).
         * @param config    GQA configuration (model_dim, num_heads, num_kv_heads).
         * @param device_id Optional DeviceId to create an owned ExecutionContext
         *                  (standalone / unit-test mode).
         */
        explicit GroupedQueryAttention(
            const std::string& name,
            const GroupedQueryAttentionConfig& config,
            std::optional<DeviceId> device_id = std::nullopt )
            : ComponentBase( name ), config_( config )
        {
            config_.validate();

            if ( device_id.has_value() )
            {
                if ( device_id->type != TDeviceType )
                {
                    throw std::invalid_argument( "GroupedQueryAttention: device type mismatch" );
                }

                context_ = createExecutionContext( device_id.value() );
                this->setExecutionContext( context_.get() );
            }
        }

        ~GroupedQueryAttention() override = default;

        // ====================================================================
        // Forward / Backward / Decode
        // ====================================================================

        /**
         * @brief Standard forward pass.
         *
         * Always available regardless of backend.  When the backend supports
         * KV caching, the first forward() call initialises and populates the
         * cache (prefill).  When called again after decode() steps, it
         * automatically resets the cache and begins a new prefill session —
         * no explicit session management required by callers.
         *
         * @param input Concatenated QKV input [B, T, (Q + 2*KV) * head_dim].
         * @return Reference to component-owned output tensor [B, T, model_dim].
         */
        TensorType& forward( const TensorType& input )
        {
            if ( !this->isBuilt() )
                throw std::runtime_error(
                    "GroupedQueryAttention must be built before calling forward()." );

            validateConcatenatedQKVShape( input.shape() );

            if ( kv_cacheable_ )
            {
                // Called again after decode steps — reset for a new session.
                if ( decode_active_ )
                {
                    kv_cacheable_->resetKVCache();
                    cache_initialized_ = false;
                    decode_active_ = false;
                }

                // Initialise cache on the first forward() if not yet done.
                if ( !cache_initialized_ )
                {
                    kv_cacheable_->initializeKVCache(
                        static_cast<int>(max_input_shape_[ 0 ]),
                        static_cast<int>(max_input_shape_[ 1 ]) );
                    cache_initialized_ = true;
                }

                // Prefill — populates KV cache as a side effect.
                kv_cacheable_->forwardPrefill( input, *owned_output_ );
                return resolveOutputView( input.shape() );
            }

            operation_->forward( input, *owned_output_ );
            return resolveOutputView( input.shape() );
        }

        /**
         * @brief Run backward pass and return the component-owned input-gradient tensor.
         *
         * @param input       Concatenated QKV input tensor used in forward.
         * @param output_grad Gradient w.r.t. the module output.
         * @return Reference to component-owned TensorType containing the input gradient.
         */
        TensorType& backward( const TensorType& input, const TensorType& output_grad )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error(
                    "GroupedQueryAttention must be built before calling backward." );
            }

            if ( !this->isTraining() )
            {
                throw std::runtime_error(
                    "GroupedQueryAttention must be in training mode to call backward. "
                    "Call setTraining(true) first." );
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
         * When the backend implements IKVCacheable and the cache has been
         * populated by a prior forward() call, uses the fast O(n) KV cache
         * path.  When the backend does not support KV caching, falls back to
         * forward().  The caller never needs to know which path was taken.
         *
         * Precondition: forward() must have been called at least once to
         * populate the KV cache before decode() is called.
         *
         * @param input    Single-token QKV input [B, 1, (Q + 2*KV) * head_dim].
         * @param position Current sequence position (0-based).
         * @return Reference to component-owned single-token output tensor.
         */
        TensorType& decode( const TensorType& input, int position )
        {
            if ( !this->isBuilt() )
                throw std::runtime_error(
                    "GroupedQueryAttention must be built before calling decode()." );

            validateConcatenatedQKVShape( input.shape() );

            if ( kv_cacheable_ && cache_initialized_ )
            {
                // Fast path — O(n) attention using cached KV state.
                kv_cacheable_->forwardDecode( input, *owned_decode_output_, position );
                decode_active_ = true;
                return *owned_decode_output_;
            }

            // Fallback — backend does not support KV caching or cache not yet
            // initialised.  Correct but not optimised; acceptable for CPU.
            operation_->forward( input, *owned_output_ );
            return resolveOutputView( input.shape() );
        }

        /**
         * @brief Returns true when the underlying operation implements IKVCacheable.
         *
         * Resolved once at build time.  CPU backends return false; CUDA backends
         * return true when CudaGroupedQueryAttentionOp is in use.
         */
        bool supportsKVCache() const noexcept
        {
            return kv_cacheable_ != nullptr;
        }

        /**
         * @brief Allocate KV cache buffers for inference.
         *
         * Intended to be called exclusively by the owning transformer's generate()
         * during session setup.  Throws if the backend does not support KV caching.
         *
         * The KV cache is sized for num_kv_heads rather than num_heads, reflecting
         * the memory savings that GQA provides over MHA.
         *
         * @param max_seq_len Maximum sequence length the cache must accommodate.
         *
         * REVIEW: Consider making private with friend TransformerBase<> once
         * that base class is introduced.
         */
        void initializeKVCache( int64_t max_seq_len )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error(
                    "GroupedQueryAttention must be built before initializeKVCache()." );
            }

            if ( !kv_cacheable_ )
            {
                throw std::runtime_error(
                    "GroupedQueryAttention: KV cache is not supported by this backend." );
            }

            kv_cacheable_->initializeKVCache(
                static_cast<int>(max_input_shape_[ 0 ]),
                static_cast<int>(max_seq_len) );
        }

        /**
         * @brief Reset KV cache state between generation sessions.
         *
         * Intended to be called exclusively by the owning transformer's generate()
         * between independent generation requests.  Throws if the backend does not
         * support KV caching.
         *
         * REVIEW: Consider making private with friend TransformerBase<> once
         * that base class is introduced.
         */
        void resetKVCache()
        {
            if ( !kv_cacheable_ )
            {
                throw std::runtime_error(
                    "GroupedQueryAttention: KV cache is not supported by this backend." );
            }

            kv_cacheable_->resetKVCache();
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
            return ComponentType::GroupedQueryAttention;
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

        std::string toString() const override
        {
            const int64_t head_dim = config_.getModelDim() / config_.getNumHeads();
            const int64_t group_size = config_.getNumHeads() / config_.getNumKvHeads();

            std::ostringstream oss;
            oss << "--------------------\n";
            oss << "GroupedQueryAttention: " << this->getName() << "\n";
            oss << "Device Id: " << this->getExecutionContext()->getDeviceId().toString() << "\n";
            oss << "Model dimension: " << config_.getModelDim() << "\n";
            oss << "Num Q heads: " << config_.getNumHeads() << "\n";
            oss << "Num KV heads: " << config_.getNumKvHeads() << "\n";
            oss << "Head size: " << head_dim << "\n";
            oss << "Group size (Q heads per KV head): " << group_size << "\n";
            oss << "Decode path: " << (kv_cacheable_ ? "KV cache (fast)" : "fallback (forward)") << "\n";
            oss << "Parameter count: " << parameterCount() << "\n";
            return oss.str();
        }

        // ====================================================================
        // Config accessors
        // ====================================================================

        int64_t getModelDim()   const noexcept
        {
            return config_.getModelDim();
        }
        int64_t getNumHeads()   const noexcept
        {
            return config_.getNumHeads();
        }
        int64_t getNumKvHeads() const noexcept
        {
            return config_.getNumKvHeads();
        }

        const GroupedQueryAttentionConfig& getConfig() const noexcept
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

            // Resolve IKVCacheable once at build time.  Null for CPU backends.
            kv_cacheable_ = dynamic_cast<IKVCacheable*>(operation_.get());

            max_input_shape_ = input_shape;

            auto device = this->getExecutionContext()->getDeviceId();

            // Full-sequence output: [B, T, model_dim]
            shape_t out_shape = max_input_shape_;
            out_shape.back() = config_.getModelDim();

            owned_output_ = std::make_unique<TensorType>( device, out_shape );
            owned_output_->setName( this->getName() + ".output" );

            // Input gradient has the same shape as the (packed QKV) input.
            owned_input_grad_ = std::make_unique<TensorType>( device, max_input_shape_ );
            owned_input_grad_->setName( this->getName() + ".input.grad" );

            // Single-token decode output: [B, 1, model_dim]
            shape_t decode_output_shape = { max_input_shape_[ 0 ], 1, config_.getModelDim() };
            owned_decode_output_ = std::make_unique<TensorType>( device, decode_output_shape );
            owned_decode_output_->setName( this->getName() + ".output.decode" );
        }

        void onTrainingChanging( bool is_training ) override
        {
            if ( operation_ )
                operation_->setTraining( is_training );

            // Entering training mode invalidates any active decode session.
            if ( is_training && kv_cacheable_ && cache_initialized_ )
            {
                kv_cacheable_->resetKVCache();
                cache_initialized_ = false;
                decode_active_ = false;
            }
        }

    private:
        GroupedQueryAttentionConfig config_;
        shape_t max_input_shape_;

        std::shared_ptr<UnaryOperation<TDeviceType, TPrecision>> operation_{ nullptr };
        std::unique_ptr<IExecutionContext> context_{ nullptr };

        // Non-owning; lifetime tied to operation_.  Null when the compute backend
        // does not implement IKVCacheable (e.g. CPU).  Resolved once in onBuilding().
        IKVCacheable* kv_cacheable_{ nullptr };

        // KV cache session state.
        bool cache_initialized_{ false };
        bool decode_active_{ false };

        std::unique_ptr<TensorType> owned_output_{ nullptr };
        std::unique_ptr<TensorType> output_view_{ nullptr };
        std::unique_ptr<TensorType> owned_input_grad_{ nullptr };
        std::unique_ptr<TensorType> owned_decode_output_{ nullptr };

        // ====================================================================
        // Private helpers
        // ====================================================================

        /**
         * @brief Return a view of owned_output_ trimmed to the actual input shape.
         *
         * When the input batch is smaller than max_input_shape_ (e.g. the final
         * partial batch), we return a view with the correct leading dimensions
         * rather than the full pre-allocated buffer.
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

        /**
         * @brief Validate that the input tensor has the expected GQA-packed QKV shape.
         *
         * Expected trailing dimension:
         *   (num_heads + 2 * num_kv_heads) * head_dim
         *
         * This differs from MHA where all three projections share the same head
         * count and the trailing dim is simply 3 * model_dim.
         */
        void validateConcatenatedQKVShape( const shape_t& shape ) const
        {
            if ( shape.size() != 3 )
            {
                throw std::invalid_argument(
                    "GroupedQueryAttention: expected 3D model-layout shape [B, T, features]" );
            }

            const int64_t head_dim = config_.getModelDim() / config_.getNumHeads();
            const int64_t trailing = shape.back();
            const int64_t expected =
                (config_.getNumHeads() + 2 * config_.getNumKvHeads()) * head_dim;

            if ( trailing != expected )
            {
                std::ostringstream oss;
                oss << "GroupedQueryAttention: expected concatenated QKV trailing dimension "
                    << expected
                    << " = (num_heads=" << config_.getNumHeads()
                    << " + 2 * num_kv_heads=" << config_.getNumKvHeads()
                    << ") * head_dim=" << head_dim
                    << ", got " << trailing;
                throw std::invalid_argument( oss.str() );
            }
        }

        void createOperation()
        {
            operation_ = OperationRegistry::instance()
                .createUnaryOperation<TDeviceType, TPrecision>(
                    "GroupedQueryAttentionOp",
                    this->getExecutionContext(),
                    config_ );

            if ( !operation_ )
            {
                throw std::runtime_error(
                    "Failed to create GroupedQueryAttention compute backend operation." );
            }
        }
    };
}
