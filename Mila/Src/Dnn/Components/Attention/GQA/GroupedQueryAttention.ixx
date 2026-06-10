/**
 * @file GroupedQueryAttention.ixx
 * @brief Grouped-Query Attention module (concatenated QKV input).
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

export module Dnn.Components.Gqa;
export import Dnn.Components.GqaConfig;

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
import Compute.GqaState;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.IKvInference;
import Compute.IKvCacheLifecycle;
import Serialization.ModelArchive;
import Serialization.Mode;
import Dnn.Quantization.KvCache.Policy;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;
    using namespace Mila::Dnn::Quant::KvCache;

    /**
     * @brief Grouped-Query Attention module that accepts concatenated QKV input.
     *
     * GQA generalises MHA by allowing num_kv_heads < num_heads. Each K/V head
     * is shared by a group of (num_heads / num_kv_heads) Q heads, reducing
     * KV cache memory and bandwidth during inference.
     *
     * The module requires a single input tensor in model-layout containing
     * concatenated Q, K and V along the feature axis:
     *
     *   input shape  == [B, T, (num_heads + 2 * num_kv_heads) * head_dim]
     *   output shape == [B, T, model_dim]       (model_dim = num_heads * head_dim)
     *
     * The backend compute implementation (registered as "GroupedQueryAttentionOp")
     * must accept this layout and produce the output above.
     *
     * KV-cache inference is an optional backend capability. After build(),
     * supportsKVCache() indicates whether the underlying operation implements
     * both IPositionalUnaryOp (prefill/decode dispatch) and IKVCacheLifecycle
     * (cache init/reset). Both pointers are resolved once at build time.
     *
     * The KV cache lifecycle (initializeKVCache / resetKVCache) is intended to
     * be driven exclusively by the owning transformer's generate() method.
     *
     * REVIEW: initializeKVCache() and resetKVCache() are currently public.
     * When TransformerBase<> is introduced as the common base for GptTransformer,
     * LlamaTransformer, MistralTransformer etc., revisit whether these should
     * become private with 'friend class TransformerBase<TDeviceType, TPrecision>'
     * to enforce that only the generate() orchestration path may manage the
     * KV cache lifecycle.
     */
    export template<DeviceType TDeviceType, TensorDataType TComputePrecision, KvCachePolicy TKvPolicy = NoKvCompression>
        requires PrecisionSupportedOnDevice<TComputePrecision, TDeviceType>
    class GroupedQueryAttention : public Component<TDeviceType, TComputePrecision>
    {
    public:
        using ComponentBase = Component<TDeviceType, TComputePrecision>;
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using TensorType = Tensor<TComputePrecision, MR>;

        static constexpr bool kKvCompressed = TKvPolicy::kIsActive;

        static constexpr TensorDataType kCacheDtype = [] {
            if constexpr ( requires { TKvPolicy::kStorageDtype; } )
                return TKvPolicy::kStorageDtype;
            else
                return TComputePrecision;
        }();

        using KvCacheTensorType = Tensor<kCacheDtype, MR>;
        using OpType = typename Compute::OperationTraits<Compute::OperationType::GroupedQueryAttentionOp, TDeviceType, TComputePrecision, TKvPolicy>::type;
    
        /**
         * @brief Construct a GroupedQueryAttention component.
         *
         * @param name      Component name identifier (mandatory).
         * @param config    GQA configuration (model_dim, num_heads, num_kv_heads).
         * @param device_id Optional DeviceId to create an owned ExecutionContext
         *                  (standalone / unit-test mode).
         */
        explicit GroupedQueryAttention( const std::string& name, const GqaConfig& config, std::optional<DeviceId> device_id = std::nullopt )
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
        // Forward / Backward
        // ====================================================================

        /**
         * @brief Standard forward pass.
         *
         * Always available regardless of backend. When the backend supports
         * KV caching, the first forward() call initialises and populates the
         * cache (prefill with position_offset=0). When called again after
         * decode() steps, it automatically resets the cache and begins a new
         * prefill session — no explicit session management required by callers.
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

            shape_t output_shape = input.shape();
            output_shape.back() = config_.getModelDim();

            if ( output_view_->shape() != output_shape )
            {
                output_view_.emplace( output_->view( output_shape ) );
            }

            if ( kv_cache_op_ && positional_op_ )
            {
                if ( decode_active_ )
                {
                    kv_cache_op_->resetKvCache();
                    cache_initialized_ = false;
                    decode_active_ = false;
                }

                if ( !cache_initialized_ )
                {
                    kv_cache_op_->initializeKvCache(
                        static_cast<int>(max_input_shape_[ 0 ]),
                        static_cast<int>(max_input_shape_[ 1 ]) );
                    cache_initialized_ = true;
                }

                // FIXME: positional_op_->prefill( input, *output_view_, 0 );
                return *output_view_;
            }

            operation_->forward( input, *output_view_ );
            return *output_view_;
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

            if ( !this->isTrainingMode() )
            {
                throw std::runtime_error(
                    "GroupedQueryAttention must be in training mode to call backward. "
                    "Call setTraining(true) first." );
            }

            validateConcatenatedQKVShape( input.shape() );

            zero( *input_grad_ );
            operation_->backward( input, output_grad, *input_grad_ );

            return *input_grad_;
        }

        // ====================================================================
        // Decode path / KV Cache
        // ====================================================================
        /**
         * @brief Chunked prefill pass with explicit position offset.
         *
         * Called by the transformer block during chunked prefill. The KV cache
         * must already be initialized (via onBuilding or forward()).
         *
         * @param input           Concatenated QKV input [B, T_chunk, (Q + 2*KV) * head_dim].
         * @param position_offset Absolute position of the first token in this chunk.
         * @return Reference to component-owned output tensor.
         */
        TensorType& prefill( const TensorType& q, const TensorType& k, const TensorType& v, int position_offset )
        {
            if ( !this->isBuilt() )
                throw std::runtime_error(
                    "GroupedQueryAttention must be built before calling prefill()." );

            //validateConcatenatedQKVShape( input.shape() );

            if ( !positional_op_ )
                throw std::runtime_error(
                    "GroupedQueryAttention: backend does not support positional inference." );

            if ( !cache_initialized_ )
                throw std::runtime_error(
                    "GroupedQueryAttention: KV cache must be initialized before prefill()." );

            const int64_t B = q.shape()[ 0 ];
            const int64_t T_actual = q.shape()[ 1 ];
            shape_t output_shape = { B, T_actual, config_.getModelDim() };

            if ( output_view_->shape() != output_shape )
            {
                output_view_.emplace( output_->view( output_shape ) );
            }

            positional_op_->prefill( q, k, v, *output_view_, position_offset );
            
            return *output_view_;
        }

                /**
         * @brief Inference-only single-token decode pass.
         *
         * When the backend implements IPositionalUnaryOp and the cache has been
         * populated by a prior forward() call, uses the fast O(n) KV cache
         * path. When the backend does not support positional dispatch, falls
         * back to forward(). The caller never needs to know which path was taken.
         *
         * Precondition: forward() must have been called at least once to
         * populate the KV cache before decode() is called.
         *
         * @param input    Single-token QKV input [B, 1, (Q + 2*KV) * head_dim].
         * @param position Current sequence position (0-based).
         * @return Reference to component-owned single-token output tensor.
         */
        TensorType& decode( const TensorType& q, const TensorType& k, const TensorType& v, int position_offset )
        {
            if ( !this->isBuilt() )
                throw std::runtime_error(
                    "GroupedQueryAttention must be built before calling decode()." );

            //validateConcatenatedQKVShape( input.shape() );

            if ( positional_op_ && cache_initialized_ )
            {
                positional_op_->decode( q, k, v, *decode_output_, position_offset );
                decode_active_ = true;

                return *decode_output_;
            }

            // Fallback — backend does not support KV caching or cache not yet initialized.
            // FIXME:
            //shape_t output_shape = input.shape();
            //output_shape.back() = config_.getModelDim();

            //if ( output_view_->shape() != output_shape )
            //{
            //    output_view_.emplace( output_->view( output_shape ) );
            //}

            //operation_->forward( input, *output_view_ );
            
            return *output_view_;
        }

        /**
         * @brief Returns true when the underlying operation implements both
         * IPositionalUnaryOp and IKVCacheLifecycle.
         *
         * Resolved once at build time. CPU backends return false; CUDA backends
         * return true when CudaGroupedQueryAttentionOp is in use.
         */
        bool supportsKVCache() const noexcept
        {
            return kv_cache_op_ != nullptr && positional_op_ != nullptr;
        }

        /**
         * @brief Forward the shared transient workspace to the underlying operation.
         *
         * Must be called after build() and before prefill() or decode().
         * A no-op on backends that do not implement setState (e.g. CPU stub).
         *
         * @param state Non-owning pointers to the shared GQA scratch tensors.
         */
        void setState( const GqaState& state )
        {
            operation_->setState( state );
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

        MemoryStats getMemoryStats() const override
        {
            MemoryStats stats;

            stats.device_state_bytes += operation_->getStateMemorySize();

            if ( output_ != nullptr )
                stats.device_state_bytes += output_->getStorageSize();

            if ( decode_output_ != nullptr )
                stats.device_state_bytes += decode_output_->getStorageSize();

            if ( input_grad_ != nullptr )
                stats.device_gradient_bytes += input_grad_->getStorageSize();

            return stats;
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
            oss << "Decode path: " << (supportsKVCache() ? "KV cache (fast)" : "fallback (forward)") << "\n";
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

        const GqaConfig& getConfig() const noexcept
        {
            return config_;
        }

    protected:

        void onExecutionContextSet() override
        {
            createOperation();
        }

        void onBuilding( const BuildContext& context ) override
        {
            const auto& input_shape = context.inputShape();
            validateConcatenatedQKVShape( input_shape );

            auto B = input_shape[ 0 ];
            auto T = input_shape[ 1 ];

            operation_->setParameters( nullptr, nullptr );
            operation_->build( context );

            // Resolve capability interfaces once at build time.
            // Null for CPU backends that lack KV cache / positional dispatch.
            kv_cache_op_ = dynamic_cast<IKvCacheLifecycle*>( operation_.get() );
            positional_op_ = dynamic_cast<IKvInference*>( operation_.get() );

            max_input_shape_ = input_shape;

            auto device = this->getExecutionContext()->getDeviceId();

            if ( context.isInferenceMode() )
            {
                // KV cache sized to full context_length at build time.
                if ( kv_cache_op_ )
                {
                    // REVIEW: int? Should be dim_t?
                    kv_cache_op_->initializeKvCache(
                        static_cast<int>(B),
                        static_cast<int>(T) );
                    cache_initialized_ = true;
                }

                // Decode path output: T=1
                shape_t decode_output_shape = { input_shape[ 0 ], 1, config_.getModelDim() };
                decode_output_ = std::make_unique<TensorType>( device, decode_output_shape, this->getName() + ".output_decode" );

                // Prefill path output — sized for one prefill chunk at a time
                shape_t output_shape = { B, context.getPrefillSize(), config_.getModelDim() };
                output_ = std::make_unique<TensorType>( device, output_shape, this->getName() + ".output_prefill" );
                output_view_.emplace( output_->view( output_->shape() ) );
            }
            else
            {
                // Training — full sequence output buffer.
                shape_t output_shape = input_shape;
                output_shape.back() = config_.getModelDim();
                output_ = std::make_unique<TensorType>( device, output_shape, this->getName() + ".output" );
                output_view_.emplace( output_->view( output_->shape() ) );

                // Input gradient — same shape as packed QKV input.
                input_grad_ = std::make_unique<TensorType>( device, input_shape, this->getName() + ".input.grad" );
            }
        }

        void onTrainingModeChanging( TrainingMode training_mode ) override
        {
            operation_->setTrainingMode( training_mode );

            auto is_training = (training_mode == TrainingMode::Normal);

            if ( is_training && kv_cache_op_ && cache_initialized_ )
            {
                // Entering training mode invalidates any active decode session.
                kv_cache_op_->resetKvCache();
                cache_initialized_ = false;
                decode_active_ = false;
            }
        }

    private:
        GqaConfig config_;
        shape_t max_input_shape_;

        std::unique_ptr<IExecutionContext> context_{ nullptr };
        std::shared_ptr<OpType> operation_{ nullptr };

        // Non-owning capability interface pointers. Lifetime tied to operation_.
        // Resolved once in onBuilding(). Null for backends that do not implement
        // the corresponding interface (e.g. CPU).
        IKvCacheLifecycle* kv_cache_op_{ nullptr };
        IKvInference* positional_op_{ nullptr };

        // KV cache session state.
        bool cache_initialized_{ false };
        bool decode_active_{ false };

        std::unique_ptr<TensorType> output_{ nullptr };
        std::optional<TensorType> output_view_;
        std::unique_ptr<TensorType> input_grad_{ nullptr };
        std::unique_ptr<TensorType> decode_output_{ nullptr };

        // ====================================================================
        // Private helpers
        // ====================================================================

        /**
         * @brief Validate that the input tensor has the expected GQA-packed QKV shape.
         *
         * Expected trailing dimension:
         *   (num_heads + 2 * num_kv_heads) * head_dim
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
            const int64_t expected = (config_.getNumHeads() + 2 * config_.getNumKvHeads()) * head_dim;

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
            operation_ = std::make_shared<OpType>(
                this->getExecutionContext(), config_ );

            if ( !operation_ )
            {
                throw std::runtime_error( "GroupedQueryAttention: failed to create operation." );
            }
        }
    };
}
