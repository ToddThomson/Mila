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
import Compute.Observation;
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
     * supportsKvCache() indicates whether the underlying operation implements
     * both IPositionalUnaryOp (prefill/decode dispatch) and IKvCacheLifecycle
     * (cache init/reset). Both pointers are resolved once at build time.
     *
     * The cache self-initializes on the first prefill/forward; resetKvCache() is
     * the public hook the owning decoder layer / transformer drives to start a new
     * generation session.
     *
     * REVIEW: resetKvCache() is public (initialization stays internal to prefill).
     * When TransformerBase<> is introduced as the common base for GptTransformer,
     * LlamaTransformer, MistralTransformer etc., revisit whether it should
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
         * prefill session -- no explicit session management required by callers.
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
                        max_input_shape_[ 0 ],
                        max_input_shape_[ 1 ] );
                    cache_initialized_ = true;
                }

                // REVIEW: dead branch -- for the KV-cache backend forward() returns an
                // un-computed output_view_ and is unreached in the validated path (Llama/Gemma
                // drive prefill()/decode() directly). The retired call below used a stale 3-arg
                // prefill(input, output, 0) signature that no longer exists. Retire-vs-wire is
                // tracked: see BACKLOG, "GroupedQueryAttention::forward standalone path is a no-op stub".
                // positional_op_->prefill( input, *output_view_, 0 );
                return *output_view_;
            }

            operation_->forward( input, *output_view_ );

            this->publish( ComputePass::Forward, "output", *output_view_ );

            return *output_view_;
        }

        /**
         * @brief Not implemented -- GQA is inference only, and this is where that is decided.
         *
         * No GQA backend implements a backward pass. The refusal is stated here, at the
         * component's own boundary, rather than reached by dispatching into an operation
         * whose body is an unconditional throw: the built/training-mode guards below it were
         * checking preconditions for a call that could never succeed, and the tail after the
         * dispatch was unreachable code the compiler reported.
         *
         * The retired sketch ran on the expanded [B,NH,T,HS] layout that CudaGqaOp no longer
         * allocates; whether a real backward keeps that layout for training or is derived on
         * the compact NKV layout is an open design question recorded in GqaMemory.md.
         *
         * MultiHeadAttention implements both directions and is the component to train through.
         *
         * @throws std::runtime_error Always.
         */
        TensorType& backward( const TensorType&, const TensorType& )
        {
            throw std::runtime_error(
                "GroupedQueryAttention::backward is not implemented -- GQA is inference only. "
                "Train through MultiHeadAttention, or drive GQA with prefill()/decode()." );
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
         * @param q               Query tensor [B, T_chunk, Q * head_dim].
         * @param k               Key tensor [B, T_chunk, KV * head_dim].
         * @param v               Value tensor [B, T_chunk, KV * head_dim].
         * @param position_offset Absolute position of the first token in this chunk.
         * @return Reference to component-owned output tensor.
         */
        TensorType& prefill( const TensorType& q, const TensorType& k, const TensorType& v, dim_t position_offset )
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

            const dim_t B = q.shape()[ 0 ];
            const dim_t T_actual = q.shape()[ 1 ];
            shape_t output_shape = { B, T_actual, config_.getModelDim() };

            if ( output_view_->shape() != output_shape )
            {
                output_view_.emplace( output_->view( output_shape ) );
            }

            positional_op_->prefill( q, k, v, *output_view_, position_offset );

            this->publish( ComputePass::Prefill, "output", *output_view_ );

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
         * @param q               Query tensor [B, 1, Q * head_dim].
         * @param k               Key tensor [B, 1, KV * head_dim].
         * @param v               Value tensor [B, 1, KV * head_dim].
         * @param position_offset Absolute position of the token (0-based).
         * @return Reference to component-owned single-token output tensor.
         */
        TensorType& decode( const TensorType& q, const TensorType& k, const TensorType& v, dim_t position_offset )
        {
            if ( !this->isBuilt() )
                throw std::runtime_error(
                    "GroupedQueryAttention must be built before calling decode()." );

            //validateConcatenatedQKVShape( input.shape() );

            if ( positional_op_ && cache_initialized_ )
            {
                positional_op_->decode( q, k, v, *decode_output_, position_offset );
                decode_active_ = true;

                this->publish( ComputePass::Decode, "output", *decode_output_ );

                return *decode_output_;
            }

            // Fallback -- backend does not support KV caching or cache not yet initialized.

            // REVIEW: The Fallback here is stale and needs to be reviewed for correctness.

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
         * IPositionalUnaryOp and IKvCacheLifecycle.
         *
         * Resolved once at build time. CPU backends return false; CUDA backends
         * return true when CudaGroupedQueryAttentionOp is in use.
         */
        bool supportsKvCache() const noexcept
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

        /**
         * @brief Route the BF16 prefill through the fused FlashAttention kernel.
         *
         * Only the CUDA BF16 ops honor this -- the unbounded/global kernel and the bounded
         * sliding-window ring variant; other backends and FP32 ignore it. The transformer
         * couples this to the shared preatt/att workspace width (flash on -> the
         * O(chunk x T_ctx) score buffer is reclaimed), so it must be set consistently with
         * that sizing -- a narrow workspace with flash off would overflow.
         * See GqaFlashAttention.md 5.6.
         */
        void setUseFlashPrefill( bool enabled )
        {
            if constexpr ( TDeviceType == DeviceType::Cuda )
                operation_->setUseFlashPrefill( enabled );
        }

        /**
         * @brief Route the BF16 decode through the fused decode-attention kernel.
         *
         * Only the CUDA BF16 ops honor this; other backends, FP32, and unsupported
         * geometries keep the cuBLASLt decode pipeline. Unlike setUseFlashPrefill
         * there is no workspace-width coupling -- the fused path draws its split-K
         * scratch from the shared execution-context buffer at decode time.
         */
        void setUseFlashDecode( bool enabled )
        {
            if constexpr ( TDeviceType == DeviceType::Cuda )
                operation_->setUseFlashDecode( enabled );
        }

        /**
         * @brief Reset the KV cache for a new generation session.
         *
         * Drops any active decode state so the next prefill starts a fresh
         * sequence. A no-op on backends without KV-cache support, and harmless
         * when no decode session is active.
         */
        void resetKvCache()
        {
            if ( kv_cache_op_ && cache_initialized_ )
            {
                kv_cache_op_->resetKvCache();
                cache_initialized_ = false;
                decode_active_ = false;
            }
        }

        /**
         * @brief Rewind the cache fill position for prompt-prefix reuse
         * (PromptCaching.md). Unlike resetKvCache() the cache session stays live:
         * initialization state and device contents are untouched, and positions
         * [0, position) remain valid for a subsequent prefillFrom.
         *
         * @return true when the underlying operation accepted the rewind.
         */
        bool rewindKvCache( dim_t position )
        {
            if ( !kv_cache_op_ || !cache_initialized_ )
                return false;

            return kv_cache_op_->rewindKvCache( position );
        }

        // ====================================================================
        // Serialization
        // ====================================================================

        void save_( ModelArchive&, SerializationMode ) const override
        {
            // Deliberately empty: parameterCount() is 0. The projections that carry the
            // weights are Linear components owned by the enclosing block, and they save
            // themselves.
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

        dim_t parameterCount() const override
        {
            return 0;
        }

        /**
         * @brief Install a shared prefill-output slot (activation pooling).
         *
         * Must be called before build(): onBuilding then skips the prefill output
         * self-allocation after validating the slot's storage covers the build
         * shape; prefill() already always returns a shape-adjusted view. The tiny
         * per-token decode output (decode_output_) stays component-owned. Mirrors
         * Linear::installSharedWeight; self-allocation remains the default. The
         * slot is owned and memory-accounted by the installer.
         */
        void installSharedOutput( std::shared_ptr<TensorType> output )
        {
            if ( this->isBuilt() )
                throw std::logic_error(
                    "GroupedQueryAttention '" + this->getName() + "': installSharedOutput must be called before build()" );

            output_ = std::move( output );
            output_installed_ = true;
        }

        /**
         * @brief What onBuilding() would allocate for this context, without allocating.
         *
         * The KV cache -- the dominant, context-scaling term -- is reported by the
         * operation, which is where it is allocated.
         * See Specifications/MemoryFootprint.md.
         */
        MemoryStats getRequiredMemory( const BuildContext& context ) const override
        {
            const auto& input_shape = context.inputShape();

            validateConcatenatedQKVShape( input_shape );

            const dim_t batch = input_shape[ 0 ];
            const dim_t model_dim = config_.getModelDim();

            MemoryStats stats;

            stats.device_state_bytes += operation_->getRequiredStateMemorySize( context );

            if ( context.isInferenceMode() )
            {
                // Decode output is T=1 and always component-owned -- never pooled.
                stats.device_state_bytes += storageBytes<TComputePrecision>( batch * model_dim );

                // Prefill output is one chunk wide, not the whole context.
                if ( !output_installed_ && !context.hasInstalledOutput() )
                {
                    stats.device_state_bytes +=
                        storageBytes<TComputePrecision>( batch * context.getPrefillSize() * model_dim );
                }
            }
            else
            {
                shape_t output_shape = input_shape;
                output_shape.back() = model_dim;

                stats.device_state_bytes +=
                    storageBytes<TComputePrecision>( elementCount( output_shape ) );
            }

            return stats;
        }

        std::vector<const ITensor*> getOutputs() const override
        {
            std::vector<const ITensor*> outputs;

            if ( output_ != nullptr )
            {
                outputs.push_back( output_.get() );
            }

            if ( decode_output_ != nullptr )
            {
                outputs.push_back( decode_output_.get() );
            }

            return outputs;
        }

        std::vector<ObservableStage> getObservableStages() const override
        {
            return { { "output",
                ComputePassMask{ ComputePass::Forward, ComputePass::Prefill, ComputePass::Decode } } };
        }

        MemoryStats getMemoryStats() const override
        {
            MemoryStats stats;

            stats.device_state_bytes += operation_->getStateMemorySize();

            // An installed shared output slot is owned and counted by the installer.
            if ( output_ != nullptr && !output_installed_ )
                stats.device_state_bytes += output_->getStorageSize();

            if ( decode_output_ != nullptr )
                stats.device_state_bytes += decode_output_->getStorageSize();

            if ( input_grad_ != nullptr )
                stats.device_gradient_bytes += input_grad_->getStorageSize();

            return stats;
        }

        std::string toString() const override
        {
            const dim_t head_dim = config_.getModelDim() / config_.getNumHeads();
            const dim_t group_size = config_.getNumHeads() / config_.getNumKvHeads();

            std::ostringstream oss;
            oss << "--------------------\n";
            oss << "GroupedQueryAttention: " << this->getName() << "\n";
            oss << "Device Id: " << this->getExecutionContext()->getDeviceId().toString() << "\n";
            oss << "Model dimension: " << config_.getModelDim() << "\n";
            oss << "Num Q heads: " << config_.getNumHeads() << "\n";
            oss << "Num KV heads: " << config_.getNumKvHeads() << "\n";
            oss << "Head size: " << head_dim << "\n";
            oss << "Group size (Q heads per KV head): " << group_size << "\n";
            oss << "Decode path: " << (supportsKvCache() ? "KV cache (fast)" : "fallback (forward)") << "\n";
            oss << "Parameter count: " << parameterCount() << "\n";

            return oss.str();
        }

        // ====================================================================
        // Config accessors
        // ====================================================================

        dim_t getModelDim()   const noexcept
        {
            return config_.getModelDim();
        }
        dim_t getNumHeads()   const noexcept
        {
            return config_.getNumHeads();
        }
        dim_t getNumKvHeads() const noexcept
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
                    kv_cache_op_->initializeKvCache( B, T );
                    cache_initialized_ = true;
                }

                // Decode path output: T=1. Always component-owned (tiny, not pooled).
                shape_t decode_output_shape = { input_shape[ 0 ], 1, config_.getModelDim() };
                decode_output_ = std::make_unique<TensorType>( device, decode_output_shape, this->getName() + ".output_decode" );

                // Prefill path output -- sized for one prefill chunk at a time.
                shape_t output_shape = { B, context.getPrefillSize(), config_.getModelDim() };

                if ( output_installed_ )
                {
                    dim_t needed = 1;
                    for ( auto d : output_shape )
                        needed *= d;

                    if ( !output_ || output_->size() < needed )
                        throw std::invalid_argument(
                            "GroupedQueryAttention '" + this->getName() + "': installed shared output slot is smaller than the build shape requires" );
                }
                else
                {
                    output_ = std::make_shared<TensorType>( device, output_shape, this->getName() + ".output_prefill" );
                }

                // View the BUILD shape, not the slot shape -- an installed slot may be wider.
                output_view_.emplace( output_->view( output_shape ) );
            }
            else
            {
                // Training -- full sequence output buffer (never pooled).
                shape_t output_shape = input_shape;
                output_shape.back() = config_.getModelDim();
                output_ = std::make_shared<TensorType>( device, output_shape, this->getName() + ".output" );
                output_view_.emplace( output_->view( output_shape ) );

                // Input gradient -- same shape as packed QKV input.
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

        // Self-allocated at build, or an installed shared slot (installSharedOutput)
        // that the component views a prefix of. decode_output_ is always owned.
        std::shared_ptr<TensorType> output_{ nullptr };
        bool output_installed_{ false };
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

            const dim_t head_dim = config_.getModelDim() / config_.getNumHeads();
            const dim_t trailing = shape.back();
            const dim_t expected = (config_.getNumHeads() + 2 * config_.getNumKvHeads()) * head_dim;

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
