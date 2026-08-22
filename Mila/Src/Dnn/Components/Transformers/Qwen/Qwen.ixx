/**
 * @file Qwen.ixx
 * @brief Qwen 3.8 decoder-only transformer network (inference: prefill + decode).
 *
 * The heterogeneous layer list is copied from GemmaTransformer, which is the one structure
 * in the tree that already solves this problem: build two block types, store both as
 * IDecoderLayer*, and drive them polymorphically (one virtual call per layer per token step,
 * negligible against the per-layer GEMMs). What is NOT copied is Gemma's `bool kGlobal`
 * flag: that is right for one mixer at two geometries, and Qwen's two kinds are different
 * mixers (Specifications/Qwen3.8.md section 8, Phase 2 revision).
 *
 * ## Two block kinds, two memory models
 *
 * The published 27B geometry (`full_attention_interval: 4`) builds three QwenDeltaNetBlocks
 * per QwenAttentionBlock. The two kinds differ in more than their mixer:
 *
 *  - The attention layers hold a KV cache that GROWS with context, and share the pooled
 *    workspace and the GQA transient this network owns.
 *  - The DeltaNet layers hold a fixed-size recurrent state plus a short convolution window,
 *    neither of which depends on context length, and self-allocate their transients. Their
 *    slots share nothing with the attention workspace's, so pooling them means a SECOND
 *    workspace struct rather than a wider one -- a memory optimization, still owed.
 *
 * One consequence reaches the product rather than the code: `rewindKvCache` asks every layer,
 * and a DeltaNet layer always refuses, because a recurrent state is a lossy summary that
 * cannot be rolled back to an earlier position. PROMPT-PREFIX REUSE IS THEREFORE UNAVAILABLE
 * for any configuration containing these layers.
 *
 * ## What this network cannot do yet
 *
 * Prefill runs the DeltaNet recurrence sequentially over the chunk (there is no chunked
 * UT-transform kernel yet), so long-prompt prefill on those layers is O(T) in steps.
 *
 * ## Untied tables
 *
 * Qwen 3.8 sets `tie_word_embeddings: false` -- two full-size tables, 1.271 B parameters
 * each. So unlike Gemma there is no weight-tying path here, and the two tables take
 * DIFFERENT policies from the plan: the embedding stays BF16 (section 5 moves it to host
 * memory, where it is a gather and never a multiply) while the head is FP4 because it drives
 * sampling and is re-read in full every decode step.
 *
 * Host-resident `embed_tokens` is section 8 item 6 and is NOT implemented here: the table is
 * device-resident, which costs the 0.610 GiB that section 5's budget hands back.
 */

module;
#include <string>
#include <vector>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <cstdint>
#include <format>
#include <algorithm>
#include <type_traits>

export module Dnn.Components.QwenTransformer;

import Dnn.Components.QwenConfig;
import Dnn.Components.QwenBlock;
import Dnn.Components.QwenDeltaNetBlock;
import Dnn.Components.QwenPrecisionPlan;
import Dnn.Components.IDecoderLayer;

import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Logging.Logger;
import Dnn.LanguageNetwork;
import Dnn.Component;
import Dnn.ComponentType;
import Dnn.ModelType;
import Dnn.Components.TokenEmbedding;
import Dnn.Components.Linear;
import Dnn.Components.RmsNorm;
import Dnn.Quantization.Weight.Policies;
import Dnn.Quantization.Weight.PrecisionPlan;
import Dnn.Quantization.KvCache.Policy;
import Compute.Device;
import Compute.DeviceType;
import Compute.DeviceId;
import Compute.DeviceTypeTraits;
import Compute.GqaState;
import Compute.CpuMemoryResource;
#ifdef MILA_HAS_CUDA
import Compute.CudaPinnedMemoryResource;
#endif
import Compute.ExecutionContext;
import Compute.ExecutionContextFactory;
import Serialization.ModelArchive;
import Serialization.Mode;
import Serialization.Metadata;
import Serialization.PretrainedReader;
import Serialization.Tensor;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;
    using namespace Mila::Dnn::Quant::Weight;
    using namespace Mila::Dnn::Quant::KvCache;

    // The chunk rungs, largest first, and the floor below which the GEMM M dimension is
    // tensor-core-hostile on top of the weight re-read.
    //
    // These are the Gemma values, INHERITED RATHER THAN MEASURED: nothing has run this
    // model, so a Qwen-specific ladder would be a guess dressed as a decision. The row-cost
    // model below is Qwen's own, so the rung this picks already reflects Qwen's geometry;
    // what is unvalidated is the ladder and the budget, and Phase 5 is where they get
    // measured against the 12 GiB card.
    inline constexpr dim_t kQwenPrefillChunkRungs[] = { 1024, 512, 256, 128, 64 };
    inline constexpr dim_t kQwenPrefillChunkFloor = 64;

    // Activation budget for one prefill pass. Section 5 allots 0.45 GiB to activations and
    // scratch at the 16K baseline, well under this cap; the cap is the ceiling a chunk may
    // not cross, not the expected occupancy.
    inline constexpr dim_t kQwenPrefillActivationBudgetBytes = dim_t{ 1536 } * 1024 * 1024;

    /**
     * @brief Qwen 3.8 transformer (decoder-only) for autoregressive inference.
     *
     * Graph: TokenEmbedding -> [QwenAttentionBlock | QwenDeltaNetBlock] x N -> RmsNorm ->
     * Linear (lm_head).
     *
     * @tparam TWeightPlan A precision plan (or a bare policy, which lifts to a uniform one).
     *                     The network requires the embedding and head roles on top of what
     *                     its blocks require -- no block instantiates a table.
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision,
        typename TWeightPlan = NoWeightQuant, KvCachePolicy TKvCachePolicy = NoKvCompression>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
              && EmbeddingPrecisionRole<PrecisionPlanFor<TWeightPlan>>
              && LanguageModelHeadRole<PrecisionPlanFor<TWeightPlan>>
    class QwenTransformer : public LanguageNetwork<TDeviceType, TPrecision>
    {
    public:
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using NetworkBase = LanguageNetwork<TDeviceType, TPrecision>;
        using TensorType = Tensor<TPrecision, MR>;

        using PrecisionPlan = PrecisionPlanFor<TWeightPlan>;

        // The two tables take different policies because they are different allocations.
        // Gemma's TableQuantizationPolicy exists to keep ONE shared table consistent across
        // two consumers; with tying off there is nothing to keep consistent, and section 5
        // prices the two separately.
        using TokenEmbeddingType = TokenEmbedding<TDeviceType, dtype_t::INT32, TPrecision,
            typename PrecisionPlan::EmbeddingTable>;
        using LmHeadLinearType = Linear<TDeviceType, TPrecision, typename PrecisionPlan::LanguageModelHead>;
        using RmsNormType = RmsNorm<TDeviceType, TPrecision>;

        using AttentionBlockType = QwenAttentionBlock<TDeviceType, TPrecision, TWeightPlan, TKvCachePolicy>;
        using DeltaNetBlockType = QwenDeltaNetBlock<TDeviceType, TPrecision, TWeightPlan>;
        using DecoderLayerType = IDecoderLayer<TDeviceType, TPrecision>;
        using TokenIndexType = Tensor<dtype_t::INT32, MR>;
        using ComponentPtr = typename NetworkBase::ComponentPtr;

        explicit QwenTransformer( const std::string& name, const QwenConfig& config, DeviceId device_id )
            : NetworkBase( name ), config_( config ), exec_context_( createExecutionContext( device_id ) )
        {
            config_.validate();

            if ( device_id.type != TDeviceType )
            {
                throw std::invalid_argument(
                    std::format( "QwenTransformer: device type mismatch: expected {}, got {}",
                        deviceTypeToString( TDeviceType ),
                        deviceTypeToString( device_id.type ) ) );
            }

            createGraph();

            this->setExecutionContext( exec_context_.get() );
        }

        ~QwenTransformer() override = default;

        // ====================================================================
        // Compute interface (inference-only)
        // ====================================================================

        TensorType& forward( const TokenIndexType& /*input*/ ) override
        {
            throw std::runtime_error(
                "QwenTransformer is inference-only; use prefill()/decode() for autoregressive generation." );
        }

        TokenIndexType& backward( const TokenIndexType& /*input*/, const TensorType& /*output_grad*/ ) override
        {
            throw std::runtime_error( "QwenTransformer is inference-only; backward() is not implemented." );
        }

        TensorType& prefill( const TokenIndexType& input ) override
        {
            return prefillFrom( input, 0 );
        }

        void setStageProbe( typename NetworkBase::StageProbe probe ) override
        {
            stage_probe_ = std::move( probe );
        }

        /**
         * @brief Chunked prefill starting at an absolute position (prompt-prefix reuse).
         *
         * `input` is the FULL prompt tensor, so token index and absolute position coincide.
         *
         * Prefix reuse will not survive Phase 3: a recurrent state cannot be rewound (section
         * 7), so with 48 of 64 layers refusing, the all-or-nothing rewind always fails and
         * every start_offset above 0 becomes unreachable through the model layer. The path
         * stays because it is correct for the layers that do cache, and because snapshot and
         * restore -- which does work on a constant-size state -- is Phase 6.
         */
        TensorType& prefillFrom( const TokenIndexType& input, dim_t start_offset ) override
        {
            if ( !this->isBuilt() )
                throw std::runtime_error( "QwenTransformer must be built before calling prefill()." );

            const int64_t B = input.shape()[ 0 ];
            const int64_t T_prompt = input.shape()[ 1 ];

            if ( start_offset < 0 || start_offset >= T_prompt )
                throw std::invalid_argument( std::format(
                    "QwenTransformer::prefillFrom: start_offset {} must lie inside the prompt (length {})",
                    start_offset, T_prompt ) );

            int64_t offset = start_offset;
            int64_t T_last = 0;

            TensorType* last_block_out = nullptr;

            while ( offset < T_prompt )
            {
                const int64_t T_actual = std::min<int64_t>( prefill_chunk_size_, T_prompt - offset );
                T_last = T_actual;

                auto chunk_input = input.view( shape_t{ B, T_actual }, offset );

                TensorType* block_input = &token_embedding_->forward( chunk_input );

                if ( stage_probe_ )
                {
                    stage_probe_( "embedding", *block_input );
                }

                int layer_index = 0;

                for ( auto* layer : layers_ )
                {
                    auto& block_out = layer->prefill( *block_input, offset );
                    block_input = &block_out;

                    if ( stage_probe_ )
                    {
                        stage_probe_( std::format( "layer_{}", layer_index ), *block_input );
                    }

                    ++layer_index;
                }

                last_block_out = block_input;
                offset += T_actual;
            }

            // Last position only. At 248,320 vocabulary a single FP32 logit row is 0.95 MiB,
            // so materializing a whole chunk's rows would allocate 0.48 GiB at chunk 512 --
            // the first of the two constraints section 5 draws from the decode budget.
            dim_t last_pos_offset = (T_last - 1) * config_.getModelDim();
            auto last_pos = last_block_out->view(
                shape_t{ B, 1, config_.getModelDim() }, last_pos_offset );

            normalized_ptr_ = &final_rmsnorm_->forward( last_pos );
            logits_ptr_ = &lm_head_->forward( *normalized_ptr_ );

            return *logits_ptr_;
        }

        TensorType& decode( const TokenIndexType& input, dim_t position ) override
        {
            if ( !this->isBuilt() )
                throw std::runtime_error( "QwenTransformer must be built before calling decode()." );

            TensorType* block_input = &token_embedding_->forward( input );

            for ( auto* layer : layers_ )
            {
                auto& block_out = layer->decode( *block_input, position );
                block_input = &block_out;
            }

            normalized_ptr_ = &final_rmsnorm_->forward( *block_input );
            logits_ptr_ = &lm_head_->forward( *normalized_ptr_ );

            return *logits_ptr_;
        }

        // ====================================================================
        // KV-cache orchestration
        // ====================================================================

        void resetKVCache()
        {
            for ( auto* layer : layers_ )
                layer->resetKVCache();
        }

        /**
         * @brief Rewind every layer's cache to `position`. All-or-nothing to the caller.
         */
        bool rewindKvCache( dim_t position ) override
        {
            bool all_accepted = true;

            for ( auto* layer : layers_ )
                all_accepted = layer->rewindKvCache( position ) && all_accepted;

            return all_accepted;
        }

        // ====================================================================
        // Accessors / Diagnostics
        // ====================================================================

        MemoryStats getMemoryStats() const override
        {
            MemoryStats stats;

            for ( const auto& child : this->getComponents() )
                stats += child->getMemoryStats();

            stats.device_state_bytes += block_workspace_.deviceStorageBytes();
            stats.device_state_bytes += gqa_workspace_.deviceStorageBytes();

            // No weight-tying correction: the tables are untied, so nothing is shared and
            // nothing is double-counted.
            return stats;
        }

        /**
         * @brief What build( context ) would allocate for the whole model, without allocating.
         */
        MemoryStats getRequiredMemory( const BuildContext& context ) const override
        {
            const auto& input_shape = context.inputShape();
            const dim_t B = input_shape[ 0 ];
            const dim_t T = input_shape[ 1 ];

            // The chunk must be resolved before recursing: it sizes every block's activation
            // scratch and the pooled workspace below.
            const dim_t prefill_chunk = resolvePrefillChunkSize( B, T );

            BuildContext block_context =
                BuildContext( shape_t{ B, T, config_.getModelDim() },
                    context.getRuntimeMode(), context.shouldInitializeParameters() )
                .withPrefillSize( prefill_chunk )
                .withInstalledOutput( context.isInferenceMode() );

            const shape_t final_shape = context.isInferenceMode()
                ? shape_t{ B, 1, config_.getModelDim() }
                : shape_t{ B, T, config_.getModelDim() };

            BuildContext final_context(
                final_shape, context.getRuntimeMode(), context.shouldInitializeParameters() );

            const std::string n = this->getName();

            MemoryStats stats;

            stats += this->template getComponentAs<TokenEmbeddingType>( n + ".temb" )
                ->getRequiredMemory( context );

            for ( dim_t i = 0; i < config_.getNumLayers(); ++i )
            {
                if ( config_.isFullAttentionLayer( i ) )
                {
                    stats += this->template getComponentAs<AttentionBlockType>( blockName( i ) )
                        ->getRequiredMemory( block_context );
                }
                else
                {
                    stats += this->template getComponentAs<DeltaNetBlockType>( blockName( i ) )
                        ->getRequiredMemory( block_context );
                }
            }

            stats += this->template getComponentAs<RmsNormType>( n + ".rmsn_final" )
                ->getRequiredMemory( final_context );

            stats += this->template getComponentAs<LmHeadLinearType>( n + ".lm_head" )
                ->getRequiredMemory( final_context );

            if ( context.isInferenceMode() )
            {
                stats.device_state_bytes += blockWorkspaceBytes( B, prefill_chunk );
                stats.device_state_bytes += gqaWorkspaceBytes( B, T, prefill_chunk );
            }

            // RoPE cos/sin caches are process-wide, deduplicated by RopeCacheRegistry on
            // (theta, max_seq_len, head_dim). Every block above reported one, but Qwen has a
            // single base and a single head width, so exactly one is ever allocated.
            stats.device_state_bytes -=
                std::max<dim_t>( config_.getNumFullAttentionLayers() - 1, 0 )
                * ropeCacheBytes( config_.getHeadDim() );

            return stats;
        }

        /**
         * @brief The prefill chunk this context length would use, and the one it could have used.
         *
         * Pure arithmetic over the config: no device is touched and nothing is allocated, so
         * a caller may ask before the network is built -- which is the point, since choosing
         * a context length by memory alone can land on one where the chunk has walked down
         * to its floor.
         */
        PrefillChunking prefillChunking( dim_t B, dim_t T_ctx ) const
        {
            PrefillChunking chunking;

            // Short-context build: the whole context is a single chunk, and no rung applies.
            if ( T_ctx < kQwenPrefillChunkFloor )
            {
                chunking.chunk_rows = T_ctx;
                chunking.unconstrained_chunk_rows = T_ctx;

                return chunking;
            }

            const dim_t row_cost = computeChunkRowCostBytes( B, T_ctx );
            const dim_t kv_bytes = prefillKvBytes( B, T_ctx );
            const dim_t budget = ( kQwenPrefillActivationBudgetBytes > kv_bytes )
                ? ( kQwenPrefillActivationBudgetBytes - kv_bytes )
                : dim_t{ 0 };

            for ( dim_t candidate : kQwenPrefillChunkRungs )
            {
                if ( candidate > T_ctx )
                    continue;

                // The first rung the context admits at all, budget aside.
                if ( chunking.unconstrained_chunk_rows == 0 )
                    chunking.unconstrained_chunk_rows = candidate;

                if ( row_cost * candidate <= budget )
                {
                    chunking.chunk_rows = candidate;

                    return chunking;
                }
            }

            // Nothing fit, including the floor. The floor is used regardless -- the caller
            // that builds warns, the caller that is choosing a context reads the flag.
            chunking.chunk_rows = kQwenPrefillChunkFloor;
            chunking.fits_activation_budget = false;

            return chunking;
        }

        ModelType getModelType() const
        {
            return ModelType::Qwen;
        }

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << std::endl;
            oss << "Qwen Network: " << this->getName() << std::endl;
            oss << "Device: " << this->getDeviceId().toString() << std::endl;
            oss << config_.toString();

            if ( this->isBuilt() )
            {
                oss << "  Parameters: " << this->parameterCount() << std::endl;
                oss << "  Prefill chunk: " << prefill_chunk_size_ << std::endl;
            }

            return oss.str();
        }

        IExecutionContext* getExecutionContext() const
        {
            return NetworkBase::getExecutionContext();
        }

        void loadParameters( PretrainedModelReader& reader )
        {
            const int device_index = this->getExecutionContext()->getDeviceId().index;

            auto consume = [&]( const std::string& full_name, const Serialization::ITensorBlob& blob )
            {
                auto [component_path, param_name] = parseParameterPath( full_name );

                ComponentPtr target = this->findComponent( component_path );
                target->loadParameter( param_name, blob );

                // The reader reuses its pinned staging slot as soon as this returns, and a
                // quantize-on-load H2D is async on the op stream and does not self-
                // synchronize, so force completion here.
                if constexpr ( TDeviceType == DeviceType::Cuda )
                {
                    this->getExecutionContext()->synchronize();
                }
            };

#ifdef MILA_HAS_CUDA
            if constexpr ( TDeviceType == DeviceType::Cuda )
            {
                reader.streamTensorBlobs<CudaPinnedMemoryResource>( consume, device_index );
            }
            else
#endif
            {
                reader.streamTensorBlobs<CpuMemoryResource>( consume );
            }

            if constexpr ( TDeviceType == DeviceType::Cuda )
            {
                this->getExecutionContext()->synchronize();
            }

            // No tying step: Qwen's tables are independent and both arrive from the file.
        }

    protected:

        void onBuilding( const BuildContext& context ) override
        {
            validateBuildContext( context );

            const auto& input_shape = context.inputShape();
            const dim_t B = input_shape[ 0 ];
            const dim_t T = input_shape[ 1 ];

            prefill_chunk_size_ = resolvePrefillChunkSize( B, T );

            // Blocks need the full context length so the attention can size its KV cache;
            // the block handles the prefill/decode split internally.
            BuildContext block_context =
                BuildContext( shape_t{ B, T, config_.getModelDim() },
                    context.getRuntimeMode(), context.shouldInitializeParameters() )
                .withPrefillSize( prefill_chunk_size_ );

            // Inference: final_rmsnorm and lm_head only process the last position.
            shape_t final_shape = context.isInferenceMode()
                ? shape_t{ B, 1, config_.getModelDim() }
                : shape_t{ B, T, config_.getModelDim() };

            BuildContext final_context(
                final_shape, context.getRuntimeMode(), context.shouldInitializeParameters() );

            token_embedding_ = this->template getComponentAs<TokenEmbeddingType>( this->getName() + ".temb" );
            token_embedding_->build( context );

            if ( context.isInferenceMode() )
                allocateBlockWorkspace( B );

            layers_.clear();
            layers_.reserve( static_cast<size_t>( config_.getNumLayers() ) );

            for ( dim_t i = 0; i < config_.getNumLayers(); ++i )
            {
                if ( config_.isFullAttentionLayer( i ) )
                {
                    auto block = this->template getComponentAs<AttentionBlockType>( blockName( i ) );

                    if ( context.isInferenceMode() )
                        block->installSharedWorkspace( block_workspace_ );

                    block->build( block_context );

                    if ( context.isInferenceMode() )
                    {
                        // The full-attention layers are unbounded, so the flash decision is
                        // one number for the whole stack and MUST agree with
                        // prefillScoreWidth() below.
                        block->setUseFlashPrefill( useFlashPrefillForContext( T ) );
                        block->setUseFlashDecode( true );
                    }

                    layers_.push_back( static_cast<DecoderLayerType*>( block.get() ) );
                }
                else
                {
                    // The DeltaNet block self-allocates. Its transients are shaped by the
                    // DeltaNet geometry -- key/value widths, a per-head view, a recurrent
                    // state -- and share nothing with the attention workspace's slots, so
                    // pooling them would mean a second workspace struct rather than a
                    // wider one. Qwen3.8.md section 8 already anticipates that; it is a
                    // memory optimization, and it is not free to get wrong.
                    auto block = this->template getComponentAs<DeltaNetBlockType>( blockName( i ) );

                    block->build( block_context );

                    layers_.push_back( static_cast<DecoderLayerType*>( block.get() ) );
                }
            }

            final_rmsnorm_ = this->template getComponentAs<RmsNormType>( this->getName() + ".rmsn_final" );
            final_rmsnorm_->build( final_context );

            lm_head_ = this->template getComponentAs<LmHeadLinearType>( this->getName() + ".lm_head" );
            lm_head_->build( final_context );

            if ( context.isInferenceMode() )
                allocateAndWireGqaWorkspace( B, T );

            normalized_ptr_ = nullptr;
            logits_ptr_ = nullptr;
        }

        void onTrainingModeChanging( TrainingMode training_mode ) override
        {
            NetworkBase::onTrainingModeChanging( training_mode );
        }

        void save_( ModelArchive& archive, SerializationMode /*mode*/ ) const override
        {
            SerializationMetadata meta = config_.toMetadata();
            meta.set( "type", "QwenTransformer" )
                .set( "version", int64_t( 1 ) )
                .set( "name", this->getName() );

            archive.writeMetadata( "transformer_meta.json", meta );
        }

    private:

        QwenConfig config_;

        // Tuned prefill chunk -- single source of truth, set in onBuilding and threaded to
        // child components via BuildContext::withPrefillSize().
        dim_t prefill_chunk_size_{ 0 };

        // Diagnostic only; unset in normal operation, where it costs one null check per layer.
        typename NetworkBase::StageProbe stage_probe_{};

        std::shared_ptr<TokenEmbeddingType> token_embedding_{ nullptr };
        // Non-owning, polymorphic view of the layer list; the concrete blocks are owned by
        // the component tree (addComponent). Valid after build. Held as IDecoderLayer* even
        // though every element is an attention block today -- that is what the DeltaNet block
        // slots into without touching the loops.
        std::vector<DecoderLayerType*> layers_;
        std::shared_ptr<RmsNormType> final_rmsnorm_{ nullptr };
        std::shared_ptr<LmHeadLinearType> lm_head_{ nullptr };

        QwenAttentionBlockWorkspace<TDeviceType, TPrecision> block_workspace_{};

        // Shared GQA transient workspace -- inference only, owned here, shared across all
        // attention blocks (exactly one layer is live at a time on the sequential path).
        QwenGqaWorkspace<TDeviceType, TPrecision> gqa_workspace_{};

        // Activation pointers -- valid between prefill/decode and the next call.
        TensorType* normalized_ptr_{ nullptr };
        TensorType* logits_ptr_{ nullptr };

        // Declared last so it is destroyed first -- cudaStreamSynchronize() fires in
        // releaseResources() before any tensor cudaFree() from the members above.
        std::unique_ptr<IExecutionContext> exec_context_{ nullptr };

        std::string blockName( dim_t layer_index ) const
        {
            return this->getName() + ".tf_layer_" + std::to_string( layer_index );
        }

        void createGraph()
        {
            TokenEmbeddingConfig embedding_config;
            embedding_config.withVocabSize( config_.getVocabSize() )
                .withEmbeddingDim( static_cast<size_t>( config_.getModelDim() ) );

            this->addComponent(
                std::make_shared<TokenEmbeddingType>( this->getName() + ".temb", embedding_config ) );

            // The heterogeneous layer list: three DeltaNet layers per full-attention layer
            // at the published interval of 4. Both kinds are stored as IDecoderLayer* and
            // driven polymorphically.
            for ( dim_t i = 0; i < config_.getNumLayers(); ++i )
            {
                if ( config_.isFullAttentionLayer( i ) )
                {
                    this->addComponent(
                        std::make_shared<AttentionBlockType>( blockName( i ), config_, std::nullopt ) );
                }
                else
                {
                    this->addComponent(
                        std::make_shared<DeltaNetBlockType>( blockName( i ), config_, std::nullopt ) );
                }
            }

            // Unit offset, like every stream norm in this family: Qwen scales by
            // (1 + weight) with weights stored zero-centered.
            auto rms_config = RmsNormConfig( shape_t{ config_.getModelDim() } )
                .withEpsilon( config_.getRMSNormEpsilon() )
                .withBias( false )
                .withUnitOffset( 1.0f );

            this->addComponent(
                std::make_shared<RmsNormType>( this->getName() + ".rmsn_final", rms_config, std::nullopt ) );

            // Language model head -- model_dim -> vocab_size, no bias, its OWN weight. The
            // tables are untied, so nothing replaces this allocation later.
            auto lm_head_config = LinearConfig( config_.getModelDim(), config_.getVocabSize() )
                .withBias( false );

            this->addComponent(
                std::make_shared<LmHeadLinearType>( this->getName() + ".lm_head", lm_head_config, std::nullopt ) );
        }

        // ====================================================================
        // Prefill chunk + shared workspaces
        // ====================================================================

        // Context length (>=) at which the BF16 attention layers switch from the cuBLASLt
        // prefill path to the fused FlashAttention kernel. Below it cuBLASLt is faster and
        // its O(chunk x T_ctx) score workspace still fits; at or above it that workspace is
        // the memory wall. 0 disables flash entirely.
        static constexpr dim_t kFlashPrefillMinContext = 16384;

        bool useFlashPrefillForContext( dim_t T_ctx ) const noexcept
        {
            return TPrecision == TensorDataType::BF16
                && kFlashPrefillMinContext > 0
                && T_ctx >= kFlashPrefillMinContext;
        }

        /**
         * @brief Width of the shared prefill score buffer.
         *
         * Every layer here is unbounded, so unlike Gemma there is no window-bounded residue
         * to keep the buffer alive for: with flash on, nothing reads it, and the chunk-scaled
         * O(chunk x T_ctx) term disappears from the row cost entirely. It is kept at one row
         * rather than zero so the allocation stays a valid tensor.
         */
        dim_t prefillScoreWidth( dim_t T_ctx ) const noexcept
        {
            return useFlashPrefillForContext( T_ctx ) ? dim_t{ 1 } : T_ctx;
        }

        // Max slot widths, shared by the workspace allocation and the row-cost model so the
        // two cannot drift apart.
        struct WorkspaceWidths
        {
            dim_t model_dim;
            dim_t hidden_dim;
            dim_t q_width;
            dim_t kv_width;
            dim_t qkv_width;

            // q + gate + q_normed + attn + gated + query_gate(2x); k + v + k_normed; the six
            // model_dim-wide stream-side slots; qkv; gate_up (2h) + ffn_act (h).
            dim_t totalRowElements() const
            {
                return 7 * q_width + 3 * kv_width + 6 * model_dim + qkv_width + 3 * hidden_dim;
            }
        };

        WorkspaceWidths computeWorkspaceWidths() const
        {
            WorkspaceWidths widths;
            widths.model_dim = config_.getModelDim();
            widths.hidden_dim = config_.getHiddenDimension();
            widths.q_width = config_.getQProjectionWidth();
            widths.kv_width = config_.getKVProjectionWidth();
            widths.qkv_width = config_.getPackedQKVWidth();

            return widths;
        }

        std::size_t blockWorkspaceBytes( dim_t B, dim_t prefill_chunk ) const
        {
            return storageBytes<TPrecision>(
                computeWorkspaceWidths().totalRowElements() * B * prefill_chunk );
        }

        std::size_t gqaWorkspaceBytes( dim_t B, dim_t T_ctx, dim_t prefill_chunk ) const
        {
            const dim_t NH = config_.getNumHeads();
            const dim_t HD = config_.getHeadDim();
            const dim_t score_width = prefillScoreWidth( T_ctx );

            const dim_t prefill_elements =
                ( 2 * B * NH * prefill_chunk * HD )            // q_permute, v_out
                + ( 2 * B * NH * prefill_chunk * score_width ); // preatt, att

            const dim_t decode_elements =
                ( 2 * B * NH * T_ctx )                         // preatt_decode, att_decode
                + ( B * NH * HD );                             // v_out_decode

            return storageBytes<TPrecision>( prefill_elements + decode_elements );
        }

        /**
         * @brief Bytes one RoPE cos/sin cache occupies for a given head width.
         *
         * MUST match CudaRopeOp::getRequiredStateMemorySize -- FP32 regardless of the model
         * precision, half the head dimension, two caches.
         */
        std::size_t ropeCacheBytes( dim_t head_dim ) const noexcept
        {
            const dim_t cache_elements = config_.getMaxSequenceLength() * ( head_dim / 2 );

            return static_cast<std::size_t>( cache_elements ) * sizeof( float ) * 2;
        }

        dim_t computeChunkRowCostBytes( dim_t B, dim_t T_ctx ) const
        {
            const dim_t precision_bytes =
                static_cast<dim_t>( TensorDataTypeTraits<TPrecision>::size_in_bytes );
            const dim_t NH = config_.getNumHeads();
            const dim_t HD = config_.getHeadDim();

            const dim_t workspace_bytes =
                computeWorkspaceWidths().totalRowElements() * B * precision_bytes;
            const dim_t attention_bytes =
                B * NH * ( 2 * prefillScoreWidth( T_ctx ) + 2 * HD ) * precision_bytes;

            return workspace_bytes + attention_bytes;
        }

        /**
         * @brief KV cache bytes at this context -- the term that shrinks the chunk budget.
         *
         * Only the full-attention layers cache, which is the property that makes a 27B model
         * plausible on 12 GiB at all (section 3): a dense stack of this width would cost 4x.
         * The DeltaNet layers hold a recurrent state whose size does not depend on T_ctx, so
         * it is a constant and does not belong in a per-context budget.
         */
        dim_t prefillKvBytes( dim_t B, dim_t T_ctx ) const
        {
            const dim_t precision_bytes =
                static_cast<dim_t>( TensorDataTypeTraits<TPrecision>::size_in_bytes );

            return config_.getNumFullAttentionLayers() * 2 * B
                * config_.getNumKVHeads() * T_ctx * config_.getHeadDim() * precision_bytes;
        }

        // The build-time reading of prefillChunking: same number, plus the warning. A
        // prediction must not warn -- a scan asks this at a hundred context lengths the user
        // never chose -- so the warning belongs on the path that is about to allocate.
        dim_t resolvePrefillChunkSize( dim_t B, dim_t T_ctx ) const
        {
            const PrefillChunking chunking = prefillChunking( B, T_ctx );

            if ( !chunking.fits_activation_budget )
            {
                Logging::Logger::warning( std::format(
                    "QwenTransformer: the prefill chunk floor ({}) exceeds the activation budget "
                    "(row cost {} bytes) -- this device/model combination cannot prefill efficiently",
                    kQwenPrefillChunkFloor,
                    computeChunkRowCostBytes( B, T_ctx ) ) );
            }

            return chunking.chunk_rows;
        }

        void allocateBlockWorkspace( dim_t B )
        {
            block_workspace_ = makeQwenAttentionBlockWorkspace<TDeviceType, TPrecision>(
                config_, this->getExecutionContext()->getDeviceId(), B, prefill_chunk_size_,
                this->getName() + ".block_ws." );
        }

        void allocateAndWireGqaWorkspace( dim_t B, dim_t T_ctx )
        {
            // score_width MUST match the op's flash decision (set on each block above via
            // setUseFlashPrefill) or the cuBLASLt path would overflow a narrow buffer; both
            // derive from useFlashPrefillForContext(T_ctx).
            gqa_workspace_ = makeQwenGqaWorkspace<TDeviceType, TPrecision>(
                config_, this->getExecutionContext()->getDeviceId(), B, T_ctx,
                prefill_chunk_size_, prefillScoreWidth( T_ctx ),
                this->getName() + ".gqa_ws." );

            const GqaState gqa_state = gqa_workspace_.state();

            for ( auto* layer : layers_ )
                layer->setState( gqa_state );
        }

        // ====================================================================
        // Helpers
        // ====================================================================

        std::pair<std::string, std::string> parseParameterPath( const std::string& full_name ) const
        {
            auto last_dot = full_name.rfind( '.' );

            if ( last_dot == std::string::npos )
                throw std::runtime_error( std::format( "Invalid parameter path: {}", full_name ) );

            return { full_name.substr( 0, last_dot ), full_name.substr( last_dot + 1 ) };
        }

        void validateBuildContext( const BuildContext& context ) const
        {
            const auto& input_shape = context.inputShape();

            if ( input_shape.size() != 2 )
            {
                throw std::invalid_argument( std::format(
                    "QwenTransformer: input must be rank 2 [B, T], got rank {}",
                    input_shape.size() ) );
            }

            if ( input_shape[ 0 ] < 1 || input_shape[ 1 ] < 1 )
            {
                throw std::invalid_argument( std::format(
                    "QwenTransformer: B and T must be >= 1, got [{}, {}]",
                    input_shape[ 0 ], input_shape[ 1 ] ) );
            }
        }
    };
}
