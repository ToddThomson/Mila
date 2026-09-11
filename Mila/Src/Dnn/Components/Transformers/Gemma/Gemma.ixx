/**
 * @file Gemma.ixx
 * @brief Gemma 4 decoder-only transformer network (inference: prefill + decode).
 *
 * Device-templated Gemma 4 autoregressive decoder. Modeled on LlamaTransformer,
 * with the two structural deltas that make Gemma heterogeneous:
 *
 *  - The layer list is NOT homogeneous. Gemma interleaves sliding (local) and
 *    full-attention (global) blocks 5:1 over 48 layers (final layer global), and
 *    the two are distinct GemmaBlock instantiations (kGlobal false/true) that
 *    differ in head_dim / KV-head count / K=V / window / RoPE. The transformer
 *    drives them polymorphically through ITransformerBlock (one virtual call per layer
 *    per token step, negligible against the per-layer GEMMs). See Gemma.md section 8.
 *
 *  - One shared GQA transient workspace serves both geometries. CudaGqaOp::setState
 *    takes only the raw scratch pointer and indexes it with its own HS_, so sizing
 *    q_permute / v_out at the MAX head_dim (global 512) lets the local layers
 *    (head_dim 256) use a prefix of the same buffer; preatt / att are head_dim-
 *    independent ([B, NH, chunk, T], NH shared at 16).
 *
 * Inference-only (Gemma is an inference target): forward()/backward() are not
 * implemented; the generation loop drives prefill()/decode().
 *
 * Two Gemma deltas are handled by deliberate design decision:
 *  - Embedding scale (x sqrt(hidden_size)) is applied at runtime in
 *    TokenEmbedding::forward via TokenEmbeddingConfig::embedding_scale (set in
 *    createGraph). The table is stored raw so it can be shared with the tied
 *    lm_head; see WeightTying.md D5. (Superseded the earlier converter-fold
 *    decision, BACKLOG Step 5d 2026-06-20, when weight tying landed.)
 *  - Final logit softcap (30 * tanh(logits / 30)) is applied host-side at the
 *    sampler: it is strictly monotonic, so it does not change greedy argmax, and
 *    GemmaConfig::getFinalLogitSoftcapping() carries the scalar for samplers that
 *    need it.
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
#include <cmath>

export module Dnn.Components.GemmaTransformer;

import Dnn.Components.GemmaConfig;
import Dnn.Components.GemmaBlock;
import Dnn.Components.ITransformerBlock;

import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Logging.Logger;
import Dnn.LanguageModelNetwork;
import Dnn.Component;
import Dnn.ComponentType;
import Dnn.ModelType;
import Dnn.Components.TokenEmbedding;
import Dnn.Components.Linear;
import Dnn.Components.RmsNorm;
import Dnn.Quantization.Weight.Policies;
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

    // Manual prefill-chunk override for VRAM sweeps: > 0 forces the chunk (clamped to
    // the context length); 0 = the activation-aware heuristic v2 on the transformer
    // (resolvePrefillChunkSize). Kept as the debug escape hatch; the chunk-32
    // operating point it used to pin is obsolete now that the block activations are
    // pooled (Gemma4InferenceReview.md sections 6-7).
    inline constexpr int64_t kGemmaPrefillChunkOverride = 0;

    // Context length (>=) at which the BF16 attention layers switch from the cuBLASLt
    // prefill path to the fused FlashAttention kernels -- the unbounded kernel on the
    // global layers, the bounded-ring variant on the local sliding layers. Below it
    // cuBLASLt is faster and its O(chunk x T_ctx) score workspace still fits; at/above it
    // that workspace is the memory wall, so flash both removes the O(S^2) score
    // materialization and lets the shared preatt/att buffer shrink to the window-bounded
    // sliding width -- the change that makes long context (target 64K) fit on a 12-16 GB
    // card (GqaFlashAttention.md 5.6). The build-time context length alone decides this,
    // so the workspace sizing and the op toggle stay coupled. 0 disables flash entirely
    // (always cuBLASLt).
    inline constexpr int64_t kGemmaFlashPrefillMinContext = 16384;

    // Activation budget for one prefill pass: every chunk-scaled term (the shared
    // block workspace, the GQA attention scratch, and the bounded-ring KV growth)
    // must fit under this cap. A fixed conservative cap rather than a live
    // cudaMemGetInfo budget (follow-up tracked in BACKLOG) -- the complete row-cost
    // model is the correction that matters: heuristic v1 budgeted only the attention
    // scratch and was blind to the terms that actually bound the chunk.
    inline constexpr int64_t kGemmaPrefillActivationBudgetBytes = int64_t{ 1536 } * 1024 * 1024;

    // The chunk rungs, largest first. Named once because two callers walk them: the resolution
    // that sizes the workspaces at build time, and the query a caller uses to choose a context
    // length before anything is built.
    inline constexpr int64_t kGemmaPrefillChunkRungs[] = { 1024, 512, 256, 128, 64 };

    // Below this the GEMM M dimension is tensor-core-hostile on top of the weight re-read, so
    // it is the floor rather than another rung: if it does not fit, warn instead of limping.
    inline constexpr int64_t kGemmaPrefillChunkFloor = 64;

    /**
     * @brief Gemma 4 transformer (decoder-only) for autoregressive inference.
     *
     * Graph: TokenEmbedding -> GemmaBlock x N (heterogeneous local/global) ->
     * RmsNorm -> Linear (lm_head). The embedding sqrt(d) scale and the final logit
     * softcap are handled by the converter and the sampler respectively (see the
     * file header).
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision,
        WeightQuantPolicy TWeightQuantization = NoWeightQuant, KvCachePolicy TKvCachePolicy = NoKvCompression>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class GemmaTransformer : public LanguageModelNetwork<TDeviceType, TPrecision>
    {
    public:
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using NetworkBase = LanguageModelNetwork<TDeviceType, TPrecision>;
        using TensorType = Tensor<TPrecision, MR>;

        // D4 Design B: weight-quantized bodies (FP4/FP8) convert the tied
        // embedding/lm_head table to per-vocab-row FP8 -- one shared FP8 table plus
        // one FP32 scale tensor read by both consumers. The NoWeightQuant body keeps
        // the BF16 table and head, preserving the exact HF token-parity oracle in
        // the reference configuration.
        using TableQuantizationPolicy = std::conditional_t<TWeightQuantization::kIsQuantized, PerChannelFp8<>, NoWeightQuant>;

        using TokenEmbeddingType = TokenEmbedding<TDeviceType, dtype_t::INT32, TPrecision, TableQuantizationPolicy>;
        using LmHeadLinearType = Linear<TDeviceType, TPrecision, TableQuantizationPolicy>;
        using RmsNormType = RmsNorm<TDeviceType, TPrecision>;
        // TKvCachePolicy applies to the LOCAL (sliding) layers only -- they attend a
        // bounded window, so their KV cache can be a ring (SlidingWindowKvCache.md D4).
        // GLOBAL (full-attention) layers attend the entire context and therefore always
        // use the full-context cache (NoKvCompression), regardless of the sliding policy.
        using LocalBlockType = GemmaBlock<TDeviceType, TPrecision, /*kGlobal*/ false, TWeightQuantization, TKvCachePolicy>;
        using GlobalBlockType = GemmaBlock<TDeviceType, TPrecision, /*kGlobal*/ true, TWeightQuantization, NoKvCompression>;
        using TransformerBlockType = ITransformerBlock<TDeviceType, TPrecision>;
        using TokenIndexType = Tensor<dtype_t::INT32, MR>;
        using ComponentPtr = typename NetworkBase::ComponentPtr;

        explicit GemmaTransformer( const std::string& name, const GemmaConfig& config, DeviceId device_id )
            : NetworkBase( name ), config_( config ), exec_context_( createExecutionContext( device_id ) )
        {
            config_.validate();

            if ( device_id.type != TDeviceType )
            {
                throw std::invalid_argument(
                    std::format( "GemmaTransformer: device type mismatch: expected {}, got {}",
                        deviceTypeToString( TDeviceType ),
                        deviceTypeToString( device_id.type ) ) );
            }

            createGraph();

            this->setExecutionContext( exec_context_.get() );
        }

        ~GemmaTransformer() override = default;

        // ====================================================================
        // Compute interface (inference-only)
        // ====================================================================

        TensorType& forward( const TokenIndexType& /*input*/ ) override
        {
            throw std::runtime_error(
                "GemmaTransformer is inference-only; use prefill()/decode() for autoregressive generation." );
        }

        TokenIndexType& backward( const TokenIndexType& /*input*/, const TensorType& /*output_grad*/ ) override
        {
            throw std::runtime_error( "GemmaTransformer is inference-only; backward() is not implemented." );
        }

        TensorType& prefill( const TokenIndexType& input ) override
        {
            return prefillFrom( input, 0 );
        }

        /**
         * @brief Chunked prefill starting at an absolute position (prompt-prefix reuse).
         *
         * `input` is the FULL prompt tensor, so the token index and the absolute
         * position coincide; chunking simply starts at start_offset instead of 0.
         * Positions [0, start_offset) must already be resident in the KV caches
         * (rewindKvCache). start_offset must lie inside the prompt so at least one
         * position is prefilled and the returned last-position logits are fresh.
         */
        TensorType& prefillFrom( const TokenIndexType& input, dim_t start_offset ) override
        {
            if ( !this->isBuilt() )
                throw std::runtime_error( "GemmaTransformer must be built before calling prefill()." );

            const int64_t B = input.shape()[ 0 ];
            const int64_t T_prompt = input.shape()[ 1 ];

            if ( start_offset < 0 || start_offset >= T_prompt )
                throw std::invalid_argument( std::format(
                    "GemmaTransformer::prefillFrom: start_offset {} must lie inside the prompt (length {})",
                    start_offset, T_prompt ) );

            int64_t offset = start_offset;
            int64_t T_last = 0;

            TensorType* last_block_out = nullptr;

            // Chunked prefill: slice the prompt into prefill_chunk_size_ chunks and
            // feed each through the heterogeneous layer list to populate the KV cache.
            while ( offset < T_prompt )
            {
                const int64_t T_actual = std::min( prefill_chunk_size_, T_prompt - offset );
                T_last = T_actual;

                auto chunk_input = input.view( shape_t{ B, T_actual }, offset );

                TensorType* block_input = &token_embedding_->forward( chunk_input );

                for ( auto* block : blocks_ )
                {
                    auto& block_out = block->prefill( *block_input, offset );
                    block_input = &block_out;
                }

                last_block_out = block_input;
                offset += T_actual;
            }

            // Extract the last position from the final chunk output -> [B, 1, model_dim].
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
                throw std::runtime_error( "GemmaTransformer must be built before calling decode()." );

            TensorType* block_input = &token_embedding_->forward( input );

            for ( auto* block : blocks_ )
            {
                auto& block_out = block->decode( *block_input, position );
                block_input = &block_out;
            }

            normalized_ptr_ = &final_rmsnorm_->forward( *block_input );
            logits_ptr_ = &lm_head_->forward( *normalized_ptr_ );

            return *logits_ptr_;
        }

        // ====================================================================
        // KV-cache orchestration
        // ====================================================================

        void resetKvCache()
        {
            for ( auto* block : blocks_ )
                block->resetKvCache();
        }

        /**
         * @brief Rewind every layer's KV cache to `position` for prefix reuse.
         *
         * All-or-nothing from the caller's perspective: returns true only when
         * every layer accepted. On false the caller falls back to a full prefill,
         * which positionally overwrites all caches -- so a partial rewind (some
         * layers moved, a bounded ring refused) needs no cleanup.
         */
        bool rewindKvCache( dim_t position ) override
        {
            bool all_accepted = true;

            for ( auto* block : blocks_ )
                all_accepted = block->rewindKvCache( position ) && all_accepted;

            return all_accepted;
        }

        // ====================================================================
        // Accessors / Diagnostics
        // ====================================================================

        // Structural kind comes from the Network base (ComponentType::Network);
        // the architecture family is reported here.
        ModelType getModelType() const
        {
            return ModelType::Gemma;
        }

        MemoryStats getMemoryStats() const override
        {
            MemoryStats stats;

            for ( const auto& child : this->getComponents() )
                stats += child->getMemoryStats();

            for ( auto* t : { gqa_q_permute_.get(), gqa_preatt_.get(), gqa_att_.get(),
                              gqa_v_out_.get(), gqa_preatt_decode_.get(),
                              gqa_att_decode_.get(), gqa_v_out_decode_.get(),
                              block_workspace_.q.get(), block_workspace_.k.get(),
                              block_workspace_.v.get(), block_workspace_.normed.get(),
                              block_workspace_.qkv.get(), block_workspace_.q_normed.get(),
                              block_workspace_.k_normed.get(), block_workspace_.v_normed.get(),
                              block_workspace_.attn.get(), block_workspace_.o.get(),
                              block_workspace_.o_normed.get(), block_workspace_.res1.get(),
                              block_workspace_.ffn_in.get(), block_workspace_.gate_up.get(),
                              block_workspace_.ffn_act.get(), block_workspace_.ffn_down.get(),
                              block_workspace_.ffn_normed.get(), block_workspace_.stream.get() } )
            {
                if ( t )
                    stats.device_state_bytes += t->getStorageSize();
            }

            // When tied, lm_head and token_embedding report the same shared allocation;
            // subtract the lm_head contribution once so it is not double-counted (D7).
            if ( tie_word_embeddings_ && lm_head_ )
                stats.device_parameter_bytes -= lm_head_->getMemoryStats().device_parameter_bytes;

            return stats;
        }

        /**
         * @brief What build( context ) would allocate for the whole model, without allocating.
         *
         * Mirrors onBuilding(): resolve the prefill chunk first, then recurse with the same
         * per-child contexts, then add the transformer's own pooled buffers and apply the two
         * no-double-count corrections. See Specifications/MemoryFootprint.md.
         */
        MemoryStats getRequiredMemory( const BuildContext& context ) const override
        {
            const auto& input_shape = context.inputShape();
            const dim_t B = input_shape[ 0 ];
            const dim_t T = input_shape[ 1 ];

            // The chunk must be resolved before recursing: it sizes every block's activation
            // scratch and the pooled workspace below.
            const int64_t prefill_chunk = resolvePrefillChunkSize( B, T );

            // The pooled workspace is installed on every block in inference mode
            // (allocateBlockWorkspace + installSharedWorkspace in onBuilding), and this
            // transformer accounts for it once below. Declaring it here is what stops each
            // block and each of its children from also counting their own slot.
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

            dim_t local_layers = 0;
            dim_t global_layers = 0;

            for ( int64_t i = 0; i < config_.getNumLayers(); ++i )
            {
                const std::string block_name = n + ".tf_layer_" + std::to_string( i );

                if ( config_.isGlobalLayer( static_cast<dim_t>( i ) ) )
                {
                    stats += this->template getComponentAs<GlobalBlockType>( block_name )
                        ->getRequiredMemory( block_context );
                    ++global_layers;
                }
                else
                {
                    stats += this->template getComponentAs<LocalBlockType>( block_name )
                        ->getRequiredMemory( block_context );
                    ++local_layers;
                }
            }

            stats += this->template getComponentAs<RmsNormType>( n + ".rmsn_final" )
                ->getRequiredMemory( final_context );

            MemoryStats head_stats =
                this->template getComponentAs<LmHeadLinearType>( n + ".lm_head" )
                    ->getRequiredMemory( final_context );

            stats += head_stats;

            if ( context.isInferenceMode() )
            {
                stats.device_state_bytes += blockWorkspaceBytes( B, prefill_chunk );
                stats.device_state_bytes += gqaWorkspaceBytes( B, T, prefill_chunk );
            }

            // Correction 1 -- weight tying. The head reports the shared table (Linear reports
            // an installed weight rather than hiding it), so subtract it once. Matches
            // getMemoryStats above.
            if ( config_.getTieWordEmbeddings() && context.isInferenceMode() )
            {
                stats.device_parameter_bytes -= head_stats.device_parameter_bytes;
            }

            // Correction 2 -- RoPE cos/sin caches are process-wide, deduplicated by
            // RopeCacheRegistry on (theta, max_seq_len, head_dim). Every block above reported
            // one cache, but only one per distinct key is ever allocated: Gemma has two, the
            // local and global theta. Without this the sum invents (layers - 2) phantom caches.
            stats.device_state_bytes -=
                std::max<dim_t>( local_layers - 1, 0 ) * ropeCacheBytes( config_.getHeadDim() );
            stats.device_state_bytes -=
                std::max<dim_t>( global_layers - 1, 0 ) * ropeCacheBytes( config_.getGlobalHeadDim() );

            return stats;
        }

        /**
         * @brief The prefill chunk this context length would use, and the one it could have used.
         *
         * Pure arithmetic over the config -- no device is touched and nothing is allocated -- so
         * a caller may ask before the network is built, which is the point: choosing a context
         * length by memory alone can land on one where the chunk has walked down to its floor.
         *
         * Heuristic v2 (Gemma4InferenceReview.md section 6.4): the largest rung whose complete
         * row cost fits the activation budget. The 1024 rung is enabled by the flash-prefill
         * score-buffer reclaim (5.6): with the O(chunk x T_ctx) preatt/att gone on the global
         * layers, the row cost drops far enough that a bigger chunk fits at long context. Its
         * payoff is NOT linear-GEMM weight amortization (saturated by 512) but (a) halving the
         * per-chunk FP4 weight DEQUANT passes (~14% of prefill at 16K, scales with chunk count)
         * and (b) fattening each flash launch so its K/V loads amortize over more query rows.
         *
         * KV-AWARE BUDGET: the activation workspace shares VRAM with the KV cache, whose global
         * term grows with T_ctx. Subtracting it from the fixed activation budget makes the
         * effective budget shrink at long context, so the big-chunk rung self-limits (chunk 1024
         * through ~40K, back to 512 at 64K where the bigger chunk would collide with the weight-
         * load transient) -- no hard context cap needed. 2048 is the next rung.
         */
        PrefillChunking prefillChunking( int64_t B, int64_t T_ctx ) const
        {
            PrefillChunking chunking;

            if constexpr ( kGemmaPrefillChunkOverride > 0 )
            {
                chunking.chunk_rows = std::min<int64_t>( kGemmaPrefillChunkOverride, T_ctx );
                chunking.unconstrained_chunk_rows = chunking.chunk_rows;

                return chunking;
            }
            else
            {
                // Short-context build: the whole context is a single chunk, and no rung applies.
                if ( T_ctx < kGemmaPrefillChunkFloor )
                {
                    chunking.chunk_rows = T_ctx;
                    chunking.unconstrained_chunk_rows = T_ctx;

                    return chunking;
                }

                const int64_t row_cost = computeChunkRowCostBytes( B, T_ctx );
                const int64_t kv_global = prefillGlobalKvBytes( B, T_ctx );
                const int64_t budget = ( kGemmaPrefillActivationBudgetBytes > kv_global )
                    ? ( kGemmaPrefillActivationBudgetBytes - kv_global )
                    : int64_t{ 0 };

                for ( int64_t candidate : kGemmaPrefillChunkRungs )
                {
                    if ( candidate > T_ctx )
                        continue;

                    // The first rung the context admits at all, budget aside. That it is reached
                    // before any budget test is what makes it the unconstrained answer.
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
                chunking.chunk_rows = kGemmaPrefillChunkFloor;
                chunking.fits_activation_budget = false;

                return chunking;
            }
        }

        // The base sums children; when tied, lm_head shares the embedding table, so its
        // elements would be counted twice. Subtract them once to match getMemoryStats (D7).
        dim_t parameterCount() const override
        {
            dim_t count = NetworkBase::parameterCount();

            if ( tie_word_embeddings_ && lm_head_ )
                count -= lm_head_->parameterCount();

            return count;
        }

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << std::endl;
            oss << "Gemma Network: " << this->getName() << std::endl;
            oss << "Device: " << this->getDeviceId().toString() << std::endl;
            oss << config_.toString();

            if ( this->isBuilt() )
            {
                oss << "  Parameters: " << this->parameterCount() << std::endl;
                oss << "  Prefill chunk: " << prefill_chunk_size_ << std::endl;
            }

            return oss.str();
        }

        void loadParameters( PretrainedModelReader& reader )
        {
            tie_word_embeddings_ = reader.getPretrainedMetadata().tie_word_embeddings;

            const int device_index = this->getExecutionContext()->getDeviceId().index;

            auto consume = [&]( const std::string& full_name, const Serialization::ITensorBlob& blob )
            {
                auto [component_path, param_name] = parseParameterPath( full_name );

                ComponentPtr target = this->findComponent( component_path );
                target->loadParameter( param_name, blob );

                // The reader reuses its pinned staging slot as soon as this returns,
                // and the quantize-on-load H2D is async on the op stream and does not
                // self-synchronize, so force completion here.
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

            // Tie lm_head to the (raw) embedding table after all blobs stream. When tied,
            // lm_head.weight is absent from the file, so nothing was loaded into lm_head's
            // own allocation; we replace it with the shared table here (WeightTying.md D2).
            // On quantized bodies the table is FP8 and the head also adopts the shared
            // per-vocab-row scales (D4 Design B).
            if ( tie_word_embeddings_ )
            {
                if constexpr ( TableQuantizationPolicy::kIsQuantized )
                {
                    lm_head_->installSharedWeight(
                        token_embedding_->getWeightTensorShared(),
                        token_embedding_->getWeightScalesTensorShared() );
                }
                else
                {
                    lm_head_->installSharedWeight( token_embedding_->getWeightTensorShared() );
                }
            }
        }

    protected:

        void onBuilding( const BuildContext& context ) override
        {
            validateBuildContext( context );

            const auto& input_shape = context.inputShape();
            const int64_t B = input_shape[ 0 ];
            const int64_t T = input_shape[ 1 ];

            // Tune the prefill chunk once and thread it to every block (and its GQA op)
            // via block_context. The heuristic budgets the complete chunk-scaled
            // activation cost (workspace + attention scratch + ring growth).
            prefill_chunk_size_ = resolvePrefillChunkSize( B, T );

            // Blocks need full context length so GQA can size the KV cache; the block
            // handles the prefill/decode split internally.
            shape_t block_shape = { B, T, config_.getModelDim() };
            BuildContext block_context =
                BuildContext( block_shape, context.getRuntimeMode(), context.shouldInitializeParameters() )
                .withPrefillSize( prefill_chunk_size_ );

            // Inference: final_rmsnorm and lm_head only process the last position.
            shape_t final_shape = context.isInferenceMode() ?
                shape_t{ B, 1, config_.getModelDim() } : shape_t{ B, T, config_.getModelDim() };

            BuildContext final_context( final_shape, context.getRuntimeMode(), context.shouldInitializeParameters() );

            token_embedding_ = this->template getComponentAs<TokenEmbeddingType>( this->getName() + ".temb" );
            token_embedding_->build( context );

            // Shared per-block activation workspace (pooling): one slot set serves all
            // layers because the inference path runs exactly one block at a time.
            // Allocated before the layer loop so each block (and its components) can
            // skip self-allocation at build.
            if ( context.isInferenceMode() )
                allocateBlockWorkspace( B );

            blocks_.clear();
            blocks_.reserve( static_cast<size_t>(config_.getNumLayers()) );

            for ( int64_t i = 0; i < config_.getNumLayers(); ++i )
            {
                const std::string block_name = this->getName() + ".tf_layer_" + std::to_string( i );

                // Heterogeneous layer list: the per-layer kind selects the GemmaBlock
                // instantiation (the two differ in head_dim / KV-heads / K=V / window / RoPE).
                if ( config_.isGlobalLayer( static_cast<dim_t>( i ) ) )
                {
                    auto block = this->template getComponentAs<GlobalBlockType>( block_name );

                    if ( context.isInferenceMode() )
                        block->installSharedWorkspace( block_workspace_ );

                    block->build( block_context );

                    // Global (unbounded) layers own the O(chunk x T_ctx) score buffer. Route
                    // them through fused flash prefill at long context so that buffer can be
                    // reclaimed -- MUST agree with prefillScoreWidth() in the workspace sizing.
                    // Fused decode is unconditional (band-limited, no workspace coupling; the
                    // BF16 op honors it, FP32 ignores it).
                    if ( context.isInferenceMode() )
                    {
                        block->setUseFlashPrefill( useFlashPrefillForContext( T ) );
                        block->setUseFlashDecode( true );
                    }

                    blocks_.push_back( static_cast<TransformerBlockType*>( block.get() ) );
                }
                else
                {
                    auto block = this->template getComponentAs<LocalBlockType>( block_name );

                    if ( context.isInferenceMode() )
                        block->installSharedWorkspace( block_workspace_ );

                    block->build( block_context );

                    // Local (sliding) layers flash at the same threshold via the bounded-
                    // ring kernel variant, replacing the ring softmax + separate QK/AV
                    // GEMMs. They stop reading the shared preatt/att buffer when flashed;
                    // prefillScoreWidth() deliberately still sizes it for them so the
                    // cuBLASLt fallback stays valid (further reclaim tracked in BACKLOG).
                    // Fused decode is unconditional, same as the global blocks above.
                    if ( context.isInferenceMode() )
                    {
                        block->setUseFlashPrefill( useFlashPrefillForContext( T ) );
                        block->setUseFlashDecode( true );
                    }

                    blocks_.push_back( static_cast<TransformerBlockType*>( block.get() ) );
                }
            }

            final_rmsnorm_ = this->template getComponentAs<RmsNormType>( this->getName() + ".rmsn_final" );
            final_rmsnorm_->build( final_context );

            lm_head_ = this->template getComponentAs<LmHeadLinearType>( this->getName() + ".lm_head" );

            // Tied lm_head: install the shared embedding table BEFORE build so the head
            // never allocates its own [vocab_size, model_dim] weight (~1 GB FP8). Without
            // this, build allocates that weight and loadParameters immediately frees it when
            // it installs the shared table -- a wasted ~1 GB load-time VRAM transient that
            // raises the load high-water and lowers the loadable-context ceiling. The tie
            // flag comes from checkpoint metadata via config; token_embedding_ is built above
            // so its table/scales allocations already exist. Inference only (the shared table
            // is loaded/quantized by loadParameters). See WeightTying.md / GqaAttentionExtent
            // sibling BACKLOG item.
            // Adopt the config's tie decision now, not at loadParameters. onBuilding is where
            // tying actually happens, so leaving the member false until load left getMemoryStats
            // subtracting nothing while the head already held the shared table -- a ~2.0 GB
            // double-count on 12B for the whole window between build() and load.
            tie_word_embeddings_ = config_.getTieWordEmbeddings();

            if ( context.isInferenceMode() && config_.getTieWordEmbeddings() )
            {
                if constexpr ( TableQuantizationPolicy::kIsQuantized )
                    lm_head_->installSharedWeight(
                        token_embedding_->getWeightTensorShared(),
                        token_embedding_->getWeightScalesTensorShared() );
                else
                    lm_head_->installSharedWeight( token_embedding_->getWeightTensorShared() );
            }

            lm_head_->build( final_context );

            if ( context.isInferenceMode() )
                allocateAndWireGqaWorkspace( B, input_shape[ 1 ] );

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
            meta.set( "type", "GemmaTransformer" )
                .set( "version", int64_t( 1 ) )
                .set( "name", this->getName() );

            archive.writeMetadata( "transformer_meta.json", meta );
        }

    private:

        GemmaConfig config_;

        // Tuned prefill chunk size -- single source of truth, set in onBuilding and
        // threaded to child components via BuildContext::withPrefillSize().
        int64_t prefill_chunk_size_{ 0 };

        std::shared_ptr<TokenEmbeddingType> token_embedding_{ nullptr };
        // Non-owning, polymorphic view of the heterogeneous block list; the concrete
        // blocks are owned by the component tree (addComponent). Valid after build.
        std::vector<TransformerBlockType*> blocks_;
        std::shared_ptr<RmsNormType> final_rmsnorm_{ nullptr };
        std::shared_ptr<LmHeadLinearType> lm_head_{ nullptr };

        // Set from checkpoint metadata in loadParameters. When true, lm_head shares the
        // token embedding table (WeightTying.md) and lm_head.weight is absent from the file.
        bool tie_word_embeddings_{ false };

        // Shared per-block activation workspace (split scratch + one output slot per
        // graph position) -- inference only, owned here, one set serves all blocks
        // (exactly one layer is live at a time on the sequential path); blocks and
        // their components view prefixes. Sized at the wider (local vs global)
        // geometry, the same max-geometry convention as the GQA workspace below.
        GemmaBlockWorkspace<TDeviceType, TPrecision> block_workspace_{};

        // Shared GQA transient workspace -- inference only, owned here, shared across
        // all blocks. q_permute/v_out are sized at the MAX head_dim (global) so the
        // local layers reuse a prefix; preatt/att are head_dim-independent.
        std::unique_ptr<TensorType> gqa_q_permute_{ nullptr };
        std::unique_ptr<TensorType> gqa_preatt_{ nullptr };
        std::unique_ptr<TensorType> gqa_att_{ nullptr };
        std::unique_ptr<TensorType> gqa_v_out_{ nullptr };
        std::unique_ptr<TensorType> gqa_preatt_decode_{ nullptr };
        std::unique_ptr<TensorType> gqa_att_decode_{ nullptr };
        std::unique_ptr<TensorType> gqa_v_out_decode_{ nullptr };

        // Activation pointers -- valid between prefill/decode and the next call.
        TensorType* normalized_ptr_{ nullptr };
        TensorType* logits_ptr_{ nullptr };

        // Declared last so it is destroyed first -- cudaStreamSynchronize() fires in
        // releaseResources() before any tensor cudaFree() from members above.
        std::unique_ptr<IExecutionContext> exec_context_{ nullptr };

        // ====================================================================
        // Graph construction
        // ====================================================================

        void createGraph()
        {
            // Gemma scales the embedding output by sqrt(hidden_size). With weight tying
            // the table is stored raw and shared with lm_head, so the scale is applied at
            // runtime here instead of being folded into the converted table (WeightTying.md D5).
            TokenEmbeddingConfig embedding_config;
            embedding_config.withVocabSize( config_.getVocabSize() )
                .withEmbeddingDim( static_cast<size_t>(config_.getModelDim()) )
                .withEmbeddingScale( static_cast<float>(
                    std::sqrt( static_cast<double>( config_.getModelDim() ) ) ) );

            this->addComponent(
                std::make_shared<TokenEmbeddingType>( this->getName() + ".temb", embedding_config ) );

            // Heterogeneous transformer blocks: the same network config drives every
            // block; the kGlobal template flag selects the per-layer geometry from it.
            for ( int64_t i = 0; i < config_.getNumLayers(); ++i )
            {
                const std::string block_name = this->getName() + ".tf_layer_" + std::to_string( i );

                if ( config_.isGlobalLayer( static_cast<dim_t>( i ) ) )
                {
                    this->addComponent(
                        std::make_shared<GlobalBlockType>( block_name, config_, std::nullopt ) );
                }
                else
                {
                    this->addComponent(
                        std::make_shared<LocalBlockType>( block_name, config_, std::nullopt ) );
                }
            }

            // Final RMSNorm. Norm convention under investigation (Gemma 4 != Gemma 3); using RAW
            // (withUnitOffset default 0) to match the block norms -- see Gemma.Block.ixx createGraph.
            auto rms_config = RmsNormConfig( shape_t{ config_.getModelDim() } )
                .withEpsilon( config_.getRMSNormEpsilon() )
                .withBias( false );

            this->addComponent(
                std::make_shared<RmsNormType>( this->getName() + ".rmsn_final", rms_config, std::nullopt ) );

            // Language model head -- model_dim -> vocab_size, no bias. Allocated with its
            // own weight here; when the checkpoint sets tie_word_embeddings, loadParameters
            // replaces that weight with the shared (raw) embedding table (WeightTying.md).
            auto lm_head_config = LinearConfig( config_.getModelDim(), config_.getVocabSize() )
                .withBias( false );

            this->addComponent(
                std::make_shared<LmHeadLinearType>( this->getName() + ".lm_head", lm_head_config, std::nullopt ) );
        }

        // ====================================================================
        // Prefill chunk heuristic v2 + shared block scratch + GQA workspace
        // ====================================================================

        // Max-geometry slot widths shared by the workspace allocation and the
        // chunk heuristic's row-cost model, so the two cannot drift apart.
        struct WorkspaceWidths
        {
            int64_t model_dim;
            int64_t hidden_dim;
            int64_t q_width;
            int64_t kv_width;
            int64_t qkv_width;

            // q + q_normed + attn; k + v + k_normed + v_normed; the eight
            // model_dim-wide stream-side slots; qkv; gate_up (2h) + ffn_act (h).
            int64_t totalRowElements() const
            {
                return 3 * q_width + 4 * kv_width + 8 * model_dim + qkv_width + 3 * hidden_dim;
            }
        };

        WorkspaceWidths computeWorkspaceWidths() const
        {
            const int64_t NH = config_.getNumHeads();

            WorkspaceWidths widths;
            widths.model_dim = config_.getModelDim();
            widths.hidden_dim = config_.getHiddenDimension();
            widths.q_width = NH * std::max( config_.getHeadDim(), config_.getGlobalHeadDim() );
            widths.kv_width = std::max(
                config_.getNumKVHeads() * config_.getHeadDim(),
                config_.getNumGlobalKVHeads() * config_.getGlobalHeadDim() );

            // Packed QKV width per layer kind: global K=V layers drop the V section.
            const int64_t packed_local =
                (NH + 2 * config_.getNumKVHeads()) * config_.getHeadDim();
            const int64_t packed_global =
                (NH + (config_.keyEqualsValue() ? 1 : 2) * config_.getNumGlobalKVHeads())
                * config_.getGlobalHeadDim();
            widths.qkv_width = std::max( packed_local, packed_global );

            return widths;
        }

        // Complete chunk-scaled activation cost per prefill chunk row
        // (Gemma4InferenceReview.md section 6.3, post-pooling constants): the shared
        // block workspace slots, the chunk-scaled GQA attention scratch
        // (preatt/att span the context; q_permute/v_out span the max head width),
        // and -- bounded sliding policy only -- the per-row ring capacity growth
        // across the local layers.
        // Whether the BF16 layers (global AND local sliding) run fused flash prefill at
        // this build-time context length (kGemmaFlashPrefillMinContext). A pure function of
        // T_ctx so the op toggle and the shared preatt/att workspace width stay coupled.
        bool useFlashPrefillForContext( int64_t T_ctx ) const noexcept
        {
            return TPrecision == TensorDataType::BF16
                && kGemmaFlashPrefillMinContext > 0
                && T_ctx >= kGemmaFlashPrefillMinContext;
        }

        // Width of the shared prefill preatt/att score buffer. With flash on, the global
        // layers no longer touch it, so it shrinks to the window-bounded sliding layers'
        // exact need (matching CudaGqaOp cache_capacity_ for kBounded); otherwise the
        // global cuBLASLt path needs the full context width. With the local layers now
        // also flashed, no prefill path reads it at all above the threshold -- the width
        // is deliberately KEPT at the sliding need so the cuBLASLt fallback (flash toggled
        // off on a standalone op) stays valid; shrinking further is a tracked follow-up.
        /**
         * @brief Bytes the pooled per-block activation workspace would take.
         *
         * Mirrors allocateBlockWorkspace(): eighteen slots, each [B, chunk, width], summed
         * through the same WorkspaceWidths the allocation uses.
         */
        std::size_t blockWorkspaceBytes( dim_t B, int64_t prefill_chunk ) const
        {
            return storageBytes<TPrecision>(
                computeWorkspaceWidths().totalRowElements() * B * prefill_chunk );
        }

        /**
         * @brief Bytes the shared GQA prefill/decode scratch would take.
         *
         * Mirrors allocateAndWireGqaWorkspace(). The score buffers are the term that made
         * flash prefill worth ~1 GB at 64K -- score_width collapses to the ring capacity
         * once the global layers flash, so this must use the same prefillScoreWidth().
         */
        std::size_t gqaWorkspaceBytes( dim_t B, int64_t T_ctx, int64_t prefill_chunk ) const
        {
            const dim_t NH = config_.getNumHeads();
            const dim_t HS_max = std::max( config_.getHeadDim(), config_.getGlobalHeadDim() );
            const dim_t score_width = prefillScoreWidth( T_ctx, prefill_chunk );

            const dim_t prefill_elements =
                ( 2 * B * NH * prefill_chunk * HS_max )      // q_permute, v_out
                + ( 2 * B * NH * prefill_chunk * score_width ); // preatt, att

            const dim_t decode_elements =
                ( 2 * B * NH * T_ctx )                        // preatt_decode, att_decode
                + ( B * NH * HS_max );                        // v_out_decode

            return storageBytes<TPrecision>( prefill_elements + decode_elements );
        }

        /**
         * @brief Bytes one RoPE cos/sin cache occupies for a given head width.
         *
         * MUST match CudaRopeOp::getRequiredStateMemorySize -- FP32 regardless of the
         * model precision, half the head dimension, two caches. Duplicated here because
         * the deduplication is the transformer's to apply and it needs the per-key size;
         * the model-level Gate A comparison is what holds the two together.
         */
        std::size_t ropeCacheBytes( dim_t head_dim ) const noexcept
        {
            const dim_t cache_elements = config_.getMaxSequenceLength() * ( head_dim / 2 );

            return static_cast<std::size_t>( cache_elements ) * sizeof( float ) * 2;
        }

        int64_t prefillScoreWidth( int64_t T_ctx ) const noexcept
        {
            return prefillScoreWidth( T_ctx, prefill_chunk_size_ );
        }

        /**
         * @brief Score width for an explicit chunk, for callers that run before build().
         *
         * getRequiredMemory() reports the workspace before prefill_chunk_size_ has been
         * assigned, so it must pass the chunk it resolved rather than read the member.
         */
        int64_t prefillScoreWidth( int64_t T_ctx, int64_t prefill_chunk ) const noexcept
        {
            if ( useFlashPrefillForContext( T_ctx ) )
                return std::min<int64_t>( T_ctx, config_.getWindow() + prefill_chunk - 1 );

            return T_ctx;
        }

        int64_t computeChunkRowCostBytes( int64_t B, int64_t T_ctx ) const
        {
            const int64_t precision_bytes =
                static_cast<int64_t>( TensorDataTypeTraits<TPrecision>::size_in_bytes );
            const int64_t NH = config_.getNumHeads();
            const int64_t HS_max = std::max( config_.getHeadDim(), config_.getGlobalHeadDim() );

            const int64_t workspace_bytes =
                computeWorkspaceWidths().totalRowElements() * B * precision_bytes;
            // With flash on the global layers, preatt/att shrink to the window-bounded
            // sliding width, so the chunk heuristic is no longer throttled by the O(T_ctx)
            // score span -- this is what decouples the prefill chunk size from long context.
            const int64_t attention_bytes =
                B * NH * ( 2 * prefillScoreWidth( T_ctx ) + 2 * HS_max ) * precision_bytes;

            int64_t ring_bytes = 0;

            if constexpr ( TKvCachePolicy::kIsActive )
            {
                int64_t local_layers = 0;

                for ( int64_t i = 0; i < config_.getNumLayers(); ++i )
                {
                    if ( !config_.isGlobalLayer( static_cast<dim_t>( i ) ) )
                        ++local_layers;
                }

                ring_bytes = local_layers * 2 * B
                    * config_.getNumKVHeads()
                    * config_.getHeadDim() * precision_bytes;
            }

            return workspace_bytes + attention_bytes + ring_bytes;
        }

        // Global-layer KV cache bytes -- the context-dependent VRAM term. The global
        // (unbounded) layers cache K and V over the full T_ctx; the sliding layers are
        // window-bounded and fixed, so only this term grows with context. The cache is BF16
        // (the config's FP8-KV label is inert -- OperationTraits has no PerChannelKvFp8
        // specialization; the op stores TPrecision).
        int64_t prefillGlobalKvBytes( int64_t B, int64_t T_ctx ) const
        {
            const int64_t precision_bytes =
                static_cast<int64_t>( TensorDataTypeTraits<TPrecision>::size_in_bytes );

            int64_t global_layers = 0;
            for ( int64_t i = 0; i < config_.getNumLayers(); ++i )
            {
                if ( config_.isGlobalLayer( static_cast<dim_t>( i ) ) )
                    ++global_layers;
            }

            return global_layers * 2 * B
                * config_.getNumGlobalKVHeads()
                * T_ctx * config_.getGlobalHeadDim() * precision_bytes;
        }

        // The build-time reading of prefillChunking: same number, plus the warning. A prediction
        // must not warn -- a scan asks this at a hundred context lengths the user never chose --
        // so the warning belongs on the path that is about to allocate, not on the query.
        int64_t resolvePrefillChunkSize( int64_t B, int64_t T_ctx ) const
        {
            const PrefillChunking chunking = prefillChunking( B, T_ctx );

            if ( !chunking.fits_activation_budget )
            {
                Logging::Logger::warning( std::format(
                    "GemmaTransformer: the prefill chunk floor ({}) exceeds the activation budget "
                    "(row cost {} bytes) -- this device/model combination cannot prefill efficiently",
                    kGemmaPrefillChunkFloor,
                    computeChunkRowCostBytes( B, T_ctx ) ) );
            }

            return chunking.chunk_rows;
        }

        void allocateBlockWorkspace( int64_t B )
        {
            const auto widths = computeWorkspaceWidths();
            const int64_t model_dim = widths.model_dim;
            const int64_t hidden_dim = widths.hidden_dim;
            const int64_t q_width = widths.q_width;
            const int64_t kv_width = widths.kv_width;
            const int64_t qkv_width = widths.qkv_width;

            auto device = this->getExecutionContext()->getDeviceId();
            const std::string n = this->getName();

            auto slot = [&]( int64_t width, const char* name )
            {
                return std::make_shared<TensorType>(
                    device, shape_t{ B, prefill_chunk_size_, width }, n + ".block_ws." + name );
            };

            block_workspace_.q = slot( q_width, "q" );
            block_workspace_.k = slot( kv_width, "k" );
            block_workspace_.v = slot( kv_width, "v" );
            block_workspace_.normed = slot( model_dim, "normed" );
            block_workspace_.qkv = slot( qkv_width, "qkv" );
            block_workspace_.q_normed = slot( q_width, "q_normed" );
            block_workspace_.k_normed = slot( kv_width, "k_normed" );
            block_workspace_.v_normed = slot( kv_width, "v_normed" );
            block_workspace_.attn = slot( q_width, "attn" );
            block_workspace_.o = slot( model_dim, "o" );
            block_workspace_.o_normed = slot( model_dim, "o_normed" );
            block_workspace_.res1 = slot( model_dim, "res1" );
            block_workspace_.ffn_in = slot( model_dim, "ffn_in" );
            block_workspace_.gate_up = slot( 2 * hidden_dim, "gate_up" );
            block_workspace_.ffn_act = slot( hidden_dim, "ffn_act" );
            block_workspace_.ffn_down = slot( model_dim, "ffn_down" );
            block_workspace_.ffn_normed = slot( model_dim, "ffn_normed" );
            block_workspace_.stream = slot( model_dim, "stream" );
        }

        void allocateAndWireGqaWorkspace( int64_t B, int64_t T_ctx )
        {
            const int64_t NH = config_.getNumHeads();
            const int64_t HS_max = std::max( config_.getHeadDim(), config_.getGlobalHeadDim() );
            auto device = this->getExecutionContext()->getDeviceId();
            const std::string n = this->getName();

            gqa_q_permute_ = std::make_unique<TensorType>(
                device, shape_t{ B, NH, prefill_chunk_size_, HS_max }, n + ".gqa_ws.q_perm" );
            // preatt/att carry the O(chunk x score_width) score matrix for the cuBLASLt
            // path. With flash on the global layers, only the window-bounded sliding layers
            // still use these, so score_width collapses from T_ctx to the ring capacity --
            // reclaiming ~1 GB at 64K (GqaFlashAttention.md 5.6). MUST match the op's flash
            // decision (set on the global blocks in the build loop via setUseFlashPrefill) or
            // the cuBLASLt global path would overflow a narrow buffer; both derive from
            // useFlashPrefillForContext(T_ctx).
            const int64_t score_width = prefillScoreWidth( T_ctx );

            gqa_preatt_ = std::make_unique<TensorType>(
                device, shape_t{ B, NH, prefill_chunk_size_, score_width }, n + ".gqa_ws.preatt" );
            gqa_att_ = std::make_unique<TensorType>(
                device, shape_t{ B, NH, prefill_chunk_size_, score_width }, n + ".gqa_ws.att" );
            gqa_v_out_ = std::make_unique<TensorType>(
                device, shape_t{ B, NH, prefill_chunk_size_, HS_max }, n + ".gqa_ws.v_out" );

            gqa_preatt_decode_ = std::make_unique<TensorType>(
                device, shape_t{ B, NH, 1, T_ctx }, n + ".gqa_ws.preatt_dec" );
            gqa_att_decode_ = std::make_unique<TensorType>(
                device, shape_t{ B, NH, 1, T_ctx }, n + ".gqa_ws.att_dec" );
            gqa_v_out_decode_ = std::make_unique<TensorType>(
                device, shape_t{ B, NH, 1, HS_max }, n + ".gqa_ws.v_out_dec" );

            GqaState gqa_state;
            gqa_state.q_permute = gqa_q_permute_.get();
            gqa_state.preatt = gqa_preatt_.get();
            gqa_state.att = gqa_att_.get();
            gqa_state.v_out = gqa_v_out_.get();
            gqa_state.preatt_decode = gqa_preatt_decode_.get();
            gqa_state.att_decode = gqa_att_decode_.get();
            gqa_state.v_out_decode = gqa_v_out_decode_.get();

            for ( auto* block : blocks_ )
                block->setState( gqa_state );
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
                    "GemmaTransformer: input must be rank 2 [B, T], got rank {}",
                    input_shape.size() ) );
            }

            if ( input_shape[ 0 ] < 1 || input_shape[ 1 ] < 1 )
            {
                throw std::invalid_argument( std::format(
                    "GemmaTransformer: B and T must be >= 1, got [{}, {}]",
                    input_shape[ 0 ], input_shape[ 1 ] ) );
            }
        }
    };
}
