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
 *    need it. sequenceLogLikelihood() applies it before its log-softmax, since a
 *    probability is not invariant to it the way an argmax is.
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
import Dnn.Components.GemmaFeedForward;
import Dnn.Components.GemmaBlock;
import Dnn.Components.ITransformerBlock;

import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorOps;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.LanguageModelNetwork;
import Dnn.Component;
import Dnn.ComponentType;
import Dnn.ModelType;
import Dnn.RuntimeMode;
import Dnn.Components.TokenEmbedding;
import Dnn.Components.Linear;
import Dnn.Components.RmsNorm;
import Dnn.Quantization.Weight.Policies;
import Dnn.Quantization.KvCache.Policy;
import Dnn.Quantization.KvCache.SlidingWindowKvFp8;
import Compute.Device;
import Compute.DeviceAllocation;
import Compute.DeviceType;
import Compute.DeviceId;
import Compute.DeviceTypeTraits;
import Compute.GqaState;
import Compute.GqaWorkspace;
import Compute.CpuMemoryResource;
#ifdef MILA_HAS_CUDA
import Compute.CudaPinnedMemoryResource;
#endif
import Compute.ExecutionContext;
import Compute.ExecutionContextFactory;
import Compute.OperationTraits;
import Compute.OperationType;
import Dnn.SequenceLogLikelihood;
import Serialization.ModelArchive;
import Serialization.Mode;
import Serialization.Metadata;
import Serialization.WeightsReader;
import Serialization.Tensor;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;
    using namespace Mila::Dnn::Quant::Weight;
    using namespace Mila::Dnn::Quant::KvCache;

    /**
     * @brief Gemma 4 transformer (decoder-only) for autoregressive inference.
     *
     * Graph: TokenEmbedding -> GemmaBlock x N (heterogeneous local/global) ->
     * RmsNorm -> Linear (lm_head). The embedding sqrt(d) scale and the final logit
     * softcap are handled by the converter and the sampler respectively (see the
     * file header). kFeedForward is every block's feed-forward sublayer (GemmaBlock). TKvCachePolicy is the local
     * (sliding) layers' cache, TGlobalKvCachePolicy the global layers': uncompressed, or PerTokenKvFp8.
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision,
        WeightQuantPolicy TWeightQuantization = NoWeightQuant, KvCachePolicy TKvCachePolicy = NoKvCompression,
        GemmaFeedForward kFeedForward = GemmaFeedForward::Dense, KvCachePolicy TGlobalKvCachePolicy = NoKvCompression>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class GemmaTransformer : public LanguageModelNetwork<TDeviceType, TPrecision>
    {
    public:
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using NetworkBase = LanguageModelNetwork<TDeviceType, TPrecision>;
        using TensorType = Tensor<TPrecision, MR>;

        // D4 Design B: weight-quantized bodies store the tied embedding/lm_head table
        // as six-bit codes with one FP16 scale per 32 elements of a row -- one code
        // tensor and one scale tensor read by both consumers. Against FP8 per row it is
        // closer to BF16 on every measure in 18% fewer bytes (Quantization.md, "The tied
        // table -- six bits per 32"). The NoWeightQuant body keeps the BF16 table and
        // head, preserving the exact HF token-parity oracle in the reference configuration.
        using TableQuantizationPolicy = std::conditional_t<TWeightQuantization::kIsQuantized, PerGroupInt6<32>, NoWeightQuant>;

        using TokenEmbeddingType = TokenEmbedding<TDeviceType, dtype_t::INT32, TPrecision, TableQuantizationPolicy>;
        using LmHeadLinearType = Linear<TDeviceType, TPrecision, TableQuantizationPolicy>;
        using RmsNormType = RmsNorm<TDeviceType, TPrecision>;
        // TKvCachePolicy applies to the LOCAL (sliding) layers only -- they attend a
        // bounded window, so their KV cache can be a ring (SlidingWindowKvCache.md D4).
        // GLOBAL (full-attention) layers attend the entire context, so their cache holds
        // every position, uncompressed or in FP8 (TGlobalKvCachePolicy), never a ring.
        using LocalBlockType = GemmaBlock<TDeviceType, TPrecision, /*kGlobal*/ false, TWeightQuantization, TKvCachePolicy, kFeedForward>;
        using GlobalBlockType = GemmaBlock<TDeviceType, TPrecision, /*kGlobal*/ true, TWeightQuantization, TGlobalKvCachePolicy, kFeedForward>;
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

            this->setCachedLength( T_prompt );

            return *logits_ptr_;
        }

    protected:

        TensorType& onDecode( const TokenIndexType& input, dim_t position ) override
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

    public:

        /**
         * @brief Teacher-forced log-likelihood of `input` -- the corpus-perplexity path.
         *
         * Runs the same chunked prefill generation runs, but evaluates the head at EVERY
         * position rather than the last, in windows of `getLogLikelihoodWindow()` rows, and
         * reduces each window where the logits are (NextTokenLogProbabilityOp) before the next
         * overwrites it; the model synchronizes once, at the end. The final-logit softcap is
         * applied to each row before the log-softmax: the sampler applies it before sampling,
         * and a probability, unlike an argmax, is not invariant to it.
         *
         * The prefill overwrites the KV caches from position 0, as prefill() does.
         *
         * @throws std::out_of_range when a token past the first is outside the vocabulary.
         */
        SequenceLogLikelihood sequenceLogLikelihood( const TokenIndexType& input ) override
        {
            if ( !this->isBuilt() )
                throw std::runtime_error( "GemmaTransformer must be built before calling sequenceLogLikelihood()." );

            const dim_t B = input.shape()[ 0 ];
            const dim_t T = input.shape()[ 1 ];

            if ( B != 1 )
                throw std::invalid_argument( std::format(
                    "GemmaTransformer::sequenceLogLikelihood: batch must be 1, got {} -- the targets are the "
                    "sequence's own next tokens, which two rows cannot share", B ) );

            if ( T < 2 )
                throw std::invalid_argument( std::format(
                    "GemmaTransformer::sequenceLogLikelihood: need at least 2 tokens to score one position, got {}", T ) );

            const dim_t model_dim = config_.getModelDim();
            const dim_t vocab_size = config_.getVocabSize();
            const dim_t window = resolveLogLikelihoodWindow( prefill_chunk_size_ );

            auto host_tokens = toHost<TensorDataType::INT32>( input, this->getExecutionContext() );

            requireTokensInVocabulary( host_tokens.data(), T, vocab_size );

            if ( !log_likelihood_op_ )
            {
                log_likelihood_op_ = std::make_unique<LogLikelihoodOpType>(
                    this->getExecutionContext(), config_.getFinalLogitSoftcapping() );
            }

            log_likelihood_op_->begin( T - 1 );

            dim_t offset = 0;

            while ( offset < T )
            {
                const dim_t chunk_length = std::min<dim_t>( prefill_chunk_size_, T - offset );

                auto chunk_input = input.view( shape_t{ B, chunk_length }, offset );

                TensorType* block_input = &token_embedding_->forward( chunk_input );

                for ( auto* block : blocks_ )
                {
                    block_input = &block->prefill( *block_input, offset );
                }

                for ( dim_t start = 0; start < chunk_length; start += window )
                {
                    const dim_t rows = std::min<dim_t>( window, chunk_length - start );

                    // The sequence's last token predicts nothing, so a window holding only it
                    // has no work. Every other window has at least one scored position.
                    if ( offset + start + 1 >= T )
                        break;

                    auto window_input = block_input->view(
                        shape_t{ B, rows, model_dim }, start * model_dim );

                    auto& normalized = final_rmsnorm_->forward( window_input );
                    auto& logits = lm_head_->forward( normalized );

                    log_likelihood_op_->forward( logits, input, offset + start,
                        std::min<dim_t>( rows, T - 1 - ( offset + start ) ) );
                }

                offset += chunk_length;
            }

            this->setCachedLength( T );
            this->synchronize();

            SequenceLogLikelihood result;

            for ( const float log_probability : log_likelihood_op_->logProbabilities( T - 1 ) )
            {
                result.total_log_probability += log_probability;
            }

            result.scored_positions = T - 1;

            return result;
        }

        // ====================================================================
        // KV-cache orchestration
        // ====================================================================

        void resetKvCache()
        {
            for ( auto* block : blocks_ )
                block->resetKvCache();

            this->setCachedLength( 0 );
            this->discardDecodeRecording();
        }

    protected:

        /**
         * @brief Whether every layer's KV cache accepts a rewind to `position` for prefix reuse.
         *
         * All-or-nothing from the caller's perspective: returns true only when
         * every layer accepted. On false the caller falls back to a full prefill,
         * which positionally overwrites all caches -- so a refusal needs no cleanup.
         */
        bool onRewindKvCache( dim_t position, dim_t cached_length ) override
        {
            bool all_accepted = true;

            for ( auto* block : blocks_ )
                all_accepted = block->rewindKvCache( position, cached_length ) && all_accepted;

            return all_accepted;
        }

    public:

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

            stats.device_state_bytes += block_workspace_.deviceStorageBytes();
            stats.device_state_bytes += gqa_workspace_.deviceStorageBytes();

            // When tied, lm_head and token_embedding report the same shared allocation;
            // subtract the lm_head contribution once so it is not double-counted (D7).
            if ( tie_word_embeddings_ && lm_head_ )
                stats.device_parameter_bytes -= lm_head_->getMemoryStats().device_parameter_bytes;

            return stats;
        }

        /**
         * @brief What build( context ) would allocate for the whole model, without allocating.
         *
         * At the prefill chunk the context carries. Reads nothing from the device this network is
         * bound to, so one graph prices any device (Deployment.md section 4).
         */
        MemoryStats getRequiredMemory( const BuildContext& context ) const override
        {
            return requiredMemoryAtChunk( context, context.getResolvedPrefillSize() );
        }

        /**
         * @brief The chunk rungs choosePrefillChunk() walks for this family, largest first.
         *
         * Below 64 rows the GEMM M dimension is tensor-core-hostile on top of the weight re-read,
         * so the smallest rung is the floor rather than a step toward a smaller one.
         */
        static constexpr dim_t kPrefillChunkRungs[] = { 1024, 512, 256, 128, 64 };

    private:

        /**
         * @brief What build( context ) would allocate with the given prefill chunk.
         *
         * Mirrors onBuilding(): recurse with the same per-child contexts, then add the
         * transformer's own pooled buffers and apply the two no-double-count corrections.
         */
        MemoryStats requiredMemoryAtChunk( const BuildContext& context, int64_t prefill_chunk ) const
        {
            const auto& input_shape = context.inputShape();
            const dim_t B = input_shape[ 0 ];
            const dim_t T = input_shape[ 1 ];

            // The pooled workspace is installed on every block in inference mode
            // (allocateBlockWorkspace + installSharedWorkspace in onBuilding), and this
            // transformer accounts for it once below. Declaring it here is what stops each
            // block and each of its children from also counting their own slot.
            BuildContext block_context =
                context.forChild( shape_t{ B, T, config_.getModelDim() } )
                .withPrefillSize( prefill_chunk )
                .withInstalledOutput( context.isInferenceMode() )
                .withFusedDecode( context.isInferenceMode() );

            const shape_t final_shape = context.isInferenceMode()
                ? shape_t{ B, resolveLogLikelihoodWindow( prefill_chunk ), config_.getModelDim() }
                : shape_t{ B, T, config_.getModelDim() };

            const BuildContext final_context = context.forChild( final_shape );

            const std::string n = this->getName();

            MemoryStats stats;

            stats += this->template getComponentAs<TokenEmbeddingType>( n + ".temb" )
                ->getRequiredMemory( embeddingContext( context, prefill_chunk ) );

            for ( int64_t i = 0; i < config_.getNumLayers(); ++i )
            {
                const std::string block_name = n + ".tf_layer_" + std::to_string( i );

                if ( config_.isGlobalLayer( static_cast<dim_t>( i ) ) )
                {
                    stats += this->template getComponentAs<GlobalBlockType>( block_name )
                        ->getRequiredMemory( block_context );
                }
                else
                {
                    stats += this->template getComponentAs<LocalBlockType>( block_name )
                        ->getRequiredMemory( block_context );
                }
            }

            stats += this->template getComponentAs<RmsNormType>( n + ".rmsn_final" )
                ->getRequiredMemory( final_context );

            MemoryStats head_stats =
                this->template getComponentAs<LmHeadLinearType>( n + ".lm_head" )
                    ->getRequiredMemory( final_context );

            stats += head_stats;

            const std::size_t granularity = context.getAllocationGranularity();

            if ( context.isInferenceMode() )
            {
                stats.device_state_bytes +=
                    GemmaBlockWorkspace<TDeviceType, TPrecision>::requiredBytes( config_, B, prefill_chunk, granularity );
                stats.device_state_bytes += gqaWorkspaceBytes( B, T, prefill_chunk, granularity );
            }

            // Weight tying. The head reports the shared table (Linear reports an installed weight
            // rather than hiding it), so subtract it once. Matches getMemoryStats above.
            if ( config_.getTieWordEmbeddings() && context.isInferenceMode() )
            {
                stats.device_parameter_bytes -= head_stats.device_parameter_bytes;
            }

            return stats;
        }

    public:

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

        void loadParameters( WeightsReader& reader )
        {
            tie_word_embeddings_ = reader.getWeightsMetadata().tie_word_embeddings;

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

            // Every tensor has landed, so the buffer full-precision weights were fitted through is
            // not held for the model's lifetime.
            this->getExecutionContext()->releaseLoadStaging();

            // Tie lm_head to the (raw) embedding table after all blobs stream. When tied,
            // lm_head.weight is absent from the file, so nothing was loaded into lm_head's
            // own allocation; we replace it with the shared table here (WeightTying.md D2).
            // On quantized bodies the table is INT6 and the head also adopts the shared
            // scales (D4 Design B).
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

            // The chunk the caller resolved, threaded to every block (and its GQA op) via
            // block_context. The build never chooses one (Deployment.md section 7).
            prefill_chunk_size_ = context.getResolvedPrefillSize();

            // Blocks need full context length so GQA can size the KV cache; the block
            // handles the prefill/decode split internally.
            shape_t block_shape = { B, T, config_.getModelDim() };
            BuildContext block_context =
                BuildContext( block_shape, context.getRuntimeMode(), context.shouldInitializeParameters() )
                .withPrefillSize( prefill_chunk_size_ )
                .withFusedDecode( context.isInferenceMode() );

            // Inference: final_rmsnorm and lm_head process the log-likelihood window, which is one
            // row for generation. MUST agree with requiredMemoryAtChunk().
            shape_t final_shape = context.isInferenceMode() ?
                shape_t{ B, resolveLogLikelihoodWindow( prefill_chunk_size_ ), config_.getModelDim() }
                : shape_t{ B, T, config_.getModelDim() };

            BuildContext final_context( final_shape, context.getRuntimeMode(), context.shouldInitializeParameters() );

            token_embedding_ = this->template getComponentAs<TokenEmbeddingType>( this->getName() + ".temb" );
            token_embedding_->build( embeddingContext( context, prefill_chunk_size_ ) );

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

                    // The global (unbounded) layers' flash decision MUST agree with prefillScoreWidth(): the
                    // cuBLASLt path would need an O(chunk x T_ctx) score buffer. Fused decode is declared on
                    // block_context above, so its scratch is part of the reservation.
                    if ( context.isInferenceMode() )
                    {
                        block->setUseFlashPrefill( usesFlashPrefillOnGlobalLayers() );
                    }

                    blocks_.push_back( static_cast<TransformerBlockType*>( block.get() ) );
                }
                else
                {
                    auto block = this->template getComponentAs<LocalBlockType>( block_name );

                    if ( context.isInferenceMode() )
                        block->installSharedWorkspace( block_workspace_ );

                    block->build( block_context );

                    // Local (sliding) layers flash through the bounded-ring kernel variant, and read no shared
                    // preatt/att buffer when they do. MUST agree with prefillScoreWidth(), which allocates none
                    // once every layer flashes.
                    if ( context.isInferenceMode() )
                    {
                        block->setUseFlashPrefill( usesFlashPrefillOnLocalLayers() );
                    }

                    blocks_.push_back( static_cast<TransformerBlockType*>( block.get() ) );
                }
            }

            final_rmsnorm_ = this->template getComponentAs<RmsNormType>( this->getName() + ".rmsn_final" );
            final_rmsnorm_->build( final_context );

            lm_head_ = this->template getComponentAs<LmHeadLinearType>( this->getName() + ".lm_head" );

            // Tied lm_head: install the shared embedding table BEFORE build so the head
            // never allocates its own [vocab_size, model_dim] weight (~0.7 GB INT6). Without
            // this, build allocates that weight and loadParameters immediately frees it when
            // it installs the shared table -- a wasted load-time VRAM transient that
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

            // Every operation shares the context's one scratch buffer. Reserving it at the largest
            // request, the figure the footprint reports, is what puts it in the footprint.
            this->getExecutionContext()->reserveScratch( this->getMemoryStats().device_scratch_bytes );

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

        // The prefill chunk the caller resolved, taken from the BuildContext in onBuilding and
        // threaded to child components via BuildContext::withPrefillSize().
        int64_t prefill_chunk_size_{ 0 };

        std::shared_ptr<TokenEmbeddingType> token_embedding_{ nullptr };
        // Non-owning, polymorphic view of the heterogeneous block list; the concrete
        // blocks are owned by the component tree (addComponent). Valid after build.
        std::vector<TransformerBlockType*> blocks_;
        std::shared_ptr<RmsNormType> final_rmsnorm_{ nullptr };
        std::shared_ptr<LmHeadLinearType> lm_head_{ nullptr };

        using LogLikelihoodOpType =
            typename OperationTraits<OperationType::NextTokenLogProbabilityOp, TDeviceType, TPrecision>::type;

        // Created at the first sequenceLogLikelihood(); holds no device memory.
        std::unique_ptr<LogLikelihoodOpType> log_likelihood_op_;

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
        GqaWorkspace<TDeviceType, TPrecision> gqa_workspace_{};

        // Activation pointers -- valid between prefill/decode and the next call.
        TensorType* normalized_ptr_{ nullptr };
        TensorType* logits_ptr_{ nullptr };

        // Declared last so it is destroyed first -- cudaStreamSynchronize() fires in
        // releaseResources() before any tensor cudaFree() from members above.
        std::unique_ptr<IExecutionContext> exec_context_{ nullptr };

        /**
         * @brief Positions the head evaluates per pass, bounded by what a pass can supply.
         *
         * The head reads the block stack's output, and a prefill pass produces at most
         * prefill_chunk rows of it. Both onBuilding() and requiredMemoryAtChunk() resolve
         * through here, so the head cannot be built at one width and priced at another.
         */
        dim_t resolveLogLikelihoodWindow( dim_t prefill_chunk ) const
        {
            return std::min<dim_t>( config_.getLogLikelihoodWindow(), prefill_chunk );
        }

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
        // Shared block scratch + GQA workspace
        // ====================================================================

        /**
         * @brief Whether the global layers prefill through the unbounded flash kernel: every BF16 build whose global
         *        head size it serves.
         *
         * No context threshold: the kernel is 1.4x to 3.7x faster than the cuBLASLt pipeline at the global layers'
         * geometry from a 512-token prompt up (GqaFlashAttention.md 5.7). FP32 has no flash kernel.
         */
        bool usesFlashPrefillOnGlobalLayers() const
        {
            return TPrecision == TensorDataType::BF16
                && GlobalBlockType::AttentionType::supportsFlashPrefill( config_.getGlobalHeadDim() );
        }

        /**
         * @brief Whether the sliding layers prefill through the bounded-ring flash kernel, 2.0x to 3.5x faster than
         *        their cuBLASLt pipeline: every BF16 build whose sliding head size it serves.
         */
        bool usesFlashPrefillOnLocalLayers() const
        {
            return TPrecision == TensorDataType::BF16
                && LocalBlockType::AttentionType::supportsFlashPrefill( config_.getHeadDim() );
        }

        /**
         * @brief Bytes the shared GQA prefill/decode scratch would take.
         *
         * Mirrors allocateAndWireGqaWorkspace(). The score buffers are the term that made
         * flash prefill worth ~1 GB at 64K -- score_width collapses to the ring capacity
         * once the global layers flash, and to nothing once every layer does, so this must
         * use the same prefillScoreWidth().
         */
        std::size_t gqaWorkspaceBytes( dim_t B, int64_t T_ctx, int64_t prefill_chunk, std::size_t granularity ) const
        {
            const dim_t HS_max = std::max( config_.getHeadDim(), config_.getGlobalHeadDim() );

            return gqaWorkspaceDeviceBytes<TPrecision>( granularity, B, config_.getNumHeads(), HS_max, T_ctx,
                prefill_chunk, prefillScoreWidth( T_ctx, prefill_chunk ) );
        }

        /**
         * @brief The context the token embedding is built and priced at: in inference, one prefill chunk wide,
         *        since every inference pass embeds at most a chunk at a time.
         */
        static BuildContext embeddingContext( const BuildContext& context, int64_t prefill_chunk )
        {
            if ( !context.isInferenceMode() )
                return context;

            const auto& input_shape = context.inputShape();

            return context.forChild( shape_t{ input_shape[ 0 ], std::min<int64_t>( input_shape[ 1 ], prefill_chunk ) } );
        }

        int64_t prefillScoreWidth( int64_t T_ctx ) const
        {
            return prefillScoreWidth( T_ctx, prefill_chunk_size_ );
        }

        /**
         * @brief Width of the shared prefill score buffer, for an explicit chunk so callers before build() can ask.
         *
         * A layer on the cuBLASLt path reads it across every cached column: the whole context for a global layer, or
         * for a sliding layer built without the ring cache, and the ring capacity (CudaGqaOp's cache_capacity_ for
         * kBounded) for a ring one. So it is the whole context when any layer of the first two kinds misses flash,
         * the ring capacity when only ring layers miss it, and zero -- no buffer -- when every layer flashes.
         */
        int64_t prefillScoreWidth( int64_t T_ctx, int64_t prefill_chunk ) const
        {
            constexpr bool kLocalLayersRing =
                std::is_same_v<TKvCachePolicy, SlidingWindowKvCache> || std::is_same_v<TKvCachePolicy, SlidingWindowKvFp8>;

            const bool needs_whole_context = !usesFlashPrefillOnGlobalLayers()
                || ( !kLocalLayersRing && !usesFlashPrefillOnLocalLayers() );

            if ( needs_whole_context )
                return T_ctx;

            if ( !usesFlashPrefillOnLocalLayers() )
                return std::min<int64_t>( T_ctx, config_.getWindow() + prefill_chunk - 1 );

            return 0;
        }

        void allocateBlockWorkspace( int64_t B )
        {
            block_workspace_ = makeGemmaBlockWorkspace<TDeviceType, TPrecision>(
                config_, this->getExecutionContext()->getDeviceId(), B, prefill_chunk_size_,
                this->getName() + ".block_ws." );
        }

        void allocateAndWireGqaWorkspace( int64_t B, int64_t T_ctx )
        {
            // preatt/att carry the O(chunk x score_width) score matrix for the cuBLASLt path. The width MUST
            // match the flash decision set on the blocks in the build loop, or the cuBLASLt global path would
            // overflow a narrow buffer; both derive from usesFlashPrefillOnGlobalLayers().
            gqa_workspace_ = makeGqaWorkspace<TDeviceType, TPrecision>(
                this->getExecutionContext()->getDeviceId(), B, config_.getNumHeads(),
                std::max( config_.getHeadDim(), config_.getGlobalHeadDim() ),
                T_ctx, prefill_chunk_size_, prefillScoreWidth( T_ctx ), this->getName() + ".gqa_ws." );

            const GqaState gqa_state = gqa_workspace_.state();

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
