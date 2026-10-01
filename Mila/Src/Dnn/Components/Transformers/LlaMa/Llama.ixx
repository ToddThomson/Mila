/**
 * @file Llama.ixx
 * @brief LLaMA-style decoder-only transformer network.
 *
 * Device-templated network implementing a LLaMA-style autoregressive decoder.
 */

module;
#include <string>
#include <vector>
#include <memory>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <cstdint>
#include <format>
#include <optional>
#include <algorithm>

export module Dnn.Components.LlamaTransformer;
export import :Config;
export import :Presets;
export import :BlockWorkspace;
export import :Block;

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
import Dnn.Components.Rope;
import Dnn.ActivationType;
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
import Serialization.WeightsReader;
import Serialization.Tensor;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;
    using namespace Mila::Dnn::Quant::Weight;
    using namespace Mila::Dnn::Quant::KvCache;

    /**
     * @brief LLaMA-style transformer (decoder-only) for autoregressive token prediction.
     *
     * Graph: TokenEmbedding -> RoPE -> LlamaBlock x N -> RmsNorm -> Linear (lm_head).
     * RoPE is applied to the full embedding stream after the token lookup; each
     * LlamaBlock receives rotary-encoded embeddings as input.
     *
     * Template parameters:
     *  - TDeviceType: device type (Cpu/Cuda)
     *  - TPrecision: tensor precision
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision,
        WeightQuantPolicy TWeightQuantization = NoWeightQuant, KvCachePolicy TKvCachePolicy = NoKvCompression>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class LlamaTransformer : public LanguageModelNetwork<TDeviceType, TPrecision>
    {
    public:
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using NetworkBase = LanguageModelNetwork<TDeviceType, TPrecision>;
        using TensorType = Tensor<TPrecision, MR>;
        using TokenEmbeddingType = TokenEmbedding<TDeviceType, dtype_t::INT32, TPrecision>;
        using LinearType = Linear<TDeviceType, TPrecision, TWeightQuantization>;
        using LmHeadLinearType = Linear<TDeviceType, TPrecision>;
        using RmsNormType = RmsNorm<TDeviceType, TPrecision>;
        using TransformerBlockType = LlamaBlock<TDeviceType, TPrecision, TWeightQuantization, TKvCachePolicy>;
        using TokenIndexType = Tensor<dtype_t::INT32, MR>;
        using ComponentPtr = typename NetworkBase::ComponentPtr;

        explicit LlamaTransformer( const std::string& name, const LlamaConfig& config, DeviceId device_id )
            : NetworkBase( name ), config_( config ), exec_context_( createExecutionContext( device_id ) )
        {
            config_.validate();

            if ( device_id.type != TDeviceType )
            {
                throw std::invalid_argument(
                    std::format( "LlamaTransformer: device type mismatch: expected {}, got {}",
                        deviceTypeToString( TDeviceType ),
                        deviceTypeToString( device_id.type ) ) );
            }

            createGraph();

            this->setExecutionContext( exec_context_.get() );
        }

        ~LlamaTransformer() override = default;


        TensorType& forward( const TokenIndexType& input ) override
        {
            if ( !this->isBuilt() )
                throw std::runtime_error( "LlamaTransformer must be built before calling forward()." );

            auto& embed_out = token_embedding_->forward( input );

            token_embed_out_ptr_ = &embed_out;

            if ( block_input_ptrs_.empty() || block_input_ptrs_.size() != transformer_blocks_.size() )
                throw std::runtime_error( "LlamaTransformer: forward internal state not initialized" );

            block_input_ptrs_[ 0 ] = token_embed_out_ptr_;

            for ( size_t i = 0; i < transformer_blocks_.size(); ++i )
            {
                auto& block_out = transformer_blocks_[ i ]->forward( *block_input_ptrs_[ i ] );

                block_output_ptrs_[ i ] = &block_out;

                if ( i + 1 < transformer_blocks_.size() )
                    block_input_ptrs_[ i + 1 ] = &block_out;
            }

            normalized_ptr_ = &final_rmsnorm_->forward( *block_output_ptrs_.back() );

            logits_ptr_ = &lm_head_->forward( *normalized_ptr_ );

            return *logits_ptr_;
        }

        TensorType& prefill( const TokenIndexType& input ) override
        {
            const int64_t B = input.shape()[ 0 ];
            const int64_t T_prompt = input.shape()[ 1 ];
            
            int64_t offset = 0;
            int64_t T_last = 0;

            TensorType* last_block_out = nullptr;

            // Chunked prefill loop -- input is sliced into prefill_chunk_size_ chunks and fed through the network sequentially to populate the KV cache.
            // The final chunk output is used to extract the last token representation for LM head inference.
            while ( offset < T_prompt )
            {
                const int64_t T_actual = std::min( prefill_chunk_size_, T_prompt - offset );
                T_last = T_actual;

                auto chunk_input = input.view( shape_t{ B, T_actual }, offset );

                // Embed directly -- output buffer lives in token_embedding_
                TensorType* block_input = &token_embedding_->forward( chunk_input );

                for ( size_t i = 0; i < transformer_blocks_.size(); ++i )
                {
                    auto& block_out = transformer_blocks_[ i ]->prefill( *block_input, offset );

                    block_input = &block_out;
                }

                last_block_out = block_input;
                offset += T_actual;
            }

            // Extract last position from final chunk output -- [B, 1, model_dim]
            dim_t last_pos_offset = (T_last - 1) * config_.getModelDim();
            auto last_pos = last_block_out->view(
                shape_t{ B, 1, config_.getModelDim() },
                last_pos_offset );
            
            normalized_ptr_ = &final_rmsnorm_->forward( last_pos );

            logits_ptr_ = &lm_head_->forward( *normalized_ptr_ );

            this->setCachedLength( T_prompt );

            return *logits_ptr_;
        }

        /**
         * @brief Teacher-forced log-likelihood of `input`: the log-probability the model gives each next token.
         *
         * Runs the prefill the generation path runs, but evaluates the head at every position rather than
         * the last, in windows of `getLogLikelihoodWindow()` rows, each reduced where the logits are
         * (NextTokenLogProbabilityOp) before the next overwrites it; the model synchronizes once, at the end.
         * A window of 1 costs one head evaluation per token; widening trades memory for passes and changes
         * no result.
         *
         * @throws std::out_of_range when a token past the first is outside the vocabulary.
         */
        SequenceLogLikelihood sequenceLogLikelihood( const TokenIndexType& input ) override
        {
            if ( !this->isBuilt() )
                throw std::runtime_error( "LlamaTransformer must be built before calling sequenceLogLikelihood()." );

            const dim_t B = input.shape()[ 0 ];
            const dim_t T = input.shape()[ 1 ];

            if ( B != 1 )
                throw std::invalid_argument( std::format(
                    "LlamaTransformer::sequenceLogLikelihood: batch must be 1, got {} -- the targets are the "
                    "sequence's own next tokens, which two rows cannot share", B ) );

            if ( T < 2 )
                throw std::invalid_argument( std::format(
                    "LlamaTransformer::sequenceLogLikelihood: need at least 2 tokens to score one position, got {}", T ) );

            const dim_t model_dim = config_.getModelDim();
            const dim_t vocab_size = config_.getVocabSize();
            const dim_t head_positions = resolveLogLikelihoodWindow( prefill_chunk_size_ );

            auto host_tokens = toHost<TensorDataType::INT32>( input, this->getExecutionContext() );

            requireTokensInVocabulary( host_tokens.data(), T, vocab_size );

            if ( !log_likelihood_op_ )
            {
                log_likelihood_op_ = std::make_unique<LogLikelihoodOpType>( this->getExecutionContext(), 0.0f );
            }

            log_likelihood_op_->begin( T - 1 );

            dim_t offset = 0;

            while ( offset < T )
            {
                const dim_t chunk_length = std::min<dim_t>( prefill_chunk_size_, T - offset );

                auto chunk_input = input.view( shape_t{ B, chunk_length }, offset );

                TensorType* block_input = &token_embedding_->forward( chunk_input );

                for ( auto& block : transformer_blocks_ )
                {
                    block_input = &block->prefill( *block_input, offset );
                }

                for ( dim_t start = 0; start < chunk_length; start += head_positions )
                {
                    const dim_t rows = std::min<dim_t>( head_positions, chunk_length - start );

                    // The sequence's last token predicts nothing, so a window holding only it has no work.
                    if ( offset + start + 1 >= T )
                        break;

                    auto window = block_input->view( shape_t{ B, rows, model_dim }, start * model_dim );

                    auto& normalized = final_rmsnorm_->forward( window );
                    auto& logits = lm_head_->forward( normalized );

                    // Llama applies no final-logit softcap.
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

    protected:

        TensorType& onDecode( const TokenIndexType& input, dim_t position ) override
        {
            auto& embed_out = token_embedding_->forward( input );

            token_embed_out_ptr_ = &embed_out;

            if ( block_input_ptrs_.empty() || block_input_ptrs_.size() != transformer_blocks_.size() )
                throw std::runtime_error( "LlamaTransformer: decode internal state not initialized" );

            block_input_ptrs_[ 0 ] = token_embed_out_ptr_;

            for ( size_t i = 0; i < transformer_blocks_.size(); ++i )
            {
                auto& block_out = transformer_blocks_[ i ]->decode( *block_input_ptrs_[ i ], position );

                block_output_ptrs_[ i ] = &block_out;

                if ( i + 1 < transformer_blocks_.size() )
                    block_input_ptrs_[ i + 1 ] = &block_out;
            }

            normalized_ptr_ = &final_rmsnorm_->forward( *block_output_ptrs_.back() );

            logits_ptr_ = &lm_head_->forward( *normalized_ptr_ );

            return *logits_ptr_;
        }

    public:

        TokenIndexType& backward( const TokenIndexType& input, const TensorType& output_grad ) override
        {
            if ( !this->isBuilt() )
                throw std::runtime_error( "LlamaTransformer must be built before calling backward()." );

            if ( !this->isTrainingMode() )
            {
                throw std::runtime_error( "LlamaTransformer: backward requires training mode." );
            }

            for ( size_t i = 0; i < transformer_blocks_.size(); ++i )
            {
                if ( !block_input_ptrs_[ i ] || !block_output_ptrs_[ i ] )
                    throw std::runtime_error( std::format( "LlamaTransformer: missing cached activation for block {}", i ) );
            }

            if ( !normalized_ptr_ || !logits_ptr_ )
                throw std::runtime_error( "LlamaTransformer: missing final activations for backward" );

            auto& normalized_grad = lm_head_->backward( *normalized_ptr_, output_grad );
            this->getExecutionContext()->synchronize();

            auto& last_block_grad = final_rmsnorm_->backward( *block_output_ptrs_.back(), normalized_grad );
            this->getExecutionContext()->synchronize();

            TensorType* curr_grad = &last_block_grad;

            for ( int64_t i = static_cast<int64_t>(transformer_blocks_.size()) - 1; i >= 0; --i )
            {
                auto& block_grad = transformer_blocks_[ static_cast<size_t>(i) ]->backward(
                    *block_input_ptrs_[ static_cast<size_t>(i) ], *curr_grad );

                curr_grad = &block_grad;

                this->getExecutionContext()->synchronize();
            }

            auto& input_grad = token_embedding_->backward( input, *curr_grad );
            this->getExecutionContext()->synchronize();

            return input_grad;
        }

        // ====================================================================
        // Gradient management
        // ====================================================================

        void zeroGradients() override
        {
            if ( !this->isBuilt() )
                return;

            token_embedding_->zeroGradients();

            for ( auto& block : transformer_blocks_ )
                block->zeroGradients();

            final_rmsnorm_->zeroGradients();
            lm_head_->zeroGradients();
        }

        // ====================================================================
        // Accessors / Diagnostics
        // ====================================================================

        // Structural kind comes from the Network base (ComponentType::Network);
        // the architecture family is reported here.
        ModelType getModelType() const
        {
            return ModelType::Llama;
        }

        MemoryStats getMemoryStats() const override
        {
            MemoryStats stats;

            for ( const auto& child : this->getComponents() )
            {
                stats += child->getMemoryStats();
            }

            stats.device_state_bytes += block_workspace_.deviceStorageBytes();
            stats.device_state_bytes += gqa_workspace_.deviceStorageBytes();

            return stats;
        }

        /**
         * @brief What build( context ) would allocate for the whole model, without allocating.
         *
         * Mirrors onBuilding(): at the prefill chunk the context carries, recurse with the same
         * per-child contexts, then add the shared block and GQA workspaces this transformer owns.
         * Llama does not tie the embedding to the head, so the two largest tensors are counted
         * separately and in full.
         */
        MemoryStats getRequiredMemory( const BuildContext& context ) const override
        {
            validateBuildContext( context );

            return requiredMemoryAtChunk( context, context.getResolvedPrefillSize() );
        }

        /**
         * @brief The chunk rungs choosePrefillChunk() walks for this family, largest first.
         */
        static constexpr dim_t kPrefillChunkRungs[] = { 1024, 512, 256, 128 };

    private:

        /**
         * @brief Positions the head evaluates per pass, bounded by what a pass can supply.
         *
         * A prefill pass produces at most prefill_chunk rows of block output, so a wider request names
         * rows that never exist. Both build() and getRequiredMemory() resolve through here, so the head
         * cannot be built at one width and priced at another.
         */
        dim_t resolveLogLikelihoodWindow( dim_t prefill_chunk ) const
        {
            return std::min<dim_t>( config_.getLogLikelihoodWindow(), prefill_chunk );
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

        /**
         * @brief Whether prefill attention runs the fused FlashAttention kernel: every BF16 build whose head size
         *        the kernel serves.
         *
         * No context threshold: the kernel is 5.9x to 9.1x faster than the cuBLASLt pipeline at Llama's
         * geometries from a 512-token prompt up (GqaFlashAttention.md 5.7). FP32 has no flash kernel, and the
         * 3.2 1B's head size of 64 is not one it serves.
         */
        bool usesFlashPrefill() const
        {
            return TPrecision == TensorDataType::BF16
                && TransformerBlockType::AttentionType::supportsFlashPrefill( config_.getModelDim() / config_.getNumHeads() );
        }

        /**
         * @brief Width of the shared prefill score buffer.
         *
         * Flash reads none, so it is zero -- no buffer (makeGqaWorkspace) -- and the O(chunk x context) term the
         * cuBLASLt path needs leaves the footprint.
         */
        dim_t prefillScoreWidth( dim_t T_ctx ) const
        {
            return usesFlashPrefill() ? dim_t{ 0 } : T_ctx;
        }

        /**
         * @brief What build( context ) would allocate with the given prefill chunk.
         */
        MemoryStats requiredMemoryAtChunk( const BuildContext& context, int64_t prefill_chunk ) const
        {
            const auto& input_shape = context.inputShape();
            const dim_t B = input_shape[ 0 ];
            const dim_t T = input_shape[ 1 ];
            const dim_t NH = config_.getNumHeads();
            const dim_t HS = config_.getModelDim() / NH;
            const std::size_t granularity = context.getAllocationGranularity();

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
                stats += this->template getComponentAs<TransformerBlockType>(
                    n + ".tf_layer_" + std::to_string( i ) )->getRequiredMemory( block_context );
            }

            stats += this->template getComponentAs<RmsNormType>( n + ".rmsn_final" )
                ->getRequiredMemory( final_context );

            stats += this->template getComponentAs<LmHeadLinearType>( n + ".lm_head" )
                ->getRequiredMemory( final_context );

            // The shared block and GQA workspaces, mirroring onBuilding(), the GQA one at the score width the flash
            // decision gives.
            if ( context.isInferenceMode() )
            {
                for ( dim_t width : llamaBlockWorkspaceSlotWidths( config_ ) )
                {
                    stats.device_state_bytes +=
                        occupiedDeviceBytes( storageBytes<TPrecision>( B * prefill_chunk * width ), granularity );
                }

                stats.device_state_bytes +=
                    gqaWorkspaceDeviceBytes<TPrecision>( granularity, B, NH, HS, T, prefill_chunk, prefillScoreWidth( T ) );
            }

            // RoPE cos/sin caches are process-wide, deduplicated by RopeCacheRegistry on
            // (theta, context length, head_dim). Every layer above reported one, but Llama's
            // layers are homogeneous, so exactly one cache exists for the whole model.
            stats.device_state_bytes -=
                std::max<dim_t>( config_.getNumLayers() - 1, 0 ) * ropeCacheBytes( HS, T, granularity );

            return stats;
        }

    public:

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << std::endl;
            oss << "Llama Network: " << this->getName() << std::endl;
            oss << "Device: " << this->getDeviceId().toString() << std::endl;

            oss << "Architecture:" << std::endl;
            oss << "  Vocabulary: " << config_.getVocabSize() << " tokens" << std::endl;
            oss << "  Max sequence length: " << config_.getMaxSequenceLength() << std::endl;
            oss << "  Model dim: " << config_.getModelDim() << std::endl;
            oss << "  Number of heads: " << config_.getNumHeads() << std::endl;
            oss << "  Number of KV heads: " << config_.getNumKVHeads() << std::endl;
            oss << "  Number of layers: " << config_.getNumLayers() << std::endl;
            oss << "  MLP hidden dim: " << config_.getHiddenDimension() << std::endl;
            oss << "  RoPE theta: " << config_.getRoPETheta() << std::endl;
            oss << "  RoPE scaling factor: " << config_.getRoPEFrequencyScaling().factor << std::endl;

            if ( this->isBuilt() )
            {
                oss << "  Parameters: " << this->parameterCount() << std::endl;
            }

            return oss.str();
        }

        void loadParameters( WeightsReader& reader )
        {
            const int device_index = this->getExecutionContext()->getDeviceId().index;

            auto consume = [&]( const std::string& full_name, const Serialization::ITensorBlob& blob )
            {
                auto [component_path, param_name] = parseParameterPath( full_name );

                ComponentPtr target = this->findComponent( component_path );
                target->loadParameter( param_name, blob );

                // The reader reuses its pinned staging slot as soon as this returns, so the
                // device read of blob must be complete. The quantize-on-load H2D is async on
                // the op stream and does not self-synchronize; force completion here.
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
        }

    protected:

        /**
         * @brief Bytes one RoPE cos/sin cache occupies for a given head width and context length.
         *
         * MUST match CudaRopeOp::getRequiredStateMemorySize -- FP32 regardless of the model
         * precision, one row per context position, half the head dimension, two caches.
         * Duplicated here because the deduplication is the transformer's to apply and it
         * needs the per-key size; the model-level comparison against getMemoryStats is what
         * holds the two together.
         */
        std::size_t ropeCacheBytes( dim_t head_dim, dim_t T_ctx, std::size_t granularity ) const noexcept
        {
            const dim_t cache_elements = T_ctx * ( head_dim / 2 );

            return 2 * occupiedDeviceBytes( static_cast<std::size_t>( cache_elements ) * sizeof( float ), granularity );
        }

        void onBuilding( const BuildContext& context ) override
        {
            validateBuildContext( context );

            const auto& input_shape = context.inputShape();

            const auto B = input_shape[ 0 ];
            const auto T = input_shape[ 1 ];

            // The chunk the caller resolved, threaded down to every block (and its GQA op) via
            // block_context. Also reused by prefill() and the shared GQA workspace. The build never
            // chooses one (Deployment.md section 7).
            prefill_chunk_size_ = context.getResolvedPrefillSize();

            // Blocks need full context_length so GQA can size the KV cache correctly.
            // LlamaBlock handles the prefill/decode split internally.
            shape_t block_shape = { B, T, config_.getModelDim() };
            BuildContext block_context =
                BuildContext( block_shape, context.getRuntimeMode(), context.shouldInitializeParameters() )
                .withPrefillSize( prefill_chunk_size_ )
                .withInstalledOutput( context.isInferenceMode() )
                .withFusedDecode( context.isInferenceMode() );

            // Inference: final_rmsnorm and lm_head process the configured head positions, one row for
            // generation. Training: the full sequence, for the loss. MUST agree with getRequiredMemory().
            shape_t final_shape = context.isInferenceMode()
                ? shape_t{ B, resolveLogLikelihoodWindow( prefill_chunk_size_ ), config_.getModelDim() }
                : shape_t{ B, T, config_.getModelDim() };

            BuildContext final_context( final_shape, context.getRuntimeMode(), context.shouldInitializeParameters() );

            transformer_blocks_.clear();
            transformer_blocks_.reserve( static_cast<size_t>(config_.getNumLayers()) );

            token_embedding_ = this->template getComponentAs<TokenEmbeddingType>( this->getName() + ".temb" );
            token_embedding_->build( embeddingContext( context, prefill_chunk_size_ ) );

            // One activation slot set for the whole stack: the layers run one at a time. Declared on block_context
            // as installed, which is how each block's footprint knows not to count the slots.
            if ( context.isInferenceMode() )
            {
                block_workspace_ = makeLlamaBlockWorkspace<TDeviceType, TPrecision>(
                    config_, this->getExecutionContext()->getDeviceId(), B, prefill_chunk_size_,
                    this->getName() + ".block_ws." );
            }

            for ( int64_t i = 0; i < config_.getNumLayers(); ++i )
            {
                std::string block_name = this->getName() + ".tf_layer_" + std::to_string( i );
                auto block = this->template getComponentAs<TransformerBlockType>( block_name );

                if ( context.isInferenceMode() )
                    block->installSharedWorkspace( block_workspace_ );

                block->build( block_context );
                transformer_blocks_.push_back( block );
            }

            final_rmsnorm_ = this->template getComponentAs<RmsNormType>( this->getName() + ".rmsn_final" );
            final_rmsnorm_->build( final_context );

            lm_head_ = this->template getComponentAs<LmHeadLinearType>( this->getName() + ".lm_head" );
            lm_head_->build( final_context );

            // One shared GQA transient for the whole stack: the layers run one at a time. Its score width and
            // every block's flash decision MUST come from the same place, or the cuBLASLt path overflows a
            // narrow buffer.
            if ( context.isInferenceMode() )
            {
                const dim_t NH = config_.getNumHeads();
                const dim_t HS = config_.getModelDim() / NH;
                const dim_t T_ctx = input_shape[ 1 ];

                gqa_workspace_ = makeGqaWorkspace<TDeviceType, TPrecision>(
                    this->getExecutionContext()->getDeviceId(), B, NH, HS, T_ctx, prefill_chunk_size_,
                    prefillScoreWidth( T_ctx ), this->getName() + ".gqa_ws." );

                const GqaState gqa_state = gqa_workspace_.state();

                for ( auto& block : transformer_blocks_ )
                {
                    block->setState( gqa_state );
                    block->setUseFlashPrefill( usesFlashPrefill() );
                }
            }

            // Every operation shares the context's one scratch buffer. Reserving it at the largest
            // request, the figure the footprint reports, is what puts it in the footprint.
            this->getExecutionContext()->reserveScratch( this->getMemoryStats().device_scratch_bytes );

            block_input_ptrs_.assign( transformer_blocks_.size(), nullptr );
            block_output_ptrs_.assign( transformer_blocks_.size(), nullptr );
            token_embed_out_ptr_ = nullptr;
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
            meta.set( "type", "LlamaTransformer" )
                .set( "version", int64_t( 1 ) )
                .set( "name", this->getName() );

            archive.writeMetadata( "transformer_meta.json", meta );
        }

    private:

        LlamaConfig config_;

        // The prefill chunk the caller resolved, taken from the BuildContext in onBuilding and
        // threaded to child components via BuildContext::withPrefillSize().
        int64_t prefill_chunk_size_{ 0 };

        std::shared_ptr<TokenEmbeddingType> token_embedding_{ nullptr };
        std::vector<std::shared_ptr<TransformerBlockType>> transformer_blocks_;
        std::shared_ptr<RmsNormType> final_rmsnorm_{ nullptr };
        std::shared_ptr<LmHeadLinearType> lm_head_{ nullptr };

        using LogLikelihoodOpType =
            typename OperationTraits<OperationType::NextTokenLogProbabilityOp, TDeviceType, TPrecision>::type;

        // Created at the first sequenceLogLikelihood(); holds no device memory.
        std::unique_ptr<LogLikelihoodOpType> log_likelihood_op_;

        // Shared activation and GQA transient workspaces -- inference only, owned here, shared across all blocks.
        LlamaBlockWorkspace<TDeviceType, TPrecision> block_workspace_{};
        GqaWorkspace<TDeviceType, TPrecision> gqa_workspace_{};

        // Activation pointers -- valid between forward() and the next backward().
        TensorType* token_embed_out_ptr_{ nullptr };   // rope's input
        //TensorType* encoder_out_ptr_{ nullptr };       // rope's output / blocks' input
        std::vector<TensorType*> block_input_ptrs_;
        std::vector<TensorType*> block_output_ptrs_;
        TensorType* normalized_ptr_{ nullptr };
        TensorType* logits_ptr_{ nullptr };

        // Declared last so it is destroyed first -- cudaStreamSynchronize() fires in
        // releaseResources() before any tensor cudaFree() calls from members above.
        std::unique_ptr<IExecutionContext> exec_context_{ nullptr };

        // ====================================================================
        // Graph construction
        // ====================================================================

        void createGraph()
        {
            TokenEmbeddingConfig embedding_config;
            embedding_config.withVocabSize( config_.getVocabSize() )
                .withEmbeddingDim( static_cast<size_t>(config_.getModelDim()) );

            auto embedding = std::make_shared<TokenEmbeddingType>( this->getName() + ".temb", embedding_config );
            this->addComponent( embedding );

            // Transformer blocks.
            for ( int64_t i = 0; i < config_.getNumLayers(); ++i )
            {
                LlamaConfig block_cfg( config_.getModelDim(), /*num_layers*/ 1 );
                block_cfg.withNumHeads( config_.getNumHeads() )
                    .withNumKVHeads( config_.getNumKVHeads() )
                    .withHiddenDimension( config_.getHiddenDimension() )
                    .withBias( config_.useBias() )
                    .withRoPETheta( config_.getRoPETheta() )
                    .withRoPEFrequencyScaling( config_.getRoPEFrequencyScaling() )
                    .withMaxSequenceLength( config_.getMaxSequenceLength() );

                auto layer = std::make_shared<TransformerBlockType>(
                    this->getName() + ".tf_layer_" + std::to_string( i ), block_cfg, std::nullopt );

                this->addComponent( layer );
            }

            // Final RMSNorm.
            auto rms_config = RmsNormConfig( shape_t{ config_.getModelDim() } )
                .withEpsilon( config_.getRMSNormEpsilon() )
                .withBias( false );

            auto final_rmsnorm = std::make_shared<RmsNormType>(
                this->getName() + ".rmsn_final", rms_config, std::nullopt );

            this->addComponent( final_rmsnorm );

            // Language model head -- projects model_dim -> vocab_size, no bias.
            auto lm_head_config = LinearConfig( config_.getModelDim(), config_.getVocabSize() )
                .withBias( false );

            auto lm_head = std::make_shared<LmHeadLinearType>(
                this->getName() + ".lm_head", lm_head_config, std::nullopt );

            this->addComponent( lm_head );
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
                    "LlamaTransformer: input must be rank 2 [B, T], got rank {}",
                    input_shape.size() ) );
            }

            if ( input_shape[ 0 ] < 1 || input_shape[ 1 ] < 1 )
            {
                throw std::invalid_argument( std::format(
                    "LlamaTransformer: B and T must be >= 1, got [{}, {}]",
                    input_shape[ 0 ], input_shape[ 1 ] ) );
            }
        }
    };
}
