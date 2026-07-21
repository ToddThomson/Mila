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
// #include <filesystem>
#include <algorithm>

export module Dnn.Components.LlamaTransformer;
export import :Config;
export import :Presets;
export import :Block;

import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.LanguageNetwork;
import Dnn.Component;
import Dnn.ComponentType;
import Dnn.ModelType;
import Dnn.Components.TokenEmbedding;
import Dnn.Components.Linear;
import Dnn.Components.RmsNorm;
import Dnn.Quantization.Weight.Policies;
import Dnn.Quantization.KvCache.Policy;
import Dnn.Components.Rope;
import Dnn.ActivationType;
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
import Serialization.PretrainedReader;
import Serialization.Tensor;

// import Compute.LinearOpTraits;
// import Compute.CudaLinearOpTraits;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;
    using namespace Mila::Dnn::Quant::Weight;
    using namespace Mila::Dnn::Quant::KvCache;

    // Attention-scratch ceiling for a single prefill pass. The chunk-dependent GQA
    // scratch (preatt + att + q_permute + v_out) must fit under this cap. Tuned for
    // the 12 GB / 8B-FP4 / 16K-context deployment target. A VRAM-aware budget
    // (CUDA device caps / MemoryStats) can replace this fixed cap later.
    inline constexpr int64_t kPrefillScratchByteCap = int64_t{ 1536 } * 1024 * 1024;

    // Largest prefill chunk in {512, 256, 128} whose attention scratch stays under
    // kPrefillScratchByteCap. Self-adjusts to model width, context length, and
    // compute precision. Floors at min(128, context_length) when even 128 exceeds
    // the cap. Computed once at network build time; see LlamaTransformer::onBuilding.
    inline int64_t computePrefillChunkSize(
        int64_t batch, int64_t num_heads, int64_t head_dim,
        int64_t context_length, int64_t precision_bytes )
    {
        // preatt + att dominate (each B*NH*chunk*T); q_permute + v_out add B*NH*chunk*HS.
        const int64_t scratch_per_chunk_row =
            batch * num_heads * ( 2 * context_length + 2 * head_dim ) * precision_bytes;

        for ( int64_t candidate : { int64_t{ 512 }, int64_t{ 256 }, int64_t{ 128 } } )
        {
            if ( candidate > context_length )
                continue;

            if ( scratch_per_chunk_row * candidate <= kPrefillScratchByteCap )
                return candidate;
        }

        return std::min<int64_t>( 128, context_length );
    }

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
    class LlamaTransformer : public LanguageNetwork<TDeviceType, TPrecision>
    {
    public:
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using NetworkBase = LanguageNetwork<TDeviceType, TPrecision>;
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
                    auto& block_out = transformer_blocks_[ i ]->prefill( *block_input, static_cast<int>( offset ) );

                    block_input = &block_out;
                }

                last_block_out = block_input;
                offset += T_actual;
            }

            // Extract last position from final chunk output -- [B, 1, model_dim]
            size_t last_pos_offset = static_cast<size_t>((T_last - 1) * config_.getModelDim());
            auto last_pos = last_block_out->view(
                shape_t{ B, 1, config_.getModelDim() },
                last_pos_offset );
            
            normalized_ptr_ = &final_rmsnorm_->forward( last_pos );

            logits_ptr_ = &lm_head_->forward( *normalized_ptr_ );

            return *logits_ptr_;
        }

        TensorType& decode( const TokenIndexType& input, int position ) override
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

            // GQA state memory usage
            for ( auto* t : { gqa_q_permute_.get(), gqa_preatt_.get(), gqa_att_.get(),
                              gqa_v_out_.get(), gqa_preatt_decode_.get(),
                              gqa_att_decode_.get(), gqa_v_out_decode_.get() } )
            {
                if ( t )
                    stats.device_state_bytes += t->getStorageSize();
            }

            return stats;
        }

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
            oss << "  RoPE scaling factor: " << config_.getRoPEScalingFactor() << std::endl;

            if ( this->isBuilt() )
            {
                oss << "  Parameters: " << this->parameterCount() << std::endl;
                oss << "  Batch size: " << batch_size_ << std::endl;
                oss << "  Sequence length: " << seq_length_ << std::endl;
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
        }

    protected:

        void onBuilding( const BuildContext& context ) override
        {
            validateBuildContext( context );

            const auto& input_shape = context.inputShape();

            const auto B = input_shape[ 0 ];
            const auto T = input_shape[ 1 ];

            // Tune the prefill chunk size once for the whole network and thread it down
            // to every block (and its GQA op) via block_context. Single source of truth
            // for prefill granularity; also reused by prefill() and the shared GQA workspace.
            prefill_chunk_size_ = computePrefillChunkSize(
                B, config_.getNumHeads(), config_.getModelDim() / config_.getNumHeads(),
                T, static_cast<int64_t>( TensorDataTypeTraits<TPrecision>::size_in_bytes ) );

            // Blocks need full context_length so GQA can size the KV cache correctly.
            // LlamaBlock handles the prefill/decode split internally.
            shape_t block_shape = { B, T, config_.getModelDim() };
            BuildContext block_context =
                BuildContext( block_shape, context.getRuntimeMode(), context.shouldInitializeParameters() )
                .withPrefillSize( prefill_chunk_size_ );

            // Inference: final_rmsnorm and lm_head only process the last position.
            // Training: must process full sequence for loss computation.
            shape_t final_shape = context.isInferenceMode() ? 
                shape_t{ B, 1, config_.getModelDim() } : shape_t{ B, T, config_.getModelDim() };

            BuildContext final_context( final_shape, context.getRuntimeMode(), context.shouldInitializeParameters() );

            transformer_blocks_.clear();
            transformer_blocks_.reserve( static_cast<size_t>(config_.getNumLayers()) );

            token_embedding_ = this->template getComponentAs<TokenEmbeddingType>( this->getName() + ".temb" );
            token_embedding_->build( context );

            for ( int64_t i = 0; i < config_.getNumLayers(); ++i )
            {
                std::string block_name = this->getName() + ".tf_layer_" + std::to_string( i );
                auto block = this->template getComponentAs<TransformerBlockType>( block_name );
                block->build( block_context );
                transformer_blocks_.push_back( block );
            }

            final_rmsnorm_ = this->template getComponentAs<RmsNormType>( this->getName() + ".rmsn_final" );
            final_rmsnorm_->build( final_context );

            lm_head_ = this->template getComponentAs<LmHeadLinearType>( this->getName() + ".lm_head" );
            lm_head_->build( final_context );

            // Allocate one shared GQA transient workspace for inference and wire it
            // into every block's attention layer. All 28 layers execute sequentially,
            // so a single allocation is reused across the full forward pass.
            if ( context.isInferenceMode() )
            {
                const int64_t B = input_shape[ 0 ];
                const int64_t NH = config_.getNumHeads();
                const int64_t NKV = config_.getNumKVHeads();
                const int64_t HS = config_.getModelDim() / NH;
                const int64_t T_ctx = input_shape[ 1 ];
                auto device = this->getExecutionContext()->getDeviceId();

                gqa_q_permute_ = std::make_unique<TensorType>(
                    device, shape_t{ B, NH, prefill_chunk_size_, HS }, this->getName() + ".gqa_ws.q_perm" );

                gqa_preatt_ = std::make_unique<TensorType>(
                    device, shape_t{ B, NH, prefill_chunk_size_, T_ctx }, this->getName() + ".gqa_ws.preatt" );

                gqa_att_ = std::make_unique<TensorType>(
                    device, shape_t{ B, NH, prefill_chunk_size_, T_ctx }, this->getName() + ".gqa_ws.att" );

                gqa_v_out_ = std::make_unique<TensorType>(
                    device, shape_t{ B, NH, prefill_chunk_size_, HS }, this->getName() + ".gqa_ws.v_out" );

                gqa_preatt_decode_ = std::make_unique<TensorType>(
                    device, shape_t{ B, NH, 1, T_ctx }, this->getName() + ".gqa_ws.preatt_dec" );

                gqa_att_decode_ = std::make_unique<TensorType>(
                    device, shape_t{ B, NH, 1, T_ctx }, this->getName() + ".gqa_ws.att_dec" );

                gqa_v_out_decode_ = std::make_unique<TensorType>(
                    device, shape_t{ B, NH, 1, HS }, this->getName() + ".gqa_ws.v_out_dec" );

                GqaState gqa_state;
                gqa_state.q_permute = gqa_q_permute_.get();
                gqa_state.preatt = gqa_preatt_.get();
                gqa_state.att = gqa_att_.get();
                gqa_state.v_out = gqa_v_out_.get();
                gqa_state.preatt_decode = gqa_preatt_decode_.get();
                gqa_state.att_decode = gqa_att_decode_.get();
                gqa_state.v_out_decode = gqa_v_out_decode_.get();

                for ( auto& block : transformer_blocks_ )
                    block->setState( gqa_state );
            }

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
            SerializationMetadata meta;
            meta.set( "type", "LlamaTransformer" )
                .set( "version", int64_t( 1 ) )
                .set( "name", this->getName() )
                .set( "vocab_size", config_.getVocabSize() )
                .set( "max_seq_length", config_.getMaxSequenceLength() )
                .set( "model_dim", config_.getModelDim() )
                .set( "num_heads", config_.getNumHeads() )
                .set( "num_kv_heads", config_.getNumKVHeads() )
                .set( "num_layers", config_.getNumLayers() )
                .set( "hidden_dim", config_.getHiddenDimension() )
                .set( "rope_theta", static_cast<double>(config_.getRoPETheta()) )
                .set( "rope_scaling", static_cast<double>(config_.getRoPEScalingFactor()) )
                .set( "use_bias", config_.useBias() );

            if ( this->isBuilt() )
            {
                meta.set( "input_shape", input_shape_ )
                    .set( "embedding_shape", embedding_shape_ )
                    .set( "output_shape", output_shape_ )
                    .set( "batch_size", batch_size_ )
                    .set( "seq_length", seq_length_ );
            }

            archive.writeMetadata( "transformer_meta.json", meta );
        }

    private:

        LlamaConfig config_;

        shape_t input_shape_;
        shape_t embedding_shape_;
        shape_t output_shape_;
        int64_t batch_size_{ 0 };
        int64_t seq_length_{ 0 };

        // Tuned prefill chunk size -- single source of truth, set in onBuilding and
        // threaded to child components via BuildContext::withPrefillSize().
        int64_t prefill_chunk_size_{ 0 };

        std::shared_ptr<TokenEmbeddingType> token_embedding_{ nullptr };
        std::vector<std::shared_ptr<TransformerBlockType>> transformer_blocks_;
        std::shared_ptr<RmsNormType> final_rmsnorm_{ nullptr };
        std::shared_ptr<LmHeadLinearType> lm_head_{ nullptr };
        
        // Inference-only prefill buffer for autoregressive decoding.
        std::unique_ptr<TensorType> prefill_{ nullptr };

        // Shared GQA transient workspace -- inference only, owned here, shared across all blocks.
        // ~26 MB total vs ~352 MB x 28 layers in the per-layer self-owned design.
        std::unique_ptr<TensorType> gqa_q_permute_{ nullptr };
        std::unique_ptr<TensorType> gqa_preatt_{ nullptr };
        std::unique_ptr<TensorType> gqa_att_{ nullptr };
        std::unique_ptr<TensorType> gqa_v_out_{ nullptr };
        std::unique_ptr<TensorType> gqa_preatt_decode_{ nullptr };
        std::unique_ptr<TensorType> gqa_att_decode_{ nullptr };
        std::unique_ptr<TensorType> gqa_v_out_decode_{ nullptr };

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
            embedding_config.withVocabSize( static_cast<size_t>(config_.getVocabSize()) )
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

        void validateLeadingShape( const shape_t& leading_shape ) const
        {
            if ( leading_shape.size() != 2 )
                throw std::invalid_argument(
                    "LlamaTransformer: Leading shape must have rank 2 (batch_size, seq_length)" );

            if ( leading_shape[ 1 ] > config_.getMaxSequenceLength() )
                throw std::invalid_argument(
                    std::format( "LlamaTransformer: sequence length {} exceeds maximum {}",
                        leading_shape[ 1 ], config_.getMaxSequenceLength() ) );
        }

        static LlamaConfig createConfigFromMetadata( const PretrainedMetadata& metadata )
        {
            LlamaConfig config( static_cast<dim_t>(metadata.embedding_dim),
                                static_cast<dim_t>(metadata.num_layers) );

            config.withVocabularyLength( static_cast<dim_t>(metadata.vocab_size) )
                .withMaxSequenceLength( static_cast<dim_t>(metadata.max_seq_length) )
                .withNumHeads( static_cast<dim_t>(metadata.num_heads) )
                .withNumKVHeads( static_cast<dim_t>(metadata.num_kv_heads) )
                .withHiddenDimension( static_cast<dim_t>(metadata.hidden_dim) )
                .withRoPETheta( metadata.rope_theta )
                // REVIEW: There is a scaling factor but the exact reason this was commented out is unclear.
                // .withRoPEScalingFactor( metadata.rope_scaling )
                .withBias( metadata.use_bias );

            return config;
        }
    };
}
