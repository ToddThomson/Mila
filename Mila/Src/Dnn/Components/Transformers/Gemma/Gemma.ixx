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
 *    drives them polymorphically through IDecoderLayer (one virtual call per layer
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
 * Two Gemma deltas are handled OUTSIDE this file by deliberate design decision
 * (BACKLOG Step 5d, 2026-06-20):
 *  - Embedding scale (x sqrt(hidden_size)) folds into the converter, which scales
 *    the (untied) embedding table and writes lm_head its own unscaled copy. The
 *    forward path here is therefore structurally identical to LlamaTransformer.
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

export module Dnn.Components.GemmaTransformer;

import Dnn.Components.GemmaConfig;
import Dnn.Components.GemmaBlock;
import Dnn.Components.IDecoderLayer;

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
import Serialization.Metadata;
import Serialization.PretrainedReader;
import Serialization.Tensor;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;
    using namespace Mila::Dnn::Quant::Weight;
    using namespace Mila::Dnn::Quant::KvCache;

    // Attention-scratch ceiling for a single prefill pass. The chunk-dependent GQA
    // scratch (preatt + att + q_permute + v_out) must fit under this cap. Mirrors
    // LlamaTransformer's budget; a VRAM-aware budget can replace it later.
    inline constexpr int64_t kGemmaPrefillScratchByteCap = int64_t{ 1536 } * 1024 * 1024;

    // Manual prefill-chunk override for VRAM experiments: set > 0 to force the chunk
    // (clamped to context_length); 0 = use the heuristic below. The per-layer prefill
    // activation footprint (all N layers, dominated by the GeGLU FFN) scales with this,
    // so it is the primary prefill VRAM lever on memory-constrained cards.
    inline constexpr int64_t kGemmaPrefillChunkOverride = 32;

    // Largest prefill chunk in {512, 256, 128, 64, 32, 16} whose attention scratch
    // stays under the cap. head_dim is the MAX over local/global so the budget is
    // conservative for the wider global geometry.
    //
    // NOTE: this cap budgets only the GQA attention scratch, NOT the per-layer prefill
    // activation buffers (the dominant prefill VRAM term on a wide/deep model). Until the
    // budget is made activation-aware, use kGemmaPrefillChunkOverride to sweep by hand.
    inline int64_t computeGemmaPrefillChunkSize(
        int64_t batch, int64_t num_heads, int64_t head_dim,
        int64_t context_length, int64_t precision_bytes )
    {
        if constexpr ( kGemmaPrefillChunkOverride > 0 )
        {
            return std::min<int64_t>( kGemmaPrefillChunkOverride, context_length );
        }
        else
        {
            const int64_t scratch_per_chunk_row =
                batch * num_heads * ( 2 * context_length + 2 * head_dim ) * precision_bytes;

            for ( int64_t candidate : { int64_t{ 512 }, int64_t{ 256 }, int64_t{ 128 },
                                        int64_t{ 64 }, int64_t{ 32 }, int64_t{ 16 } } )
            {
                if ( candidate > context_length )
                    continue;

                if ( scratch_per_chunk_row * candidate <= kGemmaPrefillScratchByteCap )
                    return candidate;
            }

            return std::min<int64_t>( 16, context_length );
        }
    }

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
    class GemmaTransformer : public LanguageNetwork<TDeviceType, TPrecision>
    {
    public:
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using NetworkBase = LanguageNetwork<TDeviceType, TPrecision>;
        using TensorType = Tensor<TPrecision, MR>;
        using TokenEmbeddingType = TokenEmbedding<TDeviceType, dtype_t::INT32, TPrecision>;
        using LmHeadLinearType = Linear<TDeviceType, TPrecision>;
        using RmsNormType = RmsNorm<TDeviceType, TPrecision>;
        using LocalBlockType = GemmaBlock<TDeviceType, TPrecision, false, TWeightQuantization, TKvCachePolicy>;
        using GlobalBlockType = GemmaBlock<TDeviceType, TPrecision, true, TWeightQuantization, TKvCachePolicy>;
        using DecoderLayerType = IDecoderLayer<TDeviceType, TPrecision>;
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
            if ( !this->isBuilt() )
                throw std::runtime_error( "GemmaTransformer must be built before calling prefill()." );

            const int64_t B = input.shape()[ 0 ];
            const int64_t T_prompt = input.shape()[ 1 ];

            int64_t offset = 0;
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

                for ( auto* layer : layers_ )
                {
                    auto& block_out = layer->prefill( *block_input, static_cast<int>( offset ) );
                    block_input = &block_out;
                }

                last_block_out = block_input;
                offset += T_actual;
            }

            // Extract the last position from the final chunk output -> [B, 1, model_dim].
            size_t last_pos_offset = static_cast<size_t>((T_last - 1) * config_.getModelDim());
            auto last_pos = last_block_out->view(
                shape_t{ B, 1, config_.getModelDim() }, last_pos_offset );

            normalized_ptr_ = &final_rmsnorm_->forward( last_pos );

            logits_ptr_ = &lm_head_->forward( *normalized_ptr_ );

            return *logits_ptr_;
        }

        TensorType& decode( const TokenIndexType& input, int position ) override
        {
            if ( !this->isBuilt() )
                throw std::runtime_error( "GemmaTransformer must be built before calling decode()." );

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
        }

    protected:

        void onBuilding( const BuildContext& context ) override
        {
            validateBuildContext( context );

            const auto& input_shape = context.inputShape();
            const int64_t B = input_shape[ 0 ];
            const int64_t T = input_shape[ 1 ];

            // Tune the prefill chunk once and thread it to every block (and its GQA op)
            // via block_context. Use the MAX head_dim so the scratch budget is sized for
            // the wider global geometry.
            const int64_t head_dim_max = std::max( config_.getHeadDim(), config_.getGlobalHeadDim() );
            prefill_chunk_size_ = computeGemmaPrefillChunkSize(
                B, config_.getNumHeads(), head_dim_max,
                T, static_cast<int64_t>( TensorDataTypeTraits<TPrecision>::size_in_bytes ) );

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

            layers_.clear();
            layers_.reserve( static_cast<size_t>(config_.getNumLayers()) );

            for ( int64_t i = 0; i < config_.getNumLayers(); ++i )
            {
                const std::string block_name = this->getName() + ".tf_layer_" + std::to_string( i );

                // Heterogeneous layer list: the per-layer kind selects the GemmaBlock
                // instantiation (the two differ in head_dim / KV-heads / K=V / window / RoPE).
                if ( config_.isGlobalLayer( static_cast<dim_t>( i ) ) )
                {
                    auto block = this->template getComponentAs<GlobalBlockType>( block_name );
                    block->build( block_context );
                    layers_.push_back( static_cast<DecoderLayerType*>( block.get() ) );
                }
                else
                {
                    auto block = this->template getComponentAs<LocalBlockType>( block_name );
                    block->build( block_context );
                    layers_.push_back( static_cast<DecoderLayerType*>( block.get() ) );
                }
            }

            final_rmsnorm_ = this->template getComponentAs<RmsNormType>( this->getName() + ".rmsn_final" );
            final_rmsnorm_->build( final_context );

            lm_head_ = this->template getComponentAs<LmHeadLinearType>( this->getName() + ".lm_head" );
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

        // Tuned prefill chunk size — single source of truth, set in onBuilding and
        // threaded to child components via BuildContext::withPrefillSize().
        int64_t prefill_chunk_size_{ 0 };

        std::shared_ptr<TokenEmbeddingType> token_embedding_{ nullptr };
        // Non-owning, polymorphic view of the heterogeneous block list; the concrete
        // blocks are owned by the component tree (addComponent). Valid after build.
        std::vector<DecoderLayerType*> layers_;
        std::shared_ptr<RmsNormType> final_rmsnorm_{ nullptr };
        std::shared_ptr<LmHeadLinearType> lm_head_{ nullptr };

        // Shared GQA transient workspace — inference only, owned here, shared across
        // all blocks. q_permute/v_out are sized at the MAX head_dim (global) so the
        // local layers reuse a prefix; preatt/att are head_dim-independent.
        std::unique_ptr<TensorType> gqa_q_permute_{ nullptr };
        std::unique_ptr<TensorType> gqa_preatt_{ nullptr };
        std::unique_ptr<TensorType> gqa_att_{ nullptr };
        std::unique_ptr<TensorType> gqa_v_out_{ nullptr };
        std::unique_ptr<TensorType> gqa_preatt_decode_{ nullptr };
        std::unique_ptr<TensorType> gqa_att_decode_{ nullptr };
        std::unique_ptr<TensorType> gqa_v_out_decode_{ nullptr };

        // Activation pointers — valid between prefill/decode and the next call.
        TensorType* normalized_ptr_{ nullptr };
        TensorType* logits_ptr_{ nullptr };

        // Declared last so it is destroyed first — cudaStreamSynchronize() fires in
        // releaseResources() before any tensor cudaFree() from members above.
        std::unique_ptr<IExecutionContext> exec_context_{ nullptr };

        // ====================================================================
        // Graph construction
        // ====================================================================

        void createGraph()
        {
            TokenEmbeddingConfig embedding_config;
            embedding_config.withVocabSize( static_cast<size_t>(config_.getVocabSize()) )
                .withEmbeddingDim( static_cast<size_t>(config_.getModelDim()) );

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

            // Language model head — model_dim -> vocab_size, no bias. Untied from the
            // embedding table (its own blob; the converter writes an unscaled copy).
            auto lm_head_config = LinearConfig( config_.getModelDim(), config_.getVocabSize() )
                .withBias( false );

            this->addComponent(
                std::make_shared<LmHeadLinearType>( this->getName() + ".lm_head", lm_head_config, std::nullopt ) );
        }

        // ====================================================================
        // GQA workspace
        // ====================================================================

        void allocateAndWireGqaWorkspace( int64_t B, int64_t T_ctx )
        {
            const int64_t NH = config_.getNumHeads();
            const int64_t HS_max = std::max( config_.getHeadDim(), config_.getGlobalHeadDim() );
            auto device = this->getExecutionContext()->getDeviceId();
            const std::string n = this->getName();

            gqa_q_permute_ = std::make_unique<TensorType>(
                device, shape_t{ B, NH, prefill_chunk_size_, HS_max }, n + ".gqa_ws.q_perm" );
            gqa_preatt_ = std::make_unique<TensorType>(
                device, shape_t{ B, NH, prefill_chunk_size_, T_ctx }, n + ".gqa_ws.preatt" );
            gqa_att_ = std::make_unique<TensorType>(
                device, shape_t{ B, NH, prefill_chunk_size_, T_ctx }, n + ".gqa_ws.att" );
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
