/**
 * @file GptModel.ixx
 * @brief GPT inference model.
 *
 * Inference-only wrapper around a loaded GptTransformer network.
 * No training, no optimizer, no gradients.
 *
 * Two loading paths:
 *
 *  fromPretrained() -- third-party weights (e.g. HuggingFace GPT-2) via
 *                     PretrainedModelReader. Primary path for Mila chat.
 *
 *  fromCheckpoint() -- Mila-native artifact produced by GptTransformer::save()
 *                     via ModelArchive. Round-trip path after training.
 */

module;
#include <memory>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include <filesystem>
#include <format>
#include <random>
#include <optional>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <cmath>
#include <functional>
#include <stop_token>
#include <unordered_set>

export module Dnn.Models.GptModel;

import Dnn.LanguageModel;
import Dnn.LanguageNetwork;
import Dnn.GenerateParams;
import Dnn.GenerateStatus;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.Component;
import Dnn.RuntimeMode;
import Dnn.Components.GptTransformer;
import Compute.Device;
import Compute.DeviceId;
import Compute.DeviceType;
import Compute.DeviceTypeTraits;
import Compute.DeviceTypeTraits.Cpu;
import Compute.CpuMemoryResource;
import Compute.ExecutionContextFactory;
import Serialization.ModelArchive;
import Serialization.Metadata;
import Serialization.OpenMode;
import Serialization.Mode;
import Serialization.ZipSerializer;
import Serialization.PretrainedReader;
import Logging.Logger;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    /**
     * @brief GPT inference model.
     *
     * Owns a loaded, built GptTransformer and exposes generate() for
     * autoregressive text generation.
     *
     * Construction is only possible via fromPretrained() or fromCheckpoint().
     * The network is always in a built, weights-loaded, inference-mode state
     * when generation is called.
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class GptModel : public LanguageModel<TDeviceType, TPrecision>
    {
    public:
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using ModelBase = LanguageModel<TDeviceType, TPrecision>;
        using TensorType = Tensor<TPrecision, MR>;
        using TokenIndexType = Tensor<dtype_t::INT32, MR>;
        using GptTransformerType = GptTransformer<TDeviceType, TPrecision>;

        GptModel( const GptModel& ) = delete;
        GptModel& operator=( const GptModel& ) = delete;
        GptModel( GptModel&& ) = default;
        GptModel& operator=( GptModel&& ) = default;

        ~GptModel() = default;

        // ====================================================================
        // Factory -- the sole construction paths
        // ====================================================================

        /**
         * @brief Load from third-party pretrained weights.
         *
         * Reads weights from a Mila-compatible pretrained artifact produced
         * by converting third-party checkpoints (e.g. HuggingFace GPT-2)
         * via PretrainedModelReader.
         *
         * @param path           Path to the pretrained artifact.
         * @param context_length Maximum sequence length to build for.
         * @param device_id      Target device.
         * @param strict         Throws on unknown parameter names if true.
         * @return               Inference-ready GptModel.
         */
        static std::unique_ptr<GptModel> fromPretrained(
            const std::filesystem::path& path,
            dim_t context_length,
            DeviceId device_id = DeviceId{ TDeviceType, 0 },
            bool strict = true )
        {
            if ( device_id.type != TDeviceType )
                throw std::invalid_argument( std::format(
                    "GptModel::fromPretrained: device type mismatch: expected {}, got {}",
                    deviceTypeToString( TDeviceType ),
                    deviceTypeToString( device_id.type ) ) );

            PretrainedModelReader reader( path );
            const auto& metadata = reader.getPretrainedMetadata();

            GptConfig config = configFromMetadata( metadata );

            auto network = std::make_unique<GptTransformerType>(
                metadata.model_name, config, device_id );

            BuildContext build_context(
                shape_t{ 1, static_cast<int64_t>(context_length) },
                RuntimeMode::Inference,
                false );

            network->build( build_context );
            network->loadParameters( reader, strict );

            return std::unique_ptr<GptModel>( new GptModel( std::move( network ), config, context_length, RuntimeMode::Inference ) );
        }

        /**
         * @brief Load from a Mila-native serialized artifact.
         *
         * Reads a checkpoint or weights-only artifact produced by
         * GptTransformer::save() via ModelArchive.
         *
         * @param path           Path to the Mila archive.
         * @param device_id      Target device.
         * @param context_length Deployment context length. Zero takes the geometry the
         *                       checkpoint was built with, falling back to the trained maximum.
         * @return               Inference-ready GptModel.
         */
        static std::unique_ptr<GptModel> fromCheckpoint(
            const std::filesystem::path& path,
            DeviceId device_id = DeviceId{ TDeviceType, 0 },
            dim_t context_length = 0 )
        {
            if ( device_id.type != TDeviceType )
                throw std::invalid_argument( std::format(
                    "GptModel::fromCheckpoint: device type mismatch: expected {}, got {}",
                    deviceTypeToString( TDeviceType ),
                    deviceTypeToString( device_id.type ) ) );

            ModelArchive archive( path.string(), std::make_unique<ZipSerializer>(), OpenMode::Read );

            GptConfig config = GptTransformerType::configFromArchive( archive );

            // Default to the geometry the checkpoint was built with: a resumed run wants
            // the shape it left off at, not a fresh guess. Fall back to the trained
            // maximum when the archive recorded no build geometry.
            if ( context_length <= 0 )
            {
                context_length = GptTransformerType::buildSequenceLengthFromArchive( archive );
            }

            if ( context_length <= 0 )
            {
                context_length = config.getMaxSequenceLength();
            }

            SerializationMetadata net_meta = archive.readMetadata( "network/meta.json" );
            const std::string model_name = net_meta.has( "name" )
                ? net_meta.getString( "name" ) : std::string( "gpt" );

            auto network = std::make_unique<GptTransformerType>( model_name, config, device_id );

            BuildContext build_context(
                shape_t{ 1, static_cast<int64_t>( context_length ) },
                RuntimeMode::Inference,
                false );

            network->build( build_context );

            // The graph exists and is built; load restores weights into it.
            network->load( archive, SerializationMode::Checkpoint );

            return std::unique_ptr<GptModel>(
                new GptModel( std::move( network ), config, context_length, RuntimeMode::Inference ) );
        }

        /**
         * @brief Write a Mila-native archive that fromCheckpoint() can restore.
         *
         * Writes the network config, the component graph, and one blob per parameter.
         * Weights only -- optimizer state belongs to the trainer, which owns its own
         * archive scope.
         *
         * @param path Destination archive path (overwritten if it exists).
         * @param mode Serialization mode recorded in the archive.
         *
         * @throws std::runtime_error if the archive cannot be opened or a component
         *         cannot serialize its parameters.
         */
        void saveCheckpoint(
            const std::filesystem::path& path,
            SerializationMode mode = SerializationMode::Checkpoint ) const
        {
            ModelArchive archive( path.string(), std::make_unique<ZipSerializer>(), OpenMode::Write );

            // this-> is required: network_ is a member of a dependent base.
            this->network_->save( archive, mode );
        }

        // ====================================================================
        // Accessors
        // ====================================================================

        const GptConfig& getConfig() const noexcept
        {
            return config_;
        }

        // ====================================================================
        // Diagnostics
        // ====================================================================

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "GptModel\n";
            oss << "Vocabulary: " << config_.getVocabSize() << " tokens\n";

            // Two different numbers, and only the first bounds generation: the session was
            // built for context_length_, while config_ carries the maximum the checkpoint
            // declares. Reporting only the latter overstates the usable window.
            oss << "Context length: " << context_length_ << " tokens\n";
            oss << "Architectural maximum: " << config_.getMaxSequenceLength() << " tokens\n";
            oss << "Embedding dim: " << config_.getEmbeddingSize() << "\n";
            oss << "Layers: " << config_.getNumLayers() << "\n";
            oss << "Heads: " << config_.getNumHeads() << "\n";
            oss << "MLP hidden dim: " << config_.getHiddenSize() << "\n";

            return oss.str();
        }

    protected:

        // ====================================================================
        // LanguageModel overrides
        // ====================================================================

        // REVIEW: onGenerating() is the core of the autoregressive generation loop for all LanguageModel subclasses.
        // It is currently implemented in each subclass, but it could be refactored into a common implementation in the base class,
        // with subclass-specific hooks for prefill and decode. This would reduce code duplication and make it easier to maintain.

        /**
         * @brief Prefill + KV-cache decode loop with per-token streaming.
         */
        GenerateStatus onGenerating(
            std::span<const int32_t> prompt_tokens,
            const std::function<void(int32_t)>& on_token,
            const GenerateParams& params,
            std::stop_token stop ) override
        {
            // Stop set: GPT-2 <|endoftext|> by default, or the caller's per-call override.
            std::unordered_set<int32_t> stop_ids;
            if ( params.stop_tokens.empty() )
                stop_ids.insert( eos_token_ );
            else
                for ( auto id : params.stop_tokens )
                    stop_ids.insert( static_cast<int32_t>( id ) );

            // Host sampler path (device-sampler migration deferred): time-seeded.
            std::mt19937 rng( static_cast<std::mt19937::result_type>(
                std::chrono::high_resolution_clock::now().time_since_epoch().count() ) );

            if ( prompt_tokens.size() > static_cast<size_t>( context_length_ ) )
            {
                throw std::invalid_argument( std::format(
                    "GptModel::onGenerating: prompt length {} exceeds deployment context length {}",
                    prompt_tokens.size(), context_length_ ) );
            }

            const int64_t seq_len = static_cast<int64_t>( prompt_tokens.size() );

            auto prefill_input = makeTokenTensor( prompt_tokens );
            auto& logits = this->getLanguageNetwork().prefill( prefill_input );
            this->getLanguageNetwork().synchronize();

            int32_t next_token = sampleFromLogits(
                logits, 0, params.sampling.temperature, params.sampling.top_k, rng );

            if ( stop_ids.contains( next_token ) )
                return GenerateStatus::Success;

            on_token( next_token );

            dim_t position = seq_len;
            const int max_new = params.max_new_tokens.value_or( static_cast<int>( context_length_ ) );

            for ( int step = 1; step < max_new; ++step )
            {
                if ( stop.stop_requested() )
                    return GenerateStatus::ClientCancelled;

                // decode reads the learned positional embedding at `position`, and GPT-2 has
                // exactly context_length_ of them -- so one step past the end is an
                // out-of-bounds read, not a degraded answer. max_new defaults to the whole
                // context without subtracting the prompt, so the default budget alone reaches
                // here. Mirrors the cache_has_room guard in GemmaModel.
                if ( position >= context_length_ )
                    return GenerateStatus::ContextOverflow;

                auto decode_input = makeTokenTensor( std::vector{ next_token } );
                auto& decode_logits = this->getLanguageNetwork().decode( decode_input, position );
                this->getLanguageNetwork().synchronize();

                next_token = sampleFromLogits(
                    decode_logits, 0, params.sampling.temperature, params.sampling.top_k, rng );

                if ( stop_ids.contains( next_token ) )
                    return GenerateStatus::Success;

                on_token( next_token );
                ++position;
            }

            return GenerateStatus::MaxNewTokensReached;
        }

        /**
         * @brief GPT-2 end-of-text token id.
         */
        int32_t eosToken() const noexcept override
        {
            return eos_token_;
        }

        /**
         * @brief Maximum sequence length from GPT config.
         */
        dim_t maxSequenceLength() const noexcept override
        {
            return config_.getMaxSequenceLength();
        }

        /**
         * @brief Vocabulary size from GPT config.
         */
        dim_t vocabSize() const noexcept override
        {
            return config_.getVocabSize();
        }

        /**
         * @brief Training loop -- not yet implemented for GptModel.
         *
         * @throws std::runtime_error always.
         */
        void onTraining() override
        {
            throw std::runtime_error(
                "GptModel::onTraining: training not yet implemented" );
        }

    private:

        explicit GptModel(
            std::unique_ptr<GptTransformerType> network,
            const GptConfig& config,
            int64_t context_length,
            RuntimeMode runtime_mode )
            : ModelBase( std::move( network ), runtime_mode ), context_length_( context_length ), config_( config )
        {}

        explicit GptModel(
            std::unique_ptr<GptTransformerType> network,
            const GptConfig& config )
            : ModelBase( std::move( network ), RuntimeMode::Inference ), config_( config )
        {}

        GptConfig config_;
        int64_t context_length_;

        // REVIEW: Should come from tokenizer metadata when tokenizer support added.
        static constexpr int32_t eos_token_ = 50256;  // GPT-2 <|endoftext|>

        // ====================================================================
        // Generation helpers
        // ====================================================================

        void truncateIfNeeded( std::vector<int32_t>& tokens ) const
        {
            int64_t seq_len = static_cast<int64_t>(tokens.size());

            if ( seq_len > config_.getMaxSequenceLength() )
            {
                Logging::Logger::warning( std::format(
                    "GptModel: sequence length {} exceeds max {}, truncating from start",
                    seq_len, config_.getMaxSequenceLength() ) );

                tokens.erase( tokens.begin(),
                    tokens.begin() + (seq_len - config_.getMaxSequenceLength()) );
            }
        }

        TokenIndexType makeTokenTensor( std::span<const int32_t> token_ids ) const
        {
            shape_t shape = { 1, static_cast<int64_t>(token_ids.size()) };
            TokenIndexType device_tensor( this->getDeviceId(), shape );
            Tensor<dtype_t::INT32, CpuMemoryResource> cpu_tensor( Device::Cpu(), shape );
            std::memcpy( cpu_tensor.data(), token_ids.data(),
                token_ids.size() * sizeof( int32_t ) );
            copy( cpu_tensor, device_tensor );

            return device_tensor;
        }

        int32_t sampleFromLogits(
            const TensorType& logits,
            dim_t position,
            float temperature,
            int top_k,
            std::mt19937& rng ) const
        {
            dim_t seq_len = logits.shape()[ 1 ];
            shape_t shape = { 1, seq_len, config_.getVocabSize() };
            Tensor<TPrecision, CpuMemoryResource> cpu( Device::Cpu(), shape );
            copy( logits, cpu );

            const float* last = cpu.data() + position * config_.getVocabSize();

            return sampleToken( last,
                static_cast<size_t>(config_.getVocabSize()),
                temperature, top_k, rng );
        }

        static int32_t sampleToken(
            const float* logits,
            size_t vocab_size,
            float temperature,
            int top_k,
            std::mt19937& rng )
        {
            if ( temperature <= 0.0f || top_k == 1 )
                return static_cast<int32_t>(
                    std::max_element( logits, logits + vocab_size ) - logits);

            float max_logit = *std::max_element( logits, logits + vocab_size );

            std::vector<float> probs( vocab_size );
            double sum = 0.0;

            for ( size_t i = 0; i < vocab_size; ++i )
            {
                float v = std::exp( (logits[ i ] - max_logit) / temperature );
                probs[ i ] = v;
                sum += v;
            }

            for ( size_t i = 0; i < vocab_size; ++i )
                probs[ i ] /= static_cast<float>( sum );

            if ( top_k > 0 && top_k < static_cast<int>( vocab_size ) )
            {
                std::vector<size_t> indices( vocab_size );
                std::iota( indices.begin(), indices.end(), 0 );
                std::partial_sort( indices.begin(), indices.begin() + top_k,
                    indices.end(),
                    [&]( size_t a, size_t b ) { return probs[ a ] > probs[ b ]; } );

                std::vector<float> filtered( vocab_size, 0.0f );
                double filtered_sum = 0.0;

                for ( int i = 0; i < top_k; ++i )
                {
                    filtered[ indices[ i ] ] = probs[ indices[ i ] ];
                    filtered_sum += probs[ indices[ i ] ];
                }

                for ( size_t i = 0; i < vocab_size; ++i )
                    probs[ i ] = filtered[ i ] / static_cast<float>( filtered_sum );
            }

            std::uniform_real_distribution<float> dist( 0.0f, 1.0f );
            float r = dist( rng );
            float cumsum = 0.0f;

            for ( size_t i = 0; i < vocab_size; ++i )
            {
                cumsum += probs[ i ];

                if ( r < cumsum )
                    return static_cast<int32_t>( i );
            }

            return static_cast<int32_t>( vocab_size - 1 );
        }

        // ====================================================================
        // Config helpers
        // ====================================================================

        static GptConfig configFromMetadata( const PretrainedMetadata& metadata )
        {
            GptConfig config(
                static_cast<dim_t>(metadata.embedding_dim),
                static_cast<dim_t>(metadata.num_layers) );

            config.withVocabSize( static_cast<dim_t>(metadata.vocab_size) )
                .withMaxSequenceLength( static_cast<dim_t>(metadata.max_seq_length) )
                .withNumHeads( static_cast<dim_t>(metadata.num_heads) )
                .withHiddenSize( static_cast<dim_t>(metadata.hidden_dim) )
                .withBias( metadata.use_bias );

            return config;
        }
    };
}
