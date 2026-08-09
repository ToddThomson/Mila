/**
 * @file Mila_py.Wrappers.cpp
 * @brief Implementation unit for Mila.Bindings: defines the PIMPL Impl structs
 *        and all wrapper member bodies.
 *
 * This is a module IMPLEMENTATION unit (`module Mila.Bindings;`, no export), so
 * `import Mila;` is safe here (the C2079 regression only affects ordinary,
 * non-module .cpp TUs). Because implementation-unit definitions are not
 * reachable to importers, the Impl structs and the Mila types they hold stay
 * invisible to Mila_py.cpp. See [[feedback-build-in-vs]].
 */

module;

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

module Mila.Bindings;

import Mila;

namespace Mila::Bindings
{
    using namespace Mila::Data;
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    using LlamaCudaBf16 = LlamaModel<DeviceType::Cuda, TensorDataType::BF16>;
    using GemmaCudaBf16 = GemmaModel<DeviceType::Cuda, TensorDataType::BF16>;

    void initialize( const std::string& log_level )
    {
        static const std::unordered_map<std::string, Mila::Logging::LogLevel> level_map = {
            { "trace",   Mila::Logging::LogLevel::Trace   },
            { "info",    Mila::Logging::LogLevel::Info    },
            { "warning", Mila::Logging::LogLevel::Warning },
            { "error",   Mila::Logging::LogLevel::Error   },
        };

        const auto it = level_map.find( log_level );
        const auto resolved = (it != level_map.end()) ? it->second : Mila::Logging::LogLevel::Warning;

        auto sink = std::make_shared<Mila::Logging::ConsoleSink>( resolved );
        Mila::initialize( 0, std::move( sink ) );
    }

    // ---- Distribution -------------------------------------------------------

    std::string defaultHubOwner()
    {
        return std::string( Mila::Distribution::kDefaultHubOwner );
    }

    namespace
    {
        StoredModelInfo describeStoredModel( const Mila::Distribution::StoredModel& model )
        {
            StoredModelInfo info;
            info.name = model.record.name;
            info.architecture = model.record.architecture;
            info.variant = model.record.variant;
            info.weight_quantization = model.record.weight_quantization;
            info.instruct = model.record.instruct;
            info.base_model = model.record.base_model;
            info.license = model.record.license;
            info.hub = model.record.hub;
            info.owner = model.record.owner;
            info.repository = model.record.repository;
            info.revision = model.record.revision;
            info.installed_at = model.record.installed_at;
            info.weights_path = model.weights_path.string();
            info.tokenizer_path = model.tokenizer_path.string();
            info.complete = model.complete;
            info.bytes_on_disk = model.bytes_on_disk;

            return info;
        }

        /**
         * @brief The record for an installed model, or a failure that names the alternatives.
         *
         * The store is the only source consulted. A name that is not installed is an error
         * rather than a download: pull and load are separate verbs, and a load that reached
         * the network would turn an ordinary call into a multi-gigabyte transfer.
         */
        Mila::Distribution::StoredModel requireInstalledModel( const std::string& name )
        {
            Mila::Distribution::ModelStore store;

            const auto installed = store.locate( name );

            if ( installed.has_value() )
            {
                return *installed;
            }

            // locate() refuses a record whose blobs are gone, so a name that is listed and
            // still not loadable needs its own message -- otherwise this would report a
            // model as absent while the store's own listing shows it.
            std::string available;

            for ( const auto& model : store.list() )
            {
                if ( model.record.name == name && !model.complete )
                {
                    throw std::runtime_error( std::format(
                        "'{}' is installed but its files are missing, so it cannot be loaded.",
                        name ) );
                }

                available += available.empty() ? "" : ", ";
                available += model.record.name;
            }

            throw std::runtime_error( std::format(
                "No model named '{}' is installed. Installed: {}",
                name, available.empty() ? "nothing" : available ) );
        }

        /**
         * @brief Settle a model config's quantization axes from a variant name.
         *
         * One mapping for both entry points, so "fp4" means the same thing whether it came
         * from a store record or from a caller. The presets are used rather than the
         * individual setters, matching what the chat harness loads.
         */
        template<typename TModelConfig>
        void applyQuantizationVariant(
            TModelConfig& model_config, const std::string& variant, const std::string& subject )
        {
            if ( variant == "fp4" )
            {
                model_config.withFP4Quantization();
            }
            else if ( variant == "fp8" )
            {
                model_config.withFP8Quantization();
            }
            else if ( variant == "bf16" || variant == "none" )
            {
                model_config.withFullPrecision();
            }
            else
            {
                throw std::runtime_error( std::format(
                    "{}: '{}' is not a variant this binding can load. Expected bf16, fp8 or fp4."
                    "{}",
                    subject, variant,
                    variant == "fp32" ? " These sessions are BF16 instantiations." : "" ) );
            }
        }

        /// The record's architecture, checked against the session class loading it.
        void requireArchitecture(
            const Mila::Distribution::StoredModel& model, const std::string& expected )
        {
            if ( model.record.architecture != expected )
            {
                throw std::runtime_error( std::format(
                    "'{}' has architecture '{}', which this session does not load. "
                    "Read ModelStore.locate(name).architecture and pick the matching session.",
                    model.record.name,
                    model.record.architecture.empty() ? "unknown" : model.record.architecture ) );
            }
        }

        /**
         * @brief An IHttpTransport backed by one caller-supplied callable.
         *
         * Everything above this -- the URL, the redirect decision, the token rule, the
         * manifest, the digest, the record -- stays in the library. The host language moves
         * bytes and nothing else, and they reach the store's sink directly, so they are hashed
         * once as they arrive rather than staged and re-read.
         *
         * The delegate is held by value: a copy of a std::function is cheap next to a transfer,
         * and it removes any question of whether the caller's callable outlives the pull.
         */
        class DelegateHttpTransport : public Mila::Distribution::IHttpTransport
        {
        public:

            explicit DelegateHttpTransport( HttpFetchDelegate perform )
                : perform_( std::move( perform ) )
            {}

            std::string name() const override { return "delegate"; }

            Mila::Distribution::HttpResponse fetch(
                const Mila::Distribution::HttpFetch& request,
                const Mila::Distribution::SinkCallback& sink,
                const Mila::Distribution::HeadersCallback& on_headers ) const override
            {
                std::vector<HttpHeaderInfo> headers;

                for ( const auto& header : request.headers )
                {
                    headers.push_back( { header.name, header.value } );
                }

                const HttpResponseInfo answer = perform_( request.url, headers, sink );

                // The delegate reports what it saw only on return, so a caller watching
                // progress learns the total after the body rather than before it. Reporting
                // it here at least gets it into the result.
                if ( on_headers )
                {
                    on_headers( answer.http_code, answer.content_length );
                }

                Mila::Distribution::HttpResponse response;
                response.http_code = answer.http_code;
                response.location = answer.location;
                response.content_length = answer.content_length;
                response.transport_failed = answer.transport_failed;
                response.message = answer.message;

                return response;
            }

        private:

            HttpFetchDelegate perform_;
        };

        /// The caller's transport, or this build's own when they supplied none.
        std::shared_ptr<const Mila::Distribution::IHttpTransport> transportFor(
            const HttpFetchDelegate& delegate )
        {
            if ( !delegate )
            {
                return Mila::Distribution::makeDefaultHttpTransport();
            }

            return std::make_shared<const DelegateHttpTransport>( delegate );
        }
    }

    struct ModelStoreHandle::Impl
    {
        explicit Impl( std::filesystem::path root ) : store( std::move( root ) )
        {}

        Mila::Distribution::ModelStore store;
    };

    // The root is resolved here rather than left to ModelStore's default argument, so the
    // empty-string case costs one resolveStoreRoot() rather than one discarded and one kept.
    ModelStoreHandle::ModelStoreHandle( const std::string& root )
        : impl_( std::make_unique<Impl>( root.empty()
            ? Mila::Distribution::resolveStoreRoot()
            : std::filesystem::path( root ) ) )
    {}

    ModelStoreHandle::~ModelStoreHandle() = default;

    std::string ModelStoreHandle::root() const
    {
        return impl_->store.root().string();
    }

    std::vector<StoredModelInfo> ModelStoreHandle::list() const
    {
        std::vector<StoredModelInfo> models;

        for ( const auto& model : impl_->store.list() )
        {
            models.push_back( describeStoredModel( model ) );
        }

        return models;
    }

    std::optional<StoredModelInfo> ModelStoreHandle::locate( const std::string& name ) const
    {
        const auto model = impl_->store.locate( name );

        if ( !model.has_value() )
        {
            return std::nullopt;
        }

        return describeStoredModel( *model );
    }

    RemovalReportInfo ModelStoreHandle::remove( const std::string& name )
    {
        const auto report = impl_->store.remove( name );

        RemovalReportInfo info;
        info.records_removed = report.records_removed;
        info.blobs_removed = report.blobs_removed;
        info.files_removed = report.files_removed;
        info.bytes_reclaimed = report.bytes_reclaimed;
        info.retained = report.retained;

        return info;
    }

    StoreUsageInfo ModelStoreHandle::usage() const
    {
        const auto totals = impl_->store.usage();

        StoreUsageInfo info;
        info.blob_bytes = totals.blob_bytes;
        info.reclaimable_bytes = totals.reclaimable_bytes;
        info.partial_bytes = totals.partial_bytes;
        info.model_count = totals.model_count;
        info.blob_count = totals.blob_count;

        return info;
    }

    StoredModelInfo ModelStoreHandle::install(
        const std::string& package_directory,
        const std::string& name,
        bool replace,
        bool move_files )
    {
        const auto package = Mila::Distribution::ModelPackage::open(
            std::filesystem::path( package_directory ) );

        Mila::Distribution::InstallOptions options;
        options.name = name;
        options.replace = replace;
        options.move_files = move_files;

        return describeStoredModel( impl_->store.install( package, options ) );
    }

    StoredModelInfo ModelStoreHandle::pull(
        const std::string& name,
        const std::string& owner,
        const HttpFetchDelegate& transport )
    {
        const Mila::Distribution::HuggingFaceHub hub( transportFor( transport ) );

        Mila::Distribution::ModelResolver resolver( impl_->store, hub );

        return describeStoredModel( resolver.pull( name, owner ) );
    }

    std::vector<HubModelInfo> ModelStoreHandle::listHubModels(
        const std::string& owner,
        const HttpFetchDelegate& transport ) const
    {
        const Mila::Distribution::HuggingFaceHub hub( transportFor( transport ) );

        std::vector<HubModelInfo> models;

        for ( const auto& entry : hub.listModels( owner ) )
        {
            HubModelInfo model;
            model.owner = entry.owner;
            model.repository = entry.repository;
            model.gated = entry.gated;
            model.revision = entry.revision;
            model.last_modified = entry.last_modified;
            model.library = entry.library;
            model.tags = entry.tags;
            model.files = entry.files;

            models.push_back( std::move( model ) );
        }

        return models;
    }

    // ---- Tokenizer ----------------------------------------------------------

    struct Tokenizer::Impl
    {
        std::shared_ptr<BpeTokenizer> tokenizer;
    };

    Tokenizer::Tokenizer( std::unique_ptr<Impl> impl ) : impl_( std::move( impl ) ) {}
    Tokenizer::~Tokenizer() = default;

    std::shared_ptr<Tokenizer> Tokenizer::loadLlama32( const std::string& path )
    {
        auto impl = std::make_unique<Impl>();
        impl->tokenizer = BpeTokenizer::loadLlama32( std::filesystem::path( path ) );

        return std::shared_ptr<Tokenizer>( new Tokenizer( std::move( impl ) ) );
    }

    std::shared_ptr<Tokenizer> Tokenizer::loadGemma( const std::string& path )
    {
        auto impl = std::make_unique<Impl>();
        impl->tokenizer = BpeTokenizer::loadGemma( std::filesystem::path( path ) );

        return std::shared_ptr<Tokenizer>( new Tokenizer( std::move( impl ) ) );
    }

    std::shared_ptr<Tokenizer> Tokenizer::fromStore( const std::string& name )
    {
        const auto model = requireInstalledModel( name );

        const std::string& architecture = model.record.architecture;
        const std::string path = model.tokenizer_path.string();

        if ( architecture == "gemma" )
        {
            return loadGemma( path );
        }

        if ( architecture == "llama" )
        {
            return loadLlama32( path );
        }

        throw std::runtime_error( std::format(
            "'{}' has architecture '{}', which has no tokenizer in this binding.",
            name, architecture.empty() ? "unknown" : architecture ) );
    }

    std::vector<int32_t> Tokenizer::encode( const std::string& text )
    {
        return impl_->tokenizer->encode( text );
    }

    std::string Tokenizer::decode( const std::vector<int32_t>& ids )
    {
        return impl_->tokenizer->decode( std::span<const int32_t>( ids ) );
    }

    std::string Tokenizer::tokenToString( int32_t token_id ) const
    {
        return impl_->tokenizer->tokenToString( token_id );
    }

    bool Tokenizer::isValidToken( int32_t token_id ) const
    {
        return impl_->tokenizer->isValidToken( token_id );
    }

    int64_t Tokenizer::vocabSize() const
    {
        return static_cast<int64_t>( impl_->tokenizer->getVocabSize() );
    }

    std::optional<int32_t> Tokenizer::bosTokenId() const
    {
        auto id = impl_->tokenizer->getBosTokenId();

        return id ? std::optional<int32_t>( static_cast<int32_t>( *id ) ) : std::nullopt;
    }

    std::optional<int32_t> Tokenizer::eosTokenId() const
    {
        auto id = impl_->tokenizer->getEosTokenId();

        return id ? std::optional<int32_t>( static_cast<int32_t>( *id ) ) : std::nullopt;
    }

    std::optional<int32_t> Tokenizer::padTokenId() const
    {
        auto id = impl_->tokenizer->getPadTokenId();

        return id ? std::optional<int32_t>( static_cast<int32_t>( *id ) ) : std::nullopt;
    }

    // ---- LlamaSession -------------------------------------------------------

    struct LlamaSession::Impl
    {
        std::unique_ptr<LlamaCudaBf16> model;
    };

    LlamaSession::LlamaSession( std::unique_ptr<Impl> impl ) : impl_( std::move( impl ) ) {}
    LlamaSession::~LlamaSession() = default;

    std::unique_ptr<LlamaSession> LlamaSession::fromPretrained(
        const std::string& path, int64_t context_length, int device_index,
        const std::string& quantization )
    {
        DeviceId device_id{ DeviceType::Cuda, device_index };
        LlamaModelConfig model_config( static_cast<dim_t>( context_length ) );

        applyQuantizationVariant( model_config, quantization, "LlamaModel.from_pretrained" );

        auto impl = std::make_unique<Impl>();
        impl->model = LlamaCudaBf16::fromPretrained(
            std::filesystem::path( path ), model_config, device_id );

        return std::unique_ptr<LlamaSession>( new LlamaSession( std::move( impl ) ) );
    }

    std::unique_ptr<LlamaSession> LlamaSession::fromStore(
        const std::string& name, int64_t context_length, int device_index )
    {
        const auto model = requireInstalledModel( name );

        requireArchitecture( model, "llama" );

        DeviceId device_id{ DeviceType::Cuda, device_index };
        LlamaModelConfig model_config( static_cast<dim_t>( context_length ) );

        applyQuantizationVariant( model_config, model.record.variant, name );

        auto impl = std::make_unique<Impl>();
        impl->model = LlamaCudaBf16::fromPretrained(
            model.weights_path, model_config, device_id );

        return std::unique_ptr<LlamaSession>( new LlamaSession( std::move( impl ) ) );
    }

    std::vector<int32_t> LlamaSession::generate(
        const std::vector<int32_t>& prompt_tokens,
        std::size_t max_new_tokens, float temperature, int top_k, float top_p )
    {
        GenerateParams params;
        params.max_new_tokens = static_cast<int>( max_new_tokens );
        params.sampling.temperature = temperature;
        params.sampling.top_k = top_k;
        params.sampling.top_p = top_p;

        // Blocking convenience over the streaming-only core primitive: collect the
        // generated tokens onto the prompt so the caller receives prompt + completion.
        std::vector<int32_t> output( prompt_tokens.begin(), prompt_tokens.end() );
        // Finish reason is not part of this blocking convenience shape; the caller
        // infers completion from the returned token list.
        (void)impl_->model->generate(
            prompt_tokens,
            [&output]( int32_t token ) { output.push_back( token ); },
            params );

        return output;
    }

    void LlamaSession::generateStreaming(
        const std::vector<int32_t>& prompt_tokens,
        const std::function<void( int32_t )>& on_token,
        std::size_t max_new_tokens, float temperature, int top_k, float top_p,
        std::stop_token stop )
    {
        GenerateParams params;
        params.max_new_tokens = static_cast<int>( max_new_tokens );
        params.sampling.temperature = temperature;
        params.sampling.top_k = top_k;
        params.sampling.top_p = top_p;
        (void)impl_->model->generate(
            prompt_tokens, on_token, params, std::move( stop ) );
    }

    LlamaConfigInfo LlamaSession::getConfig() const
    {
        const auto& cfg = impl_->model->getConfig();

        return LlamaConfigInfo{
            .vocab_size          = static_cast<int64_t>( cfg.getVocabSize() ),
            .max_sequence_length = static_cast<int64_t>( cfg.getMaxSequenceLength() ),
            .model_dim           = static_cast<int64_t>( cfg.getModelDim() ),
            .num_layers          = static_cast<int64_t>( cfg.getNumLayers() ),
            .num_heads           = static_cast<int64_t>( cfg.getNumHeads() ),
            .num_kv_heads        = static_cast<int64_t>( cfg.getNumKVHeads() ),
            .hidden_dim          = static_cast<int64_t>( cfg.getHiddenDimension() ),
            .rope_theta          = static_cast<double>( cfg.getRoPETheta() ),
        };
    }

    std::string LlamaSession::repr() const
    {
        return impl_->model->toString();
    }

    // ---- GemmaSession -------------------------------------------------------

    struct GemmaSession::Impl
    {
        std::unique_ptr<GemmaCudaBf16> model;
    };

    GemmaSession::GemmaSession( std::unique_ptr<Impl> impl ) : impl_( std::move( impl ) ) {}
    GemmaSession::~GemmaSession() = default;

    std::unique_ptr<GemmaSession> GemmaSession::fromPretrained(
        const std::string& path, int64_t context_length, int device_index,
        const std::string& quantization )
    {
        DeviceId device_id{ DeviceType::Cuda, device_index };
        GemmaModelConfig model_config( static_cast<dim_t>( context_length ) );

        applyQuantizationVariant( model_config, quantization, "GemmaModel.from_pretrained" );

        auto impl = std::make_unique<Impl>();
        impl->model = GemmaCudaBf16::fromPretrained(
            std::filesystem::path( path ), model_config, device_id );

        return std::unique_ptr<GemmaSession>( new GemmaSession( std::move( impl ) ) );
    }

    std::unique_ptr<GemmaSession> GemmaSession::fromStore(
        const std::string& name, int64_t context_length, int device_index )
    {
        const auto model = requireInstalledModel( name );

        requireArchitecture( model, "gemma" );

        DeviceId device_id{ DeviceType::Cuda, device_index };
        GemmaModelConfig model_config( static_cast<dim_t>( context_length ) );

        applyQuantizationVariant( model_config, model.record.variant, name );

        auto impl = std::make_unique<Impl>();
        impl->model = GemmaCudaBf16::fromPretrained(
            model.weights_path, model_config, device_id );

        return std::unique_ptr<GemmaSession>( new GemmaSession( std::move( impl ) ) );
    }

    std::vector<int32_t> GemmaSession::generate(
        const std::vector<int32_t>& prompt_tokens,
        std::size_t max_new_tokens, float temperature, int top_k, float top_p )
    {
        GenerateParams params;
        params.max_new_tokens = static_cast<int>( max_new_tokens );
        params.sampling.temperature = temperature;
        params.sampling.top_k = top_k;
        params.sampling.top_p = top_p;

        // Blocking convenience over the streaming-only core primitive: collect the
        // generated tokens onto the prompt so the caller receives prompt + completion.
        std::vector<int32_t> output( prompt_tokens.begin(), prompt_tokens.end() );
        // Finish reason is not part of this blocking convenience shape; the caller
        // infers completion from the returned token list.
        (void)impl_->model->generate(
            prompt_tokens,
            [&output]( int32_t token ) { output.push_back( token ); },
            params );

        return output;
    }

    void GemmaSession::generateStreaming(
        const std::vector<int32_t>& prompt_tokens,
        const std::function<void( int32_t )>& on_token,
        std::size_t max_new_tokens, float temperature, int top_k, float top_p,
        std::stop_token stop )
    {
        GenerateParams params;
        params.max_new_tokens = static_cast<int>( max_new_tokens );
        params.sampling.temperature = temperature;
        params.sampling.top_k = top_k;
        params.sampling.top_p = top_p;
        (void)impl_->model->generate(
            prompt_tokens, on_token, params, std::move( stop ) );
    }

    GemmaConfigInfo GemmaSession::getConfig() const
    {
        const auto& cfg = impl_->model->getNetworkConfig();

        return GemmaConfigInfo{
            .vocab_size              = static_cast<int64_t>( cfg.getVocabSize() ),
            .max_sequence_length     = static_cast<int64_t>( cfg.getMaxSequenceLength() ),
            .model_dim               = static_cast<int64_t>( cfg.getModelDim() ),
            .num_layers              = static_cast<int64_t>( cfg.getNumLayers() ),
            .num_heads               = static_cast<int64_t>( cfg.getNumHeads() ),
            .num_kv_heads            = static_cast<int64_t>( cfg.getNumKVHeads() ),
            .head_dim                = static_cast<int64_t>( cfg.getHeadDim() ),
            .global_head_dim         = static_cast<int64_t>( cfg.getGlobalHeadDim() ),
            .hidden_dim              = static_cast<int64_t>( cfg.getHiddenDimension() ),
            .window                  = static_cast<int64_t>( cfg.getWindow() ),
            .rope_theta_local        = static_cast<double>( cfg.getRoPEThetaLocal() ),
            .rope_theta_global       = static_cast<double>( cfg.getRoPEThetaGlobal() ),
            .final_logit_softcapping = static_cast<double>( cfg.getFinalLogitSoftcapping() ),
        };
    }

    std::string GemmaSession::repr() const
    {
        return impl_->model->toString();
    }
}
