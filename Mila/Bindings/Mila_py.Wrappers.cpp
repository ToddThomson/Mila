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

module Mila.Bindings;

import std;
import Mila;

namespace Mila::Bindings
{
    using namespace Mila::Data;
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Deployment;

    using LlamaCudaBf16 = LlamaModel<DeviceType::Cuda, TensorDataType::BF16>;
    using GemmaCudaBf16 = GemmaModel<DeviceType::Cuda, TensorDataType::BF16>;
    using QwenCudaBf16 = QwenModel<DeviceType::Cuda, TensorDataType::BF16>;

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

    // ---- Chat protocol ------------------------------------------------------

    namespace
    {
        /// The four roles a conversation has, named as every wire protocol names them.
        Conversation::Role roleFromString( const std::string& role )
        {
            if ( role == "system" )    return Conversation::Role::System;
            if ( role == "user" )      return Conversation::Role::User;
            if ( role == "assistant" ) return Conversation::Role::Assistant;
            if ( role == "tool" )      return Conversation::Role::Tool;

            throw std::runtime_error( std::format(
                "'{}' is not a conversation role. Expected system, user, assistant or tool.",
                role ) );
        }

        std::vector<Conversation::Turn> toConversationTurns( const std::vector<TurnInfo>& history )
        {
            std::vector<Conversation::Turn> turns;
            turns.reserve( history.size() );

            for ( const auto& turn : history )
            {
                Conversation::Turn converted;
                converted.role = roleFromString( turn.role );
                converted.content = turn.content;

                for ( const auto& call : turn.tool_calls )
                {
                    converted.tool_calls.push_back( { call.name, call.arguments } );
                }

                turns.push_back( std::move( converted ) );
            }

            return turns;
        }
    }

    std::string qwenFormatPrompt(
        const std::vector<TurnInfo>& history,
        bool enable_thinking,
        int reasoning_effort_scale,
        const std::string& tools_json )
    {
        const std::vector<Conversation::Turn> turns = toConversationTurns( history );

        // std::invalid_argument, which formatPrompt throws for an empty or badly ordered
        // history, reaches Python as ValueError rather than as the RuntimeError this surface
        // documents. Translated here so one refusal shape covers the whole call.
        try
        {
            return Qwen::formatPrompt(
                turns,
                enable_thinking,
                Qwen::reasoningEffortFromScale( reasoning_effort_scale ),
                Qwen::serializeToolSignatures( tools_json ) );
        }
        catch ( const std::invalid_argument& error )
        {
            throw std::runtime_error( error.what() );
        }
    }

    std::optional<ToolCallInfo> qwenParseToolCall( const std::string& response )
    {
        const auto call = Qwen::parseToolCall( response );

        if ( !call.has_value() )
        {
            return std::nullopt;
        }

        return ToolCallInfo{ call->name, call->arguments };
    }

    ProtocolTokens qwenProtocolTokens()
    {
        return ProtocolTokens{
            .turn_open           = std::string( Qwen::kTurnOpen ),
            .turn_close          = std::string( Qwen::kTurnClose ),
            .reasoning_open      = std::string( Qwen::kThinkOpen ),
            .reasoning_close     = std::string( Qwen::kThinkClose ),
            .tool_call_open      = std::string( Qwen::kToolCallOpen ),
            .tool_call_close     = std::string( Qwen::kToolCallClose ),
            .tool_response_open  = std::string( Qwen::kToolResponseOpen ),
            .tool_response_close = std::string( Qwen::kToolResponseClose ),
        };
    }

    std::string gemmaFormatPrompt(
        const std::vector<TurnInfo>& history,
        const std::string& tool_declarations,
        bool continue_open )
    {
        const std::vector<Conversation::Turn> turns = toConversationTurns( history );

        // std::invalid_argument reaches Python as ValueError rather than as the RuntimeError
        // this surface documents. Translated so one refusal shape covers the whole call.
        try
        {
            return Gemma::formatPrompt( turns, tool_declarations, continue_open );
        }
        catch ( const std::invalid_argument& error )
        {
            throw std::runtime_error( error.what() );
        }
    }

    std::string gemmaToolDeclarations( const std::string& tools_json )
    {
        return Gemma::serializeToolDeclarations( tools_json );
    }

    std::optional<ToolCallInfo> gemmaParseToolCall( const std::string& response )
    {
        const auto call = Gemma::parseToolCall( response );

        if ( !call.has_value() )
        {
            return std::nullopt;
        }

        return ToolCallInfo{ call->name, call->arguments };
    }

    std::string gemmaFormatToolCall( const std::string& name, const std::string& arguments )
    {
        return Gemma::formatToolCall( name, arguments );
    }

    std::string gemmaFormatToolResponse( const std::string& name, const std::string& result )
    {
        return Gemma::formatToolResponse( name, result );
    }

    std::string gemmaExtractAnswer( const std::string& text )
    {
        return Gemma::extractAnswer( text );
    }

    std::string gemmaStripControlTokens( const std::string& text )
    {
        return Gemma::stripControlTokens( text );
    }

    ProtocolTokens gemmaProtocolTokens()
    {
        return ProtocolTokens{
            .turn_open           = std::string( Gemma::kTurnOpen ),
            .turn_close          = std::string( Gemma::kTurnClose ),
            .reasoning_open      = std::string( Gemma::kChannelOpen ),
            .reasoning_close     = std::string( Gemma::kChannelClose ),
            .tool_call_open      = std::string( Gemma::kToolCallOpen ),
            .tool_call_close     = std::string( Gemma::kToolCallClose ),
            .tool_response_open  = std::string( Gemma::kToolResponseOpen ),
            .tool_response_close = std::string( Gemma::kToolResponseClose ),
        };
    }

    // ---- Distribution -------------------------------------------------------

    std::string defaultHubOwner()
    {
        return std::string( Mila::Distribution::kDefaultHubOwner );
    }

    namespace
    {
        /// The family as Python reads it, the spelling a manifest's architecture uses; empty for none this build loads.
        std::string familyName( ModelType family )
        {
            switch ( family )
            {
                case ModelType::Gemma: return "gemma";
                case ModelType::Llama: return "llama";
                case ModelType::Qwen: return "qwen";
                default: return {};
            }
        }

        StoredModelInfo describeStoredModel( const Mila::Distribution::StoredModel& model )
        {
            StoredModelInfo info;
            info.name = model.record.name;
            const Mila::Deployment::InstalledModel installed( model );
            const auto capabilities = installed.capabilities();

            info.architecture = model.record.architecture;
            info.family = familyName( installed.family() );
            info.variant = model.record.variant;
            info.weight_quantization = model.record.weight_quantization;
            info.instruct = capabilities.instruct;
            info.thinking = capabilities.thinking;
            info.tools = capabilities.tools;
            info.agentic = capabilities.agentic;
            info.mtp = installed.drafter().has_value();
            info.modalities = capabilities.modalities;
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
         * @brief An installed model a session can load, or a failure that names the alternatives.
         *
         * The store is the only source consulted. A name that is not installed is an error
         * rather than a download: pull and load are separate verbs, and a load that reached
         * the network would turn an ordinary call into a multi-gigabyte transfer.
         *
         * @param family The family the calling session loads; a model of another is refused, naming its own.
         */
        Mila::Deployment::InstalledModel requireInstalledModel( const std::string& name, ModelType family )
        {
            const auto installed = Mila::Deployment::InstalledModel::find( name );

            if ( !installed.has_value() )
            {
                std::string available;

                for ( const auto& model : Mila::Deployment::InstalledModel::list() )
                {
                    available += available.empty() ? "" : ", ";
                    available += model.name();
                }

                throw std::runtime_error( std::format(
                    "No model named '{}' is installed. Installed: {}",
                    name, available.empty() ? "nothing" : available ) );
            }

            // The library's refusal: missing files, or a family this build cannot deploy.
            Mila::Deployment::requirePlannable( *installed );

            if ( installed->family() != family )
            {
                throw std::runtime_error( std::format(
                    "'{}' is a {} model, which this session does not load. "
                    "Read ModelStore.locate(name).family and pick the matching session.",
                    installed->name(), familyName( installed->family() ) ) );
            }

            return *installed;
        }

        /**
         * @brief Settle the weight format a caller names for a load from a file, in the store's words.
         *
         * Only load() takes one: a store load takes the format the weights are stored in. A format the weights
         * cannot serve is refused by the library's own rule when the load plans.
         */
        void applyQuantizationName( DeploymentRequest& request, const std::string& name, const std::string& subject )
        {
            if ( name == "bf16" || name == "none" )
            {
                request.withFullPrecision();
            }
            else if ( name == "fp8" )
            {
                request.withFP8Quantization();
            }
            else if ( name == "fp4" )
            {
                request.withFP4Quantization();
            }
            else if ( name == "q4_0" )
            {
                request.withQ4_0Quantization();
            }
            else if ( name == "cb2-3" )
            {
                request.withPrecisionPlan();
            }
            else if ( name == "fp4-cb3" )
            {
                request.withFp4Codebook3Plan();
            }
            else
            {
                throw std::runtime_error( std::format(
                    "{}: '{}' is not a weight format. Expected bf16, fp8, fp4, q4_0, cb2-3 or fp4-cb3.{}",
                    subject, name, name == "fp32" ? " These sessions are BF16 instantiations." : "" ) );
            }
        }

        /**
         * @brief The deployment request a session load sends, before its quantization is applied.
         *
         * Headroom stays the library's zero (Deployment.md 12.3): what else shares the device is known
         * only to the program calling this binding. The KV cache is the caller's choice on every load,
         * because it is a deployment decision rather than a property of the weights; a model that cannot
         * serve it is refused by the library when the load plans.
         */
        DeploymentRequest sessionRequest(
            std::optional<std::int64_t> context_length, int device_index, const std::string& kv_cache,
            const std::string& subject )
        {
            DeploymentRequest request;
            request.withDevice( DeviceId{ DeviceType::Cuda, device_index } );

            if ( kv_cache == "fp8" )
            {
                request.withKvCacheCompression( KvCacheCompression::FP8 );
            }
            else if ( kv_cache != "bf16" )
            {
                throw std::runtime_error( std::format(
                    "{}: '{}' is not a KV cache format. Expected bf16 or fp8.", subject, kv_cache ) );
            }

            if ( context_length.has_value() )
            {
                request.withContextLength( static_cast<dim_t>( *context_length ) );
            }
            else
            {
                request.withAutomaticContextLength();
            }

            return request;
        }

        std::string formatGigabytes( std::size_t bytes )
        {
            return std::format( "{:.2f} GB", static_cast<double>( bytes ) / ( 1024.0 * 1024.0 * 1024.0 ) );
        }

        /**
         * @brief A refused deployment in this binding's words: what does not fit, against what, and what
         *        to try.
         *
         * The library names the reason and the numbers; the sentence belongs to whoever presents it
         * (Deployment.md section 6).
         */
        std::string describeRefusal(
            const std::string& subject, int device_index, const DeploymentRefusal& refusal )
        {
            const std::string free = formatGigabytes( refusal.reading().free_bytes );

            switch ( refusal.reason() )
            {
                case DeploymentRefusal::Reason::WeightsExceedDevice:
                    return std::format(
                        "'{}' does not fit on CUDA device {}: its weights need {} and the device has {} free, "
                        "so no context length fits.",
                        subject, device_index, formatGigabytes( refusal.footprint().device_parameter_bytes ), free );

                case DeploymentRefusal::Reason::FixedContextDoesNotFit:
                    return std::format(
                        "'{}' does not fit on CUDA device {} at context_length {}: it needs {} and the device "
                        "has {} free. context_length='auto' chooses the longest that fits.",
                        subject, device_index, refusal.contextLength(),
                        formatGigabytes( refusal.footprint().totalDeviceBytes() ), free );

                case DeploymentRefusal::Reason::NothingAboveTheFloorFits:
                    return std::format(
                        "'{}' does not fit on CUDA device {} at any context length: even {} needs {} and the "
                        "device has {} free.",
                        subject, device_index, refusal.contextLength(),
                        formatGigabytes( refusal.footprint().totalDeviceBytes() ), free );

                case DeploymentRefusal::Reason::DeviceDoesNotReportMemory:
                    return std::format(
                        "CUDA device {} does not report its free memory, so no context length can be chosen "
                        "for '{}'. Pass context_length as a number.",
                        device_index, subject );
            }

            return refusal.toString();
        }

        /// What a session samples with when a knob is not given: the model's settings and its template's thinking.
        struct SessionSampling
        {
            std::vector<Mila::Distribution::SamplingRecommendation> recommended;
            bool default_thinking{ false };
        };

        /**
         * @brief Whether a prompt rendered at the template's defaults has thinking on.
         *
         * The template's default rather than the manifest's thinking capability, because the prompt is what the
         * sampling has to match: a package published before capabilities existed declares none, and its template
         * still opens Qwen's reasoning. An architecture with no template here has no thinking switch to default on.
         */
        bool templateThinksByDefault( std::string_view architecture, std::string_view architecture_version )
        {
            std::unique_ptr<Conversation::Protocol> protocol;

            try
            {
                protocol = chatProtocolFor( architecture, architecture_version );
            }
            catch ( const std::invalid_argument& )
            {
                return false;
            }

            for ( const auto& variable : protocol->templateVariables() )
            {
                if ( variable.name == "enable_thinking" )
                {
                    return variable.default_value == "true";
                }
            }

            return false;
        }

        SessionSampling storedSampling( const Mila::Deployment::InstalledModel& model )
        {
            const auto& record = model.stored().record;

            // A base model has no template, so nothing renders it thinking.
            const bool thinking = model.capabilities().instruct
                && templateThinksByDefault( record.architecture, record.architecture_version );

            return { Mila::Distribution::recommendedSampling( record ), thinking };
        }

        /// A load from a file has no manifest, so its family's settings and template stand.
        SessionSampling familySessionSampling( std::string_view architecture )
        {
            return { Mila::Distribution::familySampling( architecture ), templateThinksByDefault( architecture, {} ) };
        }

        /// The caller's knobs over the recommended setting for the thinking asked, over the sampler's neutral values.
        SamplingParams resolveSampling(
            const SessionSampling& session, std::optional<bool> thinking,
            std::optional<float> temperature, std::optional<int> top_k, std::optional<float> top_p,
            std::optional<float> presence_penalty )
        {
            SamplingParams sampling;

            if ( const auto* recommended = Mila::Distribution::matchingSampling(
                     session.recommended, thinking.value_or( session.default_thinking ) ) )
            {
                sampling.temperature = recommended->temperature.value_or( sampling.temperature );
                sampling.top_k = recommended->top_k.value_or( sampling.top_k );
                sampling.top_p = recommended->top_p.value_or( sampling.top_p );
                sampling.presence_penalty = recommended->presence_penalty.value_or( sampling.presence_penalty );
            }

            sampling.temperature = temperature.value_or( sampling.temperature );
            sampling.top_k = top_k.value_or( sampling.top_k );
            sampling.top_p = top_p.value_or( sampling.top_p );
            sampling.presence_penalty = presence_penalty.value_or( sampling.presence_penalty );

            return sampling;
        }

        /// Plan the request and load its best plan, with a refusal reworded for a Python caller.
        template<typename TModel>
        std::unique_ptr<TModel> loadDeployment(
            const std::filesystem::path& weights, const DeploymentRequest& request,
            const std::string& subject, int device_index )
        {
            try
            {
                return TModel::load( weights, request );
            }
            catch ( const DeploymentRefusedError& error )
            {
                throw std::runtime_error( describeRefusal( subject, device_index, error.refusal() ) );
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

    SamplingSettingsInfo recommendedSampling( const std::string& name, bool thinking )
    {
        const auto installed = Mila::Deployment::InstalledModel::find( name );

        if ( !installed.has_value() )
        {
            throw std::runtime_error( std::format( "No model named '{}' is installed.", name ) );
        }

        const SamplingParams sampling = resolveSampling(
            { Mila::Distribution::recommendedSampling( installed->stored().record ), thinking },
            thinking, std::nullopt, std::nullopt, std::nullopt, std::nullopt );

        return { sampling.temperature, sampling.top_k, sampling.top_p, sampling.presence_penalty };
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
        info.unreadable_records = report.unreadable_records;

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
            // A repository without a manifest is not something this runtime can load.
            if ( !entry.hasManifest() )
            {
                continue;
            }

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

    std::shared_ptr<Tokenizer> Tokenizer::loadQwen( const std::string& path )
    {
        auto impl = std::make_unique<Impl>();
        impl->tokenizer = BpeTokenizer::loadQwen( std::filesystem::path( path ) );

        return std::shared_ptr<Tokenizer>( new Tokenizer( std::move( impl ) ) );
    }

    std::shared_ptr<Tokenizer> Tokenizer::fromStore( const std::string& name )
    {
        const auto installed = Mila::Deployment::InstalledModel::find( name );

        if ( !installed.has_value() )
        {
            throw std::runtime_error( std::format( "No model named '{}' is installed.", name ) );
        }

        auto impl = std::make_unique<Impl>();
        impl->tokenizer = installed->loadTokenizer();

        return std::shared_ptr<Tokenizer>( new Tokenizer( std::move( impl ) ) );
    }

    std::vector<std::int32_t> Tokenizer::encode( const std::string& text )
    {
        return impl_->tokenizer->encode( text );
    }

    std::string Tokenizer::decode( const std::vector<std::int32_t>& ids )
    {
        return impl_->tokenizer->decode( std::span<const std::int32_t>( ids ) );
    }

    std::string Tokenizer::tokenToString( std::int32_t token_id ) const
    {
        return impl_->tokenizer->tokenToString( token_id );
    }

    bool Tokenizer::isValidToken( std::int32_t token_id ) const
    {
        return impl_->tokenizer->isValidToken( token_id );
    }

    std::int64_t Tokenizer::vocabSize() const
    {
        return static_cast<std::int64_t>( impl_->tokenizer->getVocabSize() );
    }

    std::optional<std::int32_t> Tokenizer::bosTokenId() const
    {
        auto id = impl_->tokenizer->getBosTokenId();

        return id ? std::optional<std::int32_t>( static_cast<std::int32_t>( *id ) ) : std::nullopt;
    }

    std::optional<std::int32_t> Tokenizer::eosTokenId() const
    {
        auto id = impl_->tokenizer->getEosTokenId();

        return id ? std::optional<std::int32_t>( static_cast<std::int32_t>( *id ) ) : std::nullopt;
    }

    std::optional<std::int32_t> Tokenizer::padTokenId() const
    {
        auto id = impl_->tokenizer->getPadTokenId();

        return id ? std::optional<std::int32_t>( static_cast<std::int32_t>( *id ) ) : std::nullopt;
    }

    // ---- LlamaSession -------------------------------------------------------

    struct LlamaSession::Impl
    {
        std::unique_ptr<LlamaCudaBf16> model;
        SessionSampling sampling;
    };

    LlamaSession::LlamaSession( std::unique_ptr<Impl> impl ) : impl_( std::move( impl ) ) {}
    LlamaSession::~LlamaSession() = default;

    std::unique_ptr<LlamaSession> LlamaSession::load(
        const std::string& path, std::optional<std::int64_t> context_length, int device_index,
        const std::string& quantization, const std::string& kv_cache )
    {
        const std::filesystem::path weights( path );
        DeploymentRequest request = sessionRequest( context_length, device_index, kv_cache, "LlamaModel.load" );

        applyQuantizationName( request, quantization, "LlamaModel.load" );

        auto impl = std::make_unique<Impl>();
        impl->model = loadDeployment<LlamaCudaBf16>(
            weights, request, weights.filename().string(), device_index );
        impl->sampling = familySessionSampling( "llama" );

        return std::unique_ptr<LlamaSession>( new LlamaSession( std::move( impl ) ) );
    }

    std::unique_ptr<LlamaSession> LlamaSession::fromStore(
        const std::string& name, std::optional<std::int64_t> context_length, int device_index,
        const std::string& kv_cache )
    {
        const auto model = requireInstalledModel( name, ModelType::Llama );

        // Names no weight format, so the library loads the one the weights are stored in.
        const DeploymentRequest request = sessionRequest( context_length, device_index, kv_cache, name );

        auto impl = std::make_unique<Impl>();
        impl->model = loadDeployment<LlamaCudaBf16>(
            model.weightsPath(), request, model.name(), device_index );
        impl->sampling = storedSampling( model );

        return std::unique_ptr<LlamaSession>( new LlamaSession( std::move( impl ) ) );
    }

    std::string LlamaSession::generate(
        const std::vector<std::int32_t>& prompt_tokens,
        const std::function<void( std::int32_t )>& on_token,
        std::size_t max_new_tokens,
        std::optional<float> temperature, std::optional<int> top_k, std::optional<float> top_p,
        std::optional<float> presence_penalty, std::optional<bool> thinking,
        std::optional<std::uint64_t> seed,
        std::stop_token stop )
    {
        if ( seed )
        {
            impl_->model->seedSampler( *seed );
        }

        GenerateParams params;
        params.max_new_tokens = static_cast<int>( max_new_tokens );
        params.reuse_kv_prefix = !seed.has_value();
        params.sampling = resolveSampling(
            impl_->sampling, thinking, temperature, top_k, top_p, presence_penalty );

        const auto status = impl_->model->generate(
            prompt_tokens, on_token, params, std::move( stop ) );

        return std::string( to_string( status ) );
    }

    LlamaConfigInfo LlamaSession::getConfig() const
    {
        const auto& cfg = impl_->model->getConfig();

        return LlamaConfigInfo{
            .vocab_size          = static_cast<std::int64_t>( cfg.getVocabSize() ),
            .max_sequence_length = static_cast<std::int64_t>( cfg.getMaxSequenceLength() ),
            .model_dim           = static_cast<std::int64_t>( cfg.getModelDim() ),
            .num_layers          = static_cast<std::int64_t>( cfg.getNumLayers() ),
            .num_heads           = static_cast<std::int64_t>( cfg.getNumHeads() ),
            .num_kv_heads        = static_cast<std::int64_t>( cfg.getNumKVHeads() ),
            .hidden_dim          = static_cast<std::int64_t>( cfg.getHiddenDimension() ),
            .rope_theta          = static_cast<double>( cfg.getRoPETheta() ),
        };
    }

    std::int64_t LlamaSession::contextLength() const
    {
        return static_cast<std::int64_t>( impl_->model->getDeploymentPlan().contextLength() );
    }

    std::string LlamaSession::repr() const
    {
        return impl_->model->toString();
    }

    // ---- GemmaSession -------------------------------------------------------

    struct GemmaSession::Impl
    {
        std::unique_ptr<GemmaCudaBf16> model;
        SessionSampling sampling;
    };

    GemmaSession::GemmaSession( std::unique_ptr<Impl> impl ) : impl_( std::move( impl ) ) {}
    GemmaSession::~GemmaSession() = default;

    std::unique_ptr<GemmaSession> GemmaSession::load(
        const std::string& path, std::optional<std::int64_t> context_length, int device_index,
        const std::string& quantization, const std::string& kv_cache )
    {
        const std::filesystem::path weights( path );
        DeploymentRequest request = sessionRequest( context_length, device_index, kv_cache, "GemmaModel.load" );

        applyQuantizationName( request, quantization, "GemmaModel.load" );

        auto impl = std::make_unique<Impl>();
        impl->model = loadDeployment<GemmaCudaBf16>(
            weights, request, weights.filename().string(), device_index );
        impl->sampling = familySessionSampling( "gemma" );

        return std::unique_ptr<GemmaSession>( new GemmaSession( std::move( impl ) ) );
    }

    std::unique_ptr<GemmaSession> GemmaSession::fromStore(
        const std::string& name, std::optional<std::int64_t> context_length, int device_index,
        const std::string& kv_cache )
    {
        const auto model = requireInstalledModel( name, ModelType::Gemma );

        // Names no weight format, so the library loads the one the weights are stored in.
        const DeploymentRequest request = sessionRequest( context_length, device_index, kv_cache, name );

        auto impl = std::make_unique<Impl>();
        impl->model = loadDeployment<GemmaCudaBf16>(
            model.weightsPath(), request, model.name(), device_index );
        impl->sampling = storedSampling( model );

        return std::unique_ptr<GemmaSession>( new GemmaSession( std::move( impl ) ) );
    }

    std::string GemmaSession::generate(
        const std::vector<std::int32_t>& prompt_tokens,
        const std::function<void( std::int32_t )>& on_token,
        std::size_t max_new_tokens,
        std::optional<float> temperature, std::optional<int> top_k, std::optional<float> top_p,
        std::optional<float> presence_penalty, std::optional<bool> thinking,
        std::optional<std::uint64_t> seed,
        std::stop_token stop )
    {
        if ( seed )
        {
            impl_->model->seedSampler( *seed );
        }

        GenerateParams params;
        params.max_new_tokens = static_cast<int>( max_new_tokens );
        params.reuse_kv_prefix = !seed.has_value();
        params.sampling = resolveSampling(
            impl_->sampling, thinking, temperature, top_k, top_p, presence_penalty );

        const auto status = impl_->model->generate(
            prompt_tokens, on_token, params, std::move( stop ) );

        return std::string( to_string( status ) );
    }

    GemmaConfigInfo GemmaSession::getConfig() const
    {
        const auto& cfg = impl_->model->getNetworkConfig();

        return GemmaConfigInfo{
            .vocab_size              = static_cast<std::int64_t>( cfg.getVocabSize() ),
            .max_sequence_length     = static_cast<std::int64_t>( cfg.getMaxSequenceLength() ),
            .model_dim               = static_cast<std::int64_t>( cfg.getModelDim() ),
            .num_layers              = static_cast<std::int64_t>( cfg.getNumLayers() ),
            .num_heads               = static_cast<std::int64_t>( cfg.getNumHeads() ),
            .num_kv_heads            = static_cast<std::int64_t>( cfg.getNumKVHeads() ),
            .head_dim                = static_cast<std::int64_t>( cfg.getHeadDim() ),
            .global_head_dim         = static_cast<std::int64_t>( cfg.getGlobalHeadDim() ),
            .hidden_dim              = static_cast<std::int64_t>( cfg.getHiddenDimension() ),
            .window                  = static_cast<std::int64_t>( cfg.getWindow() ),
            .rope_theta_local        = static_cast<double>( cfg.getRoPEThetaLocal() ),
            .rope_theta_global       = static_cast<double>( cfg.getRoPEThetaGlobal() ),
            .final_logit_softcapping = static_cast<double>( cfg.getFinalLogitSoftcapping() ),
        };
    }

    std::int64_t GemmaSession::contextLength() const
    {
        return static_cast<std::int64_t>( impl_->model->getDeploymentPlan().contextLength() );
    }

    std::string GemmaSession::repr() const
    {
        return impl_->model->toString();
    }

    // ---- QwenSession --------------------------------------------------------

    struct QwenSession::Impl
    {
        std::unique_ptr<QwenCudaBf16> model;
        SessionSampling sampling;
    };

    QwenSession::QwenSession( std::unique_ptr<Impl> impl ) : impl_( std::move( impl ) ) {}
    QwenSession::~QwenSession() = default;

    std::unique_ptr<QwenSession> QwenSession::load(
        const std::string& path, std::optional<std::int64_t> context_length, int device_index,
        const std::string& quantization, const std::string& kv_cache )
    {
        const std::filesystem::path weights( path );
        DeploymentRequest request = sessionRequest( context_length, device_index, kv_cache, "QwenModel.load" );

        applyQuantizationName( request, quantization, "QwenModel.load" );

        auto impl = std::make_unique<Impl>();
        impl->model = loadDeployment<QwenCudaBf16>(
            weights, request, weights.filename().string(), device_index );
        impl->sampling = familySessionSampling( "qwen" );

        return std::unique_ptr<QwenSession>( new QwenSession( std::move( impl ) ) );
    }

    std::unique_ptr<QwenSession> QwenSession::fromStore(
        const std::string& name, std::optional<std::int64_t> context_length, int device_index,
        const std::string& kv_cache )
    {
        const auto model = requireInstalledModel( name, ModelType::Qwen );

        // Names no weight format, so the library loads the one the weights are stored in.
        const DeploymentRequest request = sessionRequest( context_length, device_index, kv_cache, name );

        auto impl = std::make_unique<Impl>();
        impl->model = loadDeployment<QwenCudaBf16>(
            model.weightsPath(), request, model.name(), device_index );
        impl->sampling = storedSampling( model );

        return std::unique_ptr<QwenSession>( new QwenSession( std::move( impl ) ) );
    }

    std::string QwenSession::generate(
        const std::vector<std::int32_t>& prompt_tokens,
        const std::function<void( std::int32_t )>& on_token,
        std::size_t max_new_tokens,
        std::optional<float> temperature, std::optional<int> top_k, std::optional<float> top_p,
        std::optional<float> presence_penalty, std::optional<bool> thinking,
        std::optional<std::uint64_t> seed,
        std::stop_token stop )
    {
        if ( seed )
        {
            impl_->model->seedSampler( *seed );
        }

        GenerateParams params;
        params.max_new_tokens = static_cast<int>( max_new_tokens );
        params.reuse_kv_prefix = !seed.has_value();
        params.sampling = resolveSampling(
            impl_->sampling, thinking, temperature, top_k, top_p, presence_penalty );

        const auto status = impl_->model->generate(
            prompt_tokens, on_token, params, std::move( stop ) );

        return std::string( to_string( status ) );
    }

    QwenConfigInfo QwenSession::getConfig() const
    {
        const auto& cfg = impl_->model->getNetworkConfig();

        return QwenConfigInfo{
            .vocab_size              = static_cast<std::int64_t>( cfg.getVocabSize() ),
            .max_sequence_length     = static_cast<std::int64_t>( cfg.getMaxSequenceLength() ),
            .model_dim               = static_cast<std::int64_t>( cfg.getModelDim() ),
            .num_layers              = static_cast<std::int64_t>( cfg.getNumLayers() ),
            .num_heads               = static_cast<std::int64_t>( cfg.getNumHeads() ),
            .num_kv_heads            = static_cast<std::int64_t>( cfg.getNumKVHeads() ),
            .head_dim                = static_cast<std::int64_t>( cfg.getHeadDim() ),
            .hidden_dim              = static_cast<std::int64_t>( cfg.getHiddenDimension() ),
            .rope_theta              = static_cast<double>( cfg.getRoPETheta() ),
            .partial_rotary_factor   = static_cast<double>( cfg.getPartialRotaryFactor() ),
            .attention_output_gate   = cfg.hasAttentionOutputGate(),
            .full_attention_interval = static_cast<std::int64_t>( cfg.getFullAttentionInterval() ),
            .linear_num_key_heads    = static_cast<std::int64_t>( cfg.getLinearNumKeyHeads() ),
            .linear_num_value_heads  = static_cast<std::int64_t>( cfg.getLinearNumValueHeads() ),
            .linear_head_dim         = static_cast<std::int64_t>( cfg.getLinearHeadDim() ),
            .linear_conv_kernel_dim  = static_cast<std::int64_t>( cfg.getLinearConvKernelDim() ),
        };
    }

    std::int64_t QwenSession::contextLength() const
    {
        return static_cast<std::int64_t>( impl_->model->getDeploymentPlan().contextLength() );
    }

    std::string QwenSession::repr() const
    {
        return impl_->model->toString();
    }
}
