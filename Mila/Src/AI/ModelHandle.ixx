/**
 * @file ModelHandle.ixx
 * @brief An opened model, whatever its family: the model, its tokenizer, its chat template and what it can do.
 *
 * The one place a store name becomes a running model. See Specifications/ModelHandle.md.
 */

export module AI.ModelHandle;

import std;
import Compute.DeviceType;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.LanguageModel;
import Dnn.PromptPrefixReuse;
import Dnn.Models.Conversation.Protocol;
import Dnn.Models.Architectures;
import Dnn.Models.ChatProtocolForArchitecture;
import Data.BpeTokenizer;
import Deployment.DeploymentRequest;
import Deployment.DeploymentPlan;
import Deployment.DeploymentPlans;
import Deployment.DeploymentRefusal;
import Deployment.InstalledModel;
import Deployment.InstalledModelPlanner;
import Distribution.ModelCapabilities;
import Distribution.ModelStore;
import Dnn.WeightQuantization;
import AI.OpenRefusal;
import AI.OpenRefusedError;

namespace Mila::AI
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Deployment;
    using namespace Mila::Distribution;

    /**
     * @brief A model opened by its store name, with the family erased and device and precision kept.
     *
     * Holds what a model is -- the network, its tokenizer, its chat template, its capabilities, the plan it was
     * loaded on and what the store recorded -- and no conversation. Generation is the `LanguageModel` base's one
     * virtual call, so nothing here sits between a token and the network.
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class ModelHandle
    {
    public:

        using LanguageModelType = LanguageModel<TDeviceType, TPrecision>;

        /**
         * @brief Whether a name would open, without loading or reading anything but its record: the reason it would
         *        not, or nothing when it would.
         *
         * A base model opens: it has no protocol. Whether it fits the device is the planner's answer, not this one.
         */
        [[nodiscard]] static std::optional<OpenRefusal> check(
            std::string_view name, const DeploymentRequest& request = {}, const ModelStore& store = ModelStore{} )
        {
            const std::optional<InstalledModel> installed = InstalledModel::find( name, store );

            if ( !installed )
            {
                return OpenRefusal{ OpenRefusal::Reason::NotInstalled, std::string( name ),
                    std::format( "no model named '{}' is installed", name ) };
            }

            return check( *installed, request );
        }

        /// @copydoc check
        [[nodiscard]] static std::optional<OpenRefusal> check( const InstalledModel& installed, const DeploymentRequest& request )
        {
            const std::string& name = installed.name();
            const auto& record = installed.stored().record;

            if ( !installed.isComplete() )
            {
                return OpenRefusal{ OpenRefusal::Reason::FilesMissing, name,
                    std::format( "'{}' is installed but its files are missing, so it cannot be loaded", name ) };
            }

            if ( !Architectures::contains( installed.architecture() ) )
            {
                return OpenRefusal{ OpenRefusal::Reason::ArchitectureNotDeployed, name,
                    std::format( "'{}' has architecture '{}', which this build of Mila cannot deploy; it deploys {}",
                        name, installed.architecture(), Architectures::names() ) };
            }

            if ( installed.capabilities().instruct )
            {
                try
                {
                    (void)chatProtocolFor( installed.architecture(), record.architecture_version );
                }
                catch ( const std::invalid_argument& error )
                {
                    // Family-blind: an architecture with one template takes an empty version as that one, so an
                    // empty version refused is one whose record does not say which of several it is.
                    if ( record.architecture_version.empty() )
                    {
                        return OpenRefusal{ OpenRefusal::Reason::ChatTemplateNotRecorded, name, std::format(
                            "'{}' does not record which version of '{}' it is, which decides the chat template it was "
                            "trained with: {}", name, installed.architecture(), error.what() ) };
                    }

                    return OpenRefusal{ OpenRefusal::Reason::NoChatTemplate, name,
                        std::format( "'{}' was trained with a chat template this build does not have: {}", name, error.what() ) };
                }
            }

            const std::optional<WeightQuantization> stored = installed.storedWeightQuantization();

            if ( !stored )
            {
                return OpenRefusal{ OpenRefusal::Reason::WeightFormatUnknown, name, std::format(
                    "'{}' stores its weights as '{}', which this build of Mila does not read", name, record.weight_quantization ) };
            }

            if ( request.isWeightQuantizationNamed() && request.getWeightQuantization() != *stored )
            {
                // The load's own rule, asked here so it refuses before anything is released.
                const std::string_view scheme = record.weight_quantization == "none"
                    ? std::string_view{} : std::string_view( record.weight_quantization );

                try
                {
                    requireStoredQuantizationMatches(
                        "ModelHandle::check", installed.weightsPath().string(), scheme, request.getWeightQuantization() );
                }
                catch ( const std::runtime_error& error )
                {
                    return OpenRefusal{ OpenRefusal::Reason::WeightFormatNotDerivable, name, error.what() };
                }
            }

            return std::nullopt;
        }

        /**
         * @brief Open an installed model on the best plan for the request.
         *
         * Never installs. The tokenizer and the chat template are read before the weights, so a model that cannot
         * be conversed with is refused before its load.
         *
         * @throws OpenRefusedError       for every refusal check gives, before anything is read.
         * @throws DeploymentRefusedError when nothing fits the device, carrying the refusal.
         */
        static ModelHandle open(
            std::string_view name, const DeploymentRequest& request, const ModelStore& store = ModelStore{} )
        {
            const InstalledModel installed = checkedNamed( name, request, store );

            // A base model has no template to render, so it has no protocol (ModelHandle.md 3.5).
            std::unique_ptr<Conversation::Protocol> protocol = installed.capabilities().instruct
                ? chatProtocolFor( installed.architecture(), installed.stored().record.architecture_version )
                : nullptr;

            std::shared_ptr<Data::BpeTokenizer> tokenizer = installed.loadTokenizer();

            std::unique_ptr<LanguageModelType> model;
            std::optional<DeploymentPlan> plan;
            PromptPrefixReuse prefix_reuse{};

            // The one place the concrete type is known, so what only it can answer is read here.
            Architectures::visit( installed.architecture(), [&]( auto named )
                {
                    using Architecture = typename decltype( named )::type;

                    auto typed = Architecture::template Model<TDeviceType, TPrecision>::load(
                        installed.weightsPath(), request );

                    plan.emplace( typed->getDeploymentPlan() );
                    prefix_reuse = typed->promptPrefixReuse();
                    model = std::move( typed );
                } );

            return ModelHandle( installed, std::move( model ), std::move( tokenizer ), std::move( protocol ),
                std::move( *plan ), prefix_reuse );
        }

        /**
         * @brief What open would load, without loading: the plans for the request, or why none fits.
         *
         * @throws OpenRefusedError as open, before any load.
         */
        static std::expected<DeploymentPlans, DeploymentRefusal> planFor(
            std::string_view name, const DeploymentRequest& request, const ModelStore& store = ModelStore{} )
        {
            return planDeployment<TDeviceType, TPrecision>( checkedNamed( name, request, store ), request );
        }

        ModelHandle( ModelHandle&& ) noexcept = default;
        ModelHandle& operator=( ModelHandle&& ) noexcept = default;

        /// The network behind its base: generate, observe, save.
        [[nodiscard]] LanguageModelType& model() noexcept
        {
            return *model_;
        }

        /// The typed model and its components, or null when the model is not a TModel.
        template<typename TModel>
        [[nodiscard]] TModel* as() noexcept
        {
            return dynamic_cast<TModel*>( model_.get() );
        }

        [[nodiscard]] Data::BpeTokenizer& tokenizer() const noexcept
        {
            return *tokenizer_;
        }

        /// The chat template the model was trained on; null for a base model.
        [[nodiscard]] const Conversation::Protocol* protocol() const noexcept
        {
            return protocol_.get();
        }

        /// What the package says the model can do, with only the modalities this build runs.
        [[nodiscard]] const ModelCapabilities& capabilities() const noexcept
        {
            return capabilities_;
        }

        /// Where a later prompt may resume on the cache: Mila's implementation of the architecture decides it.
        [[nodiscard]] PromptPrefixReuse promptPrefixReuse() const noexcept
        {
            return prefix_reuse_;
        }

        /// The plan the model was loaded on: what it runs, not a second derivation of it.
        [[nodiscard]] const DeploymentPlan& plan() const noexcept
        {
            return plan_;
        }

        /// What the store recorded: name, lineage, licence, files.
        [[nodiscard]] const InstalledModel& installed() const noexcept
        {
            return installed_;
        }

        /// The name as the store spells it.
        [[nodiscard]] const std::string& name() const noexcept
        {
            return installed_.name();
        }

    private:

        ModelHandle(
            InstalledModel installed,
            std::unique_ptr<LanguageModelType> model,
            std::shared_ptr<Data::BpeTokenizer> tokenizer,
            std::unique_ptr<Conversation::Protocol> protocol,
            DeploymentPlan plan,
            PromptPrefixReuse prefix_reuse )
            : installed_( std::move( installed ) ),
              capabilities_( installed_.capabilities() ),
              model_( std::move( model ) ),
              tokenizer_( std::move( tokenizer ) ),
              protocol_( std::move( protocol ) ),
              plan_( std::move( plan ) ),
              prefix_reuse_( prefix_reuse )
        {}

        /// The installed model of that name, once check finds nothing to refuse.
        static InstalledModel checkedNamed( std::string_view name, const DeploymentRequest& request, const ModelStore& store )
        {
            std::optional<InstalledModel> installed = InstalledModel::find( name, store );

            if ( !installed )
            {
                throw OpenRefusedError( *check( name, request, store ) );
            }

            if ( auto refusal = check( *installed, request ) )
            {
                throw OpenRefusedError( std::move( *refusal ) );
            }

            return std::move( *installed );
        }

        InstalledModel installed_;
        ModelCapabilities capabilities_;
        std::unique_ptr<LanguageModelType> model_;
        std::shared_ptr<Data::BpeTokenizer> tokenizer_;
        std::unique_ptr<Conversation::Protocol> protocol_;
        DeploymentPlan plan_;
        PromptPrefixReuse prefix_reuse_;
    };
}
