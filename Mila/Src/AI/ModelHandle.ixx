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
         * @brief Open an installed model on the best plan for the request.
         *
         * Never installs. The tokenizer and the chat template are read before the weights, so a model that cannot
         * be conversed with is refused before its load.
         *
         * @throws std::runtime_error     for a name not installed, missing files, or an architecture this build
         *                                does not deploy.
         * @throws std::invalid_argument  for an instruct model whose chat template this build lacks.
         * @throws DeploymentRefusedError when nothing fits the device, carrying the refusal.
         */
        static ModelHandle open(
            std::string_view name, const DeploymentRequest& request, const ModelStore& store = ModelStore{} )
        {
            const InstalledModel installed = installedNamed( name, store );

            requirePlannable( installed );

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
         * @throws std::runtime_error as open, before any load.
         */
        static std::expected<DeploymentPlans, DeploymentRefusal> planFor(
            std::string_view name, const DeploymentRequest& request, const ModelStore& store = ModelStore{} )
        {
            return planDeployment<TDeviceType, TPrecision>( installedNamed( name, store ), request );
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

        static InstalledModel installedNamed( std::string_view name, const ModelStore& store )
        {
            std::optional<InstalledModel> installed = InstalledModel::find( name, store );

            if ( !installed )
            {
                throw std::runtime_error( std::format( "no model named '{}' is installed", name ) );
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
