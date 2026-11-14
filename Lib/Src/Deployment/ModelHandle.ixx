/**
 * @file ModelHandle.ixx
 * @brief An opened model, whatever its family, device and precision: the model, its tokenizer, its chat template and
 *        what it can do.
 *
 * The one place a store name becomes a running model. See Specifications/ModelHandle.md.
 */

export module Deployment.ModelHandle;

import std;
import Compute.DeviceType;
import Compute.MemoryHold;
import Dnn.TensorTypes;
import Dnn.WeightQuantization;
import Dnn.ILanguageModel;
import Dnn.PromptPrefixReuse;
import Dnn.Models.Conversation.Protocol;
import Data.BpeTokenizer;
import Deployment.DeploymentRequest;
import Deployment.DeploymentPlan;
import Deployment.DeploymentPlans;
import Deployment.DeploymentRefusal;
import Deployment.InstalledModel;
import Distribution.ModelCapabilities;
import Distribution.ModelStore;
import Distribution.ModelHub;
import Distribution.PublishedModel;
import Deployment.OpenRefusal;

namespace Mila::Deployment
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Distribution;

    /**
     * @brief A model opened by its store name, with its family, device and precision chosen at the open and erased.
     *
     * Holds what a model is -- the network, its tokenizer, its chat template, its capabilities, the plan it was
     * loaded on and what the store recorded -- and no conversation. The device type comes from the devices the
     * request allows, the precision from that device type (ModelHandle.md 10.7). Generation is ILanguageModel's one
     * virtual call, so nothing here sits between a token and the network.
     */
    export class ModelHandle
    {
    public:

        /**
         * @brief Whether a name would open, loading nothing: the reason it would not, or nothing when it would.
         *
         * Reads the model's record and, for a request that fixes a context, the header of its weights, where the
         * trained maximum is. Weights whose header cannot be read pass here, and the load says why. A base model
         * opens: it has no protocol. Whether it fits a device is the planner's answer, not this one.
         */
        [[nodiscard]] static std::optional<OpenRefusal> check(
            std::string_view name, const DeploymentRequest& request = {}, const ModelStore& store = ModelStore{} );

        /// @copydoc check
        [[nodiscard]] static std::optional<OpenRefusal> check( const InstalledModel& installed, const DeploymentRequest& request );

        /**
         * @brief Open an installed model on the best plan for the request.
         *
         * Never installs. The tokenizer and the chat template are read before the weights, so a model that cannot
         * be conversed with is refused before its load.
         *
         * @throws OpenRefusedError       for every refusal check gives, before anything is loaded.
         * @throws DeploymentRefusedError when nothing fits the allowed devices, carrying the refusal.
         * @throws std::invalid_argument  when the allowed devices are of more than one device type.
         */
        static ModelHandle open(
            std::string_view name, const DeploymentRequest& request, const ModelStore& store = ModelStore{} );

        /**
         * @brief What open would load, without loading: the plans for the request, or why none fits.
         *
         * @throws OpenRefusedError as open, before any load.
         */
        static std::expected<DeploymentPlans, DeploymentRefusal> planFor(
            std::string_view name, const DeploymentRequest& request, const ModelStore& store = ModelStore{} );

        /**
         * @brief Whether a published model would open once installed, from its manifest alone: the reason it would
         *        not, or nothing when it would. A base model opens, as check says of an installed one.
         */
        [[nodiscard]] static std::optional<OpenRefusal> check( const PublishedModel& published );

        /**
         * @brief What open would load once a published model is installed, before it is: the plans on this computer
         *        for the request, or why none fits.
         *
         * Reads the start of its weights from the hub (PublishedModel::weightsHeader) and plans it as an installed
         * model is planned, so the answer is the one its load would give on a device in the same state.
         *
         * @throws OpenRefusedError   for every refusal check gives, before anything is read from the hub.
         * @throws std::runtime_error when the start of its weights cannot be read.
         */
        static std::expected<DeploymentPlans, DeploymentRefusal> planFor( const PublishedModel& published,
            const DeploymentRequest& request, const IModelHub& hub, const ModelStore& store = ModelStore{} );

        /**
         * @brief The device type a request's model runs on: that of the devices it allows, or, when it names none
         *        present, the one GPU type this build sees.
         *
         * A model runs on one device type, so allowed devices of two types are malformed. The host is never chosen:
         * it has no compute operations.
         *
         * @throws std::invalid_argument when the allowed devices span more than one device type.
         */
        [[nodiscard]] static std::expected<DeviceType, DeploymentRefusal> deviceTypeFor( const DeploymentRequest& request );

        /**
         * @brief The weight formats an installed model has a deployment in on the devices the request allows,
         *        its stored format first (InstalledModelPlanner's fittingWeightQuantizations), with the device type
         *        and precision chosen as open chooses them.
         *
         * @return Nothing when the planner could not answer for some format; empty when it answered and nothing fits,
         *         including when no device the request allows is present.
         */
        [[nodiscard]] static std::optional<std::vector<WeightQuantization>> fittingWeightQuantizations(
            const InstalledModel& model, const DeploymentRequest& request );

        ModelHandle( ModelHandle&& ) noexcept;
        ModelHandle& operator=( ModelHandle&& ) noexcept;
        ~ModelHandle();

        /// The network behind its base: generate, observe, save.
        [[nodiscard]] ILanguageModel& model() noexcept;

        /// The typed model and its components, or null when the model is not a TModel.
        template<typename TModel>
        [[nodiscard]] TModel* as() noexcept
        {
            return dynamic_cast<TModel*>( model_.get() );
        }

        [[nodiscard]] Data::BpeTokenizer& tokenizer() const noexcept;

        /// The chat template the model was trained on; null for a base model.
        [[nodiscard]] const Conversation::Protocol* protocol() const noexcept;

        /// What the package says the model can do, with only the modalities this build runs.
        [[nodiscard]] const ModelCapabilities& capabilities() const noexcept;

        /// Where a later prompt may resume on the cache: Mila's implementation of the architecture decides it.
        [[nodiscard]] PromptPrefixReuse promptPrefixReuse() const noexcept;

        /// The plan the model was loaded on: what it runs, not a second derivation of it.
        [[nodiscard]] const DeploymentPlan& plan() const noexcept;

        /// What the store recorded: name, lineage, licence, files.
        [[nodiscard]] const InstalledModel& installed() const noexcept;

        /// The name as the store spells it.
        [[nodiscard]] const std::string& name() const noexcept;

        /**
         * @brief Keep the model's memory on its device as firmly as `hold` says, from now on.
         *
         * The application decides when: firmly while the model is in use, nothing while it is not
         * (Specifications/Deployment.md 12.10). The hold is the device's, so it covers every model open on that device.
         */
        void setMemoryHold( MemoryHold hold );

    private:

        ModelHandle(
            InstalledModel installed,
            std::unique_ptr<ILanguageModel> model,
            std::shared_ptr<Data::BpeTokenizer> tokenizer,
            std::unique_ptr<Conversation::Protocol> protocol,
            DeploymentPlan plan,
            PromptPrefixReuse prefix_reuse );

        /// The longest context the weights declare, or zero when their header cannot be read.
        static dim_t trainedMaximum( const InstalledModel& installed );

        /// The installed model of that name, once check finds nothing to refuse.
        static InstalledModel checkedNamed( std::string_view name, const DeploymentRequest& request, const ModelStore& store );

        InstalledModel installed_;
        ModelCapabilities capabilities_;
        std::unique_ptr<ILanguageModel> model_;
        std::shared_ptr<Data::BpeTokenizer> tokenizer_;
        std::unique_ptr<Conversation::Protocol> protocol_;
        DeploymentPlan plan_;
        PromptPrefixReuse prefix_reuse_;
    };
}
