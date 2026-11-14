/**
 * @file PublishedModelFit.ixx
 * @brief How a published model would run on this computer, before it is installed, and every published model planned
 *        and ranked: the planner's suggestion of which to install. See Specifications/Apps/AssistantOnboarding.md.
 */

export module Deployment.PublishedModelFit;

import std;
import Dnn.TensorTypes;
import Deployment.DeploymentRequest;
import Deployment.DeploymentPlans;
import Deployment.DeploymentRefusal;
import Deployment.OpenRefusal;
import Distribution.ModelHub;
import Distribution.ModelStore;
import Distribution.PublishedModel;

namespace Mila::Deployment
{
    using namespace Mila::Distribution;

    /**
     * @brief The context Mila's agentic verdict is measured at (Specifications/ContextProfile.md): a model is an agent
     *        on this computer only when it is planned at least this long.
     */
    export inline constexpr Mila::Dnn::dim_t kAgenticContextLength = 65536;

    /**
     * @brief One published model planned on this computer: the plans when it fits, and otherwise exactly one of why
     *        it does not, why it would not open, or why it could not be planned.
     */
    export struct PublishedModelFit
    {
        PublishedModel model;

        /// The planner's plans, best first, when it fits.
        std::optional<DeploymentPlans> plans;

        /// Planned, and nothing fits: the planner's refusal.
        std::optional<DeploymentRefusal> shortfall;

        /// Not planned, because it would not open once installed.
        std::optional<OpenRefusal> refusal;

        /// Not planned, because the start of its weights could not be read; the reason, for a log.
        std::string unplanned;

        /// It fits and takes turns, so it can be suggested.
        [[nodiscard]] bool suggestable() const noexcept
        {
            return plans.has_value() && model.manifest() && model.manifest()->capabilities.instruct;
        }

        /// It fits, Mila measured it agentic, and it is planned at the context that verdict was measured at.
        [[nodiscard]] bool agenticHere() const noexcept
        {
            return suggestable() && model.manifest()->capabilities.agentic
                && plans->best().contextLength() >= kAgenticContextLength;
        }
    };

    /**
     * @brief Every model the owner publishes, each planned on this computer for the request, ranked: those that can be
     *        suggested first, agentic here before the rest, the larger download first within each; then the rest by
     *        name. The suggestion is the first that is suggestable.
     *
     * Larger first is provisional: nothing a manifest carries ranks two models that fit by how well they answer
     * (AssistantOnboarding.md, question 3). Each model costs two small reads of its weights the first time, none after.
     *
     * @throws std::exception when the owner's listing cannot be read.
     */
    export std::vector<PublishedModelFit> planPublishedModels( const IModelHub& hub, const std::string& owner,
        const DeploymentRequest& request, const ModelStore& store = ModelStore{} );
}
