/**
 * @file DeploymentPlans.ixx
 * @brief The planner's answer when anything fits: every plan it would offer, best first.
 *
 * The ranking lives here rather than on a plan, because a loaded model keeps the plan it runs and must not
 * carry plans it is not running. See Specifications/Deployment.md section 3.3.
 */

module;
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

export module Deployment.DeploymentPlans;

export import Deployment.DeploymentPlan;

namespace Mila::Deployment
{
    /**
     * @brief Loadable plans ranked by the objective (Deployment.md section 5), never empty.
     *
     * Each entry is the best plan for one set of devices the planner considered. With one device (Phase 3)
     * there is exactly one.
     */
    export class DeploymentPlans
    {
    public:

        /**
         * @throws std::logic_error when empty: no plan is a refusal, never an empty ranking.
         */
        explicit DeploymentPlans( std::vector<DeploymentPlan> ranked_plans )
            : ranked_plans_( std::move( ranked_plans ) )
        {
            if ( ranked_plans_.empty() )
            {
                throw std::logic_error( "DeploymentPlans: a ranking holds at least one plan; no plan is a refusal" );
            }
        }

        /// The planner's choice: rankedPlans().front().
        [[nodiscard]] const DeploymentPlan& best() const noexcept
        {
            return ranked_plans_.front();
        }

        /// Every plan, best first.
        [[nodiscard]] std::span<const DeploymentPlan> rankedPlans() const noexcept
        {
            return ranked_plans_;
        }

    private:

        std::vector<DeploymentPlan> ranked_plans_;
    };
}
