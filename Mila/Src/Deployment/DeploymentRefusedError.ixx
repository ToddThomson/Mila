/**
 * @file DeploymentRefusedError.ixx
 * @brief What load( path, request ) throws when the planner refuses: the refusal itself, with a message that
 *        names the reason and the numbers.
 *
 * std::bad_expected_access names no reason, so a load that has nothing to return throws this instead.
 */

module;
#include <stdexcept>
#include <string>
#include <string_view>

export module Deployment.DeploymentRefusedError;

export import Deployment.DeploymentRefusal;

namespace Mila::Deployment
{
    /**
     * @brief A refused deployment, thrown by a load that had no plan to execute.
     */
    export class DeploymentRefusedError : public std::runtime_error
    {
    public:

        DeploymentRefusedError( std::string_view caller, const DeploymentRefusal& refusal )
            : std::runtime_error( std::string( caller ) + ": no deployment fits -- " + refusal.toString() ),
              refusal_( refusal )
        {
        }

        [[nodiscard]] const DeploymentRefusal& refusal() const noexcept
        {
            return refusal_;
        }

    private:

        DeploymentRefusal refusal_;
    };
}
