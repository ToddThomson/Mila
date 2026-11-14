/**
 * @file PublishedModelFit.cpp
 * @brief The body of planPublishedModels: each published model planned through ModelHandle, then ranked. The interface
 *        is PublishedModelFit.ixx.
 */

module Deployment.PublishedModelFit;

import std;
import Deployment.DeploymentRequest;
import Deployment.DeploymentPlans;
import Deployment.DeploymentRefusal;
import Deployment.ModelHandle;
import Deployment.OpenRefusal;
import Deployment.OpenRefusedError;
import Distribution.ModelHub;
import Distribution.ModelStore;
import Distribution.PublishedModel;

namespace Mila::Deployment
{
    namespace
    {
        /// Lower ranks first: agentic here, then suggestable, then planned and fitting, then the rest.
        int tierOf( const PublishedModelFit& fit )
        {
            if ( fit.agenticHere() )
                return 0;

            if ( fit.suggestable() )
                return 1;

            if ( fit.plans )
                return 2;

            return 3;
        }

        PublishedModelFit planOne( PublishedModel model, const IModelHub& hub, const DeploymentRequest& request,
            const ModelStore& store )
        {
            PublishedModelFit fit{ std::move( model ), std::nullopt, std::nullopt, std::nullopt, {} };

            if ( auto refusal = ModelHandle::check( fit.model ) )
            {
                fit.refusal = std::move( *refusal );

                return fit;
            }

            try
            {
                auto planned = ModelHandle::planFor( fit.model, request, hub, store );

                if ( planned )
                {
                    fit.plans.emplace( std::move( *planned ) );
                }
                else
                {
                    fit.shortfall.emplace( std::move( planned.error() ) );
                }
            }
            catch ( const OpenRefusedError& error )
            {
                fit.refusal = error.refusal();
            }
            catch ( const std::exception& error )
            {
                fit.unplanned = error.what();
            }

            return fit;
        }
    }

    std::vector<PublishedModelFit> planPublishedModels( const IModelHub& hub, const std::string& owner,
        const DeploymentRequest& request, const ModelStore& store )
    {
        std::vector<PublishedModelFit> fits;

        for ( PublishedModel& model : PublishedModel::list( hub, owner ) )
        {
            fits.push_back( planOne( std::move( model ), hub, request, store ) );
        }

        std::ranges::stable_sort( fits, []( const PublishedModelFit& left, const PublishedModelFit& right )
            {
                const int left_tier = tierOf( left );
                const int right_tier = tierOf( right );

                if ( left_tier != right_tier )
                    return left_tier < right_tier;

                if ( left_tier < 3 && left.model.downloadBytes() != right.model.downloadBytes() )
                    return left.model.downloadBytes() > right.model.downloadBytes();

                return left.model.name() < right.model.name();
            } );

        return fits;
    }
}
