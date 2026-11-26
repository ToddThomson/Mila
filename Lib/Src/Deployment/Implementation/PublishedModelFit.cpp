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

        rankPublishedModels( fits );

        return fits;
    }

    void rankPublishedModels( std::vector<PublishedModelFit>& fits )
    {
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
    }

    std::optional<std::size_t> onboardingModel( const std::vector<PublishedModelFit>& fits )
    {
        std::optional<std::size_t> chosen;

        for ( std::size_t index = 0; index < fits.size(); ++index )
        {
            const PublishedModelFit& fit = fits[ index ];

            if ( !fit.suggestable() || !fit.model.manifest()->capabilities.onboarding )
            {
                continue;
            }

            if ( !chosen )
            {
                chosen = index;
                continue;
            }

            const PublishedModelFit& best = fits[ *chosen ];

            if ( fit.agenticHere() != best.agenticHere() )
            {
                if ( fit.agenticHere() )
                {
                    chosen = index;
                }

                continue;
            }

            if ( fit.model.downloadBytes() < best.model.downloadBytes() )
            {
                chosen = index;
            }
        }

        return chosen;
    }
}
