/**
 * @file InstalledModelPlanner.ixx
 * @brief Plan an installed model by name, whatever its family, and ask which weight formats fit.
 *
 * The family switch every app made for itself, made once, through the architecture list. See
 * Specifications/Deployment.md 8.1 and 12.4.
 */

export module Deployment.InstalledModelPlanner;

import std;
import Dnn.WeightQuantization;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Compute.DeviceType;
import Deployment.InstalledModel;
import Deployment.DeploymentRequest;
import Deployment.DeploymentPlans;
import Deployment.DeploymentRefusal;
import Dnn.Models.Architectures;

namespace Mila::Deployment
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    /**
     * @brief Refuse an installed model no deployment can be planned for, naming why.
     *
     * @throws std::runtime_error when its files are missing, or its architecture is not in the list -- GPT-2,
     *         which has no planner, among them.
     */
    export inline void requirePlannable( const InstalledModel& model )
    {
        if ( !model.isComplete() )
        {
            throw std::runtime_error( std::format(
                "'{}' is installed but its files are missing, so it cannot be loaded", model.name() ) );
        }

        if ( !Architectures::contains( model.architecture() ) )
        {
            throw std::runtime_error( std::format(
                "'{}' has architecture '{}', which this build of Mila cannot deploy; it deploys {}",
                model.name(), model.architecture(), Architectures::names() ) );
        }
    }

    /**
     * @brief Plan an installed model on the request's device, through its family's planner.
     *
     * A request that names no weight format plans the one the weights are stored in (Deployment.md 12.6).
     *
     * @throws std::runtime_error as requirePlannable, and as the family's planDeployment.
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision = TensorDataType::BF16>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    std::expected<DeploymentPlans, DeploymentRefusal> planDeployment(
        const InstalledModel& model, const DeploymentRequest& request )
    {
        requirePlannable( model );

        std::optional<std::expected<DeploymentPlans, DeploymentRefusal>> planned;

        Architectures::visit( model.architecture(), [&]( auto named )
            {
                using Architecture = typename decltype( named )::type;

                planned = Architecture::template Model<TDeviceType, TPrecision>::planDeployment(
                    model.weightsPath(), request );
            } );

        return std::move( *planned );
    }

    /**
     * @brief The weight formats this model has a deployment in on the request's device: its stored format first,
     *        then each format reference weights can be quantized to on load.
     *
     * What a listing shows ("yes", "yes, at fp8 or fp4"), asked of the planner once per format. A pre-quantized
     * model offers only its own format; reference weights also offer FP8 and FP4, since Q4_0 on load is not a
     * published path. The request's own weight format is ignored.
     *
     * @return Nothing when the planner could not answer for some format: an unreadable package, or a fixed
     *         context past what the model addresses. Empty when it answered and nothing fits.
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision = TensorDataType::BF16>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    std::optional<std::vector<WeightQuantization>> fittingWeightQuantizations(
        const InstalledModel& model, const DeploymentRequest& request )
    {
        const std::optional<WeightQuantization> stored = model.storedWeightQuantization();

        if ( !stored )
            return std::nullopt;

        std::vector<WeightQuantization> candidates{ *stored };

        if ( *stored == WeightQuantization::None )
        {
            candidates.push_back( WeightQuantization::FP8 );
            candidates.push_back( WeightQuantization::FP4 );
        }

        std::vector<WeightQuantization> fitting;

        try
        {
            for ( const WeightQuantization candidate : candidates )
            {
                DeploymentRequest asked = request;
                asked.withWeightQuantization( candidate );

                if ( planDeployment<TDeviceType, TPrecision>( model, asked ).has_value() )
                {
                    fitting.push_back( candidate );
                }
            }
        }
        catch ( const std::exception& )
        {
            return std::nullopt;
        }

        return fitting;
    }
}
