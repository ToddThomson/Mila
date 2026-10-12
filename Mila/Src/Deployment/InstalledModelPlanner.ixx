/**
 * @file InstalledModelPlanner.ixx
 * @brief Plan an installed model by name, whatever its family, and ask which weight formats fit.
 *
 * The family switch every app made for itself, made once. See Specifications/Deployment.md 8.1 and 12.4.
 */

module;
#include <exception>
#include <expected>
#include <format>
#include <optional>
#include <stdexcept>
#include <vector>

export module Deployment.InstalledModelPlanner;

import Dnn.ModelType;
import Dnn.WeightQuantization;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Compute.DeviceType;
import Deployment.InstalledModel;
import Deployment.DeploymentRequest;
import Deployment.DeploymentPlans;
import Deployment.DeploymentRefusal;
import Dnn.Models.GemmaModel;
import Dnn.Models.LlamaModel;
import Dnn.Models.QwenModel;

namespace Mila::Deployment
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    /**
     * @brief Refuse an installed model no deployment can be planned for, naming why.
     *
     * @throws std::runtime_error when its files are missing, or its family has no planner -- GPT-2, whose FP32
     *         weights no instruct or agentic model shares, among them.
     */
    export inline void requirePlannable( const InstalledModel& model )
    {
        if ( !model.isComplete() )
        {
            throw std::runtime_error( std::format(
                "'{}' is installed but its files are missing, so it cannot be loaded", model.name() ) );
        }

        switch ( model.family() )
        {
            case ModelType::Gemma:
            case ModelType::Llama:
            case ModelType::Qwen:
                return;

            default:
                throw std::runtime_error( std::format(
                    "'{}' has architecture '{}', which this build of Mila cannot deploy",
                    model.name(), model.stored().record.architecture ) );
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

        switch ( model.family() )
        {
            case ModelType::Gemma:
                return GemmaModel<TDeviceType, TPrecision>::planDeployment( model.weightsPath(), request );

            case ModelType::Llama:
                return LlamaModel<TDeviceType, TPrecision>::planDeployment( model.weightsPath(), request );

            default:
                return QwenModel<TDeviceType, TPrecision>::planDeployment( model.weightsPath(), request );
        }
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
