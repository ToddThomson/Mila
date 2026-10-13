/**
 * @file Chat.Footprint.ixx
 * @brief The deployment request Chat sends for a load, and the card it asks about.
 *
 * What a load gets -- its context length, its prefill chunk, or a refusal -- is decided by the library's
 * planner (Specifications/Deployment.md); Chat only asks and reports.
 */

export module Chat.Footprint;

import std;
export import Chat.FootprintVerdict;
export import Chat.ScopedLogSuppression;

import Chat.Config;
import Mila;

namespace Mila::ChatApp
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Deployment;

    /**
     * @brief Free and total device memory, or zeros when there is no device to ask.
     *
     * Zeros are the "do not claim anything" answer rather than an error: every caller here is
     * advisory, and a missing device must not stop a model being tried.
     */
    export inline DeviceMemoryInfo queryDeviceMemory( int device_index = 0 )
    {
        const auto device = DeviceRegistry::instance().getDevice( Device::Cuda( device_index ) );

        return device ? device->getMemoryInfo() : DeviceMemoryInfo{};
    }

    /**
     * @brief The card's own name, or empty when there is no CUDA device to ask.
     *
     * `Device::getDeviceName()` answers "CUDA:0", which is a coordinate rather than a name -- it
     * tells a reader nothing about the hardware their verdicts were measured against. The marketing
     * string lives on the concrete device's properties, hence the cast; empty on failure, because a
     * listing that cannot name the card is still a listing.
     */
    export inline std::string queryDeviceName( int device_index = 0 )
    {
        const auto cuda = std::dynamic_pointer_cast<CudaDevice>(
            DeviceRegistry::instance().getDevice( Device::Cuda( device_index ) ) );

        return cuda ? cuda->getProperties().getName() : std::string{};
    }

    /**
     * @brief The deployment request a load sends, before its context length is set.
     *
     * Names a weight format only when the session quantizes on load; otherwise the library loads the format the
     * weights are stored in (Deployment.md 12.6), so Chat never restates what the package already is.
     */
    export inline DeploymentRequest deploymentRequestFor(
        QuantizationMode quantization, bool quantization_applied_at_load, int device_index )
    {
        DeploymentRequest request;
        request.withDevice( DeviceId{ DeviceType::Cuda, device_index } );

        if ( quantization_applied_at_load )
        {
            request.withWeightQuantization( quantization );
        }

        return request;
    }
}
