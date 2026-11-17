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
    export inline DeviceMemoryInfo queryDeviceMemory( std::optional<DeviceId> device_id )
    {
        if ( !device_id )
            return {};

        try
        {
            const auto device = DeviceRegistry::instance().getDevice( *device_id );

            return device ? device->getMemoryInfo() : DeviceMemoryInfo{};
        } catch ( const std::exception& )
        {
            return {};
        }
    }

    /**
     * @brief The device's own name, such as "NVIDIA GeForce RTX 4070", or empty when there is no device to ask.
     *
     * `cuda:0` is a coordinate rather than a name -- it tells a reader nothing about the hardware their verdicts
     * were measured against. Empty on failure, because a listing that cannot name the card is still a listing.
     */
    export inline std::string queryDeviceName( std::optional<DeviceId> device_id )
    {
        return device_id ? DeviceReading::describe( *device_id ).hardware_name : std::string{};
    }

    /**
     * @brief The device the session's figures describe before a model is loaded: of the present GPUs the allowed
     *        devices name, the one with the most memory, the first in device order among equals. The planner gives a
     *        model the card where it gets the longest context (Deployment.md section 5), so a listing of what fits
     *        describes that card. Nothing when none is present.
     */
    export inline std::optional<DeviceId> describedDevice( const AllowedDevices& devices )
    {
        std::optional<DeviceReading> largest;

        for ( const DeviceReading& present : DeviceReading::describePresent() )
        {
            if ( present.device.type == DeviceType::Cpu || !devices.allows( present ) )
                continue;

            if ( !largest || present.total_bytes > largest->total_bytes )
                largest = present;
        }

        return largest ? std::optional<DeviceId>( largest->device ) : std::nullopt;
    }

    /**
     * @brief Chat's devices when the settings name none: every GPU of the type this build sees, among which the planner
     *        chooses the card (LayerSplit.md 12.3, Todd, 2026-11-17), or every device when there is none, so the load's
     *        refusal says what is missing.
     */
    export inline AllowedDevices defaultDevices()
    {
        const std::expected<DeviceType, DeploymentRefusal> type = ModelHandle::deviceTypeFor( DeploymentRequest{} );

        return type ? AllowedDevices::parse( deviceTypeToString( *type ) ) : AllowedDevices{};
    }

    /**
     * @brief The deployment request a load sends, before its context length is set.
     *
     * Names a weight format only when the session quantizes on load; otherwise the library loads the format the
     * weights are stored in (Deployment.md 12.6), so Chat never restates what the package already is. The KV
     * cache is always named: it is the session's choice, not the package's.
     */
    export inline DeploymentRequest deploymentRequestFor(
        Dnn::WeightQuantization quantization, bool quantization_applied_at_load, const AllowedDevices& devices,
        Dnn::KvCacheCompression kv_cache )
    {
        DeploymentRequest request;
        request.withAllowedDevices( devices ).withKvCacheCompression( kv_cache );

        if ( quantization_applied_at_load )
        {
            request.withWeightQuantization( quantization );
        }

        return request;
    }
}
