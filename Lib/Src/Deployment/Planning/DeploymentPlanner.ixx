/**
 * @file DeploymentPlanner.ixx
 * @brief The deployment decision for any family's network: on one device from one reading, and across the
 *        devices a request allows.
 *
 * Each family's planDeployment constructs its graph on each allowed device, takes that device's reading and plans
 * against it. See Specifications/Deployment.md sections 5, 6 and 12.8.
 */

export module Deployment.DeploymentPlanner;

import std;
import Dnn.Component;
import Dnn.RuntimeMode;
import Dnn.TensorTypes;
import Compute.DeviceId;
import Compute.DeviceType;
import Deployment.AllowedDevices;
import Deployment.DeviceReading;
import Deployment.DeploymentRequest;
import Deployment.DeploymentPlan;
import Deployment.DeploymentPlans;
import Deployment.DeploymentRefusal;
import Deployment.PrefillChunkRule;
import Serialization.WeightsReader;

namespace Mila::Deployment
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    /**
     * @brief Plan a request on the device a reading describes, or refuse it.
     *
     * The objective, strictly in this order (Deployment.md section 5): honour every value the request fixed;
     * among context lengths that fit, the largest whose prefill chunk memory did not reduce; otherwise the
     * largest that fits; otherwise a refusal. Automatic context lengths are searched from the ceiling down in
     * steps of DeploymentRequest::kContextStep -- not by bisection, because a footprint need not rise with
     * context -- and the search ends as soon as the weights alone exceed the budget, since no context length
     * shrinks them.
     *
     * @param network            The constructed graph; nothing is allocated.
     * @param request            What the caller asked for.
     * @param reading            The one reading every value is decided against. A reading that reports no memory
     *                           plans a fixed context length unchecked and refuses an automatic one.
     * @param trained_maximum    The longest context the package supports.
     * @param metadata           The package facts pricing read, kept on the plan.
     * @param stored_quantization The weight format the package stores; empty for reference weights.
     * @throws std::invalid_argument when a fixed context length, or the automatic floor, exceeds the trained
     *         maximum: a request no memory could satisfy is malformed rather than refused.
     */
    export template<typename TNetwork>
    std::expected<DeploymentPlan, DeviceShortfall> planOnDevice(
        const TNetwork& network,
        const DeploymentRequest& request,
        const DeviceReading& reading,
        dim_t trained_maximum,
        const WeightsMetadata& metadata,
        std::string_view stored_quantization )
    {
        const bool reports_memory = reading.reportsMemory();
        const std::size_t headroom = request.getHeadroom();
        const std::size_t available = request.plansAgainstWholeDeviceMemory() ? reading.total_bytes : reading.free_bytes;
        const std::size_t budget = available > headroom ? available - headroom : 0;

        struct Priced
        {
            dim_t context_length;
            PrefillChunking prefill;
            MemoryStats footprint;
        };

        auto priceAt = [&]( dim_t context_length )
        {
            const BuildContext context = BuildContext( shape_t{ 1, context_length }, RuntimeMode::Inference, false )
                .withAllocationGranularity( reading.allocation_granularity );

            // Zero tells the rule the device could not say, so a device that reported memory and has none left
            // after the headroom is given one byte instead: the rule then returns its smallest rung, not fitting.
            const PrefillChunking prefill = choosePrefillChunk(
                network, context, reports_memory ? std::max<std::size_t>( budget, 1 ) : 0 );

            return Priced{ context_length, prefill,
                network.getRequiredMemory( context.withPrefillSize( prefill.chunk_rows ) ) };
        };

        auto fits = [&]( const Priced& priced )
        {
            return priced.prefill.fits_available_memory && priced.footprint.totalDeviceBytes() <= budget;
        };

        auto planOf = [&]( const Priced& priced, DeploymentPlan::ContextLimit limit )
        {
            return DeploymentPlan(
                reading, priced.context_length, limit, priced.prefill, priced.footprint,
                request.getWeightQuantization(), request.getKvCacheCompression(),
                metadata, std::string( stored_quantization ), request.getSpeculativeDecode() );
        };

        auto refusalOf = [&]( DeviceShortfall::Reason reason, const Priced& priced )
        {
            return std::unexpected( DeviceShortfall( reason, reading, headroom, priced.context_length, priced.footprint ) );
        };

        if ( !request.isContextLengthAutomatic() )
        {
            const dim_t context_length = request.getContextLength();

            if ( context_length > trained_maximum )
            {
                throw std::invalid_argument( std::format(
                    "planDeployment: context length {} exceeds the trained maximum {}", context_length, trained_maximum ) );
            }

            const Priced priced = priceAt( context_length );

            if ( reports_memory )
            {
                if ( priced.footprint.device_parameter_bytes > budget )
                    return refusalOf( DeviceShortfall::Reason::WeightsExceedDevice, priced );

                if ( !fits( priced ) )
                    return refusalOf( DeviceShortfall::Reason::FixedContextDoesNotFit, priced );
            }

            return planOf( priced, DeploymentPlan::ContextLimit::FixedByCaller );
        }

        const dim_t floor = request.getContextFloor();

        if ( floor > trained_maximum )
        {
            throw std::invalid_argument( std::format(
                "planDeployment: automatic context floor {} exceeds the trained maximum {}", floor, trained_maximum ) );
        }

        if ( !reports_memory )
        {
            return std::unexpected( DeviceShortfall(
                DeviceShortfall::Reason::DeviceDoesNotReportMemory, reading, headroom, floor, MemoryStats{} ) );
        }

        const dim_t ceiling = request.getContextCeiling() == 0
            ? trained_maximum
            : std::min( request.getContextCeiling(), trained_maximum );
        const dim_t step = DeploymentRequest::kContextStep;
        const dim_t top = std::max( ( ceiling / step ) * step, floor );

        // The largest context length that fits, held while the search continues for one whose chunk memory did
        // not reduce. It is the answer when there is none: a reduced chunk costs prefill throughput, and giving
        // up context to avoid it is worth doing only while there is context left to give up.
        std::optional<Priced> largest_fitting;
        Priced smallest_priced{ floor, {}, {} };

        for ( dim_t context_length = top; context_length >= floor; context_length -= step )
        {
            const Priced priced = priceAt( context_length );
            smallest_priced = priced;

            if ( priced.footprint.device_parameter_bytes > budget )
                return refusalOf( DeviceShortfall::Reason::WeightsExceedDevice, priced );

            if ( !fits( priced ) )
                continue;

            if ( !largest_fitting )
                largest_fitting = priced;

            if ( !priced.prefill.isMemoryConstrained() )
            {
                const DeploymentPlan::ContextLimit limit = context_length < largest_fitting->context_length
                    ? DeploymentPlan::ContextLimit::FullPrefillChunk
                    : context_length == top
                        ? DeploymentPlan::ContextLimit::TrainedMaximum
                        : DeploymentPlan::ContextLimit::DeviceMemory;

                return planOf( priced, limit );
            }
        }

        if ( largest_fitting )
        {
            return planOf( *largest_fitting, largest_fitting->context_length == top
                ? DeploymentPlan::ContextLimit::TrainedMaximum
                : DeploymentPlan::ContextLimit::DeviceMemory );
        }

        return refusalOf( DeviceShortfall::Reason::NothingAboveTheFloorFits, smallest_priced );
    }

    /**
     * @brief Plan a request on each device it allows, rank the plans, or refuse with every device's shortfall.
     *
     * Each allowed device of the model's type is planned alone, in device order; no plan splits a model
     * (LayerSplit.md). Plans are ranked by section 5's rules 3 and 4 across devices -- a full prefill chunk first,
     * then the longer context -- and plans equal on both keep device order, since nothing ranks devices by speed
     * yet (Deployment.md 12.9). A present device the request does not allow is named in a refusal and never read.
     *
     * @param request        What the caller asked for; its allowed devices decide which devices are planned.
     * @param device_type    The device type this model's build runs on.
     * @param present        Every device the build sees, described (DeviceReading::describePresent).
     * @param plan_on_device Plans one device -- constructs the graph there, reads the device after it, and calls
     *                       planOnDevice -- returning std::expected<DeploymentPlan, DeviceShortfall>.
     * @throws std::invalid_argument when the request names a present device of another type than device_type.
     */
    export template<typename TPlanOnDevice>
    std::expected<DeploymentPlans, DeploymentRefusal> planOnAllowedDevices(
        const DeploymentRequest& request,
        DeviceType device_type,
        std::span<const DeviceReading> present,
        TPlanOnDevice&& plan_on_device )
    {
        const AllowedDevices& allowed = request.getAllowedDevices();

        std::vector<DeviceId> candidates;
        std::vector<DeviceReading> not_allowed;

        for ( const DeviceReading& device : present )
        {
            const bool is_allowed = allowed.allows( device );

            if ( device.device.type != device_type )
            {
                if ( is_allowed && !allowed.isEveryDevice() )
                {
                    throw std::invalid_argument( std::format(
                        "planDeployment: device type mismatch: this model runs on {}, and the request allows {}",
                        deviceTypeToString( device_type ), device.device.toString() ) );
                }

                continue;
            }

            if ( is_allowed )
                candidates.push_back( device.device );
            else
                not_allowed.push_back( device );
        }

        std::vector<std::string> absent = allowed.absentFrom( present );

        if ( !absent.empty() )
        {
            return std::unexpected( DeploymentRefusal(
                DeploymentRefusal::Reason::AllowedDeviceAbsent, {}, std::move( not_allowed ), std::move( absent ) ) );
        }

        if ( candidates.empty() )
        {
            return std::unexpected( DeploymentRefusal(
                DeploymentRefusal::Reason::NoDeviceToPlanOn, {}, std::move( not_allowed ) ) );
        }

        std::vector<DeploymentPlan> plans;
        std::vector<DeviceShortfall> shortfalls;

        for ( const DeviceId device : candidates )
        {
            std::expected<DeploymentPlan, DeviceShortfall> planned = plan_on_device( device );

            if ( planned )
                plans.push_back( std::move( *planned ) );
            else
                shortfalls.push_back( std::move( planned.error() ) );
        }

        if ( plans.empty() )
        {
            return std::unexpected( DeploymentRefusal(
                DeploymentRefusal::Reason::NoAllowedDeviceFits, std::move( shortfalls ), std::move( not_allowed ) ) );
        }

        std::ranges::stable_sort( plans, []( const DeploymentPlan& left, const DeploymentPlan& right )
            {
                const bool left_full = !left.prefillChunking().isMemoryConstrained();
                const bool right_full = !right.prefillChunking().isMemoryConstrained();

                if ( left_full != right_full )
                    return left_full;

                return left.contextLength() > right.contextLength();
            } );

        return DeploymentPlans( std::move( plans ) );
    }

    /**
     * @brief planOnAllowedDevices against every device the build sees now.
     */
    export template<typename TPlanOnDevice>
    std::expected<DeploymentPlans, DeploymentRefusal> planOnAllowedDevices(
        const DeploymentRequest& request,
        DeviceType device_type,
        TPlanOnDevice&& plan_on_device )
    {
        const std::vector<DeviceReading> present = DeviceReading::describePresent();

        return planOnAllowedDevices( request, device_type, std::span<const DeviceReading>( present ),
            std::forward<TPlanOnDevice>( plan_on_device ) );
    }
}
