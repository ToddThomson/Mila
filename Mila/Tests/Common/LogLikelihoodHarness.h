/**
 * @file LogLikelihoodHarness.h
 * @brief A network built from a weights file for teacher-forced measurement, outside any model.
 *
 * A log-likelihood is a measurement of a model, not something a model's user calls, so it is reached
 * at the network layer and no model class carries it. Include after `import Mila;`.
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <unordered_set>
#include <vector>

namespace Mila::Tests::Common
{
    /**
     * @brief Build `TNetwork` from `weights` at `context_length` and load it, as a plan would.
     *
     * The prefill chunk is the one the library's own rule chooses against one reading of free
     * memory, taken after construction because the execution context construction creates holds
     * device memory (Deployment.md section 9). A measurement built this way runs the chunking a
     * model loaded at the same context would, which is part of what it measures.
     *
     * @param chosen When given, receives the chunking the rule chose.
     */
    template<typename TNetwork, typename TNetworkConfig>
    std::unique_ptr<TNetwork> buildMeasuredNetwork(
        const std::filesystem::path& weights, const TNetworkConfig& config,
        Mila::Dnn::Compute::DeviceId device, Mila::Dnn::dim_t context_length,
        Mila::Dnn::PrefillChunking* chosen = nullptr )
    {
        using namespace Mila::Dnn;

        Serialization::WeightsReader reader( weights );

        auto network = std::make_unique<TNetwork>( reader.getWeightsMetadata().model_name, config, device );

        const Mila::Deployment::DeviceReading reading = Mila::Deployment::DeviceReading::take( device );

        const BuildContext context = BuildContext( shape_t{ 1, context_length }, RuntimeMode::Inference, false )
            .withAllocationGranularity( reading.allocation_granularity );

        const PrefillChunking chunking = Mila::Deployment::choosePrefillChunk( *network, context, reading.free_bytes );

        if ( chosen != nullptr )
        {
            *chosen = chunking;
        }

        network->build( context.withPrefillSize( chunking.chunk_rows ) );
        network->loadParameters( reader );

        return network;
    }

    /// Token ids as the [1, T] device tensor a network's passes read.
    template<typename TNetwork>
    typename TNetwork::TokenIndexType deviceTokens( const TNetwork& network, const std::vector<std::int32_t>& ids )
    {
        using namespace Mila::Dnn;

        const shape_t shape{ 1, static_cast<dim_t>( ids.size() ) };

        Tensor<TensorDataType::INT32, Compute::CpuMemoryResource> host( Compute::Device::Cpu(), shape );
        std::copy( ids.begin(), ids.end(), host.data() );

        typename TNetwork::TokenIndexType device( network.getDeviceId(), shape );
        copy( host, device );

        return device;
    }

    template<typename TNetwork>
    Mila::Dnn::SequenceLogLikelihood sequenceLogLikelihoodOf( TNetwork& network, const std::vector<std::int32_t>& tokens )
    {
        return network.sequenceLogLikelihood( deviceTokens( network, tokens ) );
    }

    /// The logits a pass returned, on the host as FP32, once the pass that wrote them has finished.
    template<typename TNetwork, typename TLogits>
    std::vector<float> hostLogits( TNetwork& network, const TLogits& logits )
    {
        network.synchronize();

        auto host = Mila::Dnn::toHost<Mila::Dnn::TensorDataType::FP32>( logits );

        return std::vector<float>( host.data(), host.data() + host.size() );
    }

    /// Index of the largest logit; the lowest index wins a tie.
    inline std::int32_t argMax( const std::vector<float>& logits )
    {
        return static_cast<std::int32_t>(
            std::distance( logits.begin(), std::max_element( logits.begin(), logits.end() ) ) );
    }

    /// A greedy continuation and the prefill's logits, the distribution over what follows the prompt.
    struct GreedyContinuation
    {
        std::vector<std::int32_t> tokens;
        std::vector<float> prompt_logits;
    };

    /**
     * @brief Greedy generation at the network layer: prefill, then argmax and decode until a stop token,
     *        `max_new_tokens`, or the context length.
     *
     * The stop token that ends a run is not part of the continuation, as a model's generate() reports it.
     */
    template<typename TNetwork>
    GreedyContinuation greedyContinuationOf( TNetwork& network, const std::vector<std::int32_t>& prompt,
        int max_new_tokens, const std::unordered_set<std::int32_t>& stop_tokens, Mila::Dnn::dim_t context_length )
    {
        GreedyContinuation result;

        result.prompt_logits = hostLogits( network, network.prefill( deviceTokens( network, prompt ) ) );

        std::int32_t token = argMax( result.prompt_logits );
        Mila::Dnn::dim_t position = static_cast<Mila::Dnn::dim_t>( prompt.size() );

        while ( !stop_tokens.contains( token ) )
        {
            result.tokens.push_back( token );

            if ( static_cast<int>( result.tokens.size() ) >= max_new_tokens || position >= context_length )
                break;

            token = argMax( hostLogits( network, network.decode( deviceTokens( network, { token } ), position ) ) );
            ++position;
        }

        return result;
    }
}
