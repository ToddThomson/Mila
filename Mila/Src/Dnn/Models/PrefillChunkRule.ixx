/**
 * @file PrefillChunkRule.ixx
 * @brief The one place a prefill chunk is chosen against a reading of free device memory.
 *
 * A language network builds with the chunk it is given and never reads the device itself, so
 * the prediction and the build cannot disagree about it. See Specifications/Deployment.md.
 */

module;
#include <algorithm>
#include <cstddef>

export module Dnn.Models.PrefillChunkRule;

import Dnn.Component;
import Dnn.TensorTypes;

namespace Mila::Dnn
{
    /**
     * @brief The largest rung of a family's table whose whole predicted footprint fits `free_bytes`.
     *
     * Each rung is priced by the network's own getRequiredMemory() at that chunk, so the chunk
     * chosen and the memory reported for it are one computation. A context shorter than the
     * smallest rung is prefilled as one chunk. When even the smallest rung does not fit, it is
     * returned and marked as not fitting; whether that is worth a warning is the caller's call,
     * because a scan asks at many context lengths the user never chose.
     *
     * @tparam TNetwork A language network exposing `kPrefillChunkRungs`, largest first.
     * @param network    The constructed graph; nothing is allocated.
     * @param context    The build context to price. Must carry the allocation granularity; any
     *                   prefill size it carries is replaced by each candidate.
     * @param free_bytes One reading of the device's free memory. Zero means the device could not
     *                   say, and takes the largest rung the context permits.
     */
    export template<typename TNetwork>
    PrefillChunking choosePrefillChunk(
        const TNetwork& network, const BuildContext& context, std::size_t free_bytes )
    {
        const dim_t context_length = context.inputShape()[ 1 ];
        const dim_t smallest_rung = std::ranges::min( TNetwork::kPrefillChunkRungs );

        PrefillChunking chunking;

        if ( context_length < smallest_rung )
        {
            chunking.chunk_rows = context_length;
            chunking.unconstrained_chunk_rows = context_length;

            return chunking;
        }

        for ( dim_t candidate : TNetwork::kPrefillChunkRungs )
        {
            if ( candidate > context_length )
                continue;

            // The first rung the context admits at all, memory aside.
            if ( chunking.unconstrained_chunk_rows == 0 )
                chunking.unconstrained_chunk_rows = candidate;

            if ( free_bytes == 0
                || network.getRequiredMemory( context.withPrefillSize( candidate ) ).totalDeviceBytes() <= free_bytes )
            {
                chunking.chunk_rows = candidate;

                return chunking;
            }
        }

        chunking.chunk_rows = smallest_rung;
        chunking.fits_available_memory = false;

        return chunking;
    }
}
