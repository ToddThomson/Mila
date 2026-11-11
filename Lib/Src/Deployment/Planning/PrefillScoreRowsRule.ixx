/**
 * @file PrefillScoreRowsRule.ixx
 * @brief The one place a block of prefill score rows is chosen, against the memory a context and its chunk left.
 *
 * See Specifications/Deployment.md section 5 and Specifications/Metal/MetalGlobalAttentionPrefill.md section 8.
 */

export module Deployment.PrefillScoreRowsRule;

import std;
import Dnn.Component;
import Dnn.TensorTypes;

namespace Mila::Deployment
{
    using namespace Mila::Dnn;

    /**
     * @brief The blocks the rule walks, largest first: a chunk's rungs halved down to the floor.
     *
     * Under 32 query positions a block the unfused prefill is behind the fused kernel it replaces
     * (MetalGlobalAttentionPrefill.md fact 13), so 32 is the floor and not a step toward a smaller block.
     */
    export inline constexpr dim_t kPrefillScoreRowRungs[] = { 1024, 512, 256, 128, 64, 32 };

    /**
     * @brief The largest block of score rows, no larger than the chunk, that `free_bytes` holds with as much again
     *        left over; zero when there is none or when no layer of the network wants one.
     *
     * Runs after the context length and the chunk are chosen and replaces neither, so it only spends memory a plan
     * had spare. Each block is priced by the network's own getRequiredMemory(), as a chunk is.
     *
     * A block must fit twice over because one that took the last of the memory did not run: Gemma 4 12B at a context
     * of 262,144 on a 16 GB Mac, given a block of 32 rows, was planned at 98.5% of the working set and its first
     * prefill failed for want of GPU memory, where the same plan without the block, at 96.4%, runs
     * (Metal/MetalGlobalAttentionPrefill.md fact 14). What the run needs beyond the plan is not found; until it is
     * priced, a block is not what a plan spends its last memory on.
     *
     * @tparam TNetwork A language network exposing `wantsPrefillScoreRows()`.
     * @param network    The constructed graph; nothing is allocated.
     * @param context    The build context to price, carrying the allocation granularity and the chunk already
     *                   chosen; any score rows it carries are replaced by each candidate.
     * @param free_bytes The budget the context and chunk were chosen against. Zero means the device could not say,
     *                   and takes the largest block the chunk permits.
     */
    export template<typename TNetwork>
    dim_t choosePrefillScoreRows( const TNetwork& network, const BuildContext& context, std::size_t free_bytes )
    {
        if ( !network.wantsPrefillScoreRows() )
            return 0;

        const dim_t chunk_rows = context.getResolvedPrefillSize();
        const std::size_t without_a_block = network.getRequiredMemory( context.withPrefillScoreRows( 0 ) ).totalDeviceBytes();

        for ( dim_t candidate : kPrefillScoreRowRungs )
        {
            if ( candidate > chunk_rows )
                continue;

            if ( free_bytes == 0 )
                return candidate;

            const std::size_t with_the_block = network.getRequiredMemory( context.withPrefillScoreRows( candidate ) ).totalDeviceBytes();
            const std::size_t block_bytes = with_the_block > without_a_block ? with_the_block - without_a_block : 0;

            if ( with_the_block + block_bytes <= free_bytes )
                return candidate;
        }

        return 0;
    }
}
