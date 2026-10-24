/**
 * @file Chat.RoundStats.ixx
 * @brief Timing and outcome of one generate call within a turn.
 */

export module Chat.RoundStats;

import std;
import Dnn.GenerateStatus;

namespace Mila::ChatApp
{
    /**
     * @brief One round of a turn, measured from the token callback's cadence (the library owns no stopwatch).
     *
     * A turn is one or more rounds: each tool call splits generation into a new round with its own prefill (a
     * full re-prefill of the prompt and the turn so far -- Chat has no incremental-KV continuation). Lumping
     * rounds into one prefill and decode figure mixes decode with re-prefill, which is why /stats reports them
     * separately.
     */
    export struct RoundStats
    {
        float prefill_time_ms = 0.0f;          // round start -> first token of round
        float decode_tokens_per_second = 0.0f; // steady-state decode throughput this round
        int tokens_generated = 0;
        bool ended_in_tool_call = false;
        std::string tool_name;                 // populated when ended_in_tool_call

        // Inter-token gap census for GPU-utilization-dip triage: the callback cadence is the per-token wall
        // time, so a mid-response stall inside generate() shows up as an outlier gap here. Uniform gaps and a
        // dipping utilization graph mean the GPU is being time-sliced by another process, not idled by the loop.
        float median_gap_ms = 0.0f;
        float p99_gap_ms = 0.0f;
        float max_gap_ms = 0.0f;
        int max_gap_token_index = 0;           // token that ended the largest gap
        int stall_count = 0;                   // gaps > max(2.5 x median, 40 ms)
        float stall_total_ms = 0.0f;           // wall time inside those stall gaps

        // Why the round's generate() returned -- distinguishes a natural stop from hitting the token cap or the
        // context bound (a capped round looks like a hang in a buffered display: pegged GPU, no output, until
        // the cap is reached).
        Dnn::GenerateStatus finish_status = Dnn::GenerateStatus::Success;

        /// The most tokens the round was allowed: the cap a `length` finish reached.
        std::size_t token_budget = 0;
    };
}
