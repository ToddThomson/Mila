/**
 * @file Gemma.FeedForward.ixx
 * @brief Which feed-forward sublayer a Gemma 4 network's blocks carry: dense, or routed through experts.
 */

module;

export module Dnn.Components.GemmaFeedForward;

namespace Mila::Dnn
{
    /**
     * @brief The feed-forward sublayer of every block in one Gemma 4 network.
     *
     * A checkpoint decides it (`num_experts` in its metadata), and every block of the network carries the
     * same one. Dense is the 12B's GeGLU; Routed is the 26B-A4B's, a dense GeGLU branch beside a router and
     * an expert bank, the two branch outputs summed (Specifications/Gemma.md section 10.3).
     */
    export enum class GemmaFeedForward
    {
        Dense,
        Routed
    };
}
