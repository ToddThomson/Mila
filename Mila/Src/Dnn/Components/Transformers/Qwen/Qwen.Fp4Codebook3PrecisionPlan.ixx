/**
 * @file Qwen.Fp4Codebook3PrecisionPlan.ixx
 * @brief Qwen 3.8-27B's quality-first agentic allocation, `fp4-cb3`: FP4 with two roles at three-bit codebooks.
 */

export module Dnn.Components.QwenFp4Codebook3PrecisionPlan;

import std;
import Dnn.Quantization.Weight.Policies;
import Dnn.Quantization.Weight.PrecisionPlan;
import Dnn.Components.QwenPrecisionPlan;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Quant::Weight;

    /**
     * @brief FP4 everywhere the oracle quantizes, but the FFN down and DeltaNet q/k at three-bit codebooks.
     *
     * The allocation for agentic use on a 16 GB card at 64K context and more (Specifications/Qwen3.8.md section 5,
     * "fp4-cb3"). Each sub-4-bit role was chosen by measuring it alone against the FP4 oracle on held-out code:
     *
     * | Role cut from FP4            | Saves   | Mean KL to BF16 (FP4: 0.478) | Closing brackets lost (FP4: 2.1%) |
     * |------------------------------|---------|------------------------------|-----------------------------------|
     * | DeltaNetQueryKey, codebook 3 | 0.11 GB | 0.481                        | 2.2%                              |
     * | FeedForwardDown, codebook 3  | 0.62 GB | 0.505                        | 3.7%                              |
     * | DeltaNet v/gate/o, codebook 2| 0.92 GB | 0.761 -- not taken           | 3.7%                              |
     * | FeedForwardGateUp, codebook 2| --      | not taken: drops one closing bracket of a nested pair in code |
     *
     * Together the two taken cuts measured KL 0.511 and no nesting errors. FeedForwardGateUp and
     * DeltaNetValueGateOutput stay FP4 for that reason; DeltaNetGating stays BF16 for the reason the section 5 plan
     * gives, and the embedding table is host-resident as there.
     */
    export struct QwenFp4Codebook3PrecisionPlan
    {
        using QkvProjection = PerGroupFp4<128>;
        using OutputProjection = PerGroupFp4<128>;

        using FeedForwardGateUp = PerGroupFp4<128>;
        using FeedForwardDown = PerGroupCodebook3<64>;

        using DeltaNetQueryKey = PerGroupCodebook3<64>;
        using DeltaNetValueGateOutput = PerGroupFp4<128>;
        using DeltaNetGating = NoWeightQuant;

        using EmbeddingTable = NoWeightQuant;
        using LanguageModelHead = PerGroupFp4<128>;
    };

    static_assert( DecoderPrecisionPlan<QwenFp4Codebook3PrecisionPlan> );
    static_assert( DeltaNetPrecisionRoles<QwenFp4Codebook3PrecisionPlan> );
    static_assert( EmbeddingPrecisionRole<QwenFp4Codebook3PrecisionPlan> );
    static_assert( LanguageModelHeadRole<QwenFp4Codebook3PrecisionPlan> );
    static_assert( DeltaNetPrecisionRoles<QwenPrecisionPlanFor<QwenFp4Codebook3PrecisionPlan>> );
}
