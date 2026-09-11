/**
 * @file Qwen.PrecisionPlan.ixx
 * @brief The Qwen 3.8-27B per-role weight allocation, as a named plan struct.
 *
 * This file IS Specifications/Qwen3.8.md section 5, expressed as types. It is the one
 * declaration a reader consults to learn what the model spends its 12 GiB on, which is the
 * property the whole per-role mechanism exists to preserve (section 6).
 */

module;
#include <concepts>

export module Dnn.Components.QwenPrecisionPlan;

import Dnn.Quantization.Weight.Policies;
import Dnn.Quantization.Weight.PrecisionPlan;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Quant::Weight;

    /**
     * @brief Roles a Gated DeltaNet block instantiates, on top of the feed-forward pair.
     *
     * The DeltaNet block is Phase 3 and does not exist yet; this concept is here because
     * the plan below already names its roles, and a role no concept ever checks is a
     * comment. When the block lands it requires this alongside FeedForwardPrecisionRoles,
     * exactly as QwenAttentionBlock requires AttentionPrecisionRoles.
     *
     * The query/key pair separates from the value/gate/output group because section 5
     * ranks them a half-step apart: the recurrence reads q and k to address the state
     * matrix, while v, the output gate and o carry the magnitude that enters it.
     */
    export template<typename TPlan>
        concept DeltaNetPrecisionRoles =
        requires {
            typename TPlan::DeltaNetQueryKey;
            typename TPlan::DeltaNetValueGateOutput;
            typename TPlan::DeltaNetGating;
        }
        && WeightQuantPolicy<typename TPlan::DeltaNetQueryKey>
        && WeightQuantPolicy<typename TPlan::DeltaNetValueGateOutput>
        && WeightQuantPolicy<typename TPlan::DeltaNetGating>;

    /**
     * @brief The section 5 allocation: 25.62 B resident weights at 2.90 bits average, 8.65 GiB.
     *
     * | Role                      | Params  | Policy                 | Bits/w | GiB   |
     * |---------------------------|---------|------------------------|--------|-------|
     * | EmbeddingTable            | 1.271 B | BF16, host-resident    | -      | 0.000 |
     * | FeedForwardGateUp         | 11.408 B| PerGroupCodebook2<32>  | 2.5    | 3.320 |
     * | FeedForwardDown           | 5.704 B | PerGroupCodebook3<64>  | 3.25   | 2.158 |
     * | DeltaNetValueGateOutput   | 4.531 B | PerGroupCodebook2<32>  | 2.5    | 1.319 |
     * | DeltaNetQueryKey          | 1.007 B | PerGroupCodebook3<64>  | 3.25   | 0.381 |
     * | DeltaNetGating            | 0.024 B | BF16 -- never quantized| 16     | 0.045 |
     * | QkvProjection             | 1.678 B | PerGroupFp4<128>       | 4.125  | 0.806 |
     * | OutputProjection          | (above) | PerGroupFp4<128>       | 4.125  | (above)|
     * | LanguageModelHead         | 1.271 B | PerGroupFp4<128>       | 4.125  | 0.610 |
     *
     * Three assignments are not "biggest group gets fewest bits" and are load-bearing:
     *
     *  - DeltaNetGating is `NoWeightQuant`. beta and decay drive the forget gate, where
     *    error compounds EXPONENTIALLY over the sequence rather than linearly. They are
     *    0.1% of parameters; protecting them costs 0.045 GiB.
     *  - FeedForwardDown sits a half-step above the gate/up pair because down_proj reads
     *    post-SwiGLU activations, which carry the heavy outliers.
     *  - EmbeddingTable is `NoWeightQuant` because the table is host-resident: it is a
     *    gather, never a multiply, so it costs 10 KiB per token over PCIe and no VRAM.
     *    The policy states the STORAGE FORMAT, and BF16 is what the host holds; residency
     *    is the transformer's decision (item 6), not the policy's.
     *
     * The full-attention roles collapse to one policy on purpose. Section 5 puts q (with
     * its output-gate half), k, v and o all at FP4, so the fused qkv projection needs no
     * split; the per-projection divergence in this model is between BLOCK KINDS and between
     * the feed-forward pair, not inside the attention block.
     */
    export struct QwenPrecisionPlan
    {
        using QkvProjection = PerGroupFp4<128>;
        using OutputProjection = PerGroupFp4<128>;

        using FeedForwardGateUp = PerGroupCodebook2<32>;
        using FeedForwardDown = PerGroupCodebook3<64>;

        using DeltaNetQueryKey = PerGroupCodebook3<64>;
        using DeltaNetValueGateOutput = PerGroupCodebook2<32>;
        using DeltaNetGating = NoWeightQuant;

        using EmbeddingTable = NoWeightQuant;
        using LanguageModelHead = PerGroupFp4<128>;
    };

    /**
     * @brief One policy across every Qwen role, DeltaNet included.
     *
     * The generic UniformPrecisionPlan cannot name the DeltaNet roles -- they are this
     * family's, and a block kind must not be able to invalidate the plans of blocks that
     * already work -- so the family that adds roles adds its own uniform lift. Inheriting
     * keeps the shared roles defined once: a role added to the generic plan arrives here.
     *
     * Note this does NOT replace the generic lift for the attention block. That block
     * requires only the shared roles, so `QwenAttentionBlock<Cuda, BF16, PerGroupFp4<128>>`
     * still resolves through PrecisionPlanFor and stays a valid spelling.
     */
    export template<WeightQuantPolicy TPolicy>
        struct QwenUniformPrecisionPlan : UniformPrecisionPlan<TPolicy>
    {
        using DeltaNetQueryKey = TPolicy;
        using DeltaNetValueGateOutput = TPolicy;
        using DeltaNetGating = TPolicy;
    };

    /**
     * @brief The reference-precision plan: everything BF16, nothing quantized.
     *
     * Phase 4 gates the chassis against the HF reference at BF16 before Phase 5 applies the
     * allocation above, so both must be spellable at a declaration site. Named here so the
     * two arms of that comparison read as siblings rather than as a plan and a bare policy.
     */
    export using QwenReferencePrecisionPlan = QwenUniformPrecisionPlan<NoWeightQuant>;

    /**
     * @brief The Section 5 oracle: FP4 across the body, everything else as the plan has it.
     *
     * The near-lossless arm of the Phase 5 quality gate. 4.125 bits everywhere the plan
     * quantizes, 12.31 GiB of weights -- past the 12 GiB target card and onto the 16 GiB one,
     * which is what that card is for ("The 16 GiB oracle").
     *
     * NOT `QwenUniformPrecisionPlan<PerGroupFp4<128>>`, and the difference is the point. A
     * true uniform lift would also quantize the two roles the plan deliberately leaves alone,
     * and both refuse: `TokenEmbedding` static-asserts that a table policy is per-channel and
     * that a host-resident table is stored at compute precision. Those refusals are correct
     * -- the table is a gather over PCIe, never a multiply -- and the measurement wants them
     * anyway. Two arms that differ in their embedding AND their body do not attribute a
     * perplexity gap to either.
     *
     * DeltaNetGating stays BF16 for the same reason it does in the plan: error in the forget
     * gate compounds exponentially over the sequence, so quantizing 0.1% of the parameters
     * would put the oracle's own quality in question.
     */
    export struct QwenOraclePrecisionPlan
    {
        using QkvProjection = PerGroupFp4<128>;
        using OutputProjection = PerGroupFp4<128>;

        using FeedForwardGateUp = PerGroupFp4<128>;
        using FeedForwardDown = PerGroupFp4<128>;

        using DeltaNetQueryKey = PerGroupFp4<128>;
        using DeltaNetValueGateOutput = PerGroupFp4<128>;
        using DeltaNetGating = NoWeightQuant;

        using EmbeddingTable = NoWeightQuant;
        using LanguageModelHead = PerGroupFp4<128>;
    };

    namespace QwenPlanDetail
    {
        // Bool selector rather than a constrained partial specialization, for the reason
        // PrecisionPlan.ixx records against the generic one: the constrained form resolves
        // differently across MSVC/Clang/GCC and this is on every Qwen block's path.
        template<typename T, bool kIsBarePolicy>
        struct QwenPrecisionPlanSelector
        {
            using type = T;
        };

        template<typename T>
        struct QwenPrecisionPlanSelector<T, true>
        {
            using type = QwenUniformPrecisionPlan<T>;
        };
    }

    /**
     * @brief The Qwen lift: accepts a plan or a bare policy, and yields a plan WITH the
     * DeltaNet roles.
     *
     * The generic `PrecisionPlanFor` cannot serve the DeltaNet block, because the generic
     * uniform plan deliberately does not name this family's roles -- a block kind must not
     * be able to invalidate plans that already work. So the family that adds roles adds the
     * lift that carries them, and `QwenDeltaNetBlock<Cuda, BF16, PerGroupFp4<128>>` stays a
     * valid spelling alongside `<..., QwenPrecisionPlan>`.
     */
    export template<typename T>
        using QwenPrecisionPlanFor =
            typename QwenPlanDetail::QwenPrecisionPlanSelector<T, WeightQuantPolicy<T>>::type;

    static_assert( DeltaNetPrecisionRoles<QwenPrecisionPlanFor<NoWeightQuant>> );
    static_assert( FeedForwardPrecisionRoles<QwenPrecisionPlanFor<NoWeightQuant>> );
    static_assert( DeltaNetPrecisionRoles<QwenPrecisionPlanFor<QwenPrecisionPlan>> );

    // The plan must satisfy every concept the family's blocks and network require. Asserted
    // here rather than only at the block, so a role dropped from the table is caught in the
    // file that owns the table.
    static_assert( DecoderPrecisionPlan<QwenPrecisionPlan> );
    static_assert( DeltaNetPrecisionRoles<QwenPrecisionPlan> );
    static_assert( EmbeddingPrecisionRole<QwenPrecisionPlan> );
    static_assert( LanguageModelHeadRole<QwenPrecisionPlan> );

    static_assert( DecoderPrecisionPlan<QwenReferencePrecisionPlan> );
    static_assert( DeltaNetPrecisionRoles<QwenReferencePrecisionPlan> );
    static_assert( EmbeddingPrecisionRole<QwenReferencePrecisionPlan> );
    static_assert( LanguageModelHeadRole<QwenReferencePrecisionPlan> );

    static_assert( DecoderPrecisionPlan<QwenOraclePrecisionPlan> );
    static_assert( DeltaNetPrecisionRoles<QwenOraclePrecisionPlan> );
    static_assert( EmbeddingPrecisionRole<QwenOraclePrecisionPlan> );
    static_assert( LanguageModelHeadRole<QwenOraclePrecisionPlan> );
}
