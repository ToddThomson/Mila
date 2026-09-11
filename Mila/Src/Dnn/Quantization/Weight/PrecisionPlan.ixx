// Mila - Dnn/Quantization/Weight/PrecisionPlan.ixx
// Per-role weight quantization: a named struct of policy typedefs, one per tensor role.
//
// Consumers: decoder blocks and transformers that assign DIFFERENT policies to different
// projections. A family with one uniform policy needs nothing here -- passing a bare
// WeightQuantPolicy keeps working unchanged, because PrecisionPlanFor lifts it.

module;
#include <concepts>

export module Dnn.Quantization.Weight.PrecisionPlan;

import Dnn.Quantization.Weight.Policies;

namespace Mila::Dnn::Quant::Weight
{
    // -------------------------------------------------------------------------
    // Why a struct of named typedefs
    //
    // Per-projection precision cannot ride the single TWeightQuantization parameter, which
    // applies uniformly to every Linear in a block. The alternative -- widening the block's
    // template parameter list to one policy per projection -- compiles and guarantees the
    // same things, but the declaration becomes an unlabelled positional list that no reader
    // can check against the model's memory budget.
    //
    //   using QwenAttentionLayer = QwenAttentionBlock<DeviceType::Cuda, TensorDataType::BF16,
    //       QwenPrecisionPlan>;
    //
    // versus
    //
    //   using QwenAttentionLayer = QwenAttentionBlock<DeviceType::Cuda, TensorDataType::BF16,
    //       PerGroupFp4<128>, PerGroupFp4<128>, PerGroupCodebook2<32>, PerGroupCodebook3<64>>;
    //
    // The property being preserved is that the memory strategy of the model is readable at
    // the declaration site. See Specifications/Qwen3.8.md section 6.
    // -------------------------------------------------------------------------

    // -------------------------------------------------------------------------
    // Role concepts
    //
    // Grouped by the structure that instantiates them rather than gathered into one
    // all-roles concept, so a block requires exactly the roles it builds and no more. A
    // block whose plan omits a role it instantiates fails its own requires-clause, naming
    // the block -- which is the diagnostic this design exists to produce, in place of a
    // silent fallback to some default precision.
    //
    // Each `requires { typename ... }` guard precedes the policy check it protects: the
    // conjunction short-circuits, so a plan missing the typedef reports as unsatisfied
    // rather than failing substitution.
    //
    // Roles are named after the tensors they govern, matching the member and artifact
    // names already used across the tree (qkv_proj, o_proj, fc_gate_up, fc_down).
    // -------------------------------------------------------------------------

    /// Fused QKV projection and the attention output projection.
    export template<typename TPlan>
        concept AttentionPrecisionRoles =
        requires { typename TPlan::QkvProjection; typename TPlan::OutputProjection; }
        && WeightQuantPolicy<typename TPlan::QkvProjection>
        && WeightQuantPolicy<typename TPlan::OutputProjection>;

    /// Fused gate+up projection and the down projection of a gated feed-forward network.
    export template<typename TPlan>
        concept FeedForwardPrecisionRoles =
        requires { typename TPlan::FeedForwardGateUp; typename TPlan::FeedForwardDown; }
        && WeightQuantPolicy<typename TPlan::FeedForwardGateUp>
        && WeightQuantPolicy<typename TPlan::FeedForwardDown>;

    /**
     * @brief What an attention-plus-feed-forward decoder block requires.
     *
     * Every block kind in the tree today instantiates exactly these four projections. A
     * block with a different mixer -- a recurrent one, say -- composes its own concept from
     * its own roles rather than widening this one, so that adding a block kind cannot
     * invalidate the plans of the blocks that already work.
     */
    export template<typename TPlan>
        concept DecoderPrecisionPlan =
        AttentionPrecisionRoles<TPlan> && FeedForwardPrecisionRoles<TPlan>;

    /**
     * @brief The embedding table, required at the network level rather than the block.
     *
     * Deliberately separate: no block instantiates the table, so folding it into
     * DecoderPrecisionPlan would make blocks reject plans over a role they never build.
     */
    export template<typename TPlan>
        concept EmbeddingPrecisionRole =
        requires { typename TPlan::EmbeddingTable; }
        && WeightQuantPolicy<typename TPlan::EmbeddingTable>;

    /**
     * @brief The output (language-model head) table.
     *
     * Separate from EmbeddingTable because the two tables only share a policy when they
     * share an allocation. A tied family (Gemma) names one policy twice and loses nothing;
     * an untied one (Qwen 3.8, `tie_word_embeddings: false`) reads the head every decode
     * step and gathers one row from the table, so the two are priced against different
     * costs and land at different widths.
     */
    export template<typename TPlan>
        concept LanguageModelHeadRole =
        requires { typename TPlan::LanguageModelHead; }
        && WeightQuantPolicy<typename TPlan::LanguageModelHead>;

    // -------------------------------------------------------------------------
    // UniformPrecisionPlan
    // -------------------------------------------------------------------------

    /**
     * @brief Assigns one policy to every role.
     *
     * The bridge that keeps a bare policy a valid argument wherever a plan is expected, so
     * families with no per-role need are untouched by this header existing. A role added
     * here later must be added to this struct too -- that is the definition of uniform, not
     * an oversight.
     */
    export template<WeightQuantPolicy TPolicy>
        struct UniformPrecisionPlan
    {
        using QkvProjection = TPolicy;
        using OutputProjection = TPolicy;
        using FeedForwardGateUp = TPolicy;
        using FeedForwardDown = TPolicy;
        using EmbeddingTable = TPolicy;
        using LanguageModelHead = TPolicy;
    };

    namespace Detail
    {
        // Selection on a bool rather than a constrained partial specialization of a
        // single-parameter primary. The constrained form is valid C++20 and reads better,
        // but it is the kind of construct that resolves differently across MSVC, Clang and
        // GCC, and this header is on the path of every family that adopts a plan.
        template<typename T, bool kIsBarePolicy>
        struct PrecisionPlanSelector
        {
            using type = T;
        };

        template<typename T>
        struct PrecisionPlanSelector<T, true>
        {
            using type = UniformPrecisionPlan<T>;
        };
    }

    /**
     * @brief Accepts either a plan or a bare policy, and yields a plan.
     *
     * A block names its policy parameter through this alias, which is what lets
     * `Block<Cuda, BF16, PerGroupFp4<128>>` and `Block<Cuda, BF16, QwenPrecisionPlan>` both
     * be valid spellings of the same parameter.
     */
    export template<typename T>
        using PrecisionPlanFor = typename Detail::PrecisionPlanSelector<T, WeightQuantPolicy<T>>::type;

    // A uniform plan must satisfy every role concept, or the lifting above silently stops
    // accepting bare policies at whichever block requires the role that was forgotten.
    static_assert( DecoderPrecisionPlan<UniformPrecisionPlan<NoWeightQuant>> );
    static_assert( EmbeddingPrecisionRole<UniformPrecisionPlan<NoWeightQuant>> );
    static_assert( LanguageModelHeadRole<UniformPrecisionPlan<NoWeightQuant>> );
    static_assert( DecoderPrecisionPlan<PrecisionPlanFor<NoWeightQuant>> );
}
