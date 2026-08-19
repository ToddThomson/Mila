/**
 * @file PrecisionPlan.Cpu.cpp
 * @brief The per-role precision plan's contract, checked where it lives: at compile time.
 *
 * Every claim this file makes is a static_assert, and the runtime bodies exist only so the
 * suite reports them. A plan is a compile-time construct with no state and no behaviour, so
 * a test that ran would be testing nothing the compiler had not already decided.
 *
 * The negative cases are the point. A concept that accepts everything is indistinguishable
 * from no concept at all, so each role is checked to REJECT a plan that omits it -- via
 * `requires` in a static_assert, since the alternative (a plan that fails to compile) cannot
 * be expressed as a passing test.
 *
 * CPU-only and device-free, so it runs in the CPU-only CI configuration.
 */

#include <gtest/gtest.h>

#include <type_traits>

import Dnn.Quantization.Weight.Policies;
import Dnn.Quantization.Weight.PrecisionPlan;

namespace Mila::Tests::Dnn::Quantization
{
    using namespace Mila::Dnn::Quant::Weight;

    namespace
    {
        // The Qwen3.8 allocation from Specifications/Qwen3.8.md section 5, for the
        // full-attention block: q (with its output gate), k, v and o are all FP4, so the
        // fused QKV projection carries one policy; the feed-forward pair is where the
        // sub-4-bit formats land, and gate/up is a half-step below down.
        struct QwenAttentionLayerPlan
        {
            using QkvProjection = PerGroupFp4<128>;
            using OutputProjection = PerGroupFp4<128>;
            using FeedForwardGateUp = PerGroupCodebook2<32>;
            using FeedForwardDown = PerGroupCodebook3<64>;
            using EmbeddingTable = PerGroupFp4<128>;
        };

        // Names three of the four block roles. The omission is the test.
        struct PlanMissingFeedForwardDown
        {
            using QkvProjection = PerGroupFp4<128>;
            using OutputProjection = PerGroupFp4<128>;
            using FeedForwardGateUp = PerGroupCodebook2<32>;
        };

        // Every role present, but one names something that is not a weight policy. A
        // concept that only checked for the typedef's presence would accept this.
        struct PlanWithANonPolicyRole
        {
            using QkvProjection = PerGroupFp4<128>;
            using OutputProjection = PerGroupFp4<128>;
            using FeedForwardGateUp = PerGroupCodebook2<32>;
            using FeedForwardDown = int;
        };
    }

    // -- a complete plan satisfies the block concept, and its roles keep their policies ---

    static_assert( DecoderPrecisionPlan<QwenAttentionLayerPlan> );
    static_assert( EmbeddingPrecisionRole<QwenAttentionLayerPlan> );

    static_assert( std::is_same_v<QwenAttentionLayerPlan::FeedForwardGateUp, PerGroupCodebook2<32>> );
    static_assert( std::is_same_v<QwenAttentionLayerPlan::FeedForwardDown, PerGroupCodebook3<64>> );
    static_assert( std::is_same_v<QwenAttentionLayerPlan::QkvProjection, PerGroupFp4<128>> );

    // The two feed-forward roles must not collapse to one type, or the plan is expressing a
    // uniform allocation while claiming to express a mixed one.
    static_assert( !std::is_same_v<QwenAttentionLayerPlan::FeedForwardGateUp,
        QwenAttentionLayerPlan::FeedForwardDown> );

    // -- an incomplete plan is REJECTED, and only over the role it omits -----------------

    static_assert( !DecoderPrecisionPlan<PlanMissingFeedForwardDown> );
    static_assert( !FeedForwardPrecisionRoles<PlanMissingFeedForwardDown> );
    // Still satisfies the roles it does name, so the failure a block reports points at the
    // missing role rather than at the plan wholesale.
    static_assert( AttentionPrecisionRoles<PlanMissingFeedForwardDown> );

    static_assert( !DecoderPrecisionPlan<PlanWithANonPolicyRole> );

    // A bare policy is not a plan. Without the lifting below, passing one where a plan is
    // expected would be a compile error rather than the uniform allocation it means.
    static_assert( !DecoderPrecisionPlan<PerGroupFp4<128>> );

    // -- a bare policy lifts to a uniform plan, so existing spellings stay valid ----------

    using LiftedFp4 = PrecisionPlanFor<PerGroupFp4<128>>;

    static_assert( DecoderPrecisionPlan<LiftedFp4> );
    static_assert( EmbeddingPrecisionRole<LiftedFp4> );
    static_assert( std::is_same_v<LiftedFp4::QkvProjection, PerGroupFp4<128>> );
    static_assert( std::is_same_v<LiftedFp4::FeedForwardDown, PerGroupFp4<128>> );

    static_assert( DecoderPrecisionPlan<PrecisionPlanFor<NoWeightQuant>> );
    static_assert( std::is_same_v<PrecisionPlanFor<NoWeightQuant>::FeedForwardGateUp, NoWeightQuant> );

    // Lifting a plan is the identity: a block naming its parameter through PrecisionPlanFor
    // must not wrap a plan that already is one.
    static_assert( std::is_same_v<PrecisionPlanFor<QwenAttentionLayerPlan>, QwenAttentionLayerPlan> );

    TEST( PrecisionPlanCpu, CompletePlanSatisfiesTheDecoderConcept )
    {
        SUCCEED() << "asserted at compile time";
    }

    TEST( PrecisionPlanCpu, IncompletePlanIsRejectedOverTheRoleItOmits )
    {
        SUCCEED() << "asserted at compile time";
    }

    TEST( PrecisionPlanCpu, BarePolicyLiftsToAUniformPlan )
    {
        SUCCEED() << "asserted at compile time";
    }
}
