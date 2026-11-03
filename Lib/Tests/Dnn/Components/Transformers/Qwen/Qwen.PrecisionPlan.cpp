/**
 * @file Qwen.PrecisionPlan.cpp
 * @brief The Qwen 3.8 section 5 allocation, checked where it lives: at compile time.
 *
 * Companion to Dnn/Quantization/PrecisionPlan.Cpu.cpp, which covers the generic machinery.
 * This file covers the FAMILY's table: that every role the spec assigns is present and
 * assigned what the spec assigns it, and that the assignments which are load-bearing rather
 * than mechanical stay distinct from each other.
 *
 * Every claim is a static_assert; the runtime bodies exist only so the suite reports them. A
 * plan is a compile-time construct with no state and no behaviour, so a test that ran would
 * be testing nothing the compiler had not already decided.
 *
 * CPU-only and device-free, so it runs in the CPU-only CI configuration.
 */

#include <gtest/gtest.h>

#include <type_traits>

import Dnn.TensorDataType;
import Dnn.Quantization.Weight.Policies;
import Dnn.Quantization.Weight.PrecisionPlan;
import Dnn.Components.QwenPrecisionPlan;

namespace Mila::Tests::Dnn::Components::Transformers::Qwen
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Quant::Weight;

    // -- the plan satisfies every concept the family's structures require ----------------

    static_assert( DecoderPrecisionPlan<QwenPrecisionPlan> );
    static_assert( DeltaNetPrecisionRoles<QwenPrecisionPlan> );
    static_assert( EmbeddingPrecisionRole<QwenPrecisionPlan> );
    static_assert( LanguageModelHeadRole<QwenPrecisionPlan> );

    // -- the section 5 table, role by role -----------------------------------------------

    static_assert( std::is_same_v<QwenPrecisionPlan::QkvProjection, PerGroupFp4<128>> );
    static_assert( std::is_same_v<QwenPrecisionPlan::OutputProjection, PerGroupFp4<128>> );
    static_assert( std::is_same_v<QwenPrecisionPlan::FeedForwardGateUp, PerGroupCodebook2<32>> );
    static_assert( std::is_same_v<QwenPrecisionPlan::FeedForwardDown, PerGroupCodebook3<64>> );
    static_assert( std::is_same_v<QwenPrecisionPlan::DeltaNetQueryKey, PerGroupCodebook3<64>> );
    static_assert( std::is_same_v<QwenPrecisionPlan::DeltaNetValueGateOutput, PerGroupCodebook2<32>> );
    static_assert( std::is_same_v<QwenPrecisionPlan::DeltaNetGating, NoWeightQuant> );
    static_assert( std::is_same_v<QwenPrecisionPlan::EmbeddingTable, NoWeightQuant> );
    static_assert( std::is_same_v<QwenPrecisionPlan::LanguageModelHead, PerGroupFp4<128>> );

    // -- the distinctions the allocation rests on ----------------------------------------

    // The feed-forward pair is where 64% of the model lives and where the two sub-4-bit
    // formats diverge. Collapsing them would express a uniform allocation while claiming a
    // mixed one, and would miss the 12 GiB budget by section 4's third.
    static_assert( !std::is_same_v<QwenPrecisionPlan::FeedForwardGateUp,
        QwenPrecisionPlan::FeedForwardDown> );

    // beta and decay drive the forget gate, where error compounds exponentially over the
    // sequence. They are the ONE role in the model that must not be quantized at all.
    static_assert( !QwenPrecisionPlan::DeltaNetGating::kIsQuantized );

    // Untied tables at different widths -- the case Gemma's single TableQuantizationPolicy
    // cannot express, and the reason LanguageModelHeadRole is a role of its own.
    static_assert( !std::is_same_v<QwenPrecisionPlan::EmbeddingTable,
        QwenPrecisionPlan::LanguageModelHead> );

    // The scale dtype is part of the bit width the section 5 table quotes: an FP16 scale per
    // group of 32 is the 0.5 bits that make PerGroupCodebook2<32> cost 2.5 rather than 2.
    static_assert( QwenPrecisionPlan::FeedForwardGateUp::kScaleDtype == TensorDataType::FP16 );
    static_assert( QwenPrecisionPlan::FeedForwardGateUp::kQuantizationGroupSize == 32 );
    static_assert( QwenPrecisionPlan::FeedForwardDown::kQuantizationGroupSize == 64 );

    // -- the reference plan is a plan, not a bare policy ---------------------------------

    static_assert( DecoderPrecisionPlan<QwenReferencePrecisionPlan> );
    static_assert( DeltaNetPrecisionRoles<QwenReferencePrecisionPlan> );
    static_assert( LanguageModelHeadRole<QwenReferencePrecisionPlan> );
    static_assert( !QwenReferencePrecisionPlan::FeedForwardGateUp::kIsQuantized );
    static_assert( !QwenReferencePrecisionPlan::DeltaNetGating::kIsQuantized );

    // The generic uniform lift reaches the shared roles but NOT the DeltaNet ones -- which
    // is the property that keeps a new block kind from invalidating existing plans, and the
    // reason the family carries its own lift.
    static_assert( DecoderPrecisionPlan<PrecisionPlanFor<PerGroupFp4<128>>> );
    static_assert( !DeltaNetPrecisionRoles<PrecisionPlanFor<PerGroupFp4<128>>> );
    static_assert( DeltaNetPrecisionRoles<QwenUniformPrecisionPlan<PerGroupFp4<128>>> );

    TEST( QwenPrecisionPlan, CarriesEverySectionFiveRole )
    {
        SUCCEED() << "asserted at compile time";
    }

    TEST( QwenPrecisionPlan, KeepsTheAllocationsThatMustDiffer )
    {
        SUCCEED() << "asserted at compile time";
    }

    TEST( QwenPrecisionPlan, ReferencePlanIsUnquantizedAcrossEveryRole )
    {
        SUCCEED() << "asserted at compile time";
    }
}
