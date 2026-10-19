/**
 * @file Qwen.Fp4Codebook3PrecisionPlan.cpp
 * @brief Qwen 3.8's fp4-cb3 allocation, checked at compile time: every role at the format its measurement chose.
 *
 * CPU-only and device-free, so it runs in the CPU-only CI configuration.
 */

#include <gtest/gtest.h>

#include <type_traits>

import Dnn.TensorDataType;
import Dnn.Quantization.Weight.Policies;
import Dnn.Quantization.Weight.PrecisionPlan;
import Dnn.Components.QwenPrecisionPlan;
import Dnn.Components.QwenFp4Codebook3PrecisionPlan;

namespace Mila::Tests::Dnn::Components::Transformers::Qwen
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Quant::Weight;

    using Plan = QwenFp4Codebook3PrecisionPlan;

    static_assert( DecoderPrecisionPlan<Plan> );
    static_assert( DeltaNetPrecisionRoles<Plan> );
    static_assert( EmbeddingPrecisionRole<Plan> );
    static_assert( LanguageModelHeadRole<Plan> );

    static_assert( std::is_same_v<Plan::QkvProjection, PerGroupFp4<128>> );
    static_assert( std::is_same_v<Plan::OutputProjection, PerGroupFp4<128>> );
    static_assert( std::is_same_v<Plan::FeedForwardDown, PerGroupCodebook3<64>> );
    static_assert( std::is_same_v<Plan::DeltaNetQueryKey, PerGroupCodebook3<64>> );
    static_assert( std::is_same_v<Plan::DeltaNetGating, NoWeightQuant> );
    static_assert( std::is_same_v<Plan::EmbeddingTable, NoWeightQuant> );
    static_assert( std::is_same_v<Plan::LanguageModelHead, PerGroupFp4<128>> );

    // The two roles this allocation exists to protect. At two bits the gate/up pair drops one closing bracket of a
    // nested pair in code and v, gate and o cost the most of the cuts measured; neither may drift below FP4.
    static_assert( std::is_same_v<Plan::FeedForwardGateUp, PerGroupFp4<128>> );
    static_assert( std::is_same_v<Plan::DeltaNetValueGateOutput, PerGroupFp4<128>> );

    // It is a different allocation from section 5's compact plan, not a renaming of it: a load of one must never
    // build the other's network.
    static_assert( !std::is_same_v<Plan::FeedForwardGateUp, QwenPrecisionPlan::FeedForwardGateUp> );

    TEST( QwenFp4Codebook3PrecisionPlan, CarriesEveryRoleAtItsMeasuredFormat )
    {
        SUCCEED() << "asserted at compile time";
    }
}
