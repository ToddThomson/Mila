/**
 * @file LlamaModel.HuggingFaceReference.Metal.cpp
 * @brief Llama 3.2 3B Instruct on Metal, in FP16 and BF16 on its BF16 weights, against HuggingFace's FP32 run.
 *
 * MetalBackend.md Phase 4's gate, written in Notebooks/MetalFirstModel.md before the run. Needs the BF16 weights and the
 * capture, so it never runs in CI.
 */

#include <gtest/gtest.h>

import Mila;
import std;

#include "Llama32Reference.h"

namespace Mila::Tests::Dnn::Models::Llama32
{
    namespace
    {
        // The further of two accepted BF16 runs of this model from HuggingFace FP32, on the same capture: HuggingFace's
        // own BF16 (2.185e-2 and +8.97e-4) and Mila's on CUDA (1.825e-2 and -5.51e-4).
        constexpr double kWorstRelativeL2 = 2.185e-2;
        constexpr double kNegativeLogLikelihoodDifference = 8.97e-4;

        template<typename TPrecisionConstant>
        class LlamaModelHuggingFaceReferenceMetal : public ::testing::Test
        {
        protected:
            void SetUp() override
            {
                if ( getDeviceCount( DeviceType::Metal ) == 0 )
                {
                    GTEST_SKIP() << "No Metal device available";
                }

                if ( !captureExists() )
                {
                    GTEST_SKIP() << "Llama 3.2 3B weights or capture not present at: " << referencePath().parent_path().string();
                }
            }
        };

        using Precisions = ::testing::Types<
            std::integral_constant<TensorDataType, TensorDataType::FP16>,
            std::integral_constant<TensorDataType, TensorDataType::BF16>>;

        struct PrecisionNames
        {
            template<typename T>
            static std::string GetName( int )
            {
                return T::value == TensorDataType::FP16 ? "Fp16" : "Bf16";
            }
        };

        TYPED_TEST_SUITE( LlamaModelHuggingFaceReferenceMetal, Precisions, PrecisionNames );
    }

    TYPED_TEST( LlamaModelHuggingFaceReferenceMetal, Logits )
    {
        const LogitsResult result = measureLogits<DeviceType::Metal, TypeParam::value>( Device::Metal( 0 ) );

        EXPECT_EQ( result.non_finite, 0 );
        EXPECT_LE( result.worst_relative_l2, kWorstRelativeL2 );
        EXPECT_TRUE( result.argmax_disagreements.empty() ) << "first at step " << result.argmax_disagreements.front();
    }

    TYPED_TEST( LlamaModelHuggingFaceReferenceMetal, SequenceLogLikelihoodAndRanges )
    {
        const SequenceResult result = measureSequence<DeviceType::Metal, TypeParam::value>( Device::Metal( 0 ) );

        EXPECT_NEAR( result.mean_negative_log_likelihood, result.reference_mean_negative_log_likelihood, kNegativeLogLikelihoodDifference );

        for ( std::size_t layer = 0; layer < result.layers.size(); ++layer )
        {
            EXPECT_EQ( result.layers[ layer ].non_finite, 0 ) << "layer " << layer;
        }
    }

    TYPED_TEST( LlamaModelHuggingFaceReferenceMetal, Greedy )
    {
        const LogitsResult logits = measureLogits<DeviceType::Metal, TypeParam::value>( Device::Metal( 0 ) );
        const std::vector<std::int32_t> generated = generateGreedy<DeviceType::Metal, TypeParam::value>( Device::Metal( 0 ), 32 );
        const GreedyAgreement agreement = greedyAgreement( TypeParam::value, generated, logits.largest_error,
            capturedConfig<DeviceType::Metal, TypeParam::value>().getVocabSize() );

        EXPECT_TRUE( agreement.diverged_at_a_tie ) << "diverged after " << agreement.agreed << " tokens";
    }

    // The gate can fail: each decoded token one position early must miss HuggingFace.
    TYPED_TEST( LlamaModelHuggingFaceReferenceMetal, DecodedAtTheWrongPosition_MissesHuggingFace )
    {
        const LogitsResult result = measureLogits<DeviceType::Metal, TypeParam::value>( Device::Metal( 0 ), -1 );

        EXPECT_TRUE( result.worst_relative_l2 > kWorstRelativeL2 || !result.argmax_disagreements.empty() );
    }
}
