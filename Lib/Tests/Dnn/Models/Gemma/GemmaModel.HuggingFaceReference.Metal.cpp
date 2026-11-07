/**
 * @file GemmaModel.HuggingFaceReference.Metal.cpp
 * @brief Gemma 4 12B Instruct in Q4_0 on Metal, in FP16 and BF16, against HuggingFace's FP32 run on the package's decoded weights.
 *
 * MetalBackend.md Phase 6's gate, written in Notebooks/MetalGemma.md before the run. Needs the weights and the capture, so it
 * never runs in CI.
 */

#include <gtest/gtest.h>

import Mila;
import std;

#include "Gemma12bReference.h"

namespace Mila::Tests::Dnn::Models::Gemma12b
{
    namespace
    {
        // Each the further of two accepted BF16 runs of these weights from HuggingFace FP32 on the same capture, 2026-11-06:
        // HuggingFace's own BF16 run (2.711e-2, 2.540e-2, -3.984e-3, 7.608e-4) and Mila's on CUDA (2.399e-2, 3.339e-1, +2.657e-3,
        // 1.869e-3). Across the window's edge CUDA's worst is two rows far from every other and from HuggingFace BF16, so that
        // bar is HuggingFace BF16's (Notebooks/MetalGemma.md 1.3).
        constexpr double kWorstRelativeL2 = 2.711e-2;
        constexpr double kWindowEdgeWorstRelativeL2 = 2.540e-2;
        constexpr double kNegativeLogLikelihoodDifference = 3.984e-3;
        constexpr double kPiecesMeanKl = 1.869e-3;

        template<typename TPrecisionConstant>
        class GemmaModelHuggingFaceReferenceMetal : public ::testing::Test
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
                    GTEST_SKIP() << "Gemma 4 12B Q4_0 weights or capture not present at: " << referencePath().parent_path().string();
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

        TYPED_TEST_SUITE( GemmaModelHuggingFaceReferenceMetal, Precisions, PrecisionNames );
    }

    TYPED_TEST( GemmaModelHuggingFaceReferenceMetal, Logits )
    {
        const LogitsResult result = measureLogits<DeviceType::Metal, TypeParam::value>( Device::Metal( 0 ) );

        EXPECT_EQ( result.non_finite, 0 );
        EXPECT_LE( result.worst_relative_l2, kWorstRelativeL2 );
        EXPECT_TRUE( result.argmax_disagreements.empty() ) << "first at step " << result.argmax_disagreements.front();
    }

    TYPED_TEST( GemmaModelHuggingFaceReferenceMetal, SequenceLogLikelihoodAndRanges )
    {
        const SequenceResult result = measureSequence<DeviceType::Metal, TypeParam::value>( Device::Metal( 0 ) );

        EXPECT_NEAR( result.mean_negative_log_likelihood, result.reference_mean_negative_log_likelihood, kNegativeLogLikelihoodDifference );

        for ( std::size_t layer = 0; layer < result.layers.size(); ++layer )
        {
            EXPECT_EQ( result.layers[ layer ].non_finite, 0 ) << "layer " << layer;
        }
    }

    TYPED_TEST( GemmaModelHuggingFaceReferenceMetal, PiecesKlDivergence )
    {
        const PiecesResult result = measurePieces<DeviceType::Metal, TypeParam::value>( Device::Metal( 0 ) );

        EXPECT_EQ( result.non_finite, 0 );
        EXPECT_LE( result.mean_kl, kPiecesMeanKl );
    }

    TYPED_TEST( GemmaModelHuggingFaceReferenceMetal, AcrossTheWindowEdge )
    {
        const LogitsResult result = measureWindowEdge<DeviceType::Metal, TypeParam::value>( Device::Metal( 0 ) );

        EXPECT_EQ( result.non_finite, 0 );
        EXPECT_LE( result.worst_relative_l2, kWindowEdgeWorstRelativeL2 );
        EXPECT_TRUE( result.argmax_disagreements.empty() ) << "first at row " << result.argmax_disagreements.front();
    }

    TYPED_TEST( GemmaModelHuggingFaceReferenceMetal, Greedy )
    {
        const LogitsResult logits = measureLogits<DeviceType::Metal, TypeParam::value>( Device::Metal( 0 ) );
        const std::vector<std::int32_t> generated = generateGreedy<DeviceType::Metal, TypeParam::value>( Device::Metal( 0 ), 32 );
        const GreedyAgreement agreement = greedyAgreement( TypeParam::value, generated, logits.largest_error,
            capturedConfig<DeviceType::Metal, TypeParam::value>().getVocabSize() );

        EXPECT_TRUE( agreement.diverged_at_a_tie ) << "diverged after " << agreement.agreed << " tokens";
    }

    // The gate can fail: each decoded token one position early must miss HuggingFace.
    TYPED_TEST( GemmaModelHuggingFaceReferenceMetal, DecodedAtTheWrongPosition_MissesHuggingFace )
    {
        const LogitsResult result = measureLogits<DeviceType::Metal, TypeParam::value>( Device::Metal( 0 ), -1 );

        EXPECT_TRUE( result.worst_relative_l2 > kWorstRelativeL2 || !result.argmax_disagreements.empty() );
    }
}
