/**
 * @file QwenModel.HuggingFaceReference.Metal.cpp
 * @brief Qwen 3.8's four-layer packed build on Metal, in FP16 and BF16, against HuggingFace's FP32 run on its weights as Mila
 *        runs them.
 *
 * MetalBackend.md Phase 6's gate for Qwen, written in Notebooks/MetalQwen.md before the run. Needs the build and the capture,
 * so it never runs in CI.
 */

#include <gtest/gtest.h>

import Mila;
import std;

#include "Qwen38L4Reference.h"

namespace Mila::Tests::Dnn::Models::Qwen38L4
{
    namespace
    {
        // Each the further of two accepted BF16 runs of these weights from HuggingFace FP32 on the same capture, 2026-11-07:
        // HuggingFace's own BF16 run (2.844e-3, +5.076e-4, 1.930e-5) and Mila's on CUDA (3.884e-3, +8.226e-4, 5.869e-5), which is
        // the further on each (Notebooks/MetalQwen.md 1.2).
        constexpr double kWorstRelativeL2 = 3.884e-3;
        constexpr double kNegativeLogLikelihoodDifference = 8.226e-4;
        constexpr double kPiecesMeanKl = 5.869e-5;

        template<typename TPrecisionConstant>
        class QwenModelHuggingFaceReferenceMetal : public ::testing::Test
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
                    GTEST_SKIP() << "Qwen 3.8 four-layer build or capture not present at: " << referencePath().parent_path().string();
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

        TYPED_TEST_SUITE( QwenModelHuggingFaceReferenceMetal, Precisions, PrecisionNames );
    }

    TYPED_TEST( QwenModelHuggingFaceReferenceMetal, Logits )
    {
        const LogitsResult result = measureLogits<DeviceType::Metal, TypeParam::value>( Device::Metal( 0 ) );

        EXPECT_EQ( result.non_finite, 0 );
        EXPECT_LE( result.worst_relative_l2, kWorstRelativeL2 );
        EXPECT_TRUE( result.argmax_disagreements.empty() ) << "first at step " << result.argmax_disagreements.front();
    }

    TYPED_TEST( QwenModelHuggingFaceReferenceMetal, SequenceLogLikelihoodAndRanges )
    {
        const SequenceResult result = measureSequence<DeviceType::Metal, TypeParam::value>( Device::Metal( 0 ) );

        EXPECT_NEAR( result.mean_negative_log_likelihood, result.reference_mean_negative_log_likelihood, kNegativeLogLikelihoodDifference );

        for ( std::size_t layer = 0; layer < result.layers.size(); ++layer )
        {
            EXPECT_EQ( result.layers[ layer ].non_finite, 0 ) << "layer " << layer;
        }
    }

    TYPED_TEST( QwenModelHuggingFaceReferenceMetal, PiecesKlDivergence )
    {
        const PiecesResult result = measurePieces<DeviceType::Metal, TypeParam::value>( Device::Metal( 0 ) );

        EXPECT_EQ( result.non_finite, 0 );
        EXPECT_LE( result.mean_kl, kPiecesMeanKl );
    }

    TYPED_TEST( QwenModelHuggingFaceReferenceMetal, Greedy )
    {
        const LogitsResult logits = measureLogits<DeviceType::Metal, TypeParam::value>( Device::Metal( 0 ) );
        const std::vector<std::int32_t> generated = generateGreedy<DeviceType::Metal, TypeParam::value>( Device::Metal( 0 ), 32 );
        const GreedyAgreement agreement = greedyAgreement( TypeParam::value, generated, logits.largest_error,
            capturedConfig<DeviceType::Metal, TypeParam::value>().getVocabSize() );

        EXPECT_TRUE( agreement.agreed == 32 || agreement.diverged_at_a_tie ) << "agreed for " << agreement.agreed;
    }

    /// The decoded tokens one position early: the logits check must miss.
    TYPED_TEST( QwenModelHuggingFaceReferenceMetal, ForcedFailure_ShiftedPositionsMissTheBar )
    {
        const LogitsResult result = measureLogits<DeviceType::Metal, TypeParam::value>( Device::Metal( 0 ), -1 );

        EXPECT_GT( result.worst_relative_l2, kWorstRelativeL2 );
    }
}
