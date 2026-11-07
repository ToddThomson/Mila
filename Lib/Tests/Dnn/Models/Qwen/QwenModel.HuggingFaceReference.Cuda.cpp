/**
 * @file QwenModel.HuggingFaceReference.Cuda.cpp
 * @brief Qwen 3.8's four-layer packed build at BF16 on CUDA against HuggingFace's FP32 run on its weights as Mila runs them:
 *        one of the two distances Metal is held to.
 *
 * Mila's shipped path on the same weights and capture, so Metal's bars are a measurement and not a guess
 * (Notebooks/MetalQwen.md section 1.2). Each check reports its numbers; CUDA is an accepted run, so only what any run must
 * hold is asserted. Needs the weights and the capture, so it never runs in CI.
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <filesystem>
#include <format>
#include <iostream>
#include <memory>
#include <stop_token>
#include <string>
#include <vector>

import Mila;

#include "Qwen38L4Reference.h"

namespace Mila::Tests::Dnn::Models::Qwen38L4
{
    class QwenModelHuggingFaceReferenceCudaTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            if ( getDeviceCount( DeviceType::Cuda ) == 0 )
            {
                GTEST_SKIP() << "No CUDA device available";
            }

            if ( !captureExists() )
            {
                GTEST_SKIP() << "Qwen 3.8 four-layer build or capture not present at: " << referencePath().parent_path().string();
            }
        }
    };

    TEST_F( QwenModelHuggingFaceReferenceCudaTests, Plan_Logits )
    {
        const LogitsResult result = measureLogits<DeviceType::Cuda, TensorDataType::BF16>( Device::Cuda( 0 ) );

        EXPECT_EQ( result.non_finite, 0 );
    }

    TEST_F( QwenModelHuggingFaceReferenceCudaTests, Plan_SequenceLogLikelihoodAndRanges )
    {
        const SequenceResult result = measureSequence<DeviceType::Cuda, TensorDataType::BF16>( Device::Cuda( 0 ) );

        for ( const LayerRange& layer : result.layers )
        {
            EXPECT_EQ( layer.non_finite, 0 );
        }
    }

    TEST_F( QwenModelHuggingFaceReferenceCudaTests, Plan_PiecesKlDivergence )
    {
        const PiecesResult result = measurePieces<DeviceType::Cuda, TensorDataType::BF16>( Device::Cuda( 0 ) );

        EXPECT_EQ( result.non_finite, 0 );
    }

    TEST_F( QwenModelHuggingFaceReferenceCudaTests, Plan_Greedy )
    {
        const LogitsResult logits = measureLogits<DeviceType::Cuda, TensorDataType::BF16>( Device::Cuda( 0 ) );
        const std::vector<std::int32_t> generated = generateGreedy<DeviceType::Cuda, TensorDataType::BF16>( Device::Cuda( 0 ), 32 );
        const GreedyAgreement agreement = greedyAgreement( TensorDataType::BF16, generated, logits.largest_error,
            capturedConfig<DeviceType::Cuda, TensorDataType::BF16>().getVocabSize() );

        std::cout << std::format( "[ greedy ] BF16 first divergence {}\n",
            agreement.agreed == 32 ? "none" : agreement.diverged_at_a_tie ? "at a tie" : "not at a tie" );
    }
}
