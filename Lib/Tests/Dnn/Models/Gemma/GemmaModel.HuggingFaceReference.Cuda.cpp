/**
 * @file GemmaModel.HuggingFaceReference.Cuda.cpp
 * @brief Gemma 4 12B Instruct in Q4_0 at BF16 on CUDA against HuggingFace's FP32 run on the package's decoded weights:
 *        one of the two distances Metal is held to.
 *
 * Mila's shipped path on the same weights and capture, so Metal's bars are a measurement and not a guess
 * (Notebooks/MetalGemma.md section 1.3). Each check reports its numbers; CUDA is an accepted run, so only what any run
 * must hold is asserted. Needs the weights and the capture, so it never runs in CI.
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <stop_token>
#include <format>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

import Mila;

#include "Gemma12bReference.h"

namespace Mila::Tests::Dnn::Models::Gemma12b
{
    class GemmaModelHuggingFaceReferenceCudaTests : public ::testing::Test
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
                GTEST_SKIP() << "Gemma 4 12B Q4_0 weights or capture not present at: " << referencePath().parent_path().string();
            }
        }
    };

    TEST_F( GemmaModelHuggingFaceReferenceCudaTests, Q4_0_Logits )
    {
        const LogitsResult result = measureLogits<DeviceType::Cuda, TensorDataType::BF16>( Device::Cuda( 0 ) );

        EXPECT_EQ( result.non_finite, 0 );
    }

    TEST_F( GemmaModelHuggingFaceReferenceCudaTests, Q4_0_SequenceLogLikelihoodAndRanges )
    {
        const SequenceResult result = measureSequence<DeviceType::Cuda, TensorDataType::BF16>( Device::Cuda( 0 ) );

        for ( const LayerRange& layer : result.layers )
        {
            EXPECT_EQ( layer.non_finite, 0 );
        }
    }

    TEST_F( GemmaModelHuggingFaceReferenceCudaTests, Q4_0_PiecesKlDivergence )
    {
        const PiecesResult result = measurePieces<DeviceType::Cuda, TensorDataType::BF16>( Device::Cuda( 0 ) );

        EXPECT_EQ( result.non_finite, 0 );
    }

    TEST_F( GemmaModelHuggingFaceReferenceCudaTests, Q4_0_AcrossTheWindowEdge )
    {
        const LogitsResult result = measureWindowEdge<DeviceType::Cuda, TensorDataType::BF16>( Device::Cuda( 0 ) );

        EXPECT_EQ( result.non_finite, 0 );
    }

    TEST_F( GemmaModelHuggingFaceReferenceCudaTests, Q4_0_Greedy )
    {
        const LogitsResult logits = measureLogits<DeviceType::Cuda, TensorDataType::BF16>( Device::Cuda( 0 ) );
        const std::vector<std::int32_t> generated = generateGreedy<DeviceType::Cuda, TensorDataType::BF16>( Device::Cuda( 0 ), 32 );
        const GreedyAgreement agreement = greedyAgreement( TensorDataType::BF16, generated, logits.largest_error,
            capturedConfig<DeviceType::Cuda, TensorDataType::BF16>().getVocabSize() );

        std::cout << std::format( "[ greedy ] BF16 first divergence {}\n",
            agreement.agreed == 32 ? "none" : agreement.diverged_at_a_tie ? "at a tie" : "not at a tie" );
    }
}
