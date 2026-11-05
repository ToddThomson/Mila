/**
 * @file LlamaModel.HuggingFaceReference.Cuda.cpp
 * @brief Llama 3.2 3B Instruct at BF16 on CUDA against HuggingFace's FP32 run: the distances Metal is held to.
 *
 * Mila's shipped path on the same weights and capture, so Metal's bars are a measurement and not a guess
 * (Notebooks/MetalFirstModel.md). Needs the BF16 weights and the capture, so it never runs in CI.
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <stop_token>
#include <string>
#include <vector>

import Mila;

#include "Llama32Reference.h"

namespace Mila::Tests::Dnn::Models::Llama32
{
    class LlamaModelHuggingFaceReferenceCudaTests : public ::testing::Test
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
                GTEST_SKIP() << "Llama 3.2 3B weights or capture not present at: " << referencePath().parent_path().string();
            }
        }
    };

    TEST_F( LlamaModelHuggingFaceReferenceCudaTests, Bf16_Logits )
    {
        const LogitsResult result = measureLogits<DeviceType::Cuda, TensorDataType::BF16>( Device::Cuda( 0 ) );

        EXPECT_EQ( result.non_finite, 0 );
        EXPECT_TRUE( result.argmax_disagreements.empty() ) << "first at step " << result.argmax_disagreements.front();
    }

    TEST_F( LlamaModelHuggingFaceReferenceCudaTests, Bf16_SequenceLogLikelihoodAndRanges )
    {
        const SequenceResult result = measureSequence<DeviceType::Cuda, TensorDataType::BF16>( Device::Cuda( 0 ) );

        for ( const LayerRange& layer : result.layers )
        {
            EXPECT_EQ( layer.non_finite, 0 );
        }
    }

    TEST_F( LlamaModelHuggingFaceReferenceCudaTests, Bf16_Greedy )
    {
        const LogitsResult logits = measureLogits<DeviceType::Cuda, TensorDataType::BF16>( Device::Cuda( 0 ) );
        const std::vector<std::int32_t> generated = generateGreedy<DeviceType::Cuda, TensorDataType::BF16>( Device::Cuda( 0 ), 32 );
        const GreedyAgreement agreement = greedyAgreement( TensorDataType::BF16, generated, logits.largest_error,
            capturedConfig<DeviceType::Cuda, TensorDataType::BF16>().getVocabSize() );

        EXPECT_TRUE( agreement.diverged_at_a_tie ) << "diverged after " << agreement.agreed << " tokens";
    }
}
