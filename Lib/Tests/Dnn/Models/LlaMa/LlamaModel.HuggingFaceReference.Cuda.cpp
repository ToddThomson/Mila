/**
 * @file LlamaModel.HuggingFaceReference.Cuda.cpp
 * @brief Llama 3.2 3B Instruct at BF16 on CUDA against HuggingFace's FP32 run, on its BF16 weights and in each quantized
 *        format: the distances Metal is held to.
 *
 * Mila's shipped path on the same weights and capture, so Metal's bars are a measurement and not a guess
 * (Notebooks/MetalFirstModel.md, Notebooks/MetalQuantizedWeights.md). Needs the weights and the capture, so it never runs in CI.
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

    // -----------------------------------------------------------------------------------------------------------------
    // Each quantized format, on the weights Metal runs: the distances Metal's runs of the same weights are held to
    // (Notebooks/MetalQuantizedWeights.md section 1.2). Each reports its numbers; CUDA is the accepted run, so nothing bounds them here.
    // -----------------------------------------------------------------------------------------------------------------

    namespace
    {
        template<typename TWeightPolicy, typename TKvPolicy>
        void expectTheGatesChecks( const fs::path& weights )
        {
            if ( !fs::exists( weights ) )
            {
                GTEST_SKIP() << "Not present: " << weights.string();
            }

            const LogitsResult logits = measureLogits<DeviceType::Cuda, TensorDataType::BF16, TWeightPolicy, TKvPolicy>( Device::Cuda( 0 ), 0, weights );

            EXPECT_EQ( logits.non_finite, 0 );

            const SequenceResult sequence = measureSequence<DeviceType::Cuda, TensorDataType::BF16, TWeightPolicy, TKvPolicy>( Device::Cuda( 0 ), weights );

            for ( const LayerRange& layer : sequence.layers )
            {
                EXPECT_EQ( layer.non_finite, 0 );
            }

            const std::vector<std::int32_t> generated =
                generateGreedy<DeviceType::Cuda, TensorDataType::BF16, TWeightPolicy, TKvPolicy>( Device::Cuda( 0 ), 32, weights );
            const GreedyAgreement agreement = greedyAgreement( TensorDataType::BF16, generated, logits.largest_error,
                capturedConfig<DeviceType::Cuda, TensorDataType::BF16>( weights ).getVocabSize() );

            std::cout << std::format( "[ greedy ] {} first divergence {}\n", runName<TensorDataType::BF16, TKvPolicy>( weights ),
                agreement.agreed == 32 ? "none" : agreement.diverged_at_a_tie ? "at a tie" : "not at a tie" );
        }
    }

    TEST_F( LlamaModelHuggingFaceReferenceCudaTests, Fp4 )
    {
        expectTheGatesChecks<PerGroupFp4<128>, NoKvCompression>( fp4WeightsPath() );
    }

    TEST_F( LlamaModelHuggingFaceReferenceCudaTests, Fp8 )
    {
        expectTheGatesChecks<PerChannelFp8<>, NoKvCompression>( fp8WeightsPath() );
    }

    TEST_F( LlamaModelHuggingFaceReferenceCudaTests, Fp8QuantizedAtLoad )
    {
        expectTheGatesChecks<PerChannelFp8<>, NoKvCompression>( weightsPath() );
    }

    TEST_F( LlamaModelHuggingFaceReferenceCudaTests, Bf16WithAnFp8Cache )
    {
        expectTheGatesChecks<NoWeightQuant, PerTokenKvFp8<>>( weightsPath() );
    }
}
