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

    // -----------------------------------------------------------------------------------------------------------------
    // Each quantized format, held to Mila's CUDA run of the same weights on the 5060 Ti (Notebooks/MetalQuantizedWeights.md
    // section 1.2): no further from HuggingFace FP32 than CUDA is.
    // -----------------------------------------------------------------------------------------------------------------

    namespace
    {
        /// One format's weights, and CUDA's distances on them: the worst logit step and the log-likelihood's distance.
        template<typename TWeightPolicy, typename TKvPolicy>
        struct QuantizedFormat
        {
            using WeightPolicy = TWeightPolicy;
            using KvPolicy = TKvPolicy;

            fs::path weights;
            double worst_relative_l2;
            double negative_log_likelihood_difference;
        };

        template<TensorDataType TPrecision, typename TFormat>
        void expectNoFurtherThanCuda( const TFormat& format )
        {
            using W = typename TFormat::WeightPolicy;
            using K = typename TFormat::KvPolicy;

            if ( !fs::exists( format.weights ) )
            {
                GTEST_SKIP() << "Not present: " << format.weights.string();
            }

            const LogitsResult logits = measureLogits<DeviceType::Metal, TPrecision, W, K>( Device::Metal( 0 ), 0, format.weights );

            EXPECT_EQ( logits.non_finite, 0 );
            EXPECT_LE( logits.worst_relative_l2, format.worst_relative_l2 );
            EXPECT_TRUE( logits.argmax_disagreements.empty() ) << "first at step " << logits.argmax_disagreements.front();

            const SequenceResult sequence = measureSequence<DeviceType::Metal, TPrecision, W, K>( Device::Metal( 0 ), format.weights );

            EXPECT_LE( std::fabs( sequence.mean_negative_log_likelihood - sequence.reference_mean_negative_log_likelihood ),
                format.negative_log_likelihood_difference );

            for ( std::size_t layer = 0; layer < sequence.layers.size(); ++layer )
            {
                EXPECT_EQ( sequence.layers[ layer ].non_finite, 0 ) << "layer " << layer;
            }

            const std::vector<std::int32_t> generated = generateGreedy<DeviceType::Metal, TPrecision, W, K>( Device::Metal( 0 ), 32, format.weights );
            const GreedyAgreement agreement = greedyAgreement( TPrecision, generated, logits.largest_error,
                capturedConfig<DeviceType::Metal, TPrecision>( format.weights ).getVocabSize() );

            EXPECT_TRUE( agreement.diverged_at_a_tie ) << "diverged after " << agreement.agreed << " tokens";
        }

        // Measured by LlamaModel.HuggingFaceReference.Cuda.cpp on the 5060 Ti, 2026-11-05. FP8 quantized at load gave the exported
        // FP8's numbers to every digit, as its bytes are the same.
        constexpr double kCudaFp4WorstRelativeL2 = 2.952e-1;
        constexpr double kCudaFp4NllDifference = 4.681e-2;
        constexpr double kCudaFp8WorstRelativeL2 = 7.396e-2;
        constexpr double kCudaFp8NllDifference = 5.774e-3;
        constexpr double kCudaFp8CacheWorstRelativeL2 = 5.299e-2;
        constexpr double kCudaFp8CacheNllDifference = 9.795e-4;

        const QuantizedFormat<PerGroupFp4<128>, NoKvCompression> kFp4{ fp4WeightsPath(), kCudaFp4WorstRelativeL2, kCudaFp4NllDifference };
        const QuantizedFormat<PerChannelFp8<>, NoKvCompression> kFp8{ fp8WeightsPath(), kCudaFp8WorstRelativeL2, kCudaFp8NllDifference };
        const QuantizedFormat<PerChannelFp8<>, NoKvCompression> kFp8AtLoad{ weightsPath(), kCudaFp8WorstRelativeL2, kCudaFp8NllDifference };
        const QuantizedFormat<NoWeightQuant, PerTokenKvFp8<>> kFp8Cache{ weightsPath(), kCudaFp8CacheWorstRelativeL2, kCudaFp8CacheNllDifference };
    }

    TYPED_TEST( LlamaModelHuggingFaceReferenceMetal, Fp4 )
    {
        expectNoFurtherThanCuda<TypeParam::value>( kFp4 );
    }

    TYPED_TEST( LlamaModelHuggingFaceReferenceMetal, Fp8 )
    {
        expectNoFurtherThanCuda<TypeParam::value>( kFp8 );
    }

    TYPED_TEST( LlamaModelHuggingFaceReferenceMetal, Fp8QuantizedAtLoad )
    {
        expectNoFurtherThanCuda<TypeParam::value>( kFp8AtLoad );
    }

    TYPED_TEST( LlamaModelHuggingFaceReferenceMetal, Fp8Cache )
    {
        expectNoFurtherThanCuda<TypeParam::value>( kFp8Cache );
    }

    // The gate can fail on a quantized model too.
    TYPED_TEST( LlamaModelHuggingFaceReferenceMetal, Fp4DecodedAtTheWrongPosition_MissesCuda )
    {
        if ( !fs::exists( fp4WeightsPath() ) )
        {
            GTEST_SKIP() << "Not present: " << fp4WeightsPath().string();
        }

        const LogitsResult result = measureLogits<DeviceType::Metal, TypeParam::value, PerGroupFp4<128>, NoKvCompression>(
            Device::Metal( 0 ), -1, fp4WeightsPath() );

        EXPECT_TRUE( result.worst_relative_l2 > kFp4.worst_relative_l2 || !result.argmax_disagreements.empty() );
    }
}
