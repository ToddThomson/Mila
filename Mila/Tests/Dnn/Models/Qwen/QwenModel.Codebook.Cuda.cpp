/**
 * @file QwenModel.Codebook.Cuda.cpp
 * @brief The packed codebook path against the same weights decoded to BF16, on the 4-layer Qwen fixture.
 *
 * Every other Qwen 2.82-bit figure compares Mila with Mila's own FP4 build, so a fault in how the codebook
 * weights are loaded or multiplied would be counted as quantization damage. This compares the packed load
 * with a BF16 twin of the same file -- each codebook tensor decoded offline, everything else as stored --
 * through the reference BF16 path, whose chassis is gated against HuggingFace (QwenModel.Parity.Cuda.cpp).
 *
 * Produce the twin with:
 *   python Tools/Quantization/decode_codebook.py <data>/models/qwen/qwen38_27b_l4_2p9bit.safetensors \
 *       <data>/models/qwen/qwen38_27b_l4_2p9bit_decoded.safetensors
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <filesystem>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <format>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

import Mila;

#include "Measurement/LogLikelihoodHarness.h"

namespace Mila::Tests::Dnn::Models
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Quant::Weight;

    namespace fs = std::filesystem;

    namespace
    {
        using QwenBf16 = QwenModel<DeviceType::Cuda, TensorDataType::BF16>;

        /// QwenPrecisionPlan with its four codebook roles read as stored BF16. The FP4 roles and the head
        /// keep the plan's policy, so both arms quantize them on load the same way.
        struct DecodedCodebookPlan
        {
            using QkvProjection = PerGroupFp4<128>;
            using OutputProjection = PerGroupFp4<128>;

            using FeedForwardGateUp = NoWeightQuant;
            using FeedForwardDown = NoWeightQuant;

            using DeltaNetQueryKey = NoWeightQuant;
            using DeltaNetValueGateOutput = NoWeightQuant;
            using DeltaNetGating = NoWeightQuant;

            using EmbeddingTable = NoWeightQuant;
            using LanguageModelHead = PerGroupFp4<128>;
        };

        template<typename TWeightPlan>
        using FixtureQwen = QwenTransformer<DeviceType::Cuda, TensorDataType::BF16, TWeightPlan, QwenBf16::QwenKvPolicy>;

        constexpr dim_t kContextLength = 512;

        // Arbitrary ids, chosen to include the two the 2.82-bit build confuses: 67017 '_weather' and 21926 ' Weather'.
        const std::vector<int32_t> kPrompt{ 248045, 846, 198, 3838, 374, 279, 9104, 304, 12095, 30, 248046, 198, 27, 1628, 27362 };
        const std::vector<int32_t> kForced{ 10757, 67017, 29, 198, 21926, 29 };

        fs::path fixtureDirectory()
        {
            return fs::path( TEST_DATA_DIR ) / "models" / "qwen";
        }

        double relativeError( const std::vector<float>& got, const std::vector<float>& want )
        {
            double difference = 0.0;
            double reference = 0.0;

            for ( size_t index = 0; index < want.size(); ++index )
            {
                const double delta = static_cast<double>( got[ index ] ) - want[ index ];
                difference += delta * delta;
                reference += static_cast<double>( want[ index ] ) * want[ index ];
            }

            return std::sqrt( difference / reference );
        }

        /// The prefill's last row, then one row per teacher-forced decode step: the prefill and the decode
        /// kernels of a codebook linear are different code, and each is compared.
        template<typename TWeightPlan>
        std::vector<std::vector<float>> logitRows( const fs::path& weights )
        {
            using Network = FixtureQwen<TWeightPlan>;

            Serialization::WeightsReader reader( weights );
            const QwenConfig config = QwenBf16::configFromMetadata( reader.getWeightsMetadata() );

            auto network = Measurement::buildMeasuredNetwork<Network>(
                weights, config, DeviceId{ DeviceType::Cuda, 0 }, kContextLength );

            std::vector<std::vector<float>> rows;
            rows.push_back( Measurement::hostLogits( *network, network->prefill( Measurement::deviceTokens( *network, kPrompt ) ) ) );

            dim_t position = static_cast<dim_t>( kPrompt.size() );

            for ( const int32_t token : kForced )
            {
                rows.push_back( Measurement::hostLogits( *network,
                    network->decode( Measurement::deviceTokens( *network, { token } ), position ) ) );
                ++position;
            }

            return rows;
        }
    }

    class QwenCodebookPathTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            if ( getDeviceCount( DeviceType::Cuda ) == 0 )
            {
                GTEST_SKIP() << "No CUDA device available";
            }

            packed_ = fixtureDirectory() / "qwen38_27b_l4_2p9bit.safetensors";
            decoded_ = fixtureDirectory() / "qwen38_27b_l4_2p9bit_decoded.safetensors";

            if ( !fs::exists( packed_ ) || !fs::exists( decoded_ ) )
            {
                GTEST_SKIP() << "Needs the packed 4-layer fixture and its decoded twin in " << fixtureDirectory().string();
            }
        }

        fs::path packed_;
        fs::path decoded_;
    };

    // Both arms hold the same weight values, so what may differ is rounding inside the kernels. A fault in
    // the codebook load or multiply -- a wrong group, row range, plane or table -- is an O(1) error. Measured
    // when written (2026-10-13, RTX 4070): 1.6e-3 to 2.2e-3 on every row; a twin decoded with
    // --drop-high-plane, 0.64 to 1.9.
    TEST_F( QwenCodebookPathTests, PackedPath_MatchesItsDecodedWeights )
    {
        constexpr double kBound = 1e-2;

        const std::vector<std::vector<float>> packed = logitRows<QwenPrecisionPlan>( packed_ );
        const std::vector<std::vector<float>> decoded = logitRows<DecodedCodebookPlan>( decoded_ );

        ASSERT_EQ( packed.size(), decoded.size() );

        for ( size_t row = 0; row < packed.size(); ++row )
        {
            ASSERT_EQ( packed[ row ].size(), decoded[ row ].size() );

            const double error = relativeError( packed[ row ], decoded[ row ] );
            const int32_t packed_top = Measurement::argMax( packed[ row ] );
            const int32_t decoded_top = Measurement::argMax( decoded[ row ] );

            std::cout << std::format( "  {} row {}: relative L2 {:.3e}, argmax {} / {}, logit[67017] {:.4f} / {:.4f}, logit[21926] {:.4f} / {:.4f}\n",
                row == 0 ? "prefill" : "decode ", row, error, packed_top, decoded_top,
                packed[ row ][ 67017 ], decoded[ row ][ 67017 ], packed[ row ][ 21926 ], decoded[ row ][ 21926 ] ) << std::flush;

            EXPECT_LT( error, kBound ) << "row " << row << ( row == 0 ? " (prefill)" : " (decode)" );
        }
    }
}
