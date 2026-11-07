/**
 * @file QwenModel.Load.Metal.cpp
 * @brief Qwen 3.8's four-layer packed build loads and runs on Metal in FP16 and BF16, under its own per-role plan.
 *
 * Four layers of sixty-four generate nothing meaningful, but they reach every codebook format, both block kinds and the whole
 * load path. Correctness is Notebooks/MetalQwen.md's gate; this is the machinery. Needs the build, so it never runs in CI.
 */

#include <gtest/gtest.h>

import Mila;
import std;

namespace Mila::Tests::Dnn::Models::Qwen
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace
    {
        namespace fs = std::filesystem;

        fs::path fixturePath()
        {
            return fs::path( TEST_DATA_DIR ) / "models" / "qwen" / "qwen38_27b_l4_2p9bit.safetensors";
        }

        template<typename TPrecisionConstant>
        class QwenModelLoadMetal : public ::testing::Test
        {
        protected:
            void SetUp() override
            {
                if ( getDeviceCount( DeviceType::Metal ) == 0 )
                {
                    GTEST_SKIP() << "No Metal device available";
                }

                if ( !fs::exists( fixturePath() ) )
                {
                    GTEST_SKIP() << "Packed 4-layer fixture not present at: " << fixturePath().string();
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

        TYPED_TEST_SUITE( QwenModelLoadMetal, Precisions, PrecisionNames );
    }

    TYPED_TEST( QwenModelLoadMetal, FourLayerPackedLoadsAndRuns )
    {
        QwenModelConfig model_config( 512 );
        model_config.withPrecisionPlan();

        auto model = QwenModel<DeviceType::Metal, TypeParam::value>::load( fixturePath(), model_config );

        const std::vector<std::int32_t> prompt{ 9707, 11, 1879, 0 };
        std::vector<std::int32_t> produced;

        GenerateParams params;
        params.max_new_tokens = 4;
        params.sampling.temperature = 0.0f;

        const GenerateStatus status = model->generate(
            prompt, [&]( std::int32_t token ) { produced.push_back( token ); }, params, std::stop_token{} );

        EXPECT_NE( status, GenerateStatus::ContextOverflow );
        EXPECT_EQ( produced.size(), 4u );
    }
}
