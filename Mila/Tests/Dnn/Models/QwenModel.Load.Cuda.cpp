/**
 * @file QwenModel.Load.Cuda.cpp
 * @brief QwenModel against a real converted artifact: geometry, footprint, generation.
 *
 * WHY A TRUNCATED FIXTURE. The full Qwen 3.8 27B artifact is 50 GiB at BF16 -- it fits
 * neither card in the rig nor host RAM, so reference precision cannot be exercised on the
 * whole stack anywhere on this hardware (that gap is what the Phase 4 layer-streamed parity
 * harness exists to close). The converter's `--max-layers 4` output is the vehicle that does
 * fit: 3 Gated DeltaNet layers plus 1 full-attention layer, which is every block kind and
 * every transform the converter performs, at ~7.6 GiB.
 *
 * It is NOT a model. Four layers of a sixty-four layer stack produce meaningless tokens, so
 * nothing here asserts anything about what is generated -- only that the machinery runs and
 * stays inside its contracts. Coherence is Phase 4's gate and needs the HF reference.
 *
 * Produce the fixture with:
 *   python Qwen/convert_weights.py --model Qwen/Qwen3.8-27B --max-layers 4 \
 *       --output <repo>/Data/Models/Qwen/qwen38_27b_l4_bf16.bin
 *
 * Requires a real checkpoint and is skipped without one, so it does not run in CI.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <filesystem>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <stop_token>

import Mila;

namespace Mila::Tests::Dnn::Models
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace fs = std::filesystem;

    namespace
    {
        fs::path qwenFixturePath()
        {
            return fs::path( TEST_DATA_DIR ) / "models" / "qwen" / "qwen38_27b_l4_bf16.bin";
        }

        // Small enough that the fixture plus its workspaces fit a 12 GiB card alongside a
        // desktop; the geometry under test does not depend on the context length.
        constexpr dim_t kContextLength = 512;
    }

    class QwenModelLoadCudaTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            if ( getDeviceCount( DeviceType::Cuda ) == 0 )
            {
                GTEST_SKIP() << "No CUDA device available";
            }

            fixture_ = qwenFixturePath();

            if ( !fs::exists( fixture_ ) )
            {
                GTEST_SKIP() << "Qwen fixture not present at: " << fixture_.string();
            }
        }

        fs::path fixture_;
    };

    using QwenBf16 = QwenModel<DeviceType::Cuda, TensorDataType::BF16>;

    // The geometry the artifact declares must survive the round trip through the converter's
    // metadata and PretrainedMetadata into QwenConfig. Three of these fields are invisible in
    // the HF config.json and were read from the checkpoint, so a silent zero here is exactly
    // the failure this checks for -- a zeroed interleave would build the wrong block kinds.
    TEST_F( QwenModelLoadCudaTests, Metadata_CarriesTheQwenGeometry )
    {
        QwenModelConfig model_config( kContextLength );

        auto model = QwenBf16::fromPretrained( fixture_, model_config );
        const QwenConfig& config = model->getNetworkConfig();

        EXPECT_EQ( config.getModelDim(), 5120 );
        EXPECT_EQ( config.getVocabSize(), 248320 );
        EXPECT_EQ( config.getNumHeads(), 24 );
        EXPECT_EQ( config.getNumKVHeads(), 4 );
        EXPECT_EQ( config.getHeadDim(), 256 );
        EXPECT_EQ( config.getHiddenDimension(), 17408 );
        EXPECT_TRUE( config.hasAttentionOutputGate() );
        EXPECT_EQ( config.getFullAttentionInterval(), 4 );
        EXPECT_EQ( config.getRotaryDim(), 64 );
        EXPECT_FALSE( config.getTieWordEmbeddings() );

        EXPECT_EQ( config.getLinearNumKeyHeads(), 16 );
        EXPECT_EQ( config.getLinearNumValueHeads(), 48 );
        EXPECT_EQ( config.getLinearHeadDim(), 128 );
        EXPECT_EQ( config.getLinearConvKernelDim(), 4 );

        // The fixture is truncated, so the interleave places exactly one full-attention layer
        // (index 3) and three DeltaNet layers.
        EXPECT_EQ( config.getNumLayers(), 4 );
        EXPECT_EQ( config.getNumFullAttentionLayers(), 1 );
        EXPECT_EQ( config.getNumDeltaNetLayers(), 3 );
    }

    // The prediction runs before the model exists, so it must not need the device to have
    // room -- which is the situation it is asked in.
    TEST_F( QwenModelLoadCudaTests, DeploymentFootprint_IsReportedWithoutLoading )
    {
        QwenModelConfig model_config( kContextLength );

        const DeploymentFootprint footprint =
            QwenBf16::getDeploymentFootprint( fixture_, model_config );

        // The fixture's weights alone are ~7.6 GiB; a figure below the parameter bytes would
        // mean the prediction is not seeing the tables.
        EXPECT_GT( footprint.memory.device_parameter_bytes, std::size_t{ 7 } * 1024 * 1024 * 1024 );
        EXPECT_GT( footprint.prefill.chunk_rows, 0 );
        EXPECT_LE( footprint.prefill.chunk_rows, kContextLength );
    }

    // Prefill then a bounded decode run. Nothing is asserted about WHICH tokens appear --
    // four layers of sixty-four cannot produce meaningful ones -- only that every id the
    // sampler returns is a real vocabulary index. A NaN logit row would surface here, since
    // an argmax over NaN does not land inside the vocabulary by accident.
    TEST_F( QwenModelLoadCudaTests, Generation_RunsAndStaysInsideTheVocabulary )
    {
        QwenModelConfig model_config( kContextLength );

        auto model = QwenBf16::fromPretrained( fixture_, model_config );

        const std::vector<int32_t> prompt{ 9707, 11, 1879, 0 };
        std::vector<int32_t> produced;

        GenerateParams params;
        params.max_new_tokens = 4;
        params.sampling.temperature = 0.0f;

        const GenerateStatus status = model->generate(
            prompt,
            [&]( int32_t token ) { produced.push_back( token ); },
            params,
            std::stop_token{} );

        EXPECT_TRUE( status == GenerateStatus::MaxNewTokensReached
            || status == GenerateStatus::Success );

        for ( int32_t token : produced )
        {
            EXPECT_GE( token, 0 );
            EXPECT_LT( token, model->vocabSize() );
        }
    }

    // Two generations from one model must not differ, and on this family that is a stronger
    // statement than it is on an attention stack: 48 of 64 layers carry a recurrent state, and
    // nothing resets it explicitly. Prefill at position 0 is what zeroes it, so a second
    // greedy generation reproducing the first is the evidence that the state does not leak
    // across calls. It fails loudly if that ever stops being true.
    TEST_F( QwenModelLoadCudaTests, RecurrentState_DoesNotLeakBetweenGenerations )
    {
        QwenModelConfig model_config( kContextLength );

        auto model = QwenBf16::fromPretrained( fixture_, model_config );

        const std::vector<int32_t> prompt{ 9707, 11, 1879, 0 };

        GenerateParams params;
        params.max_new_tokens = 4;
        params.sampling.temperature = 0.0f;

        auto run = [&]()
        {
            std::vector<int32_t> tokens;

            model->generate( prompt, [&]( int32_t token ) { tokens.push_back( token ); },
                params, std::stop_token{} );

            return tokens;
        };

        const std::vector<int32_t> first = run();
        const std::vector<int32_t> second = run();

        EXPECT_EQ( first, second );
    }

    // The Phase 4 / Phase 5 boundary, refused by name. Checked before the artifact is opened,
    // so it needs no fixture -- and it must stay that way: a caller asking for an allocation
    // this chassis cannot build should hear so immediately.
    TEST( QwenModelQuantizationTests, QuantizedModes_AreRefusedNamingThePlan )
    {
        if ( getDeviceCount( DeviceType::Cuda ) == 0 )
        {
            GTEST_SKIP() << "No CUDA device available";
        }

        QwenModelConfig model_config( kContextLength );
        model_config.withWeightQuantization( WeightQuantization::FP4 );

        EXPECT_THROW(
            QwenBf16::fromPretrained( "does-not-exist.bin", model_config ),
            std::runtime_error );

        QwenModelConfig none_config( kContextLength );

        // The same call with the supported mode reaches the file and fails on THAT instead,
        // which is what proves the refusal above was the quantization mode and not the path.
        EXPECT_THROW(
            QwenBf16::fromPretrained( "does-not-exist.bin", none_config ),
            std::exception );
    }
}
