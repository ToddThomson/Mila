/**
 * @file GptModel.Cuda.cpp
 * @brief The deployment context bound in GptModel::generate().
 *
 * decode() reads the learned positional embedding at `position`, and GPT-2 has exactly
 * context_length of them -- so one step past the end is an out-of-bounds read, not a
 * degraded answer. The bound that stops it is what these cases pin.
 *
 * The fixture carries no real weights: the bound is a property of the generation loop,
 * not of the numbers, so a checkpoint written from a 16-position toy config reaches the
 * boundary in single-digit steps where a real GPT-2 needs ~1005. The archive is written
 * by a CPU network because it is device-neutral and the values are then deterministic;
 * the model under test is built on CUDA.
 *
 * CUDA rather than CPU for a specific reason: the CPU operation layer takes its extents
 * from build() rather than from the input (CpuLinearOp, CpuSoftmaxOp and CpuAttentionOp
 * still do), so a CPU model cannot run a prompt shorter than its built context and could
 * never reach the loop this file is about. The CUDA ops derive extents from the input.
 */

#include <gtest/gtest.h>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <format>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

import Mila;

namespace Mila::Tests::Dnn::Models
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    namespace
    {
        using GptCpu = Mila::Dnn::GptTransformer<DeviceType::Cpu, TensorDataType::FP32>;
        using GptModelCuda = Mila::Dnn::GptModel<DeviceType::Cuda, TensorDataType::FP32>;
        using HostTensor = Tensor<TensorDataType::FP32, CpuMemoryResource>;

        constexpr int64_t kEmbedding = 16;
        constexpr int64_t kLayers = 2;
        constexpr int64_t kHeads = 2;
        constexpr int64_t kVocab = 32;
        constexpr int64_t kMaxSeq = 16;
        constexpr int64_t kHidden = 64;
        constexpr int64_t kSeq = 4;

        // Small enough that the boundary is reached in single-digit steps, and well
        // inside kMaxSeq so the bound under test is the deployment context rather than
        // the trained maximum.
        constexpr dim_t kDeploymentContext = 8;
        constexpr int32_t kPromptToken = 1;
        constexpr int kPromptLength = 2;

        // Tokens the loop emits before the bound stops it: one from prefill, then one per
        // decode at positions kPromptLength .. kDeploymentContext - 1.
        constexpr int kTokensBeforeBound =
            1 + static_cast<int>( kDeploymentContext ) - kPromptLength;

        GptConfig smallConfig()
        {
            GptConfig config( kEmbedding, kLayers );
            config.withVocabSize( kVocab )
                .withNumHeads( kHeads )
                .withMaxSequenceLength( kMaxSeq )
                .withHiddenSize( kHidden )
                .withBias( true );

            return config;
        }

        std::filesystem::path makeTempArchivePath( const std::string& tag )
        {
            const auto stamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();

            return std::filesystem::temp_directory_path()
                / std::format( "mila_test_gptmodel_cuda_{}_{}.mila", tag, stamp );
        }

        void writeCheckpoint( const std::filesystem::path& path )
        {
            auto net = std::make_unique<GptCpu>( "gpt", smallConfig(), Device::Cpu() );
            net->build( BuildContext( shape_t{ 1, kSeq }, RuntimeMode::Inference ) );

            float value = 0.5f;

            for ( auto* parameter : net->getParameters() )
            {
                fill( *static_cast<HostTensor*>( parameter ), value );
                value += 0.25f;
            }

            ModelArchive archive( path.string(), std::make_unique<ZipSerializer>(), OpenMode::Write );
            net->save( archive, SerializationMode::Checkpoint );
        }

        std::unique_ptr<GptModelCuda> loadBoundedModel( const std::filesystem::path& path )
        {
            writeCheckpoint( path );

            return GptModelCuda::fromCheckpoint(
                path, DeviceId{ DeviceType::Cuda, 0 }, kDeploymentContext );
        }
    }

    class GptModelCudaGenerateTests : public ::testing::Test
    {
    };

    // The fixture's vocabulary is 32 while eos_token_ is 50256, so no sampled token can
    // end the run early -- the bound is the only exit, which is what makes the emitted
    // count an assertion rather than an observation.
    TEST_F( GptModelCudaGenerateTests, Generate_StopsAtTheContextBoundInsteadOfReadingPastIt )
    {
        const auto path = makeTempArchivePath( "context_bound" );
        std::error_code ec;
        std::filesystem::remove( path, ec );

        auto model = loadBoundedModel( path );
        ASSERT_NE( model, nullptr );

        const std::vector<int32_t> prompt( kPromptLength, kPromptToken );

        int emitted = 0;

        GenerateParams params;
        // Above the bound on purpose: the budget must not be what stops the loop, or the
        // case would pass with no guard at all.
        params.max_new_tokens = 64;

        const auto status = model->generate(
            prompt, [&emitted]( int32_t ) { ++emitted; }, params );

        EXPECT_EQ( status, GenerateStatus::ContextOverflow );
        EXPECT_EQ( emitted, kTokensBeforeBound );

        std::filesystem::remove( path, ec );
    }

    // The other half of the boundary: a run that fits must not report the overflow.
    // Without this, a guard that fired one position early would still pass the case above.
    TEST_F( GptModelCudaGenerateTests, Generate_ReportsTheBudgetWhenTheRunFitsInsideTheContext )
    {
        const auto path = makeTempArchivePath( "within_bound" );
        std::error_code ec;
        std::filesystem::remove( path, ec );

        auto model = loadBoundedModel( path );
        ASSERT_NE( model, nullptr );

        const std::vector<int32_t> prompt( kPromptLength, kPromptToken );

        int emitted = 0;

        GenerateParams params;
        params.max_new_tokens = kTokensBeforeBound - 1;

        const auto status = model->generate(
            prompt, [&emitted]( int32_t ) { ++emitted; }, params );

        EXPECT_EQ( status, GenerateStatus::MaxNewTokensReached );
        EXPECT_EQ( emitted, *params.max_new_tokens );

        std::filesystem::remove( path, ec );
    }

    // A prompt that cannot fit is rejected before any device work, rather than truncated
    // -- the caller is told, and the distinction matters because a silently truncated
    // prompt produces a plausible answer to a question nobody asked.
    TEST_F( GptModelCudaGenerateTests, Generate_RejectsAPromptLongerThanTheContext )
    {
        const auto path = makeTempArchivePath( "long_prompt" );
        std::error_code ec;
        std::filesystem::remove( path, ec );

        auto model = loadBoundedModel( path );
        ASSERT_NE( model, nullptr );

        const std::vector<int32_t> prompt(
            static_cast<size_t>( kDeploymentContext ) + 1, kPromptToken );

        EXPECT_THROW(
            (void)model->generate( prompt, []( int32_t ) {} ),
            std::invalid_argument );

        std::filesystem::remove( path, ec );
    }
}
