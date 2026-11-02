/**
 * @file PrefixReuse.Cuda.cpp
 * @brief With KV prefix reuse off, a seeded reply does not depend on what the model ran before it.
 *
 * PromptCaching.md section 5. Reuse rounds a prompt differently from a whole prefill, so the same seeded request
 * reused and prefilled whole can sample different tokens; a caller that turns reuse off gets the reply a fresh
 * model gives. Needs exported weights of each model, so it never runs in CI.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <string>
#include <vector>

#include "Common/CudaDeviceScope.h"

import Mila;

namespace Mila::Tests::Dnn::Models
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace fs = std::filesystem;

    namespace
    {
        constexpr std::uint64_t kSeed = 42;
        constexpr std::size_t kSharedPrefix = 48;
        constexpr std::size_t kExtension = 16;
        constexpr int kReply = 48;

        fs::path modelsDirectory()
        {
            return fs::path( TEST_DATA_DIR ) / "models";
        }

        // Ids no model here reads as text, so the next-token distribution is wide and a rounding difference has
        // tokens to move between.
        std::vector<std::int32_t> promptOf( std::size_t length )
        {
            std::vector<std::int32_t> prompt( length );

            for ( std::size_t index = 0; index < length; ++index )
            {
                prompt[ index ] = static_cast<std::int32_t>( 1000 + ( 37 * index ) % 7000 );
            }

            return prompt;
        }

        template<typename TModel>
        std::vector<std::int32_t> replyTo( TModel& model, const std::vector<std::int32_t>& prompt, const GenerateParams& params )
        {
            std::vector<std::int32_t> reply;
            [[maybe_unused]] const auto status = model.generate(
                prompt, [&]( std::int32_t token ) { reply.push_back( token ); }, params );

            return reply;
        }

        // The prompt the model last ran is the start of this one, which is the case every family reuses: Llama and
        // Gemma match their token history, and Qwen returns to the position saved at the end of the last prompt.
        template<typename TModel, typename TConfig>
        void expectSeededReplyIndependentOfThePromptBefore( const std::string& label, const fs::path& weights, const TConfig& config )
        {
            if ( !fs::exists( weights ) )
            {
                GTEST_SKIP() << "Not present: " << weights.string();
            }

            int ordinal = 0;
            ASSERT_EQ( cudaGetDevice( &ordinal ), cudaSuccess );
            const DeviceId device{ DeviceType::Cuda, ordinal };
            const Common::ScopedCurrentCudaDevice current( ordinal );

            ASSERT_TRUE( current.selected() );

            cudaFree( nullptr );

            std::size_t free_bytes = 0;
            std::size_t total_bytes = 0;
            ASSERT_EQ( cudaMemGetInfo( &free_bytes, &total_bytes ), cudaSuccess );

            const DeploymentFootprint footprint = TModel::getDeploymentFootprint( weights, config, device );

            if ( footprint.memory.totalDeviceBytes() >= free_bytes )
            {
                GTEST_SKIP() << std::format( "{}: needs {} bytes and CUDA device {} has {} free",
                    label, footprint.memory.totalDeviceBytes(), ordinal, free_bytes );
            }

            auto model = TModel::load( weights, config, device );
            ASSERT_NE( model, nullptr );

            const std::vector<std::int32_t> earlier = promptOf( kSharedPrefix );
            const std::vector<std::int32_t> extended = promptOf( kSharedPrefix + kExtension );

            GenerateParams seeded;
            seeded.max_new_tokens = kReply;
            seeded.sampling.temperature = 1.0f;
            seeded.sampling.top_k = 0;
            seeded.sampling.top_p = 1.0f;
            seeded.reuse_kv_prefix = false;

            GenerateParams earlier_params;
            earlier_params.max_new_tokens = 8;
            earlier_params.sampling.temperature = 0.0f;
            earlier_params.sampling.top_k = 1;

            model->seedSampler( kSeed );
            const std::vector<std::int32_t> fresh = replyTo( *model, extended, seeded );

            [[maybe_unused]] const auto earlier_reply = replyTo( *model, earlier, earlier_params );

            model->seedSampler( kSeed );
            const std::vector<std::int32_t> after = replyTo( *model, extended, seeded );

            ASSERT_EQ( fresh.size(), static_cast<std::size_t>( kReply ) ) << label;
            EXPECT_EQ( after, fresh ) << label << ": the seeded reply changed with the prompt run before it";
        }
    }

    TEST( PrefixReuseCudaTests, Llama32_3B_Bf16 )
    {
        expectSeededReplyIndependentOfThePromptBefore<LlamaModel<DeviceType::Cuda, TensorDataType::BF16>>(
            "llama 3.2 3b bf16", modelsDirectory() / "llama" / "llama32_3b_instruct_bf16.bin", LlamaModelConfig( 4096 ) );
    }

    TEST( PrefixReuseCudaTests, Gemma4_12B_Fp4 )
    {
        GemmaModelConfig config;
        config.withContextLength( 4096 ).withWeightQuantization( WeightQuantization::FP4 );

        expectSeededReplyIndependentOfThePromptBefore<GemmaModel<DeviceType::Cuda, TensorDataType::BF16>>(
            "gemma 4 12b fp4", modelsDirectory() / "gemma" / "gemma4_12b_it_fp4.safetensors", config );
    }

    // About 15 GiB of weights, so it runs only where a device that large is visible.
    TEST( PrefixReuseCudaTests, Qwen38_27B_Fp4 )
    {
        QwenModelConfig config( 4096 );
        config.withWeightQuantization( WeightQuantization::FP4 );

        expectSeededReplyIndependentOfThePromptBefore<QwenModel<DeviceType::Cuda, TensorDataType::BF16>>(
            "qwen 3.8 27b fp4", modelsDirectory() / "qwen" / "qwen38_27b_fp4.safetensors", config );
    }
}
