/**
 * @file Gemma.ToolCallStop.Cuda.cpp
 * @brief A Gemma 4 reply keeps every call it makes and stops where the model hands over to the tools.
 *
 * Gemma 4 writes all the calls a turn needs, then <|tool_response>, which its generation_config.json declares a stop.
 * Needs the exported 12B weights, so it never runs in CI.
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

    namespace Conversation = Mila::Dnn::Conversation;
    namespace fs = std::filesystem;

    namespace
    {
        fs::path gemmaData( const char* name )
        {
            return fs::path( TEST_DATA_DIR ) / "models" / "gemma" / name;
        }

        const std::string kWeatherTool = R"([{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city.", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "The city name"}}, "required": ["city"]}}}])";
    }

    TEST( GemmaToolCallStopCudaTests, Gemma4_12B_Q4_0_ATurnNeedingTwoCallsMakesBothAndStopsAfterThem )
    {
        const fs::path weights = gemmaData( "gemma4_12b_it_qat_q4_0.safetensors" );
        const fs::path tokenizer_path = gemmaData( "gemma_tokenizer.bin" );

        if ( !fs::exists( weights ) || !fs::exists( tokenizer_path ) )
        {
            GTEST_SKIP() << "Not present: " << weights.string() << " or " << tokenizer_path.string();
        }

        int ordinal = 0;
        ASSERT_EQ( cudaGetDevice( &ordinal ), cudaSuccess );
        const DeviceId device{ DeviceType::Cuda, ordinal };
        const Common::ScopedCurrentCudaDevice current( ordinal );

        ASSERT_TRUE( current.selected() );

        cudaFree( nullptr );

        GemmaModelConfig config;
        config.withContextLength( 4096 ).withWeightQuantization( WeightQuantization::Q4_0 );

        using Model = GemmaModel<DeviceType::Cuda, TensorDataType::BF16>;

        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        ASSERT_EQ( cudaMemGetInfo( &free_bytes, &total_bytes ), cudaSuccess );

        const DeploymentFootprint footprint = Model::getDeploymentFootprint( weights, config, device );

        if ( footprint.memory.totalDeviceBytes() >= free_bytes )
        {
            GTEST_SKIP() << std::format( "needs {} bytes and CUDA device {} has {} free",
                footprint.memory.totalDeviceBytes(), ordinal, free_bytes );
        }

        auto model = Model::load( weights, config, device );
        ASSERT_NE( model, nullptr );

        const auto tokenizer = Mila::Data::BpeTokenizer::loadGemma( tokenizer_path );
        const Gemma::GemmaChatProtocol protocol;
        const std::vector<Conversation::Turn> history = {
            { Conversation::Role::System, "You are a helpful assistant. Use the tools when they help." },
            { Conversation::Role::User, "What's the weather in Paris and in Tokyo right now?" } };
        const Conversation::Prompt prompt = protocol.formatPrompt( history, { {}, kWeatherTool } );

        const auto encoded = tokenizer->encode( prompt.text );
        const std::vector<std::int32_t> prompt_ids( encoded.begin(), encoded.end() );

        GenerateParams greedy;
        greedy.max_new_tokens = 256;
        greedy.sampling.temperature = 0.0f;
        greedy.sampling.top_k = 1;

        std::vector<Mila::Data::TokenId> reply_ids;
        const GenerateStatus status = model->generate(
            prompt_ids, [&]( std::int32_t token ) { reply_ids.push_back( static_cast<Mila::Data::TokenId>( token ) ); }, greedy );

        const std::string text = tokenizer->decode( reply_ids );
        const Conversation::Reply reply = protocol.parseReply( text, prompt );

        EXPECT_EQ( status, GenerateStatus::Success ) << text;
        ASSERT_EQ( reply.tool_calls.size(), 2u ) << text;
        EXPECT_EQ( reply.tool_calls[ 0 ].name, "get_weather" );
        EXPECT_NE( reply.tool_calls[ 0 ].arguments.find( "Paris" ), std::string::npos ) << reply.tool_calls[ 0 ].arguments;
        EXPECT_NE( reply.tool_calls[ 1 ].arguments.find( "Tokyo" ), std::string::npos ) << reply.tool_calls[ 1 ].arguments;
        EXPECT_TRUE( reply.malformed_tool_calls.empty() ) << text;

        // The reply ends with the second call: the model stopped at its own <|tool_response>, and wrote no result.
        EXPECT_TRUE( text.ends_with( "<tool_call|>" ) ) << text;
    }
}
