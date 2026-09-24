/**
 * @file DeploymentPlanner.Load.Cuda.cpp
 * @brief Plan, then execute, through the public entry points: a load builds what its plan priced, keeps the
 *        plan, refuses a package it was not priced for, and throws a refusal it has no plan for.
 *
 * Specifications/Deployment.md sections 3.6 and 7, and negative N4. Needs exported weights, so it never runs
 * in CI.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cctype>
#include <cstddef>
#include <stdexcept>
#include <filesystem>
#include <format>
#include <iostream>
#include <string>

#include "Common/CudaDeviceScope.h"

import Mila;

namespace Mila::Tests::Deployment
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Deployment;

    namespace fs = std::filesystem;

    namespace
    {
        using GemmaCudaModel = GemmaModel<DeviceType::Cuda, TensorDataType::BF16>;
        using LlamaCudaModel = LlamaModel<DeviceType::Cuda, TensorDataType::BF16>;

        fs::path gemmaWeights()
        {
            return fs::path( TEST_DATA_DIR ) / "models" / "gemma" / "gemma4_12b_it_fp4.safetensors";
        }

        fs::path llamaWeights()
        {
            return fs::path( TEST_DATA_DIR ) / "models" / "llama" / "llama31_8b_instruct_fp4.safetensors";
        }
    }

    class DeploymentLoadCudaTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            if ( getDeviceCount( DeviceType::Cuda ) == 0 )
            {
                GTEST_SKIP() << "No CUDA device available";
            }

            ASSERT_EQ( cudaGetDevice( &ordinal_ ), cudaSuccess );
            device_ = DeviceId{ DeviceType::Cuda, ordinal_ };
        }

        int ordinal_{ 0 };
        DeviceId device_{};
    };

    // Section 3.6, depth 2: plan, inspect, run exactly that. The loaded model reports the plan it ran, and what it
    // allocated is what the plan priced (G1 through the entry points).
    TEST_F( DeploymentLoadCudaTests, ALoadBuildsWhatItsPlanPricedAndKeepsThePlan )
    {
        if ( !fs::exists( gemmaWeights() ) )
            GTEST_SKIP() << "Not present: " << gemmaWeights().string();

        const Common::ScopedCurrentCudaDevice current( ordinal_ );

        const auto planned = GemmaCudaModel::planDeployment( gemmaWeights(),
            DeploymentRequest{}.withFP4Quantization().withContextLength( 8192 ).withDevice( device_ ) );

        if ( !planned )
            GTEST_SKIP() << "does not fit CUDA device " << ordinal_ << ": " << planned.error().toString();

        const DeploymentPlan& plan = planned->best();
        const auto model = GemmaCudaModel::load( gemmaWeights(), plan );
        const MemoryStats built = model->getMemoryStats();

        std::cout << std::format( "[load] gemma 4 12b fp4 context {} chunk {} priced {} built {}\n",
            plan.contextLength(), plan.prefillChunkRows(), plan.footprint().totalDeviceBytes(), built.totalDeviceBytes() );

        EXPECT_EQ( built.device_parameter_bytes, plan.footprint().device_parameter_bytes );
        EXPECT_EQ( built.device_state_bytes, plan.footprint().device_state_bytes );
        EXPECT_EQ( built.device_scratch_bytes, plan.footprint().device_scratch_bytes );

        EXPECT_EQ( model->getDeploymentPlan().contextLength(), plan.contextLength() );
        EXPECT_EQ( model->getDeploymentPlan().prefillChunkRows(), plan.prefillChunkRows() );
        EXPECT_EQ( model->contextLength(), plan.contextLength() );
    }

    // N4: a plan priced for one package is refused against another, naming both, before anything is built.
    TEST_F( DeploymentLoadCudaTests, APlanIsRefusedForAPackageItWasNotPricedFor )
    {
        if ( !fs::exists( gemmaWeights() ) || !fs::exists( llamaWeights() ) )
            GTEST_SKIP() << "Needs both the Gemma and the Llama weights";

        const auto planned = LlamaCudaModel::planDeployment( llamaWeights(),
            DeploymentRequest{}.withFP4Quantization().withContextLength( 1024 ).withDevice( device_ ) );

        if ( !planned )
            GTEST_SKIP() << planned.error().toString();

        try
        {
            [[maybe_unused]] auto model = GemmaCudaModel::load( gemmaWeights(), planned->best() );
            FAIL() << "a Llama plan loaded Gemma weights";
        } catch ( const std::invalid_argument& error )
        {
            std::string message = error.what();
            std::transform( message.begin(), message.end(), message.begin(),
                []( unsigned char c ) { return static_cast<char>( std::tolower( c ) ); } );

            EXPECT_NE( message.find( "llama" ), std::string::npos ) << error.what();
            EXPECT_NE( message.find( "gemma" ), std::string::npos ) << error.what();
        }
    }

    // Deployment.md 12.2 through the fixed entry point: a context length that fits at no chunk is refused before
    // anything is allocated, where it used to be attempted with a warning.
    TEST_F( DeploymentLoadCudaTests, AFixedContextThatDoesNotFitIsThrownAsARefusal )
    {
        if ( !fs::exists( llamaWeights() ) )
            GTEST_SKIP() << "Not present: " << llamaWeights().string();

        LlamaModelConfig config( 131072 );
        config.withFP4Quantization();

        try
        {
            [[maybe_unused]] auto model = LlamaCudaModel::load( llamaWeights(), config, device_ );
            GTEST_SKIP() << "a 131072 context fits CUDA device " << ordinal_ << "; nothing to refuse";
        } catch ( const DeploymentRefusedError& error )
        {
            EXPECT_EQ( error.refusal().reason(), DeploymentRefusal::Reason::FixedContextDoesNotFit ) << error.what();
            EXPECT_EQ( error.refusal().contextLength(), 131072 );
            EXPECT_NE( std::string( error.what() ).find( "FixedContextDoesNotFit" ), std::string::npos );
        }
    }
}
