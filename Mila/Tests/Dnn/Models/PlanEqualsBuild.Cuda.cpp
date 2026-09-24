/**
 * @file PlanEqualsBuild.Cuda.cpp
 * @brief Deployment.md G1 on the single-device matrix: every prefill chunk the rule can pick builds what it prices.
 *
 * Needs the exported weights of each model, so it never runs in CI. The same gate on synthetic geometries, and
 * N1, are in PrefillChunkRule.Cuda.cpp.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <filesystem>
#include <format>
#include <iostream>
#include <string>

#include "Common/CudaDeviceScope.h"

import Mila;

namespace Mila::Tests::Dnn::Models
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace fs = std::filesystem;

    namespace
    {
        BuildContext pricedFor( DeviceId device, dim_t context_length )
        {
            return BuildContext( shape_t{ 1, context_length }, RuntimeMode::Inference, false )
                .withAllocationGranularity( allocationGranularity( device ) );
        }

        // One network alive at a time: the process-wide RoPE cache makes a second one report less than
        // it allocates.
        template<typename TNetwork, typename TConfig>
        MemoryStats predictedAt( const TConfig& config, DeviceId device, const BuildContext& context )
        {
            TNetwork predictor( "network", config, device );

            return predictor.getRequiredMemory( context );
        }

        template<typename TNetwork, typename TConfig>
        MemoryStats builtAt( const TConfig& config, DeviceId device, const BuildContext& context )
        {
            TNetwork network( "network", config, device );
            network.build( context );

            return network.getMemoryStats();
        }

        void expectCategoriesEqual( const MemoryStats& predicted, const MemoryStats& built, const std::string& label )
        {
            EXPECT_EQ( predicted.device_parameter_bytes, built.device_parameter_bytes ) << label << ": parameters";
            EXPECT_EQ( predicted.device_state_bytes, built.device_state_bytes ) << label << ": state";
            EXPECT_EQ( predicted.device_scratch_bytes, built.device_scratch_bytes ) << label << ": scratch";
            EXPECT_EQ( predicted.device_gradient_bytes, built.device_gradient_bytes ) << label << ": gradients";
        }
    }

    // ====================================================================
    // G1 on the single-device matrix (Deployment.md section 10)
    //
    // On whatever device is current, for each exported model: every rung whose predicted total fits the
    // device's free memory -- the rungs the rule can pick -- is built, and the build reports what was priced.
    // Needs the exported weights, so it never runs in CI.
    // ====================================================================

    namespace
    {
        fs::path modelsDirectory()
        {
            return fs::path( TEST_DATA_DIR ) / "models";
        }

        template<typename TModel, typename TNetwork, typename TModelConfig, typename TNetworkConfig>
        void expectPlanEqualsBuildOnThisDevice(
            const std::string& label, const fs::path& weights,
            const TModelConfig& model_config, const TNetworkConfig& network_config )
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

            const DeploymentFootprint footprint = TModel::getDeploymentFootprint( weights, model_config, device );
            const BuildContext priced = pricedFor( device, model_config.getContextLength() );

            // The network built here must be the one a load builds, or this proves nothing about a load.
            ASSERT_EQ(
                predictedAt<TNetwork>( network_config, device, priced.withPrefillSize( footprint.prefill.chunk_rows ) )
                    .totalDeviceBytes(),
                footprint.memory.totalDeviceBytes() )
                << label << ": the test's network prices differently from the model's own footprint";

            const std::size_t free_bytes = readFreeDeviceBytes( device );

            if ( footprint.memory.device_parameter_bytes >= free_bytes )
            {
                GTEST_SKIP() << std::format( "{}: weights need {} bytes and CUDA device {} has {} free",
                    label, footprint.memory.device_parameter_bytes, ordinal, free_bytes );
            }

            int rungs_built = 0;

            for ( dim_t rung : TNetwork::kPrefillChunkRungs )
            {
                if ( rung > model_config.getContextLength() )
                    continue;

                const BuildContext context = priced.withPrefillSize( rung );
                const MemoryStats predicted = predictedAt<TNetwork>( network_config, device, context );

                if ( predicted.totalDeviceBytes() > free_bytes )
                {
                    std::cout << std::format( "[G1] {} CUDA device {} rung {:>4}  {} bytes priced, {} free: not a pick\n",
                        label, ordinal, rung, predicted.totalDeviceBytes(), free_bytes ) << std::flush;

                    continue;
                }

                const MemoryStats built = builtAt<TNetwork>( network_config, device, context );

                std::cout << std::format( "[G1] {} CUDA device {} rung {:>4}  priced {} built {}\n",
                    label, ordinal, rung, predicted.totalDeviceBytes(), built.totalDeviceBytes() ) << std::flush;

                expectCategoriesEqual( predicted, built, std::format( "{} rung {}", label, rung ) );
                ++rungs_built;
            }

            EXPECT_GT( rungs_built, 0 ) << label << ": no rung fits, so nothing was compared";
        }

        using GemmaCudaModel = GemmaModel<DeviceType::Cuda, TensorDataType::BF16>;
        using LlamaCudaModel = LlamaModel<DeviceType::Cuda, TensorDataType::BF16>;
        using QwenCudaModel = QwenModel<DeviceType::Cuda, TensorDataType::BF16>;

        template<typename TModel>
        auto networkConfigOf( const fs::path& weights )
        {
            Serialization::WeightsReader reader( weights );

            return TModel::configFromMetadata( reader.getWeightsMetadata() );
        }

        QwenConfig qwenNetworkConfigOf( const fs::path& weights, const QwenModelConfig& model_config )
        {
            QwenConfig config = networkConfigOf<QwenCudaModel>( weights );
            config.withLanguageModelHeadPositions( model_config.getLanguageModelHeadPositions() );

            return config;
        }
    }

    TEST( PlanEqualsBuildCudaTests, Gemma4_12B_Fp4_Context8192 )
    {
        const fs::path weights = modelsDirectory() / "gemma" / "gemma4_12b_it_fp4.safetensors";

        if ( !fs::exists( weights ) )
        {
            GTEST_SKIP() << "Not present: " << weights.string();
        }

        GemmaModelConfig config;
        config.withContextLength( 8192 ).withWeightQuantization( WeightQuantization::FP4 );

        using Network = GemmaTransformer<DeviceType::Cuda, TensorDataType::BF16,
            Quant::Weight::PerGroupFp4<128>, GemmaCudaModel::GemmaSlidingKvPolicy>;

        expectPlanEqualsBuildOnThisDevice<GemmaCudaModel, Network>(
            "gemma 4 12b fp4", weights, config, networkConfigOf<GemmaCudaModel>( weights ) );
    }

    TEST( PlanEqualsBuildCudaTests, Llama31_8B_Fp4_Context8192 )
    {
        const fs::path weights = modelsDirectory() / "llama" / "llama31_8b_instruct_fp4.safetensors";

        if ( !fs::exists( weights ) )
        {
            GTEST_SKIP() << "Not present: " << weights.string();
        }

        LlamaModelConfig config( 8192 );
        config.withWeightQuantization( WeightQuantization::FP4 );

        using Network = LlamaTransformer<DeviceType::Cuda, TensorDataType::BF16,
            Quant::Weight::PerGroupFp4<128>, LlamaCudaModel::LlamaKvPolicy>;

        expectPlanEqualsBuildOnThisDevice<LlamaCudaModel, Network>(
            "llama 3.1 8b fp4", weights, config, networkConfigOf<LlamaCudaModel>( weights ) );
    }

    TEST( PlanEqualsBuildCudaTests, Qwen38_27B_Fp4_Context8192 )
    {
        const fs::path weights = modelsDirectory() / "qwen" / "qwen38_27b_fp4.safetensors";

        if ( !fs::exists( weights ) )
        {
            GTEST_SKIP() << "Not present: " << weights.string();
        }

        QwenModelConfig config( 8192 );
        config.withWeightQuantization( WeightQuantization::FP4 );

        using Network = QwenTransformer<DeviceType::Cuda, TensorDataType::BF16,
            QwenOraclePrecisionPlan, QwenCudaModel::QwenKvPolicy>;

        expectPlanEqualsBuildOnThisDevice<QwenCudaModel, Network>(
            "qwen 3.8 27b fp4", weights, config, qwenNetworkConfigOf( weights, config ) );
    }

    TEST( PlanEqualsBuildCudaTests, Qwen38_27B_Codebook_Context4096 )
    {
        const fs::path weights = modelsDirectory() / "qwen" / "qwen38_27b_cb2-3.safetensors";

        if ( !fs::exists( weights ) )
        {
            GTEST_SKIP() << "Not present: " << weights.string();
        }

        QwenModelConfig config( 4096 );
        config.withPrecisionPlan();

        using Network = QwenTransformer<DeviceType::Cuda, TensorDataType::BF16,
            QwenPrecisionPlan, QwenCudaModel::QwenKvPolicy>;

        expectPlanEqualsBuildOnThisDevice<QwenCudaModel, Network>(
            "qwen 3.8 27b cb2-3", weights, config, qwenNetworkConfigOf( weights, config ) );
    }
}
