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
#include <optional>
#include <string>

#include "Common/CudaDeviceScope.h"

import Mila;

namespace Mila::Tests::Deployment
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

        // Outside the blocks each model allocates a few parameter tensors alone: the embedding table, the final norm,
        // the head and its scales. Each may round by up to one granule.
        constexpr std::size_t kAllocationsOutsideBlocks = 4;

        // MemoryFootprint.md 11.8.1 gate: a block's weights are one allocation, so a model's weights round by at most
        // one granule a block (plus the few outside them), where one allocation a tensor rounded by hundreds of MiB.
        // Pricing reads no weights, so only the file's metadata is needed.
        template<typename TModel, typename TNetwork>
        void expectWeightsRoundOncePerBlock( const std::string& label, const fs::path& weights, dim_t context_length )
        {
            if ( !fs::exists( weights ) )
            {
                std::cout << "[rounding] " << label << ": not present, " << weights.string() << "\n";

                return;
            }

            Serialization::WeightsReader reader( weights );
            const auto network_config = TModel::configFromMetadata( reader.getWeightsMetadata() );

            int ordinal = 0;
            ASSERT_EQ( cudaGetDevice( &ordinal ), cudaSuccess );
            const DeviceId device{ DeviceType::Cuda, ordinal };
            const std::size_t granularity = allocationGranularity( device );

            const BuildContext context = BuildContext( shape_t{ 1, context_length }, RuntimeMode::Inference, false )
                .withPrefillSize( TNetwork::kPrefillChunkRungs[ 0 ] );

            const std::size_t rounded =
                predictedAt<TNetwork>( network_config, device, context.withAllocationGranularity( granularity ) )
                    .device_parameter_bytes;
            const std::size_t placed =
                predictedAt<TNetwork>( network_config, device, context.withAllocationGranularity( 0 ) )
                    .device_parameter_bytes;

            const std::size_t blocks = static_cast<std::size_t>( network_config.getNumLayers() );
            const std::size_t bound = ( blocks + kAllocationsOutsideBlocks ) * granularity;

            std::cout << std::format( "[rounding] {}: weights {:.1f} MiB, rounding {:.1f} MiB over {} blocks (bound {:.1f})\n",
                label, rounded / 1048576.0, ( rounded - placed ) / 1048576.0, blocks, bound / 1048576.0 ) << std::flush;

            EXPECT_LE( rounded - placed, bound ) << label;
        }

        using GemmaCudaModel = GemmaModel<DeviceType::Cuda, TensorDataType::BF16>;
        using LlamaCudaModel = LlamaModel<DeviceType::Cuda, TensorDataType::BF16>;
        using QwenCudaModel = QwenModel<DeviceType::Cuda, TensorDataType::BF16>;

        // The 26B-A4B has no file under Data/Models; it is read where a user has it, installed in the model store.
        // Read only. Empty when it is not installed.
        fs::path installedWeights( const std::string& name )
        {
            const std::optional<Mila::Distribution::StoredModel> stored = Mila::Distribution::ModelStore().locate( name );

            return stored ? stored->weights_path : fs::path{};
        }

        constexpr const char* kGemma26BStoreName = "gemma-4-26B-A4B-it-q4_0";

        template<typename TKvCache>
        using Gemma26BNetwork = GemmaTransformer<DeviceType::Cuda, TensorDataType::BF16,
            Quant::Weight::PerGroupInt4<32>, GemmaCudaModel::GemmaSlidingKvPolicy, GemmaFeedForward::Routed, TKvCache>;

        template<typename TModel>
        auto networkConfigOf( const fs::path& weights )
        {
            Serialization::WeightsReader reader( weights );

            return TModel::configFromMetadata( reader.getWeightsMetadata() );
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
            Quant::Weight::PerGroupFp4<128>, Quant::KvCache::NoKvCompression>;

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
            QwenOraclePrecisionPlan, Quant::KvCache::NoKvCompression>;

        expectPlanEqualsBuildOnThisDevice<QwenCudaModel, Network>(
            "qwen 3.8 27b fp4", weights, config, networkConfigOf<QwenCudaModel>( weights ) );
    }

    // The agentic target's context (Qwen3.8.md, "Agentic deployment on the 16 GB card").
    TEST( PlanEqualsBuildCudaTests, Qwen38_27B_Fp4_Fp8Kv_Context32768 )
    {
        const fs::path weights = modelsDirectory() / "qwen" / "qwen38_27b_fp4.safetensors";

        if ( !fs::exists( weights ) )
        {
            GTEST_SKIP() << "Not present: " << weights.string();
        }

        QwenModelConfig config( 32768 );
        config.withWeightQuantization( WeightQuantization::FP4 ).withKvCacheCompression( KvCacheCompression::FP8 );

        using Network = QwenTransformer<DeviceType::Cuda, TensorDataType::BF16,
            QwenOraclePrecisionPlan, Quant::KvCache::PerTokenKvFp8<>>;

        expectPlanEqualsBuildOnThisDevice<QwenCudaModel, Network>(
            "qwen 3.8 27b fp4, fp8 kv", weights, config, networkConfigOf<QwenCudaModel>( weights ) );
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
            QwenPrecisionPlan, Quant::KvCache::NoKvCompression>;

        expectPlanEqualsBuildOnThisDevice<QwenCudaModel, Network>(
            "qwen 3.8 27b cb2-3", weights, config, networkConfigOf<QwenCudaModel>( weights ) );
    }

    // The quality-first agentic build at its target, on the 16 GB card (Qwen3.8.md, "Agentic deployment on the 16 GB
    // card"). Skips on a card that cannot hold the weights.
    TEST( PlanEqualsBuildCudaTests, Qwen38_27B_Fp4Codebook3_Fp8Kv_Context65536 )
    {
        const fs::path weights = modelsDirectory() / "qwen" / "qwen38_27b_fp4-cb3.safetensors";

        if ( !fs::exists( weights ) )
        {
            GTEST_SKIP() << "Not present: " << weights.string();
        }

        QwenModelConfig config( 65536 );
        config.withFp4Codebook3Plan().withKvCacheCompression( KvCacheCompression::FP8 );

        using Network = QwenTransformer<DeviceType::Cuda, TensorDataType::BF16,
            QwenFp4Codebook3PrecisionPlan, Quant::KvCache::PerTokenKvFp8<>>;

        expectPlanEqualsBuildOnThisDevice<QwenCudaModel, Network>(
            "qwen 3.8 27b fp4-cb3, fp8 kv", weights, config, networkConfigOf<QwenCudaModel>( weights ) );
    }

    // The 26B-A4B, from the model store. Its routed blocks carry the Q4_0 expert banks and their scales.
    TEST( PlanEqualsBuildCudaTests, Gemma4_26B_A4B_Q4_0_Context8192 )
    {
        const fs::path weights = installedWeights( kGemma26BStoreName );

        if ( weights.empty() )
        {
            GTEST_SKIP() << kGemma26BStoreName << " is not installed in the model store";
        }

        GemmaModelConfig config;
        config.withContextLength( 8192 ).withWeightQuantization( WeightQuantization::Q4_0 );

        expectPlanEqualsBuildOnThisDevice<GemmaCudaModel, Gemma26BNetwork<Quant::KvCache::NoKvCompression>>(
            "gemma 4 26b-a4b q4_0", weights, config, networkConfigOf<GemmaCudaModel>( weights ) );
    }

    // Its agentic deployment: the FP8 cache at 64K (ContextProfile.md).
    TEST( PlanEqualsBuildCudaTests, Gemma4_26B_A4B_Q4_0_Fp8Kv_Context65536 )
    {
        const fs::path weights = installedWeights( kGemma26BStoreName );

        if ( weights.empty() )
        {
            GTEST_SKIP() << kGemma26BStoreName << " is not installed in the model store";
        }

        GemmaModelConfig config;
        config.withContextLength( 65536 ).withWeightQuantization( WeightQuantization::Q4_0 )
            .withKvCacheCompression( KvCacheCompression::FP8 );

        expectPlanEqualsBuildOnThisDevice<GemmaCudaModel, Gemma26BNetwork<Quant::KvCache::PerTokenKvFp8<>>>(
            "gemma 4 26b-a4b q4_0, fp8 kv", weights, config, networkConfigOf<GemmaCudaModel>( weights ) );
    }

    // ====================================================================
    // Weights round once per block (MemoryFootprint.md 11.8.1), on every published family.
    // ====================================================================

    TEST( PlanEqualsBuildCudaTests, WeightsRoundOncePerBlock )
    {
        const fs::path models = modelsDirectory();

        expectWeightsRoundOncePerBlock<GemmaCudaModel, GemmaTransformer<DeviceType::Cuda, TensorDataType::BF16,
            Quant::Weight::PerGroupFp4<128>, GemmaCudaModel::GemmaSlidingKvPolicy>>(
            "gemma 4 12b fp4", models / "gemma" / "gemma4_12b_it_fp4.safetensors", 8192 );

        expectWeightsRoundOncePerBlock<GemmaCudaModel, Gemma26BNetwork<Quant::KvCache::NoKvCompression>>(
            "gemma 4 26b-a4b q4_0", installedWeights( kGemma26BStoreName ), 8192 );

        expectWeightsRoundOncePerBlock<LlamaCudaModel, LlamaTransformer<DeviceType::Cuda, TensorDataType::BF16,
            Quant::Weight::PerGroupFp4<128>, Quant::KvCache::NoKvCompression>>(
            "llama 3.1 8b fp4", models / "llama" / "llama31_8b_instruct_fp4.safetensors", 8192 );

        expectWeightsRoundOncePerBlock<QwenCudaModel, QwenTransformer<DeviceType::Cuda, TensorDataType::BF16,
            QwenOraclePrecisionPlan, Quant::KvCache::NoKvCompression>>(
            "qwen 3.8 27b fp4", models / "qwen" / "qwen38_27b_fp4.safetensors", 8192 );

        expectWeightsRoundOncePerBlock<QwenCudaModel, QwenTransformer<DeviceType::Cuda, TensorDataType::BF16,
            QwenPrecisionPlan, Quant::KvCache::NoKvCompression>>(
            "qwen 3.8 27b cb2-3", models / "qwen" / "qwen38_27b_cb2-3.safetensors", 8192 );

        expectWeightsRoundOncePerBlock<QwenCudaModel, QwenTransformer<DeviceType::Cuda, TensorDataType::BF16,
            QwenFp4Codebook3PrecisionPlan, Quant::KvCache::PerTokenKvFp8<>>>(
            "qwen 3.8 27b fp4-cb3", models / "qwen" / "qwen38_27b_fp4-cb3.safetensors", 8192 );
    }
}
