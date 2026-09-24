/**
 * @file DeploymentPlanner.G2.Cuda.cpp
 * @brief Deployment.md G2: given the readings Chat's automatic context saw, the planner picks what Chat picked.
 *
 * The table was recorded at 0.21.0-dev+3, before the rule moved into the library. The planner reads nothing,
 * so both cards' rows are checked on whichever device runs this. Needs the exported weights of each model, so
 * it never runs in CI.
 */

#include <gtest/gtest.h>
#include <cstddef>
#include <expected>
#include <filesystem>
#include <format>
#include <iostream>
#include <optional>
#include <string>
#include <utility>

import Mila;

namespace Mila::Tests::Deployment
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Deployment;

    namespace fs = std::filesystem;

    namespace
    {
        constexpr std::size_t kRtx4070FreeBytes = 11645485056;
        constexpr std::size_t kRtx5060TiFreeBytes = 15908995072;

        // What Chat searched below: its family ceilings (Chat.FamilyTraits.ixx), which the trained maximum caps.
        constexpr dim_t kGemmaAndLlamaCeiling = 131072;
        constexpr dim_t kQwenCeiling = 262144;

        struct Recorded
        {
            std::size_t free_bytes;

            /// Absent when Chat found no plan.
            std::optional<dim_t> context_length;
            dim_t chunk_rows{ 0 };

            /// The table's "held back for a full chunk".
            bool held_back{ false };
        };

        fs::path modelsDirectory()
        {
            return fs::path( TEST_DATA_DIR ) / "models";
        }

        DeviceReading recordedReading( std::size_t free_bytes )
        {
            DeviceReading reading;
            reading.device = Device::Cuda( 0 );
            reading.free_bytes = free_bytes;
            reading.total_bytes = free_bytes;
            reading.allocation_granularity = allocationGranularity( Device::Cuda( 0 ) );

            return reading;
        }

        template<typename TNetwork, typename TNetworkConfig>
        void expectRecordedChoices(
            const std::string& label, const fs::path& weights, const TNetworkConfig& network_config,
            const DeploymentRequest& request, const Recorded& on_4070, const Recorded& on_5060_ti )
        {
            Serialization::WeightsReader reader( weights );

            // The table recorded Chat's reading before it constructed a graph; the planner reads after, once the
            // execution context holds its share (Deployment.md section 9). What construction holds is measured
            // here -- 4 MiB on both cards, the cuBLASLt workspace -- and taken off, so the planner is given the
            // reading it would have taken on that machine at that moment.
            const std::size_t free_before_construction = readFreeDeviceBytes( Device::Cuda( 0 ) );
            const TNetwork network( "g2", network_config, Device::Cuda( 0 ) );
            const std::size_t construction_holds = free_before_construction - readFreeDeviceBytes( Device::Cuda( 0 ) );

            ASSERT_LT( construction_holds, std::size_t{ 64 } << 20 )
                << "construction held more than any execution context should; is something else using the device?";

            for ( const auto& [card, recorded] : { std::pair{ "RTX 4070", on_4070 }, std::pair{ "RTX 5060 Ti", on_5060_ti } } )
            {
                const DeviceReading reading = recordedReading( recorded.free_bytes - construction_holds );
                const auto planned = planOnDevice( network, request, reading,
                    network_config.getMaxSequenceLength(), reader.getWeightsMetadata(), reader.getWeightQuantization() );
                const std::string at = std::format( "{} on the {}", label, card );

                if ( !recorded.context_length )
                {
                    ASSERT_FALSE( planned.has_value() ) << at << ": planned " << planned->best().contextLength();
                    EXPECT_EQ( planned.error().reason(), DeploymentRefusal::Reason::WeightsExceedDevice ) << at;

                    continue;
                }

                ASSERT_TRUE( planned.has_value() ) << at << ": " << planned.error().toString();

                const DeploymentPlan& best = planned->best();

                std::cout << std::format( "[G2] {:<22} {:<12} context {:>6} chunk {:>4} limit {}\n", label, card,
                    best.contextLength(), best.prefillChunkRows(), DeploymentPlan::nameOf( best.contextLimitedBy() ) );

                EXPECT_EQ( best.contextLength(), *recorded.context_length ) << at;
                EXPECT_EQ( best.prefillChunkRows(), recorded.chunk_rows ) << at;
                EXPECT_EQ( best.contextLimitedBy() == DeploymentPlan::ContextLimit::FullPrefillChunk, recorded.held_back )
                    << at << ": limit " << DeploymentPlan::nameOf( best.contextLimitedBy() );

                // G4 on the same reading: fixing the chosen context reproduces the plan.
                const auto fixed = planOnDevice( network,
                    DeploymentRequest( request ).withContextLength( best.contextLength() ),
                    reading, network_config.getMaxSequenceLength(),
                    reader.getWeightsMetadata(), reader.getWeightQuantization() );

                ASSERT_TRUE( fixed.has_value() ) << at;
                EXPECT_EQ( fixed->best().prefillChunkRows(), best.prefillChunkRows() ) << at;
                EXPECT_EQ( fixed->best().footprint().totalDeviceBytes(), best.footprint().totalDeviceBytes() ) << at;
                EXPECT_EQ( fixed->best().contextLimitedBy(), DeploymentPlan::ContextLimit::FixedByCaller ) << at;
            }
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
    }

    class DeploymentPlannerG2CudaTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            if ( getDeviceCount( DeviceType::Cuda ) == 0 )
            {
                GTEST_SKIP() << "No CUDA device available";
            }
        }

        static bool present( const fs::path& weights )
        {
            return fs::exists( weights );
        }
    };

    TEST_F( DeploymentPlannerG2CudaTests, Gemma4_12B_Fp4 )
    {
        const fs::path weights = modelsDirectory() / "gemma" / "gemma4_12b_it_fp4.safetensors";

        if ( !present( weights ) )
            GTEST_SKIP() << "Not present: " << weights.string();

        using Network = GemmaTransformer<DeviceType::Cuda, TensorDataType::BF16,
            Quant::Weight::PerGroupFp4<128>, GemmaCudaModel::GemmaSlidingKvPolicy>;

        expectRecordedChoices<Network>( "gemma-4-12b-it-fp4", weights, networkConfigOf<GemmaCudaModel>( weights ),
            DeploymentRequest{}.withFP4Quantization().withAutomaticContextLength( 1024, kGemmaAndLlamaCeiling ),
            Recorded{ kRtx4070FreeBytes, 121856, 1024, true },
            Recorded{ kRtx5060TiFreeBytes, 131072, 1024, false } );
    }

    TEST_F( DeploymentPlannerG2CudaTests, Llama31_8B_Fp4 )
    {
        const fs::path weights = modelsDirectory() / "llama" / "llama31_8b_instruct_fp4.safetensors";

        if ( !present( weights ) )
            GTEST_SKIP() << "Not present: " << weights.string();

        using Network = LlamaTransformer<DeviceType::Cuda, TensorDataType::BF16,
            Quant::Weight::PerGroupFp4<128>, LlamaCudaModel::LlamaKvPolicy>;

        expectRecordedChoices<Network>( "llama-3.1-8b-instruct-fp4", weights, networkConfigOf<LlamaCudaModel>( weights ),
            DeploymentRequest{}.withFP4Quantization().withAutomaticContextLength( 1024, kGemmaAndLlamaCeiling ),
            Recorded{ kRtx4070FreeBytes, 13312, 512, true },
            Recorded{ kRtx5060TiFreeBytes, 33792, 512, true } );
    }

    // Chat found no context for this one on the RTX 4070 and tried the load anyway (exit 5); the planner refuses.
    TEST_F( DeploymentPlannerG2CudaTests, Qwen38_27B_Fp4 )
    {
        const fs::path weights = modelsDirectory() / "qwen" / "qwen38_27b_fp4.safetensors";

        if ( !present( weights ) )
            GTEST_SKIP() << "Not present: " << weights.string();

        using Network = QwenTransformer<DeviceType::Cuda, TensorDataType::BF16,
            QwenOraclePrecisionPlan, QwenCudaModel::QwenKvPolicy>;

        expectRecordedChoices<Network>( "qwen3.8-27b-fp4", weights, networkConfigOf<QwenCudaModel>( weights ),
            DeploymentRequest{}.withWeightQuantization( WeightQuantization::FP4 ).withAutomaticContextLength( 1024, kQwenCeiling ),
            Recorded{ kRtx4070FreeBytes, std::nullopt },
            Recorded{ kRtx5060TiFreeBytes, 4096, 1024, true } );
    }

    TEST_F( DeploymentPlannerG2CudaTests, Qwen38_27B_Codebook )
    {
        const fs::path weights = modelsDirectory() / "qwen" / "qwen38_27b_cb2-3.safetensors";

        if ( !present( weights ) )
            GTEST_SKIP() << "Not present: " << weights.string();

        using Network = QwenTransformer<DeviceType::Cuda, TensorDataType::BF16,
            QwenPrecisionPlan, QwenCudaModel::QwenKvPolicy>;

        expectRecordedChoices<Network>( "qwen3.8-27b-cb2-3", weights, networkConfigOf<QwenCudaModel>( weights ),
            DeploymentRequest{}.withPrecisionPlan().withAutomaticContextLength( 1024, kQwenCeiling ),
            Recorded{ kRtx4070FreeBytes, 3072, 1024, true },
            Recorded{ kRtx5060TiFreeBytes, 64512, 1024, true } );
    }
}
