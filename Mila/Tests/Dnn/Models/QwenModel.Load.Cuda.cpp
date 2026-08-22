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
#include <chrono>
#include <cstdint>
#include <format>
#include <iostream>
#include <memory>
#include <set>
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

        // The embedding table is host-resident (Qwen3.8.md section 5, item 6), so the
        // fixture's ~7.6 GiB of weights arrives split. Both halves are asserted: the total
        // still catches a prediction that is not seeing the tables, and the split is the
        // property the residency exists for -- 248320 x 5120 BF16 rows off the card.
        constexpr std::size_t kEmbeddingTableBytes = std::size_t{ 248320 } * 5120 * 2;

        EXPECT_EQ( footprint.memory.host_parameter_bytes, kEmbeddingTableBytes );
        EXPECT_GT( footprint.memory.device_parameter_bytes + footprint.memory.host_parameter_bytes,
            std::size_t{ 7 } * 1024 * 1024 * 1024 );
        EXPECT_LT( footprint.memory.device_parameter_bytes,
            std::size_t{ 7 } * 1024 * 1024 * 1024 );

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

    // ====================================================================
    // The Section 5 allocation, against the packed 2.9-bit artifact
    //
    // Produce it with (about 50 minutes on a 12 GiB card):
    //   python Tools/Quantization/pack_qwen.py --model Qwen/Qwen3.8-27B \
    //       --calib-text corpus/wiki.train.raw \
    //       --output <repo>/Data/Models/Qwen/qwen38_27b_2p9bit.safetensors
    //
    // Unlike the 4-layer fixture above this IS the whole model, so what it proves is not
    // just that the machinery runs: a 27B stack that fits a 12 GiB card at all is the claim
    // the entire Qwen 3.8 track exists to make.
    // ====================================================================

    class QwenPackedArtifactTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            if ( getDeviceCount( DeviceType::Cuda ) == 0 )
            {
                GTEST_SKIP() << "No CUDA device available";
            }

            artifact_ = fs::path( TEST_DATA_DIR ) / "models" / "qwen"
                / "qwen38_27b_2p9bit.safetensors";

            if ( !fs::exists( artifact_ ) )
            {
                GTEST_SKIP() << "Packed Qwen artifact not present at: " << artifact_.string();
            }
        }

        fs::path artifact_;
    };

    // The prediction needs no room on the device, so it runs even where the load would not --
    // and it is the first thing that reads the packed container at all.
    TEST_F( QwenPackedArtifactTests, Footprint_ReportsTheSectionFiveAllocation )
    {
        QwenModelConfig model_config( 16384 );
        model_config.withPrecisionPlan();

        const DeploymentFootprint footprint =
            QwenBf16::getDeploymentFootprint( artifact_, model_config );

        // Printed, not just asserted: this is the Section 5 budget table as the runtime
        // actually computes it, and the bounds below are far looser than the numbers are
        // interesting. A reader chasing a memory regression wants the breakdown.
        std::cout << footprint.memory.toString() << "\n"
                  << "  prefill chunk rows: " << footprint.prefill.chunk_rows << "\n";

        // Section 5 budgets 8.65 GiB of resident weights at 2.90 average bits, with the
        // 2.37 GiB embedding table host-resident. The bound is deliberately loose in both
        // directions: what it has to catch is an order-of-magnitude miss, which is what
        // loading these codes as BF16 -- or ignoring the plan and building BF16 -- would be.
        const double device_gib =
            static_cast<double>( footprint.memory.device_parameter_bytes ) / ( 1024.0 * 1024 * 1024 );
        const double host_gib =
            static_cast<double>( footprint.memory.host_parameter_bytes ) / ( 1024.0 * 1024 * 1024 );

        EXPECT_GT( device_gib, 7.0 );
        EXPECT_LT( device_gib, 10.5 );

        // The embedding table, off the card by design (Qwen3.8.md section 5, item 6).
        EXPECT_NEAR( host_gib, 2.37, 0.05 );

        EXPECT_GT( footprint.prefill.chunk_rows, 0 );
    }

    // The Section 5 budget across the context ladder, which is what decides whether the
    // model fits. DISABLED because it is a measurement, not an assertion:
    //   MilaTests --gtest_also_run_disabled_tests
    //       --gtest_filter=QwenPackedArtifactTests.DISABLED_FootprintAcrossContexts
    TEST_F( QwenPackedArtifactTests, DISABLED_FootprintAcrossContexts )
    {
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        cudaMemGetInfo( &free_bytes, &total_bytes );

        std::cout << "  device: " << ( total_bytes / ( 1024.0 * 1024 * 1024 ) ) << " GiB total, "
                  << ( free_bytes / ( 1024.0 * 1024 * 1024 ) ) << " GiB free\n\n"
                  << "  context   params    state    device   chunk\n";

        for ( const dim_t context : { 512, 2048, 4096, 8192, 16384, 32768 } )
        {
            QwenModelConfig model_config( context );
            model_config.withPrecisionPlan();

            const DeploymentFootprint footprint =
                QwenBf16::getDeploymentFootprint( artifact_, model_config );

            const double gib = 1024.0 * 1024 * 1024;

            std::cout << std::format(
                "  {:>7}   {:>6.2f}   {:>6.2f}   {:>6.2f}   {:>5}\n",
                static_cast<long long>( context ),
                footprint.memory.device_parameter_bytes / gib,
                footprint.memory.device_state_bytes / gib,
                footprint.memory.totalDeviceBytes() / gib,
                static_cast<long long>( footprint.prefill.chunk_rows ) );
        }
    }

    // A build and an artifact must agree on the storage format, in BOTH directions: packed
    // codes read as BF16, or a BF16 blob decoded through a codebook, produce a model that
    // loads and runs and is wrong. Neither failure announces itself downstream.
    TEST_F( QwenPackedArtifactTests, MismatchedStorageFormat_IsRefusedInBothDirections )
    {
        QwenModelConfig reference_config( 512 );

        EXPECT_THROW(
            QwenBf16::getDeploymentFootprint( artifact_, reference_config ),
            std::runtime_error );

        QwenModelConfig plan_config( 512 );
        plan_config.withPrecisionPlan();

        const fs::path bf16_fixture =
            fs::path( TEST_DATA_DIR ) / "models" / "qwen" / "qwen38_27b_l4_bf16.bin";

        if ( fs::exists( bf16_fixture ) )
        {
            EXPECT_THROW(
                QwenBf16::getDeploymentFootprint( bf16_fixture, plan_config ),
                std::runtime_error );
        }
    }

    // The packed path at fixture scale. Four layers of sixty-four generate nothing
    // meaningful, but they exercise every codebook policy, both block kinds and the whole
    // load path at a size that fits anywhere -- so a failure here is a defect in the
    // machinery rather than a memory ceiling, which the full artifact cannot distinguish.
    TEST_F( QwenPackedArtifactTests, FourLayerPacked_LoadsAndRuns )
    {
        const fs::path fixture =
            fs::path( TEST_DATA_DIR ) / "models" / "qwen" / "qwen38_27b_l4_2p9bit.safetensors";

        if ( !fs::exists( fixture ) )
        {
            GTEST_SKIP() << "Packed 4-layer fixture not present at: " << fixture.string();
        }

        QwenModelConfig model_config( 512 );
        model_config.withPrecisionPlan();

        auto model = QwenBf16::fromPretrained( fixture, model_config );

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

        EXPECT_NE( status, GenerateStatus::ContextOverflow );
        EXPECT_FALSE( produced.empty() );
    }

    // Decode rate on the packed model. DISABLED because it is a measurement:
    //   MilaTests --gtest_also_run_disabled_tests
    //       --gtest_filter=QwenPackedArtifactTests.DISABLED_DecodeRate
    //
    // Timed by SUBTRACTION between two generation lengths rather than by timing one run.
    // A single run's wall clock is dominated by the 15 GiB load and then by prefill, and
    // both cancel exactly when the same prompt is generated twice at different lengths --
    // leaving the marginal cost of a decode step, which is the number Section 5 predicts.
    //
    // This is the first DRAM-resident measurement of these kernels. Every codebook figure
    // recorded so far came off an L2-resident benchmark at 3072x8192, which the spec is
    // explicit overstates what DRAM can sustain; here the model is 8.69 GiB and every
    // weight is read once per token, so there is nowhere for it to hide.
    TEST_F( QwenPackedArtifactTests, DISABLED_DecodeRate )
    {
        QwenModelConfig model_config( 512 );
        model_config.withPrecisionPlan();

        auto model = QwenBf16::fromPretrained( artifact_, model_config );

        const std::vector<int32_t> prompt{ 760, 6511, 314, 9338, 369 };

        auto timeGeneration = [&]( int tokens ) -> double
            {
                std::vector<int32_t> produced;

                GenerateParams params;
                params.max_new_tokens = tokens;
                params.sampling.temperature = 0.0f;

                const auto start = std::chrono::steady_clock::now();

                const GenerateStatus status = model->generate(
                    prompt,
                    [&]( int32_t token ) { produced.push_back( token ); },
                    params,
                    std::stop_token{} );

                const auto elapsed = std::chrono::steady_clock::now() - start;

                EXPECT_EQ( static_cast<int>( produced.size() ), tokens )
                    << "generation stopped early (" << to_string( status )
                    << "), which breaks the subtraction";

                return std::chrono::duration<double>( elapsed ).count();
            };

        // Warm: the first generation pays for lazy cuBLASLt setup and first-touch paging.
        timeGeneration( 8 );

        constexpr int kShort = 8;
        constexpr int kLong = 72;

        const double short_seconds = timeGeneration( kShort );
        const double long_seconds = timeGeneration( kLong );

        const double per_token = ( long_seconds - short_seconds ) / ( kLong - kShort );
        const double tokens_per_second = 1.0 / per_token;

        // Section 5 derives its ceiling from bytes moved per token, so the achieved
        // bandwidth is the figure that says whether decode is bandwidth-bound at all --
        // and the spec's open question is precisely that these kernels are not.
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        cudaMemGetInfo( &free_bytes, &total_bytes );

        const double weight_gib = 8.69;
        const double achieved_gb_per_second = ( weight_gib * 1024 * 1024 * 1024 )
                                              / per_token / 1e9;

        std::cout << std::format(
            "  prefill+load cancelled by subtraction ({} vs {} tokens)\n"
            "  decode: {:.2f} ms/token, {:.1f} tok/s\n"
            "  implied weight bandwidth: {:.0f} GB/s "
            "(RTX 4070 peak 504, Section 5 ceiling 47 tok/s)\n",
            kShort, kLong, per_token * 1000.0, tokens_per_second,
            achieved_gb_per_second ) << std::flush;

        EXPECT_GT( tokens_per_second, 0.0 );
    }

    // ====================================================================
    // Coherence
    //
    // Every other test here checks that generated ids are inside the vocabulary, which
    // catches NaN logits and index errors and NOTHING ELSE -- a 2.82-bit model emitting
    // plausible garbage passes all of them. These two read the actual text.
    // ====================================================================

    // The strongest single assertion available, because it ties to a number already of
    // record: on "The capital of France is", Phase 4 measured the HF reference AND Mila at
    // BF16 both choosing 11751 = " Paris", through all 64 layers on real weights. If the
    // packed 2.82-bit build still chooses it, quantization did not destroy the model --
    // and if it does not, that is the first thing anyone would want to know.
    TEST_F( QwenPackedArtifactTests, PackedModel_StillAnswersParis )
    {
        const fs::path tokenizer_path =
            fs::path( TEST_DATA_DIR ) / "models" / "qwen" / "qwen38_tokenizer.bin";

        if ( !fs::exists( tokenizer_path ) )
        {
            GTEST_SKIP() << "Qwen tokenizer not present at: " << tokenizer_path.string();
        }

        std::shared_ptr<Mila::Data::BpeTokenizer> tokenizer =
            Mila::Data::BpeTokenizer::loadQwen( tokenizer_path );

        const auto encoded = tokenizer->encode( "The capital of France is" );
        std::vector<int32_t> prompt;

        for ( const auto id : encoded )
        {
            prompt.push_back( static_cast<int32_t>( id ) );
        }

        QwenModelConfig model_config( 512 );
        model_config.withPrecisionPlan();

        auto model = QwenBf16::fromPretrained( artifact_, model_config );

        std::vector<int32_t> produced;

        GenerateParams params;
        params.max_new_tokens = 24;
        params.sampling.temperature = 0.0f;

        const GenerateStatus status = model->generate(
            prompt,
            [&]( int32_t token ) { produced.push_back( token ); },
            params,
            std::stop_token{} );

        ASSERT_FALSE( produced.empty() );

        std::cout << "  prompt:     \"The capital of France is\"\n"
                  << "  completion: \"" << tokenizer->decode( produced ) << "\"\n"
                  << "  status:     " << to_string( status ) << std::endl;

        // 11751 = " Paris". The assertion is on the FIRST token, which is the one the
        // Phase 4 parity gate measured; what follows it is read above, not asserted.
        const std::vector<int32_t> first_only{ produced.front() };

        EXPECT_EQ( produced.front(), 11751 )
            << "first token decoded as \"" << tokenizer->decode( first_only ) << "\"";
    }

    // A longer, open-ended completion, printed for a human read. No assertion beyond
    // non-degeneracy: whether prose is good is not a thing a test can decide, but a model
    // that has collapsed into a repetition loop is, and that is the failure mode Phase 0
    // saw at these bit widths before compensation was added.
    TEST_F( QwenPackedArtifactTests, PackedModel_ProducesNonDegenerateProse )
    {
        const fs::path tokenizer_path =
            fs::path( TEST_DATA_DIR ) / "models" / "qwen" / "qwen38_tokenizer.bin";

        if ( !fs::exists( tokenizer_path ) )
        {
            GTEST_SKIP() << "Qwen tokenizer not present at: " << tokenizer_path.string();
        }

        std::shared_ptr<Mila::Data::BpeTokenizer> tokenizer =
            Mila::Data::BpeTokenizer::loadQwen( tokenizer_path );

        const auto encoded = tokenizer->encode(
            "A gardener explains why compost matters:" );
        std::vector<int32_t> prompt;

        for ( const auto id : encoded )
        {
            prompt.push_back( static_cast<int32_t>( id ) );
        }

        QwenModelConfig model_config( 512 );
        model_config.withPrecisionPlan();

        auto model = QwenBf16::fromPretrained( artifact_, model_config );

        std::vector<int32_t> produced;

        GenerateParams params;
        params.max_new_tokens = 64;
        params.sampling.temperature = 0.0f;

        const GenerateStatus status = model->generate(
            prompt,
            [&]( int32_t token ) { produced.push_back( token ); },
            params,
            std::stop_token{} );

        ASSERT_FALSE( produced.empty() );

        std::cout << "  prompt:     \"A gardener explains why compost matters:\"\n"
                  << "  completion: \"" << tokenizer->decode( produced ) << "\"\n"
                  << "  status:     " << to_string( status ) << std::endl;

        // Degeneracy check: a greedy run that emits one token over and over has collapsed.
        // Counted rather than eyeballed so it fails in CI as well as on a reading.
        const size_t distinct =
            std::set<int32_t>( produced.begin(), produced.end() ).size();

        EXPECT_GT( distinct, produced.size() / 4 )
            << "only " << distinct << " distinct tokens in " << produced.size()
            << " -- the model has collapsed into a repetition loop";
    }

    // The whole claim, end to end: 27B on a 12 GiB card, generating.
    //
    // 512 CONTEXT, NOT THE 16K BASELINE, and the number is measured rather than chosen.
    // getDeploymentFootprint predicts 9.94 GiB here and the load actually consumes the whole
    // 10.85 GiB the card has free with a desktop running -- a ~0.9 GiB gap that the model's
    // own accounting does not see (the CUDA context, the cuBLASLt workspace, per-allocation
    // rounding; the same residual BACKLOG records as unattributed on Gemma). At 2048 the
    // prediction is 10.12 GiB and the load dies, so the gap, not the prediction, is what
    // bounds the context today. Raise this the moment the residual is attributed or the run
    // moves to a headless card.
    TEST_F( QwenPackedArtifactTests, Generation_RunsAndStaysInsideTheVocabulary )
    {
        QwenModelConfig model_config( 512 );
        model_config.withPrecisionPlan();

        size_t free_before = 0;
        size_t total_bytes = 0;
        cudaMemGetInfo( &free_before, &total_bytes );

        std::cout << "  before load: " << ( free_before / ( 1024.0 * 1024 * 1024 ) )
                  << " GiB free" << std::endl;

        auto model = QwenBf16::fromPretrained( artifact_, model_config );

        size_t free_after = 0;
        cudaMemGetInfo( &free_after, &total_bytes );

        // What the load actually took, against what getDeploymentFootprint predicted. The
        // two are not expected to match exactly -- the CUDA context, the cuBLASLt workspace
        // and per-allocation rounding are outside the model's own accounting -- but the gap
        // is the number a deployment decision rests on, so it is printed rather than assumed.
        std::cout << "  load consumed "
                  << ( ( free_before - free_after ) / ( 1024.0 * 1024 * 1024 ) )
                  << " GiB device, " << ( free_after / ( 1024.0 * 1024 * 1024 ) )
                  << " GiB still free" << std::endl;

        const std::vector<int32_t> prompt{ 760, 6511, 314, 9338, 369 };
        std::vector<int32_t> produced;

        GenerateParams params;
        params.max_new_tokens = 8;
        params.sampling.temperature = 0.0f;

        const GenerateStatus status = model->generate(
            prompt,
            [&]( int32_t token ) { produced.push_back( token ); },
            params,
            std::stop_token{} );

        // Eight new tokens from a five-token prompt at 2048 context: the run ends either by
        // reaching the cap or on a stop token. ContextOverflow here would mean the bound
        // arithmetic is wrong, which is the one outcome that would not be visible in the
        // tokens themselves.
        EXPECT_NE( status, GenerateStatus::ContextOverflow );
        EXPECT_FALSE( produced.empty() );

        const int32_t vocabulary_size =
            static_cast<int32_t>( model->getNetworkConfig().getVocabSize() );

        for ( const int32_t token : produced )
        {
            EXPECT_GE( token, 0 );
            EXPECT_LT( token, vocabulary_size );
        }
    }
}
