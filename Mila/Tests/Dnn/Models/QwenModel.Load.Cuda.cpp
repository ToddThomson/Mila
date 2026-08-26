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
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <chrono>
#include <cstdint>
#include <format>
// C stdio rather than <fstream>: including any input-stream header in a TU that does
// `import Mila;` leaves std::basic_istream::sentry incomplete, and the failure surfaces
// inside <istream> rather than at the include.
#include <cstdio>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <stop_token>

import Mila;

// Instantiating a CUDA component consumer-side reaches CudaGqaOp::build and CudaLinearOp,
// which need ExecutionContext<Cuda> COMPLETE rather than merely reachable, and Mila.ixx
// exports only IExecutionContext. Same import Qwen.Block.Cuda.cpp had to add.
import Compute.ExecutionContext;

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

        /**
         * @brief Tokenize the held-out corpus. Empty when the corpus or tokenizer is absent.
         *
         * wikitext-2 TEST, and the split matters: the packer calibrates on wiki.train.raw, so
         * scoring the model on that text would measure how well it memorized its own
         * calibration set. The corpus directory is gitignored -- a download, not a repository
         * artifact -- hence the Shakespeare fallback and the empty return.
         */
        std::vector<int32_t> loadCorpusTokens( dim_t token_budget, dim_t context_length,
            fs::path& corpus_path )
        {
            const fs::path tokenizer_path =
                fs::path( TEST_DATA_DIR ) / "models" / "qwen" / "qwen38_tokenizer.bin";

            if ( !fs::exists( tokenizer_path ) )
            {
                return {};
            }

            const fs::path repository_root = fs::path( TEST_DATA_DIR ).parent_path();

            const fs::path candidates[] = {
                repository_root / "Mila" / "Tools" / "Quantization" / "corpus" / "wiki.test.raw",
                fs::path( TEST_DATA_DIR ) / "datasets" / "Shakespeare" / "raw" / "TinyShakespeare.txt"
            };

            corpus_path.clear();

            for ( const fs::path& candidate : candidates )
            {
                if ( fs::exists( candidate ) )
                {
                    corpus_path = candidate;
                    break;
                }
            }

            if ( corpus_path.empty() )
            {
                return {};
            }

            std::FILE* corpus_file = std::fopen( corpus_path.string().c_str(), "rb" );

            if ( corpus_file == nullptr )
            {
                return {};
            }

            // Four characters per token is a deliberate over-read: encoding more text than the
            // budget needs and truncating is correct, while under-reading would silently score
            // fewer positions than asked for.
            std::string text( static_cast<size_t>( token_budget ) * 4, '\0' );
            const size_t characters_read = std::fread( text.data(), 1, text.size(), corpus_file );
            std::fclose( corpus_file );

            text.resize( characters_read );

            auto tokenizer = Mila::Data::BpeTokenizer::loadQwen( tokenizer_path );
            auto tokens = tokenizer->encode( text );

            if ( tokens.size() <= static_cast<size_t>( context_length ) )
            {
                return {};
            }

            if ( tokens.size() > static_cast<size_t>( token_budget ) )
            {
                tokens.resize( static_cast<size_t>( token_budget ) );
            }

            return tokens;
        }

        /// One arm of the gate at one segment length.
        struct ArmResult
        {
            double perplexity{ 0.0 };
            double mean_negative_log_probability{ 0.0 };
            dim_t  scored_positions{ 0 };
            double elapsed_seconds{ 0.0 };
        };

        /**
         * @brief Score a token stream under one deployment.
         *
         * Shared by both arms so the protocol cannot drift between them: the comparison is
         * only meaningful if segmentation, head width and corpus are identical and the
         * allocation is the only difference.
         */
        ArmResult scoreCorpus( const fs::path& artifact, const QwenModelConfig& model_config,
            const std::vector<int32_t>& tokens, dim_t context_length )
        {
            // Announced because the oracle arm reads a 50 GiB blob and is otherwise silent
            // for half a minute before any progress is visible.
            std::cout << "  loading " << artifact.filename().string()
                << " at context " << context_length << " ...\n" << std::flush;

            auto model = QwenBf16::fromPretrained( artifact, model_config );

            SequenceLogLikelihood total;

            const auto start = std::chrono::steady_clock::now();

            for ( size_t offset = 0; offset + 1 < tokens.size(); offset += context_length )
            {
                const size_t length =
                    std::min<size_t>( static_cast<size_t>( context_length ), tokens.size() - offset );

                // A one-token tail scores nothing and cannot be scored -- scoreTokens needs
                // two. Dropping it costs one position of a segment's worth.
                if ( length < 2 )
                {
                    break;
                }

                const std::vector<int32_t> segment(
                    tokens.begin() + static_cast<std::ptrdiff_t>( offset ),
                    tokens.begin() + static_cast<std::ptrdiff_t>( offset + length ) );

                const SequenceLogLikelihood scored = model->scoreTokens( segment );

                total.total_log_probability += scored.total_log_probability;
                total.scored_positions += scored.scored_positions;
            }

            ArmResult result;

            result.elapsed_seconds =
                std::chrono::duration<double>( std::chrono::steady_clock::now() - start ).count();
            result.scored_positions = total.scored_positions;

            if ( total.scored_positions > 0 )
            {
                result.mean_negative_log_probability =
                    -total.total_log_probability / static_cast<double>( total.scored_positions );
                result.perplexity = std::exp( result.mean_negative_log_probability );
            }

            return result;
        }

        /// One arm's response to the whole prompt set, from a single load.
        /**
         * @brief What the reference thinks of two roads taken from the same prompt.
         *
         * Both continuations scored teacher-forced under the ORACLE, over the same number of
         * tokens. The prompt's own positions contribute identically to each and cancel in the
         * difference, so what survives is exactly the continuation.
         */
        struct TrajectoryComparison
        {
            double oracle_path_log_probability{ 0.0 };
            double plan_path_log_probability{ 0.0 };
            dim_t  compared_tokens{ 0 };

            /// Nats per token the plan's road gives up, judged by the oracle.
            double costPerToken() const
            {
                return compared_tokens > 0
                    ? ( oracle_path_log_probability - plan_path_log_probability )
                        / static_cast<double>( compared_tokens )
                    : 0.0;
            }
        };

        struct ArmOutcome
        {
            std::vector<std::vector<int32_t>> generated;
            std::vector<std::vector<float>> last_logits;

            /// Filled only on the arm that also scores, which must be the oracle.
            std::vector<TrajectoryComparison> trajectories;
        };

        /**
         * @brief Greedy-generate and capture the final logit row for every prompt, one load.
         *
         * Order matters: the logits are read AFTER the generation for that prompt, so the
         * prefill they come from is a fresh one over the prompt alone. Reading them first
         * would be equivalent here -- prefill always starts at position zero and zeroes the
         * recurrent state -- but relying on that silently is how a state-carrying bug hides.
         */
        ArmOutcome runPromptSet( std::string_view label, const fs::path& artifact,
            const QwenModelConfig& model_config,
            const std::vector<std::vector<int32_t>>& prompts, int generated_tokens,
            const std::vector<std::vector<int32_t>>& other_arm_continuations = {} )
        {
            std::cout << "  loading " << artifact.filename().string()
                << " for " << label << " ...\n" << std::flush;

            auto model = QwenBf16::fromPretrained( artifact, model_config );

            // The logit row comes from OBSERVATION rather than a purpose-built accessor: the
            // head already publishes its output on every pass, and the first publication of a
            // generate() call is the prefill's -- the distribution over what follows the
            // prompt, before a single token has been sampled. Every later publication is a
            // decode step, so `capturing` closes the gate after the first.
            using DeviceLogits = Tensor<TensorDataType::BF16,
                typename DeviceTypeTraits<DeviceType::Cuda>::memory_resource>;

            std::vector<float> captured;
            bool capturing = false;

            const size_t observed = model->observe( "*.lm_head", ComputePassMask::inference(),
                [&]( std::string_view, ComputePass, std::string_view stage, const ITensor& value )
                {
                    if ( !capturing || stage != "output" )
                    {
                        return;
                    }

                    const auto* typed = dynamic_cast<const DeviceLogits*>( &value );

                    if ( typed == nullptr )
                    {
                        return;
                    }

                    auto host = toHost<TensorDataType::FP32>( *typed );

                    captured.assign( host.data(), host.data() + host.size() );
                    capturing = false;
                } );

            EXPECT_EQ( observed, 1u ) << "the head was not selected, so no logits will arrive";

            ArmOutcome outcome;

            for ( size_t index = 0; index < prompts.size(); ++index )
            {
                const std::vector<int32_t>& prompt = prompts[ index ];

                std::vector<int32_t> produced;

                GenerateParams params;
                params.max_new_tokens = generated_tokens;
                params.sampling.temperature = 0.0f;

                captured.clear();
                capturing = true;

                const GenerateStatus status = model->generate(
                    prompt,
                    [&]( int32_t token ) { produced.push_back( token ); },
                    params,
                    std::stop_token{} );

                // A run that stopped early is still comparable -- both arms may hit EOS at
                // different points, and where they do IS the divergence -- but a context
                // overflow would mean the harness, not the model, ended it.
                EXPECT_NE( status, GenerateStatus::ContextOverflow );

                EXPECT_FALSE( captured.empty() ) << "the head published nothing for this prompt";

                outcome.last_logits.push_back( captured );

                if ( index < other_arm_continuations.size() )
                {
                    outcome.trajectories.push_back( compareTrajectories(
                        *model, prompt, produced, other_arm_continuations[ index ] ) );
                }

                outcome.generated.push_back( std::move( produced ) );
            }

            // The sink captures locals by reference; detaching before they leave scope is the
            // discipline even though the model is about to be destroyed with its context.
            model->stopObserving();

            return outcome;
        }

        /**
         * @brief Score both arms' continuations under `oracle` and report what the plan gives up.
         *
         * This is the criterion that replaces a bare divergence index. Where two greedy runs
         * part is chaotic -- one near-tie flips the argmax and they never re-converge, which
         * two builds of the SAME precision already do (Qwen3.8.md section 8). What is not
         * chaotic is whether the road the plan took is worse, and the reference model is the
         * only thing entitled to judge that.
         *
         * Truncated to the shorter continuation so both are scored over the same number of
         * tokens; the prompt is common to both and cancels.
         *
         * The result should be POSITIVE: the oracle's own greedy path is locally optimal under
         * the oracle, so anything else should score no better. A negative value would say the
         * plan found a road the reference likes MORE than its own -- which greedy decoding
         * does not forbid, and which would be strong evidence the divergence is noise rather
         * than damage. Either way the sign is a check on the measurement.
         */
        TrajectoryComparison compareTrajectories( QwenBf16& oracle,
            const std::vector<int32_t>& prompt,
            const std::vector<int32_t>& oracle_continuation,
            const std::vector<int32_t>& plan_continuation )
        {
            TrajectoryComparison comparison;

            const size_t length =
                std::min( oracle_continuation.size(), plan_continuation.size() );

            if ( length == 0 )
            {
                return comparison;
            }

            auto scorePath = [&]( const std::vector<int32_t>& continuation )
                {
                    std::vector<int32_t> sequence = prompt;

                    sequence.insert( sequence.end(), continuation.begin(),
                        continuation.begin() + static_cast<std::ptrdiff_t>( length ) );

                    return oracle.scoreTokens( sequence ).total_log_probability;
                };

            comparison.oracle_path_log_probability = scorePath( oracle_continuation );
            comparison.plan_path_log_probability = scorePath( plan_continuation );
            comparison.compared_tokens = static_cast<dim_t>( length );

            return comparison;
        }

        /// Index of the largest logit.
        static size_t argMax( const std::vector<float>& logits )
        {
            return static_cast<size_t>(
                std::distance( logits.begin(), std::max_element( logits.begin(), logits.end() ) ) );
        }

        /**
         * @brief KL( softmax(left) || softmax(right) ) in nats, accumulated in double.
         *
         * Asymmetric on purpose and in this order: it weighs each disagreement by how much
         * probability the ORACLE put there, which is the question being asked -- what does
         * using the quantized distribution cost, judged against the reference. The reverse
         * direction would let the plan's own confident mistakes dominate.
         */
        static double klDivergence( const std::vector<float>& left, const std::vector<float>& right )
        {
            const float left_max = *std::max_element( left.begin(), left.end() );
            const float right_max = *std::max_element( right.begin(), right.end() );

            double left_sum = 0.0;
            double right_sum = 0.0;

            for ( size_t i = 0; i < left.size(); ++i )
            {
                left_sum += std::exp( static_cast<double>( left[ i ] - left_max ) );
                right_sum += std::exp( static_cast<double>( right[ i ] - right_max ) );
            }

            const double left_log_normalizer = std::log( left_sum );
            const double right_log_normalizer = std::log( right_sum );

            double divergence = 0.0;

            for ( size_t i = 0; i < left.size(); ++i )
            {
                const double left_log_probability =
                    static_cast<double>( left[ i ] - left_max ) - left_log_normalizer;
                const double right_log_probability =
                    static_cast<double>( right[ i ] - right_max ) - right_log_normalizer;

                divergence += std::exp( left_log_probability )
                    * ( left_log_probability - right_log_probability );
            }

            return divergence;
        }

        void reportArm( std::string_view label, const fs::path& corpus_path,
            dim_t context_length, dim_t head_positions, const ArmResult& result )
        {
            std::cout << std::format(
                "  arm: {}\n"
                "  corpus: {}\n"
                "  protocol: non-overlapping segments of {} tokens, teacher-forced, head width {}\n"
                "  scored positions: {}\n"
                "  mean negative log-likelihood: {:.4f} nats/token\n"
                "  PERPLEXITY: {:.3f}\n"
                "  elapsed: {:.1f} s ({:.1f} positions/s)\n",
                label, corpus_path.filename().string(), context_length, head_positions,
                result.scored_positions, result.mean_negative_log_probability,
                result.perplexity, result.elapsed_seconds,
                result.scored_positions / result.elapsed_seconds ) << std::flush;
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

        // Predicted before, measured after: the two together are what say whether
        // getRequiredMemory describes the allocation a load actually makes.
        const DeploymentFootprint predicted =
            QwenBf16::getDeploymentFootprint( fixture, model_config );

        size_t free_before = 0;
        size_t total_bytes = 0;
        cudaMemGetInfo( &free_before, &total_bytes );

        auto model = QwenBf16::fromPretrained( fixture, model_config );

        size_t free_after = 0;
        cudaMemGetInfo( &free_after, &total_bytes );

        const double gib = 1024.0 * 1024 * 1024;

        std::cout << std::format(
            "  4-layer packed: predicted {:.2f} GiB device, actually consumed {:.2f} GiB\n"
            "  ({} layers, chunk {})\n",
            predicted.memory.totalDeviceBytes() / gib,
            ( free_before - free_after ) / gib,
            static_cast<long long>( model->getNetworkConfig().getNumLayers() ),
            static_cast<long long>( predicted.prefill.chunk_rows ) ) << std::flush;

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

    // Gate B for Qwen, on the packed 4-layer fixture.
    //
    // Run on the FIXTURE rather than the full artifact deliberately: the 64-layer model
    // oversubscribes the card, so WDDM backs part of it in system memory and cudaMemGetInfo
    // -- which sees dedicated VRAM only -- reports a number that is not the allocation. At
    // 4 layers everything is resident and the three figures are comparable.
    //
    // Gemma's equivalent asserts predicted == reported EXACTLY, on the reasoning that the
    // prediction is an accounting of the same allocations getMemoryStats counts. Which of
    // those two Qwen violates is the whole diagnostic:
    //
    //   predicted < reported   a component's getRequiredMemory disagrees with its own
    //                          getMemoryStats -- findable by walking the tree.
    //   predicted == reported  but both below consumed: the buffers are invisible to Mila's
    //                          accounting altogether, i.e. allocated outside a component's
    //                          reported state.
    TEST_F( QwenPackedArtifactTests, DISABLED_GateB_PredictionAgainstActual )
    {
        const fs::path fixture =
            fs::path( TEST_DATA_DIR ) / "models" / "qwen" / "qwen38_27b_l4_2p9bit.safetensors";

        if ( !fs::exists( fixture ) )
        {
            GTEST_SKIP() << "Packed 4-layer fixture not present at: " << fixture.string();
        }

        constexpr dim_t kContext = 512;
        const double gib = 1024.0 * 1024 * 1024;

        auto freeBytes = [] () -> std::size_t
            {
                std::size_t free_bytes = 0;
                std::size_t total_bytes = 0;
                cudaMemGetInfo( &free_bytes, &total_bytes );

                return free_bytes;
            };

        cudaFree( nullptr );

        QwenModelConfig model_config( kContext );
        model_config.withPrecisionPlan();

        const MemoryStats predicted =
            QwenBf16::getRequiredMemory( fixture, model_config );

        const std::size_t free_before = freeBytes();

        auto model = QwenBf16::fromPretrained( fixture, model_config );

        ASSERT_NE( model, nullptr );

        const std::size_t consumed = free_before - freeBytes();
        const MemoryStats reported = model->getMemoryStats();

        std::cout << std::format(
            "[gate B] Qwen 4-layer packed, context {}\n"
            "  predicted (getRequiredMemory) {:.3f} GiB  (params {:.3f} / state {:.3f})\n"
            "  reported  (getMemoryStats)    {:.3f} GiB  (params {:.3f} / state {:.3f})\n"
            "  consumed  (cudaMemGetInfo)    {:.3f} GiB\n"
            "  scratch high-water            {:.3f} GiB\n",
            static_cast<long long>( kContext ),
            predicted.totalDeviceBytes() / gib,
            predicted.device_parameter_bytes / gib,
            predicted.device_state_bytes / gib,
            reported.totalDeviceBytes() / gib,
            reported.device_parameter_bytes / gib,
            reported.device_state_bytes / gib,
            consumed / gib,
            model->getScratchHighWaterBytes() / gib ) << std::flush;

        EXPECT_EQ( predicted.device_parameter_bytes, reported.device_parameter_bytes );
        EXPECT_EQ( predicted.device_state_bytes, reported.device_state_bytes );
    }

    // Where a layer's device state actually sits, component by component.
    //
    // Gate B says the prediction understates state by ~95 MB per layer while getMemoryStats
    // sees it, so the bytes are held by some component that the block's getRequiredMemory
    // mis-sizes. This prints the built tree's own accounting so the component can be named
    // rather than reasoned about.
    //   MilaTests --gtest_also_run_disabled_tests
    //       --gtest_filter=QwenPackedArtifactTests.DISABLED_WhereTheLayerStateSits
    TEST_F( QwenPackedArtifactTests, DISABLED_WhereTheLayerStateSits )
    {
        const fs::path fixture =
            fs::path( TEST_DATA_DIR ) / "models" / "qwen" / "qwen38_27b_l4_2p9bit.safetensors";

        if ( !fs::exists( fixture ) )
        {
            GTEST_SKIP() << "Packed 4-layer fixture not present at: " << fixture.string();
        }

        // The transformer is built directly rather than loaded through QwenModel: only the
        // STATE allocation is in question, and building allocates it without reading a
        // single weight. Seconds instead of minutes, and no 5 GiB of I/O.
        Serialization::PretrainedModelReader reader( fixture );
        const QwenConfig network_config =
            QwenBf16::configFromMetadata( reader.getPretrainedMetadata() );

        using QwenPacked = QwenTransformer<DeviceType::Cuda, TensorDataType::BF16,
            QwenPrecisionPlan, Mila::Dnn::Quant::KvCache::NoKvCompression>;

        QwenPacked network( "qwen", network_config, Device::Cuda( 0 ) );
        network.build( BuildContext( shape_t{ 1, 512 }, RuntimeMode::Inference, false ) );

        const double mib = 1024.0 * 1024;

        for ( const auto& layer : network.getComponents() )
        {
            const MemoryStats layer_stats = layer->getMemoryStats();

            if ( layer_stats.device_state_bytes < 1024 * 1024 )
            {
                continue;
            }

            // The block's own prediction for itself, against what it actually holds. Agreement
            // puts the mis-sizing at the transformer level; disagreement puts it inside the
            // block's getRequiredMemory.
            MemoryStats layer_predicted;

            try
            {
                layer_predicted = layer->getRequiredMemory(
                    BuildContext( shape_t{ 1, 512, network_config.getModelDim() },
                        RuntimeMode::Inference, false ) );
            }
            catch ( const std::exception& e )
            {
                std::cout << "      (self-prediction threw: " << e.what() << ")\n";
            }

            std::cout << std::format(
                "\n  {} -- state {:.1f} MiB actual, {:.1f} MiB self-predicted\n",
                layer->getName(), layer_stats.device_state_bytes / mib,
                layer_predicted.device_state_bytes / mib );

            const auto* composite =
                dynamic_cast<const CompositeComponent<DeviceType::Cuda, TensorDataType::BF16>*>(
                    layer.get() );

            if ( composite == nullptr )
            {
                continue;
            }

            for ( const auto& child : composite->getComponents() )
            {
                const MemoryStats child_stats = child->getMemoryStats();

                if ( child_stats.device_state_bytes >= 1024 * 1024 )
                {
                    std::cout << std::format( "      {:<44} {:>8.1f} MiB\n",
                        child->getName(), child_stats.device_state_bytes / mib );
                }
            }
        }

        std::cout << std::flush;
    }

    // Decode cost per LAYER, on a model small enough to be genuinely VRAM-resident.
    //
    // The control for the full artifact's rate. The 4-layer packed fixture runs the same
    // kernels on the same policies at ~1.2 GiB resident, where nothing can be evicted.
    //
    // If per-layer cost matches between the two, the GEMVs are simply slow. If the full
    // model is several times worse per layer, its weights are not resident and the rate is
    // measuring PCIe. Nothing else in the two runs differs.
    //
    // That is how the residency defect was found and it is what settled it: the full model
    // read 4.7 tok/s against this control's 18.9, and once the DeltaNet activations were
    // pooled it reads 33.7. The extrapolation below UNDERSTATES for a known reason -- it
    // charges each of these four layers a share of lm_head and then multiplies it back
    // sixteenfold -- so the full model coming in faster than the control is expected.
    //   MilaTests --gtest_also_run_disabled_tests
    //       --gtest_filter=QwenPackedArtifactTests.DISABLED_DecodeRatePerLayer
    TEST_F( QwenPackedArtifactTests, DISABLED_DecodeRatePerLayer )
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

        size_t free_bytes = 0;
        size_t total_bytes = 0;
        cudaMemGetInfo( &free_bytes, &total_bytes );

        const std::vector<int32_t> prompt{ 760, 6511, 314, 9338, 369 };

        auto timeGeneration = [&]( int tokens ) -> double
            {
                std::vector<int32_t> produced;

                GenerateParams params;
                params.max_new_tokens = tokens;
                params.sampling.temperature = 0.0f;

                const auto start = std::chrono::steady_clock::now();

                model->generate(
                    prompt,
                    [&]( int32_t token ) { produced.push_back( token ); },
                    params,
                    std::stop_token{} );

                return std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - start ).count();
            };

        timeGeneration( 8 );

        constexpr int kShort = 8;
        constexpr int kLong = 72;

        const double per_token =
            ( timeGeneration( kLong ) - timeGeneration( kShort ) ) / ( kLong - kShort );

        // lm_head runs once per token regardless of depth, so it is NOT divided out; the
        // comparison is deliberately crude, because the effect being tested for is 5x.
        const int layers = static_cast<int>( model->getNetworkConfig().getNumLayers() );

        std::cout << std::format(
            "  4-layer packed: {:.2f} ms/token over {} layers, {:.2f} ms/layer\n"
            "  {:.2f} GiB still free after load\n"
            "  extrapolated to 64 layers: {:.0f} ms/token, {:.1f} tok/s (understates -- see above)\n"
            "  measured on the 64-layer model: 29.7 ms/token, 33.7 tok/s\n",
            per_token * 1000.0, layers, per_token * 1000.0 / layers,
            free_bytes / ( 1024.0 * 1024 * 1024 ),
            per_token * 1000.0 / layers * 64, 1.0 / ( per_token / layers * 64 ) )
            << std::flush;

        EXPECT_GT( per_token, 0.0 );
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

    // ====================================================================
    // Corpus perplexity -- the Phase 5 quality gate (Qwen3.8.md section 8, item 9)
    //
    // Perplexity is only comparable between runs that used the SAME protocol, so this one
    // states its own and the printed line repeats it. The protocol here:
    //
    //   - the corpus is tokenized once, as one stream;
    //   - the stream is cut into NON-OVERLAPPING segments of the deployment context length;
    //   - each segment is scored independently, from a cold cache and a zeroed recurrent
    //     state, and the log-probabilities are summed across segments;
    //   - the first token of each segment scores nothing, having no context, so a segment of
    //     N tokens contributes N-1 positions;
    //   - perplexity is exp( -total / positions ).
    //
    // Non-overlapping segments are the cheap protocol, not the flattering one: every segment
    // spends its early positions predicting from almost no context, which a sliding window
    // would avoid. It reads slightly worse than a sliding-window number on the same text and
    // is what both arms of a quantization comparison must use.
    // ====================================================================

    // A measurement, so DISABLED:
    //   MilaTests --gtest_also_run_disabled_tests
    //       --gtest_filter=QwenPackedArtifactTests.DISABLED_CorpusPerplexity
    TEST_F( QwenPackedArtifactTests, DISABLED_CorpusPerplexity )
    {
        // Cost is linear in scored positions: the head runs once per token and the reduction
        // is on the host, together about 190 positions/s. 16K tokens is ~90 seconds and is
        // far more than enough to separate two quantizations, whose perplexities differ by
        // percent. The whole wikitext-2 test split is ~300K tokens, so a full-corpus number
        // is ~26 minutes at this rate -- worth it for a published figure, not for iteration.
        constexpr dim_t kTokenBudget = 16384;
        constexpr dim_t kContextLength = 1024;

        // 64 rows per head pass. This was 1 until the W4A8-FP8 prefill path learned to
        // stripe its weight expansion: unstriped it asked for 1212.5 MiB of staging for the
        // head whatever the row count, which did not fit beside the model and aborted.
        constexpr dim_t kHeadPositions = 64;

        fs::path corpus_path;
        const std::vector<int32_t> tokens = loadCorpusTokens( kTokenBudget, kContextLength, corpus_path );

        if ( tokens.empty() )
        {
            GTEST_SKIP() << "No corpus or tokenizer available";
        }

        QwenModelConfig model_config( kContextLength );
        model_config.withPrecisionPlan()
            .withLanguageModelHeadPositions( kHeadPositions );

        const ArmResult result = scoreCorpus( artifact_, model_config, tokens, kContextLength );

        reportArm( "Section 5 plan, 2.82 bits", corpus_path, kContextLength, kHeadPositions, result );

        EXPECT_GT( result.perplexity, 1.0 );
    }

    // The other arm of the gate. DISABLED, and slower than the one above: it reads the
    // 50 GiB reference blob and quantizes on the way in.
    //   MilaTests --gtest_also_run_disabled_tests
    //       --gtest_filter=QwenPackedArtifactTests.DISABLED_CorpusPerplexityFp4Oracle
    //
    // Uniform FP4 over the same weights -- 4.125 bits everywhere the plan quantizes, 12.31
    // GiB, which needs the 16 GiB card. Section 5's exit criterion is the RATIO of this to
    // the packed number, so the two must be read together and under one protocol: same
    // corpus, same segment length, same head width. Only the allocation differs.
    //
    // RUN BOTH ARMS ON ONE CARD, pinned by UUID. CUDA's device 0 is NOT nvidia-smi's: this
    // rig reports the 5060 Ti at nvidia-smi index 0 while CUDA orders the 4070 first, so the
    // default DeviceId{ Cuda, 0 } lands on the 12 GiB card and 12.31 GiB of weights aborts
    // there. Set CUDA_VISIBLE_DEVICES to the 16 GiB card's UUID. The two cards also disagree
    // in the last digits -- the packed arm measured 7.506 unpinned and 7.513 on the 5060 Ti,
    // which is float non-associativity between architectures, not a defect -- so a ratio
    // built from two different cards is not a measurement.
    TEST_F( QwenPackedArtifactTests, DISABLED_CorpusPerplexityFp4Oracle )
    {
        constexpr dim_t kTokenBudget = 16384;
        constexpr dim_t kContextLength = 1024;
        constexpr dim_t kHeadPositions = 1;

        const fs::path reference_blob =
            fs::path( TEST_DATA_DIR ) / "models" / "qwen" / "qwen38_27b_bf16.bin";

        if ( !fs::exists( reference_blob ) )
        {
            GTEST_SKIP() << "Reference BF16 blob not present at: " << reference_blob.string();
        }

        fs::path corpus_path;
        const std::vector<int32_t> tokens = loadCorpusTokens( kTokenBudget, kContextLength, corpus_path );

        if ( tokens.empty() )
        {
            GTEST_SKIP() << "No corpus or tokenizer available";
        }

        QwenModelConfig model_config( kContextLength );
        model_config.withWeightQuantization( WeightQuantization::FP4 )
            .withLanguageModelHeadPositions( kHeadPositions );

        const ArmResult result = scoreCorpus( reference_blob, model_config, tokens, kContextLength );

        reportArm( "FP4 oracle, 4.125 bits", corpus_path, kContextLength, kHeadPositions, result );

        EXPECT_GT( result.perplexity, 1.0 );
    }

    // ====================================================================
    // The gate itself: both arms, three context lengths.
    //   MilaTests --gtest_also_run_disabled_tests
    //       --gtest_filter=QwenPackedArtifactTests.DISABLED_QualityGateAcrossContextLengths
    //
    // A ratio at one segment length is not the claim the model makes. 48 of the 64 layers
    // carry a recurrent state, which is where quantization error compounds ALONG a sequence
    // rather than across parameters, so a gap measured at 1024 tokens says nothing about the
    // 16K context this model is sold on. This sweep is what separates "the allocation is
    // sound" from "the allocation is sound at the length we happened to measure".
    //
    // Each cell is its own deployment: the context length sizes the KV cache and the prefill
    // chunk, so scoring 4K segments inside a 16K build would measure a deployment nobody
    // runs. Six loads, and they are cheap against the scoring.
    //
    // PIN THE CARD. Both arms must share one GPU (see the note above), and the oracle needs
    // the 16 GiB one -- 12.31 GiB of weights plus a 16K KV cache is close to its ceiling, so
    // the last row is the one that may not fit. Rows print as they complete, ascending, so a
    // failure at 16K still leaves 4K and 8K on the record.
    // ====================================================================
    TEST_F( QwenPackedArtifactTests, DISABLED_QualityGateAcrossContextLengths )
    {
        // Above every segment length, so each row scores the same span of the same corpus
        // and only the segmentation differs.
        constexpr dim_t kTokenBudget = 32768;
        constexpr dim_t kHeadPositions = 1;
        constexpr dim_t kSegmentLengths[] = { 4096, 8192, 16384 };

        const fs::path reference_blob =
            fs::path( TEST_DATA_DIR ) / "models" / "qwen" / "qwen38_27b_bf16.bin";

        if ( !fs::exists( reference_blob ) )
        {
            GTEST_SKIP() << "Reference BF16 blob not present at: " << reference_blob.string();
        }

        std::cout << "\n  context | oracle 4.125b | plan 2.82b | ratio | positions\n"
                  << "  --------+---------------+------------+-------+----------\n" << std::flush;

        for ( const dim_t context_length : kSegmentLengths )
        {
            fs::path corpus_path;
            const std::vector<int32_t> tokens =
                loadCorpusTokens( kTokenBudget, context_length, corpus_path );

            if ( tokens.empty() )
            {
                GTEST_SKIP() << "No corpus or tokenizer available";
            }

            QwenModelConfig oracle_config( context_length );
            oracle_config.withWeightQuantization( WeightQuantization::FP4 )
                .withLanguageModelHeadPositions( kHeadPositions );

            const ArmResult oracle = scoreCorpus( reference_blob, oracle_config, tokens, context_length );

            QwenModelConfig packed_config( context_length );
            packed_config.withPrecisionPlan()
                .withLanguageModelHeadPositions( kHeadPositions );

            const ArmResult packed = scoreCorpus( artifact_, packed_config, tokens, context_length );

            ASSERT_GT( oracle.scored_positions, 0 );
            ASSERT_GT( packed.scored_positions, 0 );
            ASSERT_EQ( oracle.scored_positions, packed.scored_positions )
                << "the two arms scored different positions, so the ratio compares nothing";

            std::cout << std::format(
                "  {:>7} | {:>13.3f} | {:>10.3f} | {:>5.3f} | {:>9}\n",
                context_length, oracle.perplexity, packed.perplexity,
                packed.perplexity / oracle.perplexity, packed.scored_positions ) << std::flush;

            EXPECT_GT( packed.perplexity, oracle.perplexity )
                << "fewer bits scoring BETTER is a defect in the measurement, not a result";
        }
    }

    // ====================================================================
    // Section 5's other two quality criteria: divergence point and logit divergence.
    //   MilaTests --gtest_also_run_disabled_tests
    //       --gtest_filter=QwenPackedArtifactTests.DISABLED_DivergenceAgainstTheOracle
    //
    // Perplexity is prose next-token accuracy averaged over tens of thousands of positions,
    // which is the property a quantized model keeps longest. These two read what it cannot:
    //
    //   - DIVERGENCE POINT: how many tokens of matched greedy generation the two arms agree
    //     on before they part. Averaging cannot see this. Two models with the same perplexity
    //     can fork at token 3 or track each other for a hundred, and for anything agentic --
    //     a tool call, a JSON body, a chain of reasoning -- that difference is the product.
    //   - LOGIT DIVERGENCE: how far apart the two distributions are at a fixed position,
    //     as KL( oracle || plan ) in nats. The sampled token says only which side of a
    //     boundary the argmax fell on; the KL says whether the model was nearly indifferent
    //     or confidently disagreed.
    //
    // The KL doubles as a cross-check on the perplexity gate. ln( 1.139 ) = 0.130 nats is the
    // mean log-probability the plan gives up per token, so a mean KL in that neighbourhood
    // means two independent measurements agree; a mean KL far below it would say the gap
    // lives somewhere the prompt set is not looking.
    //
    // Both arms, one card, greedy (temperature 0). Two loads, shared.
    // ====================================================================
    TEST_F( QwenPackedArtifactTests, DISABLED_DivergenceAgainstTheOracle )
    {
        constexpr dim_t kContextLength = 1024;
        constexpr int kGeneratedTokens = 128;

        // Fixed and stated, because a divergence point means nothing without the prompt that
        // produced it. Chosen to spread across what the model is actually asked to do rather
        // than to flatter: recall, code, exposition, narrative and an instruction.
        const std::vector<std::string> prompts = {
            "The capital of France is",
            "In 1969, humans first walked on the",
            "def fibonacci(n):\n    ",
            "The main difference between a list and a tuple in Python is",
            "Once upon a time, in a village at the edge of the forest,",
            "Summarize in one sentence: the committee met on Tuesday and approved the budget."
        };

        const fs::path tokenizer_path =
            fs::path( TEST_DATA_DIR ) / "models" / "qwen" / "qwen38_tokenizer.bin";
        const fs::path reference_blob =
            fs::path( TEST_DATA_DIR ) / "models" / "qwen" / "qwen38_27b_bf16.bin";

        if ( !fs::exists( tokenizer_path ) || !fs::exists( reference_blob ) )
        {
            GTEST_SKIP() << "Tokenizer or reference blob not present";
        }

        auto tokenizer = Mila::Data::BpeTokenizer::loadQwen( tokenizer_path );

        std::vector<std::vector<int32_t>> encoded;

        for ( const std::string& prompt : prompts )
        {
            encoded.push_back( tokenizer->encode( prompt ) );
        }

        QwenModelConfig oracle_config( kContextLength );
        oracle_config.withWeightQuantization( WeightQuantization::FP4 );

        QwenModelConfig packed_config( kContextLength );
        packed_config.withPrecisionPlan();

        // The PLAN loads first and the ORACLE second, which is not arbitrary: the oracle is
        // the only model entitled to judge either road, so it has to still be resident when
        // the trajectories are scored. Loading it last lets one load both generate its own
        // continuations and score the plan's -- two loads rather than three.
        const ArmOutcome packed =
            runPromptSet( "Section 5 plan", artifact_, packed_config, encoded, kGeneratedTokens );

        const ArmOutcome oracle = runPromptSet( "FP4 oracle", reference_blob, oracle_config,
            encoded, kGeneratedTokens, packed.generated );

        ASSERT_EQ( oracle.generated.size(), prompts.size() );
        ASSERT_EQ( packed.generated.size(), prompts.size() );
        ASSERT_EQ( oracle.trajectories.size(), prompts.size() );

        std::cout << "\n  prompt                          | forks at | KL(o||p) | top-1 | plan cost/tok\n"
                  <<   "  --------------------------------+----------+----------+-------+--------------\n"
                  << std::flush;

        double summed_kl = 0.0;
        double summed_trajectory_cost = 0.0;
        int top1_agreements = 0;
        int never_diverged = 0;

        for ( size_t index = 0; index < prompts.size(); ++index )
        {
            const std::vector<int32_t>& left = oracle.generated[ index ];
            const std::vector<int32_t>& right = packed.generated[ index ];

            const size_t compared = std::min( left.size(), right.size() );

            size_t divergence = compared;

            for ( size_t position = 0; position < compared; ++position )
            {
                if ( left[ position ] != right[ position ] )
                {
                    divergence = position;
                    break;
                }
            }

            const bool diverged = divergence < compared;

            if ( !diverged )
            {
                ++never_diverged;
            }

            const double kl = klDivergence( oracle.last_logits[ index ], packed.last_logits[ index ] );
            summed_kl += kl;

            const bool same_top1 =
                argMax( oracle.last_logits[ index ] ) == argMax( packed.last_logits[ index ] );

            if ( same_top1 )
            {
                ++top1_agreements;
            }

            std::string label = prompts[ index ];

            if ( label.size() > 31 )
            {
                label = label.substr( 0, 28 ) + "...";
            }

            std::replace( label.begin(), label.end(), '\n', ' ' );

            const double trajectory_cost = oracle.trajectories[ index ].costPerToken();
            summed_trajectory_cost += trajectory_cost;

            std::cout << std::format( "  {:<31} | {:>8} | {:>8.4f} | {:<5} | {:>13.4f}\n",
                label,
                diverged ? std::to_string( divergence ) : std::format( ">={}", compared ),
                kl, same_top1 ? "same" : "DIFF", trajectory_cost ) << std::flush;
        }

        const double mean_kl = summed_kl / static_cast<double>( prompts.size() );
        const double mean_trajectory_cost =
            summed_trajectory_cost / static_cast<double>( prompts.size() );

        std::cout << std::format(
            "\n  mean trajectory cost: {:.4f} nats/token -- what the plan's road gives up,\n"
            "                        judged by the oracle. THIS is the discriminating number.\n"
            "  mean KL at the prompt: {:.4f} nats\n"
            "  perplexity gate implies: {:.4f} nats  (ln 1.139)\n"
            "  top-1 agreement: {} of {}\n"
            "  generations that never forked within {} tokens: {} of {}\n",
            mean_trajectory_cost, mean_kl, std::log( 1.139 ),
            top1_agreements, prompts.size(),
            kGeneratedTokens, never_diverged, prompts.size() ) << std::flush;

        // The two arms are different models; identical output everywhere would mean the
        // harness loaded one model twice, which is the failure that would make every number
        // above meaningless.
        EXPECT_GT( summed_kl, 0.0 )
            << "the arms are indistinguishable -- did both loads resolve to the same weights?";

        // The oracle's own greedy path is locally optimal under the oracle, so the plan's
        // road should score no better. Negative would not be a bug in the model -- greedy is
        // not globally optimal -- but it would mean the divergence is at noise level, which
        // is a different conclusion and must not pass unnoticed.
        EXPECT_GE( mean_trajectory_cost, 0.0 )
            << "the plan's road scores BETTER under the oracle than the oracle's own -- "
               "read this as the fork being noise, not as the plan being better";
    }

    // ====================================================================
    // Where scoring's time actually goes.
    //   MilaTests --gtest_also_run_disabled_tests
    //       --gtest_filter=QwenPackedArtifactTests.DISABLED_ScoringCostBreakdown
    //
    // Widening the head from 1 position to 64 bought only 1.4x, which says the head was never
    // what scoring spends its time on. This separates the two things that could be: the model
    // forward itself, which no amount of optimizing the scoring path can remove, and the
    // per-position overhead scoring adds on top of it -- the head passes, the device-to-host
    // transfer of each logit row, and the host-side log-probability reduction.
    //
    // The baseline is a one-token generate() over the same segment. That is a full prefill
    // plus a single decode step, and it is the closest thing to a prefill-only measurement
    // the public surface offers; the one decode step is noise against a 1023-token prefill.
    //
    // What the answer decides: whether a device-side reduction is worth building. If the
    // overhead is small against the forward, it is not, whatever the arithmetic below says
    // about exp() calls.
    // ====================================================================
    TEST_F( QwenPackedArtifactTests, DISABLED_ScoringCostBreakdown )
    {
        constexpr dim_t kContextLength = 1024;

        // One short of the context so the baseline's single decode step has somewhere to go.
        constexpr dim_t kSegmentLength = 1023;
        constexpr dim_t kTokenBudget = 8192;
        constexpr dim_t kHeadPositions = 64;

        fs::path corpus_path;
        const std::vector<int32_t> tokens =
            loadCorpusTokens( kTokenBudget, kSegmentLength, corpus_path );

        if ( tokens.empty() )
        {
            GTEST_SKIP() << "No corpus or tokenizer available";
        }

        QwenModelConfig model_config( kContextLength );
        model_config.withPrecisionPlan()
            .withLanguageModelHeadPositions( kHeadPositions );

        std::cout << "  loading " << artifact_.filename().string() << " ...\n" << std::flush;

        auto model = QwenBf16::fromPretrained( artifact_, model_config );

        std::vector<std::vector<int32_t>> segments;

        for ( size_t offset = 0; offset + 1 < tokens.size(); offset += kSegmentLength )
        {
            const size_t length =
                std::min<size_t>( static_cast<size_t>( kSegmentLength ), tokens.size() - offset );

            if ( length < 2 )
            {
                break;
            }

            segments.emplace_back(
                tokens.begin() + static_cast<std::ptrdiff_t>( offset ),
                tokens.begin() + static_cast<std::ptrdiff_t>( offset + length ) );
        }

        ASSERT_FALSE( segments.empty() );

        // Warm: the first pass of either kind pays lazy cuBLASLt plan selection and
        // first-touch paging, and charging that to whichever ran first would invent a
        // difference between them.
        {
            GenerateParams warm_params;
            warm_params.max_new_tokens = 1;
            warm_params.sampling.temperature = 0.0f;

            (void)model->generate( segments.front(), []( int32_t ) {}, warm_params,
                std::stop_token{} );
            (void)model->scoreTokens( segments.front() );
        }

        double forward_seconds = 0.0;
        double scoring_seconds = 0.0;
        dim_t scored_positions = 0;

        for ( const std::vector<int32_t>& segment : segments )
        {
            GenerateParams params;
            params.max_new_tokens = 1;
            params.sampling.temperature = 0.0f;

            const auto forward_start = std::chrono::steady_clock::now();

            (void)model->generate( segment, []( int32_t ) {}, params, std::stop_token{} );

            forward_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - forward_start ).count();

            const auto scoring_start = std::chrono::steady_clock::now();

            const SequenceLogLikelihood scored = model->scoreTokens( segment );

            scoring_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - scoring_start ).count();

            scored_positions += scored.scored_positions;
        }

        ASSERT_GT( scored_positions, 0 );

        const double overhead_seconds = scoring_seconds - forward_seconds;
        const double vocabulary = static_cast<double>( model->getNetworkConfig().getVocabSize() );

        // What the transfer alone could possibly cost, so it can be ruled in or out rather
        // than assumed. BF16 on the wire, converted host-side.
        const double transferred_gigabytes =
            static_cast<double>( scored_positions ) * vocabulary * 2.0 / 1e9;

        std::cout << std::format(
            "\n  segments: {} of {} tokens; scored positions: {}\n"
            "  model forward (prefill baseline): {:.1f} s\n"
            "  scoring (forward + head + transfer + host reduction): {:.1f} s\n"
            "  scoring OVERHEAD: {:.1f} s  ({:.0f}% of scoring, {:.0f} us/position)\n"
            "\n"
            "  of that overhead, what the parts could be:\n"
            "    logit rows transferred: {:.1f} GB -> ~{:.1f} s at 12 GB/s\n"
            "    host exp() calls: {:.2f} billion -> ~{:.1f} s at 5 ns each\n",
            segments.size(), kSegmentLength, scored_positions,
            forward_seconds, scoring_seconds,
            overhead_seconds, 100.0 * overhead_seconds / scoring_seconds,
            1e6 * overhead_seconds / static_cast<double>( scored_positions ),
            transferred_gigabytes, transferred_gigabytes / 12.0,
            static_cast<double>( scored_positions ) * vocabulary / 1e9,
            static_cast<double>( scored_positions ) * vocabulary * 5e-9 ) << std::flush;

        EXPECT_GT( scoring_seconds, forward_seconds )
            << "scoring cannot be cheaper than the forward it contains";
    }
}
