/**
 * @file GemmaModel.Parity.Cuda.cpp
 * @brief HuggingFace token-for-token greedy parity test for GemmaModel (Step 5f).
 *
 * The end-to-end correctness oracle for the Gemma 4 dense chassis: loads the
 * converted checkpoint via GemmaModel::fromPretrained and asserts that greedy
 * decode reproduces the HuggingFace reference token-for-token. The reference is
 * captured ONCE from the HF run and hardcoded here (the same pattern the tokenizer
 * suite uses for its HF ground truth):
 *
 *   python Dev/Scripts/gemma_4_BF16/hf_gemma_greedy_validation.py
 *
 * and paste its "kPromptIds = {...}" / "kExpectedGen = {...}" lines below. The
 * tokenizer is validated separately (BpeTokenizerGemma), so feeding the reference
 * prompt ids isolates MODEL parity: the embedding sqrt(d) scale, sandwich/QK norms,
 * dual RoPE, the local AND global K=V / head_dim-512 attention, GeGLU, and lm_head.
 *
 * Gated: skips unless a CUDA device is present, the reference ids are populated,
 * and the checkpoint exists at <TEST_DATA_DIR>/models/gemma/gemma4_12b_it_bf16.bin.
 * It is an opt-in integration test — never runs in CI.
 *
 * Quantization: the 12B BF16 weights (~24 GB) do not fit on the 12 GB dev card,
 * so the model is loaded with FP4 weight quantization (PerGroupFp4<128>), which
 * brings the resident device footprint to roughly a quarter of BF16. The BF16
 * checkpoint is still read from disk and quantized host-side at load time. The
 * HF reference was captured in BF16; greedy argmax over this short horizon is
 * robust to the FP4 perturbation, so token-for-token parity is still the bar. A
 * divergence here is a real signal (either an FP4-path bug or a horizon long
 * enough that quantization noise flips an argmax) and should be investigated.
 */

#include <gtest/gtest.h>
#include <filesystem>
#include <vector>
#include <cstdint>
#include <memory>

import Mila;

namespace Mila::Tests::Dnn::Models
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace fs = std::filesystem;

    namespace
    {
        using GemmaCudaBf16 = GemmaModel<DeviceType::Cuda, TensorDataType::BF16>;

        // ---- HuggingFace ground truth -------------------------------------------
        // Pasted from hf_gemma_greedy_validation.py output (greedy, bf16, add_special_tokens=True).
        const std::vector<int32_t> kPromptIds = { 2, 105, 2364, 107, 818, 5279, 529, 7001, 563, 106, 107, 105, 4368, 107, 100, 45518, 107, 101 };
        const std::vector<int32_t> kExpectedGen = { 818, 5279, 529, 7001, 563, 5213, 50429, 84750, 106 };

        constexpr int64_t kContextLength = 512;

        fs::path gemma_checkpoint_path()
        {
            return fs::path( TEST_DATA_DIR ) / "models" / "gemma" / "gemma4_12b_it_bf16.bin";
        }
    }

    class GemmaModelParityCudaTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            if ( getDeviceCount( DeviceType::Cuda ) == 0 )
            {
                GTEST_SKIP() << "No CUDA device available";
            }

            if ( kPromptIds.empty() || kExpectedGen.empty() )
            {
                GTEST_SKIP() << "Populate kPromptIds / kExpectedGen from "
                                "Dev/Scripts/gemma_4_BF16/hf_gemma_greedy_validation.py output.";
            }

            checkpoint_ = gemma_checkpoint_path();

            if ( !fs::exists( checkpoint_ ) )
            {
                GTEST_SKIP() << "Gemma checkpoint not present at: " << checkpoint_.string();
            }
        }

        fs::path checkpoint_;
    };

    TEST_F( GemmaModelParityCudaTests, GreedyDecode_MatchesHuggingFaceTokenForToken )
    {
        // FP4 weight quantization is required to fit the 12B model on the dev card.
        // withFP4Quantization() also requests FP8 KV compression; GemmaModel does not
        // yet support a compressed KV cache and loads NoKvCompression regardless.
        auto model_config = GemmaModelConfig( static_cast<dim_t>( kContextLength ) )
            .withFP4Quantization();

        auto model = GemmaCudaBf16::fromPretrained(
            checkpoint_, model_config, DeviceId{ DeviceType::Cuda, 0 } );

        // temperature 0 + top_k 1 -> greedy argmax (GemmaModel::sampleToken).
        auto out = model->generate(
            kPromptIds, kExpectedGen.size(), /*temperature=*/0.0f, /*top_k=*/1 );

        ASSERT_GE( out.size(), kPromptIds.size() );
        const std::vector<int32_t> gen( out.begin() + kPromptIds.size(), out.end() );

        // Mila omits the trailing EOS / <end_of_turn> that HF includes, so Mila's
        // output must be a token-for-token PREFIX of the HF reference, short by at
        // most that one stop token.
        ASSERT_LE( gen.size(), kExpectedGen.size() );
        ASSERT_GE( gen.size(), kExpectedGen.size() - 1 )
            << "Mila stopped early: " << gen.size() << " tokens vs HF " << kExpectedGen.size();

        for ( size_t i = 0; i < gen.size(); ++i )
        {
            EXPECT_EQ( gen[ i ], kExpectedGen[ i ] ) << "Greedy divergence at generated token " << i;
        }
    }
}
