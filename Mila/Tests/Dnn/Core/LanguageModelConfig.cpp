/**
 * @file LanguageModelConfig.cpp
 * @brief Contract tests for the stored-weights quantization rule.
 *
 * requireStoredQuantizationMatches is the one place every family decides whether the weights
 * on disk may be loaded by the policy the build compiled for. It was previously authored
 * inline in two families and absent from the other two, and its failure mode is silent: the
 * wrong answer produces a model that loads, runs, and is wrong. Device-free and fixture-free,
 * so it rides the CPU-only CI gate.
 */

#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <string_view>

import Mila;

namespace Mila::Tests::Dnn::Core
{
    using namespace Mila::Dnn;

    namespace
    {
        constexpr const char* kCaller = "TestModel::fromPretrained";
        constexpr const char* kPath = "weights.safetensors";

        void check( std::string_view stored, WeightQuantization requested )
        {
            requireStoredQuantizationMatches( kCaller, kPath, stored, requested );
        }
    }

    // ====================================================================
    // Reference weights: derivable formats are computed at load time
    // ====================================================================

    TEST( LanguageModelConfigQuantizationRule, ReferenceWeightsAcceptReferencePrecision )
    {
        EXPECT_NO_THROW( check( "", WeightQuantization::None ) );
    }

    TEST( LanguageModelConfigQuantizationRule, ReferenceWeightsAcceptDerivableFormats )
    {
        EXPECT_NO_THROW( check( "", WeightQuantization::FP4 ) );
        EXPECT_NO_THROW( check( "", WeightQuantization::FP8 ) );
    }

    // A plan's codebooks are fitted offline against calibration data, so nothing in a
    // reference tensor recovers them. This is the asymmetry the rule exists to express.
    TEST( LanguageModelConfigQuantizationRule, ReferenceWeightsRefuseAPlan )
    {
        EXPECT_THROW( check( "", WeightQuantization::Plan ), std::runtime_error );
    }

    // ====================================================================
    // Pre-quantized weights must match exactly, in either direction
    // ====================================================================

    TEST( LanguageModelConfigQuantizationRule, MatchingSchemeIsAccepted )
    {
        EXPECT_NO_THROW( check( "per_group_fp4_128", WeightQuantization::FP4 ) );
        EXPECT_NO_THROW( check( "per_channel_fp8_e4m3", WeightQuantization::FP8 ) );
        EXPECT_NO_THROW( check( "codebook", WeightQuantization::Plan ) );
    }

    // The storage dtype cannot stand in for the scheme name -- FP4 at group 128 and FP8 are
    // both stored as U8, so a load that ignored this would reinterpret the packed layout.
    TEST( LanguageModelConfigQuantizationRule, DifferentSchemeIsRefused )
    {
        EXPECT_THROW( check( "per_group_fp4_128", WeightQuantization::FP8 ), std::runtime_error );
        EXPECT_THROW( check( "per_channel_fp8_e4m3", WeightQuantization::FP4 ), std::runtime_error );
        EXPECT_THROW( check( "codebook", WeightQuantization::FP4 ), std::runtime_error );
    }

    // The direction GPT-2 takes: a family with no quantization policy at all must still
    // refuse weights that carry one, or it loads packed bytes as BF16.
    TEST( LanguageModelConfigQuantizationRule, QuantizedWeightsRefuseReferencePrecision )
    {
        EXPECT_THROW( check( "per_group_fp4_128", WeightQuantization::None ), std::runtime_error );
        EXPECT_THROW( check( "codebook", WeightQuantization::None ), std::runtime_error );
    }

    // ====================================================================
    // Message
    // ====================================================================

    // The caller is a parameter because one shared rule serves four families and two entry
    // points each; without it the message cannot say which load refused.
    TEST( LanguageModelConfigQuantizationRule, MessageNamesTheCallerAndBothSchemes )
    {
        try
        {
            check( "codebook", WeightQuantization::FP4 );
            FAIL() << "expected the mismatched scheme to be refused";
        }
        catch ( const std::runtime_error& error )
        {
            const std::string message = error.what();

            EXPECT_NE( message.find( kCaller ), std::string::npos );
            EXPECT_NE( message.find( "codebook" ), std::string::npos );
            EXPECT_NE( message.find( "per_group_fp4_128" ), std::string::npos );
        }
    }
}
