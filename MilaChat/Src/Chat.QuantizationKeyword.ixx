/**
 * @file Chat.QuantizationKeyword.ixx
 * @brief Chat's keyword for each weight format the library names.
 */

export module Chat.QuantizationKeyword;

import std;
import Dnn.WeightQuantization;

namespace Mila::ChatApp
{
    /**
     * @brief Parse a quantization keyword ("none"/"fp8"/"fp4"/"q4_0"/"cb2-3"/"fp4-cb3").
     *
     * Returns std::nullopt for an unrecognized value. Shared by the session-config
     * loader and the in-session /model command so both accept the same vocabulary.
     */
    export constexpr std::optional<Dnn::WeightQuantization> parseQuantization( std::string_view s )
    {
        using Dnn::WeightQuantization;

        if ( s.empty() || s == "none" )  return WeightQuantization::None;
        if ( s == "fp8" )                return WeightQuantization::FP8;
        if ( s == "fp4" )                return WeightQuantization::FP4;
        if ( s == "q4_0" )               return WeightQuantization::Q4_0;

        // Accepted although it can never be applied at load, so that a user who names the
        // variant a model already IS gets the load they asked for rather than a vocabulary
        // error. Naming it for a model that is not one is refused where the weights are
        // known, which is the only place the difference can be seen.
        if ( s == "cb2-3" )              return WeightQuantization::Plan;
        if ( s == "fp4-cb3" )            return WeightQuantization::Fp4Codebook3;

        return std::nullopt;
    }

    /**
     * @brief Display name for a weight format. The spelling parseQuantization accepts.
     */
    export constexpr std::string_view quantizationName( Dnn::WeightQuantization mode )
    {
        using Dnn::WeightQuantization;

        switch ( mode )
        {
            case WeightQuantization::FP8: return "fp8";
            case WeightQuantization::FP4: return "fp4";
            case WeightQuantization::Q4_0: return "q4_0";
            case WeightQuantization::Plan: return "cb2-3";
            case WeightQuantization::Fp4Codebook3: return "fp4-cb3";
            case WeightQuantization::None: return "none";
        }

        return "none";
    }
}
