/**
 * @file Chat.QuantizationMode.ixx
 * @brief The weight format a Chat session loads, as the library names it, and Chat's keyword for each.
 */

module;
#include <optional>
#include <string_view>

export module Chat.QuantizationMode;

import Mila;

namespace Mila::ChatApp
{
    /// The library's weight format, so the one an installed model reports is the one Chat loads.
    export using QuantizationMode = Mila::Dnn::WeightQuantization;

    /**
     * @brief Parse a quantization keyword ("none"/"fp8"/"fp4"/"q4_0"/"cb2-3").
     *
     * Returns std::nullopt for an unrecognized value. Shared by the session-config
     * loader and the in-session /model command so both accept the same vocabulary.
     */
    export constexpr std::optional<QuantizationMode> parseQuantization( std::string_view s )
    {
        if ( s.empty() || s == "none" )  return QuantizationMode::None;
        if ( s == "fp8" )                return QuantizationMode::FP8;
        if ( s == "fp4" )                return QuantizationMode::FP4;
        if ( s == "q4_0" )               return QuantizationMode::Q4_0;

        // Accepted although it can never be applied at load, so that a user who names the
        // variant a model already IS gets the load they asked for rather than a vocabulary
        // error. Naming it for a model that is not one is refused where the weights are
        // known, which is the only place the difference can be seen.
        if ( s == "cb2-3" )              return QuantizationMode::Plan;

        return std::nullopt;
    }

    /**
     * @brief Display name for a QuantizationMode. The spelling parseQuantization accepts.
     */
    export constexpr std::string_view quantizationName( QuantizationMode mode )
    {
        switch ( mode )
        {
            case QuantizationMode::FP8: return "fp8";
            case QuantizationMode::FP4: return "fp4";
            case QuantizationMode::Q4_0: return "q4_0";
            case QuantizationMode::Plan: return "cb2-3";
            case QuantizationMode::None: return "none";
        }

        return "none";
    }
}
