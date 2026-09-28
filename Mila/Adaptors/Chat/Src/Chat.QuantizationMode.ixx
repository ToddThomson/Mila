/**
 * @file Chat.QuantizationMode.ixx
 * @brief The weight format a Chat session loads, and its keyword.
 */

module;
#include <optional>
#include <string_view>

export module Chat.QuantizationMode;

namespace Mila::ChatApp
{
    export enum class QuantizationMode
    {
        None,  ///< BF16 weights, no KV cache compression -- default.
        FP8,   ///< FP8 weights + FP8 KV cache (PerChannelFp8 + PerChannelKvFp8).
        FP4,   ///< FP4 E2M1 weights + FP8 KV cache (PerGroupFp4 + PerChannelKvFp8).
        Q4_0,  ///< Q4_0 weights + FP8 KV cache (PerGroupInt4<32> + PerChannelKvFp8).

        /**
         * The family's own per-role bit allocation, from pre-quantized weights
         * (WeightQuantization::Plan). Qwen 3.8 spends 2 and 3 bits per weight across the body
         * against 4.125 on full attention, which averages 2.82.
         *
         * Unlike the formats above this can never be a load-time choice: a codebook is fitted
         * offline against calibration data, so the weights either carry the codes or cannot
         * be built this way at all.
         */
        Codebook,
    };

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
        if ( s == "cb2-3" )              return QuantizationMode::Codebook;

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
            case QuantizationMode::Codebook: return "cb2-3";
            case QuantizationMode::None: return "none";
        }

        return "none";
    }
}
