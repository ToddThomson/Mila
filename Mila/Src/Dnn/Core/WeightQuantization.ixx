/**
 * @file WeightQuantization.ixx
 * @brief The weight storage format a deployment asks for, and the rule a load applies to stored weights.
 */

module;
#include <format>
#include <stdexcept>
#include <string>
#include <string_view>

export module Dnn.WeightQuantization;

namespace Mila::Dnn
{
    /**
     * @brief Weight storage and matmul strategy for Linear components.
     *
     * Maps to the TWeightQuantization template parameter on Linear and CudaLinearOp
     * via the load() runtime->compile-time bridge. The mapping is:
     *
     *   None  -> NoWeightQuant        (BF16 weights, standard cuBLASLt plan)
     *   FP8   -> PerChannelFp8<>      (FP8_E4M3 weights, per-channel float32 scales)
     *   FP4   -> PerGroupFp4<128>     (FP4_E2M1 weights, per-group float32 scales), or <64> for a
     *                                  model whose projection widths are not multiples of 128
     *   Q4_0  -> PerGroupInt4<32>     (INT4 codes, per-group IEEE half scales)
     *
     * This enum is Mila API vocabulary. Callers set it via fluent methods on
     * the concrete model config -- they do not interact with the policy structs
     * directly.
     */
    export enum class WeightQuantization
    {
        None,   ///< BF16 weights -- default; no quantization overhead.
        FP8,    ///< FP8_E4M3 per-channel weight quantization.
        FP4,    ///< Per-group FP4 weight quantization.
        Q4_0,   ///< Q4_0: 4-bit integer codes, one FP16 scale per 32 weights.

        /**
         * The family's own designed per-role allocation, rather than one uniform format.
         *
         * The values above name a STORAGE FORMAT that applies to every Linear alike.
         * This one does not name a format at all: it says "build this model the way its
         * designers allocated its bits", and which formats that means is the family's to
         * define -- Qwen 3.8 spends 2.5 bits on the feed-forward gate/up pair and 4.125 on
         * full attention (Specifications/Qwen3.8.md section 5).
         *
         * A family with no plan must REFUSE this value rather than fall back to a uniform
         * policy, which is why dispatchWeightQuantization handles it explicitly.
         */
        Plan,
    };

    /**
     * @brief The scheme name recorded in a model's weights and in its manifest.
     *
     * It lives beside the enum because it is written by the model that saves the weights and
     * read by the tool that packages them, and those two must agree exactly: the load side
     * refuses weights whose scheme disagrees with the build's compile-time policy, since
     * the bytes are packed differently per scheme and reinterpreting them produces a model
     * that runs and is wrong.
     *
     * @param quantization The scheme to name.
     * @param fp4_group_size The FP4 group the model's geometry requires; ignored for other schemes.
     */
    export inline std::string weightQuantizationName( WeightQuantization quantization, int fp4_group_size = 128 )
    {
        switch ( quantization )
        {
            case WeightQuantization::FP4: return std::format( "per_group_fp4_{}", fp4_group_size );
            case WeightQuantization::FP8: return "per_channel_fp8_e4m3";
            case WeightQuantization::Q4_0: return "q4_0";

            // A plan's weights scheme is the FAMILY's, not this enum's -- Qwen 3.8 writes
            // "codebook" because its sub-4-bit rows are what a load cannot reconstruct. One
            // family has a plan today, so the mapping is stated here; a second one with a
            // different scheme is what forces it to move behind a family accessor.
            case WeightQuantization::Plan: return "codebook";

            case WeightQuantization::None:
            default:
                return "none";
        }
    }

    /**
     * @brief True when a load can derive this format from reference weights.
     *
     * The distinction the stored-weights check turns on. FP4, FP8 and Q4_0 are computed from the
     * weights at load time -- absmax scales and a format-defined level table -- so BF16 weights
     * are a legitimate source for them, and every family already relies on that: Qwen's own
     * packed weights carry codebook tensors only and quantize its attention and head
     * projections on load (Qwen3.8.md section 8).
     *
     * A plan's codebooks are the opposite case. They are FITTED offline against calibration
     * data, so nothing in a BF16 tensor recovers them and the weights must carry them.
     */
    export inline bool isDerivableFromReferenceWeights( WeightQuantization quantization )
    {
        return quantization == WeightQuantization::FP4
            || quantization == WeightQuantization::FP8
            || quantization == WeightQuantization::Q4_0;
    }

    /**
     * @brief Refuse a load whose stored weights were packed for a different policy.
     *
     * Nothing downstream can tell the two apart: packed codes reinterpreted as BF16, or a
     * BF16 blob decoded through a codebook, produce a model that loads and runs and is wrong.
     * The storage dtype cannot stand in for this check -- FP4 at group 128 and at group 64 are
     * both U8. An empty stored name means reference weights; the reader normalizes the
     * writer's "none" to empty, so the two spellings compare as one.
     *
     * The one asymmetry: reference weights ARE a valid source for a derivable format, because
     * the load computes those scales from the weights. See isDerivableFromReferenceWeights for
     * why a codebook is the opposite case.
     *
     * @param caller Qualified name of the calling factory, for the message.
     * @param weights_path Path to the weights being loaded.
     * @param stored_quantization Scheme the weights declare; empty for reference weights.
     * @param requested_quantization Policy this build was compiled for.
     * @param fp4_group_size FP4 group this build was compiled for; FP4 at another group is refused.
     *
     * @throws std::runtime_error when the stored scheme cannot serve the requested one.
     */
    export inline void requireStoredQuantizationMatches(
        std::string_view caller,
        std::string_view weights_path,
        std::string_view stored_quantization,
        WeightQuantization requested_quantization,
        int fp4_group_size = 128 )
    {
        const std::string requested = weightQuantizationName( requested_quantization, fp4_group_size );

        const bool weights_are_quantized = !stored_quantization.empty();
        const bool build_is_quantized = requested_quantization != WeightQuantization::None;

        const bool quantizes_on_load = !weights_are_quantized
            && isDerivableFromReferenceWeights( requested_quantization );

        if ( !quantizes_on_load
            && ( weights_are_quantized != build_is_quantized
                || ( weights_are_quantized && stored_quantization != requested ) ) )
        {
            const std::string_view stored = weights_are_quantized ? stored_quantization : std::string_view{ "none" };
            const std::string codebook = weightQuantizationName( WeightQuantization::Plan );

            throw std::runtime_error( std::format(
                "{}: weights '{}' are stored as '{}' but this load requested '{}'. {}",
                caller, weights_path, stored, requested,
                stored == codebook || requested == codebook
                    ? "A codebook cannot be loaded at reference precision and reference weights cannot be "
                      "decoded through one -- the codes are fitted offline against calibration data and are "
                      "not recoverable from weights"
                    : "Load them as the format they are stored in" ) );
        }
    }
}
