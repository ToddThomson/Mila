/**
 * @file Chat.Footprint.ixx
 * @brief Predicting what a model would cost in device memory, and grading that against the card.
 *
 * One implementation, because two consumers ask the same question: the pre-flight before a load
 * and the /model list listing. A listing that graded a model differently from the load it precedes
 * would be worse than one that said nothing.
 */

module;
#include <algorithm>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <format>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

// The one preprocessor test in this module, confined to the fragment so the module body stays
// free of it: what happens when a model does not fit is a property of the driver model, and
// there is no runtime query for it.
#ifdef _WIN32
inline constexpr bool kDriverOversubscribesToHostMemory = true;
#else
inline constexpr bool kDriverOversubscribesToHostMemory = false;
#endif

export module Chat.Footprint;

import Chat.Config;
import Mila;

namespace Mila::ChatApp
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    /**
     * @brief How a predicted footprint stands against the memory available to it.
     *
     * Warn-and-proceed throughout, on every platform: the consequence of not fitting differs by
     * driver model, but neither consequence is something this can assert in advance, and an
     * over-eager refusal blocks configurations that would have run.
     */
    export enum class FootprintVerdict
    {
        /// No prediction was available, so nothing is claimed.
        Unknown,

        Fits,

        /// The weights fit but the total does not. Context is the lever.
        DoesNotFit,

        /// The weights alone exceed what is available. Context does not shrink them;
        /// quantization does.
        WeightsExceedAvailable
    };

    /// True when a row's verdict is one the user needs to be told about.
    export inline bool isOverBudget( FootprintVerdict verdict )
    {
        return verdict == FootprintVerdict::DoesNotFit
            || verdict == FootprintVerdict::WeightsExceedAvailable;
    }

    /**
     * @brief What not fitting actually means here, in one sentence.
     *
     * One explanation rather than one per degree of overflow: the difference between the weights
     * overflowing and the total overflowing changes which lever helps, not what happens. What it
     * does change is the platform -- WDDM oversubscribes into shared GPU memory and keeps going
     * (measured at 3.1 tok/s against roughly 40 for a model that fits), where a Linux driver has
     * no such fallback and the allocation simply fails.
     */
    export inline std::string doesNotFitExplanation( std::string_view available )
    {
        if constexpr ( kDriverOversubscribesToHostMemory )
        {
            // The model loads; what does not fit is what goes to shared memory. Saying the
            // model loads *into* shared memory would describe something that does not happen.
            return std::format(
                "Needs more than {}. The model may use shared GPU memory, with slow performance.",
                available );
        }
        else
        {
            return std::format( "Needs more than {}. The load will likely fail.", available );
        }
    }

    /**
     * @brief The predicted total plus an allowance for what the model does not account for.
     *
     * Gate B measured the unmodelled remainder at 6-13% of what a load consumes -- allocator
     * rounding and lazily grown scratch, which scale with the model rather than being a fixed
     * cost. A proportional allowance is closer to right than a constant, and erring high here
     * only costs a warning.
     */
    export inline std::size_t practicalDeviceBytes( const MemoryStats& required )
    {
        return required.totalDeviceBytes() + ( required.totalDeviceBytes() / 8 );
    }

    /**
     * @brief Free and total device memory, or zeros when there is no device to ask.
     *
     * Zeros are the "do not claim anything" answer rather than an error: every caller here is
     * advisory, and a missing device must not stop a model being tried.
     */
    export inline DeviceMemoryInfo queryDeviceMemory( int device_index = 0 )
    {
        const auto device = DeviceRegistry::instance().getDevice( Device::Cuda( device_index ) );

        return device ? device->getMemoryInfo() : DeviceMemoryInfo{};
    }

    /**
     * @brief The card's own name, or empty when there is no CUDA device to ask.
     *
     * `Device::getDeviceName()` answers "CUDA:0", which is a coordinate rather than a name -- it
     * tells a reader nothing about the hardware their verdicts were measured against. The marketing
     * string lives on the concrete device's properties, hence the cast; empty on failure, because a
     * listing that cannot name the card is still a listing.
     */
    export inline std::string queryDeviceName( int device_index = 0 )
    {
        const auto cuda = std::dynamic_pointer_cast<CudaDevice>(
            DeviceRegistry::instance().getDevice( Device::Cuda( device_index ) ) );

        return cuda ? cuda->getProperties().getName() : std::string{};
    }

    /**
     * @brief Memory an imminent load could claim: what is free, plus what it is about to free.
     *
     * For the load path, where the question is whether this particular load succeeds on the
     * machine as it stands. Chat releases the outgoing model before loading the replacement, so
     * resident_bytes is zero by the time this is asked -- it is passed anyway rather than
     * assumed, since a caller that has not released yet would otherwise be told the wrong thing
     * silently.
     *
     * Note what this is NOT for: grading a *catalogue* of candidates. There the question is what
     * the card can run at all, which is its capacity -- free memory would charge every candidate
     * for whatever the desktop is holding, and a model's own report understates what releasing
     * it returns by the residual it does not model.
     *
     * @param resident_bytes What the loaded model has allocated, or zero when none is loaded.
     */
    export inline std::size_t availableDeviceBytes(
        const DeviceMemoryInfo& memory, std::size_t resident_bytes )
    {
        return memory.total_bytes == 0 ? 0 : memory.free_bytes + resident_bytes;
    }

    /**
     * @brief Grade a footprint against what a load could actually claim.
     */
    export inline FootprintVerdict gradeFootprint(
        const std::optional<MemoryStats>& required, std::size_t available_bytes )
    {
        if ( !required.has_value() || available_bytes == 0 )
        {
            return FootprintVerdict::Unknown;
        }

        if ( required->device_parameter_bytes > available_bytes )
        {
            return FootprintVerdict::WeightsExceedAvailable;
        }

        return practicalDeviceBytes( *required ) > available_bytes
            ? FootprintVerdict::DoesNotFit
            : FootprintVerdict::Fits;
    }

    /**
     * @brief A context length, and how it was arrived at.
     *
     * Carried structured rather than as a sentence because the two facts have two audiences: the
     * startup line reads them as prose, the `--output-format json` payload as fields.
     */
    export struct ResolvedContext
    {
        std::size_t context_length{ 0 };

        /// True when auto chose the number, whether by measurement or by falling back.
        bool automatic{ false };

        /// Device capacity auto measured against. Zero when there was no device to ask, which is
        /// also how a caller tells a measured answer from a fallback.
        std::size_t device_total_bytes{ 0 };

        /// Why auto could not measure, from the predictor. Empty when it did.
        std::string fallback_reason;

        /// How the chosen context would chunk its prefill. Zeroed on a fallback, where no
        /// prediction was available to ask.
        PrefillChunking prefill;

        /// True when auto stopped short of the largest context that fits memory because that
        /// context would have prefilled below a full chunk. The two numbers differ by enough to
        /// be worth naming: on a 12 GB card Gemma fits 95232 and prefills well at roughly 54K.
        bool bounded_by_prefill{ false };
    };

    /**
     * @brief A footprint, or why there is not one.
     *
     * The two consumers need different halves of the same answer. A pre-flight proceeds silently
     * when nothing can be predicted, because it must never be the thing that stops a model from
     * being tried. `context_length: "auto"` cannot: a derived number with no provenance is the
     * complaint ChatConfiguration.md opens with, so it has to say what it fell back to and why.
     * Carrying the reason costs the silent caller nothing.
     */
    export struct FootprintPrediction
    {
        std::optional<MemoryStats> required;

        /// Why there is no prediction, phrased to read inside a parenthetical. Empty when
        /// required holds one.
        std::string unavailable_reason;

        /// How this context length would chunk its prefill. Zeroed when there is no prediction,
        /// and for a family whose transformer does not chunk.
        PrefillChunking prefill;
    };

    /**
     * @brief Set Qwen's weight quantization, and NOT its KV cache compression.
     *
     * The convenience setters the other families use pair each weight format with FP8 KV --
     * `withFP4Quantization()` sets both -- and Qwen has no FP8 KV type yet, so that pairing makes
     * every FP4 load throw "FP8 KV cache compression is not yet supported" before it reads a byte.
     * Its baseline is a BF16 cache, which fits at 16K uncompressed, so leaving the axis alone is
     * the deployment rather than a workaround. Same shape the FP4 oracle test builds.
     *
     * Exported so the footprint pre-flight and the load configure one model one way. Splitting
     * them is how a prediction comes to describe a deployment that never happens.
     */
    export inline void applyQwenQuantization(
        QwenModelConfig& config, QuantizationMode quantization )
    {
        switch ( quantization )
        {
            case QuantizationMode::FP8:
                config.withWeightQuantization( WeightQuantization::FP8 );
                break;

            case QuantizationMode::FP4:
                config.withWeightQuantization( WeightQuantization::FP4 );
                break;

            case QuantizationMode::Codebook:
                config.withPrecisionPlan();
                break;

            case QuantizationMode::None:
                break;
        }
    }

    /**
     * @brief What a model would allocate at a context length, without allocating any of it.
     *
     * Costs nothing on the device: the graph is constructed, asked, and discarded without a
     * weight being read. Only the artifact header is touched. See
     * Specifications/MemoryFootprint.md.
     *
     * The four axes are passed rather than a resolved model, because they are exactly what the
     * answer depends on -- and because the two callers hold them in different shapes: a session
     * config on the load path, a store record on the listing path.
     *
     * @return No footprint for a family with no entry point, for a precision that has none, and
     *         on any failure to read the artifact -- each with the reason. Never throws: a
     *         pre-flight must never be the thing that stops a model from being tried.
     */
    export inline FootprintPrediction predictFootprint(
        const std::filesystem::path& weights,
        ModelType family,
        ModelPrecision precision,
        QuantizationMode quantization,
        dim_t context_length,
        int device_index = 0 )
    {
        const DeviceId device{ DeviceType::Cuda, device_index };

        // Shared by both families that have an entry point: it is one fact about what this build
        // instantiates, not one about either architecture.
        constexpr const char* bf16_only = "a footprint is predicted for BF16 deployments only";

        try
        {
            switch ( family )
            {
                case ModelType::Llama:
                {
                    if ( precision != ModelPrecision::BF16 )
                    {
                        return { std::nullopt, bf16_only };
                    }

                    LlamaModelConfig llama_config( context_length );

                    if ( quantization == QuantizationMode::FP8 )
                        llama_config.withFP8Quantization();
                    else if ( quantization == QuantizationMode::FP4 )
                        llama_config.withFP4Quantization();

                    const DeploymentFootprint footprint =
                        LlamaModel<DeviceType::Cuda, TensorDataType::BF16>::getDeploymentFootprint(
                            weights, llama_config, device );

                    return { footprint.memory, {}, footprint.prefill };
                }

                case ModelType::Gemma:
                {
                    // The guard Llama has always had. Without it an FP32 deployment was answered
                    // by the BF16 instantiation below, which reports a footprint no load would
                    // produce -- a wrong number rather than a declined question.
                    if ( precision != ModelPrecision::BF16 )
                    {
                        return { std::nullopt, bf16_only };
                    }

                    GemmaModelConfig gemma_config( context_length );

                    if ( quantization == QuantizationMode::FP8 )
                        gemma_config.withFP8Quantization();
                    else if ( quantization == QuantizationMode::FP4 )
                        gemma_config.withFP4Quantization();

                    const DeploymentFootprint footprint =
                        GemmaModel<DeviceType::Cuda, TensorDataType::BF16>::getDeploymentFootprint(
                            weights, gemma_config, device );

                    return { footprint.memory, {}, footprint.prefill };
                }

                case ModelType::Qwen:
                {
                    if ( precision != ModelPrecision::BF16 )
                    {
                        return { std::nullopt, bf16_only };
                    }

                    QwenModelConfig qwen_config( context_length );

                    applyQwenQuantization( qwen_config, quantization );

                    const DeploymentFootprint footprint =
                        QwenModel<DeviceType::Cuda, TensorDataType::BF16>::getDeploymentFootprint(
                            weights, qwen_config, device );

                    return { footprint.memory, {}, footprint.prefill };
                }

                case ModelType::Gpt:
                default:
                    return { std::nullopt, "GPT-2 has no footprint entry point in this build" };
            }
        }
        catch ( const std::exception& error )
        {
            // The reason the bare catch used to discard. Everything reachable from here names
            // itself -- an unreadable artifact, a context past the trained maximum, a
            // quantization the dispatcher has no specialization for.
            return { std::nullopt, error.what() };
        }
    }

    /// The grid auto searches. Also the granularity it reports: a context of 83968 says it was
    /// measured, where 84131 would suggest a precision the prediction does not have.
    export inline constexpr std::size_t kContextStep = 1024;

    /**
     * @brief Silences library logging for the duration of a prediction scan.
     *
     * Constructing a graph logs, and a scan constructs one per candidate: Gemma warns that it
     * cannot prefill efficiently at long context, which arrived 35 times at startup before this.
     * A prediction is not a deployment. What a probe learns is returned, never printed -- the same
     * contract predictFootprint already holds, extended to the library it calls into.
     *
     * Warnings from the LOAD are untouched, which is the point: the one at the context actually
     * chosen is a fact about this session, where the other thirty-four were about contexts nobody
     * asked for.
     *
     * Exported because it belongs to every caller that probes in bulk, not just to the scan below.
     * The /model list ladder was written without it and leaked exactly the warning this was built to
     * suppress -- a Gemma prefill complaint about a 128K context nobody had asked to run at,
     * printed above the table it was probing for.
     */
    export class ScopedLogSuppression
    {
    public:
        ScopedLogSuppression()
            : restore_( Logging::Logger::defaultLogger().getLevel() )
        {
            Logging::Logger::defaultLogger().setLevel( Logging::LogLevel::Error );
        }

        ~ScopedLogSuppression()
        {
            Logging::Logger::defaultLogger().setLevel( restore_ );
        }

        ScopedLogSuppression( const ScopedLogSuppression& ) = delete;
        ScopedLogSuppression& operator=( const ScopedLogSuppression& ) = delete;

    private:
        Logging::LogLevel restore_;
    };

    /**
     * @brief The largest context that fits comfortably on this device, or the floor and why not.
     *
     * **Searched from the ceiling downward.** Not by bisection, and the difference is correctness
     * rather than speed: bisection assumes the footprint rises with context, and Gemma's does not
     * -- measured 2026-08-15 it climbs to 9.92 GB at 10240, drops to 9.37 at 12288 as prefill
     * chunking caps the activation buffers, then resumes at a quarter of the slope. Scanning from
     * the top finds the largest qualifying context on any curve. See ChatConfiguration.md
     * section 6 for the measurements.
     *
     * **Fitting in memory is not the same as running well, so the scan does not stop at the first
     * memory fit.** Both families resolve a prefill chunk against an activation budget that shrinks
     * as the KV cache grows, so the largest context that fits can be one where the chunk has walked
     * down toward its floor -- measured 2026-08-15, gemma-4-12b-it-fp4 fits 95232 on a 12 GB card
     * and prefills there at 64 rows of a possible 1024, where 56320 holds the full chunk. The scan
     * therefore keeps the first memory fit as a floor under the answer and continues down for the
     * largest context whose chunk is also unconstrained, preferring that when one exists.
     *
     * A probe that cannot be predicted does not end the search. Above a checkpoint's trained
     * length the prediction throws, and until ModelRecord carries that length the ceiling passed
     * here is the family's, which overshoots it -- so those probes are expected to fail and the
     * scan simply walks down past them. Only a scan where nothing could be predicted at all falls
     * back, carrying the last reason.
     *
     * @param ceiling The largest context worth trying: the family maximum today, tightened by the
     *        checkpoint's own maximum once the manifest carries one.
     * @param floor Where to land when nothing can be measured.
     */
    export inline ResolvedContext resolveAutomaticContext(
        const std::filesystem::path& weights,
        ModelType family,
        ModelPrecision precision,
        QuantizationMode quantization,
        std::size_t ceiling,
        std::size_t floor,
        int device_index = 0 )
    {
        ResolvedContext resolved;
        resolved.automatic = true;
        resolved.context_length = floor;

        const DeviceMemoryInfo memory = queryDeviceMemory( device_index );

        if ( memory.total_bytes == 0 )
        {
            resolved.fallback_reason = "no CUDA device";

            return resolved;
        }

        // Comfortably needs a number. 11.07 of 11.99 GB is a fit by arithmetic and a bad
        // experience in practice -- a 92% claim on a card that also drives a display. The margin
        // is what auto chooses for you, never a policy imposed on a length you chose yourself.
        const std::size_t margin = std::max<std::size_t>(
            memory.total_bytes / 10, std::size_t{ 512 } * 1024 * 1024 );

        if ( memory.total_bytes <= margin )
        {
            resolved.fallback_reason = "device too small to reserve a margin on";

            return resolved;
        }

        const std::size_t budget = memory.total_bytes - margin;

        const ScopedLogSuppression quiet;

        std::string last_reason;

        // The largest context that fits memory, held while the scan continues for one that also
        // prefills at a full chunk. It is the answer if no such context exists -- a constrained
        // chunk is a throughput cost, and giving up context to avoid it is only worth doing when
        // there is a context left to give up.
        std::size_t largest_fitting = 0;
        PrefillChunking largest_fitting_prefill;

        for ( std::size_t candidate = ( ceiling / kContextStep ) * kContextStep;
              candidate >= kContextStep;
              candidate -= kContextStep )
        {
            const FootprintPrediction prediction = predictFootprint(
                weights, family, precision, quantization,
                static_cast<dim_t>( candidate ), device_index );

            if ( !prediction.required )
            {
                last_reason = prediction.unavailable_reason;

                continue;
            }

            // Weights do not shrink with context, so once they alone are over there is no shorter
            // context to find and the remaining probes are all going to fail. This turns the
            // slowest case -- a card that cannot hold the model at any length -- into the fastest.
            if ( prediction.required->device_parameter_bytes > budget )
            {
                break;
            }

            if ( practicalDeviceBytes( *prediction.required ) > budget )
            {
                continue;
            }

            if ( largest_fitting == 0 )
            {
                largest_fitting = candidate;
                largest_fitting_prefill = prediction.prefill;
            }

            if ( !prediction.prefill.isBudgetConstrained() )
            {
                resolved.context_length = candidate;
                resolved.device_total_bytes = memory.total_bytes;
                resolved.prefill = prediction.prefill;
                resolved.bounded_by_prefill = candidate < largest_fitting;

                return resolved;
            }
        }

        // Every fitting context prefills below a full chunk. Memory is still the better answer
        // than the floor: the chunk costs prefill throughput, where the floor costs the length of
        // the conversation.
        if ( largest_fitting != 0 )
        {
            resolved.context_length = largest_fitting;
            resolved.device_total_bytes = memory.total_bytes;
            resolved.prefill = largest_fitting_prefill;

            return resolved;
        }

        // Nothing fit, or nothing could be predicted. The two are worth telling apart: the first
        // is a card too small for this model at any length, and no reason of ours explains it.
        resolved.fallback_reason = last_reason.empty()
            ? std::string( "no context fits this device" )
            : last_reason;

        return resolved;
    }
}
