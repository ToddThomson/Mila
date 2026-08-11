/**
 * @file Chat.Footprint.ixx
 * @brief Predicting what a model would cost in device memory, and grading that against the card.
 *
 * One implementation, because two consumers ask the same question: the pre-flight before a load
 * and the /models listing. A listing that graded a model differently from the load it precedes
 * would be worse than one that said nothing.
 */

module;
#include <cstddef>
#include <exception>
#include <filesystem>
#include <format>
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
    export inline DeviceMemoryInfo queryDeviceMemory()
    {
        const auto device = DeviceRegistry::instance().getDevice( Device::Cuda( 0 ) );

        return device ? device->getMemoryInfo() : DeviceMemoryInfo{};
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
     * @return nullopt for a family with no footprint entry point, for a precision that has none,
     *         and on any failure to read the artifact. A pre-flight must never be the thing that
     *         stops a model from being tried, so a failure here is silence rather than a throw.
     */
    export inline std::optional<MemoryStats> predictFootprint(
        const std::filesystem::path& weights,
        ModelType family,
        ModelPrecision precision,
        QuantizationMode quantization,
        dim_t context_length )
    {
        const DeviceId device{ DeviceType::Cuda, 0 };

        try
        {
            switch ( family )
            {
                case ModelType::Llama:
                {
                    if ( precision != ModelPrecision::BF16 )
                    {
                        return std::nullopt;
                    }

                    LlamaModelConfig llama_config( context_length );

                    if ( quantization == QuantizationMode::FP8 )
                        llama_config.withFP8Quantization();
                    else if ( quantization == QuantizationMode::FP4 )
                        llama_config.withFP4Quantization();

                    return LlamaModel<DeviceType::Cuda, TensorDataType::BF16>::getRequiredMemory(
                        weights, llama_config, device );
                }

                case ModelType::Gemma:
                {
                    GemmaModelConfig gemma_config( context_length );

                    if ( quantization == QuantizationMode::FP8 )
                        gemma_config.withFP8Quantization();
                    else if ( quantization == QuantizationMode::FP4 )
                        gemma_config.withFP4Quantization();

                    return GemmaModel<DeviceType::Cuda, TensorDataType::BF16>::getRequiredMemory(
                        weights, gemma_config, device );
                }

                case ModelType::Gpt:
                default:
                    return std::nullopt;
            }
        }
        catch ( const std::exception& )
        {
            return std::nullopt;
        }
    }
}
