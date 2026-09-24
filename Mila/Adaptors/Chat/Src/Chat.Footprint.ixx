/**
 * @file Chat.Footprint.ixx
 * @brief The deployment request Chat sends for a load, what Chat says when the library refuses one, and the
 *        /model list pricing against the card.
 *
 * What a load gets -- its context length, its prefill chunk, or a refusal -- is decided by the library's
 * planner (Specifications/Deployment.md); Chat only asks and reports.
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

export module Chat.Footprint;

import Chat.Config;
import Mila;

namespace Mila::ChatApp
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Deployment;

    /**
     * @brief How a predicted footprint stands against a card's capacity, for the /model list column.
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
     * @brief Grade a footprint against a card's capacity.
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

        // The prediction stands on its own: it reserves and reports scratch, and counts the
        // driver's rounding of every allocation (MemoryFootprint.md 11.8). What it leaves out is
        // under 30 MiB, and on a card that drives a display the Windows budget cut, neither of
        // which an allowance here could size honestly.
        return required->totalDeviceBytes() > available_bytes
            ? FootprintVerdict::DoesNotFit
            : FootprintVerdict::Fits;
    }

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
     * Shared by the /model list pricing and the load's deployment request, so both configure one
     * model one way. Splitting them is how a prediction comes to describe a deployment that never
     * happens.
     */
    export template<typename TConfig>
    void applyQwenQuantization( LanguageModelConfig<TConfig>& config, QuantizationMode quantization )
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
     * @brief The deployment request a load of this family sends, before its context length is set.
     *
     * Configured exactly as the /model list pricing configures the same model, so a row and the load it
     * describes cannot be two different deployments.
     */
    export inline DeploymentRequest deploymentRequestFor(
        ModelType family, QuantizationMode quantization, int device_index )
    {
        DeploymentRequest request;
        request.withDevice( DeviceId{ DeviceType::Cuda, device_index } );

        if ( family == ModelType::Qwen )
        {
            applyQwenQuantization( request, quantization );
        }
        else if ( quantization == QuantizationMode::FP8 )
        {
            request.withFP8Quantization();
        }
        else if ( quantization == QuantizationMode::FP4 )
        {
            request.withFP4Quantization();
        }

        return request;
    }

    /**
     * @brief What a model would allocate at a context length, without allocating any of it.
     *
     * Costs nothing on the device: the graph is constructed, asked, and discarded without a
     * weight being read. Only the weights header is touched. See
     * Specifications/MemoryFootprint.md.
     *
     * The four axes are passed rather than a resolved model, because they are exactly what the
     * answer depends on -- and because the two callers hold them in different shapes: a session
     * config on the load path, a store record on the listing path.
     *
     * @return No footprint for a family with no entry point, for a precision that has none, and
     *         on any failure to read the weights -- each with the reason. Never throws: a
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

                default:
                    return { std::nullopt, "this model's family has no footprint entry point" };
            }
        }
        catch ( const std::exception& error )
        {
            // The reason the bare catch used to discard. Everything reachable from here names
            // itself -- unreadable weights, a context past the trained maximum, a
            // quantization the dispatcher has no specialization for.
            return { std::nullopt, error.what() };
        }
    }

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
     * Exported because it belongs to every caller that probes in bulk. The /model list ladder was
     * first written without it and leaked exactly the warning this was built to
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
}
