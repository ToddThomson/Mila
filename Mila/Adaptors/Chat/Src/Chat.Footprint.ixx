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

export import Chat.FootprintVerdict;
export import Chat.FootprintPrediction;
export import Chat.ScopedLogSuppression;

import Chat.Config;
import Mila;

namespace Mila::ChatApp
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Deployment;

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

            // Qwen refuses it; passed through so the refusal is the library's.
            case QuantizationMode::Q4_0:
                config.withWeightQuantization( WeightQuantization::Q4_0 );
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
        else if ( quantization == QuantizationMode::Q4_0 )
        {
            request.withQ4_0Quantization();
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
                    else if ( quantization == QuantizationMode::Q4_0 )
                        llama_config.withQ4_0Quantization();

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
                    else if ( quantization == QuantizationMode::Q4_0 )
                        gemma_config.withQ4_0Quantization();

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
}
