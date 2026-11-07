/**
 * @file ILanguageModel.ixx
 * @brief A loaded language model with its device and precision erased: what a program calls on any model.
 *
 * See Specifications/ModelHandle.md 10.7.
 */

export module Dnn.ILanguageModel;

import std;
import Dnn.Component;
import Dnn.GenerateParams;
import Dnn.GenerateStatus;
import Dnn.RuntimeMode;
import Dnn.TensorDataType;
import Compute.DeviceId;
import Compute.Observation;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;

    /**
     * @brief The language model behind a handle, whatever its family, device and precision.
     *
     * LanguageModel<TDeviceType, TPrecision> implements it; a program that needs the typed model and its components
     * casts to that type. Each call here is one virtual call: generate runs a whole reply, so nothing here sits
     * between a token and the network.
     */
    export class ILanguageModel
    {
    public:

        virtual ~ILanguageModel() = default;

        /**
         * @brief Generate tokens from a prompt, streaming each through on_token (LanguageModel::generate).
         */
        [[nodiscard]] virtual GenerateStatus generate(
            std::span<const std::int32_t> prompt_tokens,
            const std::function<void( std::int32_t )>& on_token,
            const GenerateParams& params = {},
            std::stop_token stop = {} ) = 0;

        /// Watch activations through matching components on matching passes; returns how many matched.
        virtual std::size_t observe(
            std::string_view pattern, ComputePassMask passes, ActivationObserver sink ) = 0;

        /// Detach every observer.
        virtual void stopObserving() = 0;

        /// Every observable component path.
        [[nodiscard]] virtual std::vector<std::string> componentPaths() const = 0;

        /// Write the live weights as a safetensors weights file that loads by the same path.
        virtual void save( const std::filesystem::path& path ) const = 0;

        /// The scheme name save records for these weights, e.g. "per_group_fp4_64".
        [[nodiscard]] virtual std::string weightQuantizationScheme() const = 0;

        /// Seed the sampler for reproducible generation: once, before a run.
        virtual void seedSampler( std::uint64_t seed ) = 0;

        /// Replay each generated token's forward pass from one recording, or run it call by call.
        virtual void setDecodeReplay( bool enabled ) = 0;

        [[nodiscard]] virtual bool isDecodeReplayed() const noexcept = 0;

        /// The device the model runs on; with a split model, the one holding its input.
        [[nodiscard]] virtual DeviceId getDeviceId() const noexcept = 0;

        /// The precision its activations are computed in.
        [[nodiscard]] virtual TensorDataType getPrecision() const noexcept = 0;

        [[nodiscard]] virtual RuntimeMode getRuntimeMode() const noexcept = 0;

        /// What the model holds now, category by category.
        [[nodiscard]] virtual MemoryStats getMemoryStats() const = 0;

        /// A summary of the model's configuration, for a developer's log.
        [[nodiscard]] virtual std::string toString() const = 0;

    protected:

        ILanguageModel() = default;
        ILanguageModel( const ILanguageModel& ) = default;
        ILanguageModel& operator=( const ILanguageModel& ) = default;
        ILanguageModel( ILanguageModel&& ) = default;
        ILanguageModel& operator=( ILanguageModel&& ) = default;
    };
}
