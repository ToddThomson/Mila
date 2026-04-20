/**
 * @file Model.ixx
 * @brief Abstract base class for all Mila models.
 *
 * Model defines the universal contract across all model families —
 * language models, image classifiers, regression models, and any
 * future model type.
 *
 * ## Architecture
 *
 * Model sits at the top of the Mila DNN pipeline:
 *
 *   Component          — leaf node, shape-driven buffer allocation
 *   CompositeComponent — structural aggregation, cascades BuildConfig
 *   Network            — graph topology, forward/backward
 *   Model              — RuntimeMode, lifecycle, universal API boundary
 *       LanguageModel  — generate(), sampling, EOS, vocabulary
 *           GptModel   — GPT-specific factory and config
 *           LlamaModel — LLaMA-specific factory and config
 *       ImageClassifier— classify(), top-k predictions
 *
 * ## RuntimeMode
 *
 * A Model is constructed for either Inference or Training — immutable
 * after construction. The mode governs which public API methods are
 * valid:
 *
 * | Mode      | Valid                                  |
 * |-----------|----------------------------------------|
 * | Inference | model-family inference API             |
 * | Training  | train() → onTraining() hook            |
 *
 * ## Training
 *
 * train() enforces the RuntimeMode::Training precondition then
 * delegates entirely to the pure virtual onTraining() hook. The
 * derived class owns the training loop — data loading, optimizer,
 * loss, backward pass, checkpointing, and sampling are all derived
 * class concerns.
 *
 * ## Threading
 *
 * Not thread-safe. External synchronization required if shared.
 */

module;
#include <memory>
#include <string>
#include <stdexcept>
//#include <format>

export module Dnn.Model;

import Dnn.Network;
import Dnn.Component;
import Dnn.RuntimeMode;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Compute.DeviceType;
import Compute.DeviceId;
//import Compute.DeviceTypeTraits;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;

    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class Model
    {
    public:

        using NetworkType = Network<TDeviceType, TPrecision>;

        // Non-copyable, movable
        Model( const Model& ) = delete;
        Model& operator=( const Model& ) = delete;
        Model( Model&& ) = default;
        Model& operator=( Model&& ) = default;

        virtual ~Model() = default;

        // ====================================================================
        // Runtime Mode
        // ====================================================================

        /**
         * @brief The runtime mode this model was constructed for.
         *
         * Immutable after construction. Governs which public API
         * methods are valid.
         */
        RuntimeMode getRuntimeMode() const noexcept
        {
            return runtime_mode_;
        }

        /**
         * @brief True if this model was constructed for inference.
         *
         * The model-family inference API (e.g. generate()) is valid.
         * train() will throw.
         */
        bool isInferenceMode() const noexcept
        {
            return runtime_mode_ == RuntimeMode::Inference;
        }

        /**
         * @brief True if this model was constructed for training.
         *
         * train() is valid. The model-family inference API will throw.
         */
        bool isTrainingMode() const noexcept
        {
            return runtime_mode_ == RuntimeMode::Training;
        }

        // ====================================================================
        // Training lifecycle
        // ====================================================================

        /**
         * @brief Run the training loop for this model.
         *
         * Enforces RuntimeMode::Training precondition then delegates
         * entirely to onTraining(). The derived class owns the loop —
         * data loading, optimizer, loss, checkpointing, and sampling
         * are all derived class concerns.
         *
         * @throws std::runtime_error if called on an Inference-mode model.
         */
        void train()
        {
            ensureTrainingMode( "train" );
            onTraining();
        }

        // ====================================================================
        // Eval toggle — Training mode only
        // ====================================================================

        /**
         * @brief Toggle eval sub-state for this model.
         *
         * When eval is true, the forward pass runs without gradients,
         * dropout is disabled, and batch norm uses running statistics.
         * When eval is false, the full training pass is restored.
         *
         * Cascades through Network to every Component and Operation
         * in the graph via their onEvalChanging() hooks.
         *
         * Only valid on models constructed with RuntimeMode::Training.
         * Inference-mode models are always in eval state by definition.
         *
         * @param eval true to enter eval sub-state, false to restore training.
         * @throws std::runtime_error if called on a RuntimeMode::Inference model.
         */
        void setEval( bool eval )
        {
            ensureTrainingMode( "setEval" );
            network_->setEval( eval );
        }

        /**
         * @brief True if this model is currently in eval sub-state.
         *
         * For RuntimeMode::Inference models always returns true —
         * inference models never compute gradients.
         * For RuntimeMode::Training models reflects the last setEval() call.
         */
        bool isEval() const noexcept
        {
            if ( isInferenceMode() )
            {
                return true;
            }
            return network_->isEval();
        }

        // ====================================================================
        // Accessors
        // ====================================================================

        /**
         * @brief The device this model runs on.
         */
        DeviceId getDeviceId() const noexcept
        {
            return network_->getExecutionContext()->getDeviceId();
        }

        /**
         * @brief Current memory allocation breakdown for this model.
         */
        MemoryStats getMemoryStats() const
        {
            return network_->getMemoryStats();
        }

        // ====================================================================
        // Diagnostics
        // ====================================================================

        /**
         * @brief Human-readable summary of this model's configuration.
         */
        virtual std::string toString() const = 0;

    protected:

        /**
         * @brief Construct with a fully built network and runtime mode.
         *
         * Called by derived class constructors only. The network must
         * already be built and have parameters loaded before this
         * constructor is called.
         *
         * @param network      Fully built and loaded Network.
         * @param runtime_mode Inference or Training — immutable after
         *                     construction.
         */
        explicit Model( std::unique_ptr<NetworkType> network, RuntimeMode runtime_mode )
            : network_( std::move( network ) ), runtime_mode_( runtime_mode )
        {}

        /**
         * @brief Training loop hook — derived class owns the implementation.
         *
         * Called by train() after precondition enforcement. The derived
         * class has total control over data loading, optimizer construction,
         * loss computation, backward pass, checkpointing, and sampling.
         *
         * Pure virtual — a model declaring RuntimeMode::Training must
         * provide a training loop.
         */
        virtual void onTraining() = 0;

        /**
         * @brief The owned Network instance.
         *
         * Accessible to derived classes for model-specific operations
         * not covered by the base class API.
         */
        std::unique_ptr<NetworkType> network_;

    private:

        RuntimeMode runtime_mode_;

        // ====================================================================
        // Precondition guards
        // ====================================================================

        void ensureTrainingMode( const char* method ) const
        {
            if ( runtime_mode_ != RuntimeMode::Training )
            {
                throw std::runtime_error( "Model::{}: only valid in Training mode" );
                    //std::format(
                    //    , method ) );
            }
        }
    };
}