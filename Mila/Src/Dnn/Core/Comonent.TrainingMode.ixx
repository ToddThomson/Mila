/**
 * @file ComponentTrainingMode.ixx
 * @brief Runtime behavioral mode for training-built Components.
 */
module;
#include <cstdint>

export module Dnn.Component:TrainingMode;

namespace Mila::Dnn
{
    /**
     * @brief Runtime behavioral state for Components built with RuntimeMode::Training.
     *
     * TrainingMode governs the runtime behavioral state of a Component
     * that was built with RuntimeMode::Training. It is orthogonal to
     * RuntimeMode — RuntimeMode is a build-time allocation policy while
     * TrainingMode is a runtime behavioral toggle.
     *
     * ## States
     *
     * | TrainingMode | Gradients | Dropout | Batch Norm          |
     * |--------------|-----------|---------|---------------------|
     * | Normal       | active    | on      | uses batch stats    |
     * | Eval         | inactive  | off     | uses running stats  |
     *
     * ## Validity
     *
     * TrainingMode is only meaningful on Components built with
     * RuntimeMode::Training. Calling setTrainingMode() on a Component
     * built with RuntimeMode::Inference throws std::runtime_error.
     *
     * ## Ownership
     *
     * Model drives transitions via Network::setTrainingMode() — the
     * toggle is never exposed directly to the user.
     */
    export enum class TrainingMode : uint8_t
    {
        Normal,    ///< Gradients active, dropout on, batch norm uses batch stats.
        Eval       ///< No gradients, dropout off, batch norm uses running stats.
    };
}