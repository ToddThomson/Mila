/**
 * @file ComponentBuildContext.ixx
 * @brief Build-time context passed to Component::build().
 *
 * BuildContext carries the build-time concerns down the Component
 * hierarchy during build().
 */
module;
#include <cstddef>
#include <stdexcept>
#include <format>

export module Dnn.Component:BuildContext;

import Dnn.RuntimeMode;
import Dnn.TensorTypes;

namespace Mila::Dnn
{
    /**
     * @brief Build-time context for Component::build().
     *
     * Carries four orthogonal concerns down the Component hierarchy:
     *
     * 1. **Input shape**             — the full input shape the component receives.
     *                                  Used for parameter sizing, output buffer
     *                                  allocation, and build-time validation against
     *                                  component config.
     *
     * 2. **RuntimeMode**             — allocation policy governing output buffer
     *                                  sizing and gradient buffer allocation.
     *
     *                                  Inference — T=1 decode path output buffers.
     *                                  Training  — full sequence output buffers,
     *                                              gradient buffers allocated.
     *
     * 3. **Prefill size**            — number of tokens processed per prefill pass.
     *                                  Only meaningful for RuntimeMode::Inference.
     *                                  Zero indicates no prefill support.
     *                                  Components that cannot reuse the shared prefill
     *                                  scratch buffer allocate their own prefill buffer
     *                                  sized to prefillSize() in onBuilding().
     *
     * 4. **Parameter initialization** — whether components should initialize parameter
     *                                   tensors after allocation. Set to false when
     *                                   building for a pretrained weight load to avoid
     *                                   computing initializers (Xavier, normal, zeros)
     *                                   that are immediately overwritten by loadParameter().
     *                                   Defaults to true (training from scratch).
     *
     * ## Caller responsibility
     *
     * The Network or Transformer constructing BuildContext is responsible
     * for providing the correct full input shape for each child component.
     * Each component validates this shape against its own config in
     * onBuilding().
     *
     * ## Threading
     *
     * Not synchronized. Used only during the single-threaded build phase.
     */
    export class BuildContext
    {
    public:

        // ====================================================================
        // Construction
        // ====================================================================

        /**
         * @brief Default constructor — sentinel value for pre-build state.
         *
         * Produces a minimal valid BuildContext with parameter initialization
         * enabled. Never read before build() is called — Component::ensureBuilt()
         * guards all access paths.
         */
        BuildContext()
            : input_shape_{ 1 }
            , runtime_mode_( RuntimeMode::Training )
            , prefill_size_( 0 )
            , initialize_parameters_( true )
        {
        }

        /**
         * @brief Construct from full input shape, runtime mode, prefill size,
         *        and parameter initialization policy.
         *
         * @param input_shape            Complete input shape this component receives.
         *                               Must have at least one dimension.
         * @param runtime_mode           Allocation policy: Inference or Training.
         * @param prefill_size           Tokens per prefill pass. Only meaningful for
         *                               RuntimeMode::Inference. Zero disables prefill
         *                               buffer allocation.
         * @param initialize_parameters  When false, components allocate parameter
         *                               tensors but skip value initialization. Use
         *                               false when weights will be loaded from a
         *                               pretrained checkpoint immediately after build().
         *
         * @throws std::invalid_argument if input_shape is empty.
         */
        explicit BuildContext(
            shape_t input_shape,
            RuntimeMode runtime_mode,
            int64_t prefill_size = 0,
            bool initialize_parameters = true )
            : input_shape_( std::move( input_shape ) )
            , runtime_mode_( runtime_mode )
            , prefill_size_( prefill_size )
            , initialize_parameters_( initialize_parameters )
        {
            if ( input_shape_.empty() )
            {
                throw std::invalid_argument(
                    "BuildContext: input_shape must have at least one dimension" );
            }
        }

        // ====================================================================
        // Shape
        // ====================================================================

        /**
         * @brief The full input shape this component receives.
         *
         * Components use this for parameter sizing, output buffer
         * allocation, and build-time validation against their config.
         *
         * @return Const reference to the full input shape.
         */
        const shape_t& inputShape() const noexcept
        {
            return input_shape_;
        }

        // ====================================================================
        // RuntimeMode
        // ====================================================================

        /**
         * @brief The runtime mode governing output buffer allocation.
         */
        RuntimeMode getRuntimeMode() const noexcept
        {
            return runtime_mode_;
        }

        /**
         * @brief True if output buffers should be allocated at T=1.
         *
         * Components allocate decode path output buffers at T=1.
         * Components that cannot reuse the shared prefill scratch buffer
         * allocate their own prefill buffer sized to prefillSize().
         */
        bool isInferenceMode() const noexcept
        {
            return runtime_mode_ == RuntimeMode::Inference;
        }

        /**
         * @brief True if output buffers should be allocated at full
         * input shape sequence length with gradient buffers.
         */
        bool isTrainingMode() const noexcept
        {
            return runtime_mode_ == RuntimeMode::Training;
        }

        // ====================================================================
        // Prefill
        // ====================================================================

        /**
         * @brief Number of tokens processed per prefill pass.
         *
         * Only meaningful for RuntimeMode::Inference builds. Components
         * that cannot reuse the shared prefill scratch buffer — such as
         * Residual connections that must survive across sub-graphs — use
         * this value to size their own prefill buffer in onBuilding().
         *
         * Zero indicates prefill is not supported or not required.
         *
         * @return Prefill size in tokens.
         */
        int64_t prefillSize() const noexcept
        {
            return prefill_size_;
        }

        /**
         * @brief True if this build context supports prefill.
         *
         * Convenience accessor — equivalent to
         * isInferenceMode() && prefillSize() > 0.
         */
        bool hasPrefill() const noexcept
        {
            return runtime_mode_ == RuntimeMode::Inference && prefill_size_ > 0;
        }

        // ====================================================================
        // Parameter initialization
        // ====================================================================

        /**
         * @brief True if components should initialize parameter values after allocation.
         *
         * When false, allocateParameters() runs but initializeParameters() is
         * skipped. Callers loading a pretrained checkpoint should set this to
         * false so Xavier, normal, and zero initializers are not computed for
         * tensors that are immediately overwritten by loadParameter().
         *
         * @return True if parameter initialization should be performed.
         */
        bool shouldInitializeParameters() const noexcept
        {
            return initialize_parameters_;
        }

    private:

        shape_t     input_shape_;
        RuntimeMode runtime_mode_;
        int64_t     prefill_size_{ 0 };
        bool        initialize_parameters_{ true };
    };
}