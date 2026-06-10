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
     * Carries six orthogonal concerns down the Component hierarchy:
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
         * enabled, Auto precision policy, and no quantization.
         * Never read before build() is called — Component::ensureBuilt()
         * guards all access paths.
         */
        BuildContext()
            : input_shape_{ 1 }, runtime_mode_( RuntimeMode::Training ), initialize_parameters_( true )
        {
        }

        /**
         * @brief Construct from all six concerns explicitly.
         *
         * precision_policy and quantization are extracted from ModelConfig
         * by fromPretrained() and passed here as raw values, keeping
         * BuildContext free of any model-layer dependency.
         *
         * @param input_shape            Complete input shape this component receives.
         *                               Must have at least one dimension.
         * @param runtime_mode           Allocation policy: Inference or Training.
         * @param initialize_parameters  When false, components allocate parameter
         *                               tensors but skip value initialization.
         * @throws std::invalid_argument if input_shape is empty.
         */
        explicit BuildContext(
            shape_t input_shape,
            RuntimeMode runtime_mode,
            bool initialize_parameters = true )
            : input_shape_( std::move( input_shape ) ), runtime_mode_( runtime_mode ), initialize_parameters_( initialize_parameters )
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
         */
        const shape_t& inputShape() const noexcept
        {
            return input_shape_;
        }

        /**
         * @brief Return a copy of this context with a different input shape.
         *
         * All other fields are preserved — RuntimeMode, prefill_size, and
         * initialize_parameters.
         *
         * @param new_shape  Replacement input shape. Must be non-empty.
         * @return New BuildContext with new_shape and all other fields unchanged.
         */
        [[nodiscard]] BuildContext withShape( shape_t new_shape ) const
        {
            if ( new_shape.empty() )
            {
                throw std::invalid_argument(
                    "BuildContext::withShape: input_shape must have at least one dimension" );
            }

            BuildContext copy( *this );
            copy.input_shape_ = std::move( new_shape );

            return copy;
        }

        /**
         * @brief Return a copy of this context with a different prefill size.
         *
         * All other fields are preserved. Used by the network to stamp the tuned
         * prefill chunk size onto the contexts it builds its child components with.
         *
         * @param prefill_size  Tokens per prefill pass.
         * @return New BuildContext with prefill_size and all other fields unchanged.
         */
        [[nodiscard]] BuildContext withPrefillSize( int64_t prefill_size ) const
        {
            BuildContext copy( *this );
            copy.prefill_size_ = prefill_size;

            return copy;
        }

        /**
         * @brief Return a copy of this context with a different quantization config.
         *
         * All other parameters are preserved. Provided for sub-graphs that
         * require a different quantization policy than the parent context.
         *
         * @param quantization  Replacement quantization config.
         * @return New BuildContext with quantization and all other fields unchanged.
         */
        // DEPRECATED:
        //[[nodiscard]] BuildContext withQuantization( QuantizationConfig quantization ) const
        //{
        //    return BuildContext(
        //        input_shape_,
        //        runtime_mode_,
        //        //prefill_size_,
        //        initialize_parameters_,
        //        quantization );
        //}

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
         */
        bool isInferenceMode() const noexcept
        {
            return runtime_mode_ == RuntimeMode::Inference;
        }

        /**
         * @brief True if output buffers should be allocated at full
         *        input shape sequence length with gradient buffers.
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
         * The tuned prefill chunk size, computed once at network build time and
         * threaded down to every component that sizes prefill buffers or attention
         * scratch. Zero on training-mode contexts (no chunking).
         */
        int64_t getPrefillSize() const noexcept
        {
            return prefill_size_;
        }

        // ====================================================================
        // Parameter initialization
        // ====================================================================

        /**
         * @brief True if components should initialize parameter values after allocation.
         */
        bool shouldInitializeParameters() const noexcept
        {
            return initialize_parameters_;
        }

        // ====================================================================
        // Quantization
        // ====================================================================

        /**
         * @brief The weight storage dtype and scale allocation policy.
         *
         * Consumed by Linear::build() and Linear::loadParameters() only.
         * All other components ignore this field.
         */
         // DEPRECATED:
        /*const QuantizationConfig& getQuantization() const noexcept
        {
            return quantization_;
        }*/

    private:

        shape_t                  input_shape_;
        RuntimeMode              runtime_mode_;
        int64_t                  prefill_size_{ 0 };
        bool                     initialize_parameters_{ true };
    };
}
