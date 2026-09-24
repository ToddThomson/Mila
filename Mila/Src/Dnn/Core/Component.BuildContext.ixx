/**
 * @file Component.BuildContext.ixx
 * @brief Build-time context passed to Component::build().
 *
 * BuildContext carries the build-time concerns down the Component
 * hierarchy during build().
 */
module;
#include <cstddef>
#include <optional>
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
     * Carries five orthogonal concerns down the Component hierarchy, plus the per-component
     * declarations a composite makes for its children (installed output, fused decode):
     *
     * 1. **Input shape**             -- the full input shape the component receives.
     *                                  Used for parameter sizing, output buffer
     *                                  allocation, and build-time validation against
     *                                  component config.
     *
     * 2. **RuntimeMode**             -- allocation policy governing output buffer
     *                                  sizing and gradient buffer allocation.
     *
     *                                  Inference -- T=1 decode path output buffers.
     *                                  Training  -- full sequence output buffers,
     *                                              gradient buffers allocated.
     *
     * 3. **Parameter initialization** -- whether components should initialize parameter
     *                                   tensors after allocation. Set to false when
     *                                   building for a weights load to avoid
     *                                   computing initializers (Xavier, normal, zeros)
     *                                   that are immediately overwritten by loadParameter().
     *                                   When not specified, the default is derived from
     *                                   RuntimeMode: Training initializes (train from
     *                                   scratch), Inference skips (weights are loaded).
     *                                   An inference-mode build therefore cannot silently
     *                                   run then discard parameter initialization by
     *                                   omitting the flag.
     *
     * 4. **Device facts**            -- the allocation granularity a prediction prices
     *                                   against. getRequiredMemory() reads it from here
     *                                   and never from the device the component is bound
     *                                   to, so one graph prices any device (Deployment.md
     *                                   section 4). build() does not need it.
     *
     * 5. **Prefill chunk**           -- rows per prefill pass. A language network builds
     *                                   and prices at the chunk its caller resolved, and
     *                                   stamps it onto the contexts of its children. No
     *                                   component reads free memory to choose one.
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
         * @brief Default constructor -- sentinel value for pre-build state.
         *
         * Produces a minimal valid BuildContext with parameter initialization
         * enabled, Auto precision policy, and no quantization.
         * Never read before build() is called -- Component::ensureBuilt()
         * guards all access paths.
         */
        BuildContext()
            : input_shape_{ 1 }, runtime_mode_( RuntimeMode::Training ), initialize_parameters_( true )
        {
        }

        /**
         * @brief Construct from the input shape, runtime mode and parameter initialization.
         *
         * The granularity and the prefill chunk are added with their own with*() calls.
         *
         * @param input_shape            Complete input shape this component receives.
         *                               Must have at least one dimension.
         * @param runtime_mode           Allocation policy: Inference or Training.
         * @param initialize_parameters  When false, components allocate parameter
         *                               tensors but skip value initialization. When
         *                               omitted (nullopt), the default is derived from
         *                               runtime_mode -- Training initializes, Inference
         *                               skips -- so a load path cannot regress by
         *                               forgetting the flag.
         * @throws std::invalid_argument if input_shape is empty.
         */
        explicit BuildContext(
            shape_t input_shape,
            RuntimeMode runtime_mode,
            std::optional<bool> initialize_parameters = std::nullopt )
            : input_shape_( std::move( input_shape ) ),
              runtime_mode_( runtime_mode ),
              initialize_parameters_( initialize_parameters.value_or( runtime_mode == RuntimeMode::Training ) )
        {
            if ( input_shape_.empty() )
            {
                throw std::invalid_argument(
                    "BuildContext: input_shape must have at least one dimension" );
            }
        }

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
         * All other fields are preserved -- RuntimeMode, prefill_size, and
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
         * All other fields are preserved. A caller gives a language network the chunk it
         * resolved this way, and the network stamps it onto the contexts it builds its child
         * components with.
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
         * @brief A context for a child component with a different input shape.
         *
         * Carries the runtime mode, parameter initialization and the allocation granularity, and
         * none of the per-component declarations -- prefill size, installed output, fused decode --
         * which the composite states for each child itself. withShape() is the call that keeps those.
         */
        [[nodiscard]] BuildContext forChild( shape_t input_shape ) const
        {
            BuildContext child( std::move( input_shape ), runtime_mode_, initialize_parameters_ );
            child.allocation_granularity_ = allocation_granularity_;

            return child;
        }

        // ====================================================================
        // Device facts
        // ====================================================================

        /**
         * @brief Return a copy of this context priced at a device's allocation granularity.
         *
         * A prediction rounds each allocation over 1 MiB up to this (MemoryFootprint.md 11.8) and
         * never asks the device it is bound to, so one graph can be priced for any device. Zero
         * rounds nothing, which is right for the CPU.
         */
        [[nodiscard]] BuildContext withAllocationGranularity( std::size_t granularity ) const
        {
            BuildContext copy( *this );
            copy.allocation_granularity_ = granularity;

            return copy;
        }

        bool hasAllocationGranularity() const noexcept
        {
            return allocation_granularity_.has_value();
        }

        /**
         * @brief The allocation granularity a prediction rounds to.
         *
         * @throws std::logic_error when none was given. A missing granularity is a programming
         *         error rather than zero: zero is a real answer (no rounding), and taking it by
         *         default under-predicts every CUDA allocation over 1 MiB.
         */
        std::size_t getAllocationGranularity() const
        {
            if ( !allocation_granularity_ )
            {
                throw std::logic_error(
                    "BuildContext: a prediction needs the allocation granularity of the device it prices; "
                    "call withAllocationGranularity()" );
            }

            return *allocation_granularity_;
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
        // Output installation
        // ====================================================================

        /**
         * @brief Declare that the caller will install this component's output buffer.
         *
         * A composite that pools activations installs a shared slot into each child
         * *before* calling build(), so the child skips self-allocating its output. That
         * decision is invisible to getRequiredMemory(), which runs before any installation
         * has happened and would otherwise count a buffer the build never allocates --
         * once in the child and again in the pooling parent.
         *
         * Only prediction reads this; onBuilding() continues to use the component's own
         * installed flag, which by then is accurate.
         */
        [[nodiscard]] BuildContext withInstalledOutput( bool installed ) const
        {
            BuildContext copy( *this );
            copy.installed_output_ = installed;

            return copy;
        }

        /**
         * @brief True if the caller will install this component's output buffer.
         */
        bool hasInstalledOutput() const noexcept
        {
            return installed_output_;
        }

        /**
         * @brief Declare that single-token decode attention runs the fused kernel.
         *
         * The transformer decides this before build rather than after, so the scratch the
         * kernel requests is in both the prediction and the reservation the network makes.
         */
        [[nodiscard]] BuildContext withFusedDecode( bool fused ) const
        {
            BuildContext copy( *this );
            copy.fused_decode_ = fused;

            return copy;
        }

        /**
         * @brief True if single-token decode attention runs the fused kernel.
         */
        bool usesFusedDecode() const noexcept
        {
            return fused_decode_;
        }

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

        /**
         * @brief The prefill chunk a language network builds and prices at, which its caller resolved.
         *
         * An inference build executes the chunk it is given and never chooses one (Deployment.md
         * section 7), so an inference context without one is a programming error rather than an
         * invitation to decide. A training build does not chunk its prefill, and without a chunk
         * takes the whole context as one.
         *
         * @throws std::logic_error      when an inference context carries no chunk.
         * @throws std::invalid_argument when the context is not [B, T], or the chunk exceeds T.
         */
        dim_t getResolvedPrefillSize() const
        {
            if ( input_shape_.size() < 2 )
            {
                throw std::invalid_argument(
                    "BuildContext: a prefill chunk belongs to a [B, T] context" );
            }

            const dim_t context_length = input_shape_[ 1 ];

            if ( prefill_size_ <= 0 )
            {
                if ( isTrainingMode() )
                    return context_length;

                throw std::logic_error(
                    "BuildContext: an inference build needs the prefill chunk its caller resolved; "
                    "call withPrefillSize()" );
            }

            if ( prefill_size_ > context_length )
            {
                throw std::invalid_argument( std::format(
                    "BuildContext: prefill chunk {} exceeds the context length {}",
                    prefill_size_, context_length ) );
            }

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

    private:

        shape_t                  input_shape_;
        RuntimeMode              runtime_mode_;
        int64_t                  prefill_size_{ 0 };
        bool                     initialize_parameters_{ true };
        bool                     installed_output_{ false };
        bool                     fused_decode_{ false };
        std::optional<std::size_t> allocation_granularity_;
    };
}
