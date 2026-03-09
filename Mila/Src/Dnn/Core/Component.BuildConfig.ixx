/**
 * @file ComponentBuildConfig.ixx
 * @brief Build-time configuration for component construction.
 *
 * Provides a small, fluent config used by components during build.
 */

module;
#include <cstddef>
#include <stdexcept>
#include <format>
#include <limits>

export module Dnn.Component:BuildConfig;

import Dnn.TensorTypes;

namespace Mila::Dnn
{
    /**
     * @brief Build-time configuration passed to Component::build() and onBuilding() hooks.
     *
     * Carries the leading shape dimensions { B, T, ... } that the network and all
     * its components use as allocation bounds for their buffers. The leading shape
     * is not the full input tensor shape — each component derives its own full tensor
     * shapes by appending trailing dimensions from its own component config.
     *
     * Example — for a transformer network:
     * @code
     *   shape_t leading_shape = { config.batch_size, config.seq_length };
     *   model->build( leading_shape );
     * @endcode
     *
     * The parent network cascades the same BuildConfig unchanged to all child
     * components. Each child appends its own trailing dims from its own config.
     *
     * Optional micro-batching settings are modified with fluent setters and are
     * orthogonal to the leading shape concern.
     *
     * Call validate() before use — Component::build() does this automatically.
     */
    export struct BuildConfig
    {
        /**
         * @brief Construct with the leading shape dimensions.
         *
         * @param leading_shape The leading dimensions { B, T, ... } used as
         *                      allocation bounds. Must be non-empty with
         *                      leading_shape[0] > 0.
         */
        explicit BuildConfig( shape_t leading_shape ) noexcept
            : leading_shape_( std::move( leading_shape ) )
        {}

        /**
         * @brief Set the micro-batch size.
         *
         * @param n Micro-batch size. Must be >= 1.
         * @return *this for fluent chaining.
         */
        BuildConfig& setMicroBatchSize( std::size_t n ) noexcept
        {
            micro_batch_size_ = n;
            return *this;
        }

        /**
         * @brief Set the number of gradient accumulation steps.
         *
         * @param n Gradient accumulation steps. Must be >= 1.
         * @return *this for fluent chaining.
         */
        BuildConfig& setGradientAccumulationSteps( std::size_t n ) noexcept
        {
            gradient_accumulation_steps_ = n;
            return *this;
        }

        /**
         * @brief The leading shape dimensions { B, T, ... } passed at build time.
         *
         * Used by components as the allocation bound for their buffers.
         * Components append their own trailing dims from their component config
         * to form their full tensor shapes.
         *
         * @return The leading shape.
         */
        [[nodiscard]] const shape_t& leadingShape() const noexcept
        {
            return leading_shape_;
        }

        /**
         * @brief The leading dimension at the given index.
         *
         * @param index Index into the leading shape.
         * @return The dimension value.
         */
        [[nodiscard]] std::size_t leadingDim( std::size_t index ) const
        {
            return leading_shape_.at( index );
        }

        /**
         * @brief The batch size — leading_shape[0].
         *
         * @return Batch size, or 0 if leading_shape is empty.
         */
        [[nodiscard]] std::size_t batchSize() const noexcept
        {
            return leading_shape_.empty() ? 0u : leading_shape_[ 0 ];
        }

        [[nodiscard]] std::size_t microBatchSize() const noexcept
        {
            return micro_batch_size_;
        }

        [[nodiscard]] std::size_t gradientAccumulationSteps() const noexcept
        {
            return gradient_accumulation_steps_;
        }

        [[nodiscard]] bool isMicroBatchingEnabled() const noexcept
        {
            return micro_batch_size_ > 1;
        }

        [[nodiscard]] std::size_t effectiveBatchSize() const noexcept
        {
            return micro_batch_size_ * gradient_accumulation_steps_;
        }

        /**
         * @brief Validate configuration invariants.
         *
         * Throws std::invalid_argument when:
         * - leading_shape is empty
         * - leading_shape[0] (batch size) is 0
         * - micro_batch_size or gradient_accumulation_steps is 0
         * - batch size is not divisible by effective batch size
         *   (micro_batch_size * gradient_accumulation_steps)
         *
         * Component::build() calls this automatically before invoking onBuilding().
         */
        void validate() const
        {
            if ( leading_shape_.empty() )
            {
                throw std::invalid_argument(
                    "BuildConfig::validate: leading_shape must be non-empty" );
            }

            const std::size_t batch_size = leading_shape_[ 0 ];

            if ( batch_size == 0 )
            {
                throw std::invalid_argument(
                    "BuildConfig::validate: leading_shape[0] (batch size) must be > 0" );
            }

            if ( micro_batch_size_ == 0 )
            {
                throw std::invalid_argument(
                    "BuildConfig::validate: micro_batch_size must be >= 1" );
            }

            if ( gradient_accumulation_steps_ == 0 )
            {
                throw std::invalid_argument(
                    "BuildConfig::validate: gradient_accumulation_steps must be >= 1" );
            }

            if ( gradient_accumulation_steps_ > (std::numeric_limits<std::size_t>::max() / micro_batch_size_) )
            {
                throw std::invalid_argument(
                    "BuildConfig::validate: micro_batch_size * gradient_accumulation_steps would overflow" );
            }

            const std::size_t effective_batch = micro_batch_size_ * gradient_accumulation_steps_;

            if ( batch_size % effective_batch != 0 )
            {
                throw std::invalid_argument(
                    std::format( "BuildConfig::validate: batch size {} is not divisible by effective batch size ({})",
                        batch_size, effective_batch ) );
            }
        }

    private:
        shape_t leading_shape_;
        std::size_t micro_batch_size_{ 1 };
        std::size_t gradient_accumulation_steps_{ 1 };
    };
}
