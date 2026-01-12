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
     * @brief Build-time configuration used by Component::build and onBuilding hooks.
     *
     * Constructor requires the full `input_shape` (input_shape[0] = global batch).
     * Optional settings are modified with fluent setters. Call `validate()` before
     * using the config (Component::build should perform this).
     */
    export struct BuildConfig
    {
        // Construct with required full input shape. Defaults make the config valid
        // when micro-batching is not used (micro_batch_size == 1).
        explicit BuildConfig( const shape_t& input_shape ) noexcept
            : input_shape_( input_shape )
        {
        }

        BuildConfig& setMicroBatchSize( std::size_t n ) noexcept
        {
            micro_batch_size_ = n;
            return *this;
        }

        BuildConfig& setGradientAccumulationSteps( std::size_t n ) noexcept
        {
            gradient_accumulation_steps_ = n;
            return *this;
        }

        const shape_t& inputShape() const noexcept { return input_shape_; }

        std::size_t microBatchSize() const noexcept { return micro_batch_size_; }

        std::size_t gradientAccumulationSteps() const noexcept { return gradient_accumulation_steps_; }

        bool isMicroBatchingEnabled() const noexcept { return micro_batch_size_ > 1; }

        std::size_t fullBatchSize() const noexcept
        {
            return input_shape_.empty() ? 0u : input_shape_[ 0 ];
        }

        std::size_t effectiveBatch() const noexcept
        {
            return micro_batch_size_ * gradient_accumulation_steps_;
        }

        shape_t microBatchShape() const
        {
            if ( input_shape_.empty() ) return {};
            shape_t s = input_shape_;
            s[ 0 ] = static_cast<std::size_t>( micro_batch_size_ );
            return s;
        }

        /**
         * @brief Validate configuration invariants.
         *
         * Throws std::invalid_argument when configuration is invalid:
         * - input_shape must be provided and input_shape[0] (full batch) must be > 0
         * - micro_batch_size and gradient_accumulation_steps must be >= 1
         * - full_batch must be divisible by the effective batch (micro_batch_size * gradient_accumulation_steps)
         *
         * Note: Component::build(const BuildConfig&) should call this before use.
         */
        void validate() const
        {
            if ( micro_batch_size_ == 0 )
            {
                throw std::invalid_argument( "BuildConfig::validate: micro_batch_size must be >= 1" );
            }

            if ( gradient_accumulation_steps_ == 0 )
            {
                throw std::invalid_argument( "BuildConfig::validate: gradient_accumulation_steps must be >= 1" );
            }

            if ( input_shape_.empty() )
            {
                throw std::invalid_argument( "BuildConfig::validate: input_shape must be provided and include the full batch size (input_shape[0])" );
            }

            const std::size_t full_batch = input_shape_[ 0 ];

            if ( full_batch == 0 )
            {
                throw std::invalid_argument( "BuildConfig::validate: full batch size (input_shape[0]) must be > 0" );
            }

            // Defensive overflow check before computing product.
            if ( micro_batch_size_ > 0 &&
                 gradient_accumulation_steps_ > (std::numeric_limits<std::size_t>::max() / micro_batch_size_) )
            {
                throw std::invalid_argument( "BuildConfig::validate: micro_batch_size * gradient_accumulation_steps would overflow" );
            }

            const std::size_t divisor = micro_batch_size_ * gradient_accumulation_steps_;

            if ( divisor == 0 )
            {
                throw std::invalid_argument( "BuildConfig::validate: computed divisor is zero" );
            }

            if ( full_batch % divisor != 0 )
            {
                throw std::invalid_argument(
                    std::format( "BuildConfig::validate: full batch size {} is not divisible by effective batch ({})",
                                 full_batch, divisor )
                );
            }
        }

    private:
        shape_t input_shape_;
        std::size_t micro_batch_size_{ 1 };
        std::size_t gradient_accumulation_steps_{ 1 };
    };
}