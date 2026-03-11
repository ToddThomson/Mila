/**
 * @file ComponentBuildConfig.ixx
 * @brief Build-time configuration passed to Component::build().
 *
 * BuildConfig carries the leading shape that drives buffer allocation
 * across the Component hierarchy. It is an opaque shape carrier —
 * it imposes no interpretation on the dimensions. Each Component
 * interprets the leading shape according to its own semantics in
 * its onBuilding() override.
 *
 * ## Design
 *
 * BuildConfig is constructed by Model and cascaded through the
 * Component hierarchy during build(). It intentionally has no
 * concept of RuntimeMode, batch size, sequence length, or any
 * other domain-specific dimension — those are Component concerns.
 *
 * ## Example
 *
 * A transformer component expecting [ B, T ] accesses dimensions
 * by known index in its own onBuilding():
 *
 * @code
 * void onBuilding( const BuildConfig& config ) override
 * {
 *     auto batch_size = config.leadingShape()[ 0 ];
 *     auto seq_len    = config.leadingShape()[ 1 ];
 *     // allocate output buffers sized to batch_size x seq_len x features
 * }
 * @endcode
 *
 * A convolution component expecting [ B, C, H, W ] does the same:
 *
 * @code
 * void onBuilding( const BuildConfig& config ) override
 * {
 *     auto batch_size = config.leadingShape()[ 0 ];
 *     auto channels   = config.leadingShape()[ 1 ];
 *     auto height     = config.leadingShape()[ 2 ];
 *     auto width      = config.leadingShape()[ 3 ];
 * }
 * @endcode
 *
 * ## Threading
 *
 * Not synchronized. BuildConfig is used only during the single-threaded
 * build phase and is not retained after build() completes.
 */

module;
#include <cstddef>
#include <stdexcept>
#include <format>

export module Dnn.Component:BuildConfig;

import Dnn.TensorTypes;

namespace Mila::Dnn
{
    /**
     * @brief Opaque shape carrier for Component::build().
     *
     * Lightweight value type constructed by Model and passed down
     * through the Component hierarchy. Components interpret the
     * leading shape according to their own dimensional semantics.
     *
     * Setters follow the Mila fluent with* convention for any future
     * optional behavioral parameters. Currently there are none —
     * BuildConfig is intentionally minimal.
     */
    export class BuildConfig
    {
    public:

        // ====================================================================
        // Construction
        // ====================================================================

        /**
         * @brief Construct from a leading shape.
         *
         * The shape must have at least one dimension. Interpretation
         * of the dimensions is left entirely to the Component.
         *
         * @param leading_shape Shape driving buffer allocation.
         * @throws std::invalid_argument if leading_shape is empty.
         */
        explicit BuildConfig( shape_t leading_shape )
            : leading_shape_( std::move( leading_shape ) )
        {
            if ( leading_shape_.empty() )
            {
                throw std::invalid_argument(
                    "BuildConfig: leading_shape must have at least one dimension" );
            }
        }

        // ====================================================================
        // Accessors
        // ====================================================================

        /**
         * @brief The leading shape passed at construction.
         *
         * Components access dimensions by their own known indices.
         * BuildConfig imposes no interpretation.
         *
         * @return Const reference to the leading shape.
         */
        const shape_t& leadingShape() const noexcept
        {
            return leading_shape_;
        }

        /**
         * @brief Number of dimensions in the leading shape.
         *
         * Components may use this to validate their expected rank
         * at the start of onBuilding().
         *
         * @return Number of dimensions.
         */
        size_t rank() const noexcept
        {
            return leading_shape_.size();
        }

        /**
         * @brief Access a single dimension by index.
         *
         * Convenience accessor for components that need a specific
         * dimension without holding a reference to the full shape.
         *
         * @param index Zero-based dimension index.
         * @return Size of the requested dimension.
         * @throws std::out_of_range if index >= rank().
         */
        int64_t dim( size_t index ) const
        {
            if ( index >= leading_shape_.size() )
            {
                throw std::out_of_range(
                    std::format(
                        "BuildConfig::dim: index {} out of range for rank {}",
                        index, leading_shape_.size() ) );
            }

            return leading_shape_[ index ];
        }

    private:

        shape_t leading_shape_;
    };
}