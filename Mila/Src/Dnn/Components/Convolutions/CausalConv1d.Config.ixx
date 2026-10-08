/**
 * @file CausalConv1d.Config.ixx
 * @brief Configuration for the depthwise causal 1-D convolution component.
 *
 * Channels and kernel width are structurally required -- there is no sensible default for
 * either -- so both are constructor parameters. Bias is optional and defaults to false,
 * which is what a convolution inside a gated recurrence normally carries. Dilation is optional
 * and defaults to 1.
 */

module;
#include <stdexcept>
#include <string>
#include <sstream>
#include <utility>

export module Dnn.Components.CausalConv1dConfig;

import Dnn.TensorTypes;
import Dnn.ComponentConfig;
import Serialization.Metadata;

namespace Mila::Dnn
{
    using Serialization::SerializationMetadata;

    export class CausalConv1dConfig : public ComponentConfig
    {
    public:
        /**
         * @brief Construct a depthwise causal 1-D convolution configuration.
         *
         * @param channels     Channel count. Depthwise, so this is both in and out channels
         *                     and the group count -- there is no cross-channel mixing.
         * @param kernel_width Filter taps. The convolution looks back (kernel_width - 1) *
         *                     dilation positions, which is also the retained state depth.
         */
        CausalConv1dConfig( dim_t channels, dim_t kernel_width )
            : channels_( channels ), kernel_width_( kernel_width )
        {
            if ( channels <= 0 )
            {
                throw std::invalid_argument( "CausalConv1dConfig: channels must be > 0" );
            }

            if ( kernel_width <= 0 )
            {
                throw std::invalid_argument( "CausalConv1dConfig: kernel_width must be > 0" );
            }
        }

        /**
         * @brief Enable or disable a learnable per-channel bias. Default: false.
         */
        template <typename Self>
        decltype(auto) withBias( this Self&& self, bool has_bias )
        {
            self.has_bias_ = has_bias;
            return std::forward<Self>( self );
        }

        /**
         * @brief Space the taps @p dilation positions apart. Default: 1, adjacent taps.
         *
         * Tap i reads position t - (kernel_width - 1 - i) * dilation. Qwen 4's per-layer
         * embedding convolves with kernel 4 at dilation 3.
         */
        template <typename Self>
        decltype(auto) withDilation( this Self&& self, dim_t dilation )
        {
            self.dilation_ = dilation;
            return std::forward<Self>( self );
        }

        dim_t getChannels() const noexcept { return channels_; }
        dim_t getKernelWidth() const noexcept { return kernel_width_; }
        dim_t getDilation() const noexcept { return dilation_; }
        bool hasBias() const noexcept { return has_bias_; }

        /// Retained input rows = (kernel_width - 1) * dilation. The convolution's whole memory.
        dim_t getStateRows() const noexcept { return ( kernel_width_ - 1 ) * dilation_; }

        /// The state shift stages every retained row in registers, so the bound is a kernel property.
        static constexpr dim_t kMaximumStateRows = 16;

        void validate() const override
        {
            if ( channels_ <= 0 )
            {
                throw std::invalid_argument( "CausalConv1dConfig: channels must be > 0" );
            }

            if ( kernel_width_ <= 0 )
            {
                throw std::invalid_argument( "CausalConv1dConfig: kernel_width must be > 0" );
            }

            if ( dilation_ <= 0 )
            {
                throw std::invalid_argument( "CausalConv1dConfig: dilation must be > 0" );
            }

            // Qwen 3.8 retains 3 rows (kernel 4); Qwen 4's per-layer embedding retains 9 (kernel 4,
            // dilation 3).
            if ( getStateRows() > kMaximumStateRows )
            {
                throw std::invalid_argument(
                    "CausalConv1dConfig: (kernel_width - 1) * dilation must be <= "
                    + std::to_string( kMaximumStateRows )
                    + " (the kernel stages every retained row in registers)" );
            }
        }

        SerializationMetadata toMetadata() const override
        {
            SerializationMetadata meta;

            meta.set( "channels", static_cast<int64_t>( channels_ ) )
                .set( "kernel_width", static_cast<int64_t>( kernel_width_ ) )
                .set( "dilation", static_cast<int64_t>( dilation_ ) )
                .set( "has_bias", has_bias_ );

            return meta;
        }

        void fromMetadata( const SerializationMetadata& meta ) override
        {
            if ( auto v = meta.tryGetInt( "channels" ) )
            {
                channels_ = static_cast<dim_t>( *v );
            }

            if ( auto v = meta.tryGetInt( "kernel_width" ) )
            {
                kernel_width_ = static_cast<dim_t>( *v );
            }

            if ( auto v = meta.tryGetInt( "dilation" ) )
            {
                dilation_ = static_cast<dim_t>( *v );
            }

            if ( auto v = meta.tryGetBool( "has_bias" ) )
            {
                has_bias_ = *v;
            }
        }

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "CausalConv1dConfig( channels=" << channels_
                << ", kernel_width=" << kernel_width_
                << ", dilation=" << dilation_
                << ", has_bias=" << (has_bias_ ? "true" : "false") << " )";

            return oss.str();
        }

    private:
        dim_t channels_{ 0 };
        dim_t kernel_width_{ 0 };
        dim_t dilation_{ 1 };
        bool  has_bias_{ false };
    };
}
