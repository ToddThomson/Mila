/**
 * @file CausalConv1d.Config.ixx
 * @brief Configuration for the depthwise causal 1-D convolution component.
 *
 * Channels and kernel width are structurally required -- there is no sensible default for
 * either -- so both are constructor parameters. Bias is optional and defaults to false,
 * which is what a convolution inside a gated recurrence normally carries.
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
         * @param kernel_width Filter taps. The convolution looks back kernel_width - 1
         *                     positions, which is also the retained state depth.
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

        dim_t getChannels() const noexcept { return channels_; }
        dim_t getKernelWidth() const noexcept { return kernel_width_; }
        bool hasBias() const noexcept { return has_bias_; }

        /// Retained input rows = kernel_width - 1. The convolution's whole memory.
        dim_t getStateRows() const noexcept { return kernel_width_ - 1; }

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

            // The state shift stages kernel_width - 1 rows in registers, so the bound is a
            // kernel property rather than a modelling one. Qwen 3.8 uses 4.
            if ( kernel_width_ > 8 )
            {
                throw std::invalid_argument(
                    "CausalConv1dConfig: kernel_width must be <= 8 (the kernel stages "
                    "kernel_width - 1 rows in registers)" );
            }
        }

        SerializationMetadata toMetadata() const override
        {
            SerializationMetadata meta;

            meta.set( "channels", static_cast<int64_t>( channels_ ) )
                .set( "kernel_width", static_cast<int64_t>( kernel_width_ ) )
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
                << ", has_bias=" << (has_bias_ ? "true" : "false") << " )";

            return oss.str();
        }

    private:
        dim_t channels_{ 0 };
        dim_t kernel_width_{ 0 };
        bool  has_bias_{ false };
    };
}
