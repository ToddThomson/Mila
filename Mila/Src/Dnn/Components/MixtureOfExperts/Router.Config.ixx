/**
 * @file Router.Config.ixx
 * @brief Configuration for mixture-of-experts routing: how many experts, and how many run per token.
 *
 * Gemma 4 26B-A4B routes each 2816-wide token to 8 of 128 experts. See Specifications/Gemma4MoE.md Phase 5.
 */

module;
#include <stdexcept>
#include <string>
#include <sstream>
#include <format>
#include <utility>
#include <cstdint>

export module Dnn.Components.RouterConfig;

import Dnn.TensorTypes;
import Dnn.ComponentConfig;
import Serialization.Metadata;

namespace Mila::Dnn
{
    using Serialization::SerializationMetadata;

    export class RouterConfig : public ComponentConfig
    {
    public:
        RouterConfig( dim_t hidden_size, dim_t num_experts, dim_t top_k )
            : hidden_size_( hidden_size ), num_experts_( num_experts ), top_k_( top_k )
        {
            validate();
        }

        /// Epsilon of the router's unscaled RMS norm (Gemma 4: rms_norm_eps).
        template<typename Self>
        decltype(auto) withEpsilon( this Self&& self, float epsilon )
        {
            self.epsilon_ = epsilon;
            self.validate();

            return std::forward<Self>( self );
        }

        dim_t getHiddenSize() const noexcept { return hidden_size_; }
        dim_t getNumExperts() const noexcept { return num_experts_; }

        /// Experts each token is dispatched to.
        dim_t getTopK() const noexcept { return top_k_; }

        float getEpsilon() const noexcept { return epsilon_; }

        void validate() const override
        {
            if ( hidden_size_ <= 0 )
            {
                throw std::invalid_argument( "RouterConfig: hidden_size must be > 0" );
            }

            if ( num_experts_ <= 0 )
            {
                throw std::invalid_argument( "RouterConfig: num_experts must be > 0" );
            }

            if ( top_k_ <= 0 || top_k_ > num_experts_ )
            {
                throw std::invalid_argument( std::format(
                    "RouterConfig: top_k must lie in [1, num_experts = {}], got {}", num_experts_, top_k_ ) );
            }

            if ( !( epsilon_ >= 0.0f ) )
            {
                throw std::invalid_argument( "RouterConfig: epsilon must be >= 0" );
            }
        }

        SerializationMetadata toMetadata() const override
        {
            SerializationMetadata meta;

            meta.set( "hidden_size", static_cast<int64_t>( hidden_size_ ) )
                .set( "num_experts", static_cast<int64_t>( num_experts_ ) )
                .set( "top_k", static_cast<int64_t>( top_k_ ) )
                .set( "epsilon", epsilon_ );

            return meta;
        }

        void fromMetadata( const SerializationMetadata& meta ) override
        {
            if ( auto v = meta.tryGetInt( "hidden_size" ) )
            {
                hidden_size_ = static_cast<dim_t>( *v );
            }

            if ( auto v = meta.tryGetInt( "num_experts" ) )
            {
                num_experts_ = static_cast<dim_t>( *v );
            }

            if ( auto v = meta.tryGetInt( "top_k" ) )
            {
                top_k_ = static_cast<dim_t>( *v );
            }

            if ( auto v = meta.tryGetFloat( "epsilon" ) )
            {
                epsilon_ = *v;
            }
        }

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "RouterConfig( hidden_size=" << hidden_size_ << ", num_experts=" << num_experts_
                << ", top_k=" << top_k_ << ", epsilon=" << epsilon_ << " )";

            return oss.str();
        }

    private:
        dim_t hidden_size_{ 0 };
        dim_t num_experts_{ 0 };
        dim_t top_k_{ 0 };
        float epsilon_{ 1e-6f };
    };
}
