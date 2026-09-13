/**
 * @file MixtureOfExperts.Config.ixx
 * @brief Configuration for a mixture-of-experts bank: expert geometry, count, and experts per token.
 *
 * Gemma 4 26B-A4B: 128 experts of intermediate width 704 over a 2816-wide stream, 8 per token.
 * See Specifications/Gemma4MoE.md Phase 6.
 */

module;
#include <stdexcept>
#include <string>
#include <sstream>
#include <format>
#include <cstdint>

export module Dnn.Components.MixtureOfExpertsConfig;

import Dnn.TensorTypes;
import Dnn.ComponentConfig;
import Serialization.Metadata;

namespace Mila::Dnn
{
    using Serialization::SerializationMetadata;

    export class MixtureOfExpertsConfig : public ComponentConfig
    {
    public:
        MixtureOfExpertsConfig( dim_t hidden_size, dim_t expert_intermediate_size, dim_t num_experts, dim_t top_k )
            : hidden_size_( hidden_size ), expert_intermediate_size_( expert_intermediate_size ),
              num_experts_( num_experts ), top_k_( top_k )
        {
            validate();
        }

        dim_t getHiddenSize() const noexcept { return hidden_size_; }

        /// One expert's gated width I: gate_up_proj is [E, 2I, H], down_proj is [E, H, I].
        dim_t getExpertIntermediateSize() const noexcept { return expert_intermediate_size_; }

        dim_t getNumExperts() const noexcept { return num_experts_; }

        /// Experts each token is dispatched to; the routing tensors are [..., top_k].
        dim_t getTopK() const noexcept { return top_k_; }

        void validate() const override
        {
            if ( hidden_size_ <= 0 || expert_intermediate_size_ <= 0 )
            {
                throw std::invalid_argument( "MixtureOfExpertsConfig: hidden_size and expert_intermediate_size must be > 0" );
            }

            if ( num_experts_ <= 0 )
            {
                throw std::invalid_argument( "MixtureOfExpertsConfig: num_experts must be > 0" );
            }

            if ( top_k_ <= 0 || top_k_ > num_experts_ )
            {
                throw std::invalid_argument( std::format(
                    "MixtureOfExpertsConfig: top_k must lie in [1, num_experts = {}], got {}", num_experts_, top_k_ ) );
            }
        }

        SerializationMetadata toMetadata() const override
        {
            SerializationMetadata meta;

            meta.set( "hidden_size", static_cast<int64_t>( hidden_size_ ) )
                .set( "expert_intermediate_size", static_cast<int64_t>( expert_intermediate_size_ ) )
                .set( "num_experts", static_cast<int64_t>( num_experts_ ) )
                .set( "top_k", static_cast<int64_t>( top_k_ ) );

            return meta;
        }

        void fromMetadata( const SerializationMetadata& meta ) override
        {
            if ( auto v = meta.tryGetInt( "hidden_size" ) )
            {
                hidden_size_ = static_cast<dim_t>( *v );
            }

            if ( auto v = meta.tryGetInt( "expert_intermediate_size" ) )
            {
                expert_intermediate_size_ = static_cast<dim_t>( *v );
            }

            if ( auto v = meta.tryGetInt( "num_experts" ) )
            {
                num_experts_ = static_cast<dim_t>( *v );
            }

            if ( auto v = meta.tryGetInt( "top_k" ) )
            {
                top_k_ = static_cast<dim_t>( *v );
            }
        }

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "MixtureOfExpertsConfig( hidden_size=" << hidden_size_
                << ", expert_intermediate_size=" << expert_intermediate_size_
                << ", num_experts=" << num_experts_ << ", top_k=" << top_k_ << " )";

            return oss.str();
        }

    private:
        dim_t hidden_size_{ 0 };
        dim_t expert_intermediate_size_{ 0 };
        dim_t num_experts_{ 0 };
        dim_t top_k_{ 0 };
    };
}
