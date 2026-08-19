/**
 * @file GatedDeltaRule.Config.ixx
 * @brief Configuration for the gated delta rule (linear-attention mixer).
 *
 * All four geometry values are structurally required. Qwen 3.8 27B uses 16 key heads,
 * 48 value heads, and 128 for both head dimensions.
 */

module;
#include <stdexcept>
#include <string>
#include <sstream>
#include <utility>

export module Dnn.Components.GatedDeltaRuleConfig;

import Dnn.TensorTypes;
import Dnn.ComponentConfig;
import Serialization.Metadata;

namespace Mila::Dnn
{
    using Serialization::SerializationMetadata;

    export class GatedDeltaRuleConfig : public ComponentConfig
    {
    public:
        GatedDeltaRuleConfig( dim_t num_key_heads, dim_t num_value_heads,
            dim_t head_key_dim, dim_t head_value_dim )
            : num_key_heads_( num_key_heads ), num_value_heads_( num_value_heads ),
              head_key_dim_( head_key_dim ), head_value_dim_( head_value_dim )
        {
            validate();
        }

        dim_t getNumKeyHeads() const noexcept { return num_key_heads_; }
        dim_t getNumValueHeads() const noexcept { return num_value_heads_; }
        dim_t getHeadKeyDim() const noexcept { return head_key_dim_; }
        dim_t getHeadValueDim() const noexcept { return head_value_dim_; }

        /// Value heads served by each key head (27B: 3).
        dim_t getHeadGroupSize() const noexcept { return num_value_heads_ / num_key_heads_; }

        /// Width of the q and k projections: num_key_heads * head_key_dim.
        dim_t getKeyWidth() const noexcept { return num_key_heads_ * head_key_dim_; }

        /// Width of the v projection and of the output: num_value_heads * head_value_dim.
        dim_t getValueWidth() const noexcept { return num_value_heads_ * head_value_dim_; }

        /// Carried state elements per batch item: heads * head_key_dim * head_value_dim.
        dim_t getStateElementsPerBatch() const noexcept
        {
            return num_value_heads_ * head_key_dim_ * head_value_dim_;
        }

        void validate() const override
        {
            if ( num_key_heads_ <= 0 || num_value_heads_ <= 0 )
            {
                throw std::invalid_argument( "GatedDeltaRuleConfig: head counts must be > 0" );
            }

            if ( num_value_heads_ % num_key_heads_ != 0 )
            {
                throw std::invalid_argument(
                    "GatedDeltaRuleConfig: num_value_heads must be divisible by num_key_heads" );
            }

            if ( head_key_dim_ <= 0 || head_value_dim_ <= 0 )
            {
                throw std::invalid_argument( "GatedDeltaRuleConfig: head dims must be > 0" );
            }

            // The kernel holds one state COLUMN per thread in registers, unrolled to a
            // compile-time bound, and launches one thread per value dimension. Both limits
            // are kernel properties rather than modelling ones, and both are hard: past
            // them the launch would silently compute against a truncated state.
            if ( head_key_dim_ > 128 )
            {
                throw std::invalid_argument(
                    "GatedDeltaRuleConfig: head_key_dim must be <= 128 (the kernel keeps a "
                    "state column of that many floats in registers)" );
            }

            if ( head_value_dim_ > 1024 )
            {
                throw std::invalid_argument(
                    "GatedDeltaRuleConfig: head_value_dim must be <= 1024 (one thread per "
                    "value dimension)" );
            }
        }

        SerializationMetadata toMetadata() const override
        {
            SerializationMetadata meta;

            meta.set( "num_key_heads", static_cast<int64_t>( num_key_heads_ ) )
                .set( "num_value_heads", static_cast<int64_t>( num_value_heads_ ) )
                .set( "head_key_dim", static_cast<int64_t>( head_key_dim_ ) )
                .set( "head_value_dim", static_cast<int64_t>( head_value_dim_ ) );

            return meta;
        }

        void fromMetadata( const SerializationMetadata& meta ) override
        {
            if ( auto v = meta.tryGetInt( "num_key_heads" ) )
            {
                num_key_heads_ = static_cast<dim_t>( *v );
            }

            if ( auto v = meta.tryGetInt( "num_value_heads" ) )
            {
                num_value_heads_ = static_cast<dim_t>( *v );
            }

            if ( auto v = meta.tryGetInt( "head_key_dim" ) )
            {
                head_key_dim_ = static_cast<dim_t>( *v );
            }

            if ( auto v = meta.tryGetInt( "head_value_dim" ) )
            {
                head_value_dim_ = static_cast<dim_t>( *v );
            }
        }

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "GatedDeltaRuleConfig( key_heads=" << num_key_heads_
                << ", value_heads=" << num_value_heads_
                << ", head_key_dim=" << head_key_dim_
                << ", head_value_dim=" << head_value_dim_ << " )";

            return oss.str();
        }

    private:
        dim_t num_key_heads_{ 0 };
        dim_t num_value_heads_{ 0 };
        dim_t head_key_dim_{ 0 };
        dim_t head_value_dim_{ 0 };
    };
}
