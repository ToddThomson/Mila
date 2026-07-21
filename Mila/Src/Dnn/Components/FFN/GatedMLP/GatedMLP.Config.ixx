/**
 * @file GatedMLP.Config.ixx
 * @brief Configuration for the gated feed-forward (GatedMLP) block.
 *
 * The gated FFN (Llama, Mistral, Qwen, ... and MoE experts) projects to a fused
 * 2H gate+up, applies a gated activation, then projects back down. See
 * Specifications/FfnAndMoE.md section 7.
 */

module;
#include <stdexcept>
#include <cstdint>
#include <string>
#include <utility>
#include <sstream>

export module Dnn.Components.GatedMLP:Config;

import Dnn.TensorTypes;
import Dnn.ComponentConfig;
import Dnn.ActivationType;
import Serialization.Metadata;

namespace Mila::Dnn
{
    using Serialization::SerializationMetadata;

    /**
     * @brief Configuration class for the gated feed-forward (GatedMLP) block.
     *
     * Block structure:
     *   Input -> fc_gate_up Linear(in -> 2H, fused) -> Swiglu gate (2H -> H)
     *         -> fc_down Linear(H -> in) -> Output
     *
     * Gated FFNs are typically bias-free; has_bias defaults to false. gate_activation
     * is serializable metadata; the model factory bridges it to the compile-time gate.
     */
    export class GatedMLPConfig : public ComponentConfig
    {
    public:
        /**
         * @brief Construct GatedMLP configuration.
         *
         * @param input_features Number of input (and output) features (must be > 0).
         * @param hidden_size Size of the gated intermediate dimension H (must be > 0).
         */
        GatedMLPConfig( dim_t input_features, dim_t hidden_size )
            : input_features_( input_features ), hidden_size_( hidden_size )
        {
        }

        template <typename Self>
        Self&& withBias( this Self&& self, bool has_bias )
        {
            self.has_bias_ = has_bias;
            return std::forward<Self>( self );
        }

        /**
         * @brief Set the gate activation function (serializable metadata).
         */
        template <typename Self>
        decltype(auto) withGateActivation( this Self&& self, ActivationType gate_activation )
        {
            self.gate_activation_ = gate_activation;
            return std::forward<Self>( self );
        }

        dim_t getInputFeatures() const noexcept { return input_features_; }
        dim_t getHiddenSize() const noexcept { return hidden_size_; }
        bool hasBias() const noexcept { return has_bias_; }
        ActivationType getGateActivation() const noexcept { return gate_activation_; }

        void validate() const override
        {
            if ( input_features_ <= 0 )
            {
                throw std::invalid_argument( "GatedMLPConfig: Input features must be greater than zero" );
            }

            if ( hidden_size_ <= 0 )
            {
                throw std::invalid_argument( "GatedMLPConfig: Hidden size must be greater than zero" );
            }
        }

        SerializationMetadata toMetadata() const override
        {
            SerializationMetadata meta;

            meta.set( "input_features", static_cast<int64_t>( input_features_ ) )
                .set( "hidden_size", static_cast<int64_t>( hidden_size_ ) )
                .set( "has_bias", has_bias_ )
                .set( "gate_activation", static_cast<int64_t>( gate_activation_ ) );

            return meta;
        }

        void fromMetadata( const SerializationMetadata& meta ) override
        {
            if ( auto in = meta.tryGetInt( "input_features" ) )
            {
                input_features_ = static_cast<dim_t>( *in );
            }

            if ( auto hs = meta.tryGetInt( "hidden_size" ) )
            {
                hidden_size_ = static_cast<dim_t>( *hs );
            }

            if ( auto hb = meta.tryGetBool( "has_bias" ) )
            {
                has_bias_ = *hb;
            }

            if ( auto ga = meta.tryGetInt( "gate_activation" ) )
            {
                gate_activation_ = static_cast<ActivationType>( *ga );
            }
        }

        std::string toString() const override
        {
            std::ostringstream oss;

            oss << "GatedMLPConfig( ";
            oss << "input_features=" << input_features_ << ", ";
            oss << "hidden_size=" << hidden_size_ << ", ";
            oss << "has_bias=" << (has_bias_ ? "true" : "false") << ", ";
            oss << "gate_activation=" << static_cast<int>( gate_activation_ );
            oss << " )";

            return oss.str();
        }

    private:

        dim_t input_features_{ 0 };
        dim_t hidden_size_{ 0 };
        bool has_bias_{ false };
        ActivationType gate_activation_{ ActivationType::Silu };
    };
}
