/**
 * @file LinearConfig.ixx
 * @brief Configuration for the Linear (fully connected) layer.
 *
 * Provides fluent setters, validation, and metadata serialization for Linear.
 */

module;
#include <memory>
#include <stdexcept>
#include <cstdint>
#include <string>
#include <utility>
#include <sstream>
#include <ios>

export module Dnn.Components.Linear:Config;

import Dnn.Component;
import Dnn.ComponentConfig;
import Dnn.TensorTypes;
import Serialization.Metadata;

namespace Mila::Dnn
{
    using Serialization::SerializationMetadata;

    /**
     * @class LinearConfig
     * @brief Configuration object for a Linear (fully connected) layer.
     *
     * LinearConfig describes parameters required to construct a Linear layer:
     * input and output feature dimensions and whether a bias is present.
     *
     * Instances are lightweight value objects intended to be passed into module
     * factories or constructors. Call `validate()` prior to constructing runtime
     * objects to surface configuration errors early.
     */
    export class LinearConfig : public ComponentConfig
    {
    public:
        /**
         * @brief Construct a LinearConfig with required feature dimensions.
         *
         * @param input_features Number of input features (must be > 0).
         * @param output_features Number of output features (must be > 0).
         */
        LinearConfig( dim_t input_features, dim_t output_features )
            : input_features_( input_features ), output_features_( output_features )
        {
        }

        // DEBUG: Temp withRowMajor

        template <typename Self>
        Self&& withRowMajor( this Self&& self, bool row_major )
        {
            self.uses_row_major_ = row_major;
            return std::forward<Self>( self );
        }

        /**
         * @brief Fluent setter for bias enable flag.
         *
         * @tparam Self Concrete config type (deduced via explicit object parameter)
         * @param has_bias True to include a bias parameter, false to omit it.
         * @return Self&& Reference to this configuration for chaining.
         */
        template <typename Self>
        Self&& withBias( this Self&& self, bool has_bias )
        {
            self.has_bias_ = has_bias;
            return std::forward<Self>( self );
        }

        /**
         * @brief Fluent setter for input features.
         *
         * @param input_features Number of input features.
         * @return Self&& Reference to this configuration for chaining.
         */
        template <typename Self>
        Self&& withInputFeatures( this Self&& self, dim_t input_features )
        {
            self.input_features_ = input_features;
            return std::forward<Self>( self );
        }

        /**
         * @brief Fluent setter for output features.
         *
         * @param output_features Number of output features.
         * @return Self&& Reference to this configuration for chaining.
         */
        template <typename Self>
        Self&& withOutputFeatures( this Self&& self, dim_t output_features )
        {
            self.output_features_ = output_features;
            return std::forward<Self>( self );
        }

        /**
         * @brief Get the configured number of input features.
         * @return dim_t Number of input features configured.
         */
        dim_t getInputFeatures() const noexcept
        {
            return input_features_;
        }

        /**
         * @brief Get the configured number of output features.
         * @return dim_t Number of output features configured.
         */
        dim_t getOutputFeatures() const noexcept
        {
            return output_features_;
        }

        /**
         * @brief Query whether the bias term is enabled.
         * @return bool True if bias is enabled; false otherwise.
         */
        bool hasBias() const noexcept
        {
            return has_bias_;
        }

        bool getRowMajor() const noexcept
        {
            return uses_row_major_;
        }

        /**
         * @brief Validate the configuration values.
         *
         * Throws std::invalid_argument when the configuration is invalid.
         */
        void validate() const override
        {
            if ( input_features_ <= 0 || output_features_ <= 0 )
            {
                throw std::invalid_argument( "LinearConfig: Input and output features must be greater than zero" );
            }
        }

        /**
         * @brief Convert configuration into SerializationMetadata.
         *
         * Produces keys:
         * - "precision" : integer (ComputePrecision::Policy)
         * - "input_features" : integer
         * - "output_features" : integer
         * - "has_bias" : boolean
         *
         * @return SerializationMetadata Metadata representing this configuration.
         */
        SerializationMetadata toMetadata() const override
        {
            SerializationMetadata meta;
            meta.set( "precision", static_cast<int64_t>( precision_ ) )
                .set( "input_features", static_cast<int64_t>( input_features_ ) )
                .set( "output_features", static_cast<int64_t>( output_features_ ) )
                .set( "has_bias", has_bias_ );

            return meta;
        }

        /**
         * @brief Populate configuration from provided metadata.
         *
         * Missing keys are ignored, leaving defaults intact. Type mismatches
         * result in no assignment (use tryGet* helpers).
         *
         * @param meta Metadata to read configuration values from.
         */
        void fromMetadata( const SerializationMetadata& meta ) override
        {
            if ( auto p = meta.tryGetInt( "precision" ) )
            {
                precision_ = static_cast<decltype( precision_ )>( *p );
            }

            if ( auto in = meta.tryGetInt( "input_features" ) )
            {
                input_features_ = static_cast<dim_t>( *in );
            }

            if ( auto out = meta.tryGetInt( "output_features" ) )
            {
                output_features_ = static_cast<dim_t>( *out );
            }

            if ( auto hb = meta.tryGetBool( "has_bias" ) )
            {
                has_bias_ = *hb;
            }
        }

        /**
         * @brief Human-readable summary suitable for logging.
         * @return std::string Compact description of the configuration.
         */
        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "LinearConfig(input_features=" << input_features_
                << ", output_features=" << output_features_
                << ", has_bias=" << std::boolalpha << has_bias_ << ")";

            return oss.str();
        }

    private:
        /** Number of input features (must be > 0). */
        dim_t input_features_;

        /** Number of output features (must be > 0). */
        dim_t output_features_;

        /** Whether the layer has a bias term. Default is true. */
        bool has_bias_{ true };

        // DEBUG: Temp row-major flag (not yet used in LinearConfig or Linear)
        bool uses_row_major_{ false };
    };
}