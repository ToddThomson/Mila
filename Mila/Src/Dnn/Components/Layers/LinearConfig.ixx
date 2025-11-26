/**
 * @file LinearConfig.ixx
 * @brief Configuration interface for the Linear module in the Mila DNN framework.
 *
 * Defines the LinearConfig class, providing a type-safe fluent interface for configuring
 * Linear (fully connected) layer modules. Inherits from ModuleConfig CRTP base and adds
 * Linear-specific options: input/output feature dimensions and bias configuration.
 *
 * Exposed as part of the Linear module via module partitions.
 */

module;
#include <memory> // for nlohmann::json to compile in VS2026
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
import nlohmann.json;

namespace Mila::Dnn
{
    using json = nlohmann::json;

    /**
     * @class LinearConfig
     * @brief Configuration object for a Linear (fully connected) layer.
     *
     * LinearConfig provides a minimal, type-safe fluent interface for describing the
     * parameters required to construct a Linear layer: the number of input features,
     * the number of output features, and whether the layer contains a bias term.
     *
     * Typical usage:
     * @code
     * LinearConfig cfg{128, 64};
     * cfg.withBias(true).validate();
     * @endcode
     *
     * @note Instances are lightweight value objects intended to be passed into module
     *       factories or constructors. Validation should be invoked prior to creating
     *       runtime objects to surface configuration errors early.
     */
    export class LinearConfig : public ComponentConfig
    {
    public:
        /**
         * @brief Construct a LinearConfig with required feature dimensions.
         *
         * The constructor initializes the two required dimensions for the Linear layer.
         *
         * @param input_features Number of input features (channels). Must be > 0.
         * @param output_features Number of output features (channels). Must be > 0.
         */
        LinearConfig( dim_t input_features, dim_t output_features )
			: input_features_( input_features ), output_features_( output_features ), ComponentConfig( "linear" )
        {
        }

        /**
         * @brief C++23-style fluent setter for bias enable flag.
         *
         * Uses the explicit object parameter style so chaining preserves the
         * exact value category (lvalue/rvalue) of the caller.
         *
         * @param has_bias True to include a bias parameter, false to omit it.
         * @return Self&& Reference to this configuration (for chaining).
         */
        template <typename Self>
        decltype(auto) withBias( this Self&& self, bool has_bias )
        {
            self.has_bias_ = has_bias;
            return std::forward<Self>( self );
        }

        /**
         * @brief C++23-style fluent setter for input features.
         *
         * Provided to support fluent construction patterns when the caller
         * prefers setters over the constructor.
         */
        template <typename Self>
        decltype(auto) withInputFeatures( this Self&& self, dim_t input_features )
        {
            self.input_features_ = input_features;
            return std::forward<Self>( self );
        }

        /**
         * @brief C++23-style fluent setter for output features.
         *
         * Provided to support fluent construction patterns when the caller
         * prefers setters over the constructor.
         */
        template <typename Self>
        decltype(auto) withOutputFeatures( this Self&& self, dim_t output_features )
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

        /**
         * @brief Validate the configuration values.
         *
         * This method performs checks that the configuration is usable for constructing
         * a runtime Linear module. It calls the base class validate implementation and
         * then checks Linear-specific constraints.
         *
         * @throws std::invalid_argument If any required parameter is invalid.
         */
        void validate() const override
        {
            if (input_features_ <= 0 || output_features_ <= 0)
            {
                throw std::invalid_argument( "LinearConfig: Input and output features must be greater than zero" );
            }
        }

        /**
         * @brief Serialize this configuration to JSON (ModuleConfig interface).
         *
         * Produces keys:
         * - "name" : string
         * - "precision" : integer (underlying value of ComputePrecision::Policy)
         * - "input_features" : integer
         * - "output_features" : integer
         * - "has_bias" : boolean
         */
        json toJson() const
        {
            json j;
            j["name"] = name_;
            j["precision"] = static_cast<int>( precision_ );
            j["input_features"] = static_cast<int64_t>( input_features_ );
            j["output_features"] = static_cast<int64_t>( output_features_ );
            j["has_bias"] = has_bias_;

            return j;
        }

        /**
         * @brief Deserialize configuration from JSON (ModuleConfig interface).
         *
         * Missing keys leave fields at their current values. Type errors are
         * propagated from nlohmann::json getters.
         */
        void fromJson( const json& j )
        {
            if ( j.contains( "name" ) )
            {
                name_ = j.at( "name" ).get<std::string>();
            }

            if ( j.contains( "precision" ) )
            {
                precision_ = static_cast<decltype( precision_)>( j.at( "precision" ).get<int>() );
            }

            if ( j.contains( "input_features" ) )
            {
                input_features_ = static_cast<dim_t>( j.at( "input_features" ).get<int64_t>() );
            }

            if ( j.contains( "output_features" ) )
            {
                output_features_ = static_cast<dim_t>( j.at( "output_features" ).get<int64_t>() );
            }

            if ( j.contains( "has_bias" ) )
            {
                has_bias_ = j.at( "has_bias" ).get<bool>();
            }
        }

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "LinearConfig(input_features=" << input_features_
                << ", output_features=" << output_features_
                << ", has_bias=" << std::boolalpha << has_bias_ << ")";
            return oss.str();
		}

    private:
        /**
         * @brief Number of input features (channels) expected by the layer.
         *
         * Must be greater than zero.
         */
        dim_t input_features_;

        /**
         * @brief Number of output features (channels) produced by the layer.
         *
         * Must be greater than zero.
         */
        dim_t output_features_;

        /**
         * @brief Whether the layer has a bias term.
         *
         * Default is true.
         */
        bool has_bias_{ true };
    };
}