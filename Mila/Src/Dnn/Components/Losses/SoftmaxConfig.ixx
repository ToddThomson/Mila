/**
 * @file SoftmaxConfig.ixx
 * @brief Configuration interface for the Softmax module in the Mila DNN framework.
 *
 * Defines the SoftmaxConfig class, providing a type-safe fluent interface for configuring
 * the Softmax activation function module.
 */

module;
#include <cstdint>
#include <string>
#include <memory> // TJT: Note: Without this include, VS2026 will not complile nlohmann::json

export module Dnn.Components.Softmax:Config;

import Dnn.ComponentConfig;
import nlohmann.json;

namespace Mila::Dnn
{
    using json = nlohmann::json;

    /**
     * @brief Configuration class for Softmax module.
     *
     * Provides a type-safe fluent interface for configuring Softmax modules.
     */
    export class SoftmaxConfig : public ComponentConfig 
    {
    public:
        
        /**
         * @brief Default constructor with name "softmax".
         *
         * @note When adding multiple Softmax components to a container,
         *       use .withName() to provide unique names.
         */
        SoftmaxConfig() : ComponentConfig( "softmax" )
        {
        }

        /**
         * @brief C++23-style fluent setter for the axis along which to apply softmax.
         *
         * @param axis Dimension for softmax computation (default: -1 for last dimension)
         * @return Self&& for method chaining
         */
        template <typename Self>
        Self&& withAxis( this Self&& self, int64_t axis )
        {
            self.axis_ = axis;
            return std::forward<Self>( self );
        }

        /**
         * @brief Get the configured axis value.
         *
         * @return int64_t The axis along which softmax will be computed
         */
        int64_t getAxis() const noexcept { return axis_; }

        /**
         * @brief Validate configuration parameters.
         *
         * @throws std::invalid_argument If validation fails
         */
        void validate() const override {
            // No additional validation needed for Softmax
        }

        /**
         * @brief Serialize this configuration to JSON.
         *
         * Keys:
         * - "name" : string
         * - "precision" : integer (underlying value of ComputePrecision::Policy)
         * - "axis" : integer (axis for softmax)
         */
        json toJson() const
        {
            json j;
            j["name"] = name_;
            //j["precision"] = static_cast<int>( precision_ );
            //j["axis"] = axis_;

            return j;
        }

        /**
         * @brief Deserialize configuration from JSON.
         *
         * Missing keys leave fields at their current values.
         */
   //     void fromJson( const json& j )
   //     {
   //         if ( j.contains( "name" ) ) {
   //             name_ = j.at( "name" ).get<std::string>();
   //         }

			///*

   //         if ( j.contains( "precision" ) ) {
   //             precision_ = static_cast<decltype(precision_)>( j.at( "precision" ).get<int>() );
   //         }

   //         if ( j.contains( "axis" ) ) {
   //             axis_ = j.at( "axis" ).get<int64_t>();
   //         }*/
   //     }

        /**
         * @brief Get a string representation of the configuration.
         *
         * @return std::string Human-readable representation
		 */
        std::string toString() const override {
            return "SoftmaxConfig(name=" + name_ +
                   ", precision=" + std::to_string( static_cast<int>( precision_ ) ) +
                   ", axis=" + std::to_string( axis_ ) + ")";
		}

    private:
        int64_t axis_ = -1;
    };
}