/**
 * @file SoftmaxConfig.ixx
 * @brief Configuration interface for the Softmax module in the Mila DNN framework.
 */

module;
#include <cstdint>
#include <string>

export module Dnn.Components.Softmax:Config;

import Dnn.ComponentConfig;
import Serialization.Metadata;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Serialization;

    /**
     * @brief Configuration class for Softmax module.
     *
     * Provides a type-safe fluent interface for configuring Softmax modules.
     */
    export class SoftmaxConfig : public ComponentConfig 
    {
    public:
        
        /**
         * @brief fluent setter for the axis along which to apply softmax.
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
        void validate() const override
        {
            ComponentConfig::validate();
        }

        /**
         * @brief Serialize this configuration to SerializationMetadata.
         *
         * @return SerializationMetadata containing configuration parameters
         */
        SerializationMetadata toSerializationMetadata() const
        {
            SerializationMetadata meta;
            meta.set( "precision", static_cast<int64_t>( precision_ ) )
                .set( "axis", axis_ );

            return meta;
        }

        /**
         * @brief Deserialize configuration from SerializationMetadata.
         *
         * Missing keys leave fields at their current values.
         *
         * @param meta SerializationMetadata containing configuration parameters
         */
        void fromSerializationMetadata( const SerializationMetadata& meta )
        {
            if ( meta.has( "precision" ) )
            {
                precision_ = static_cast<decltype( precision_ )>( meta.getInt( "precision" ) );
            }

            if ( meta.has( "axis" ) )
            {
                axis_ = meta.getInt( "axis" );
            }
        }

        /**
         * @brief Get a string representation of the configuration.
         *
         * @return std::string Human-readable representation
         */
        std::string toString() const override
        {
            return "SoftmaxConfig( precision=" + std::to_string( static_cast<int>( precision_ ) ) +
                   ", axis=" + std::to_string( axis_ ) + ")";
        }

    private:
        int64_t axis_ = -1;
    };
}