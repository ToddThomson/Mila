/**
 * @file OperationAttributes.ixx
 * @brief Common attributes for neural network operations.
 */

module;
#include <unordered_map>
#include <string>
#include <any>
#include <sstream>
#include <limits>

export module Compute.OperationAttributes;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Common attributes for neural network operations.
     *
     * This structure provides a centralized way to manage operation attributes,
     * combining frequently used properties as direct members for efficiency and
     * an extensible map for less common properties.
     */
    export struct OperationAttributes {
        /**
         * @brief Default constructor with default values.
         */
        OperationAttributes() = default;

        /**
         * @brief Constructor with common attributes.
         *
         * @param axisVal The axis parameter value.
         * @param epsilonVal The epsilon parameter value.
         * @param temperatureVal The temperature parameter value.
         */
        OperationAttributes( int64_t axisVal, float epsilonVal = 1e-5f, float temperatureVal = 1.0f )
            : axis( axisVal ), epsilon( epsilonVal ), temperature( temperatureVal ) {}

        /**
         * @brief Axis along which the operation is applied.
         *
         * Used by operations like Softmax, Reduction, and LayerNorm.
         * Negative values count backward from the end (-1 means the last dimension).
         */
        int64_t axis = -1;

        /**
         * @brief Small constant added for numerical stability.
         *
         * Used by operations like LayerNorm, BatchNorm, and some activation functions
         * to prevent division by zero or to stabilize gradients.
         */
        float epsilon = 1e-5f;

        /**
         * @brief Temperature parameter for Softmax-like operations.
         *
         * Controls the "peakiness" of probability distributions. Higher values
         * produce more uniform distributions, while lower values make distributions
         * more peaked.
         */
        float temperature = 1.0f;

        /**
         * @brief Gradient clipping threshold.
         *
         * Used to prevent exploding gradients during training by clipping
         * gradient values that exceed this threshold.
         */
        float clipThreshold = std::numeric_limits<float>::max();

        /**
         * @brief Dropout probability.
         *
         * Used by Dropout layers to specify the probability of zeroing an element.
         */
        float dropoutProb = 0.0f;

        /**
         * @brief Indicates whether the operation is in training mode.
         *
         * Some operations like BatchNorm and Dropout behave differently during
         * training versus inference.
         */
        bool trainingMode = false;

        /**
         * @brief Extensible property storage for operation-specific attributes.
         *
         * Allows storing arbitrary attributes that aren't common enough to warrant
         * direct member variables.
         */
        std::unordered_map<std::string, std::any> propsMap;

        /**
         * @brief Helper method to get values from the propsMap with type safety.
         *
         * @tparam TElementType The expected type of the property.
         * @param key The property key to retrieve.
         * @param defaultValue The default value to return if the property doesn't exist or has the wrong type.
         * @return TElementType The property value or defaultValue if not found.
         */
        template<typename T>
        T get( const std::string& key, const T& defaultValue ) const {
            auto it = propsMap.find( key );
            if ( it != propsMap.end() ) {
                try {
                    return std::any_cast<T>(it->second);
                }
                catch ( const std::bad_any_cast& ) {
                    // Return default if type doesn't match
                    return defaultValue;
                }
            }
            return defaultValue;
        }

        /**
         * @brief Helper method to set a value in the propsMap.
         *
         * @tparam TElementType The type of the property to set.
         * @param key The property key.
         * @param value The property value.
         */
        template<typename T>
        void set( const std::string& key, const T& value ) {
            propsMap[ key ] = value;
        }

        /**
         * @brief Check if a custom property exists in the propsMap.
         *
         * @param key The property key to check.
         * @return bool True if the property exists.
         */
        bool has( const std::string& key ) const {
            return propsMap.find( key ) != propsMap.end();
        }

        /**
         * @brief Remove a custom property from the propsMap.
         *
         * @param key The property key to remove.
         * @return bool True if the property was found and removed.
         */
        bool remove( const std::string& key ) {
            return propsMap.erase( key ) > 0;
        }

        /**
         * @brief Get the number of custom properties.
         *
         * @return size_t The number of properties in the propsMap.
         */
        size_t customPropertyCount() const {
            return propsMap.size();
        }

        /**
         * @brief Check if a property exists and is of the expected type.
         *
         * @tparam TElementType The expected property type.
         * @param key The property key to check.
         * @return bool True if the property exists and is of type TElementType.
         */
        template<typename T>
        bool isType( const std::string& key ) const {
            auto it = propsMap.find( key );
            if ( it == propsMap.end() ) return false;

            try {
                std::any_cast<T>(it->second);
                return true;
            }
            catch ( const std::bad_any_cast& ) {
                return false;
            }
        }

        /**
         * @brief Serialize attributes to a string representation.
         *
         * Useful for debugging, logging, and potentially for simple serialization.
         *
         * @return std::string A string representation of the attributes.
         */
        std::string toString() const {
            std::ostringstream oss;
            oss << "OperationAttributes{axis=" << axis
                << ", epsilon=" << epsilon
                << ", temperature=" << temperature
                << ", clipThreshold=" << clipThreshold
                << ", dropoutProb=" << dropoutProb
                << ", trainingMode=" << (trainingMode ? "true" : "false")
                << ", customProps=[";

            bool first = true;
            for ( const auto& [key, value] : propsMap ) {
                if ( !first ) oss << ", ";
                first = false;
                oss << key << "=<custom>";
            }
            oss << "]}";

            return oss.str();
        }
    };
}