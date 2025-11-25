/**
 * @file ActivationType.ixx
 * @brief Definition of activation function types used throughout the Mila library.
 */

module;
#include <string>
#include <stdexcept>

export module Dnn.ActivationType;

namespace Mila::Dnn
{
    /**
     * @brief Enumeration of supported activation function types.
     *
     * This enum class defines the different activation functions that can
     * be used throughout the Mila library, particularly in neural network layers.
     */
    export enum class ActivationType
    {
        None,       ///< No activation (identity function)
        Relu,       ///< Rectified Linear Unit: max(0, x)
        Gelu,       ///< Gaussian Error Linear Unit: x * phi(x) where phi() is the standard Gaussian CDF
        Silu,       ///< Sigmoid Linear Unit (Swish): x * sigmoid(x)
        Tanh,       ///< Hyperbolic Tangent: tanh(x)
        Sigmoid,    ///< Sigmoid function: 1 / (1 + exp(-x))
        LeakyRelu,  ///< Leaky ReLU: max(alpha * x, x) where alpha is typically 0.01
        Mish,       ///< Mish: x * tanh(softplus(x))
    };

    /**
     * @brief Converts an ActivationType enum value to its string representation.
     *
     * @param type The ActivationType to convert
     * @return std::string The string representation of the activation type
     */
    export inline std::string activationTypeToString( ActivationType type )
    {
        switch ( type ) {
            case ActivationType::None:      return "None";
            case ActivationType::Relu:      return "ReLU";
            case ActivationType::Gelu:      return "GELU";
            case ActivationType::Silu:      return "SiLU";
            case ActivationType::Tanh:      return "Tanh";
            case ActivationType::Sigmoid:   return "Sigmoid";
            case ActivationType::LeakyRelu: return "LeakyReLU";
            case ActivationType::Mish:      return "Mish";
            default:                        return "Unknown";
        }
    }

    /**
     * @brief Converts a string to its corresponding ActivationType enum value.
     *
     * @param name The string representation of an activation function
     * @return ActivationType The corresponding enum value
     * @throws std::invalid_argument if the string doesn't match any known activation function
     */
    export inline ActivationType stringToActivationType( const std::string& name )
    {
        if ( name == "None" )       return ActivationType::None;
        if ( name == "ReLU" )       return ActivationType::Relu;
        if ( name == "GELU" )       return ActivationType::Gelu;
        if ( name == "SiLU" )       return ActivationType::Silu;
        if ( name == "Tanh" )       return ActivationType::Tanh;
        if ( name == "Sigmoid" )    return ActivationType::Sigmoid;
        if ( name == "LeakyReLU" )  return ActivationType::LeakyRelu;
        if ( name == "Mish" )       return ActivationType::Mish;

        throw std::invalid_argument( "Unknown activation type: " + name );
    }
}