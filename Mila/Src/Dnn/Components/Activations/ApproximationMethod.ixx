/**
 * @file ApproximationMethod.ixx
 * @brief Shared approximation method enum for activation functions.
 *
 * Export a small enum and helper to-string for reuse by GELU, SwiGLU and others.
 */

module;
#include <string_view>

export module Dnn.ApproximationMethod;

namespace Mila::Dnn
{
    /**
     * @brief Approximation methods usable by activation functions.
     */
    export enum class ApproximationMethod {
        Exact,   ///< Exact implementation using erf
        Tanh,    ///< Fast tanh-based approximation
        Sigmoid  ///< Sigmoid-based approximation
    };

    /**
     * @brief Convert ApproximationMethod to a short string.
     *
     * Returns a constexpr std::string_view suitable for logging/serialization.
     */
    export constexpr std::string_view toString( ApproximationMethod m ) noexcept
    {
        switch ( m )
        {
            case ApproximationMethod::Exact:  return "Exact";
            case ApproximationMethod::Tanh:   return "Tanh";
            case ApproximationMethod::Sigmoid:return "Sigmoid";
            default:                          return "Unknown";
        }
    }
}