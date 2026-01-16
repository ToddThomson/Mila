/**
 * @file NormType.ixx
 * @brief Normalization layer type enumeration used by Transformer components.
 */

    module;
#include <string>
#include <stdexcept>

export module Dnn.NormType;

namespace Mila::Dnn
{
    /**
     * @brief Normalization type selection.
     */
    export enum class NormType
    {
        LayerNorm,  ///< Standard LayerNorm (mean + variance)
        RMSNorm     ///< Root Mean Square Norm (variance-only)
    };

    /**
     * @brief Convert NormType to string.
     */
    export inline std::string normTypeToString( NormType n )
    {
        switch ( n )
        {
            case NormType::LayerNorm: return "LayerNorm";
            case NormType::RMSNorm:   return "RMSNorm";
            default:                  return "Unknown";
        }
    }

    /**
     * @brief Parse string to NormType.
     *
     * @throws std::invalid_argument on unknown value.
     */
    export inline NormType stringToNormType( const std::string& v )
    {
        if ( v == "LayerNorm" ) return NormType::LayerNorm;
        if ( v == "RMSNorm" )   return NormType::RMSNorm;

        throw std::invalid_argument( "Unknown NormType: " + v );
    }
}