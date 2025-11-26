/**
 * @file ComponentType.ixx
 * @brief Enumeration of built-in component types supported by the deserializer.
 */

module;
#include <string>
#include <string_view>
#include <algorithm>
#include <cctype>

export module Dnn.ComponentType;

namespace Mila::Dnn
{
    export enum class ComponentType : int
    {
        Unknown = 0,
        Linear,
        Gelu,
        LayerNorm,
        Attention,
        Residual,
        MLP,
        Transformer,
        Encoder
    };

    export inline std::string toString( ComponentType t ) noexcept
    {
        switch ( t )
        {
            case ComponentType::Linear:
                return "Linear";
            case ComponentType::Gelu:
                return "Gelu";
            case ComponentType::LayerNorm:
                return "LayerNorm";
            case ComponentType::Attention:
                return "Attention";
            case ComponentType::Residual:
                return "Residual";
            case ComponentType::MLP:
                return "MLP";
            case ComponentType::Transformer:
                return "Transformer";
            case ComponentType::Encoder:
                return "Encoder";
            default:
                return "Unknown";
        }
    }

    export inline ComponentType fromString( std::string_view s ) noexcept
    {
        std::string low; low.reserve( s.size() );
        for ( unsigned char c : std::string_view( s ) ) low.push_back( static_cast<char>(std::tolower( c )) );

        if ( low == "linear" ) 
            return ComponentType::Linear;
        if ( low == "gelu" )
            return ComponentType::Gelu;
        if ( low == "layernorm" )
            return ComponentType::LayerNorm;
        if ( low == "attention" )
            return ComponentType::Attention;
        if ( low == "residual" )
            return ComponentType::Residual;
        if ( low == "mlp" )
            return ComponentType::MLP;
        if ( low == "transformer" )
            return ComponentType::Transformer;
        if ( low == "encoder" )
            return ComponentType::Encoder;

        return ComponentType::Unknown;
    }
}