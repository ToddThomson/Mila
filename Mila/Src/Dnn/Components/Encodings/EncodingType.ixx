/**
 * @file EncodingType.ixx
 * @brief Positional encoding strategy selection used by Transformer components.
 */

    module;
#include <string>
#include <stdexcept>

export module Dnn.EncodingType;

namespace Mila::Dnn
{
    /**
     * @brief Positional encoding strategies.
     */
    export enum class EncodingType
    {
        Learned,    ///< Learned absolute position embeddings (GPT-2 style)
        RoPE,       ///< Rotary Position Embeddings (LLaMA style)
        ALiBi       ///< Attention with Linear Biases (MPT / BLOOM style)
    };

    /**
     * @brief Convert EncodingType to string.
     */
    export inline std::string encodingTypeToString( EncodingType p )
    {
        switch ( p )
        {
            case EncodingType::Learned: return "Learned";
            case EncodingType::RoPE:    return "RoPE";
            case EncodingType::ALiBi:   return "ALiBi";
            default:                    return "Unknown";
        }
    }

    /**
     * @brief Parse string to PositionalEncodingType.
     *
     * @throws std::invalid_argument on unknown value.
     */
    export inline EncodingType stringToEncodingType( const std::string& v )
    {
        if ( v == "Learned" ) return EncodingType::Learned;
        if ( v == "RoPE" )    return EncodingType::RoPE;
        if ( v == "ALiBi" )   return EncodingType::ALiBi;

        throw std::invalid_argument( "Unknown EncodingType: " + v );
    }
}