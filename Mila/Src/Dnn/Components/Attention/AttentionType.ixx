/**
 * @file AttentionType.ixx
 * @brief Defines attention mechanism types used by transformer components.
 */

module;
#include <string>
#include <stdexcept>

export module Dnn.AttentionType;

namespace Mila::Dnn
{
    /**
     * @brief Enumeration of supported attention mechanism types.
     */
    export enum class AttentionType
    {
        MultiHead,      ///< Multi-Head Attention (MHA): independent Q, K, V per head
        GroupedQuery,   ///< Grouped Query Attention (GQA): Q heads grouped over shared K/V heads
        MultiQuery      ///< Multi-Query Attention (MQA): all Q heads share a single K/V head
    };

    /**
     * @brief Convert AttentionType to string.
     */
    export inline std::string attentionTypeToString( AttentionType t )
    {
        switch ( t )
        {
            case AttentionType::MultiHead:    return "MultiHead";
            case AttentionType::GroupedQuery: return "GroupedQuery";
            case AttentionType::MultiQuery:   return "MultiQuery";
            default:                          return "Unknown";
        }
    }

    /**
     * @brief Parse string to AttentionType.
     *
     * @throws std::invalid_argument on unknown value.
     */
    export inline AttentionType stringToAttentionType( const std::string& v )
    {
        if ( v == "MultiHead" )       return AttentionType::MultiHead;
        if ( v == "GroupedQuery" )   return AttentionType::GroupedQuery;
        if ( v == "MultiQuery" )     return AttentionType::MultiQuery;

        throw std::invalid_argument( "Unknown AttentionType: " + v );
    }
}