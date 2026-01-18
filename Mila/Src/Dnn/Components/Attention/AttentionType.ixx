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
        Standard,       ///< Standard multi-head attention (MHA)
        GroupedQuery,   ///< Grouped Query Attention (GQA)
        MultiQuery      ///< Multi-Query Attention (MQA)
    };

    /**
     * @brief Convert AttentionType to string.
     */
    export inline std::string attentionTypeToString( AttentionType t )
    {
        switch ( t )
        {
            case AttentionType::Standard:     return "Standard";
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
        if ( v == "Standard" )       return AttentionType::Standard;
        if ( v == "GroupedQuery" )   return AttentionType::GroupedQuery;
        if ( v == "MultiQuery" )     return AttentionType::MultiQuery;

        throw std::invalid_argument( "Unknown AttentionType: " + v );
    }
}