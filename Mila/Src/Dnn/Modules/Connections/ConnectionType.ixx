/**
 * @file ConnectionType.ixx
 * @brief Definition of connection function types used by the Mila DNN library.
 */

module;
#include <string>
#include <stdexcept>

export module Dnn.ConnectionType;

namespace Mila::Dnn
{
    /**
     * @brief Connection types supported by residual and skip-connection modules.
     *
     * Defines how the input and transformed output are combined in
     * residual and skip-connection architectures.
     *
     * Currently only Addition is implemented. Other types (multiplication,
     * concatenation) may be added in the future.
     */
    export enum class ConnectionType
    {
        Addition    ///< Element-wise addition (y = x + F(x))
    };

    /**
     * @brief Converts a ConnectionType enum value to its string representation.
     *
     * @param type The ConnectionType to convert
     * @return std::string The string representation of the connection type
     *
     * Example:
     * @code
     * auto name = connectionTypeToString(ConnectionType::Addition);
     * // name == "Addition"
     * @endcode
     */
    export inline std::string connectionTypeToString( ConnectionType type )
    {
        switch (type)
        {
            case ConnectionType::Addition:
                return "Addition";
            
            default:
                return "Unknown";
        }
    }

    /**
     * @brief Converts a string to its corresponding ConnectionType enum value.
     *
     * @param name The string representation of a connection type
     * @return ConnectionType The corresponding enum value
     * @throws std::invalid_argument if the string doesn't match any known connection type
     *
     * Example:
     * @code
     * auto type = stringToConnectionType("Addition");
     * // type == ConnectionType::Addition
     * @endcode
     */
    export inline ConnectionType stringToConnectionType( const std::string& name )
    {
        if (name == "Addition") 
            return ConnectionType::Addition;

        throw std::invalid_argument( "Unknown connection type: " + name );
    }
}