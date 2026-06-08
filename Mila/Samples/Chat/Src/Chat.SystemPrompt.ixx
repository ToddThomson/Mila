/**
 * @file Chat.SystemPrompt.ixx
 * @brief System prompt and tool definition loader for the Mila chat application.
 *
 * Reads a JSON file containing a system_prompt string and an optional tools
 * array. Parsed results are held as plain value types for use by Chat::run().
 */

module;
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <format>

export module Chat.SystemPrompt;

import Chat.Json;

namespace Mila::ChatApp
{
    /**
     * @brief A single parameter property in a tool's JSON schema.
     */
    export struct ToolParameterProperty
    {
        std::string type;
        std::string description;
    };

    /**
     * @brief JSON Schema descriptor for a tool's parameter object.
     *
     * Mirrors the subset of JSON Schema used by Llama 3.x tool definitions:
     * an object with named properties and a required array.
     */
    export struct ToolParameterSchema
    {
        std::string type{ "object" };
        std::vector<std::string> required;
        std::vector<std::pair<std::string, ToolParameterProperty>> properties;
    };

    /**
     * @brief Definition of a callable tool exposed to the model.
     *
     * Serialized into the system prompt by Chat::run() using the Llama 3.x
     * tool-calling JSON format before being passed to MessageFormatter.
     */
    export struct ToolDefinition
    {
        std::string name;
        std::string description;
        ToolParameterSchema parameters;
    };

    /**
     * @brief Parsed contents of a system prompt JSON file.
     *
     * Loaded once at Chat construction time via SystemPromptLoader::load().
     * tools is empty when the file contains no tools array or the array is empty.
     */
    export struct SystemPromptConfig
    {
        std::string system_prompt;
        std::vector<ToolDefinition> tools;
    };

    /**
     * @brief Loads and parses a system prompt JSON file.
     *
     * Expected file format:
     * @code
     * {
     *     "system_prompt": "You are a helpful assistant.",
     *     "tools": [
     *         {
     *             "name": "get_weather",
     *             "description": "Get current weather for a location.",
     *             "parameters": {
     *                 "type": "object",
     *                 "properties": {
     *                     "location": {
     *                         "type": "string",
     *                         "description": "City and country, e.g. London, UK"
     *                     }
     *                 },
     *                 "required": ["location"]
     *             }
     *         }
     *     ]
     * }
     * @endcode
     *
     * The tools array is optional. system_prompt is required.
     */
    export class SystemPromptLoader
    {
    public:

        /**
         * @brief Load and parse a system prompt JSON file.
         *
         * @param path Path to the JSON file.
         * @return     Parsed SystemPromptConfig.
         * @throws std::runtime_error if the file cannot be opened, is not valid
         *         JSON, or is missing the required system_prompt field.
         */
        static SystemPromptConfig load( const std::filesystem::path& path )
        {
            std::ifstream file( path );

            if ( !file )
            {
                throw std::runtime_error(
                    std::format( "SystemPromptLoader: cannot open '{}'", path.string() ) );
            }

            nlohmann::json j;

            try
            {
                file >> j;
            }
            catch ( const nlohmann::json::parse_error& e )
            {
                throw std::runtime_error(
                    std::format( "SystemPromptLoader: JSON parse error in '{}': {}",
                        path.string(), e.what() ) );
            }

            if ( !j.contains( "system_prompt" ) || !j[ "system_prompt" ].is_string() )
            {
                throw std::runtime_error(
                    std::format( "SystemPromptLoader: '{}' must contain a string 'system_prompt' field",
                        path.string() ) );
            }

            SystemPromptConfig config;
            config.system_prompt = j[ "system_prompt" ].get<std::string>();

            if ( j.contains( "tools" ) && j[ "tools" ].is_array() )
            {
                for ( const auto& tool_json : j[ "tools" ] )
                {
                    config.tools.push_back( parseTool( tool_json, path ) );
                }
            }

            return config;
        }

    private:

        static ToolDefinition parseTool(
            const nlohmann::json& j,
            const std::filesystem::path& path )
        {
            if ( !j.contains( "name" ) || !j[ "name" ].is_string() )
            {
                throw std::runtime_error(
                    std::format( "SystemPromptLoader: tool in '{}' missing required 'name' field",
                        path.string() ) );
            }

            if ( !j.contains( "description" ) || !j[ "description" ].is_string() )
            {
                throw std::runtime_error(
                    std::format( "SystemPromptLoader: tool '{}' in '{}' missing required 'description' field",
                        j[ "name" ].get<std::string>(), path.string() ) );
            }

            ToolDefinition tool;
            tool.name = j[ "name" ].get<std::string>();
            tool.description = j[ "description" ].get<std::string>();

            if ( j.contains( "parameters" ) && j[ "parameters" ].is_object() )
            {
                tool.parameters = parseParameterSchema( j[ "parameters" ] );
            }

            return tool;
        }

        static ToolParameterSchema parseParameterSchema( const nlohmann::json& j )
        {
            ToolParameterSchema schema;

            if ( j.contains( "type" ) && j[ "type" ].is_string() )
            {
                schema.type = j[ "type" ].get<std::string>();
            }

            if ( j.contains( "required" ) && j[ "required" ].is_array() )
            {
                for ( const auto& r : j[ "required" ] )
                {
                    if ( r.is_string() )
                    {
                        schema.required.push_back( r.get<std::string>() );
                    }
                }
            }

            if ( j.contains( "properties" ) && j[ "properties" ].is_object() )
            {
                for ( const auto& item : j[ "properties" ].items() )
                {
                    ToolParameterProperty p;

                    const auto& prop = item.value();
                    const auto& name = item.key();

                    if ( prop.contains( "type" ) && prop[ "type" ].is_string() )
                    {
                        p.type = prop[ "type" ].get<std::string>();
                    }

                    if ( prop.contains( "description" ) && prop[ "description" ].is_string() )
                    {
                        p.description = prop[ "description" ].get<std::string>();
                    }

                    schema.properties.emplace_back( name, std::move( p ) );
                }
            }

            return schema;
        }
    };
}