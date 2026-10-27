/**
 * @file Assistant.PromptFile.ixx
 * @brief An activity's prompt: the system prompt it begins with and the tools it offers, read from its file.
 */

export module Assistant.PromptFile;

import std;
import nlohmann.json;
import Assistant.PromptTool;

namespace Mila::Assistant
{
    /**
     * @brief One activity's prompt file, shipped beside the session in Prompts (MilaApp.md 3.1).
     *
     * @code
     * { "system_prompt": "...", "tools": [ { "name": "read_file", "description": "...", "summary": "Reading {path}",
     *   "parameters": { "type": "object", ... } } ] }
     * @endcode
     *
     * The tools are optional; each needs a name and a description, and parameters, when given, an object.
     */
    export struct PromptFile
    {
        std::string system_prompt;
        std::vector<PromptTool> tools;

        /// @throws std::runtime_error naming the file and what is wrong in it.
        static PromptFile load( const std::filesystem::path& path )
        {
            const std::u8string spelled = path.u8string();
            const std::string name( spelled.begin(), spelled.end() );

            std::ifstream file( path, std::ios::binary );

            if ( !file )
            {
                throw std::runtime_error( std::format( "the prompt {} could not be opened", name ) );
            }

            // Ordered, so each tool's schema reaches the model with its keys as the file writes them.
            nlohmann::ordered_json stored;

            try
            {
                stored = nlohmann::ordered_json::parse( file );
            }
            catch ( const nlohmann::ordered_json::parse_error& error )
            {
                throw std::runtime_error( std::format( "the prompt {} is not JSON: {}", name, error.what() ) );
            }

            if ( !stored.is_object() || !stored.contains( "system_prompt" ) || !stored[ "system_prompt" ].is_string() )
            {
                throw std::runtime_error( std::format( "the prompt {} has no system_prompt text", name ) );
            }

            PromptFile prompt;
            prompt.system_prompt = stored[ "system_prompt" ].get<std::string>();

            const nlohmann::ordered_json tools = stored.value( "tools", nlohmann::ordered_json::array() );

            if ( !tools.is_array() )
            {
                throw std::runtime_error( std::format( "the prompt {} has tools that are not a list", name ) );
            }

            for ( const nlohmann::ordered_json& tool : tools )
            {
                prompt.tools.push_back( readTool( tool, name ) );
            }

            return prompt;
        }

    private:

        static PromptTool readTool( const nlohmann::ordered_json& stored, const std::string& file )
        {
            if ( !stored.is_object() || !stored.contains( "name" ) || !stored[ "name" ].is_string() )
            {
                throw std::runtime_error( std::format( "a tool in the prompt {} has no name", file ) );
            }

            PromptTool tool;
            tool.name = stored[ "name" ].get<std::string>();

            if ( !stored.contains( "description" ) || !stored[ "description" ].is_string() )
            {
                throw std::runtime_error( std::format( "the tool {} in the prompt {} has no description", tool.name, file ) );
            }

            tool.description = stored[ "description" ].get<std::string>();

            const nlohmann::ordered_json parameters = stored.value( "parameters", nlohmann::ordered_json::object() );

            if ( !parameters.is_object() )
            {
                throw std::runtime_error( std::format( "the tool {} in the prompt {} has parameters that are not an object",
                    tool.name, file ) );
            }

            tool.parameters = parameters.dump();

            if ( stored.contains( "summary" ) && stored[ "summary" ].is_string() )
            {
                tool.summary = stored[ "summary" ].get<std::string>();
            }

            return tool;
        }
    };
}
