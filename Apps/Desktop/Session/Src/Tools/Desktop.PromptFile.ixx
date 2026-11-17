/**
 * @file Desktop.PromptFile.ixx
 * @brief The chat's prompt: the system prompt's passages and the tools it offers, read from its file.
 */

export module Desktop.PromptFile;

import std;
import nlohmann.json;
import Desktop.PromptTool;
import Desktop.PromptPart;

namespace Mila::Desktop
{
    /**
     * @brief The prompt file shipped beside the session in Prompts (AssistantInterface.md 5.5).
     *
     * @code
     * { "system_prompt": [ "You are Mila...", { "when": "read_web_page", "unless": "web_search", "text": "..." } ],
     *   "tools": [ "read_file", { "name": "propose_plan", "description": "...", "summary": "Proposing a plan",
     *   "parameters": { "type": "object", ... } } ] }
     * @endcode
     *
     * The system prompt is a text, or a list of passages: a text for every chat, or an object kept by the tools offered
     * (`PromptPart`). The tools are optional. A tool Mila::Agents provides is named by a string; one of the session's own
     * needs a name and a description, and parameters, when given, an object.
     */
    export struct PromptFile
    {
        std::vector<PromptPart> system_prompt;
        std::vector<PromptTool> tools;

        /// The system prompt for a chat offered these tools: the passages kept, joined by a space.
        [[nodiscard]] std::string systemPromptFor( const std::set<std::string, std::less<>>& offered ) const
        {
            std::string joined;

            for ( const PromptPart& part : system_prompt )
            {
                if ( !part.keptFor( offered ) )
                {
                    continue;
                }

                if ( !joined.empty() )
                {
                    joined += ' ';
                }

                joined += part.text;
            }

            return joined;
        }

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

            if ( !stored.is_object() || !stored.contains( "system_prompt" ) )
            {
                throw std::runtime_error( std::format( "the prompt {} has no system_prompt text", name ) );
            }

            PromptFile prompt;
            prompt.system_prompt = readParts( stored[ "system_prompt" ], name );

            if ( prompt.system_prompt.empty() )
            {
                throw std::runtime_error( std::format( "the prompt {} has no system_prompt text", name ) );
            }

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

        static std::vector<PromptPart> readParts( const nlohmann::ordered_json& stored, const std::string& file )
        {
            if ( stored.is_string() )
            {
                return { PromptPart{ stored.get<std::string>(), {}, {} } };
            }

            if ( !stored.is_array() )
            {
                throw std::runtime_error( std::format( "the prompt {} has a system_prompt that is neither text nor a list", file ) );
            }

            std::vector<PromptPart> parts;

            for ( const nlohmann::ordered_json& part : stored )
            {
                if ( part.is_string() )
                {
                    parts.push_back( { part.get<std::string>(), {}, {} } );

                    continue;
                }

                if ( !part.is_object() || !part.contains( "text" ) || !part[ "text" ].is_string() )
                {
                    throw std::runtime_error( std::format( "a passage of the prompt {} has no text", file ) );
                }

                const auto named = [&]( const char* key )
                    {
                        if ( !part.contains( key ) )
                        {
                            return std::string{};
                        }

                        if ( !part[ key ].is_string() )
                        {
                            throw std::runtime_error( std::format( "a passage of the prompt {} has a {} that is not a tool's name",
                                file, key ) );
                        }

                        return part[ key ].get<std::string>();
                    };

                parts.push_back( { part[ "text" ].get<std::string>(), named( "when" ), named( "unless" ) } );
            }

            return parts;
        }

        static PromptTool readTool( const nlohmann::ordered_json& stored, const std::string& file )
        {
            if ( stored.is_string() )
            {
                PromptTool named;
                named.name = stored.get<std::string>();

                return named;
            }

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
