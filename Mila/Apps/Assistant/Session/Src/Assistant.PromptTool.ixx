/**
 * @file Assistant.PromptTool.ixx
 * @brief One tool as an activity's prompt file describes it to the model, and the line the page shows for a call.
 */

export module Assistant.PromptTool;

import std;
import nlohmann.json;

namespace Mila::Assistant
{
    /**
     * @brief A tool an activity's prompt offers, as written in its file; the session gives it the code that runs it.
     */
    export struct PromptTool
    {
        std::string name;

        /// What the model is told the tool does.
        std::string description;

        /// The arguments' JSON Schema, as the file writes it.
        std::string parameters;

        /// The line shown for a call, each {argument} replaced by the call's value: "Reading {path}".
        std::string summary;
    };

    /**
     * @brief The line the page shows for a call: the tool's summary with the call's arguments in it.
     *
     * "Calling <name>" for a tool with no summary; the summary as written when the arguments are not JSON.
     */
    export inline std::string callSummary( std::span<const PromptTool> tools, std::string_view name, std::string_view arguments )
    {
        const auto tool = std::ranges::find( tools, name, &PromptTool::name );

        if ( tool == tools.end() || tool->summary.empty() )
        {
            return std::format( "Calling {}", name );
        }

        nlohmann::json parsed;

        try
        {
            parsed = nlohmann::json::parse( arguments );
        }
        catch ( const nlohmann::json::parse_error& )
        {
            return tool->summary;
        }

        std::string line = tool->summary;

        if ( !parsed.is_object() )
        {
            return line;
        }

        for ( auto argument = parsed.begin(); argument != parsed.end(); ++argument )
        {
            const std::string placeholder = "{" + argument.key() + "}";
            const std::string value = argument.value().is_string()
                ? argument.value().get<std::string>()
                : argument.value().dump();

            for ( std::size_t at = line.find( placeholder ); at != std::string::npos; at = line.find( placeholder, at + value.size() ) )
            {
                line.replace( at, placeholder.size(), value );
            }
        }

        return line;
    }
}
