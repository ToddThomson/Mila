/**
 * @file Conversation.TemplateVariable.ixx
 * @brief A variable a checkpoint's chat template takes, with the values it accepts and its default.
 */

export module Dnn.Models.Conversation.TemplateVariable;

import std;

namespace Mila::Dnn::Conversation
{
    /**
     * @brief One variable of a chat template, as the template defines it.
     *
     * What HF's apply_chat_template(**kwargs) passes: Qwen 3.8's enable_thinking, reasoning_effort
     * and preserve_thinking, Gemma 4's enable_thinking and preserve_thinking. Thinking and effort
     * are the template's own controls, so they are named, valued and defaulted by the template and
     * never by Mila (ModelHandle.md 3.4).
     */
    export struct TemplateVariable
    {
        /// As the template spells it.
        std::string name;

        /// Every value the template accepts, as text; a boolean is "true" or "false". Empty when the
        /// template takes any text, as Llama's date_string does.
        std::vector<std::string> values;

        /// What the template uses when the variable is not set. Empty when the template computes its
        /// default when it renders -- Llama 3.2's date_string is today's date -- and the protocol says so.
        std::string default_value;
    };

    /**
     * @brief Refuse a setting the template does not accept.
     *
     * @throws std::invalid_argument naming the variable and what is accepted, for a name the
     *         template does not take or a value it does not accept. Never approximated: a value
     *         close to an accepted one is a prompt the checkpoint was not published with.
     */
    export inline void validateTemplateVariables(
        std::span<const TemplateVariable> variables, const std::map<std::string, std::string>& settings )
    {
        for ( const auto& [ name, value ] : settings )
        {
            const auto variable = std::ranges::find( variables, name, &TemplateVariable::name );

            if ( variable == variables.end() )
            {
                std::string accepted;

                for ( const auto& known : variables )
                {
                    accepted += accepted.empty() ? "" : ", ";
                    accepted += known.name;
                }

                throw std::invalid_argument( std::format(
                    "the chat template takes no variable '{}'; it takes: {}",
                    name, accepted.empty() ? "none" : accepted ) );
            }

            if ( !variable->values.empty() && std::ranges::find( variable->values, value ) == variable->values.end() )
            {
                std::string accepted;

                for ( const auto& known : variable->values )
                {
                    accepted += accepted.empty() ? "" : ", ";
                    accepted += known;
                }

                throw std::invalid_argument( std::format(
                    "the chat template's '{}' does not accept '{}'; it accepts: {}", name, value, accepted ) );
            }
        }
    }

    /// The value a variable takes for these settings: the one set, or the template's default.
    export inline std::string_view resolveTemplateVariable(
        std::span<const TemplateVariable> variables, const std::map<std::string, std::string>& settings,
        std::string_view name )
    {
        const auto setting = settings.find( std::string( name ) );

        if ( setting != settings.end() )
        {
            return setting->second;
        }

        const auto variable = std::ranges::find( variables, name, &TemplateVariable::name );

        if ( variable == variables.end() )
        {
            throw std::logic_error( std::format( "no template variable '{}'", name ) );
        }

        return variable->default_value;
    }
}
