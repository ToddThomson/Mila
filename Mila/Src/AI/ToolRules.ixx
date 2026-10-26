/**
 * @file ToolRules.ixx
 * @brief The person's rules for an agent's tool calls, and the permission they give one call.
 */

export module AI.ToolRules;

import std;
import nlohmann.json;
import AI.ToolPermission;
import AI.ToolRule;
import AI.ArgumentCondition;

namespace Mila::AI
{
    /**
     * @brief The rules a call is decided by (ToolCalling.md 3.3). The application supplies them and keeps them.
     *
     * Of the rules that apply to a call, a deny wins over an ask and an ask over an allow, so adding a rule never
     * lets through a call another rule stops. A call no rule applies to is asked about.
     */
    export class ToolRules
    {
    public:

        /**
         * @throws std::invalid_argument for a rule that names no tool, or a condition that names no argument as a
         *         JSON Pointer or gives no values.
         */
        ToolRules& add( ToolRule rule ) &
        {
            addRule( std::move( rule ) );

            return *this;
        }

        ToolRules&& add( ToolRule rule ) &&
        {
            addRule( std::move( rule ) );

            return std::move( *this );
        }

        /// The permission the rules give a call of this tool with these arguments, a JSON object.
        [[nodiscard]] ToolPermission decide( std::string_view tool, std::string_view arguments ) const
        {
            const auto parsed = nlohmann::ordered_json::parse( arguments, nullptr, false );

            bool asks = false;
            bool allows = false;

            for ( const auto& rule : rules_ )
            {
                if ( !matches( rule.tool, tool ) || !std::ranges::all_of( rule.when,
                    [&parsed]( const ArgumentCondition& condition ) { return holds( condition, parsed ); } ) )
                {
                    continue;
                }

                switch ( rule.permission )
                {
                    case ToolPermission::Deny:
                        return ToolPermission::Deny;

                    case ToolPermission::Ask:
                        asks = true;
                        break;

                    case ToolPermission::Allow:
                        allows = true;
                        break;
                }
            }

            return asks || !allows ? ToolPermission::Ask : ToolPermission::Allow;
        }

        [[nodiscard]] std::span<const ToolRule> rules() const noexcept
        {
            return rules_;
        }

    private:

        void addRule( ToolRule rule )
        {
            if ( rule.tool.empty() )
            {
                throw std::invalid_argument( "a rule names the tool it applies to" );
            }

            for ( const auto& condition : rule.when )
            {
                if ( !condition.argument.starts_with( '/' ) )
                {
                    throw std::invalid_argument( std::format(
                        "a condition names its argument as a JSON Pointer, such as \"/path\", not \"{}\"", condition.argument ) );
                }

                try
                {
                    [[maybe_unused]] const nlohmann::ordered_json::json_pointer pointer( condition.argument );
                }
                catch ( const std::exception& )
                {
                    throw std::invalid_argument( std::format( "\"{}\" is not a JSON Pointer", condition.argument ) );
                }

                if ( condition.values.empty() )
                {
                    throw std::invalid_argument( std::format(
                        "the condition on \"{}\" gives no values to compare with", condition.argument ) );
                }
            }

            rules_.push_back( std::move( rule ) );
        }

        static bool holds( const ArgumentCondition& condition, const nlohmann::ordered_json& arguments )
        {
            if ( arguments.is_discarded() )
            {
                return false;
            }

            std::string text;

            try
            {
                const nlohmann::ordered_json::json_pointer pointer( condition.argument );

                if ( !arguments.contains( pointer ) )
                {
                    return false;
                }

                const auto& value = arguments.at( pointer );
                text = value.is_string() ? value.get<std::string>() : value.dump();
            }
            catch ( const std::exception& )
            {
                // A pointer through a value that is not an object or array names nothing.
                return false;
            }

            return std::ranges::any_of( condition.values, [&condition, &text]( const std::string& value )
                {
                    return condition.test == ArgumentCondition::Test::IsOneOf ? text == value : matches( value, text );
                } );
        }

        /// Whether text matches a pattern in which * is any run of characters, and every other character itself.
        static bool matches( std::string_view pattern, std::string_view text ) noexcept
        {
            std::size_t at = 0;
            std::size_t star = std::string_view::npos;
            std::size_t resume = 0;

            for ( std::size_t position = 0; position < text.size(); )
            {
                if ( at < pattern.size() && pattern[ at ] == '*' )
                {
                    star = at++;
                    resume = position;
                }
                else if ( at < pattern.size() && pattern[ at ] == text[ position ] )
                {
                    ++at;
                    ++position;
                }
                else if ( star != std::string_view::npos )
                {
                    at = star + 1;
                    position = ++resume;
                }
                else
                {
                    return false;
                }
            }

            while ( at < pattern.size() && pattern[ at ] == '*' )
            {
                ++at;
            }

            return at == pattern.size();
        }

        std::vector<ToolRule> rules_;
    };
}
