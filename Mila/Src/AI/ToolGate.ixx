/**
 * @file ToolGate.ixx
 * @brief The gate every tool call passes: its arguments checked, the rules and the person asked, and the tool run
 *        within its time.
 */

export module AI.ToolGate;

import std;
import AI.Tool;
import AI.ToolQuestion;
import AI.ToolAnswer;
import AI.ToolRules;
import AI.ToolPermission;
import AI.ToolDecision;
import AI.ToolOutcome;
import AI.ArgumentSchema;
import AI.AgentLimits;

namespace Mila::AI
{
    /**
     * @brief An agent's tools and the one path by which a call of them runs (ToolCalling.md 3.2 and 3.3).
     *
     * A call of a tool not offered, or with arguments its parameters refuse, is never asked about, since nothing
     * will run. Otherwise the rules decide; where they ask, an answer for the conversation stands in for asking
     * again, and a deny rule is never overridden by one. Needs nothing of a model.
     */
    export class ToolGate
    {
    public:

        using Asker = std::function<ToolAnswer( const ToolQuestion& )>;

        /**
         * @brief Offer a tool.
         *
         * @throws std::invalid_argument for a tool with no name, a name of other than letters, digits, _ and -, a
         *         name already registered, no function, parameters ArgumentSchema refuses, or a time or result limit
         *         of zero or less.
         */
        void add( Tool tool )
        {
            if ( tool.name.empty() )
            {
                throw std::invalid_argument( "a tool needs a name" );
            }

            if ( !std::ranges::all_of( tool.name, []( char character )
                {
                    return std::isalnum( static_cast<unsigned char>( character ) ) != 0 || character == '_' || character == '-';
                } ) )
            {
                throw std::invalid_argument( std::format(
                    "tool '{}': a name is letters, digits, _ and -, which every template accepts", tool.name ) );
            }

            if ( find( tool.name ) != nullptr )
            {
                throw std::invalid_argument( std::format( "a tool named '{}' is already registered", tool.name ) );
            }

            if ( !tool.run )
            {
                throw std::invalid_argument( std::format( "tool '{}' has no function to run it", tool.name ) );
            }

            if ( tool.time && tool.time->count() <= 0 )
            {
                throw std::invalid_argument( std::format( "tool '{}' needs some time to run, not {}", tool.name, *tool.time ) );
            }

            if ( tool.result_tokens && *tool.result_tokens <= 0 )
            {
                throw std::invalid_argument( std::format(
                    "tool '{}' needs at least one token of result, not {}", tool.name, *tool.result_tokens ) );
            }

            std::optional<ArgumentSchema> schema;

            try
            {
                schema.emplace( tool.parameters );
            }
            catch ( const std::invalid_argument& error )
            {
                throw std::invalid_argument( std::format( "tool '{}': {}", tool.name, error.what() ) );
            }

            tools_.push_back( std::move( tool ) );
            schemas_.push_back( std::move( *schema ) );
        }

        [[nodiscard]] const Tool* find( std::string_view name ) const noexcept
        {
            const auto tool = std::ranges::find( tools_, name, &Tool::name );

            return tool == tools_.end() ? nullptr : &*tool;
        }

        [[nodiscard]] std::span<const Tool> tools() const noexcept
        {
            return tools_;
        }

        /// Decide calls by these rules from the next call on; with none set, every call is asked about.
        void setRules( ToolRules rules )
        {
            rules_ = std::move( rules );
        }

        [[nodiscard]] const ToolRules& rules() const noexcept
        {
            return rules_;
        }

        /// Forget every tool the person allowed for the conversation.
        void newConversation() noexcept
        {
            allowed_for_conversation_.clear();
        }

        /// Offer no tool from the next call on; a tool once offered can then be added again. The rules stay.
        void clear() noexcept
        {
            tools_.clear();
            schemas_.clear();
            allowed_for_conversation_.clear();
        }

        /**
         * @brief Decide one call, ask the person where the rules say to, and run it when allowed.
         *
         * @param name      The tool the model called.
         * @param arguments The arguments as the model wrote them.
         * @param ask       How the person is asked; empty when no one can be.
         * @param halted    The response's stop: once requested, a call not yet run does not run and a call running
         *                  is asked to stop.
         * @param limits    The call's time and result limits where the tool sets none.
         */
        [[nodiscard]] ToolOutcome answer( std::string_view name, std::string_view arguments, const Asker& ask,
            std::stop_token halted, const AgentLimits& limits = {} )
        {
            ToolOutcome outcome;

            if ( halted.stop_requested() )
            {
                outcome.stopped = true;

                return outcome;
            }

            outcome.decision = decide( name, arguments, ask );

            const Tool* tool = find( name );

            // The person may have taken long enough to answer for the reply to be stopped meanwhile.
            if ( halted.stop_requested() )
            {
                outcome.stopped = true;

                return outcome;
            }

            if ( !isAllowed( outcome.decision ) )
            {
                outcome.result = refusal( outcome.decision, name, arguments );

                return outcome;
            }

            outcome.result_tokens = tool->result_tokens ? tool->result_tokens : limits.result_tokens;
            run( *tool, arguments, halted, tool->time ? tool->time : limits.tool_time, outcome );

            return outcome;
        }

    private:

        ToolDecision decide( std::string_view name, std::string_view arguments, const Asker& ask )
        {
            const auto tool = std::ranges::find( tools_, name, &Tool::name );

            if ( tool == tools_.end() )
            {
                return ToolDecision::NotOffered;
            }

            if ( schemas_[ static_cast<std::size_t>( tool - tools_.begin() ) ].check( arguments ) )
            {
                return ToolDecision::InvalidArguments;
            }

            switch ( rules_.decide( name, arguments ) )
            {
                case ToolPermission::Deny:
                    return ToolDecision::DeniedByRule;

                case ToolPermission::Allow:
                    return ToolDecision::AllowedByRule;

                case ToolPermission::Ask:
                    break;
            }

            if ( allowed_for_conversation_.contains( name ) )
            {
                return ToolDecision::AllowedForConversation;
            }

            if ( !ask )
            {
                return ToolDecision::NoOneToAsk;
            }

            switch ( ask( ToolQuestion{ tool->name, tool->source, tool->summary, arguments, tool->needs } ) )
            {
                case ToolAnswer::Once:
                    return ToolDecision::AllowedByPerson;

                case ToolAnswer::ForConversation:
                    allowed_for_conversation_.insert( tool->name );

                    return ToolDecision::AllowedByPersonForConversation;

                case ToolAnswer::Deny:
                    break;
            }

            return ToolDecision::DeclinedByPerson;
        }

        /// What the model reads of a call that did not run, so it can recover.
        std::string refusal( ToolDecision decision, std::string_view name, std::string_view arguments ) const
        {
            switch ( decision )
            {
                case ToolDecision::NotOffered:
                {
                    std::string offered;

                    for ( const auto& tool : tools_ )
                    {
                        offered += std::format( "{}{}", offered.empty() ? "" : ", ", tool.name );
                    }

                    return std::format( "Error: there is no tool named '{}'. The tools are: {}.", name, offered );
                }

                case ToolDecision::InvalidArguments:
                {
                    const auto tool = std::ranges::find( tools_, name, &Tool::name );
                    const auto why = schemas_[ static_cast<std::size_t>( tool - tools_.begin() ) ].check( arguments );

                    return std::format( "Error: the arguments do not match the parameters of '{}': {}.", name, *why );
                }

                case ToolDecision::DeniedByRule:
                    return "The user's rules do not allow this tool call.";

                default:
                    // Declined, or no one to ask: either way the person did not allow it.
                    return "The user declined to run this tool call.";
            }
        }

        static void run( const Tool& tool, std::string_view arguments, std::stop_token halted,
            std::optional<std::chrono::milliseconds> time, ToolOutcome& outcome )
        {
            std::stop_source call_stop;
            const std::stop_callback forward_stop( halted, [&call_stop] { call_stop.request_stop(); } );
            std::atomic<bool> timed_out{ false };
            std::jthread timer;

            if ( time )
            {
                timer = std::jthread( [&call_stop, &timed_out, limit = *time]( std::stop_token finished )
                    {
                        std::mutex mutex;
                        std::condition_variable_any never;
                        std::unique_lock<std::mutex> lock( mutex );

                        // Only the call finishing ends the wait early.
                        never.wait_for( lock, finished, limit, [] { return false; } );

                        if ( !finished.stop_requested() )
                        {
                            timed_out = true;
                            call_stop.request_stop();
                        }
                    } );
            }

            outcome.ran = true;

            const auto started = std::chrono::steady_clock::now();

            try
            {
                outcome.result = tool.run( arguments, call_stop.get_token() );
                outcome.returned_bytes = outcome.result.size();
                outcome.is_error = false;
            }
            catch ( const std::exception& error )
            {
                outcome.result = std::format( "Error: tool '{}' failed: {}", tool.name, error.what() );
            }
            catch ( ... )
            {
                outcome.result = std::format( "Error: tool '{}' failed", tool.name );
            }

            outcome.duration = std::chrono::steady_clock::now() - started;

            if ( timer.joinable() )
            {
                timer.request_stop();
                timer.join();
            }

            if ( halted.stop_requested() )
            {
                outcome.result.clear();
                outcome.stopped = true;
            }
            else if ( timed_out )
            {
                // What it returned once asked to stop may be partial, so the model is told only that it ran out.
                outcome.result = std::format(
                    "Error: tool '{}' ran out of its {} ms, so it may not have finished.", tool.name, time->count() );
                outcome.is_error = true;
                outcome.timed_out = true;
            }
        }

        std::vector<Tool> tools_;
        std::vector<ArgumentSchema> schemas_;
        ToolRules rules_;
        std::set<std::string, std::less<>> allowed_for_conversation_;
    };
}
