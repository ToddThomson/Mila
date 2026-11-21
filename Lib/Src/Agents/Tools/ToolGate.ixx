/**
 * @file ToolGate.ixx
 * @brief The gate every tool call passes: its arguments checked, the rules and the person asked, and the tool run
 *        within its time.
 */

export module Agents.ToolGate;

import std;
import Agents.Tool;
import Agents.ToolQuestion;
import Agents.ToolAnswer;
import Agents.ToolRules;
import Agents.ToolPermission;
import Agents.ToolDecision;
import Agents.ToolOutcome;
import Agents.ArgumentSchema;
import Agents.AgentLimits;
import Agents.ToolGrant;
import Agents.ToolNeeds;
import Agents.ResultOrigin;
import Agents.ToolResult;
import Agents.ExposureSource;
import Agents.GrantRefusal;

namespace Mila::Agents
{
    /**
     * @brief An agent's tools and the one path by which a call of them runs (ToolCalling.md 3.2 and 3.3).
     *
     * A call of a tool not offered, or with arguments its parameters refuse, is never asked about, since nothing
     * will run. Otherwise the rules decide; where they ask, an answer for the conversation stands in for asking
     * again, and a deny rule is never overridden by one. A tool that uses the network is never allowed for the
     * conversation: what it sends is approved call by call, so that answer counts as once. A call allowed runs with
     * the gate's grant narrowed to the tool's needs (ToolCalling.md 3.4). Needs nothing of a model.
     *
     * Once a call of a tool whose result comes from outside has run, the conversation is exposed, and from then on a
     * call that writes files, uses the network or runs programs is asked every time: no allow rule or answer for the
     * conversation covers it (ToolCalling.md 3.3, *What a result may cause*).
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

        /// The most any call may reach from the next call on; each is given this narrowed to its tool's needs. With
        /// none set, a call reaches no folder and no network.
        void setGrant( ToolGrant grant )
        {
            grant_ = std::move( grant );
        }

        [[nodiscard]] const ToolGrant& grant() const noexcept
        {
            return grant_;
        }

        /// Forget every tool the person allowed for the conversation, and what it read from outside.
        void newConversation() noexcept
        {
            allowed_for_conversation_.clear();
            exposure_.clear();
        }

        /// What the conversation has read from outside, oldest first; empty while it is not exposed.
        [[nodiscard]] std::span<const ExposureSource> exposure() const noexcept
        {
            return exposure_;
        }

        [[nodiscard]] bool isExposed() const noexcept
        {
            return !exposure_.empty();
        }

        /// Continue with what a stored conversation had read from outside, as `exposure()` gave it.
        void restoreExposure( std::vector<ExposureSource> exposure )
        {
            exposure_ = std::move( exposure );
        }

        /// The tools the person allowed for the conversation, by name.
        [[nodiscard]] const std::set<std::string, std::less<>>& allowedForConversation() const noexcept
        {
            return allowed_for_conversation_;
        }

        /// Ask about this tool again from the next call on.
        void withdrawAllowance( std::string_view name )
        {
            const auto allowed = allowed_for_conversation_.find( name );

            if ( allowed != allowed_for_conversation_.end() )
            {
                allowed_for_conversation_.erase( allowed );
            }
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
            const ResultOrigin origin = run( *tool, arguments, grant_.narrowedTo( tool->needs ), halted,
                tool->time ? tool->time : limits.tool_time, outcome );

            // A stopped or timed-out call's result is the gate's own words, so only a call that returned exposes.
            if ( outcome.ran && !outcome.stopped && !outcome.timed_out && origin == ResultOrigin::Outside )
            {
                outcome.exposes = true;
                exposure_.push_back( ExposureSource{ tool->name, std::string( arguments ), std::chrono::system_clock::now() } );
            }

            return outcome;
        }

        /**
         * @brief Run a call the program placed itself, as `answer` runs one allowed: with the grant narrowed to the
         *        tool's needs, and exposing the conversation when its result comes from outside.
         *
         * No rule or person decides it, since the program, not the model, chose to make it.
         *
         * @throws std::invalid_argument for a tool not offered or arguments its parameters refuse, which is the
         *         program's error rather than the model's.
         */
        [[nodiscard]] ToolOutcome runPlaced( std::string_view name, std::string_view arguments, std::stop_token halted,
            const AgentLimits& limits = {} )
        {
            checkPlaced( name, arguments );

            ToolOutcome outcome;
            outcome.decision = ToolDecision::PlacedByProgram;

            if ( halted.stop_requested() )
            {
                outcome.stopped = true;

                return outcome;
            }

            const Tool& tool = *find( name );

            outcome.result_tokens = tool.result_tokens ? tool.result_tokens : limits.result_tokens;
            const ResultOrigin origin = run( tool, arguments, grant_.narrowedTo( tool.needs ), halted,
                tool.time ? tool.time : limits.tool_time, outcome );

            if ( outcome.ran && !outcome.stopped && !outcome.timed_out && origin == ResultOrigin::Outside )
            {
                outcome.exposes = true;
                exposure_.push_back( ExposureSource{ tool.name, std::string( arguments ), std::chrono::system_clock::now() } );
            }

            return outcome;
        }

        /**
         * @brief Refuse a call the program would place that could not run.
         *
         * @throws std::invalid_argument for a tool not offered or arguments its parameters refuse.
         */
        void checkPlaced( std::string_view name, std::string_view arguments ) const
        {
            const auto tool = std::ranges::find( tools_, name, &Tool::name );

            if ( tool == tools_.end() )
            {
                throw std::invalid_argument( std::format( "a placed call names '{}', which is not offered", name ) );
            }

            if ( const auto why = schemas_[ static_cast<std::size_t>( tool - tools_.begin() ) ].check( arguments ) )
            {
                throw std::invalid_argument( std::format(
                    "a placed call of '{}' has arguments its parameters refuse: {}", name, *why ) );
            }
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

            // What an exposed conversation asks for may come from what it read, so nothing earlier covers a call that
            // acts outward or changes anything.
            const bool asked_every_time = isExposed() && actsOutward( tool->needs );

            switch ( rules_.decide( name, arguments ) )
            {
                case ToolPermission::Deny:
                    return ToolDecision::DeniedByRule;

                case ToolPermission::Allow:
                    if ( !asked_every_time )
                    {
                        return ToolDecision::AllowedByRule;
                    }

                    break;

                case ToolPermission::Ask:
                    break;
            }

            if ( asked_every_time )
            {
                withdrawAllowance( name );
            }
            else if ( allowed_for_conversation_.contains( name ) )
            {
                return ToolDecision::AllowedForConversation;
            }

            if ( !ask )
            {
                return ToolDecision::NoOneToAsk;
            }

            const ToolGrant grant = grant_.narrowedTo( tool->needs );

            switch ( ask( ToolQuestion{ tool->name, tool->source, tool->summary, arguments, tool->needs, grant, exposure_ } ) )
            {
                case ToolAnswer::Once:
                    return ToolDecision::AllowedByPerson;

                case ToolAnswer::ForConversation:
                    if ( tool->needs.uses_network || asked_every_time )
                    {
                        return ToolDecision::AllowedByPerson;
                    }

                    allowed_for_conversation_.insert( tool->name );

                    return ToolDecision::AllowedByPersonForConversation;

                case ToolAnswer::Deny:
                    break;
            }

            return ToolDecision::DeclinedByPerson;
        }

        static bool actsOutward( const ToolNeeds& needs ) noexcept
        {
            return needs.writes_files || needs.uses_network || needs.runs_programs;
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

        /// @return Where the result came from: the result's own word, else the tool's.
        static ResultOrigin run( const Tool& tool, std::string_view arguments, const ToolGrant& grant, std::stop_token halted,
            std::optional<std::chrono::milliseconds> time, ToolOutcome& outcome )
        {
            ResultOrigin origin = tool.result_origin;

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
                ToolResult returned = tool.run( arguments, grant, call_stop.get_token() );
                origin = returned.origin.value_or( tool.result_origin );
                outcome.result = std::move( returned.text );
                outcome.returned_bytes = outcome.result.size();
                outcome.is_error = false;
            }
            catch ( const GrantRefusal& refused )
            {
                // Decided from the grant before anything was opened or sent, so the call read nothing from outside.
                outcome.result = std::format( "Error: tool '{}' failed: {}", tool.name, refused.what() );
                origin = ResultOrigin::Application;
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

            return origin;
        }

        std::vector<Tool> tools_;
        std::vector<ArgumentSchema> schemas_;
        ToolRules rules_;
        ToolGrant grant_;
        std::set<std::string, std::less<>> allowed_for_conversation_;
        std::vector<ExposureSource> exposure_;
    };
}
