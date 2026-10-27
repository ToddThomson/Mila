/**
 * @file Session.ixx
 * @brief Everything a program gives an agent, and the conversation: what stays when the model changes.
 *
 * See Specifications/Direction.md section 4.3.
 */

export module AI.Session;

import std;
import Dnn.Models.Conversation.Protocol;
import AI.Tool;
import AI.ToolRules;
import AI.ToolGrant;
import AI.ToolGate;
import AI.AgentLimits;
import AI.SamplingOverride;

namespace Mila::AI
{
    using namespace Mila::Dnn;

    /**
     * @brief The program's part of an agent: its prompt, its tools and their gate, how the person is asked, its limits,
     *        the template variables and sampling it asks for, and the conversation.
     *
     * Needs no model, so a program builds one before it has chosen a model, and an agent gives it back when its model
     * goes (`release`). A copy is a fork: the same settings, and the conversation so far, each changed apart from the
     * other. What it asks of the model -- template variables and sampling -- is checked by the agent it is given to,
     * since only a model's template knows what it accepts.
     */
    export class Session
    {
    public:

        using Asker = ToolGate::Asker;

        /**
         * @brief Offer the model a tool.
         *
         * @throws std::invalid_argument as ToolGate::add.
         */
        Session& withTool( Tool tool ) &
        {
            gate_.add( std::move( tool ) );

            return *this;
        }

        Session&& withTool( Tool tool ) &&
        {
            gate_.add( std::move( tool ) );

            return std::move( *this );
        }

        /**
         * @brief Offer the model no tool from the next response on, so one conversation can be given a different set
         *        of tools than the last.
         *
         * Tools allowed for the conversation are forgotten with them; the rules stay.
         */
        Session& withoutTools() &
        {
            gate_.clear();

            return *this;
        }

        Session&& withoutTools() &&
        {
            gate_.clear();

            return std::move( *this );
        }

        /// Decide the model's calls by these rules from the next call on. With none, every call is asked about.
        Session& withRules( ToolRules rules ) &
        {
            gate_.setRules( std::move( rules ) );

            return *this;
        }

        Session&& withRules( ToolRules rules ) &&
        {
            gate_.setRules( std::move( rules ) );

            return std::move( *this );
        }

        /**
         * @brief The most the model's calls may reach from the next call on: folders and the network. Each call
         *        allowed is given this narrowed to its tool's needs; with none, a call reaches nothing.
         */
        Session& withGrant( ToolGrant grant ) &
        {
            gate_.setGrant( std::move( grant ) );

            return *this;
        }

        Session&& withGrant( ToolGrant grant ) &&
        {
            gate_.setGrant( std::move( grant ) );

            return std::move( *this );
        }

        /**
         * @brief How the person is asked whether a call may run. With none, no one can be asked, so every call the
         *        rules would ask about is declined.
         */
        Session& onAsk( Asker ask ) &
        {
            ask_ = std::move( ask );

            return *this;
        }

        Session&& onAsk( Asker ask ) &&
        {
            ask_ = std::move( ask );

            return std::move( *this );
        }

        /// The system message every prompt begins with; empty for none.
        Session& withSystemPrompt( std::string text ) &
        {
            system_prompt_ = std::move( text );

            return *this;
        }

        Session&& withSystemPrompt( std::string text ) &&
        {
            system_prompt_ = std::move( text );

            return std::move( *this );
        }

        /**
         * @brief Limit how far each response may run. None is set until this is called.
         *
         * @throws std::invalid_argument for a limit set to zero or less.
         */
        Session& withLimits( AgentLimits limits ) &
        {
            setLimits( limits );

            return *this;
        }

        Session&& withLimits( AgentLimits limits ) &&
        {
            setLimits( limits );

            return std::move( *this );
        }

        /**
         * @brief Ask for one of the chat template's own variables, such as its thinking switch.
         *
         * Not checked here: the agent this session is given to checks it against its model's template, and drops one
         * the template does not accept when it is created (`droppedTemplateVariables`). Set through an agent instead,
         * it is checked at once.
         */
        Session& withTemplateVariable( std::string name, std::string value ) &
        {
            template_variables_[ std::move( name ) ] = std::move( value );

            return *this;
        }

        Session&& withTemplateVariable( std::string name, std::string value ) &&
        {
            template_variables_[ std::move( name ) ] = std::move( value );

            return std::move( *this );
        }

        /// Ask for every one of the template's variables at once; a variable not named takes its default.
        Session& withTemplateVariables( std::map<std::string, std::string> variables ) &
        {
            template_variables_ = std::move( variables );

            return *this;
        }

        Session&& withTemplateVariables( std::map<std::string, std::string> variables ) &&
        {
            template_variables_ = std::move( variables );

            return std::move( *this );
        }

        /// Sample with these settings over what the model's publisher recommends; empty settings are the publisher's.
        Session& withSampling( SamplingOverride sampling ) &
        {
            sampling_ = sampling;

            return *this;
        }

        Session&& withSampling( SamplingOverride sampling ) &&
        {
            sampling_ = sampling;

            return std::move( *this );
        }

        /// Forget the conversation, and every tool allowed for it. Tools and settings stay.
        void newConversation()
        {
            conversation_.clear();
            gate_.newConversation();
        }

        /// Continue a conversation saved earlier, as `conversation()` gave it; no tool is allowed for it yet.
        void openConversation( std::vector<Conversation::Turn> turns )
        {
            conversation_ = std::move( turns );
            gate_.newConversation();
        }

        /// Every message, reply and result so far, without the system prompt.
        [[nodiscard]] std::span<const Conversation::Turn> conversation() const noexcept
        {
            return conversation_;
        }

        /// The conversation to change in place: what an agent adds to as it responds, and a program may edit between
        /// responses.
        [[nodiscard]] std::vector<Conversation::Turn>& turns() noexcept
        {
            return conversation_;
        }

        [[nodiscard]] std::span<const Tool> tools() const noexcept
        {
            return gate_.tools();
        }

        /// The tools and the one path a call of them runs by: its rules, its grant and the tools allowed for the
        /// conversation.
        [[nodiscard]] ToolGate& gate() noexcept
        {
            return gate_;
        }

        [[nodiscard]] const ToolGate& gate() const noexcept
        {
            return gate_;
        }

        [[nodiscard]] const Asker& asker() const noexcept
        {
            return ask_;
        }

        [[nodiscard]] const std::string& systemPrompt() const noexcept
        {
            return system_prompt_;
        }

        [[nodiscard]] const AgentLimits& limits() const noexcept
        {
            return limits_;
        }

        [[nodiscard]] const std::map<std::string, std::string>& templateVariables() const noexcept
        {
            return template_variables_;
        }

        /// The variables the session asks for, for an agent to drop those its template does not accept.
        [[nodiscard]] std::map<std::string, std::string>& templateVariables() noexcept
        {
            return template_variables_;
        }

        [[nodiscard]] const SamplingOverride& sampling() const noexcept
        {
            return sampling_;
        }

    private:

        void setLimits( const AgentLimits& limits )
        {
            if ( limits.rounds && *limits.rounds <= 0 )
            {
                throw std::invalid_argument( std::format( "a response needs at least one round, not {}", *limits.rounds ) );
            }

            if ( limits.time && limits.time->count() <= 0 )
            {
                throw std::invalid_argument( std::format( "a response needs some time, not {}", *limits.time ) );
            }

            if ( limits.reply_tokens && *limits.reply_tokens <= 0 )
            {
                throw std::invalid_argument(
                    std::format( "a response needs at least one reply token, not {}", *limits.reply_tokens ) );
            }

            if ( limits.tool_time && limits.tool_time->count() <= 0 )
            {
                throw std::invalid_argument( std::format( "a tool call needs some time to run, not {}", *limits.tool_time ) );
            }

            if ( limits.result_tokens && *limits.result_tokens <= 0 )
            {
                throw std::invalid_argument(
                    std::format( "a result needs at least one token, not {}", *limits.result_tokens ) );
            }

            limits_ = limits;
        }

        ToolGate gate_;
        Asker ask_;
        std::string system_prompt_;
        AgentLimits limits_;
        std::map<std::string, std::string> template_variables_;
        SamplingOverride sampling_;
        std::vector<Conversation::Turn> conversation_;
    };
}
