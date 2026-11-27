/**
 * @file RoundRecord.cpp
 * @brief The bodies of RoundRecord's members; the interface is Assistant.RoundRecord.ixx.
 */

module Assistant.RoundRecord;

import std;
import nlohmann.json;
import Mila;

namespace Mila::Assistant
{
    namespace
    {
        using Kind = Mila::Agents::AgentEvent::Kind;
        using Mila::Agents::ToolDecision;

        std::string_view eventName( Kind kind ) noexcept
        {
            switch ( kind )
            {
                case Kind::RoundBegan: return "round_began";
                case Kind::Token: return "token";
                case Kind::ReasoningOpened: return "reasoning_opened";
                case Kind::Reasoning: return "reasoning";
                case Kind::ReasoningClosed: return "reasoning_closed";
                case Kind::Answer: return "answer";
                case Kind::ToolCallOpened: return "tool_call_opened";
                case Kind::ToolCalled: return "tool_called";
                case Kind::MalformedToolCall: return "malformed_tool_call";
                case Kind::RoundEnded: return "round_ended";
                case Kind::CompactionBegan: return "compaction_began";
                case Kind::Compacted: return "compacted";
                case Kind::CompactionFailed: return "compaction_failed";
            }

            return "unknown";
        }

        std::string_view decisionName( ToolDecision decision ) noexcept
        {
            switch ( decision )
            {
                case ToolDecision::AllowedByRule: return "allowed_by_rule";
                case ToolDecision::AllowedByPerson: return "allowed_by_person";
                case ToolDecision::AllowedByPersonForConversation: return "allowed_by_person_for_conversation";
                case ToolDecision::AllowedForConversation: return "allowed_for_conversation";
                case ToolDecision::DeniedByRule: return "denied_by_rule";
                case ToolDecision::DeclinedByPerson: return "declined_by_person";
                case ToolDecision::NoOneToAsk: return "no_one_to_ask";
                case ToolDecision::NotOffered: return "not_offered";
                case ToolDecision::InvalidArguments: return "invalid_arguments";
                case ToolDecision::NotRun: return "not_run";
                case ToolDecision::PlacedByProgram: return "placed_by_program";
            }

            return "unknown";
        }
    }

    std::filesystem::path RoundRecord::pathFor( const std::filesystem::path& conversations, std::string_view id )
    {
        return conversations / ( std::string( id ) + ".rounds.jsonl" );
    }

    RoundRecord::RoundRecord( std::filesystem::path file )
    {
        std::error_code ignored;
        std::filesystem::create_directories( file.parent_path(), ignored );

        out_.open( file, std::ios::binary | std::ios::app );
    }

    bool RoundRecord::good() const noexcept
    {
        return out_.good();
    }

    void RoundRecord::record( const Mila::Agents::AgentEvent& event, std::string_view turn, int round, std::string_view time )
    {
        if ( event.kind == Kind::RoundBegan )
        {
            round_began_ = std::chrono::steady_clock::now();
        }

        nlohmann::json line = { { "event", eventName( event.kind ) }, { "round", round },
            { "ms", std::chrono::duration_cast<std::chrono::milliseconds>( std::chrono::steady_clock::now() - round_began_ ).count() } };

        switch ( event.kind )
        {
            case Kind::RoundBegan:
            {
                line[ "turn" ] = turn;
                line[ "time" ] = time;
                line[ "prompt_tokens" ] = event.prompt_tokens ? nlohmann::json( *event.prompt_tokens ) : nlohmann::json( nullptr );
                line[ "budget" ] = event.budget ? nlohmann::json( *event.budget ) : nlohmann::json( nullptr );

                if ( event.sampling )
                {
                    line[ "sampling" ] = { { "temperature", event.sampling->temperature }, { "top_k", event.sampling->top_k },
                        { "top_p", event.sampling->top_p }, { "presence_penalty", event.sampling->presence_penalty } };
                }

                if ( !continues_.empty() && event.prompt.starts_with( continues_ ) )
                {
                    line[ "prompt_added" ] = event.prompt.substr( continues_.size() );
                } else {
                    line[ "prompt" ] = event.prompt;
                }

                continues_ = event.prompt;
                break;
            }

            case Kind::Token:
                line[ "id" ] = event.token ? nlohmann::json( *event.token ) : nlohmann::json( nullptr );
                line[ "text" ] = event.text;
                continues_ += event.text;
                break;

            case Kind::Reasoning:
            case Kind::Answer:
            case Kind::MalformedToolCall:
            case Kind::CompactionFailed:
                line[ "text" ] = event.text;
                break;

            case Kind::ToolCalled:
                line[ "name" ] = event.call != nullptr ? event.call->name : std::string{};
                line[ "arguments" ] = event.call != nullptr ? event.call->arguments : std::string{};
                line[ "decision" ] = event.decision ? nlohmann::json( decisionName( *event.decision ) ) : nlohmann::json( nullptr );
                line[ "is_error" ] = event.is_error;
                line[ "exposes" ] = event.exposes;
                line[ "duration_ms" ] = std::chrono::duration_cast<std::chrono::milliseconds>( event.duration ).count();
                line[ "returned_bytes" ] = event.returned_bytes ? nlohmann::json( *event.returned_bytes ) : nlohmann::json( nullptr );
                line[ "result" ] = event.text;
                break;

            case Kind::RoundEnded:
                line[ "status" ] = Mila::Dnn::to_string( event.status );
                line[ "tool_calls" ] = event.reply != nullptr ? event.reply->tool_calls.size() : 0u;
                line[ "malformed_tool_calls" ] = event.reply != nullptr ? event.reply->malformed_tool_calls.size() : 0u;
                break;

            case Kind::CompactionBegan:
            case Kind::Compacted:
                line[ "tokens_before" ] = event.tokens_before ? nlohmann::json( *event.tokens_before ) : nlohmann::json( nullptr );
                line[ "tokens_after" ] = event.tokens_after ? nlohmann::json( *event.tokens_after ) : nlohmann::json( nullptr );
                line[ "dropped_turns" ] = event.dropped_turns;
                line[ "text" ] = event.text;

                // What the next round reads is rendered anew.
                continues_.clear();
                break;

            case Kind::ReasoningOpened:
            case Kind::ReasoningClosed:
            case Kind::ToolCallOpened:
                break;
        }

        write( std::move( line ) );
    }

    void RoundRecord::write( nlohmann::json line )
    {
        // A token can end part way through a character, which is not text JSON can hold.
        out_ << line.dump( -1, ' ', false, nlohmann::json::error_handler_t::replace ) << '\n';
        out_.flush();
    }
}
