/**
 * @file Mila_py.ChatProtocol.cpp
 * @brief Implementation unit for Mila.Bindings.ChatProtocol: JSON to Conversation and back.
 */

module Mila.Bindings.ChatProtocol;

import std;
import Mila;
import nlohmann.json;

namespace Mila::Bindings
{
    namespace Conversation = Mila::Dnn::Conversation;

    struct ChatProtocolHandle::Implementation
    {
        std::unique_ptr<const Conversation::Protocol> protocol;
    };

    namespace
    {
        Conversation::Role roleFromName( const std::string& name )
        {
            if ( name == "system" )
            {
                return Conversation::Role::System;
            }

            if ( name == "user" )
            {
                return Conversation::Role::User;
            }

            if ( name == "assistant" )
            {
                return Conversation::Role::Assistant;
            }

            if ( name == "tool" )
            {
                return Conversation::Role::Tool;
            }

            throw std::invalid_argument( std::format(
                "a turn's role must be system, user, assistant or tool, not '{}'", name ) );
        }

        std::string stringField( const nlohmann::json& object, std::string_view key )
        {
            const auto field = object.find( std::string( key ) );

            if ( field == object.end() || field->is_null() )
            {
                return {};
            }

            if ( !field->is_string() )
            {
                throw std::invalid_argument( std::format( "a turn's '{}' must be text", key ) );
            }

            return field->get<std::string>();
        }

        std::vector<Conversation::Turn> turnsFromJson( const std::string& history_json )
        {
            const nlohmann::json history = nlohmann::json::parse( history_json, nullptr, false );

            if ( history.is_discarded() || !history.is_array() )
            {
                throw std::invalid_argument( "history must be a list of turns" );
            }

            std::vector<Conversation::Turn> turns;

            for ( const auto& item : history )
            {
                if ( !item.is_object() )
                {
                    throw std::invalid_argument( "each turn must be a mapping" );
                }

                Conversation::Turn turn{ roleFromName( stringField( item, "role" ) ), stringField( item, "content" ) };
                turn.tool_name = stringField( item, "tool_name" );
                turn.reasoning = stringField( item, "reasoning" );

                const auto calls = item.find( "tool_calls" );

                if ( calls != item.end() && !calls->is_null() )
                {
                    if ( !calls->is_array() )
                    {
                        throw std::invalid_argument( "a turn's tool_calls must be a list" );
                    }

                    for ( const auto& call : *calls )
                    {
                        turn.tool_calls.push_back( { stringField( call, "name" ), stringField( call, "arguments" ) } );
                    }
                }

                turns.push_back( std::move( turn ) );
            }

            return turns;
        }
    }

    ChatProtocolHandle::ChatProtocolHandle( std::shared_ptr<const Implementation> implementation )
        : implementation_( std::move( implementation ) )
    {
    }

    ChatProtocolHandle ChatProtocolHandle::forInstalledModel( const std::string& name )
    {
        const Mila::Distribution::ModelStore store;
        const auto record = store.readRecord( name );

        if ( !record.has_value() )
        {
            throw std::invalid_argument( std::format( "no model named '{}' is installed", name ) );
        }

        if ( !record->capabilities.instruct )
        {
            throw std::invalid_argument( std::format( "'{}' is a base model, which has no chat template", name ) );
        }

        return ChatProtocolHandle( std::make_shared<const Implementation>(
            Implementation{ Dnn::chatProtocolFor( record->architecture, record->architecture_version ) } ) );
    }

    ChatProtocolHandle ChatProtocolHandle::gemma()
    {
        return ChatProtocolHandle( std::make_shared<const Implementation>(
            Implementation{ std::make_unique<const Dnn::Gemma::GemmaChatProtocol>() } ) );
    }

    ChatProtocolHandle ChatProtocolHandle::qwen()
    {
        return ChatProtocolHandle( std::make_shared<const Implementation>(
            Implementation{ std::make_unique<const Dnn::Qwen::QwenChatProtocol>() } ) );
    }

    ChatProtocolHandle ChatProtocolHandle::llama31()
    {
        return ChatProtocolHandle( std::make_shared<const Implementation>(
            Implementation{ std::make_unique<const Dnn::Llama::LlamaChatProtocol>( Dnn::Llama::LlamaChatProtocol::llama31() ) } ) );
    }

    ChatProtocolHandle ChatProtocolHandle::llama32()
    {
        return ChatProtocolHandle( std::make_shared<const Implementation>(
            Implementation{ std::make_unique<const Dnn::Llama::LlamaChatProtocol>( Dnn::Llama::LlamaChatProtocol::llama32() ) } ) );
    }

    std::pair<std::string, bool> ChatProtocolHandle::formatPrompt( const std::string& history_json,
        const std::map<std::string, std::string>& template_variables, const std::string& tools_json ) const
    {
        const auto turns = turnsFromJson( history_json );
        auto prompt = implementation_->protocol->formatPrompt( turns, { template_variables, tools_json } );

        return { std::move( prompt.text ), prompt.reply_begins_in_reasoning };
    }

    std::string ChatProtocolHandle::parseReply( const std::string& text, bool reply_begins_in_reasoning ) const
    {
        const auto reply = implementation_->protocol->parseReply( text, { {}, reply_begins_in_reasoning } );

        nlohmann::json calls = nlohmann::json::array();

        for ( const auto& call : reply.tool_calls )
        {
            calls.push_back( { { "name", call.name }, { "arguments", call.arguments } } );
        }

        return nlohmann::json{
            { "reasoning", reply.reasoning },
            { "content", reply.content },
            { "tool_calls", calls },
            { "malformed_tool_calls", reply.malformed_tool_calls } }.dump();
    }

    std::string ChatProtocolHandle::formatToolResult( const std::string& name, const std::string& result ) const
    {
        return implementation_->protocol->formatToolResult( name, result );
    }

    std::string ChatProtocolHandle::markersJson() const
    {
        const auto& markers = implementation_->protocol->markers();

        return nlohmann::json{
            { "reasoning_open", markers.reasoning_open },
            { "reasoning_close", markers.reasoning_close },
            { "tool_call_open", markers.tool_call_open },
            { "tool_call_close", markers.tool_call_close },
            { "tool_response_open", markers.tool_response_open },
            { "stop_after", markers.stop_after } }.dump();
    }

    std::string ChatProtocolHandle::templateVariablesJson() const
    {
        nlohmann::json variables = nlohmann::json::array();

        for ( const auto& variable : implementation_->protocol->templateVariables() )
        {
            variables.push_back( { { "name", variable.name }, { "values", variable.values }, { "default", variable.default_value } } );
        }

        return variables.dump();
    }

    bool ChatProtocolHandle::rendersLaterSystemTurns() const
    {
        return implementation_->protocol->rendersLaterSystemTurns();
    }
}
