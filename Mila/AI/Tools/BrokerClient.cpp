/**
 * @file BrokerClient.cpp
 * @brief The broker client's messages: MCP's `_meta`, tool listing, calls and the approval retry.
 */

module AI.Tools.BrokerClient;

import std;
import AI.Json;
import AI.Tools.ApprovalChoice;
import AI.Tools.ApprovalRequest;
import AI.Tools.BrokerConnection;
import AI.Tools.BrokerTool;
import AI.Tools.ToolResult;

namespace Mila::AI
{
    namespace
    {
        constexpr std::string_view kProtocolVersion = "2026-07-28";

        // The broker's own keys (ToolBroker.Broker.ixx), under the vendor prefix MCP asks for.
        constexpr std::string_view kAgentKey = "me.toddt.mila/agent";
        constexpr std::string_view kConversationKey = "me.toddt.mila/conversation";

        nlohmann::json requestMeta( const std::string& agent, const std::string& application,
            std::string_view conversation, bool can_ask )
        {
            nlohmann::json capabilities = nlohmann::json::object();

            if ( can_ask )
            {
                capabilities[ "elicitation" ] = { { "form", nlohmann::json::object() } };
            }

            nlohmann::json meta = {
                { "io.modelcontextprotocol/protocolVersion", kProtocolVersion },
                { "io.modelcontextprotocol/clientInfo", { { "name", application }, { "version", MILA_AI_VERSION } } },
                { "io.modelcontextprotocol/clientCapabilities", capabilities },
                { kAgentKey, agent }
            };

            if ( !conversation.empty() )
            {
                meta[ kConversationKey ] = conversation;
            }

            return meta;
        }

        nlohmann::json requestLine( std::int64_t id, std::string_view method, nlohmann::json params )
        {
            return { { "jsonrpc", "2.0" }, { "id", id }, { "method", method }, { "params", std::move( params ) } };
        }

        std::string errorText( const nlohmann::json& response, std::string_view fallback )
        {
            const auto& error = response[ "error" ];

            return error.is_object() && error.contains( "message" ) && error[ "message" ].is_string()
                ? error[ "message" ].get<std::string>() : std::string( fallback );
        }

        ToolResult toResult( const nlohmann::json& result )
        {
            std::string text;

            if ( result.contains( "content" ) && result[ "content" ].is_array() )
            {
                for ( const auto& item : result[ "content" ] )
                {
                    const auto type = item.is_object() ? item.value( "type", "" ) : std::string();
                    const auto piece = type == "text" ? item.value( "text", "" ) : std::format( "[{} content omitted]", type );
                    text += text.empty() ? piece : "\n" + piece;
                }
            }

            if ( text.empty() && result.contains( "structuredContent" ) )
            {
                text = result[ "structuredContent" ].dump();
            }

            const bool failed = result.contains( "isError" ) && result[ "isError" ].is_boolean() && result[ "isError" ].get<bool>();

            return ToolResult{ std::move( text ), failed };
        }

        /** The choices the broker's approval form offers, or nothing if the input request is not one. */
        std::optional<std::vector<std::string>> approvalChoices( const nlohmann::json& input )
        {
            if ( !input.is_object() || input.value( "method", "" ) != "elicitation/create" || !input.contains( "params" ) )
            {
                return std::nullopt;
            }

            const auto& params = input[ "params" ];

            if ( !params.is_object() || params.value( "mode", "form" ) != "form" )
            {
                return std::nullopt;
            }

            const auto decision = nlohmann::json::json_pointer( "/requestedSchema/properties/decision/oneOf" );

            if ( !params.contains( decision ) || !params[ decision ].is_array() )
            {
                return std::nullopt;
            }

            std::vector<std::string> choices;

            for ( const auto& choice : params[ decision ] )
            {
                if ( choice.is_object() && choice.contains( "const" ) && choice[ "const" ].is_string() )
                {
                    choices.push_back( choice[ "const" ].get<std::string>() );
                }
            }

            if ( std::ranges::find( choices, "allow_once" ) == choices.end() )
            {
                return std::nullopt;
            }

            return choices;
        }
    }

    BrokerClient::BrokerClient( const std::wstring& pipe_name, std::string agent, std::string application )
        : connection_( pipe_name ), agent_( std::move( agent ) ), application_( std::move( application ) )
    {
    }

    std::string BrokerClient::exchange( const std::string& request, std::int64_t id )
    {
        connection_.send( request );

        // Notifications the broker passes on while a call runs carry no id and are not the answer.
        while ( const auto line = connection_.receive() )
        {
            const auto message = nlohmann::json::parse( *line, nullptr, false );

            if ( message.is_object() && message.contains( "id" ) && message[ "id" ] == id )
            {
                return *line;
            }
        }

        throw std::runtime_error( "The tool broker closed the connection" );
    }

    std::vector<BrokerTool> BrokerClient::listTools()
    {
        std::vector<BrokerTool> tools;
        nlohmann::json cursor = nullptr;

        do
        {
            nlohmann::json params = { { "_meta", requestMeta( agent_, application_, {}, false ) } };

            if ( cursor.is_string() )
            {
                params[ "cursor" ] = cursor;
            }

            const auto id = next_id_++;
            const auto response = nlohmann::json::parse( exchange( requestLine( id, "tools/list", params ).dump(), id ) );

            if ( !response.contains( "result" ) )
            {
                throw std::runtime_error( errorText( response, "The tool broker did not list its tools" ) );
            }

            const auto& result = response[ "result" ];

            if ( result.contains( "tools" ) && result[ "tools" ].is_array() )
            {
                for ( const auto& tool : result[ "tools" ] )
                {
                    if ( !tool.is_object() || !tool.contains( "name" ) || !tool[ "name" ].is_string() )
                    {
                        continue;
                    }

                    BrokerTool entry{ tool[ "name" ].get<std::string>(), tool.value( "description", "" ),
                        tool.contains( "inputSchema" ) ? tool[ "inputSchema" ].dump() : std::string( "{}" ) };
                    descriptions_.insert_or_assign( entry.name, entry.description );
                    tools.push_back( std::move( entry ) );
                }
            }

            cursor = result.value( "nextCursor", nlohmann::json() );
        } while ( cursor.is_string() );

        return tools;
    }

    ToolResult BrokerClient::call( std::string_view tool, std::string_view arguments, std::string_view conversation,
        const Approver& approve )
    {
        auto parsed = arguments.empty() ? nlohmann::json::object() : nlohmann::json::parse( arguments, nullptr, false );

        if ( parsed.is_discarded() || !parsed.is_object() )
        {
            return ToolResult{ std::format( "The arguments to {} are not a JSON object.", tool ), true };
        }

        const auto shown_arguments = parsed.dump();
        const auto description = descriptions_.find( tool );

        nlohmann::json params = {
            { "name", tool },
            { "arguments", std::move( parsed ) },
            { "_meta", requestMeta( agent_, application_, conversation, static_cast<bool>( approve ) ) }
        };

        for ( int round = 0; round < kApprovalRounds; ++round )
        {
            const auto id = next_id_++;
            const auto response = nlohmann::json::parse( exchange( requestLine( id, "tools/call", params ).dump(), id ) );

            if ( !response.contains( "result" ) || !response[ "result" ].is_object() )
            {
                return ToolResult{ errorText( response, "The tool broker refused the call." ), true };
            }

            const auto& result = response[ "result" ];
            const auto type = result.value( "resultType", "complete" );

            if ( type == "complete" )
            {
                return toResult( result );
            }

            if ( type != "input_required" )
            {
                return ToolResult{ std::format( "The tool broker answered {} with a result this client does not know.", tool ), true };
            }

            nlohmann::json responses = nlohmann::json::object();

            if ( result.contains( "inputRequests" ) && result[ "inputRequests" ].is_object() )
            {
                const auto& requests = result[ "inputRequests" ];

                for ( auto entry = requests.begin(); entry != requests.end(); ++entry )
                {
                    const auto key = entry.key();
                    const auto& input = entry.value();
                    const auto choices = approvalChoices( input );

                    if ( !choices || !approve )
                    {
                        responses[ key ] = { { "action", "cancel" } };
                        continue;
                    }

                    const bool offers_conversation = std::ranges::find( *choices, "allow_conversation" ) != choices->end();
                    const ApprovalRequest request{ std::string( tool ),
                        description == descriptions_.end() ? std::string() : description->second, shown_arguments,
                        input[ "params" ].value( "message", "" ), offers_conversation };

                    const auto choice = approve( request );

                    if ( choice == ApprovalChoice::Deny )
                    {
                        responses[ key ] = { { "action", "decline" } };
                    } else {
                        const bool whole_conversation = choice == ApprovalChoice::AllowForConversation && offers_conversation;
                        responses[ key ] = { { "action", "accept" },
                            { "content", { { "decision", whole_conversation ? "allow_conversation" : "allow_once" } } } };
                    }
                }
            }

            params.erase( "inputResponses" );
            params.erase( "requestState" );

            if ( !responses.empty() )
            {
                params[ "inputResponses" ] = std::move( responses );
            }

            if ( result.contains( "requestState" ) )
            {
                params[ "requestState" ] = result[ "requestState" ];
            }
        }

        return ToolResult{ std::format( "The tool broker asked about {} more than {} times, so it did not run.", tool,
            kApprovalRounds ), true };
    }
}
