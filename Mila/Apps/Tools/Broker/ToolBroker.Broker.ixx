/**
 * @file ToolBroker.Broker.ixx
 * @brief The broker's MCP front: lists every worker's tools under one name each, and runs a call only as the
 *        person's rules and approvals allow.
 */

export module ToolBroker.Broker;

import std;
import ToolBroker.ClientMessage;
import ToolBroker.Json;
import ToolBroker.LineChannel;
import ToolBroker.Policy;
import ToolBroker.ProtocolError;
import ToolBroker.Relay;
import ToolBroker.ToolRule;
import ToolBroker.Worker;

namespace Mila::ToolBroker
{
    /** The MCP revision the broker speaks on both hops. */
    export inline constexpr std::string_view kProtocolVersion = "2026-07-28";

    /** The `_meta` key naming which of the application's registered agents a request is for (ToolBroker.md 2). */
    export inline constexpr std::string_view kAgentKey = "me.toddt.mila/agent";

    /**
     * The `_meta` key naming the conversation a call belongs to. MCP has no session, so an approval "for this
     * conversation" can only be keyed on an identifier every request carries.
     */
    export inline constexpr std::string_view kConversationKey = "me.toddt.mila/conversation";

    /** The key of the broker's one input request, the approval, in an input-required result. */
    export inline constexpr std::string_view kApprovalInput = "approval";

    /**
     * @brief The broker's answer to every client request: `tools/list` and `tools/call`, over the workers added to it.
     *
     * A tool is shown as `<server>.<tool>`, since MCP tool names are unique only within one server. Each server's tools
     * are listed once, when it is added, and that list is the one the broker offers and accepts; a tool a worker adds
     * later is never offered.
     *
     * A call the rules leave at ask is answered with MCP's input-required result carrying one form elicitation, the
     * approval, and a request state the retry must echo (*Multi Round-Trip Requests*). The state is a random token
     * the broker keeps, bound to the agent, the conversation, the tool and its exact arguments, used once and
     * expiring after `kApprovalLifetime`; anything else presented as one is refused. A client that declares no form
     * elicitation has nobody to ask, so its ask is a refusal (ToolBroker.md section 4). A refusal or a declined
     * approval reaches the model as a tool result saying so (section 5).
     *
     * Add every server before the first request; after that, requests may come from several threads at once, and
     * each worker takes one call at a time.
     */
    export class Broker
    {
    public:
        static constexpr std::chrono::minutes kApprovalLifetime{ 10 };

        explicit Broker( Policy policy )
            : policy_( std::move( policy ) )
        {
        }

        /**
         * @brief Adds a worker as the server `name`, listing its tools.
         * @throws std::invalid_argument if the name is empty, taken, or holds anything but ASCII letters, digits,
         *         `_` and `-`, since a dot would make a tool's full name ambiguous.
         * @throws std::runtime_error if the worker does not list its tools.
         */
        void addServer( std::string name, Worker worker )
        {
            const bool plain = !name.empty() && std::ranges::all_of( name, []( char character )
                {
                    return std::isalnum( static_cast<unsigned char>( character ) ) || character == '_' || character == '-';
                } );

            if ( !plain || servers_.contains( name ) )
            {
                throw std::invalid_argument( std::format( "'{}' cannot name a tool server", name ) );
            }

            auto server = std::make_unique<Server>( std::move( worker ) );
            nlohmann::json cursor = nullptr;

            for ( int page = 0; page < kListPageLimit; ++page )
            {
                const nlohmann::json id = std::format( "broker-list-{}", page );
                nlohmann::json params = { { "_meta", workerMeta() } };

                if ( !cursor.is_null() )
                {
                    params[ "cursor" ] = cursor;
                }

                const nlohmann::json request = { { "jsonrpc", "2.0" }, { "id", id }, { "method", "tools/list" },
                    { "params", params } };
                const auto answer = nlohmann::json::parse(
                    forwardRequest( server->worker, request.dump(), id, []( std::string_view ) {} ) );

                if ( !answer.contains( "result" ) || !answer[ "result" ].value( "tools", nlohmann::json() ).is_array() )
                {
                    throw std::runtime_error( std::format( "The tool server '{}' did not list its tools: {}", name, answer.dump() ) );
                }

                for ( auto tool : answer[ "result" ][ "tools" ] )
                {
                    if ( !tool.is_object() || !tool.value( "name", nlohmann::json() ).is_string() )
                    {
                        continue;
                    }

                    const auto full_name = name + "." + tool[ "name" ].get<std::string>();
                    tool[ "name" ] = full_name;
                    server->names.insert( full_name );
                    server->tools.push_back( std::move( tool ) );
                }

                cursor = answer[ "result" ].value( "nextCursor", nlohmann::json() );

                if ( !cursor.is_string() )
                {
                    break;
                }
            }

            servers_.emplace( std::move( name ), std::move( server ) );
        }

        /**
         * @brief The response line to one client message, or nothing for a notification.
         *
         * Notifications a worker sends while it runs a call go to `pass_on`.
         */
        std::optional<std::string> handle( std::string_view line, const std::function<void( std::string_view )>& pass_on )
        {
            auto message = parseClientMessage( line );

            if ( !message )
            {
                return errorResponse( message.error() );
            }

            if ( message->isNotification() )
            {
                return std::nullopt;
            }

            if ( message->method == "tools/list" )
            {
                return listTools( *message );
            }

            if ( message->method == "tools/call" )
            {
                return callTool( *message, pass_on );
            }

            return errorResponse( ProtocolError{ ProtocolError::kMethodNotFound,
                std::format( "Method not found: {}", message->method ), message->id } );
        }

        /** Answers one client until it closes. */
        void serve( LineChannel& client )
        {
            const auto pass_on = [&client]( std::string_view notification ) { client.send( notification ); };

            while ( const auto line = client.receive() )
            {
                if ( const auto response = handle( *line, pass_on ) )
                {
                    client.send( *response );
                }
            }
        }

    private:
        static constexpr int kListPageLimit = 100;

        struct Server
        {
            explicit Server( Worker running )
                : worker( std::move( running ) )
            {
            }

            Worker worker;
            std::mutex mutex;
            nlohmann::json tools = nlohmann::json::array();
            std::set<std::string> names;
        };

        struct PendingApproval
        {
            std::string agent;
            std::string conversation;
            std::string tool;
            nlohmann::json arguments;
            bool offers_conversation = false;
            std::chrono::steady_clock::time_point expires;
        };

        enum class Decision
        {
            Missing,
            Declined,
            AllowOnce,
            AllowForConversation
        };

        static nlohmann::json workerMeta()
        {
            return { { "io.modelcontextprotocol/protocolVersion", kProtocolVersion },
                { "io.modelcontextprotocol/clientCapabilities", nlohmann::json::object() } };
        }

        static std::string metaString( const nlohmann::json& meta, std::string_view key )
        {
            const auto found = meta.find( key );

            return found != meta.end() && found->is_string() ? found->get<std::string>() : std::string();
        }

        static std::string complete( const nlohmann::json& id, nlohmann::json result )
        {
            result[ "resultType" ] = "complete";

            return nlohmann::json{ { "jsonrpc", "2.0" }, { "id", id }, { "result", std::move( result ) } }.dump();
        }

        /** A call that did not run, told to the model as a tool result it can act on. */
        static std::string notRun( const nlohmann::json& id, std::string text )
        {
            return complete( id, { { "content", nlohmann::json::array( { { { "type", "text" }, { "text", std::move( text ) } } } ) },
                { "isError", true } } );
        }

        /** Form elicitation, declared as `elicitation: {}` (form only) or with a `form` member. */
        static bool canAsk( const nlohmann::json& meta )
        {
            const auto& capabilities = meta[ "io.modelcontextprotocol/clientCapabilities" ];
            const auto elicitation = capabilities.find( "elicitation" );

            if ( elicitation == capabilities.end() || !elicitation->is_object() )
            {
                return false;
            }

            return elicitation->empty() || elicitation->contains( "form" );
        }

        static std::string newToken()
        {
            // MSVC's random_device draws from the operating system's cryptographic generator.
            std::random_device source;
            std::string token;

            for ( int index = 0; index < 4; ++index )
            {
                token += std::format( "{:08x}", source() );
            }

            return token;
        }

        static Decision readDecision( const nlohmann::json& params, bool offers_conversation )
        {
            const auto responses = params.find( "inputResponses" );

            if ( responses == params.end() || !responses->is_object() || !responses->contains( kApprovalInput ) )
            {
                return Decision::Missing;
            }

            const auto& response = ( *responses )[ kApprovalInput ];
            const auto action = response.is_object() ? response.value( "action", "" ) : std::string();

            if ( action == "decline" || action == "cancel" )
            {
                return Decision::Declined;
            }

            if ( action != "accept" || !response.contains( "content" ) || !response[ "content" ].is_object() )
            {
                return Decision::Missing;
            }

            const auto choice = response[ "content" ].value( "decision", "" );

            if ( choice == "allow_once" )
            {
                return Decision::AllowOnce;
            }

            if ( choice == "allow_conversation" && offers_conversation )
            {
                return Decision::AllowForConversation;
            }

            return choice == "deny" ? Decision::Declined : Decision::Missing;
        }

        std::string listTools( const ClientMessage& message ) const
        {
            nlohmann::json tools = nlohmann::json::array();

            for ( const auto& [name, server] : servers_ )
            {
                for ( const auto& tool : server->tools )
                {
                    tools.push_back( tool );
                }
            }

            return complete( message.id, { { "tools", std::move( tools ) } } );
        }

        std::string askFor( const nlohmann::json& id, PendingApproval approval )
        {
            nlohmann::json choices = nlohmann::json::array( { { { "const", "allow_once" }, { "title", "Allow once" } } } );

            if ( approval.offers_conversation )
            {
                choices.push_back( { { "const", "allow_conversation" }, { "title", "Allow for this conversation" } } );
            }

            choices.push_back( { { "const", "deny" }, { "title", "Deny" } } );

            const auto who = approval.agent.empty() ? std::string( "An agent" ) : approval.agent;
            const auto text = std::format( "{} asks to run {} with these arguments: {}", who, approval.tool,
                approval.arguments.dump() );

            const nlohmann::json request = {
                { "method", "elicitation/create" },
                { "params", {
                    { "mode", "form" },
                    { "message", text },
                    { "requestedSchema", {
                        { "type", "object" },
                        { "properties", { { "decision", { { "type", "string" }, { "title", "Decision" }, { "oneOf", choices } } } } },
                        { "required", nlohmann::json::array( { "decision" } ) } } } } }
            };

            const auto token = newToken();
            approval.expires = std::chrono::steady_clock::now() + kApprovalLifetime;

            {
                std::scoped_lock lock( approvals_mutex_ );
                const auto now = std::chrono::steady_clock::now();
                std::erase_if( pending_, [now]( const auto& entry ) { return entry.second.expires < now; } );
                pending_.emplace( token, std::move( approval ) );
            }

            const nlohmann::json result = {
                { "resultType", "input_required" },
                { "inputRequests", { { kApprovalInput, request } } },
                { "requestState", token }
            };

            return nlohmann::json{ { "jsonrpc", "2.0" }, { "id", id }, { "result", result } }.dump();
        }

        /** The approval a retry presents, removed whether or not it matches, so no state is used twice. */
        std::optional<PendingApproval> takeApproval( const nlohmann::json& state, const PendingApproval& call )
        {
            if ( !state.is_string() )
            {
                return std::nullopt;
            }

            std::scoped_lock lock( approvals_mutex_ );
            const auto found = pending_.find( state.get<std::string>() );

            if ( found == pending_.end() )
            {
                return std::nullopt;
            }

            auto approval = std::move( found->second );
            pending_.erase( found );

            const bool matches = approval.agent == call.agent && approval.conversation == call.conversation
                && approval.tool == call.tool && approval.arguments == call.arguments;

            if ( !matches || approval.expires < std::chrono::steady_clock::now() )
            {
                return std::nullopt;
            }

            return approval;
        }

        std::string callTool( const ClientMessage& message, const std::function<void( std::string_view )>& pass_on )
        {
            const auto& params = message.params;
            const auto& id = message.id;
            const auto name_field = params.find( "name" );

            if ( name_field == params.end() || !name_field->is_string() )
            {
                return errorResponse( ProtocolError{ ProtocolError::kInvalidParams, "A tool call must name its tool", id } );
            }

            const auto name = name_field->get<std::string>();
            const auto arguments = params.value( "arguments", nlohmann::json::object() );

            if ( !arguments.is_object() )
            {
                return errorResponse( ProtocolError{ ProtocolError::kInvalidParams, "A tool's arguments must be an object", id } );
            }

            const auto dot = name.find( '.' );
            const auto server = dot == std::string::npos ? servers_.end() : servers_.find( name.substr( 0, dot ) );

            if ( server == servers_.end() || !server->second->names.contains( name ) )
            {
                return errorResponse( ProtocolError{ ProtocolError::kInvalidParams, std::format( "Unknown tool: {}", name ), id } );
            }

            const auto& meta = params[ "_meta" ];
            PendingApproval call{ metaString( meta, kAgentKey ), metaString( meta, kConversationKey ), name, arguments };
            call.offers_conversation = !call.conversation.empty();

            auto rule = policy_.ruleFor( call.agent, name );

            if ( rule == ToolRule::Ask && call.offers_conversation )
            {
                std::scoped_lock lock( approvals_mutex_ );

                if ( conversation_grants_.contains( { call.agent, call.conversation, name } ) )
                {
                    rule = ToolRule::Allow;
                }
            }

            if ( rule == ToolRule::Deny )
            {
                return notRun( id, std::format( "The person's rules do not allow {}, so it did not run.", name ) );
            }

            if ( rule == ToolRule::Ask )
            {
                if ( !params.contains( "requestState" ) )
                {
                    if ( !canAsk( meta ) )
                    {
                        return notRun( id, std::format( "Nobody was present to approve {}, so it did not run.", name ) );
                    }

                    return askFor( id, std::move( call ) );
                }

                const auto approval = takeApproval( params[ "requestState" ], call );

                if ( !approval )
                {
                    return errorResponse( ProtocolError{ ProtocolError::kInvalidParams,
                        "This call's approval is unknown, expired, used or for another call", id } );
                }

                switch ( readDecision( params, approval->offers_conversation ) )
                {
                    case Decision::Missing:
                        return askFor( id, *approval );

                    case Decision::Declined:
                        return notRun( id, std::format( "The person declined to run {}.", name ) );

                    case Decision::AllowForConversation:
                    {
                        std::scoped_lock lock( approvals_mutex_ );
                        conversation_grants_.insert( { call.agent, call.conversation, name } );
                        break;
                    }

                    case Decision::AllowOnce:
                        break;
                }
            }

            const nlohmann::json request = {
                { "jsonrpc", "2.0" },
                { "id", id },
                { "method", "tools/call" },
                { "params", { { "name", name.substr( dot + 1 ) }, { "arguments", arguments }, { "_meta", workerMeta() } } }
            };

            std::scoped_lock lock( server->second->mutex );

            return forwardRequest( server->second->worker, request.dump(), id, pass_on );
        }

        Policy policy_;
        std::map<std::string, std::unique_ptr<Server>> servers_;
        std::mutex approvals_mutex_;
        std::map<std::string, PendingApproval> pending_;
        std::set<std::tuple<std::string, std::string, std::string>> conversation_grants_;
    };
}
