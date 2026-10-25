/**
 * @file AI.Tools.BrokerClient.ixx
 * @brief An application's connection to the tool broker: list its tools, call one, and ask the person when the
 *        broker says to.
 */

export module AI.Tools.BrokerClient;

import std;
import AI.Tools.ApprovalRequest;
import AI.Tools.BrokerConnection;
import AI.Tools.BrokerTool;
import AI.Tools.ToolResult;

namespace Mila::AI
{
    /**
     * @brief Speaks MCP 2026-07-28 to the tool broker for one agent.
     *
     * Every request carries the protocol's `_meta` and the agent's name, since MCP keeps no session. When the
     * broker's rules say ask, it answers with an input-required result; the client turns that into an
     * `ApprovalRequest` for the application's `Approver`, and retries the call with the answer (ToolBroker.md
     * section 5). The broker decides; the application only presents.
     *
     * One call at a time: the connection is used request then reply from one thread.
     */
    export class BrokerClient
    {
    public:
        /** How many times one call may be asked about before the client gives up on it. */
        static constexpr int kApprovalRounds = 3;

        /**
         * @param pipe_name The broker's pipe, `\\.\pipe\...`.
         * @param agent Which of the application's registered agents the calls are for.
         * @param application The application's name, sent as MCP's client information.
         * @throws std::system_error if the broker's pipe cannot be opened.
         */
        BrokerClient( const std::wstring& pipe_name, std::string agent, std::string application );

        /**
         * @brief Every tool the broker offers this agent, in the broker's order.
         * @throws std::runtime_error if the broker refuses or closes.
         */
        std::vector<BrokerTool> listTools();

        /**
         * @brief Runs `tool` with `arguments` (a JSON object as text; empty means none).
         *
         * @param conversation Names the conversation the call belongs to, so the person can allow a tool for the
         *        rest of it; empty offers no such choice.
         * @param approve Asks the person; an empty one has nobody to ask, and the broker refuses what it would ask.
         * @return The tool's result, or a failed result saying why the call did not run.
         * @throws std::runtime_error if the broker closes the connection.
         */
        ToolResult call( std::string_view tool, std::string_view arguments, std::string_view conversation,
            const Approver& approve );

    private:
        std::string exchange( const std::string& request, std::int64_t id );

        BrokerConnection connection_;
        std::string agent_;
        std::string application_;
        std::int64_t next_id_ = 1;
        std::map<std::string, std::string, std::less<>> descriptions_;
    };
}
