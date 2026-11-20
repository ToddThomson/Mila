/**
 * @file ToolResult.ixx
 * @brief What one call of a tool returns: the text the model reads, and where it came from when that differs from
 *        call to call.
 */

export module Agents.ToolResult;

import std;
import Agents.ResultOrigin;

namespace Mila::Agents
{
    /**
     * @brief One call's result (ToolCalling.md 3.3, *What a result may cause*).
     *
     * A tool whose results always come from the same place returns its text alone, and its `Tool::result_origin`
     * says where. One whose results differ, such as a search whose matches may come from content that may not direct
     * the agent or from the program's own records, says so on each result.
     */
    export struct ToolResult
    {
        ToolResult( std::string result_text ) : text( std::move( result_text ) ) {}

        ToolResult( std::string result_text, ResultOrigin result_origin )
            : text( std::move( result_text ) ), origin( result_origin )
        {
        }

        ToolResult( const char* result_text ) : text( result_text ) {}

        std::string text;

        /// Where this result came from; the tool's own `result_origin` when not set.
        std::optional<ResultOrigin> origin;
    };
}
