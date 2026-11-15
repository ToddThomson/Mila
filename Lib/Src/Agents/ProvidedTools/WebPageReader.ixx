/**
 * @file WebPageReader.ixx
 * @brief The web tool both applications offer: a page fetched over HTTP or HTTPS and given to the model as readable
 *        text (ToolCalling.md section 5).
 */

export module Agents.WebPageReader;

import std;
import Networking.HttpTransport;
import Agents.Tool;

namespace Mila::Agents
{
    /**
     * @brief Reads one web page for the model, a part at a time.
     *
     * The page is fetched through the call's grant, which connects only to public addresses, each redirect's
     * connection checked as the first was (ToolCalling.md 3.4). At most five redirects are followed, only to http
     * and https addresses, and the page is cut at a size no page of text needs. A failure throws, and the model
     * reads its message.
     */
    export class WebPageReader
    {
    public:

        /// The tool, with what the model is told, what the person is shown and what it needs.
        [[nodiscard]] static Tool tool();

        /// `url`, and optionally `start`, the character the part read begins at; fetched through this transport.
        [[nodiscard]] static std::string read( const std::string& arguments_text, const Mila::Networking::IHttpTransport& transport );
    };
}
