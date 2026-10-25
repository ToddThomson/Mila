/**
 * @file Chat.ParsedResponse.ixx
 * @brief A turn's reasoning and answer, as it is shown whole.
 */

export module Chat.ParsedResponse;

import std;

namespace Mila::ChatApp
{
    /**
     * @brief Every round's reasoning and answer text, joined in the order the rounds ran.
     *
     * thinking is empty when the model did not reason. The answer is what the person reads.
     */
    export struct ParsedResponse
    {
        std::string thinking;
        std::string answer;
    };
}
