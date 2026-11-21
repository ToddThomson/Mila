/**
 * @file Assistant.ExposureRecord.ixx
 * @brief What a chat read from outside, as its stored conversation keeps it (AssistantInterface.md 5.6 and 8).
 */

export module Assistant.ExposureRecord;

import std;
import nlohmann.json;

import Mila;

namespace Mila::Assistant
{
    /// The exposure as a conversation's `exposure` array: each source's tool, arguments and time, the time ISO 8601
    /// UTC to the millisecond or null.
    export nlohmann::json exposureToJson( std::span<const Mila::Agents::ExposureSource> exposure );

    /**
     * @brief The exposure a stored conversation kept.
     *
     * An entry that cannot be read still counts, with what could be read of it: a chat whose record is damaged stays
     * exposed rather than coming back clear. Anything but an array or null counts as one source of unknown tool.
     */
    export std::vector<Mila::Agents::ExposureSource> exposureFromJson( const nlohmann::json& stored );
}
