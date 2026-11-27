/**
 * @file Assistant.RoundRecord.ixx
 * @brief Everything each round of a conversation read and wrote, kept beside it when the person turns it on
 *        (ToolCalling.md 4.1).
 */

export module Assistant.RoundRecord;

import std;
import nlohmann.json;
import Mila;

namespace Mila::Assistant
{
    /**
     * @brief One conversation's round record: a JSON line per agent event, appended in the order the agent reported
     *        them, each written out as it comes so a session that ends mid-round leaves the round as far as it got.
     *
     * Every line has `event`, the event's kind in snake case, `round`, the turn's round, and `ms`, the milliseconds since
     * the round began. A round's prompt is written whole when it does not extend the last round's prompt and reply
     * (`prompt`), and otherwise only what it adds (`prompt_added`), so a long conversation does not write its history
     * every round. A token's text that is not text alone -- part of a character split across tokens -- is written with
     * the replacement character; its `id` is exact.
     */
    export class RoundRecord
    {
    public:

        /// The file a conversation's record is kept in, beside the conversation's own.
        [[nodiscard]] static std::filesystem::path pathFor( const std::filesystem::path& conversations, std::string_view id );

        /// Appends to the file, made with its folder when missing; whether it could be opened is good.
        explicit RoundRecord( std::filesystem::path file );

        [[nodiscard]] bool good() const noexcept;

        /**
         * @brief Write one event of a turn's response.
         *
         * @param turn  The turn's id, written on the line that begins each round.
         * @param round The round the event belongs to, from 1.
         * @param time  When a round began, ISO 8601 in UTC; read for RoundBegan only.
         */
        void record( const Mila::Agents::AgentEvent& event, std::string_view turn, int round, std::string_view time );

    private:

        void write( nlohmann::json line );

        std::ofstream out_;

        // The last round's prompt and the reply as it generated it, which a continued round's prompt begins with.
        std::string continues_;

        std::chrono::steady_clock::time_point round_began_{};
    };
}
