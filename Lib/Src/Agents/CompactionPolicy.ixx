/**
 * @file CompactionPolicy.ixx
 * @brief What a program asks of compaction: when it runs, what it keeps word for word, and how the summary is written.
 *
 * See Specifications/Agents/Compaction.md.
 */

export module Agents.CompactionPolicy;

import std;
import Dnn.TensorTypes;
import Dnn.Models.Conversation.Turn;

namespace Mila::Agents
{
    /**
     * @brief What a summary is asked to keep, where a policy names no instruction of its own.
     *
     * A summary cut by its budget fails, so the length asked for leaves it room: Qwen wrote about two tokens to a word,
     * and two fifths of the budget in words leaves a model a fifth to run over.
     *
     * @param summary_tokens The most the summary may generate.
     */
    export [[nodiscard]] inline std::string defaultCompactionInstruction( Mila::Dnn::dim_t summary_tokens )
    {
        return std::format(
            "Write a summary of this conversation so far, for whoever continues the work with no other record of it. Begin "
            "with every instruction and requirement the user gave that still holds, including any in an earlier summary, "
            "quoted word for word as instructions to follow. Do not apply them to the summary itself. Then write what was "
            "decided and why; the files, names, numbers and exact values that matter; what has been done, and what failed "
            "and why; and what remains to do next. Keep it under {} words. Do not call any tool. Reply with the summary "
            "alone.",
            summary_tokens * 2 / 5 );
    }

    /**
     * @brief A session's request to carry its conversation past its context, or its model's reliable depth.
     *
     * Belongs to the session, so it outlives a model, and holds no number about any one model. Off until a program sets
     * one (Session::withCompaction).
     */
    export struct CompactionPolicy
    {
        /// Writes the summary in place of the model: from the turns it replaces, or nothing when it cannot.
        using Summarizer = std::function<std::optional<std::string>(
            std::span<const Mila::Dnn::Conversation::Turn>, std::stop_token )>;

        /// The prompt length that sets compaction off. Empty: the context limit less summary_tokens, which is also
        /// the latest it ever runs, so a summary always has its room.
        std::optional<Mila::Dnn::dim_t> trigger_tokens;

        /// The most a summary may generate, its reasoning included; one cut by this fails, since a partial summary
        /// loses context without saying so. Required unless a summarizer writes it. A policy whose budget is the
        /// model's whole context never runs by itself.
        Mila::Dnn::dim_t summary_tokens{ 0 };

        /// Copied word for word into the turn that replaces the history, each time. The system prompt needs no entry:
        /// it is not in the conversation, and every prompt renders it.
        std::vector<std::string> kept;

        /// What the model is asked; empty for defaultCompactionInstruction. Nothing is added to a program's own.
        std::string instruction;

        /// Laid over the session's template variables for the summary alone, such as a thinking switch. One this
        /// model's template does not accept is left out.
        std::map<std::string, std::string> template_variables;

        /// Empty: the model writes the summary.
        Summarizer summarizer;
    };
}
