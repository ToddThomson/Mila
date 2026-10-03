/**
 * @file PromptPrefixReuse.ixx
 * @brief Which earlier positions a model can continue from without prefilling them again.
 */

module;
#include <cstdint>

export module Dnn.PromptPrefixReuse;

namespace Mila::Dnn
{
    /**
     * @brief Where a model can resume a conversation it has already read (PromptCaching.md).
     *
     * A model whose every layer caches each position separately can drop any suffix, so a new prompt
     * sharing any prefix with the last one prefills only what differs. A model with recurrent layers
     * holds a summary that cannot be rolled back; it resumes only from a position whose state it saved,
     * and a prompt that diverges before that position prefills from the start.
     */
    export enum class PromptPrefixReuse : int32_t
    {
        AnyPosition,
        SavedPosition
    };
}
