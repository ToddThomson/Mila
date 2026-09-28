/**
 * @file Chat.FootprintBudget.ixx
 * @brief What the model listing needs to cost a row against this machine, right now.
 */

module;
#include <cstddef>
#include <string>

export module Chat.FootprintBudget;

import Mila;

namespace Mila::ChatApp
{
    /**
     * @brief What the listing needs to cost a row against this machine, right now.
     *
     * Assembled by the session rather than read here, because both figures are facts about the
     * live session: the context it is running at, and the model it already has resident.
     */
    export struct FootprintBudget
    {
        /// The context every row is answered at, when the session names one. Zero when context is
        /// automatic, where each row is answered at the largest context THAT model would get.
        ///
        /// The distinction is the whole defect this replaced: one auto-derived number priced every
        /// row, so Gemma's 56320 -- affordable only because most of its layers are sliding-window
        /// -- was charged to Llama rows that would never be given it, and three of six carried a
        /// warning that could not happen. A context the user NAMED is different: that one really
        /// does apply to every row, because it is what loading any of them would use.
        Mila::Dnn::dim_t fixed_context_length{ 0 };

        /// The memory a load may claim. The caller decides what that means -- the listing asks
        /// what this device can run, which is its capacity, not what happens to be free now.
        std::size_t available_bytes{ 0 };

        /// The card's own name, for the line beneath the table. Empty when it could not be read,
        /// which drops the name and keeps the capacity -- a verdict has to say what it was measured
        /// against even when it cannot say what that thing is called.
        std::string device_name;

        /// Which card the rows are priced against. Carried alongside its capacity rather than
        /// derived here, because a row is measured by BUILDING the graph on that device -- the
        /// two must name the same card or a row would be sized on one and graded against another.
        int device_index{ 0 };

        /// The loaded model's name, marked in the listing. Empty when none is loaded.
        std::string resident_model;
    };
}
