/**
 * @file Chat.FootprintVerdict.ixx
 * @brief How a predicted footprint stands against a card's capacity, for the /model list column.
 */

export module Chat.FootprintVerdict;

namespace Mila::ChatApp
{
    export enum class FootprintVerdict
    {
        /// No prediction was available, so nothing is claimed.
        Unknown,

        Fits,

        /// The weights fit but the total does not. Context is the lever.
        DoesNotFit,

        /// The weights alone exceed what is available. Context does not shrink them;
        /// quantization does.
        WeightsExceedAvailable
    };

    /// True when a row's verdict is one the user needs to be told about.
    export inline bool isOverBudget( FootprintVerdict verdict )
    {
        return verdict == FootprintVerdict::DoesNotFit
            || verdict == FootprintVerdict::WeightsExceedAvailable;
    }
}
