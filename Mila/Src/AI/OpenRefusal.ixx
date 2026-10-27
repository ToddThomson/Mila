/**
 * @file OpenRefusal.ixx
 * @brief Why a name does not open as a model, or as an agent, decided before anything is loaded.
 */

export module AI.OpenRefusal;

import std;

namespace Mila::AI
{
    /**
     * @brief The reason a model will not open, and the name it was asked for by (Direction.md 4.3).
     *
     * Whether the model fits the device is not among them: that is the planner's answer (DeploymentRefusal), read
     * from free memory a resident model still holds. The reason is what an application words for its own reader;
     * the detail is the library's sentence, for a log or a developer.
     */
    export struct OpenRefusal
    {
        enum class Reason
        {
            /// The store holds no model of that name.
            NotInstalled,

            /// Installed, but a file its record names is missing from the store.
            FilesMissing,

            /// A base model: it has no chat template to converse in, so it opens as a model but not as an agent.
            BaseModel,

            /// Its architecture has more than one chat template, and its record does not say which it was trained with.
            ChatTemplateNotRecorded,

            /// It was trained with a chat template this build does not have.
            NoChatTemplate,

            /// Its architecture is not one this build deploys.
            ArchitectureNotDeployed,

            /// Its weights are stored in a format this build does not read.
            WeightFormatUnknown,

            /// The request names a weight format its stored weights cannot be loaded as.
            WeightFormatNotDerivable
        };

        Reason reason{ Reason::NotInstalled };

        /// The name as the store spells it, or as it was asked for when the store holds none.
        std::string name;

        std::string detail;
    };
}
