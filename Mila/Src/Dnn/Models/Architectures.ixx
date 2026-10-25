/**
 * @file Architectures.ixx
 * @brief Every architecture this build runs: the one place an architecture's name becomes its types.
 */

export module Dnn.Models.Architectures;

export import Dnn.Models.ArchitectureList;
export import Dnn.Models.GemmaArchitecture;
export import Dnn.Models.LlamaArchitecture;
export import Dnn.Models.QwenArchitecture;

namespace Mila::Dnn
{
    /**
     * @brief The architectures a manifest may name.
     *
     * Adding one is its family's directory, which is written anyway, and an entry here (ModelHandle.md 3.3).
     * GPT-2 is absent: it has no planner, so nothing that opens a model by name can deploy it.
     */
    export using Architectures = ArchitectureList<GemmaArchitecture, LlamaArchitecture, QwenArchitecture>;
}
