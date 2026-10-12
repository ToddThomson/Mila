/**
 * @file Mila_py.Deployment.ixx
 * @brief Whether an installed model runs on a device, as the library's planner answers it, with a std-only surface.
 *
 * Kept apart from Mila.Bindings for the reason Mila.Bindings.Runtime is: Mila_py.cpp must not import Mila.
 */

module;

#include <optional>
#include <string>
#include <vector>

export module Mila.Bindings.Deployment;

namespace Mila::Bindings
{
    /**
     * @brief The weight format an installed model is stored in, in the words a load takes: bf16, fp8, fp4, q4_0
     *        or cb2-3.
     *
     * @throws std::runtime_error when no model of that name is installed, or its record names a format this
     *         build does not read.
     */
    export std::string storedWeightQuantization( const std::string& name );

    /**
     * @brief The weight formats an installed model has a deployment in on the device, its stored format first
     *        when that fits, then each format its reference weights can be quantized to on load.
     *
     * Planned against the device's whole memory, so a model already loaded elsewhere does not count against it.
     *
     * @return Nothing when the planner could not answer: missing files, a family this build cannot deploy, or
     *         an unreadable package. Empty when it answered and nothing fits.
     *
     * @throws std::runtime_error when no model of that name is installed.
     */
    export std::optional<std::vector<std::string>> fittingWeightQuantizations(
        const std::string& name, int device_index );
}
