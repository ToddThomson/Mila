/**
 * @file Mila_py.Wrappers.SamplingSettingsInfo.ixx
 * @brief The four sampling knobs, as the sampler would draw with them.
 */

export module Mila.Bindings:SamplingSettingsInfo;

import std;

namespace Mila::Bindings
{
    /// Every knob filled: a value the publisher did not state is the sampler's neutral one.
    export struct SamplingSettingsInfo
    {
        float temperature{ 1.0f };
        int top_k{ 0 };
        float top_p{ 1.0f };
        float presence_penalty{ 0.0f };
    };
}
