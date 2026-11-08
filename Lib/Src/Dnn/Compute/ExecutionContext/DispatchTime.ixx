/**
 * @file DispatchTime.ixx
 * @brief One kernel's GPU time over the dispatches a profile counted.
 */

export module Compute.DispatchTime;

import std;

namespace Mila::Dnn::Compute
{
    /// A kernel, how many times it ran while dispatch profiling was on, and the GPU's time for all of them.
    export struct DispatchTime
    {
        std::string kernel;
        std::int64_t dispatches{ 0 };
        double gpu_seconds{ 0.0 };
    };
}
