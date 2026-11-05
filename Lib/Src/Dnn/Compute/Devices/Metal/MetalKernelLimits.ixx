/**
 * @file MetalKernelLimits.ixx
 * @brief What a compiled kernel reports about how it may be dispatched.
 */

export module Compute.MetalKernelLimits;

import std;

namespace Mila::Dnn::Compute
{
    export struct MetalKernelLimits
    {
        std::size_t thread_execution_width{ 0 };
        std::size_t max_total_threads_per_threadgroup{ 0 };
    };
}
