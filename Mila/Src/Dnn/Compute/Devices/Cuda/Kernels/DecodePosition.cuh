// DecodePosition.cuh
//
// The decode position a CUDA execution context holds on the device, written once per decode step and read by every
// decode kernel whose work depends on it. DecodeGraph.md section 4.1.

#pragma once

#include <cuda_runtime.h>

namespace Mila::Dnn::Compute::Cuda
{
    /**
     * @brief Enqueue a write of `position` to `target` on `stream`.
     *
     * A one-thread kernel rather than a copy: the position is a launch argument, copied when the launch is enqueued,
     * so the host may enqueue the next step's position while this step still runs.
     */
    void cuda_set_decode_position( int* target, int position, cudaStream_t stream );
}
