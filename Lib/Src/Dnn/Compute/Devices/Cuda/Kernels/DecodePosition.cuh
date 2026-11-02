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

    /**
     * @brief Enqueue `*position + offset` into `target` on `stream`, both read and written on the device.
     *
     * For a step whose kernels need a position derived from the context's: Gemma 4's draft model rotates its query at
     * the decode position and attends keys only up to the one before it (Gemma4Mtp.md 4.3). The offset is fixed for
     * a step's life, so the step stays a pure function of device memory.
     */
    void cuda_offset_decode_position( int* target, const int* position, int offset, cudaStream_t stream );
}
