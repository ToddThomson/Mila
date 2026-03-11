/**
 * @file ComponentExecutionMode.ixx
 * @brief Build-time execution mode for component buffer allocation policy.
 */
module;
#include <cstdint>

export module Dnn.Model:RuntimeMode;

namespace Mila::Dnn
{
    /**
     * @brief Runtime mode governing Model API and Network build policy.
     *
     * Immutable after Model construction. Determines which public API
     * methods are valid and how the Network allocates its buffers.
     *
     * | Mode      | Network build shape  | Valid Model API         |
     * |-----------|----------------------|-------------------------|
     * | Inference | { 1, context_len }   | generate()              |
     * | Training  | { batch, seq_len }   | eval(), sample()        |
     */
    export enum class RuntimeMode : uint8_t
    {
        Inference,  // decode path, T=1 buffers, generate() only
        Training    // full sequence buffers, eval/sample/backward
    };
}
