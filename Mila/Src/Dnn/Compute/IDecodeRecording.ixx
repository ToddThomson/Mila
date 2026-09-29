/**
 * @file IDecodeRecording.ixx
 * @brief A decode step's launches, recorded once and replayed (DecodeGraph.md).
 */

module;
#include <functional>

export module Compute.IDecodeRecording;

namespace Mila::Dnn::Compute
{
    /**
     * @brief The launches one decode step enqueues on an execution context, held so they can be enqueued again.
     *
     * Made by the context (IExecutionContext::createDecodeRecording) and owned by the network that decodes on it.
     * A recording replays exactly what was recorded -- the same kernels, arguments and buffers -- so it stays
     * correct only while the step is a pure function of device memory (DecodeGraph.md section 5.3).
     */
    export class IDecodeRecording
    {
    public:
        virtual ~IDecodeRecording() = default;

        /**
         * @brief Record the launches `step` enqueues, without running them.
         *
         * @return false when the step could not be recorded: it made a call a recording cannot hold, or threw.
         *         Nothing is recorded then, the context's stream is usable, and the step has not run.
         */
        virtual bool record( const std::function<void()>& step ) = 0;

        /// Enqueue the recorded launches on the context's stream.
        virtual void replay() = 0;

    protected:
        IDecodeRecording() = default;
    };
}
