/**
 * @file ToolBroker.Worker.ixx
 * @brief A running worker process, and the one function that starts one.
 */

export module ToolBroker.Worker;

import std;
import ToolBroker.LineChannel;
import ToolBroker.WorkerLimits;
import ToolBroker.WorkerProgram;

namespace Mila::ToolBroker
{
    /**
     * @brief A worker: its channel, its process, and the job that ends it and everything it started.
     *
     * Destroying it closes the worker's input, which MCP's stdio binding makes the signal to exit, waits up to
     * `kExitGrace` for that, then closes the job. The job ends every process still in it, so a worker cannot leave
     * a process behind by starting one.
     */
    export class Worker
    {
    public:
        static constexpr std::chrono::milliseconds kExitGrace{ 2000 };

        /** Takes ownership of the process and job handles; `launchWorker` is the only caller. */
        Worker( void* process, void* job, std::uint32_t process_id, LineChannel channel );

        ~Worker();

        Worker( Worker&& other ) noexcept;
        Worker& operator=( Worker&& other ) noexcept;

        Worker( const Worker& ) = delete;
        Worker& operator=( const Worker& ) = delete;

        LineChannel& channel() noexcept;

        std::uint32_t processId() const noexcept;

        bool isRunning() const;

    private:
        void stop() noexcept;

        void* process_ = nullptr;
        void* job_ = nullptr;
        std::uint32_t process_id_ = 0;
        LineChannel channel_;
    };

    /**
     * @brief Starts `program` as a worker under `limits`.
     *
     * The one place a worker is started, so a sandbox (ToolBroker.md section 6.3) replaces this function and nothing
     * that calls it. The process joins its job as it is created, with no breakaway; it inherits only its standard
     * handles; its standard error goes nowhere, which MCP allows; and it sees only the environment in `limits`.
     *
     * @throws std::invalid_argument if an environment variable's name is empty or holds `=` or a null character.
     * @throws std::system_error if the process cannot be started.
     */
    export Worker launchWorker( const WorkerProgram& program, const WorkerLimits& limits );
}
