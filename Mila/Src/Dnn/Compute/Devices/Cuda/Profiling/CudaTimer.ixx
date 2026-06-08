/**
 * @file CudaTimer.ixx
 * @brief GPU-accurate interval timer using a CUDA event pair.
 *
 * Provides CudaTimer, a lightweight RAII wrapper around two cudaEvent_t handles
 * (start and stop). The timer records events into a CUDA stream at the desired
 * measurement points and reads the elapsed GPU time after the stream reaches the
 * stop event.
 *
 * Intended use:
 *   - Beta.1 profiling pass: per-operation GPU timing in CudaProfileScope.
 *   - Fine-grained kernel sequence measurement without host-side synchronization
 *     at every boundary.
 */

module;
#include <cuda_runtime.h>
#include <stdexcept>
#include <format>

export module Compute.Cuda.Timer;

namespace Mila::Dnn::Compute
{
    /**
     * @brief GPU-accurate interval timer using a CUDA event pair.
     *
     * Records start and stop events on a given CUDA stream. Reading the elapsed
     * time via elapsedMilliseconds() synchronizes on the stop event, ensuring
     * the GPU has reached that point before the measurement is read back.
     *
     * This approach avoids inserting a full cudaStreamSynchronize() at every
     * measurement boundary, so it is suitable for fine-grained per-operation
     * profiling during the beta.1 pass.
     *
     * Example usage:
     * @code
     *   CudaTimer timer;
     *   timer.start( stream );
     *   // ... dispatch CUDA kernels ...
     *   timer.stop( stream );
     *   float elapsed_ms = timer.elapsedMilliseconds();
     * @endcode
     *
     * Thread safety:
     *   A CudaTimer instance must not be used concurrently from multiple threads.
     *   Each CUDA stream should have its own CudaTimer.
     */
    export class CudaTimer
    {
    public:

        /**
         * @brief Construct a CudaTimer and allocate its CUDA event pair.
         *
         * @throws std::runtime_error if either CUDA event cannot be created.
         */
        CudaTimer()
        {
            cudaError_t error = cudaEventCreate( &start_event_ );

            if ( error != cudaSuccess )
            {
                throw std::runtime_error(
                    std::format( "CudaTimer: failed to create start event: {}",
                        cudaGetErrorString( error ) ) );
            }

            error = cudaEventCreate( &stop_event_ );

            if ( error != cudaSuccess )
            {
                cudaEventDestroy( start_event_ );
                start_event_ = nullptr;
                throw std::runtime_error(
                    std::format( "CudaTimer: failed to create stop event: {}",
                        cudaGetErrorString( error ) ) );
            }
        }

        /**
         * @brief Destructor. Releases both CUDA events.
         */
        ~CudaTimer()
        {
            if ( start_event_ )
            {
                cudaEventDestroy( start_event_ );
                start_event_ = nullptr;
            }

            if ( stop_event_ )
            {
                cudaEventDestroy( stop_event_ );
                stop_event_ = nullptr;
            }
        }

        CudaTimer( const CudaTimer& ) = delete;
        CudaTimer& operator=( const CudaTimer& ) = delete;
        CudaTimer( CudaTimer&& ) = delete;
        CudaTimer& operator=( CudaTimer&& ) = delete;

        /**
         * @brief Record the start event into the given CUDA stream.
         *
         * The GPU timestamp is recorded at the point the stream reaches this event,
         * which may be later than the host-side call.
         *
         * @param stream CUDA stream on which to record the start event.
         * @throws std::runtime_error if event recording fails.
         */
        void start( cudaStream_t stream )
        {
            cudaError_t error = cudaEventRecord( start_event_, stream );

            if ( error != cudaSuccess )
            {
                throw std::runtime_error(
                    std::format( "CudaTimer: failed to record start event: {}",
                        cudaGetErrorString( error ) ) );
            }
        }

        /**
         * @brief Record the stop event into the given CUDA stream.
         *
         * The GPU timestamp is recorded at the point the stream reaches this event.
         * Call elapsedMilliseconds() afterwards to read the measured interval.
         *
         * @param stream CUDA stream on which to record the stop event.
         * @throws std::runtime_error if event recording fails.
         */
        void stop( cudaStream_t stream )
        {
            cudaError_t error = cudaEventRecord( stop_event_, stream );

            if ( error != cudaSuccess )
            {
                throw std::runtime_error(
                    std::format( "CudaTimer: failed to record stop event: {}",
                        cudaGetErrorString( error ) ) );
            }
        }

        /**
         * @brief Returns the elapsed GPU time in milliseconds between start and stop.
         *
         * Synchronizes on the stop event (cudaEventSynchronize) before reading back
         * the elapsed time. The caller's thread blocks until the GPU has reached the
         * stop event.
         *
         * @return Elapsed time in milliseconds as reported by the CUDA event subsystem.
         * @throws std::runtime_error if synchronization or elapsed time query fails.
         */
        [[nodiscard]] float elapsedMilliseconds() const
        {
            cudaError_t error = cudaEventSynchronize( stop_event_ );

            if ( error != cudaSuccess )
            {
                throw std::runtime_error(
                    std::format( "CudaTimer: failed to synchronize stop event: {}",
                        cudaGetErrorString( error ) ) );
            }

            float elapsed_ms = 0.0f;
            error = cudaEventElapsedTime( &elapsed_ms, start_event_, stop_event_ );

            if ( error != cudaSuccess )
            {
                throw std::runtime_error(
                    std::format( "CudaTimer: failed to read elapsed time: {}",
                        cudaGetErrorString( error ) ) );
            }

            return elapsed_ms;
        }

    private:

        cudaEvent_t start_event_{ nullptr };
        cudaEvent_t stop_event_{ nullptr };
    };
}
