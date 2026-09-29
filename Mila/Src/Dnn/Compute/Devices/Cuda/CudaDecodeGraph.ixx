/**
 * @file CudaDecodeGraph.ixx
 * @brief A decode step recorded as one CUDA graph under stream capture, replayed with one launch (DecodeGraph.md).
 */

module;
#include <cuda_runtime.h>
#include <format>
#include <functional>
#include <stdexcept>

export module Compute.CudaDecodeGraph;

import Compute.IDecodeRecording;
import Cuda.Error;

namespace Mila::Dnn::Compute
{
    /**
     * @brief The CUDA recording of a decode step: the launches captured from one stream, instantiated once.
     *
     * Capture is thread-local, so a call on this thread that a graph cannot hold -- an allocation, a synchronizing
     * copy -- fails the recording rather than running; other threads are unaffected.
     */
    export class CudaDecodeGraph : public IDecodeRecording
    {
    public:
        explicit CudaDecodeGraph( cudaStream_t stream )
            : stream_( stream )
        {
        }

        ~CudaDecodeGraph() override
        {
            release();
        }

        CudaDecodeGraph( const CudaDecodeGraph& ) = delete;
        CudaDecodeGraph& operator=( const CudaDecodeGraph& ) = delete;
        CudaDecodeGraph( CudaDecodeGraph&& ) = delete;
        CudaDecodeGraph& operator=( CudaDecodeGraph&& ) = delete;

        bool record( const std::function<void()>& step ) override
        {
            release();

            if ( cudaStreamBeginCapture( stream_, cudaStreamCaptureModeThreadLocal ) != cudaSuccess )
            {
                cudaDiscardLastError();

                return false;
            }

            // Whatever the step throws, the capture must be ended before the stream is used again.
            bool stepped = true;

            try
            {
                step();
            }
            catch ( ... )
            {
                stepped = false;
            }

            cudaGraph_t graph = nullptr;
            const cudaError_t ended = cudaStreamEndCapture( stream_, &graph );

            if ( !stepped || ended != cudaSuccess || graph == nullptr )
            {
                if ( graph != nullptr )
                    cudaGraphDestroy( graph );

                cudaDiscardLastError();

                return false;
            }

            const cudaError_t instantiated = cudaGraphInstantiate( &executable_, graph, 0 );
            cudaGraphDestroy( graph );

            if ( instantiated != cudaSuccess )
            {
                executable_ = nullptr;
                cudaDiscardLastError();

                return false;
            }

            return true;
        }

        void replay() override
        {
            if ( executable_ == nullptr )
                throw std::logic_error( "CudaDecodeGraph::replay: nothing has been recorded" );

            const cudaError_t launched = cudaGraphLaunch( executable_, stream_ );

            if ( launched != cudaSuccess )
            {
                throw std::runtime_error(
                    std::format( "CudaDecodeGraph::replay: {}", cudaGetErrorString( launched ) ) );
            }
        }

    private:

        void release() noexcept
        {
            if ( executable_ != nullptr )
            {
                cudaGraphExecDestroy( executable_ );
                executable_ = nullptr;
            }
        }

        cudaStream_t stream_;
        cudaGraphExec_t executable_{ nullptr };
    };
}
