module;
#include <mutex>
#include <cuda_runtime.h>
#include <cublasLt.h>
#ifdef USE_CUDNN
#include <cudnn.h>
#endif
#include <unordered_map> // for per-thread stream map
#include <utility>
#include <stdexcept>

export module Compute.CudaDeviceResources;

import Cuda.Helpers;

namespace Mila::Dnn::Compute
{
    export class CudaDeviceResources 
    {
    public:
        explicit CudaDeviceResources( int device_id )
            : device_id_( device_id )
        {
			initializeResources();
        }

        ~CudaDeviceResources() {
			releaseResources();
        }

        cublasLtHandle_t getCublasLtHandle() {
            std::call_once( cublas_init_flag_, [this]() {

                Cuda::setCurrentDevice( device_id_ );

                cublasStatus_t status = cublasLtCreate( &cublas_handle_ );

                if (status != CUBLAS_STATUS_SUCCESS)
                {
                    throw std::runtime_error(
                        "Failed to create cuBLASLt handle: " /* FIXME: + cublasStatusToString(status) */
                    );
                }

            } );
            
            return cublas_handle_;
        }

#ifdef USE_CUDNN
        /**
         * @brief Gets the cuDNN handle (lazy initialization).
         *
         * Thread-safe lazy creation of cuDNN handle bound to this context's stream.
         *
         * @return cuDNN handle
         * @throws std::runtime_error If cuDNN handle creation fails
         */

        cudnnHandle_t getCudnnHandle() {
            std::call_once( cudnn_init_flag_, [this]() {
                Cuda::setCurrentDevice( device_id_ );
                cudnnCreate( &cudnn_handle_ );
                } );
            
            return cudnn_handle_;
        }
#endif

        int getDeviceId() const { return device_id_; }

        /**
         * @brief Get the device-global stream created during initialization.
         *
         * This is a single stream associated with the device resources instance.
         */
        cudaStream_t getStream() const { return stream_; }

        /**
         * @brief Get or create a per-thread non-blocking stream for this device.
         *
         * Creates one non-blocking stream per-device per-thread on demand. Streams are
         * destroyed automatically on thread exit. This should be used as a fallback when
         * no explicit ExecutionContext / bound stream is provided.
         *
         * @throws std::runtime_error If stream creation fails
         */
        cudaStream_t getPerThreadStream()
        {
            // Lightweight wrapper type that destroys stream on thread exit
            struct ThreadStreamWrapper
            {
                cudaStream_t s{ nullptr };
                ~ThreadStreamWrapper()
                {
                    if (s)
                    {
                        // best-effort destroy; do not throw in destructor
                        cudaStreamDestroy( s );
                        s = nullptr;
                    }
                }
            };

            // thread-local storage: map device_id -> ThreadStreamWrapper
            static thread_local std::unordered_map<int, ThreadStreamWrapper> tls_streams;

            auto it = tls_streams.find( device_id_ );

            if (it != tls_streams.end() && it->second.s != nullptr)
            {
                return it->second.s;
            }

            // Need to create a new per-thread stream for this device
            Cuda::setCurrentDevice( device_id_ );

            cudaStream_t new_stream = nullptr;
            cudaError_t err = cudaStreamCreateWithFlags( &new_stream, cudaStreamNonBlocking );
            
            if (err != cudaSuccess)
            {
                throw std::runtime_error(
                    "Failed to create per-thread CUDA stream: " + std::string( cudaGetErrorString( err ) )
                );
            }

            // Emplace wrapper into thread-local map; wrapper will destroy stream on thread exit.
            ThreadStreamWrapper wrapper;
            wrapper.s = new_stream;
            tls_streams.emplace( device_id_, std::move( wrapper ) );

            return new_stream;
        }

    private:
        int device_id_;
        
        cudaStream_t stream_{ nullptr };
        bool stream_created_{ false };

        mutable cublasLtHandle_t cublas_handle_{ nullptr };

#ifdef USE_CUDNN
        mutable cudnnHandle_t cudnn_handle_{ nullptr };
#endif        
        
        mutable std::once_flag cublas_init_flag_;
        mutable std::once_flag cudnn_init_flag_;

        /**
         * @brief Initializes CUDA execution resources (stream).
         *
         * Creates non-blocking CUDA stream for asynchronous operations.
         *
         * @throws std::runtime_error If stream creation fails
         */
        void initializeResources() {
            Cuda::setCurrentDevice( device_id_ );

            cudaError_t error = cudaStreamCreateWithFlags(
                &stream_,
                cudaStreamNonBlocking
            );

            if (error != cudaSuccess)
            {
                throw std::runtime_error(
                    "Failed to create CUDA stream: " +
                    std::string( cudaGetErrorString( error ) )
                );
            }

            stream_created_ = true;
        }

        /**
         * @brief Releases all CUDA resources.
         *
         * Destroys library handles and stream in reverse order of creation.
         * Called by destructor - does not throw exceptions.
         */
        void releaseResources() noexcept {
            // Destroy library handles first
#ifdef USE_CUDNN
            if (cudnn_handle_)
            {
                cudnnDestroy( cudnn_handle_ );
                cudnn_handle_ = nullptr;
            }
#endif

            if (cublas_handle_)
            {
                cublasLtDestroy( cublas_handle_ );
                cublas_handle_ = nullptr;
            }

            // Destroy stream last
            if (stream_created_ && stream_)
            {
                cudaStreamDestroy( stream_ );
                stream_ = nullptr;
                stream_created_ = false;
            }
        }
    };
}