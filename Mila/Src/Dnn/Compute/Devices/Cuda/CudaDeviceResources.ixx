module;
#include <mutex>
#include <cuda_runtime.h>
#include <cublasLt.h>
#ifdef USE_CUDNN
#include <cudnn.h>
#endif

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
                cudaSetDevice( device_id_ );
                cudnnCreate( &cudnn_handle_ );
                } );
            
            return cudnn_handle_;
        }
#endif

        int getDeviceId() const { return device_id_; }

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