/**
 * @file CudaExecutionContext.ixx
 * @brief CUDA-specific execution context implementation.
 *
 * Manages CUDA execution resources including streams and library handles (cuBLAS, cuDNN).
 * Separated from CudaDeviceContext to allow multiple independent execution streams
 * on the same device, enabling proper module isolation and stream management.
 */

module;
#include <memory>
#include <string>
#include <stdexcept>
#include <mutex>
#include <cuda_runtime.h>
#include <cublasLt.h>
#ifdef USE_CUDNN
#include <cudnn.h>
#endif

export module Compute.CudaExecutionContext;

import Compute.ExecutionContext;
import Compute.DeviceContext;
import Compute.DeviceType;
import Cuda.Helpers;

namespace Mila::Dnn::Compute
{
    /**
     * @brief CUDA-specific execution context implementation.
     *
     * Manages CUDA execution resources that are independent per execution stream:
     * - CUDA stream for asynchronous operations
     * - cuBLAS handle bound to the stream
     * - cuDNN handle bound to the stream (if enabled)
     *
     * Multiple CudaExecutionContexts can share the same DeviceContext (device),
     * each with independent streams for concurrent execution or module isolation.
     *
     * Design rationale:
     * - DeviceContext handles device selection (cudaSetDevice)
     * - ExecutionContext handles stream management and library handles
     * - Modules own ExecutionContexts to control their execution streams
     */
    export class CudaExecutionContext : public ExecutionContext {
    public:
        /**
         * @brief Constructs CUDA execution context from device context.
         *
         * Creates a new CUDA stream and initializes library handles bound to that stream.
         * Multiple execution contexts can share the same device context.
         *
         * @param device_context Device context for this execution context
         * @throws std::invalid_argument If device_context is null or not a CUDA device
         * @throws std::runtime_error If CUDA stream or handle creation fails
         */
        explicit CudaExecutionContext( std::shared_ptr<DeviceContext> device_context )
            : device_context_( device_context ) {

            if (!device_context_) {
                throw std::invalid_argument( "Device context cannot be null" );
            }

            if (!device_context_->isCudaDevice()) {
                throw std::invalid_argument(
                    "CudaExecutionContext requires CUDA device context"
                );
            }

            device_id_ = device_context_->getDeviceId();

            initializeExecutionResources();
        }

        /**
         * @brief Destructor with CUDA resource cleanup.
         *
         * Destroys CUDA stream and library handles in reverse order of creation.
         * Ensures proper resource cleanup even in multi-GPU scenarios.
         */
        ~CudaExecutionContext() override {
            releaseResources();
        }

        // ExecutionContext interface implementation

        /**
         * @brief Synchronizes the CUDA stream.
         *
         * Blocks until all operations submitted to this execution context's
         * stream have completed. Does not affect other streams on the same device.
         */
        void synchronize() override {
            if (stream_) {
                setCurrentDevice( device_id_ );
                cudaError_t error = cudaStreamSynchronize( stream_ );
                if (error != cudaSuccess) {
                    throw std::runtime_error(
                        "CUDA stream synchronization failed: " +
                        std::string( cudaGetErrorString( error ) )
                    );
                }
            }
        }

        // CUDA-specific methods

        /**
         * @brief Gets the CUDA stream for this execution context.
         *
         * Returns the non-blocking CUDA stream associated with this execution context.
         * Operations submitted to this stream execute asynchronously and independently
         * from other streams on the same device.
         *
         * @return CUDA stream handle
         */
        cudaStream_t getStream() const {
            return stream_;
        }

        /**
         * @brief Gets the cuBLAS handle for this execution context.
         *
         * Lazily creates the cuBLAS handle on first access and binds it to this
         * execution context's stream. Thread-safe via mutex protection.
         *
         * @return cuBLAS handle bound to this execution context's stream
         * @throws std::runtime_error If cuBLAS handle creation fails
         */
        cublasLtHandle_t getCublasLtHandle() {
            std::lock_guard<std::mutex> lock( handle_mutex_ );

            if (!cublasLtHandle_) {
                setCurrentDevice( device_id_ );

                cublasStatus_t status = cublasLtCreate( &cublasLtHandle_ );

                if (status != CUBLAS_STATUS_SUCCESS) {
                    const char* errorMsg;
                    switch (status) {
                    case CUBLAS_STATUS_NOT_INITIALIZED:
                        errorMsg = "CUBLAS library not initialized";
                        break;
                    case CUBLAS_STATUS_ALLOC_FAILED:
                        errorMsg = "Resource allocation failed";
                        break;
                    default:
                        errorMsg = "Unknown cuBLASLt error";
                        break;
                    }
                    throw std::runtime_error(
                        std::string( "Failed to create cuBLASLt handle: " ) + errorMsg
                    );
                }
            }
            return cublasLtHandle_;
        }

#ifdef USE_CUDNN
        /**
         * @brief Gets the cuDNN handle for this execution context.
         *
         * Lazily creates the cuDNN handle on first access and binds it to this
         * execution context's stream. Thread-safe via mutex protection.
         *
         * @return cuDNN handle bound to this execution context's stream
         * @throws std::runtime_error If cuDNN handle creation fails
         */
        cudnnHandle_t getCudnnHandle() {
            std::lock_guard<std::mutex> lock( handle_mutex_ );

            if (!cudnnHandle_) {
                setCurrentDevice( device_id_ );

                cudnnStatus_t status = cudnnCreate( &cudnnHandle_ );
                if (status != CUDNN_STATUS_SUCCESS) {
                    throw std::runtime_error( "Failed to create cuDNN handle" );
                }

                if (stream_) {
                    status = cudnnSetStream( cudnnHandle_, stream_ );
                    if (status != CUDNN_STATUS_SUCCESS) {
                        cudnnDestroy( cudnnHandle_ );
                        cudnnHandle_ = nullptr;
                        throw std::runtime_error( "Failed to set cuDNN stream" );
                    }
                }
            }
            return cudnnHandle_;
        }
#endif

    private:
        std::shared_ptr<DeviceContext> device_context_;
        int device_id_{ -1 };
        cudaStream_t stream_ = nullptr;
        bool stream_created_ = false;
        mutable cublasLtHandle_t cublasLtHandle_ = nullptr;
#ifdef USE_CUDNN
        mutable cudnnHandle_t cudnnHandle_ = nullptr;
#endif
        mutable std::mutex handle_mutex_;

        /**
         * @brief Initializes CUDA execution resources.
         *
         * Creates a non-blocking CUDA stream for this execution context.
         * Library handles are created lazily on first access.
         *
         * @throws std::runtime_error If stream creation fails
         */
        void initializeExecutionResources() {
            setCurrentDevice( device_id_ );

            cudaError_t streamErr = cudaStreamCreateWithFlags(
                &stream_,
                cudaStreamNonBlocking
            );

            if (streamErr != cudaSuccess) {
                throw std::runtime_error(
                    "Failed to create CUDA stream: " +
                    std::string( cudaGetErrorString( streamErr ) )
                );
            }
            stream_created_ = true;
        }

        /**
         * @brief Releases all CUDA execution resources.
         *
         * Destroys library handles and CUDA stream in reverse order of creation.
         * Called by destructor. Does not throw exceptions.
         */
        void releaseResources() noexcept {
            // Destroy library handles first
#ifdef USE_CUDNN
            if (cudnnHandle_) {
                cudnnDestroy( cudnnHandle_ );
                cudnnHandle_ = nullptr;
            }
#endif

            if (cublasLtHandle_) {
                cublasLtDestroy( cublasLtHandle_ );
                cublasLtHandle_ = nullptr;
            }

            // Destroy stream last
            if (stream_created_ && stream_) {
                cudaStreamDestroy( stream_ );
                stream_ = nullptr;
                stream_created_ = false;
            }
        }
    };
}