/**
 * @file CudaExecutionContext.ixx
 * @brief CUDA-specific execution context specialization.
 */

module;
#include <cuda_runtime.h>
#include <cublasLt.h>
#ifdef USE_CUDNN
#include <cudnn.h>
#endif
#include <memory>
#include <string>
#include <format>
#include <stdexcept>
#include <mutex>

export module Compute.ExecutionContext:Cuda;

import Compute.ExecutionContextTemplate;
import Compute.IExecutionContext;
import Compute.DeviceId;
import Compute.DeviceType;
import Compute.DeviceRegistry;
import Cuda.Helpers;
import Cuda.Error;

namespace Mila::Dnn::Compute
{
    /**
     * @brief CUDA execution context specialization.
     *
     * Manages CUDA execution resources including streams and library handles.
     * Each context owns an independent CUDA stream and owns the library
     * handles used by callers bound to this context's stream.
     *
     * Thread Safety:
     * - Multiple contexts can safely exist on the same device
     * - Each context has its own stream for isolated execution
     * - Library handles are owned by the context and must not be used
     *   concurrently from multiple threads without external synchronization.
     */
    export template<>
    class ExecutionContext<DeviceType::Cuda> : public IExecutionContext
    {
    public:
        /**
         * @brief Constructs CUDA execution context for a specific device.
         *
         * Creates an independent CUDA stream.
         *
         * @param device_id CUDA device identifier.
         * @throws std::invalid_argument If device_id type is not Cuda.
         * @throws std::runtime_error If CUDA stream creation fails.
         */
        explicit ExecutionContext( DeviceId device_id )
            : device_id_( validateDeviceId( device_id ) )
        {
            initializeResources();
        }

        /**
         * @brief Destructor with proper CUDA resource cleanup.
         *
         * Destroys the CUDA stream and any library handles owned by this context.
         */
        ~ExecutionContext()
        {
            releaseResources();
        }

        ExecutionContext( const ExecutionContext& ) = delete;
        ExecutionContext& operator=( const ExecutionContext& ) = delete;
        ExecutionContext( ExecutionContext&& ) = delete;
        ExecutionContext& operator=( ExecutionContext&& ) = delete;

        /**
         * @brief Gets the device identifier.
         *
         * @return DeviceId The CUDA device identifier (type + index).
         */
        [[nodiscard]] DeviceId getDeviceId() const noexcept override
        {
            return device_id_;
        }

        /**
         * @brief Synchronizes the CUDA stream.
         *
         * Blocks until all operations submitted to this context's stream complete.
         *
         * @throws std::runtime_error If stream synchronization fails.
         */
        void synchronize() override
        {
            cudaCheckLastError();

            Cuda::setCurrentDevice( device_id_.index );

            cudaCheckLastError();

            cudaError_t error = cudaStreamSynchronize( stream_ );

            if ( error != cudaSuccess )
            {
                throw std::runtime_error(
                    std::format( "CUDA stream synchronization failed: {}", cudaGetErrorString( error ) )
                );
            }
        }

        /**
         * @brief Gets the CUDA stream for asynchronous operations.
         *
         * @return cudaStream_t The CUDA stream owned by this context.
         */
        [[nodiscard]] cudaStream_t getStream() const noexcept
        {
            return stream_;
        }

        /**
         * @brief Get a cuBLASLt handle for this context.
         *
         * The handle is created lazily and owned by this context. Callers
         * should bind it to this context's stream if the cuBLASLt API in use
         * requires a stream binding (e.g. via cublasLtSetStream or explicit
         * stream parameters where applicable).
         *
         * @return cublasLtHandle_t The cuBLASLt handle.
         * @throws std::runtime_error If handle creation fails.
         */
        [[nodiscard]] cublasLtHandle_t getCublasLtHandle() const
        {
            std::call_once( cublas_init_flag_, [this]() {
                Cuda::setCurrentDevice( device_id_.index );

                cublasStatus_t status = cublasLtCreate( &cublas_handle_ );

                if ( status != CUBLAS_STATUS_SUCCESS )
                {
                    throw std::runtime_error(
                        "Failed to create cuBLASLt handle"
                    );
                }
            } );

            return cublas_handle_;
        }

#ifdef USE_CUDNN
        /**
         * @brief Get a cuDNN handle bound to this context's stream.
         *
         * The cuDNN handle is created lazily and owned by this context. The
         * returned handle is bound to the context's stream (via cudnnSetStream)
         * on each call to ensure correct stream association.
         *
         * @return cudnnHandle_t The cuDNN handle bound to this stream.
         * @throws std::runtime_error If handle creation or stream binding fails.
         */
        [[nodiscard]] cudnnHandle_t getCudnnHandle() const
        {
            std::call_once( cudnn_init_flag_, [this]() {
                Cuda::setCurrentDevice( device_id_.index );

                cudnnStatus_t status = cudnnCreate( &cudnn_handle_ );

                if ( status != CUDNN_STATUS_SUCCESS )
                {
                    throw std::runtime_error( "Failed to create cuDNN handle" );
                }
            } );

            if ( cudnn_handle_ )
            {
                cudnnStatus_t status = cudnnSetStream( cudnn_handle_, stream_ );

                if ( status != CUDNN_STATUS_SUCCESS )
                {
                    throw std::runtime_error( "Failed to bind cuDNN handle to stream" );
                }
            }

            return cudnn_handle_;
        }
#endif

    private:

        DeviceId device_id_;

        cudaStream_t stream_{ nullptr };
        bool stream_created_{ false };

        mutable cublasLtHandle_t cublas_handle_{ nullptr };
        mutable std::once_flag cublas_init_flag_;

#ifdef USE_CUDNN
        mutable cudnnHandle_t cudnn_handle_{ nullptr };
        mutable std::once_flag cudnn_init_flag_;
#endif

        /**
         * @brief Validates device identifier for CUDA context.
         *
         * @param device_id Device identifier to validate.
         * @return DeviceId The validated device identifier.
         * @throws std::invalid_argument If device type is not Cuda.
         */
        static DeviceId validateDeviceId( DeviceId device_id )
        {
            if ( device_id.type != DeviceType::Cuda )
            {
                throw std::invalid_argument(
                    std::format( "CudaExecutionContext requires Cuda device type, got '{}'",
                                device_id.toString() )
                );
            }

            if ( !DeviceRegistry::instance().hasDevice( device_id ) )
            {
                throw std::invalid_argument(
                    std::format( "CudaExecutionContext device '{}' is not registered",
                        device_id.toString() )
                );
            }

            return device_id;
        }

        /**
         * @brief Initializes CUDA execution resources (stream).
         *
         * Creates a non-blocking CUDA stream for asynchronous operations.
         *
         * @throws std::runtime_error If stream creation fails.
         */
        void initializeResources()
        {
            Cuda::setCurrentDevice( device_id_.index );

            cudaError_t error = cudaStreamCreateWithFlags(
                &stream_,
                cudaStreamDefault // DEBUG: changed from cudaStreamNonBlocking
            );

            if ( error != cudaSuccess )
            {
                throw std::runtime_error(
                    std::format( "Failed to create CUDA stream: {}",
                                cudaGetErrorString( error ) )
                );
            }

            stream_created_ = true;
        }

        /**
         * @brief Releases all CUDA resources.
         *
         * Destroys library handles and the stream. Called by destructor -
         * does not throw.
         */
        void releaseResources() noexcept
        {
            // Destroy library handles first
#ifdef USE_CUDNN
            if ( cudnn_handle_ )
            {
                cudnnDestroy( cudnn_handle_ );
                cudnn_handle_ = nullptr;
            }
#endif

            if ( cublas_handle_ )
            {
                cublasLtDestroy( cublas_handle_ );
                cublas_handle_ = nullptr;
            }

            // Destroy stream last
            if ( stream_created_ && stream_ )
            {
                cudaError_t err = cudaStreamSynchronize( stream_ );
                if ( err != cudaSuccess )
                {
                    std::fprintf( stderr,
                        "ExecutionContext: Failed to synchronize CUDA stream: %s\n",
                        cudaGetErrorString( err ) );
                }

                err = cudaStreamDestroy( stream_ );
                if ( err != cudaSuccess )
                {
                    std::fprintf( stderr,
                        "ExecutionContext: Failed to destroy CUDA stream: %s\n",
                        cudaGetErrorString( err ) );
                }

                stream_ = nullptr;
                stream_created_ = false;
            }
        }
    };

    export using CudaExecutionContext = ExecutionContext<DeviceType::Cuda>;
}