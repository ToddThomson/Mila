/**
 * @file DeviceContext.ixx
 * @brief Manages device contexts for compute operations in the Mila deep neural network framework.
 *
 * This file provides the implementation of the DeviceContext class which serves as an abstraction
 * over different compute devices (CPU, CUDA GPUs). It facilitates device selection, resource management,
 * and execution control for neural network operations across heterogeneous hardware platforms.
 *
 * The DeviceContext class enables:
 * - Automatic detection and selection of available compute devices
 * - Management of device-specific resources (streams, memory)
 * - CUDA stream handling for asynchronous execution
 * - Thread-safe handling of device contexts and resources
 *
 * This is a core component of the Mila compute infrastructure that other operations
 * depend on for device-specific execution.
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

export module Compute.DeviceContext;

import Compute.ComputeDevice;
import Compute.CudaDevice;
import Compute.DeviceRegistry;
import Compute.DeviceType;

namespace Mila::Dnn::Compute
{
    /**
    * @brief The DeviceContext class manages device contexts for module and tensor computations.
    *
    * This class provides functionality for managing compute devices and their associated resources,
    * such as CUDA streams and optional cuBLASLt and cuDNN handles. Multiple instances can be created
    * to manage different devices.
    */
    export class DeviceContext {
    public:

        /**
        * @brief Constructor with a specific device.
        * @param device_name The name of the device to use (e.g., "CUDA:0", "CPU").
        * @throws std::runtime_error If the device name is invalid or device initialization fails.
        */
        explicit DeviceContext( const std::string& device_name ) {
            setDevice( device_name );
        }

        /**
        * @brief Destructor. Cleans up any associated resources.
        */
        ~DeviceContext() {
            releaseResources();
        }

        /**
        * @brief Copy constructor (deleted).
        * @note DeviceContext is not copyable due to unique resource ownership.
        */
        DeviceContext( const DeviceContext& ) = delete;

        /**
        * @brief Copy assignment operator (deleted).
        * @note DeviceContext is not copyable due to unique resource ownership.
        */
        DeviceContext& operator=( const DeviceContext& ) = delete;

        /**
        * @brief Move constructor.
        * @param other The source DeviceContext to move from.
        */
        DeviceContext( DeviceContext&& other ) noexcept {
            moveFrom( std::move( other ) );
        }

        /**
        * @brief Move assignment operator.
        * @param other The source DeviceContext to move from.
        * @return A reference to this DeviceContext.
        */
        DeviceContext& operator=( DeviceContext&& other ) noexcept {
            if ( this != &other ) {
                releaseResources();
                moveFrom( std::move( other ) );
            }
            return *this;
        }

        /**
        * @brief Checks if the current device is of a specific type.
        *
        * @param type The device type to check against.
        * @return True if the device matches the specified type, false otherwise.
        */
        bool isDeviceType( DeviceType type ) const {
            return device_ && device_->getDeviceType() == type;
        }

        /**
         * @brief Checks if the current device is a CUDA device.
         *
         * @return True if the device is a CUDA device, false otherwise.
         */
        bool isCudaDevice() const {
            return isDeviceType( DeviceType::Cuda );
        }

        /**
         * @brief Gets the compute capability of the current CUDA device.
         *
         * @return std::pair<int, int> The major and minor versions of the compute capability,
         *         or {0,0} if the device is not a CUDA device or compute capability couldn't be determined.
         */
        std::pair<int, int> getComputeCapability() const {
            if ( !isCudaDevice() ) {
                return { 0, 0 };
            }

            auto cudaDevice = std::dynamic_pointer_cast<CudaDevice>(device_);
            if ( cudaDevice ) {
                return cudaDevice->getProperties().getComputeCapability();
            }

            return { 0, 0 };
        }

        /**
        * @brief Gets the current device.
        * @return A shared pointer to the current device.
        */
        std::shared_ptr<ComputeDevice> getDevice() const {
            return device_;
        }

        /**
        * @brief Gets the ID of the current CUDA device.
        * @return The CUDA device ID, or -1 if not using a CUDA device.
        */
        int getDeviceId() const {
            return isCudaDevice() ? device_id_ : -1;
        }

        /**
        * @brief Gets the current CUDA stream.
        * @return The current CUDA stream, or nullptr if not using CUDA.
        */
        cudaStream_t getStream() const {
            return stream_;
        }

        /**
        * @brief Gets the cuBLASLt handle, initializing it if necessary.
        * @return The cuBLASLt handle.
        * @throws std::runtime_error If creating the cuBLASLt handle fails.
        */
        cublasLtHandle_t getCublasLtHandle() {
            std::lock_guard<std::mutex> lock( handle_mutex_ );
            if ( !cublasLtHandle_ && isCudaDevice() ) {
                cublasStatus_t status = cublasLtCreate( &cublasLtHandle_ );
                if ( status != CUBLAS_STATUS_SUCCESS ) {
                    // Convert cuBLAS error to a readable message
                    const char* errorMsg;
                    switch ( status ) {
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
                    throw std::runtime_error( std::string( "Failed to create cuBLASLt handle: " ) + errorMsg );
                }
            }
            return cublasLtHandle_;
        }

        /**
        * @brief Gets the cuDNN handle, initializing it if necessary.
        * @return The cuDNN handle.
        * @throws std::runtime_error If creating the cuDNN handle fails.
        */
    #ifdef USE_CUDNN
        cudnnHandle_t getCudnnHandle() {
            std::lock_guard<std::mutex> lock( handle_mutex_ );
            if ( !cudnnHandle_ && isCudaDevice() ) {
                cudnnStatus_t status = cudnnCreate( &cudnnHandle_ );
                if ( status != CUDNN_STATUS_SUCCESS ) {
                    throw std::runtime_error( "Failed to create cuDNN handle" );
                }
                // Associate with the current stream
                if ( stream_ ) {
                    status = cudnnSetStream( cudnnHandle_, stream_ );
                    if ( status != CUDNN_STATUS_SUCCESS ) {
                        throw std::runtime_error( "Failed to set cuDNN stream" );
                    }
                }
            }
            return cudnnHandle_;
        }
    #endif

        /**
        * @brief Sets the current device as active in the current thread.
        *
        * This method ensures that subsequent CUDA operations are executed on the correct device
        * by setting the current device in the thread if it's different from the previously set device.
        * The method optimizes performance by tracking the currently active device per thread and
        * avoiding unnecessary device switches.
        *
        * @note This method is thread-safe and optimized for multi-threaded environments.
        * @throws std::runtime_error If setting the CUDA device fails.
        */
        void makeCurrent() const {
            static thread_local int current_device = -1;
            if ( isCudaDevice() && current_device != device_id_ ) {
                cudaError_t error = cudaSetDevice( device_id_ );
                if ( error != cudaSuccess ) {
                    std::string errorMsg = "Failed to set CUDA device " +
                        std::to_string( device_id_ ) + ": " +
                        cudaGetErrorString( error );
                    throw std::runtime_error( errorMsg );
                }
                current_device = device_id_;
            }
        }

        /**
        * @brief Synchronizes the device, waiting for all operations to complete.
        *
        * When using a CUDA device, this method ensures the current device is active and then
        * synchronizes the CUDA stream, waiting for all enqueued operations to complete.
        */
        void synchronize() {
            if ( isCudaDevice() && stream_ ) {
                makeCurrent();
                cudaStreamSynchronize( stream_ );
            }
        }

    private:
        /** @brief The compute device used by this context */
        std::shared_ptr<ComputeDevice> device_;

        /** @brief The CUDA device ID, -1 indicates uninitialized */
        int device_id_ = -1;

        /** @brief The CUDA stream for asynchronous operations */
        cudaStream_t stream_ = nullptr;

        /** @brief Indicates if the stream was created by this context and needs to be destroyed */
        bool stream_created_ = false;

        /** @brief Handle for cuBLASLt operations */
        mutable cublasLtHandle_t cublasLtHandle_ = nullptr;

    #ifdef USE_CUDNN
        /** @brief Handle for cuDNN operations */
        mutable cudnnHandle_t cudnnHandle_ = nullptr;
    #endif

        /** @brief Mutex for thread-safe handle initialization */
        mutable std::mutex handle_mutex_;

        /**
        * @brief Sets the current device by name.
        * @param device_name The name of the device to set.
        * @throws std::runtime_error If the device name is invalid or device initialization fails.
        */
        void setDevice( const std::string& device_name ) {
            releaseResources();

            device_ = DeviceRegistry::instance().createDevice( device_name );

            if ( !device_ ) {
                throw std::runtime_error( "Invalid device name: " + device_name );
            }

            initializeDeviceResources();
        }

        /**
        * @brief Initializes resources specific to the current device.
        *
        * For CUDA devices, this retrieves the device ID, sets the device as current,
        * and creates a CUDA stream.
        */
        void initializeDeviceResources() {
            if ( isCudaDevice() ) {
                // Get the device ID from CudaDevice
                auto cudaDevice = std::dynamic_pointer_cast<CudaDevice>(device_);
                if ( cudaDevice ) {
                    device_id_ = cudaDevice->getDeviceId();
                }

                makeCurrent();
                
                cudaError_t streamErr = cudaStreamCreateWithFlags( &stream_, cudaStreamNonBlocking );
                
                if ( streamErr != cudaSuccess ) {
                    throw std::runtime_error( "Failed to create CUDA stream: " +
                        std::string( cudaGetErrorString( streamErr ) ) );
                }

                stream_created_ = true;
            }
        }

        /**
        * @brief Releases all device-specific resources.
        *
        * Frees CUDA streams and library handles when applicable.
        */
        void releaseResources() {
            // Clean up CUDA resources if necessary
            if ( isCudaDevice() ) {
                if ( stream_created_ && stream_ ) {
                    cudaStreamDestroy( stream_ );
                    stream_ = nullptr;
                    stream_created_ = false;
                }

                if ( cublasLtHandle_ ) {
                    cublasLtDestroy( cublasLtHandle_ );
                    cublasLtHandle_ = nullptr;
                }
            #ifdef USE_CUDNN
                if ( cudnnHandle_ ) {
                    cudnnDestroy( cudnnHandle_ );
                    cudnnHandle_ = nullptr;
                }
            #endif
            }
        }

        /**
        * @brief Moves resources from another DeviceContext.
        * @param other The DeviceContext to move resources from.
        */
        void moveFrom( DeviceContext&& other ) {
            device_ = std::move( other.device_ );
            device_id_ = other.device_id_;
            stream_ = other.stream_;
            stream_created_ = other.stream_created_;
            cublasLtHandle_ = other.cublasLtHandle_;
        #ifdef USE_CUDNN
            cudnnHandle_ = other.cudnnHandle_;
        #endif

            other.stream_ = nullptr;
            other.stream_created_ = false;
            other.cublasLtHandle_ = nullptr;
        #ifdef USE_CUDNN
            other.cudnnHandle_ = nullptr;
        #endif
        }
    };
}
