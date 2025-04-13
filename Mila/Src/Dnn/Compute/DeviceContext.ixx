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
 * - CUDA graph capture and replay for optimized execution of recurring operations
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

        // Disable copy, enable move
        DeviceContext( const DeviceContext& ) = delete;
        DeviceContext& operator=( const DeviceContext& ) = delete;

        DeviceContext( DeviceContext&& other ) noexcept {
            moveFrom( std::move( other ) );
        }

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
        * @param type The device type to check against
        * @return bool True if the device matches the specified type
        */
        bool isDeviceType( DeviceType type ) const {
            return device_ && device_->getDeviceType() == type;
        }

        /**
         * @brief Checks if the current device is a CUDA device.
         *
         * @return bool True if the device is a CUDA device
         */
        bool isCudaDevice() const {
            return isDeviceType( DeviceType::Cuda );
        }

        /**
        * @brief Gets the current device.
        * @return A shared pointer to the current device.
        */
        std::shared_ptr<ComputeDevice> getDevice() const {
            return device_;
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
        */
        cublasLtHandle_t getCublasLtHandle() {
            std::lock_guard<std::mutex> lock( handle_mutex_ );
            if ( !cublasLtHandle_ && isCudaDevice() ) {
                cublasLtCreate( &cublasLtHandle_ );
            }
            return cublasLtHandle_;
        }

        /**
        * @brief Gets the cuDNN handle, initializing it if necessary.
        * @return The cuDNN handle.
        */
	#ifdef USE_CUDNN
        cudnnHandle_t getCudnnHandle() {
            std::lock_guard<std::mutex> lock( handle_mutex_ );
            if ( !cudnnHandle_ && isCudaDevice() ) {
                cudnnCreate( &cudnnHandle_ );
            }
            return cudnnHandle_;
        }
	#endif

        /**
        * @brief Synchronizes the device, waiting for all operations to complete.
        */
        void synchronize() {
            if ( isCudaDevice() && stream_ ) {
                cudaStreamSynchronize( stream_ );
            }
        }

    private:
        std::shared_ptr<ComputeDevice> device_;
        cudaStream_t stream_ = nullptr;
        bool stream_created_ = false;

        // Optional handles
        mutable cublasLtHandle_t cublasLtHandle_ = nullptr;
	#ifdef USE_CUDNN
        mutable cudnnHandle_t cudnnHandle_ = nullptr;
	#endif
        mutable std::mutex handle_mutex_;

        /**
        * @brief Sets the current device by name.
        * @param device_name The name of the device to set.
        * @throws std::runtime_error if the device name is invalid.
        */
        void setDevice( const std::string& device_name ) {
            // Clean up old resources
            releaseResources();

            device_ = DeviceRegistry::instance().createDevice( device_name );

            if ( !device_ ) {
                throw std::runtime_error( "Invalid device name: " + device_name );
            }

            initializeDeviceResources();
        }

        /**
        * @brief Initializes resources specific to the current device.
        */
        void initializeDeviceResources() {
            if ( isCudaDevice() ) {
                cudaStreamCreate( &stream_ );
                stream_created_ = true;
            }
        }

        /**
        * @brief Releases all device-specific resources.
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
        * @param other The other DeviceContext to move from.
        */
        void moveFrom( DeviceContext&& other ) {
            device_ = std::move( other.device_ );
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
