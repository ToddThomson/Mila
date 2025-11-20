/**
 * @file CudaExecutionContext.ixx
 * @brief CUDA-specific execution context specialization.
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
import Compute.IExecutionContext;
import Compute.ComputeDevice;
import Compute.CudaDevice;
import Compute.CudaDeviceResources;
import Compute.DeviceType;
import Compute.DeviceRegistry;
import Cuda.Helpers;

namespace Mila::Dnn::Compute
{
    /**
     * @brief CUDA execution context specialization.
     *
     * Manages CUDA execution resources including streams and library handles.
     * Each context owns an independent CUDA stream for asynchronous execution and
     * reuses device-level resources exposed by CudaDeviceResources.
     */
    export template<>
    class ExecutionContext<DeviceType::Cuda> : public IExecutionContext
    {
    public:
        /**
         * @brief Constructs CUDA execution context for a specific device.
         *
         * Obtains the device through DeviceRegistry to ensure shared device instances
         * are reused rather than creating a new CudaDevice directly.
         *
         * @param device_id CUDA device ID (0-based)
         * @throws std::invalid_argument If device_id is invalid
         * @throws std::runtime_error If CUDA stream creation fails
         */
        explicit ExecutionContext( int device_id )
            : ExecutionContext( DeviceRegistry::instance().getDevice( std::string("CUDA:") + std::to_string(device_id) ) ) {
        }

        /**
         * @brief Constructs CUDA execution context from existing device.
         *
         * Shares device ownership and creates independent execution resources.
         *
         * @param device CUDA device instance
         * @throws std::invalid_argument If device is null or not CUDA
         * @throws std::runtime_error If CUDA stream creation fails
         */
        explicit ExecutionContext( std::shared_ptr<ComputeDevice> device )
            : IExecutionContext( DeviceType::Cuda ), device_( validateDevice( device ) ) {

            resources_ = std::make_shared<CudaDeviceResources>( device_->getDeviceId() );

            initializeResources();
        }

        /**
         * @brief Destructor with proper CUDA resource cleanup.
         */
        ~ExecutionContext() {
        }

        // Non-copyable, non-movable (owns CUDA resources)
        ExecutionContext( const ExecutionContext& ) = delete;
        ExecutionContext& operator=( const ExecutionContext& ) = delete;
        ExecutionContext( ExecutionContext&& ) = delete;
        ExecutionContext& operator=( ExecutionContext&& ) = delete;

        /**
         * @brief Synchronizes the CUDA stream.
         *
         * Blocks until all operations submitted to this context's stream complete.
         *
         * @throws std::runtime_error If stream synchronization fails
         */
        void synchronize() {
            /*if (!stream_)
            {
                return;
            }*/

            Cuda::setCurrentDevice( device_->getDeviceId() );

            cudaError_t error = cudaStreamSynchronize( stream_ );
            
            if (error != cudaSuccess)
            {
                throw std::runtime_error(
                    "CUDA stream synchronization failed: " + std::string( cudaGetErrorString( error ) )
                );
            }
        }

        /**
         * @brief Gets the associated CUDA device.
         */
        std::shared_ptr<ComputeDevice> getDevice() const {
            return device_;
        }

        /**
         * @brief Gets the device type (always Cuda).
         */
        static constexpr DeviceType getDeviceType() {
            return DeviceType::Cuda;
        }

        /**
         * @brief Gets the device name (e.g., "CUDA:0").
         */
        std::string getDeviceName() const {
            return device_->getDeviceName();
        }

        /**
         * @brief Gets the CUDA device ID.
         */
        int getDeviceId() const {
            return device_->getDeviceId();
        }

        /**
         * @brief Checks if this is a CUDA device (always true).
         */
        static constexpr bool isCudaDevice() {
            return true;
        }

        /**
         * @brief Checks if this is a CPU device (always false).
         */
        static constexpr bool isCpuDevice() {
            return false;
        }

        /**
         * @brief Gets the CUDA stream for asynchronous operations.
         */
        cudaStream_t getStream() const {
            return stream_;
        }

        /**
         * @brief Returns the shared device-level resources backing this context.
         */
        std::shared_ptr<CudaDeviceResources> getResources() const { return resources_; }

        /**
         * @brief Get a cuBLASLt handle bound to this context's stream.
         *
         * The handle is provided by the device resources; we bind it to the
         * context stream just prior to use to guarantee correct stream ordering.
         *
         * Note: binding the same handle concurrently to different streams is
         * a race unless callers serialize. For concurrent multi-stream use,
         * prefer per-context or per-thread handles created via device resources.
         */
        cublasLtHandle_t getCublasLtHandle() const {
            return resources_->getCublasLtHandle();
        }

#ifdef USE_CUDNN
        /**
         * @brief Get a cuDNN handle bound to this context's stream.
         *
         * Similar semantics/notes as getCublasLtHandle().
         */
        cudnnHandle_t getCudnnHandle() const {
            auto handle = resources_->getCudnnHandle();
            if (handle && stream_) {
                cudnnStatus_t status = cudnnSetStream( handle, stream_ );
                if (status != CUDNN_STATUS_SUCCESS) {
                    throw std::runtime_error( "Failed to set cuDNN handle stream." );
                }
            }
            return handle;
        }
#endif

    private:

        std::shared_ptr<ComputeDevice> device_;
        std::shared_ptr<CudaDeviceResources> resources_;

        cudaStream_t stream_{ nullptr };
        bool stream_created_{ false };

        /**
         * @brief Validates device for construction.
         */
        static std::shared_ptr<ComputeDevice> validateDevice(
            std::shared_ptr<ComputeDevice> device ) {

            if (!device)
            {
                throw std::invalid_argument( "Device cannot be null" );
            }

            if (device->getDeviceType() != DeviceType::Cuda)
            {
                throw std::invalid_argument(
                    "CudaExecutionContext requires CUDA device, got: " +
                    std::string( deviceTypeToString( device->getDeviceType() ) )
                );
            }

            return device;
        }

        /**
         * @brief Initializes CUDA execution resources (stream).
         *
         * Creates non-blocking CUDA stream for asynchronous operations.
         *
         * @throws std::runtime_error If stream creation fails
         */
        void initializeResources() {
            Cuda::setCurrentDevice( device_->getDeviceId() );

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
         * Destroys stream in reverse order of creation.
         * Called by destructor - does not throw exceptions.
         */
        void releaseResources() noexcept {
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