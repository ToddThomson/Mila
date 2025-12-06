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

export module Compute.ExecutionContext:Cuda;

import Compute.ExecutionContextTemplate;
import Compute.IExecutionContext;
//import Compute.Device;
import Compute.DeviceId;
import Compute.DeviceType;
//import Compute.CudaDevice;
import Compute.CudaDeviceResources;
//import Compute.DeviceRegistry;
import Cuda.Helpers;

namespace Mila::Dnn::Compute
{
    /**
     * @brief CUDA execution context specialization.
     *
     * Manages CUDA execution resources including streams and library handles.
     * Each context owns an independent CUDA stream for asynchronous execution
     * and shares device-level resources (cuBLAS, cuDNN handles) via CudaDeviceResources.
     *
     * Thread Safety:
     * - Multiple contexts can safely exist on the same device
     * - Each context has its own stream for isolated execution
     * - Shared library handles require external serialization for concurrent use
     */
    export template<>
    class ExecutionContext<DeviceType::Cuda> : public IExecutionContext
    {
    public:
        /**
         * @brief Constructs CUDA execution context for a specific device.
         *
         * Creates an independent CUDA stream and obtains shared device-level
         * resources (library handles).
         *
         * @param device_id CUDA device identifier.
         * @throws std::invalid_argument If device_id type is not Cuda.
         * @throws std::runtime_error If CUDA stream creation fails.
         */
        explicit ExecutionContext( DeviceId device_id )
            : device_id_( validateDeviceId( device_id ) )
        {
            resources_ = std::make_shared<CudaDeviceResources>( device_id_ );

            initializeResources();
        }

        /**
         * @brief Destructor with proper CUDA resource cleanup.
         *
         * Destroys the CUDA stream. Shared device resources are released when
         * the last reference is dropped.
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
            Cuda::setCurrentDevice( device_id_.index );
            
            cudaError_t error = cudaStreamSynchronize( stream_ );

            if ( error != cudaSuccess )
            {
                throw std::runtime_error(
                    std::format( "CUDA stream synchronization failed: {}",
                                cudaGetErrorString( error ) )
                );
            }
        }

        /**
         * @brief Gets the CUDA stream for asynchronous operations.
         *
         * @return cudaStream_t The CUDA stream owned by this context.
         */
        /*[[nodiscard]] cudaStream_t getStream() const noexcept
        {
            return stream_;
        }*/

        /**
         * @brief Returns the shared device-level resources backing this context.
         *
         * @return std::shared_ptr<CudaDeviceResources> Shared device resources.
         */
        /*[[nodiscard]] std::shared_ptr<CudaDeviceResources> getResources() const noexcept
        {
            return resources_;
        }
*/
        /**
         * @brief Get a cuBLASLt handle for this context.
         *
         * The handle is shared across all contexts on the same device.
         * Callers must bind it to this context's stream before use via
         * cublasLtSetStream() if stream ordering is required.
         *
         * @return cublasLtHandle_t The cuBLASLt handle.
         */
        /*[[nodiscard]] cublasLtHandle_t getCublasLtHandle() const
        {
            return resources_->getCublasLtHandle();
        }*/

#ifdef USE_CUDNN
        /**
         * @brief Get a cuDNN handle bound to this context's stream.
         *
         * The handle is shared across contexts on the same device but is
         * automatically bound to this context's stream before returning.
         *
         * @return cudnnHandle_t The cuDNN handle bound to this stream.
         * @throws std::runtime_error If stream binding fails.
         */
        [[nodiscard]] cudnnHandle_t getCudnnHandle() const
        {
            auto handle = resources_->getCudnnHandle();

            if ( handle )
            {
                cudnnStatus_t status = cudnnSetStream( handle, stream_ );

                if ( status != CUDNN_STATUS_SUCCESS )
                {
                    throw std::runtime_error(
                        "Failed to bind cuDNN handle to stream"
                    );
                }
            }

            return handle;
        }
#endif

    private:

        DeviceId device_id_;
        std::shared_ptr<CudaDeviceResources> resources_{ nullptr };

        cudaStream_t stream_{ nullptr };
        bool stream_created_{ false };

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
                cudaStreamNonBlocking
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
         * Destroys the CUDA stream. Called by destructor - does not throw.
         */
        void releaseResources() noexcept
        {
            if ( stream_created_ && stream_ )
            {
                cudaStreamDestroy( stream_ );
                stream_ = nullptr;
                stream_created_ = false;
            }
        }
    };

    export using CudaExecutionContext = ExecutionContext<DeviceType::Cuda>;
}