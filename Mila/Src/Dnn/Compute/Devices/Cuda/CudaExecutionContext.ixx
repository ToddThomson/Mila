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
import Compute.ComputeDevice;
import Compute.CudaDevice;
import Compute.DeviceType;
import Cuda.Helpers;

namespace Mila::Dnn::Compute
{
    /**
     * @brief CUDA execution context specialization.
     *
     * Manages CUDA execution resources including streams and library handles.
     */
    export template<>
        class ExecutionContext<DeviceType::Cuda> {
        public:
            /**
             * @brief Constructs CUDA execution context for a specific device.
             *
             * @param device_id CUDA device ID (0-based)
             * @throws std::invalid_argument If device_id is invalid
             * @throws std::runtime_error If CUDA stream creation fails
             */
            explicit ExecutionContext( int device_id )
                : device_( std::make_shared<CudaDevice>( device_id ) ),
                device_id_( device_id ) {

                if (device_id_ < 0) {
                    throw std::invalid_argument( "Device ID must be non-negative" );
                }

                initializeExecutionResources();
            }

            /**
             * @brief Constructs CUDA execution context from device.
             *
             * @param device CUDA device instance
             * @throws std::invalid_argument If device is null or not CUDA
             * @throws std::runtime_error If CUDA stream creation fails
             */
            explicit ExecutionContext( std::shared_ptr<ComputeDevice> device )
                : device_( device ),
                device_id_( device ? device->getDeviceId() : -1 ) {

                if (!device_) {
                    throw std::invalid_argument( "Device cannot be null" );
                }

                if (device_->getDeviceType() != DeviceType::Cuda) {
                    throw std::invalid_argument(
                        "CudaExecutionContext requires CUDA device"
                    );
                }

                initializeExecutionResources();
            }

            ~ExecutionContext() {
                releaseResources();
            }

            ExecutionContext( const ExecutionContext& ) = delete;
            ExecutionContext& operator=( const ExecutionContext& ) = delete;
            ExecutionContext( ExecutionContext&& ) = delete;
            ExecutionContext& operator=( ExecutionContext&& ) = delete;

            /**
             * @brief Synchronizes the CUDA stream.
             */
            void synchronize() {
                if (stream_) {
                    Cuda::setCurrentDevice( device_id_ );
                    cudaError_t error = cudaStreamSynchronize( stream_ );
                    if (error != cudaSuccess) {
                        throw std::runtime_error(
                            "CUDA stream synchronization failed: " +
                            std::string( cudaGetErrorString( error ) )
                        );
                    }
                }
            }

            /**
             * @brief Gets the CUDA device.
             */
            std::shared_ptr<ComputeDevice> getDevice() const {
                return device_;
            }

            /**
             * @brief Gets the device type (always CUDA).
             */
            static constexpr DeviceType getDeviceType() {
                return DeviceType::Cuda;
            }

            /**
             * @brief Gets the device name.
             */
            std::string getDeviceName() const {
                return device_->getDeviceName();
            }

            /**
             * @brief Gets the CUDA device ID.
             */
            int getDeviceId() const {
                return device_id_;
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
             * @brief Gets the CUDA stream.
             */
            cudaStream_t getStream() const {
                return stream_;
            }

            /**
             * @brief Gets the cuBLAS handle.
             */
            cublasLtHandle_t getCublasLtHandle() {
                std::lock_guard<std::mutex> lock( handle_mutex_ );

                if (!cublasLtHandle_) {
                    Cuda::setCurrentDevice( device_id_ );

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
            cudnnHandle_t getCudnnHandle() {
                std::lock_guard<std::mutex> lock( handle_mutex_ );

                if (!cudnnHandle_) {
                    Cuda::setCurrentDevice( device_id_ );

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
            std::shared_ptr<ComputeDevice> device_;
            int device_id_{ -1 };
            cudaStream_t stream_ = nullptr;
            bool stream_created_ = false;
            mutable cublasLtHandle_t cublasLtHandle_ = nullptr;
#ifdef USE_CUDNN
            mutable cudnnHandle_t cudnnHandle_ = nullptr;
#endif
            mutable std::mutex handle_mutex_;

            void initializeExecutionResources() {
                Cuda::setCurrentDevice( device_id_ );

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

            void releaseResources() noexcept {
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

                if (stream_created_ && stream_) {
                    cudaStreamDestroy( stream_ );
                    stream_ = nullptr;
                    stream_created_ = false;
                }
            }
    };
}