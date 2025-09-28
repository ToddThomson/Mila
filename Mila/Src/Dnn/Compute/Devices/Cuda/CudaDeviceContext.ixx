/**
 * @file CudaDeviceContext.ixx
 * @brief CUDA-specific device context implementation.
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

export module Compute.CudaDeviceContext;

import Compute.DeviceContext;
import Compute.ComputeDevice;
import Compute.CudaDevice;
import Compute.DeviceRegistry;
import Compute.DeviceType;

namespace Mila::Dnn::Compute
{
    /**
     * @brief CUDA-specific device context implementation.
     *
     * Manages CUDA devices and their associated resources including streams,
     * cuBLAS handles, and cuDNN handles. Optimized for NVIDIA GPU operations
     * and multi-GPU environments.
     */
    export class CudaDeviceContext : public DeviceContext {
    public:
        /**
         * @brief Constructor with CUDA device name.
         * @param device_name CUDA device identifier (e.g., "CUDA:0").
         * @throws std::runtime_error If device initialization fails.
         */
        explicit CudaDeviceContext( const std::string& device_name ) {
            device_ = DeviceRegistry::instance().createDevice( device_name );

            if (!device_ || device_->getDeviceType() != DeviceType::Cuda) {
                throw std::runtime_error( "Invalid CUDA device name: " + device_name );
            }

            auto cudaDevice = std::dynamic_pointer_cast<CudaDevice>(device_);

            if (cudaDevice) {
                device_id_ = cudaDevice->getDeviceId();
            }

            initializeDeviceResources();
        }

        /**
         * @brief Destructor with CUDA resource cleanup.
         */
        ~CudaDeviceContext() override {
            releaseResources();
        }

        // DeviceContext interface implementation
        DeviceType getDeviceType() const override {
            return DeviceType::Cuda;
        }

        std::string getDeviceName() const override {
            return device_ ? device_->getName() : "CUDA:INVALID";
        }

        int getDeviceId() const override {
            return device_id_;
        }

        void makeCurrent() override {
            static thread_local int current_device = -1;
            if (current_device != device_id_) {
                cudaError_t error = cudaSetDevice(device_id_);
                if (error != cudaSuccess) {
                    throw std::runtime_error("Failed to set CUDA device " +
                        std::to_string(device_id_) + ": " + cudaGetErrorString(error));
                }
                current_device = device_id_;
            }
        }

        void synchronize() override {
            if (stream_) {
                makeCurrent();
                cudaStreamSynchronize(stream_);
            }
        }

        std::shared_ptr<ComputeDevice> getDevice() const override {
            return device_;
        }

        // CUDA-specific methods
        cudaStream_t getStream() const {
            return stream_;
        }

        cublasLtHandle_t getCublasLtHandle() {
            std::lock_guard<std::mutex> lock(handle_mutex_);
            if (!cublasLtHandle_) {
                makeCurrent();

                cublasStatus_t status = cublasLtCreate(&cublasLtHandle_);
                
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
                    throw std::runtime_error(std::string("Failed to create cuBLASLt handle: ") + errorMsg);
                }
            }
            return cublasLtHandle_;
        }

#ifdef USE_CUDNN
        cudnnHandle_t getCudnnHandle() {
            std::lock_guard<std::mutex> lock(handle_mutex_);
            if (!cudnnHandle_) {
                cudnnStatus_t status = cudnnCreate(&cudnnHandle_);
                if (status != CUDNN_STATUS_SUCCESS) {
                    throw std::runtime_error("Failed to create cuDNN handle");
                }
                if (stream_) {
                    status = cudnnSetStream(cudnnHandle_, stream_);
                    if (status != CUDNN_STATUS_SUCCESS) {
                        throw std::runtime_error("Failed to set cuDNN stream");
                    }
                }
            }
            return cudnnHandle_;
        }
#endif

        std::pair<int, int> getComputeCapability() const {
            auto cudaDevice = std::dynamic_pointer_cast<CudaDevice>(device_);
            return cudaDevice ? cudaDevice->getProperties().getComputeCapability() : std::make_pair(0, 0);
        }

    private:
        std::shared_ptr<ComputeDevice> device_;
        int device_id_ = -1;
        cudaStream_t stream_ = nullptr;
        bool stream_created_ = false;
        mutable cublasLtHandle_t cublasLtHandle_ = nullptr;
#ifdef USE_CUDNN
        mutable cudnnHandle_t cudnnHandle_ = nullptr;
#endif
        mutable std::mutex handle_mutex_;

        void initializeDeviceResources() {
            makeCurrent();

            cudaError_t streamErr = cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
            
            if (streamErr != cudaSuccess) {
                throw std::runtime_error("Failed to create CUDA stream: " +
                    std::string(cudaGetErrorString(streamErr)));
            }
            stream_created_ = true;
        }

        void releaseResources() {
            if (stream_created_ && stream_) {
                cudaStreamDestroy(stream_);
                stream_ = nullptr;
                stream_created_ = false;
            }

            if (cublasLtHandle_) {
                cublasLtDestroy(cublasLtHandle_);
                cublasLtHandle_ = nullptr;
            }

#ifdef USE_CUDNN
            if (cudnnHandle_) {
                cudnnDestroy(cudnnHandle_);
                cudnnHandle_ = nullptr;
            }
#endif
        }
    };
}