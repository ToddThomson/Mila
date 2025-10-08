/**
 * @file CudaDeviceContext.ixx
 * @brief CUDA-specific device context implementation.
 *
 * Manages CUDA device identification and activation. Execution resources
 * (streams, library handles) are managed by CudaExecutionContext.
 */

module;
#include <memory>
#include <string>
#include <stdexcept>
#include <cuda_runtime.h>

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
     * Manages CUDA device identification and activation. Used for memory
     * allocation operations that require device selection via cudaSetDevice().
     *
     * For compute operations, use CudaExecutionContext which manages streams
     * and library handles (cuBLAS, cuDNN).
     *
     * Responsibilities:
     * - Device identification (device ID, name, type)
     * - Device activation via makeCurrent() for memory allocation
     * - Device property queries
     *
     * Not responsible for:
     * - Stream management (handled by CudaExecutionContext)
     * - Library handles (handled by CudaExecutionContext)
     * - Execution synchronization (handled by CudaExecutionContext)
     */
    //export class CudaDeviceContext : public DeviceContext {
    //public:
    //    /**
    //     * @brief Constructor with CUDA device name.
    //     *
    //     * Creates a device context for the specified CUDA device.
    //     * Does NOT create streams or library handles.
    //     *
    //     * @param device_name CUDA device identifier (e.g., "CUDA:0")
    //     * @throws std::runtime_error If device initialization fails
    //     */
    //    explicit CudaDeviceContext( const std::string& device_name ) {
    //        device_ = DeviceRegistry::instance().createDevice( device_name );

    //        if (!device_ || device_->getDeviceType() != DeviceType::Cuda) {
    //            throw std::runtime_error( "Invalid CUDA device name: " + device_name );
    //        }

    //        auto cudaDevice = std::dynamic_pointer_cast<CudaDevice>(device_);
    //        if (cudaDevice) {
    //            device_id_ = cudaDevice->getDeviceId();
    //        }
    //    }

    //    /**
    //     * @brief Destructor.
    //     *
    //     * No CUDA resources to clean up - streams and handles are managed
    //     * by CudaExecutionContext.
    //     */
    //    ~CudaDeviceContext() override = default;

    //    // DeviceContext interface implementation

    //    /**
    //     * @brief Gets the device type.
    //     * @return DeviceType::Cuda
    //     */
    //    DeviceType getDeviceType() const override {
    //        return DeviceType::Cuda;
    //    }

    //    /**
    //     * @brief Gets the device name.
    //     * @return Device name string (e.g., "CUDA:0")
    //     */
    //    std::string getDeviceName() const override {
    //        return device_ ? device_->getDeviceName() : "CUDA:INVALID";
    //    }

    //    /**
    //     * @brief Gets the CUDA device ID.
    //     * @return CUDA device ID (0, 1, 2, etc.)
    //     */
    //    int getDeviceId() const override {
    //        return device_id_;
    //    }

    //    /**
    //     * @brief Gets the compute device.
    //     * @return Shared pointer to ComputeDevice
    //     */
    //    std::shared_ptr<ComputeDevice> getDevice() const override {
    //        return device_;
    //    }

    //    // CUDA-specific methods

    //    /**
    //     * @brief Gets the compute capability of this device.
    //     * @return Pair of (major, minor) version numbers
    //     */
    //    std::pair<int, int> getComputeCapability() const {
    //        auto cudaDevice = std::dynamic_pointer_cast<CudaDevice>(device_);
    //        return cudaDevice ?
    //            cudaDevice->getProperties().getComputeCapability() :
    //            std::make_pair( 0, 0 );
    //    }

    //private:
    //    std::shared_ptr<ComputeDevice> device_;
    //    int device_id_ = -1;
    //};
}