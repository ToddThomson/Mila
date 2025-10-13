/**
 * @file CudaDevice.ixx
 * @brief Implementation of CUDA-based compute device for the Mila framework.
 */

module;
#include <cuda_runtime.h>
#include <string>
#include <stdexcept>
#include <source_location>

export module Compute.CudaDevice;

import Compute.ComputeDevice;
import Compute.DeviceType;
import Compute.CudaDeviceProps;
import Cuda.Error;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Class representing a CUDA compute device instance.
     *
     * Provides an interface to interact with a specific NVIDIA CUDA-capable GPU.
     * Handles device properties and capabilities for a single device instance.
     * Device discovery and registration is handled by CudaDevicePlugin.
     * Device activation and context management is handled by ExecutionContext.
     *
     * Precision Support:
     * - FP32: All CUDA devices (SM 1.0+)
     * - FP16: Pascal and newer (SM 6.0+)
     * - BF16: Ampere and newer (SM 8.0+)
     * - FP8: Hopper and newer (SM 9.0+)
     * - INT8: Turing and newer (SM 7.5+)
     */
    export class CudaDevice : public ComputeDevice 
    {
    public:

        /**
         * @brief Constructs a CudaDevice with specified device ID.
         *
         * Validates the device ID and initializes device properties.
         *
         * @param device_id The CUDA device ID to initialize (0-based)
         * @throws std::invalid_argument If device_id is negative
         * @throws std::runtime_error If device_id exceeds available device count
         */
        explicit CudaDevice( int device_id )
            : device_id_( validateDeviceId( device_id ) ),
            props_( device_id_ ) {
        }

        /**
         * @brief Gets the CUDA device ID.
         * @return int The device ID for this CUDA device (0-based)
         */
        constexpr int getDeviceId() const override {
            return device_id_;
        }

        /**
         * @brief Gets the type of this compute device.
         * @return DeviceType The device type (Cuda)
         */
        constexpr DeviceType getDeviceType() const override {
            return DeviceType::Cuda;
        }

        /**
         * @brief Gets the name of this CUDA device.
         * @return std::string The device name in format "CUDA:N" where N is device_id
         */
        std::string getDeviceName() const override {
            return "CUDA:" + std::to_string( device_id_ );
        }

        /**
         * @brief Gets the properties of this CUDA device.
         * @return const CudaDeviceProps& Reference to the device properties
         */
        const CudaDeviceProps& getProperties() const {
            return props_;
        }

        /**
         * @brief Gets the compute capability version.
         * @return std::pair<int, int> Major and minor version (e.g., {8, 6} for SM 8.6)
         */
        std::pair<int, int> getComputeCapability() const {
            return { props_.major, props_.minor };
        }

        /**
         * @brief Gets the compute capability as a single number.
         * @return int Compute capability (e.g., 86 for SM 8.6)
         */
        int getComputeCapabilityVersion() const {
            return props_.major * 10 + props_.minor;
        }

        /**
         * @brief Checks if the device supports FP16 (half precision).
         *
         * FP16 is supported on Pascal and newer architectures (SM 6.0+).
         *
         * @return bool True if FP16 is supported
         */
        bool isFp16Supported() const {
            return props_.major >= 6;
        }

        /**
         * @brief Checks if the device supports BF16 (bfloat16 precision).
         *
         * BF16 is supported on Ampere and newer architectures (SM 8.0+).
         *
         * @return bool True if BF16 is supported
         */
        bool isBf16Supported() const {
            return props_.major >= 8;
        }

        /**
         * @brief Checks if the device supports FP8 (8-bit float precision).
         *
         * FP8 is supported on Hopper and newer architectures (SM 9.0+).
         *
         * @return bool True if FP8 is supported
         */
        bool isFp8Supported() const {
            return props_.major >= 9;
        }

        /**
         * @brief Checks if the device supports INT8 tensor cores.
         *
         * INT8 tensor cores are supported on Turing and newer (SM 7.5+).
         *
         * @return bool True if INT8 tensor cores are supported
         */
        bool isInt8Supported() const {
            return getComputeCapabilityVersion() >= 75;
        }

        /**
         * @brief Checks if the device has Tensor Cores.
         *
         * Tensor Cores are available on Volta and newer (SM 7.0+).
         *
         * @return bool True if Tensor Cores are available
         */
        bool hasTensorCores() const {
            return props_.major >= 7;
        }

        /**
         * @brief Gets the maximum number of threads per block.
         */
        int getMaxThreadsPerBlock() const {
            return props_.maxThreadsPerBlock;
        }

        /**
         * @brief Gets the total global memory size in bytes.
         */
        size_t getTotalGlobalMemory() const {
            return props_.totalGlobalMem;
        }

        /**
         * @brief Gets the shared memory per block in bytes.
         */
        size_t getSharedMemoryPerBlock() const {
            return props_.sharedMemPerBlock;
        }

        /**
         * @brief Gets the number of multiprocessors.
         */
        int getMultiprocessorCount() const {
            return props_.multiProcessorCount;
        }

        /**
         * @brief Gets the warp size.
         */
        int getWarpSize() const {
            return props_.warpSize;
        }

    private:
        int device_id_;
        CudaDeviceProps props_;

        /**
         * @brief Validates CUDA device ID.
         *
         * Ensures device_id is non-negative and within the range of available devices.
         *
         * @param device_id Device ID to validate
         * @return int The validated device ID
         * @throws std::invalid_argument If device_id is negative
         * @throws std::runtime_error If device_id exceeds device count or CUDA error occurs
         */
        static int validateDeviceId( int device_id ) {
            if (device_id < 0) {
                throw std::invalid_argument(
                    "CUDA device ID must be non-negative, got: " + std::to_string( device_id )
                );
            }

            int device_count = 0;
            cudaError_t error = cudaGetDeviceCount( &device_count );

            if (error != cudaSuccess) {
                throw std::runtime_error(
                    "Failed to get CUDA device count: " +
                    std::string( cudaGetErrorString( error ) )
                );
            }

            if (device_count == 0) {
                throw std::runtime_error( "No CUDA devices available" );
            }

            if (device_id >= device_count) {
                throw std::runtime_error(
                    "CUDA device ID " + std::to_string( device_id ) +
                    " exceeds available device count " + std::to_string( device_count ) +
                    " (valid range: 0-" + std::to_string( device_count - 1 ) + ")"
                );
            }

            return device_id;
        }
    };
}