/**
 * @file CudaDevice.ixx
 * @brief Implementation of CUDA-based compute device for the Mila framework.
 */

module;
#include <cuda_runtime.h>
#include <string>
#include <format>
#include <stdexcept>
#include <utility>
#include <memory>
#include <exception>

export module Compute.CudaDevice;

import Compute.Device;
import Compute.DeviceId;
import Compute.DeviceType;
import Compute.CudaDeviceResources;
import Compute.CudaDeviceProps;
import Compute.DeviceRegistry;
import Cuda.Error;
import Utils.Logger;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Class representing a CUDA compute device instance.
     *
     * Provides an interface to interact with a specific NVIDIA CUDA-capable GPU.
     * Handles device properties and capabilities for a single device instance.
     *
     * Device instances are created exclusively by DeviceFactory (via DeviceRegistry).
     * Users should obtain devices through DeviceRegistry::getDevice().
     *
     * Precision Support:
     * - FP32: All CUDA devices (SM 1.0+)
     * - FP16: Pascal and newer (SM 6.0+)
     * - BF16: Ampere and newer (SM 8.0+)
     * - FP8: Hopper and newer (SM 9.0+)
     * - INT8: Turing and newer (SM 7.5+)
     */
    export class CudaDevice : public Device
    {

    public:

        /**
         * @brief Construct CUDA device with validation.
         *
         * Validates that the device ID is registered with DeviceRegistry and
         * queries/caches device properties from CUDA runtime.
         *
         * @param key Construction key ensuring only DeviceRegistry can create instances
         * @param device_id Device identifier to initialize
         * @throws std::invalid_argument If device_id validation fails
         * @throws std::runtime_error If device is not registered or CUDA operations fail
         */
        explicit CudaDevice( DeviceConstructionKey key, DeviceId device_id )
            : device_id_( validateDeviceId( device_id ) ), props_( device_id.index )
        {
            (void)key;
        }

        /**
         * @brief Gets the device identifier.
         *
         * @return DeviceId The identifier for this CUDA device (type + index).
         */
        DeviceId getDeviceId() const override
        {
            return device_id_;
        }

        /**
         * @brief Gets the device type.
         *
         * @return DeviceType The device type (Cuda).
         */
        constexpr DeviceType getDeviceType() const override
        {
            return DeviceType::Cuda;
        }

        /**
         * @brief Gets the device name.
         *
         * @return std::string The device name (e.g., "CUDA:0", "CUDA:1").
         */
        std::string getDeviceName() const override
        {
            return device_id_.toString();
        }

        /**
         * @brief Gets the properties of this CUDA device.
         *
         * @return const CudaDeviceProps& Reference to the device properties.
         */
        const CudaDeviceProps& getProperties() const
        {
            return props_;
        }

        /**
         * @brief Gets the compute capability version.
         *
         * @return std::pair<int, int> Major and minor version (e.g., {8, 6} for SM 8.6).
         */
        std::pair<int, int> getComputeCapability() const
        {
            return { props_.major, props_.minor };
        }

        /**
         * @brief Gets the compute capability as a single number.
         *
         * @return int Compute capability (e.g., 86 for SM 8.6).
         */
        int getComputeCapabilityVersion() const
        {
            return props_.major * 10 + props_.minor;
        }

        /**
         * @brief Checks if the device supports FP16 (half precision).
         *
         * FP16 is supported on Pascal and newer architectures (SM 6.0+).
         *
         * @return bool True if FP16 is supported.
         */
        bool isFp16Supported() const
        {
            return props_.major >= 6;
        }

        /**
         * @brief Checks if the device supports BF16 (bfloat16 precision).
         *
         * BF16 is supported on Ampere and newer architectures (SM 8.0+).
         *
         * @return bool True if BF16 is supported.
         */
        bool isBf16Supported() const
        {
            return props_.major >= 8;
        }

        /**
         * @brief Checks if the device supports FP8 (8-bit float precision).
         *
         * FP8 is supported on Hopper and newer architectures (SM 9.0+).
         *
         * @return bool True if FP8 is supported.
         */
        bool isFp8Supported() const
        {
            return props_.major >= 9;
        }

        /**
         * @brief Checks if the device supports INT8 tensor cores.
         *
         * INT8 tensor cores are supported on Turing and newer (SM 7.5+).
         *
         * @return bool True if INT8 tensor cores are supported.
         */
        bool isInt8Supported() const
        {
            return getComputeCapabilityVersion() >= 75;
        }

        /**
         * @brief Checks if the device has Tensor Cores.
         *
         * Tensor Cores are available on Volta and newer (SM 7.0+).
         *
         * @return bool True if Tensor Cores are available.
         */
        bool hasTensorCores() const
        {
            return props_.major >= 7;
        }

        /**
         * @brief Gets the maximum number of threads per block.
         *
         * @return int Maximum threads per block.
         */
        int getMaxThreadsPerBlock() const
        {
            return props_.maxThreadsPerBlock;
        }

        /**
         * @brief Gets the total global memory size in bytes.
         *
         * @return size_t Total global memory in bytes.
         */
        size_t getTotalGlobalMemory() const
        {
            return props_.totalGlobalMem;
        }

        /**
         * @brief Gets the shared memory per block in bytes.
         *
         * @return size_t Shared memory per block in bytes.
         */
        size_t getSharedMemoryPerBlock() const
        {
            return props_.sharedMemPerBlock;
        }

        /**
         * @brief Gets the number of multiprocessors.
         *
         * @return int Number of streaming multiprocessors.
         */
        int getMultiprocessorCount() const
        {
            return props_.multiProcessorCount;
        }

        /**
         * @brief Gets the warp size.
         *
         * @return int Warp size (typically 32).
         */
        int getWarpSize() const
        {
            return props_.warpSize;
        }

    private:

        DeviceId device_id_;
        CudaDeviceProps props_;

        /**
         * @brief Validates CUDA device ID.
         *
         * Ensures device_id has correct type (Cuda), non-negative index,
         * and is within the range of available CUDA devices.
         *
         * @param device_id Device identifier to validate.
         * @return DeviceId The validated device identifier.
         * @throws std::invalid_argument If device_id type is not Cuda or index is negative.
         * @throws std::runtime_error If CUDA device count query fails or index is out of range.
         */
        static DeviceId validateDeviceId( DeviceId device_id )
        {
			// REVIEW: DeviceRegistry ensures only registered devices are constructed so this validation
			// is redundant, but kept for defense in depth for now.
            if ( device_id.type != DeviceType::Cuda )
            {
                throw std::invalid_argument(
                    "CudaDevice requires Cuda device type, got: " +
                    deviceTypeToString( device_id.type )
                );
            }

            if ( device_id.index < 0 )
            {
                throw std::invalid_argument(
                    "CUDA device index must be non-negative, got: " +
                    std::to_string( device_id.index )
                );
            }

            return device_id;
        }
    };

    /**
     * @brief CUDA device plugin for device-agnostic registration.
     *
     * Encapsulates CUDA-specific logic for device discovery and registration,
     * providing a clean static interface while handling CUDA runtime API
     * interactions internally.
     */
    export class CudaDeviceRegistrar
    {
    public:

        /**
         * @brief Register CUDA support with the DeviceRegistry.
         *
         * Registers discovered CUDA devices with the global DeviceRegistry.
         * If no devices are present, a warning is emitted and registration
         * is skipped.
         */
        static void registerDevices()
        {
            auto& registry = DeviceRegistry::instance();

            int count = getDeviceCount();

            if ( count == 0 )
            {
                Utils::Logger::warning( "CudaDeviceRegistrar: CUDA not available or no usable devices" );
                return;
            }

            for ( int i = 0; i < count; ++i )
            {
                if ( !isDeviceUsable( i ) )
                {
                    Utils::Logger::warning( std::format( "CudaDeviceRegistrar: CUDA device index '{}' is not useable", i ) );
                    
                    continue;
                }
                
                try
                {
                    DeviceId device_id = Device::Cuda( i );

                    registry.registerDevice( device_id, [device_id]( DeviceConstructionKey key ) -> std::shared_ptr<Device> {
                        return std::make_shared<CudaDevice>( key, device_id );
                        } );
                }
                catch ( const std::exception& ex )
                {
                    Utils::Logger::warning( std::format( "CudaDeviceRegistrar: failed to register CUDA device {}: {}", i, ex.what() ) );
                }
                catch ( ... )
                {
                    Utils::Logger::warning( std::format( "CudaDeviceRegistrar: failed to register CUDA device {}", i ) );
                }
            }
        }

        /**
         * @brief Gets the number of available CUDA devices.
         *
         * @return Number of CUDA devices available, or 0 if CUDA is not available
         */
         static int getDeviceCount() {
             try {
                 int deviceCount = 0;
                 cudaError_t error = cudaGetDeviceCount( &deviceCount );

                 return (error == cudaSuccess) ? deviceCount : 0;
             }
             catch (...) {
                 return 0;
             }
         }

    private:
        /**
         * @brief Checks if a specific CUDA device is usable for computation.
         *
         * Ensures device meets minimal compute capability and memory requirements
         * and that it can be selected via the CUDA runtime.
         */
        static bool isDeviceUsable( int deviceId )
        {
            try
            {
                cudaDeviceProp deviceProp;
                cudaError_t error = cudaGetDeviceProperties( &deviceProp, deviceId );

                if ( error != cudaSuccess )
                {
                    return false;
                }

                if ( deviceProp.major < 3 )
                {
                    return false;
                }

                if ( deviceProp.totalGlobalMem < (1ULL << 30) )
                {
                    return false;
                }

                int computeMode = -1;
                error = cudaDeviceGetAttribute( &computeMode, cudaDevAttrComputeMode, deviceId );

                if ( computeMode == cudaComputeModeProhibited )
                {
                    return false;
                }

                cudaError_t setError = cudaSetDevice( deviceId );
                if ( setError != cudaSuccess )
                {
                    return false;
                }

                cudaDeviceReset();

                return true;
            }
            catch ( ... )
            {
                return false;
            }
        }
    };
}