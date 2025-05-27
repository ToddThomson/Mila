/**
 * @file CudaHelpers.ixx
 * @brief CUDA utility functions for device management and kernel execution
 *
 * Provides helper functions for CUDA device discovery, selection,
 * and kernel execution optimization. These utilities simplify working
 * with CUDA devices and provide consistent error handling.
 */

module;
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>

export module Cuda.Helpers;

import Cuda.Error;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Calculates ceiling division for kernel grid/block dimensions
     * @param M Dividend value
     * @param N Divisor value
     * @return Ceiling of M/N as an integer
     */
    export constexpr int ceil_div( int M, int N ) {
        return (M + N - 1) / N;
    }

    /**
     * @brief Gets the installed CUDA driver version
     * @return Integer representation of the CUDA driver version
     * @throws CudaError If driver version cannot be determined
     */
    export int getDriverVersion() {
        int driverVersion;
        cudaCheckStatus( cudaDriverGetVersion( &driverVersion ) );
        return driverVersion;
    };

    /**
     * @brief Gets the installed CUDA runtime version
     * @return Integer representation of the CUDA runtime version
     * @throws CudaError If runtime version cannot be determined
     */
    export int getRuntimeVersion() {
        int runtimeVersion;
        cudaCheckStatus( cudaRuntimeGetVersion( &runtimeVersion ) );
        return runtimeVersion;
    };

    /**
     * @brief Gets the number of available CUDA devices
     * @return Number of CUDA devices available to the application
     * @throws CudaError If device enumeration fails
     */
    export inline int getDeviceCount() {
        int devCount;
        cudaCheckStatus( cudaGetDeviceCount( &devCount ) );
        return devCount;
    };

    /**
     * @brief Validates that a device ID is valid and available
     * @param deviceId CUDA device ID to check
     * @return The same device ID if valid
     * @throws std::invalid_argument If device ID is negative
     * @throws std::runtime_error If no CUDA devices are available
     * @throws std::out_of_range If device ID exceeds available device count
     * @throws std::runtime_error If device is in prohibited compute mode
     */
    export int checkDevice( int deviceId ) {
        if ( deviceId < 0 ) {
            throw std::invalid_argument( "Invalid device id." );
        }

        int devCount = getDeviceCount();

        if ( devCount == 0 ) {
            throw std::runtime_error( "No CUDA devices found." );
        }

        if ( deviceId > devCount - 1 ) {
            throw std::out_of_range( "Device id out of range." );
        }

        int computeMode = -1;
        cudaCheckStatus( cudaDeviceGetAttribute( &computeMode, cudaDevAttrComputeMode, deviceId ) );

        if ( computeMode == cudaComputeModeProhibited ) {
            throw std::runtime_error( "Device is running in Compute ModeProhibited." );
        }

        return deviceId;
    };

    /**
     * @brief Identifies the best CUDA device based on performance characteristics
     *
     * Evaluates available CUDA devices and selects the one with highest performance
     * potential. Selection criteria vary based on the intended workload type.
     *
     * @param preferMemory When true, prioritizes memory bandwidth over compute
     * @return Device ID of the best available CUDA device
     * @throws CudaError If device properties cannot be accessed
     */
    export inline int getBestDeviceId( bool preferMemory = false ) {
        int deviceCount = getDeviceCount();
        int bestDevice = 0;
        uint64_t bestScore = 0;

        for ( int device = 0; device < deviceCount; device++ ) {
            cudaDeviceProp props;
            if ( cudaGetDeviceProperties( &props, device ) != cudaSuccess ) continue;
            if ( props.computeMode == cudaComputeModeProhibited ) continue;

            uint64_t score;
            if ( preferMemory ) {
                score = (uint64_t)props.totalGlobalMem * props.memoryBusWidth;
            }
            else {
                score = (uint64_t)props.multiProcessorCount * props.maxThreadsPerMultiProcessor *
                    props.clockRate * (props.major >= 7 ? 10 : 1);
            }

            if ( score > bestScore ) {
                bestScore = score;
                bestDevice = device;
            }
        }

        return bestDevice;
    }

    /**
     * @brief Finds the most appropriate CUDA device for computation
     *
     * Either validates a specific device ID if provided or finds
     * the best available device when no preference is specified.
     *
     * @param deviceId Preferred device ID, or -1 to select the best device
     * @return Valid CUDA device ID
     * @throws std::runtime_error If no CUDA devices are found
     */
    export inline int findCudaDevice( int deviceId = -1, bool preferMemory = false ) {
        int device_count = getDeviceCount();

        if ( device_count == 0 ) {
            throw std::runtime_error( "No CUDA devices found." );
        }

        if ( deviceId < 0 ) {
            return getBestDeviceId( preferMemory );
        }

        return checkDevice( deviceId );
    };
}