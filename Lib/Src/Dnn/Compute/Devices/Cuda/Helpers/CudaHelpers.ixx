/**
 * @file CudaHelpers.ixx
 * @brief CUDA utility functions for device management and kernel execution
 *
 * Binding a thread to a device, the runtime's versions and device count, and grid arithmetic. Choosing a device is
 * the deployment planner's (Specifications/Deployment.md 12.8, 12.9).
 */

module;
#include <cuda_runtime.h>

export module Cuda.Helpers;

import std;
import Cuda.Error;

namespace Mila::Dnn::Compute::Cuda
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
     * @brief Bind the calling thread to a CUDA device.
     *
     * The current device is per-thread global state, so anything that allocates or launches
     * must state which device it means rather than inherit whatever was set last. On a
     * multi-GPU host the difference is not academic: a stream created under one device and
     * pointers allocated under another produce an illegal memory access on first launch.
     *
     * @param device_id CUDA device ID to activate
     * @throws std::invalid_argument If device_id is negative
     * @throws std::runtime_error If cudaSetDevice fails
     */
    export inline void setCurrentDevice( int device_id )
    {
        // NOTE: the wrapper earns its keep as the single place the contract above is stated;
        // an earlier comment here described thread-local caching that was never implemented.
        if ( device_id < 0 )
        {
            throw std::invalid_argument( "Invalid CUDA device id: must be non-negative." );
        }

        //assert( device_id >= 0 && "Invalid CUDA device id" );

        cudaError_t error = cudaSetDevice( device_id );

        if ( error != cudaSuccess )
        {
            throw std::runtime_error(
                std::format( "Failed to set CUDA device {}: {}", device_id, cudaGetErrorString( error ) )
            );

        }
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
}
