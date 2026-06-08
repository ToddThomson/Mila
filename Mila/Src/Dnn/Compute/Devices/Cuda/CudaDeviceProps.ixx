/**
 * @file CudaDeviceProps.ixx
 * @brief CUDA device properties wrapper with caching and convenience methods.
 *
 * Provides a type-safe wrapper around cudaDeviceProp with additional
 * convenience methods for querying device capabilities and formatting.
 * Compatible with CUDA 13.0+ which deprecated clockRate fields.
 *
 * cudaDeviceProp is intentionally not stored as a member: the struct is
 * large (100+ fields) and causes an MSVC IFC-writer C1001 ICE when it
 * appears in the exported class layout of any module imported into a
 * module that exports a class template. All needed values are cached as
 * individual members during construction.
 */

module;
#include <string>
#include <iostream>
#include <sstream>
#include <vector>
#include <stdexcept>
#include <format>
#include <cuda_runtime.h>

export module Compute.CudaDeviceProps;

import Cuda.Helpers;
import Cuda.Error;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Wrapper for CUDA device properties with cached values.
     *
     * Queries and caches CUDA device properties on construction. The underlying
     * cudaDeviceProp struct is used only during construction as a local variable
     * and is not retained as a member to keep this class safe to import from
     * any module that exports a class template.
     *
     * CUDA 13.0+ Compatibility:
     * - clockRate and memoryClockRate fields are deprecated (always 0).
     * - Uses cudaDeviceGetAttribute for clock rates instead.
     */
    export class CudaDeviceProps
    {
    public:
        /**
         * @brief Constructs device properties for specified CUDA device.
         *
         * Queries CUDA runtime for device properties and caches all needed values.
         * For CUDA 13.0+, clock rates are queried via device attributes.
         *
         * @param device_id CUDA device ID (0-based).
         * @throws std::runtime_error If device properties query fails.
         */
        explicit CudaDeviceProps( int device_id )
            : device_id_( device_id )
        {
            cudaDeviceProp props{};
            cudaCheckStatus( cudaGetDeviceProperties( &props, device_id ) );

            name = props.name;
            major = props.major;
            minor = props.minor;
            totalGlobalMem = props.totalGlobalMem;
            sharedMemPerBlock = props.sharedMemPerBlock;
            maxThreadsPerBlock = props.maxThreadsPerBlock;
            maxThreadsDim[0] = props.maxThreadsDim[0];
            maxThreadsDim[1] = props.maxThreadsDim[1];
            maxThreadsDim[2] = props.maxThreadsDim[2];
            maxGridSize[0] = props.maxGridSize[0];
            maxGridSize[1] = props.maxGridSize[1];
            maxGridSize[2] = props.maxGridSize[2];
            warpSize = props.warpSize;
            multiProcessorCount = props.multiProcessorCount;
            memoryBusWidth = props.memoryBusWidth;
            l2CacheSize = props.l2CacheSize;
            pciBusID = props.pciBusID;
            pciDeviceID = props.pciDeviceID;
            pciDomainID = props.pciDomainID;

            int clockRateKHz = 0;
            int memoryClockRateKHz = 0;

            cudaError_t clockErr = cudaDeviceGetAttribute(
                &clockRateKHz, cudaDevAttrClockRate, device_id );

            cudaError_t memClockErr = cudaDeviceGetAttribute(
                &memoryClockRateKHz, cudaDevAttrMemoryClockRate, device_id );

            clockRate = ( clockErr == cudaSuccess ) ? clockRateKHz : 0;
            memoryClockRate = ( memClockErr == cudaSuccess ) ? memoryClockRateKHz : 0;
        }

        std::string name;
        int major{ 0 };
        int minor{ 0 };
        size_t totalGlobalMem{ 0 };
        size_t sharedMemPerBlock{ 0 };
        int maxThreadsPerBlock{ 0 };
        int maxThreadsDim[3]{ 0, 0, 0 };
        int maxGridSize[3]{ 0, 0, 0 };
        int warpSize{ 0 };
        int multiProcessorCount{ 0 };
        int clockRate{ 0 };
        int memoryClockRate{ 0 };
        int memoryBusWidth{ 0 };
        int l2CacheSize{ 0 };
        int pciBusID{ 0 };
        int pciDeviceID{ 0 };
        int pciDomainID{ 0 };

        /**
         * @brief Gets compute capability as major/minor version pair.
         * @return std::pair<int, int> e.g. {8, 9} for SM 8.9.
         */
        std::pair<int, int> getComputeCapability() const
        {
            return { major, minor };
        }

        /**
         * @brief Gets the device name.
         * @return std::string Device name.
         */
        std::string getName() const
        {
            return name;
        }

    private:
        int device_id_{ 0 };
    };
}