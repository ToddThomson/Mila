/**
 * @file CudaDeviceProps.ixx
 * @brief CUDA device properties wrapper with caching and convenience methods.
 *
 * Provides a type-safe wrapper around cudaDeviceProp with additional
 * convenience methods for querying device capabilities and formatting.
 * Compatible with CUDA 13.0+ which deprecated clockRate fields.
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
     * Queries and caches CUDA device properties on construction, providing
     * convenient accessors for commonly used properties and formatted output.
     *
     * Properties are cached to avoid repeated CUDA API calls and stored as
     * public members for direct access when needed.
     *
     * CUDA 13.0+ Compatibility:
     * - clockRate and memoryClockRate fields are deprecated (always 0)
     * - Uses cudaDeviceGetAttribute for clock rates instead
     */
    export class CudaDeviceProps
    {
    public:
        /**
         * @brief Constructs device properties for specified CUDA device.
         *
         * Queries CUDA runtime for device properties and caches commonly used values.
         * For CUDA 13.0+, clock rates are queried via device attributes.
         *
         * @param device_id CUDA device ID (0-based)
         * @throws std::runtime_error If device properties query fails
         */
        explicit CudaDeviceProps( int device_id )
            : device_id_( device_id )
        {
            cudaCheckStatus( cudaGetDeviceProperties( &props_, device_id ) );

            // Cache commonly used properties as public members
            name = props_.name;
            major = props_.major;
            minor = props_.minor;
            totalGlobalMem = props_.totalGlobalMem;
            sharedMemPerBlock = props_.sharedMemPerBlock;
            maxThreadsPerBlock = props_.maxThreadsPerBlock;
            maxThreadsDim[0] = props_.maxThreadsDim[0];
            maxThreadsDim[1] = props_.maxThreadsDim[1];
            maxThreadsDim[2] = props_.maxThreadsDim[2];
            maxGridSize[0] = props_.maxGridSize[0];
            maxGridSize[1] = props_.maxGridSize[1];
            maxGridSize[2] = props_.maxGridSize[2];
            warpSize = props_.warpSize;
            multiProcessorCount = props_.multiProcessorCount;
            memoryBusWidth = props_.memoryBusWidth;
            l2CacheSize = props_.l2CacheSize;
            pciBusID = props_.pciBusID;
            pciDeviceID = props_.pciDeviceID;
            pciDomainID = props_.pciDomainID;

            // CUDA 13.0+: clockRate and memoryClockRate are deprecated
            // Use device attributes instead for accurate values
            int clockRateKHz = 0;
            int memoryClockRateKHz = 0;

            cudaError_t clockErr = cudaDeviceGetAttribute(
                &clockRateKHz,
                cudaDevAttrClockRate,
                device_id
            );

            cudaError_t memClockErr = cudaDeviceGetAttribute(
                &memoryClockRateKHz,
                cudaDevAttrMemoryClockRate,
                device_id
            );

            // Store clock rates (will be 0 if attributes not available)
            clockRate = (clockErr == cudaSuccess) ? clockRateKHz : 0;
            memoryClockRate = (memClockErr == cudaSuccess) ? memoryClockRateKHz : 0;
        }

        // ====================================================================
        // Public Cached Properties (for direct access)
        // ====================================================================

        std::string name;                   ///< Device name
        int major;                          ///< Compute capability major version
        int minor;                          ///< Compute capability minor version
        size_t totalGlobalMem;              ///< Total global memory (bytes)
        size_t sharedMemPerBlock;           ///< Shared memory per block (bytes)
        int maxThreadsPerBlock;             ///< Maximum threads per block
        int maxThreadsDim[3];               ///< Maximum threads per dimension
        int maxGridSize[3];                 ///< Maximum grid size per dimension
        int warpSize;                       ///< Warp size in threads
        int multiProcessorCount;            ///< Number of multiprocessors
        int clockRate;                      ///< Clock frequency (kHz) - queried via attribute API
        int memoryClockRate;                ///< Memory clock frequency (kHz) - queried via attribute API
        int memoryBusWidth;                 ///< Memory bus width (bits)
        int l2CacheSize;                    ///< L2 cache size (bytes)
        int pciBusID;                       ///< PCI bus ID
        int pciDeviceID;                    ///< PCI device ID
        int pciDomainID;                    ///< PCI domain ID

        // ====================================================================
        // Accessors
        // ====================================================================

        /**
         * @brief Gets the raw CUDA device properties structure.
         * @return const cudaDeviceProp* Pointer to underlying cudaDeviceProp
         */
        const cudaDeviceProp* getProperties() const
        {
            return &props_;
        }

        /**
         * @brief Gets the device name.
         * @return std::string Device name (e.g., "NVIDIA GeForce RTX 4090")
         */
        std::string getName() const {
            return name;
        }

        /**
         * @brief Gets compute capability as major/minor version pair.
         * @return std::pair<int, int> Major and minor version (e.g., {8, 9})
         */
        std::pair<int, int> getComputeCapability() const {
            return { major, minor };
        }

        /**
         * @brief Gets compute capability as single version number.
         * @return int Compute capability version (e.g., 89 for SM 8.9)
         */
        int getComputeCapabilityVersion() const {
            return major * 10 + minor;
        }

        /**
         * @brief Gets total global memory in bytes.
         * @return size_t Total global memory
         */
        size_t getTotalGlobalMem() const {
            return totalGlobalMem;
        }

        /**
         * @brief Gets total global memory in GB.
         * @return double Total global memory in gigabytes
         */
        double getTotalGlobalMemGB() const {
            return static_cast<double>(totalGlobalMem) / (1024.0 * 1024.0 * 1024.0);
        }

        /**
         * @brief Gets shared memory per block in bytes.
         * @return size_t Shared memory per block
         */
        size_t getSharedMemPerBlock() const {
            return sharedMemPerBlock;
        }

        /**
         * @brief Gets maximum threads per block.
         * @return int Maximum threads per block
         */
        int getMaxThreadsPerBlock() const {
            return maxThreadsPerBlock;
        }

        /**
         * @brief Gets warp size.
         * @return int Warp size (typically 32)
         */
        int getWarpSize() const {
            return warpSize;
        }

        /**
         * @brief Gets number of streaming multiprocessors.
         * @return int Number of SMs
         */
        int getMultiprocessorCount() const {
            return multiProcessorCount;
        }

        /**
         * @brief Gets device clock rate in MHz.
         *
         * Uses cudaDeviceGetAttribute API for CUDA 13.0+ compatibility.
         *
         * @return double Clock rate in megahertz (0.0 if unavailable)
         */
        double getClockRateMHz() const {
            return clockRate > 0 ? static_cast<double>(clockRate) / 1000.0 : 0.0;
        }

        /**
         * @brief Gets memory clock rate in MHz.
         *
         * Uses cudaDeviceGetAttribute API for CUDA 13.0+ compatibility.
         *
         * @return double Memory clock rate in megahertz (0.0 if unavailable)
         */
        double getMemoryClockRateMHz() const {
            return memoryClockRate > 0 ? static_cast<double>(memoryClockRate) / 1000.0 : 0.0;
        }

        /**
         * @brief Gets memory bandwidth in GB/s (theoretical peak).
         *
         * Calculated as: (Memory Clock * Bus Width * 2) / 8 / 1000
         * Returns 0.0 if memory clock rate is unavailable.
         *
         * @return double Memory bandwidth in gigabytes per second (0.0 if unavailable)
         */
        double getMemoryBandwidthGBs() const {
            if (memoryClockRate == 0) {
                return 0.0;
            }

            // Bandwidth = (Memory Clock Rate * Memory Bus Width * 2) / 8 / 1000
            // *2 for DDR, /8 for bits to bytes, /1000 for MHz to GHz
            return (memoryClockRate / 1000.0) * (memoryBusWidth / 8.0) * 2.0 / 1000.0;
        }

        /**
         * @brief Gets L2 cache size in bytes.
         * @return int L2 cache size
         */
        int getL2CacheSize() const {
            return l2CacheSize;
        }

        /**
         * @brief Gets PCI location string.
         * @return std::string PCI location (e.g., "0000:01:00.0")
         */
        std::string getPciLocation() const {
            return std::format( "{:04x}:{:02x}:{:02x}.0",
                pciDomainID, pciBusID, pciDeviceID );
        }

        // ====================================================================
        // Capability Checks
        // ====================================================================

        /**
         * @brief Checks if device supports FP16 (half precision).
         * @return bool True if FP16 is supported (SM 6.0+)
         */
        bool supportsFp16() const {
            return major >= 6;
        }

        /**
         * @brief Checks if device supports BF16 (bfloat16).
         * @return bool True if BF16 is supported (SM 8.0+)
         */
        bool supportsBf16() const {
            return major >= 8;
        }

        /**
         * @brief Checks if device supports FP8.
         * @return bool True if FP8 is supported (SM 9.0+)
         */
        bool supportsFp8() const {
            return major >= 9;
        }

        /**
         * @brief Checks if device has Tensor Cores.
         * @return bool True if Tensor Cores are available (SM 7.0+)
         */
        bool hasTensorCores() const {
            return major >= 7;
        }

        /**
         * @brief Checks if device supports INT8 tensor cores.
         * @return bool True if INT8 tensor cores are supported (SM 7.5+)
         */
        bool supportsInt8TensorCores() const {
            return getComputeCapabilityVersion() >= 75;
        }

        /**
         * @brief Checks if device supports unified memory.
         * @return bool True if unified memory is supported
         */
        bool supportsUnifiedMemory() const {
            return props_.unifiedAddressing != 0;
        }

        /**
         * @brief Checks if device supports concurrent kernels.
         * @return bool True if concurrent kernel execution is supported
         */
        bool supportsConcurrentKernels() const {
            return props_.concurrentKernels != 0;
        }

        // ====================================================================
        // Formatted Output
        // ====================================================================

        /**
         * @brief Converts device properties to formatted string.
         * @return std::string Formatted device properties
         */
        std::string toString() const
        {
            int driver_version = Cuda::getDriverVersion();
            int runtime_version = Cuda::getRuntimeVersion();

            std::ostringstream ss;

            ss << "Device Properties:\n"
                << "  Name: " << name << "\n"
                << "  CUDA Driver Version: " << (driver_version / 1000) << "."
                << (driver_version % 100) / 10 << "\n"
                << "  CUDA Runtime Version: " << (runtime_version / 1000) << "."
                << (runtime_version % 100) / 10 << "\n"
                << "  Compute Capability: " << major << "." << minor
                << " (SM " << getComputeCapabilityVersion() << ")\n"
                << "  Total Global Memory: " << std::format( "{:.2f}", getTotalGlobalMemGB() ) << " GB\n";

            // Only show clock rates if available
            if (memoryClockRate > 0) {
                ss << "  Memory Clock Rate: " << std::format( "{:.0f}", getMemoryClockRateMHz() ) << " MHz\n";
            }

            ss << "  Memory Bus Width: " << memoryBusWidth << " bits\n";

            if (memoryClockRate > 0) {
                ss << "  Memory Bandwidth: " << std::format( "{:.1f}", getMemoryBandwidthGBs() ) << " GB/s\n";
            }

            ss << "  L2 Cache Size: " << (l2CacheSize / 1024) << " KB\n";

            if (clockRate > 0) {
                ss << "  Clock Rate: " << std::format( "{:.0f}", getClockRateMHz() ) << " MHz\n";
            }

            ss << "  Multiprocessors: " << multiProcessorCount << "\n"
                << "  Warp Size: " << warpSize << "\n"
                << "  Max Threads/Block: " << maxThreadsPerBlock << "\n"
                << "  Max Grid Size: [" << maxGridSize[0] << ", " << maxGridSize[1]
                << ", " << maxGridSize[2] << "]\n"
                << "  Shared Memory/Block: " << (sharedMemPerBlock / 1024) << " KB\n"
                << "  PCI Location: " << getPciLocation() << "\n"
                << "  Capabilities:\n"
                << "    FP16: " << (supportsFp16() ? "Yes" : "No") << "\n"
                << "    BF16: " << (supportsBf16() ? "Yes" : "No") << "\n"
                << "    FP8: " << (supportsFp8() ? "Yes" : "No") << "\n"
                << "    Tensor Cores: " << (hasTensorCores() ? "Yes" : "No") << "\n"
                << "    INT8 Tensor Cores: " << (supportsInt8TensorCores() ? "Yes" : "No") << "\n"
                << "    Unified Memory: " << (supportsUnifiedMemory() ? "Yes" : "No") << "\n"
                << "    Concurrent Kernels: " << (supportsConcurrentKernels() ? "Yes" : "No");

            return ss.str();
        }

        /**
         * @brief Converts to brief summary string.
         * @return std::string Brief device summary
         */
        std::string toSummary() const {
            return std::format( "{} (SM {}.{}, {:.2f} GB, {} SMs)",
                name, major, minor, getTotalGlobalMemGB(), multiProcessorCount );
        }

    private:
        int device_id_;
        cudaDeviceProp props_;
    };
}