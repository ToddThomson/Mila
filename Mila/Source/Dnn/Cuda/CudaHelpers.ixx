/*
 * Copyright 2021 Todd Thomson, Achilles Software.  All rights reserved.
 *
 * Please refer to the Mila end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

module;
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdexcept>
#include <cuda_runtime.h>

export module Cuda.Helpers;

import Cuda.Error;

namespace Mila::Dnn::Cuda 
{
    export inline void cudaCheckStatus( cudaError_t status_ )
    {
        switch ( status_ )
        {
        case cudaSuccess:
            return;

        default:
            cudaGetLastError();
            throw CudaError( status_ );
        }
    }

    export template <typename T>
    inline void CUDA_CALL( T status_ )
    {
        return cudaCheckStatus( status_ );
    }

    static const char* _cudaGetErrorEnum( cudaError_t error )
    {
        return cudaGetErrorName( error );
    }

    // Beginning of GPU Architecture definitions
    inline int _ConvertSMVer2Cores( int major, int minor )
    {
        // Defines for GPU Architecture types (using the SM version to determine
        // the # of cores per SM
        typedef struct
        {
            int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
            // and m = SM minor version
            int Cores;
        } sSMtoCores;

        sSMtoCores nGpuArchCoresPerSM[] = {
            {0x30, 192},
            {0x32, 192},
            {0x35, 192},
            {0x37, 192},
            {0x50, 128},
            {0x52, 128},
            {0x53, 128},
            {0x60,  64},
            {0x61, 128},
            {0x62, 128},
            {0x70,  64},
            {0x72,  64},
            {0x75,  64},
            {0x80,  64},
            {0x86, 128},
            {-1, -1} };

        int index = 0;

        while ( nGpuArchCoresPerSM[ index ].SM != -1 )
        {
            if ( nGpuArchCoresPerSM[ index ].SM == ((major << 4) + minor) )
            {
                return nGpuArchCoresPerSM[ index ].Cores;
            }

            index++;
        }

        // If we don't find the values, we default use the previous one
        // to run properly
        printf(
            "MapSMtoCores for SM %d.%d is undefined."
            "  Default to use %d Cores/SM\n",
            major, minor, nGpuArchCoresPerSM[ index - 1 ].Cores );
        return nGpuArchCoresPerSM[ index - 1 ].Cores;
    }

    inline const char* _ConvertSMVer2ArchName( int major, int minor )
    {
        // Defines for GPU Architecture types (using the SM version to determine
        // the GPU Arch name)
        typedef struct
        {
            int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
            // and m = SM minor version
            const char* name;
        } sSMtoArchName;

        sSMtoArchName nGpuArchNameSM[] = {
            {0x30, "Kepler"},
            {0x32, "Kepler"},
            {0x35, "Kepler"},
            {0x37, "Kepler"},
            {0x50, "Maxwell"},
            {0x52, "Maxwell"},
            {0x53, "Maxwell"},
            {0x60, "Pascal"},
            {0x61, "Pascal"},
            {0x62, "Pascal"},
            {0x70, "Volta"},
            {0x72, "Xavier"},
            {0x75, "Turing"},
            {0x80, "Ampere"},
            {0x86, "Ampere"},
            {-1, "Graphics Device"} };

        int index = 0;

        while ( nGpuArchNameSM[ index ].SM != -1 )
        {
            if ( nGpuArchNameSM[ index ].SM == ((major << 4) + minor) )
            {
                return nGpuArchNameSM[ index ].name;
            }

            index++;
        }

        // If we don't find the values, we default use the previous one
        // to run properly
        printf(
            "MapSMtoArchName for SM %d.%d is undefined."
            "  Default to use %s\n",
            major, minor, nGpuArchNameSM[ index - 1 ].name );
        return nGpuArchNameSM[ index - 1 ].name;
    }

    export inline int getDeviceCount()
    {
        int devCount;
        CUDA_CALL( cudaGetDeviceCount( &devCount ) );

        return devCount;
    }

    inline int deviceInit( int device )
    {
        if ( device < 0 )
        {
            throw std::invalid_argument( "Invalid device." );
        }

        int deviceCount = getDeviceCount();

        if ( deviceCount == 0 )
        {
            throw std::runtime_error( "No devices found." );
        }

        if ( device > deviceCount - 1 )
        {
            throw std::out_of_range( "device out of range." );
        }

        int computeMode = -1,

            CUDA_CALL( cudaDeviceGetAttribute( &computeMode, cudaDevAttrComputeMode, device ) );

        if ( computeMode == cudaComputeModeProhibited )
        {
            throw std::runtime_error( "Device is running in Compute ModeProhibited." );
        }

        cudaSetDevice( device );

        return device;
    }
    
    /// <summary>
    /// Returns the best GPU (with maximum GFLOPS) 
    /// </summary>
    /// <returns>GPU device Id</returns>
    inline int gpuGetMaxGflopsDeviceId()
    {
        int current_device = 0, sm_per_multiproc = 0;
        int max_perf_device = 0;
        int devices_prohibited = 0;

        uint64_t max_compute_perf = 0;

        int devCount = getDeviceCount();

        if ( devCount == 0 )
        {
            throw std::runtime_error( "No CUDA devices found." );
        }

        // Find the best CUDA capable GPU device
        current_device = 0;

        while ( current_device < devCount )
        {
            int computeMode = -1, major = 0, minor = 0;

            CUDA_CALL( cudaDeviceGetAttribute( &computeMode, cudaDevAttrComputeMode, current_device ) );
            CUDA_CALL( cudaDeviceGetAttribute( &major, cudaDevAttrComputeCapabilityMajor, current_device ) );
            CUDA_CALL( cudaDeviceGetAttribute( &minor, cudaDevAttrComputeCapabilityMinor, current_device ) );

            // If this GPU is not running on Compute Mode prohibited,
            // then we can add it to the list
            if ( computeMode != cudaComputeModeProhibited )
            {
                if ( major == 9999 && minor == 9999 )
                {
                    sm_per_multiproc = 1;
                }
                else
                {
                    sm_per_multiproc =
                        _ConvertSMVer2Cores( major, minor );
                }

                int multiProcessorCount = 0, clockRate = 0;
                CUDA_CALL( cudaDeviceGetAttribute( &multiProcessorCount, cudaDevAttrMultiProcessorCount, current_device ) );
                cudaError_t result = cudaDeviceGetAttribute( &clockRate, cudaDevAttrClockRate, current_device );

                if ( result != cudaSuccess )
                {
                    // If cudaDevAttrClockRate attribute is not supported we
                    // set clockRate as 1, to consider GPU with most SMs and CUDA Cores.
                    if ( result == cudaErrorInvalidValue )
                    {
                        clockRate = 1;
                    }
                    else
                    {
                        fprintf( stderr, "CUDA error at %s:%d code=%d(%s) \n", __FILE__, __LINE__,
                            static_cast<unsigned int>(result), _cudaGetErrorEnum( result ) );
                        exit( EXIT_FAILURE );
                    }
                }
                uint64_t compute_perf = (uint64_t)multiProcessorCount * sm_per_multiproc * clockRate;

                if ( compute_perf > max_compute_perf )
                {
                    max_compute_perf = compute_perf;
                    max_perf_device = current_device;
                }
            }
            else
            {
                devices_prohibited++;
            }

            ++current_device;
        }

        if ( devices_prohibited == devCount )
        {
            throw std::runtime_error( "All devices have compute mode prohibited." );
        }

        return max_perf_device;
    }

    // Initialization code to find the best CUDA Device
    inline int findBestCudaDevice( int device_id = -1 )
    {
        int devID = 0;

        devID = gpuGetMaxGflopsDeviceId();

        CUDA_CALL( cudaSetDevice( devID ) );

        return devID;
    }

    // General check for CUDA GPU SM Capabilities
    inline bool checkCudaComputeCapabily( int major_version, int minor_version )
    {
        int dev;
        int major = 0, minor = 0;

        CUDA_CALL( cudaGetDevice( &dev ) );
        CUDA_CALL( cudaDeviceGetAttribute( &major, cudaDevAttrComputeCapabilityMajor, dev ) );
        CUDA_CALL( cudaDeviceGetAttribute( &minor, cudaDevAttrComputeCapabilityMinor, dev ) );

        if ( (major > major_version) ||
            (major == major_version &&
                minor >= minor_version) )
        {
            return true;
        }
        else
        {
            return false;
        }
    }
}