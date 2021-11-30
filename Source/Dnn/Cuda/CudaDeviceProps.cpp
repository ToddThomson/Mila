/**
 * Copyright 2021 Todd Thomson, Achilles Software.  All rights reserved.
 *
 * Please refer to the ACHILLES end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */

#include <cuda_runtime.h>
#include "CudaHelper.h"
#include "CudaDeviceProps.h"

namespace Mila::Dnn::Cuda {

    CudaDeviceProps::CudaDeviceProps()
    {
        int devCount = 0;
        CUDA_CALL( cudaGetDeviceCount( &devCount ) );

        if ( devCount > 0 )
        {
            props_.resize( devCount );

            for ( int devId = 0; devId < devCount; ++devId )
            {
                CUDA_CALL( cudaGetDeviceProperties( &props_[ devId ], devId ) );
            }
        }
    }

    const cudaDeviceProp* CudaDeviceProps::get( int devId ) const
    {
        if ( static_cast<size_t>(devId) < props_.size() )
        {
            throw std::out_of_range( "device id out of range." );
        }

        return &props_[ devId ];
    };

} // Mila::Compute::Cuda namespace
