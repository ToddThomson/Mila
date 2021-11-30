/*
 * Copyright 2021 Todd Thomson, Achilles Software.  All rights reserved.
 *
 * Please refer to the ACHILLES end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */

#ifndef MILA_DNN_CUDA_DEVICE_H_
#define MILA_DNN_CUDA_DEVICE_H_

#include <cuda_runtime.h>
#include "CudaHelper.h"

namespace Mila::Dnn::Cuda
{
	/// <summary>
	/// 
	/// </summary>
    class CudaDevice // : public unique_handle<int, CudaDevice>
    {
    public:

        /// <summary>
        /// 
        /// </summary>
        /// <param name="device_id"></param>
        CudaDevice( int device = 0 )
        {
            device_ = Init( device );
        }

    private: // Methods

        int Init( int device )
        {
            if ( device < 0 )
            {
                throw std::invalid_argument( "Invalid device." );
            }

            int devCount = getDeviceCount();

            if ( devCount == 0 )
            {
                throw std::runtime_error( "No devices found." );
            }

            if ( device > devCount - 1 )
            {
                throw std::out_of_range( "device id out of range." );
            }

            int computeMode = -1,
            CUDA_CALL( cudaDeviceGetAttribute( &computeMode, cudaDevAttrComputeMode, device ) );

            if ( computeMode == cudaComputeModeProhibited )
            {
                throw std::runtime_error( "Device is running in Compute ModeProhibited." );
            }

            //CUDA_CALL( cudaSetDevice( device ) );

            return device;
        };
    
    private:

        int device_;
    };
}
#endif