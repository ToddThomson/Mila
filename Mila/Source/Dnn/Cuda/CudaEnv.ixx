/*
 * Copyright 2021 Todd Thomson, Achilles Software.  All rights reserved.
 *
 * Please refer to the Mila end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */

module;
#include <cuda_runtime.h>

export module Cuda.Environment;

import Cuda.Helpers;

namespace Mila::Dnn::Cuda 
{
    export class CudaEnv
    {
    public:

        CudaEnv()
        {
            int num_devices = 0;
            CUDA_CALL( cudaGetDeviceCount( &num_devices ) );

            //num_devices_ = num_devices;
        };

        ~CudaEnv();

        /**
         * @brief Returns a reference to the singleton instance.
         */
        static CudaEnv& instance();

    private:

    };
}