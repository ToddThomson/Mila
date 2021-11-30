/*
 * Copyright 2021 Todd Thomson, Achilles Software.  All rights reserved.
 *
 * Please refer to the ACHILLES end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */

#ifndef MILA_DNN_CUDA_ENV_H_
#define MILA_DNN_CUDA_ENV_H_

namespace Mila::Dnn::Cuda {

    class CudaEnv
    {
    public:

        explicit CudaEnv();

        ~CudaEnv();

        /**
         * @brief Returns a reference to the singleton instance.
         */
        static CudaEnv& instance();

    private:

    };
}
#endif