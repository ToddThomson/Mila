/*
 * Copyright 2021 Todd Thomson, Achilles Software.  All rights reserved.
 *
 * Please refer to the ACHILLES end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */

#ifndef MILA_DNN_CUDA_ERROR_H_
#define MILA_DNN_CUDA_ERROR_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <cstring>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <ostream>
#include <iostream>
#include <sstream>

namespace Mila::Dnn::Cuda {

    class CudaError : public std::runtime_error
    {
    public:

        explicit CudaError( cudaError_t status_ ) : std::runtime_error( get_message( status_ ) ), rt_err( status_ )
        {
        }
        explicit CudaError( CUresult status_ ) : std::runtime_error( get_message( status_ ) ), drv_err( status_ )
        {
        }

        CUresult drv_error() const noexcept
        {
            return drv_err;
        }
        cudaError_t rt_error() const noexcept
        {
            return rt_err;
        }

        bool is_drv_api() const noexcept
        {
            return drv_err != CUDA_SUCCESS;
        }
        bool is_rt_api() const noexcept
        {
            return rt_err != cudaSuccess;
        }

    private:

        CUresult drv_err = CUDA_SUCCESS;
        cudaError_t rt_err = cudaSuccess;

        static std::string get_message( CUresult status_ )
        {
            const char* name = nullptr, * desc = nullptr;
            cuGetErrorName( status_, &name );
            cuGetErrorString( status_, &desc );
            std::ostringstream ss;
            if ( !name ) name = "<unknown error>";
            ss << "CUDA driver API error "
                << name << " (" << static_cast<unsigned>(status_) << ")";
            if ( desc && *desc ) ss << ":\n" << desc;
            
            return ss.str();
        }

        static std::string get_message( cudaError_t status_ )
        {
            const char* name = cudaGetErrorName( status_ );
            const char* desc = cudaGetErrorString( status_ );
            if ( !name ) name = "<unknown error>";
            std::ostringstream ss;
            ss << "CUDA runtime API error "
                << name << " (" << static_cast<unsigned>(status_) << ")";
            if ( desc && *desc ) ss << ":\n" << desc;
            return ss.str();
        }
    };
}
#endif