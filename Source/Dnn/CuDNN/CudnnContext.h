/*
 * Copyright 2021 Todd Thomson, Achilles Software.  All rights reserved.
 *
 * Please refer to the ACHILLES end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */

#ifndef MILA_DNN_CUDNN_CONTEXT_H_
#define MILA_DNN_CUDNN_CONTEXT_H_

#include <string>
#include <memory>

#include <cudnn.h>

#include "../Cuda/CudaDevice.h"
#include "OpaqueHandle.h"
#include "error.h"

namespace Mila::Dnn::CuDNN
{
    class CudnnContext;

    using ManagedCudnnContext = std::shared_ptr<CudnnContext>;
    /// <summary>
    /// CuDNN library context.
    /// </summary>
    class CudnnContext
    {
    public:
        /// <summary>
        /// CuDNN context class constructor
        /// </summary>
        CudnnContext() /* CudaDevice device ) */
        {
            std::cout << "In CudnnContext constructor\n";

            // Initialize the managed handle
            auto status = InitializeManagedCudnnHandle();

            if ( status != CUDNN_STATUS_SUCCESS )
            {
                throw cudnnException(
                    std::string( std::string( "Failed to create CuDNN handle. Status: " ) + to_string( status ) ).c_str(), status );
            }
        }

        ManagedCudnnHandle& GetCudnnHandle()
        {
            return handle_;
        }

    private:

        CudnnContext( CudnnContext const& ) = delete;
        CudnnContext& operator=( CudnnContext const& ) = delete;

        /// <summary>
        /// Initializes the managed CuDNN handle.
        /// </summary>
        /// <returns>CuDNN status</returns>
        cudnnStatus_t InitializeManagedCudnnHandle()
        {
            handle_ = MakeManagedCudnnHandle();

            return handle_->GetStatus();
        }

        ManagedCudnnHandle handle_ = nullptr;

        mutable cudnnStatus_t status_ = CUDNN_STATUS_SUCCESS;
        mutable std::string err_msg_;
    };
}
#endif