/*
 * Copyright 2021 Todd Thomson, Achilles Software.  All rights reserved.
 *
 * Please refer to the ACHILLES end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */

module;
#include <iostream>
#include <string>
#include <cuda_runtime.h>

export module Cuda.Memory;

import Cuda.Helpers;

namespace Mila::Dnn::Cuda
{
    /// <summary>
    /// Represents Cuda device memory.
    /// </summary>
    export class CudaMemory
    {
    public:

        CudaMemory( size_t bufferSize = 0 )
        {
            std::cout << "CudaMemory::CudaMemory( bufferSize: " 
                << std::to_string( bufferSize ) << " )" << std::endl;

            buffer_size_ = bufferSize;

            if ( bufferSize > 0 )
            {
                std::cout << "CudaMemory:: allocating memory: "
                    << std::to_string( buffer_size_ ) << " Bytes.. ";

                cudaCheckStatus( cudaMalloc( (void**)&buffer_, buffer_size_ ) );
                
                std::cout << "Done memory allocation\n";
            }
        }

        ~CudaMemory()
        {
            std::cout << "~CudaMemory()\n";
            if ( buffer_size_ > 0 )
            {
                std::cout << " --> freeing Cuda memory..";
                cudaCheckStatus( cudaFree( buffer_ ) );
                std::cout << " Done freeing cuda memory" << std::endl;
            }
        }

        /// <summary>
        /// Move constructor.
        /// </summary>
        CudaMemory( CudaMemory&& other )
            : buffer_( other.buffer_ ), buffer_size_( other.buffer_size_)
        {
            std::cout << "CudaMemory move constuctor" << std::endl;

            other.buffer_ = nullptr;
            other.buffer_size_ = 0;
        }

        /// <summary>
        /// Move assignment operator.
        /// </summary>
        /// <param name="other">CudaMemory object being moved.</param>
        /// <returns>This pointer</returns>
        CudaMemory& operator=( CudaMemory&& other )
        {
            buffer_ = other.buffer_;
            buffer_size_ = other.buffer_size_;
            
            other.buffer_ = nullptr;
            other.buffer_size_ = 0;

            return *this;
        }

        /// <summary>
        /// Sets device memory to a specified value.
        /// </summary>
        /// <param name="value"></param>
        /// <param name="count"></param>
        void Set( int value, size_t count = -1 )
        {
            if ( count < 0 )
            {
                count = buffer_size_;
            }

            if ( count > buffer_size_ )
            {
                // TJT: Exception here.
                return;
            }

            cudaMemset( buffer_, value, count );
        }

        /// <summary>
        /// Get the pointer to the device memory buffer.
        /// </summary>
        /// <returns>Buffer pointer</returns>
        void* GetBuffer()
        {
            return buffer_;
        }

        /// <summary>
        /// Gets the size of the memory buffer.
        /// </summary>
        /// <returns>The buffer size.</returns>
        size_t GetBufferSize()
        {
            return buffer_size_;
        }

        /// <summary>
        /// Copy constructor is deleted as the device memory buffer resource is
        /// non-copyable.
        /// </summary>
        CudaMemory( const CudaMemory& ) = delete;

        /// <summary>
        /// Copy assignment operation is deleted as the device memory buffer resource is
        /// non-copyable. 
        /// </summary>
        CudaMemory& operator=( const CudaMemory& ) = delete;

    private:

        void* buffer_ = nullptr;
        size_t buffer_size_ = 0;
    };
}
