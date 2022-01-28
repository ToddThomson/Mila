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
#include <cudnn.h>
#include <memory>
#include <iostream>

export module CuDnn.OpaqueHandle;

import CuDnn.Error;

export namespace Mila::Dnn::CuDnn
{
    // Forward Declaration for ManagedCudnnHandle
    class OpaqueCudnnHandle;

    /// <summary>
    /// A shared_ptr wrapper on top of the OpaqueHandle
    /// </summary>
    export using ManagedCudnnHandle = std::shared_ptr<OpaqueCudnnHandle>;

    /// <summary>
    /// A wrapper on top of the std::make_shared for the OpaqueHandle
    /// </summary>
    /// <returns></returns>
    export ManagedCudnnHandle MakeManagedCudnnHandle()
    {
        return std::make_shared<OpaqueCudnnHandle>();
    };

    export class OpaqueCudnnHandle
    {
    public:

        /// <summary>
        /// Delete the copy constructor to prevent copies.
        /// </summary>
        OpaqueCudnnHandle( const OpaqueCudnnHandle& ) = delete;
        OpaqueCudnnHandle& operator=( const OpaqueCudnnHandle& ) = delete;

        OpaqueCudnnHandle( OpaqueCudnnHandle&& ) = default;

        /// <summary>
        /// Constructor
        /// </summary>
        OpaqueCudnnHandle()
        {
            std::cout << "OpaqueCudnnHandle() Creating cuDNN handle...\n";

            status_ = cudnnCreate( &handle_ );

            if ( status_ != CUDNN_STATUS_SUCCESS )
            {
                throw std::runtime_error( "Failed to create CuDNN handle." );
                    //cudnnException(
                    //std::string( "Failed to create CuDNN handle. Status: " ).c_str(),
                    //status_ );
            }
        }

        /// <summary>
        /// Gets the const reference to the opaque CuDNN library context.
        /// </summary>
        /// <returns>Opaque CuDNN library context</returns>
        cudnnHandle_t GetOpaqueHandle() const
        {
            return handle_;
        }

        /// <summary>
        /// Gets the status of the cudnnCreate call. 
        /// </summary>
        cudnnStatus_t GetStatus() const
        {
            return status_;
        }

        /// <summary>
        /// Returns true if the handle creation was completed successfully.
        /// </summary>
        bool IsStatusSuccess() const
        {
            return status_ == CUDNN_STATUS_SUCCESS;
        }

        /**
         * OpaqueHandle destructor.
         * Calls the cudnnDestroy to release resources.
         */
        ~OpaqueCudnnHandle() noexcept(false)
        {
            std::cout << "~OpaqueCudnnHandle() calling cudnnDestroy().. ";

            if ( handle_ != nullptr )
            {
                auto status = cudnnDestroy( handle_ );

                if ( status != CUDNN_STATUS_SUCCESS )
                {
                    throw std::runtime_error( "Failed to destroy cudnn handle." );// , status );
                }
            }

            std::cout << "Done. Completed successfully." << std::endl;
        }

    private:

        /// <summary>
        /// CuDNN handle returned from cudnnCreate() call.
        /// </summary>
        cudnnHandle_t handle_ = nullptr;

        /// <summary>
        /// Status of handle creation.
        /// </summary>
        cudnnStatus_t status_ = CUDNN_STATUS_SUCCESS;
    };
}