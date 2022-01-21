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
#include <string>
#include <memory>
#include <iostream>

export module CuDnn.Context;

import Cuda.Device;
import CuDnn.OpaqueHandle;
import CuDnn.Error;
import CuDnn.Utils;

namespace Mila::Dnn::CuDnn
{
    class CudnnContext;

    export using ManagedCudnnContext = std::shared_ptr<CudnnContext>;
    
    /// <summary>
    /// CuDNN library context.
    /// </summary>
    export class CudnnContext
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