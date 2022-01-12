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
#include <cstring>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <ostream>
#include <iostream>
#include <sstream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

export module Cuda.Error;

namespace Mila::Dnn::Cuda {

    export class CudaError : public std::runtime_error
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

            // FIXME: Linking errors
            
            //cuGetErrorName( status_, &name );
            //cuGetErrorString( status_, &desc );
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
