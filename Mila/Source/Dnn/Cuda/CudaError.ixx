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

export module Cuda.Error;

namespace Mila::Dnn::Cuda {

    export class CudaError : public std::runtime_error
    {
    public:

        CudaError( cudaError_t status )
            : std::runtime_error( get_message( status ) ), cuda_error_( status )
        {
        }

        cudaError_t Error() const noexcept
        {
            return cuda_error_;
        }

    private:

        cudaError_t cuda_error_ = cudaSuccess;

        static std::string get_message( cudaError_t status )
        {
            const char* name = cudaGetErrorName( status );
            const char* desc = cudaGetErrorString( status );

            if ( !name )
                name = "<unknown error>";

            std::ostringstream ss;
            ss << "CUDA runtime API error "
                << name << " (" << static_cast<unsigned>(status) << ")";
            
            if ( desc && *desc )
                ss << ":\n" << desc;

            return ss.str();
        }
    };
}
