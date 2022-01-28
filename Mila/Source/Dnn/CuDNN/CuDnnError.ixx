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
#include <stdexcept>
#include <exception>
#include <functional>
#include <string>

export module CuDnn.Error;

import Cuda.Error;
import CuDnn.Utils;

namespace Mila::Dnn::CuDnn
{
    export class CudnnException : public std::runtime_error
    {
    public:

        CudnnException( const char* message, cudnnStatus_t status_ )
            throw() : std::runtime_error( message )
        {
            error_status = status_;
        }
        
        cudnnStatus_t GetStatus()
        {
            return error_status;
        }

    private:

        cudnnStatus_t error_status;
    };

    

    static inline void cudnnStatusCheck( cudnnStatus_t status )
    {
        if ( status != CUDNN_STATUS_SUCCESS )
        {
            throw CudnnException(
                std::string( "CuDNN Error: " + std::string(" cudnn_status: ") + to_string(status)).c_str(),
                status );
        }
    }
}