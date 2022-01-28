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
#include <cuda_runtime.h>
#include <stdexcept>

export module Cuda.Device;

import Cuda.DeviceProps;
import Cuda.Helpers;

namespace Mila::Dnn::Cuda
{
	/// <summary>
	/// Represents a Cuda device. 
	/// </summary>
    export class CudaDevice // : public unique_handle<int, CudaDevice>
    {
    public:

        /// <summary>
        /// Creates a CudaDevice object with the specified device id.
        /// </summary>
        /// <param name="deviceId">Cuda GPU device id</param>
        CudaDevice( int deviceId = 0 )
            : device_id_( CheckDevice( deviceId ) ), props_( CudaDeviceProps( device_id_ ) )
        {
        }

        const CudaDeviceProps& GetProperties() const
        {
            return props_;
        }

    private:

        int device_id_;

        CudaDeviceProps props_;
    };
}