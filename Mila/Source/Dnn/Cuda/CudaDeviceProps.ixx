/**
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
#include <string>
#include <iostream>
#include <sstream>
#include <vector>
#include <stdexcept>
#include <cuda_runtime.h>

export module Cuda.DeviceProperties;

import Cuda.Helpers;

namespace Mila::Dnn::Cuda 
{
	export class CudaDeviceProperties
	{
	public:

        CudaDeviceProperties( int deviceId )
        {
            CUDA_CALL( cudaGetDeviceProperties( &props_, deviceId ) );
        }

        const cudaDeviceProp* GetProperties() const
        {
            return &props_;
        }

        std::string ToString() const
        {
            // Include CUDA driver and runtime versions
            int driver_version = GetDriverVersion();
            int runtime_version = GetRuntimeVersion();
            
            std::stringstream ss;

            ss << "Device Properties: " << std::endl
                << "Device: " << props_.name << std::endl
                << " CUDA Driver Version: " << (driver_version / 1000) << ", " << (driver_version % 100) / 10 << std::endl
                << " CUDA Runtime Version: " << (runtime_version / 1000) << "," << (runtime_version % 100) / 10 << std::endl;

            ss << " CUDA Capability Major/Minor version number: " << props_.major << "/" << props_.minor << std::endl;

            ss << " Total amount of global memory (bytes): " << props_.totalGlobalMem << std::endl;

            return ss.str();
        }

        const std::string GetName() const {
            return props_.name;
        }

        const std::pair<int, int> GetComputeCaps() const {
            return std::pair<int, int>( props_.major, props_.minor );
        }

        const size_t GetTotalGlobalMem() const {
            return props_.totalGlobalMem;
        }

	private:

		cudaDeviceProp props_;
	};
}