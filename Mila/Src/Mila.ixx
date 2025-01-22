/*
 * Copyright 2025 Todd Thomson, Achilles Software.  All rights reserved.
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
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#ifdef USE_OMP
#include <omp.h>
#endif

export module Mila;

import Mila.Version;

export import Cuda.Error;
export import Cuda.Helpers;

export import Dnn.Module;
export import Dnn.Model;
export import Dnn.ModelContext;
export import Dnn.Tensor;
export import Dnn.TensorHelpers;

//export import Compute.Device;
export import Compute.DeviceInterface;
import Compute.DeviceRegistry;
import Compute.CpuDevice;
import Compute.CudaDevice;
export import Compute.DeviceHelpers;

export import Dnn.Modules.LayerNorm;
export import Dnn.Modules.MatMul;

import Compute.Cpu.Ops.LayerNormOp;

namespace Mila {
    /// <summary>
    /// Gets the Mila API version.
    /// </summary>
    export Version GetAPIVersion() {
        return Version{0, 9, 25, "alpha", 1 };
    }

	export void device( const std::string& name ) {
		Mila::Dnn::ModelContext::instance().setDevice( name );
	}

    namespace {
        struct DeviceRegistrar {
            DeviceRegistrar() {
                Mila::Dnn::Compute::Cpu::CpuDevice::RegisterDevice();
                Mila::Dnn::Compute::Cuda::CudaDevice::RegisterDevices();
            }
        };

		struct OperationRegistrar {
			OperationRegistrar() {
                Mila::Dnn::Compute::Cpu::Ops::LayerNormOp<float>::RegisterOperation();
			}
		};

        // Static instance to trigger registration
        static DeviceRegistrar deviceRegistrar;
		static OperationRegistrar operationRegistrar;
    }

    // Ensure the static instance is referenced to trigger the constructor
    export void Initialize() {
        (void)deviceRegistrar;
		(void)operationRegistrar;
    }
}