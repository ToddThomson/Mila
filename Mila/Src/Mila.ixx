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

export import Dnn.Module;
export import Dnn.Model;
export import Dnn.Tensor;
export import Dnn.TensorHelpers;

//export import Compute.Device;
export import Compute.DeviceInterface;
export import Compute.DeviceRegistry;
export import Compute.CpuDevice;
export import Compute.CudaDevice;

//export import Dnn.Session;

export import Compute.Cpu.Ops.layernorm;

export namespace Mila {
	/// <summary>
	/// Gets the Mila API version.
	/// </summary>
	export Version GetAPIVersion() {
		return Version{0, 9, 17};
	}

	// TODO: Static device registration should be automatic
	bool Dnn::Compute::Cpu::CpuDevice::registered_ = (CpuDevice::RegisterDevice(), true);
	bool Dnn::Compute::Cuda::CudaDevice::registered_ = (CudaDevice::RegisterDevices(), true);
}