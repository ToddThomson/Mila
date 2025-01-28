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
#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "Dnn/TensorBuffer.cuh"

export module Mila;

import Mila.Version;

export import Cuda.Error;
export import Cuda.Helpers;

export import Dnn.Module;
export import Dnn.Model;

export import Dnn.Tensor;
export import Dnn.TensorHelpers;

export import Compute.DeviceInterface;
export import Compute.DeviceContext;
import Compute.DeviceRegistry;
import Compute.CpuDevice;
import Compute.CudaDevice;
export import Compute.DeviceHelpers;

import Compute.CudaMatMulOp;

export import Dnn.Modules.LayerNorm;
export import Dnn.Modules.MatMul;

import Compute.CpuLayerNormOp;
import Compute.CpuMatMulOp;

namespace Mila {
    /// <summary>
    /// Gets the Mila API version.
    /// </summary>
    export Version GetAPIVersion() {
        return Version{0, 9, 35, "alpha", 1 };
    }

	export void setDevice( const std::string& name ) {
		Dnn::Compute::DeviceContext::instance().setDevice( name );
	}

    // TJT: Remove. Ensure the static instance is referenced to trigger the constructor
    void Initialize() {
    }

    // Explicit instantiation definitions
    //export template class Dnn::TensorBuffer<float, thrust::device_vector>;
    //export template class Dnn::TensorBuffer<int, thrust::device_vector>;

    export template class Dnn::TensorBuffer<float, thrust::host_vector>;
    export template class Dnn::TensorBuffer<int, thrust::host_vector>;

}