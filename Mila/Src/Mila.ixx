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
#include <memory>

export module Mila;

import Mila.Version;

export import Cuda.Error;
export import Cuda.Helpers;

export import Dnn.Module;
export import Dnn.Model;
export import Dnn.ModelBase;

export import Dnn.Tensor;
export import Dnn.TensorBuffer; // TJT: Remove after testing
export import Dnn.TensorTraits;
export import Dnn.TensorHelpers;

export import Compute.ComputeDevice;
export import Compute.DeviceRegistry;
export import Compute.DeviceType;
export import Compute.CpuDevice;
export import Compute.CudaDevice;
export import Compute.DeviceContext;
export import Compute.DeviceHelpers;

export import Compute.MemoryResource;
export import Compute.CpuMemoryResource;
export import Compute.CudaMemoryResource;
export import Compute.CudaManagedMemoryResource;
export import Compute.CudaPinnedMemoryResource;

export import Dnn.Modules.Attention;
export import Dnn.Modules.Encoder;
export import Dnn.Modules.Gelu;
export import Dnn.Modules.LayerNorm;
export import Dnn.Modules.FullyConnected;
export import Dnn.Modules.Residual;
export import Dnn.Modules.Softmax;

export import Dnn.Blocks.MLP;
export import Dnn.Blocks.TransformerBlock;

import Compute.Operations;

namespace Mila {
    /// <summary>
    /// Gets the Mila API version.
    /// </summary>
    export Version GetAPIVersion() {
        return Version{0, 9, 62, "alpha", 1 };
    }

	export void setDevice( const std::string& name ) {
		Dnn::Compute::DeviceContext::instance().setDevice( name );
	}

    export std::shared_ptr<Dnn::Compute::ComputeDevice> getDevice() {
        return Dnn::Compute::DeviceContext::instance().getDevice();
    }

    // TJT: Remove. Ensure the static instance is referenced to trigger the constructor
    export void Initialize() {
        Dnn::Compute::Operations::instance();
		Dnn::Compute::DeviceContext::instance();
    }
}