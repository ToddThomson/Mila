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
#include "Version.h"

export module Mila;

import Mila.Version;

export import Cuda.Error;
export import Cuda.Helpers;

export import Utils.Logger;
import Utils.DefaultLogger;

export import Dnn.Module;
export import Dnn.Model;

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

export import Dnn.Gpt2.DatasetReader;

import Compute.OperationsRegistrar;

namespace Mila
{
    namespace detail
    {
        std::shared_ptr<Utils::DefaultLogger> g_defaultLogger;
    }

    void initializeLogger( Utils::LogLevel level = Utils::LogLevel::Info ) {
        detail::g_defaultLogger = std::make_shared<Utils::DefaultLogger>( level );
        Utils::Logger::setDefaultLogger( detail::g_defaultLogger.get() );
    }

    /// <summary>
    /// Gets the current Mila API version.
    /// </summary>
    /// <returns>A Version object containing the version information</returns>
    export Version getAPIVersion() {
        return Version{
                MILA_VERSION_MAJOR,
                MILA_VERSION_MINOR,
                MILA_VERSION_PATCH,
                MILA_VERSION_PRERELEASE_TAG,
                MILA_VERSION_PRERELEASE
        };
    }

    export void setDevice( const std::string& name ) {
        Dnn::Compute::DeviceContext::instance().setDevice( name );
    }

    export std::shared_ptr<Dnn::Compute::ComputeDevice> getDevice() {
        return Dnn::Compute::DeviceContext::instance().getDevice();
    }

    /// <summary>
    /// Initializes the Mila framework.
    /// Must be called before using any other Mila functionality.
    /// </summary>
    /// <returns>True if initialization succeeded, false otherwise</returns>
    export bool initialize() {
        try {
            initializeLogger( Utils::LogLevel::Info );

			// Initialize operations and device context
            Dnn::Compute::OperationsRegistrar::instance();
            Dnn::Compute::DeviceContext::instance();

            Utils::Logger::info( "Mila framework initialized successfully" );
            return true;
        }
        catch ( const std::exception& e ) {
            // Fall back to std::cerr if logger isn't initialized yet
            std::cerr << "Mila initialization failed: " << e.what() << std::endl;
            return false;
        }
    }

    export void shutdown() {
        Utils::Logger::info( "Shutting down Mila framework" );

        detail::g_defaultLogger.reset();
        Utils::Logger::setDefaultLogger( nullptr );

        // TODO: Add other cleanup code here...
    }
}