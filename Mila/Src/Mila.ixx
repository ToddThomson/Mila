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
#include <cuda_fp16.h>
#include <string>
#include <iostream>
#include <memory>
#include "Version.h"

export module Mila;

import Mila.Version;

export import Cuda.Error;
export import Cuda.Helpers;

export import Utils.Logger;
import Utils.DefaultLogger;

export import Core.RandomGenerator;

export import Compute.OperationBase;
export import Compute.OperationType;
export import Compute.UnaryOperation;
export import Compute.BinaryOperation;
export import Compute.Precision;

export import Dnn.Module;
export import Dnn.ComponentConfig;
export import Dnn.CompositeModule;
export import Dnn.Model;

export import Dnn.Tensor;
export import Dnn.TensorBuffer; // TJT: Remove after testing
export import Dnn.TensorTraits;
export import Dnn.TensorHelpers;
export import Dnn.ActivationType;

export import Data.DataLoader;

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
export import Compute.DynamicMemoryResource;

export import Compute.OperationRegistry;
export import Compute.OperationAttributes;

export import Compute.CpuEncoderOp;
export import Compute.CpuGeluOp;
export import Compute.CpuLayerNormOp;
export import Compute.CpuLinearOp;
export import Compute.CpuResidualOp;
export import Compute.CpuSoftmaxOp;

export import Compute.CudaEncoderOp;
export import Compute.CudaGeluOp;
export import Compute.CudaSoftmaxOp;
export import Compute.CudaMHAOp;
export import Compute.CudaLinearOp;
export import Compute.CudaLayerNormOp;
export import Compute.CudaResidualOp;

export import Dnn.Modules.Attention;
export import Dnn.Modules.Encoder;
export import Dnn.Modules.Gelu;
export import Dnn.Modules.LayerNorm;
export import Dnn.Modules.Linear;
export import Dnn.Modules.Residual;
export import Dnn.Modules.Softmax;

export import Dnn.Blocks.MLP;
export import Dnn.Blocks.TransformerBlock;

export import Dnn.Gpt2.DatasetReader;

import Compute.OperationsRegistrar;
import Compute.DeviceRegistrar;

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

    /// <summary>
    /// Initializes the Mila framework.
    /// Must be called before using any other Mila functionality.
    /// </summary>
    /// <param name="randomSeed">Random seed for reproducibility (0 = use non-deterministic seed)</param>
    /// <returns>True if initialization succeeded, false otherwise</returns>
    export bool initialize( unsigned int randomSeed = 0 ) {
        try {
            initializeLogger( Utils::LogLevel::Info );

            // Initialize random generator with provided seed
            Core::RandomGenerator::getInstance().setSeed( randomSeed );
            if ( randomSeed != 0 ) {
                Utils::Logger::info( "Initialized random generator with seed: " + std::to_string( randomSeed ) );
            }
            else {
                Utils::Logger::info( "Initialized random generator with non-deterministic seed." );
            }

            // Initialize operations and devices
            Dnn::Compute::OperationsRegistrar::instance();
            Dnn::Compute::DeviceRegistrar::instance();

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