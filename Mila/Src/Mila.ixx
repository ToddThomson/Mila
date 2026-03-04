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
#include <string>
#include <iostream>
#include <memory>
#include "Version.h"
#include <exception>

export module Mila;

import Mila.Version;

// ====================================================================
// Core
// ====================================================================
export import Core.RandomGenerator;

// ====================================================================
// Utils
// ====================================================================
export import Utils.Logger;
import Utils.DefaultLogger;

// ====================================================================
// Cuda 
// REVIEW: TODO: Make internal. We don't want to expose CUDA details
// in the main API.
// ====================================================================
//export import Cuda.Error;
//export import Cuda.Helpers;

// ====================================================================
// Compute - Base
// ====================================================================
// REVIEW: Should we make Operations internal only?
export import Compute.OperationBase;
export import Compute.OperationType;
export import Compute.UnaryOperation;
export import Compute.BinaryOperation;
export import Compute.Precision;

// ====================================================================
// Compute - Execution Context
// ====================================================================
export import Compute.IExecutionContext;
export import Compute.ExecutionContextFactory;

// ====================================================================
// Compute - Devices
// ====================================================================
export import Compute.Device;
export import Compute.DeviceId;
export import Compute.DeviceType;
export import Compute.DeviceTypeTraits;
export import Compute.DeviceTypeTraits.Cpu;
export import Compute.DeviceTypeTraits.Cuda;
export import Compute.CpuDevice;
export import Compute.CudaDevice;

// ============================================================================
// Compute - Optimizers
// ============================================================================
export import Compute.OptimizerBase;

// ====================================================================
// Compute - Device Registry
// ====================================================================
import Compute.DeviceRegistrar; // Not part of the Mila public API
export import Compute.DeviceRegistry;
export import Compute.DeviceRegistryHelpers;

// ====================================================================
// Compute - Memory Resources
// ====================================================================
export import Compute.MemoryResource;
//export import Compute.MemoryResourceTracker;
export import Compute.CpuMemoryResource;
export import Compute.CudaDeviceMemoryResource;
export import Compute.CudaManagedMemoryResource;
export import Compute.CudaPinnedMemoryResource;

// ====================================================================
// Compute - Operations Registry
// ====================================================================
import Compute.OperationsRegistrar;
export import Compute.OperationRegistry;
export import Compute.OperationRegistryHelpers;

// Deprecated to be removed later
// export import Compute.OperationAttributes;

// ====================================================================
// Compute - CPU Operations ( internal )
// ====================================================================
//export import Compute.CpuEncoderOp;
//export import Compute.CpuGeluOp;
//export import Compute.CpuLayerNormOp;
//export import Compute.CpuLinearOp;
//export import Compute.CpuResidualOp;
//export import Compute.CpuSoftmaxOp;

// ====================================================================
// Compute - CUDA Operations ( internal )
// ====================================================================
//export import Compute.CudaEncoderOp;
//export import Compute.CudaGeluOp;
//export import Compute.CudaMHAOp;
//export import Compute.CudaLinearOp;
//export import Compute.CudaLayerNormOp;
//export import Compute.CudaResidualOp;
//export import Compute.CudaSoftmaxOp;

// ====================================================================
// Compute - Tensor Data Types
// ====================================================================
//export import Compute.CudaTensorDataType;
//export import Compute.CpuTensorDataTypeTraits;
// FUTURE: export import Compute.MetalTensorTraits;
// FUTURE: export import Compute.OpenCLTensorTraits;
// FUTURE: export import Compute.VulkanTensorTraits;

// ====================================================================
// Dnn - Core, Components, and Composite Components
// ====================================================================
export import Dnn.Component;
export import Dnn.ComponentType;
export import Dnn.ComponentConfig;
export import Dnn.CompositeComponent;

// ============================================================================
// Dnn -Core Network
// ============================================================================
export import Dnn.Network;
export import Dnn.NetworkFactory;

// ====================================================================
// Dnn - Tensors
// ====================================================================
export import Dnn.Tensor;
export import Dnn.ITensor;
export import Dnn.TensorBuffer; // TJT: Remove after testing
export import Dnn.TensorTypes;
export import Dnn.TensorDataType;
export import Dnn.TensorDataTypeTraits;
export import Dnn.TensorDataTypeMap;
export import Dnn.TensorHostTypeMap;

// ====================================================================
// Dnn - Tensor Operations
// ====================================================================
export import Dnn.TensorOps;

// ====================================================================
// Dnn - Tensor Initializers
// ====================================================================
export import Dnn.TensorInitializers;

// ====================================================================
// Dnn - Components
// ====================================================================
export import Dnn.ActivationType;
export import Dnn.ApproximationMethod;
export import Dnn.ConnectionType;

export import Dnn.Components.MultiHeadAttention;
export import Dnn.Components.GroupedQueryAttention;
export import Dnn.Components.Lpe;
export import Dnn.Components.Rope;
export import Dnn.Components.Gelu;
export import Dnn.Components.Swiglu;
export import Dnn.Components.LayerNorm;
export import Dnn.Components.RmsNorm;
export import Dnn.Components.Linear;
export import Dnn.Components.Residual;
export import Dnn.Components.Softmax;
//export import Dnn.Components.SoftmaxCrossEntropy;

// ============================================================================
// Dnn - Composite Components
// ============================================================================
export import Dnn.Components.MLP;
export import Dnn.Components.GptBlock;
export import Dnn.Components.LlamaBlock;

// ============================================================================
// Networks - Open Source Transformer Networks
// ============================================================================
export import Dnn.Components.GptTransformer;
export import Dnn.Components.LlamaTransformer;

// ============================================================================
// Models - Open Source Models
// ============================================================================
export import Dnn.Models.GptModel;
export import Dnn.Models.LlamaModel;

// ============================================================================
// Dnn - Optimizers
// ============================================================================
export import Dnn.Optimizers.AdamW;
export import Dnn.Optimizers.AdamWConfig;

// ============================================================================
// Dnn - LossFunctions
// ============================================================================
//export import Dnn.Loss;

// ============================================================================
// Dnn - Data
// ============================================================================
export import Data.DataLoader;
export import Data.Tokenizer;
export import Data.TokenizerType;
export import Data.Gpt2Tokenizer;

// ============================================================================
// Serialization
// ============================================================================
export import Serialization.Mode;
export import Serialization.OpenMode;
export import Serialization.Metadata;
export import Serialization.ModelArchive;
export import Serialization.ArchiveSerializer;
export import Serialization.ZipSerializer;

// ============================================================================
// Modeling
// NOTE: Moved to Dev/Training
// ============================================================================
//export import Dnn.Model;
//export import Dnn.ModelConfig;

// ============================================================================
// Data - Core
// ============================================================================

// Data - Tokenizers
export import Data.CharTokenizer;
export import Data.CharTrainer;
export import Data.CharVocabulary;
export import Data.CharVocabularyConfig;
export import Data.SpecialTokens;

export import Data.BpeVocabulary;
export import Data.BpeVocabularyConfig;
export import Data.BpeTokenizer;
export import Data.BpeTrainer;
// FIXME: export import Data.BpeSpecialTokens;

// ============================================================================
// Data - Datasets
// ============================================================================
export import Data.DataLoader;
export import Data.TokenSequenceLoader;

/**
 * @brief Mila main API namespace.
 */
namespace Mila
{
    namespace detail
    {
        std::shared_ptr<Utils::DefaultLogger> g_defaultLogger;
    }

    static void initializeLogger( Utils::LogLevel level = Utils::LogLevel::Info ) {
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
            initializeLogger( Utils::LogLevel::Debug );

            Core::RandomGenerator::getInstance().setSeed( randomSeed );

            if (randomSeed != 0) {
                Utils::Logger::info( "Initialized random generator with seed: " + std::to_string( randomSeed ) );
            }
            else {
                Utils::Logger::info( "Initialized random generator with non-deterministic seed." );
            }

            Dnn::Compute::DeviceRegistrar::instance();
            Dnn::Compute::OperationsRegistrar::instance();

            Utils::Logger::info( "Mila framework initialized successfully." );

            return true;
        }
        catch (const std::exception& e) {
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