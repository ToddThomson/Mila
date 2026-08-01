/**
 * @file Mila.ixx
 * @brief Mila public API umbrella module - the single supported entry point (import Mila;).
 *
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2021..2026 Todd J. Thomson
 */

module;
#include <string>
#include <memory>
#include <format>
#include <exception>

export module Mila;

// The export list below IS the public API spec: `import Mila;` is the single supported entry point,
// and the public surface is exactly what this umbrella re-exports.
//
// WHY THIS LIST IS BROAD (investigated 2026-07-19 -- do not "narrow" it without a full rebuild):
// under MSVC C++23 modules a type named in a public template's interface -- a member or a base, e.g.
// TensorBuffer inside Tensor<>, or OperationTraits/OperationType/Operation inside every component --
// must be VISIBLE in the consumer's TU at instantiation, not merely reachable. Dropping such a module
// from this list breaks every consumer that instantiates the template, even though no consumer ever
// names the type, so grepping for the symbol finds nothing. Modules exporting free functions have the
// same property: unqualified calls (e.g. getDeviceCount) resolve only through this re-export.
// Widening later is non-breaking; narrowing is breaking -- and how far it can narrow is bounded by
// the above, not by taste.

export import Mila.Version;

// ====================================================================
// Core
// ====================================================================
export import Core.RandomGenerator;

// ====================================================================
// Logging
// ====================================================================
export import Logging.Logger;
export import Logging.ConsoleSink;
export import Logging.FileSink;
export import Logging.NullSink;

// ====================================================================
// Compute - Base ( PUBLIC -- required visible for component instantiation )
// ====================================================================
// The old "make Operations internal only?" REVIEW is answered NO by the module rules: public
// component templates name these at instantiation (e.g. Linear<>'s
// `using OpType = OperationTraits<OperationType::LinearOp, ...>::type`, and every component derives
// from Operation<>). Under MSVC modules those names must be VISIBLE, not merely reachable, in any
// consumer TU that instantiates a component -- so they are public and must be re-exported.
export import Compute.OperationBase;
export import Compute.OperationType;

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
export import Compute.CpuDevice;
#ifdef MILA_HAS_CUDA
export import Compute.DeviceTypeTraits.Cuda;
export import Compute.CudaDevice;
#endif

// ============================================================================
// Compute - Optimizers
// ============================================================================
export import Compute.OptimizerBase;

// ====================================================================
// Compute - Device Registry
// ====================================================================
// DeviceRegistryHelpers exports free functions (getDeviceCount / getBestDevice /
// listDevicesByName / listDevicesByType) that the test suite uses pervasively as
// CUDA-availability guards; unqualified calls resolve only if the module is re-exported here.
import Compute.DeviceRegistrar; // Not part of the Mila public API
export import Compute.DeviceRegistry;
export import Compute.DeviceRegistryHelpers;

// ====================================================================
// Compute - Memory Resources
// ====================================================================
export import Compute.MemoryResource;
//export import Compute.MemoryResourceTracker;
export import Compute.CpuMemoryResource;
#ifdef MILA_HAS_CUDA
export import Compute.CudaDeviceMemoryResource;
export import Compute.CudaManagedMemoryResource;
export import Compute.CudaPinnedMemoryResource;
#endif

// ====================================================================
// Compute - Operations Registry
// ====================================================================
// DEPRECATED: the runtime OperationRegistry (and OperationsRegistrar/Helpers) has been retired
// in favor of compile-time OperationTraits dispatch. The modules are kept on disk but removed
// from the build; they are no longer re-exported from the public umbrella.

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
export import Dnn.ModelType;
export import Dnn.ComponentConfig;
export import Dnn.CompositeComponent;

// ============================================================================
// Dnn - Core Network
// ============================================================================
export import Dnn.Network;
export import Dnn.NetworkFactory;

// ============================================================================
// Dnn - Core Model
// ============================================================================
export import Dnn.Model;
export import Dnn.LanguageNetwork;
export import Dnn.LanguageModel;
export import Dnn.SamplingParams;
export import Dnn.GenerateParams;
export import Dnn.GenerateStatus;
export import Dnn.RuntimeMode;

// ====================================================================
// Dnn - Tensors
// ====================================================================
export import Dnn.Tensor;
export import Dnn.ITensor;
// Dnn.TensorBuffer is PUBLIC: it is a member type of the Tensor<> template (Tensor::buffer_),
// so consumers instantiating Tensor need it visible under MSVC modules.
export import Dnn.TensorBuffer;
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
// Dnn - Components
// ====================================================================
export import Dnn.ActivationType;
export import Dnn.ApproximationMethod;
export import Dnn.ConnectionType;

export import Dnn.Components.MultiHeadAttention;
export import Dnn.Components.Gqa;
export import Dnn.Components.Lpe;
export import Dnn.Components.Rope;
export import Dnn.Components.Gelu;
export import Dnn.Components.Activation;
export import Dnn.Components.Swiglu;
export import Dnn.Components.LayerNorm;
export import Dnn.Components.RmsNorm;
export import Dnn.Components.TokenEmbedding;

// Compute.OperationTraits is PUBLIC: component templates name it at instantiation (see the
// Compute - Base note) -- consumers instantiating any component need it visible.
export import Compute.OperationTraits;

// The quantization policies are PUBLIC for the same reason, and the rule is the one recorded
// when the export surface was frozen: a type in a public template's interface must be VISIBLE,
// not merely reachable, at instantiation. TWeightQuantization is Linear's third template
// parameter and TKvPolicy is GroupedQueryAttention's, so both policy families are part of a
// public interface.
//
// The omission was easy to miss because it fails asymmetrically: `Linear<Cuda, BF16>` compiles
// through this umbrella since a DEFAULT template argument only needs its type reachable, while
// `Linear<Cuda, BF16, PerChannelFp8<>>` -- the spelling the quantization design is documented in
// terms of -- did not, until these two lines.
export import Dnn.Quantization.Weight.Policies;
export import Dnn.Quantization.KvCache.Policy;

export import Dnn.Components.Linear;

export import Dnn.Components.Residual;
export import Dnn.Components.Softmax;
// BACKLOG: export import Dnn.Components.SoftmaxCrossEntropy;

// ============================================================================
// Dnn - Composite Components
// ============================================================================
export import Dnn.Components.MLP;
export import Dnn.Components.GatedMLP;
export import Dnn.Components.GptBlock;

// ============================================================================
// Networks - Open Source Transformer Networks
// ============================================================================
export import Dnn.Components.GptTransformer;
export import Dnn.Components.LlamaTransformer;
export import Dnn.Components.GemmaConfig;
export import Dnn.Components.IDecoderLayer;
export import Dnn.Components.GemmaBlock;
export import Dnn.Components.GemmaTransformer;
export import Dnn.Components.GemmaProtocol;

// ============================================================================
// Models - Open Source Models
// ============================================================================
export import Dnn.Models.GptModel;

export import Dnn.Models.LlamaModel;
export import Dnn.Models.LlamaModelConfig;

export import Dnn.Models.GemmaModel;
export import Dnn.Models.GemmaModelConfig;

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

// ============================================================================
// Serialization
// ============================================================================
export import Serialization.Mode;
export import Serialization.OpenMode;
export import Serialization.Metadata;
export import Serialization.ModelArchive;
// Component::loadParameter takes an ITensorBlob, so this module's types are part of the
// public API whether or not they were exported. Without this, a consumer can call nothing
// that names one and cannot override loadParameter at all.
export import Serialization.Tensor;
export import Serialization.ArchiveSerializer;
export import Serialization.ZipSerializer;
export import Serialization.SafeTensors;
export import Serialization.PretrainedReader;

// ============================================================================
// Distribution - the local store, always present; the hub is gated
// ============================================================================
export import Distribution.Sha256;
export import Distribution.Environment;
export import Distribution.ModelCoordinate;
export import Distribution.ModelStore;

#ifdef MILA_HAS_MODEL_DOWNLOAD
export import Distribution.HttpClient;
export import Distribution.ModelHub;
export import Distribution.ModelResolver;
#endif

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
export import Data.BpePreTokenizationMode;

// ============================================================================
// Data - Datasets
// ============================================================================
export import Data.TokenSequenceLoader;

/**
 * @brief Mila main API namespace.
 */
namespace Mila
{
    namespace detail
    {
        std::shared_ptr<Logging::Logger> g_defaultLogger;
    }

    /**
     * @brief Initializes the Mila framework.
     *
     * Must be called before using any other Mila functionality. If no sink is
     * provided a NullSink is used, suppressing all log output -- appropriate for
     * applications linking Mila as a static library that manage their own logging.
     * Pass an explicit sink to opt in to Mila log output.
     *
     * @param randomSeed  Random seed for reproducibility. 0 = non-deterministic.
     * @param sink        Logging sink to register. nullptr = NullSink (silent).
     * @return True if initialization succeeded, false otherwise.
     * @throws            Any exception thrown during initialization is propagated
     *                    to the caller; the application is responsible for handling it.
     *
     * @code
     * // Silent -- appropriate default for apps linking Mila as a library
     * Mila::initialize();
     *
     * // Development / CLI tool -- opt in to Info-level console output
     * auto sink = std::make_shared<Mila::Logging::ConsoleSink>( Logging::LogLevel::Info );
     * Mila::initialize( 0, sink );
     *
     * // FastAPI server -- structured file logging at Warning+
     * auto sink = std::make_shared<Mila::Logging::FileSink>( "mila.log", Logging::LogLevel::Warning );
     * Mila::initialize( 0, sink );
     * @endcode
     */
    export bool initialize(
        unsigned int randomSeed = 0,
        std::shared_ptr<Logging::Logger> sink = nullptr )
    {
        if ( sink )
        {
            detail::g_defaultLogger = std::move( sink );
        }
        else
        {
            detail::g_defaultLogger = std::make_shared<Logging::NullSink>();
        }

        Logging::Logger::setDefaultLogger( detail::g_defaultLogger.get() );

        Core::RandomGenerator::getInstance().setSeed( randomSeed );

        if ( randomSeed != 0 )
        {
            auto message = std::format( "Initialized random generator with seed: {}", randomSeed );
            Logging::Logger::info( message );
        }
        else
        {
            Logging::Logger::info( "Initialized random generator with non-deterministic seed." );
        }

        Dnn::Compute::DeviceRegistrar::instance();

        Logging::Logger::info( "Mila framework initialized successfully." );

        return true;
    }

    /**
     * @brief Shuts down the Mila framework and releases all resources.
     *
     * Flushes any pending log output through the registered sink before
     * releasing it. After this call no further log calls should be made
     * until initialize() is called again.
     *
     * @throws Any exception thrown during shutdown is propagated to the caller.
     */
    export void shutdown()
    {
        Logging::Logger::info( "Shutting down Mila framework." );

        detail::g_defaultLogger.reset();
        Logging::Logger::setDefaultLogger( nullptr );
    }
}
