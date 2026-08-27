# CMake configuration for the Mila library

set(CMAKE_CXX_SCAN_FOR_MODULES ON)

# Sources are added below via target_sources, partitioned by category and guarded
# by backend. CUDA kernel .cu (compiled, never shipped) and .cuh/.h (shipped
# declarations) live in the if(MILA_HAS_CUDA) block, so a CPU-only configure no
# longer drags in CUDA sources.
add_library( Mila STATIC )

# CUDA architectures the Mila library target is compiled for. The default is a
# portable fat binary spanning every supported arch, so a find_package consumer's
# Mila.lib runs on any of them. This is a target property, so it overrides any global
# CMAKE_CUDA_ARCHITECTURES -- which is why the library always builds the whole list
# regardless of a caller's -DCMAKE_CUDA_ARCHITECTURES. A local build that targets one
# known GPU (the devcontainer/WSL convenience path, or any single-card dev build) can
# pin a single arch for a much faster build via -DMILA_LIBRARY_CUDA_ARCHITECTURES=89.
# Opt-in: the default is unchanged, so releases and packaged consumers stay portable.
#
# 120 is Blackwell consumer (RTX 50-series). It is listed so those cards get native
# cubin rather than a PTX JIT from 90 at first launch; nothing in Mila uses sm_120
# instructions yet. Each added arch costs .cu compile time, so pin a single arch when
# iterating on one card.
set(MILA_LIBRARY_CUDA_ARCHITECTURES "75;80;86;89;90;120" CACHE STRING
    "CUDA architectures for the Mila library target (default: portable fat binary)")

set_target_properties( Mila PROPERTIES
	CXX_STANDARD 23
	CXX_STANDARD_REQUIRED ON
	CXX_EXTENSIONS OFF
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
    # Mila uses relocatable device code (separable compilation above), so each .cu
    # leaves device symbols that need a device-link step to emit the
    # __cudaRegisterLinkedBinary stubs. For a STATIC library CMake defers that to
    # the final executable -- fine for in-tree samples, but a find_package consumer
    # linking Mila.lib into a plain C++ executable never device-links and fails with
    # unresolved __cudaRegisterLinkedBinary_*. Resolving device symbols here runs the
    # device link when Mila.lib is created, making the shipped archive self-contained.
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES "${MILA_LIBRARY_CUDA_ARCHITECTURES}"

    # The MilaPy binding is a SHARED object, and on Linux a shared object may contain only
    # position-independent code. A non-PIC static Mila library (the default) makes the
    # pybind11 module link fail: "relocation R_X86_64_PC32 against symbol ... can not be
    # used when making a shared object; recompile with -fPIC". MSVC has no PIC/non-PIC
    # distinction, so this never surfaces on Windows. PIC static code also links cleanly
    # into the Chat/sample executables, so enable it unconditionally.
    POSITION_INDEPENDENT_CODE ON
)

# C++23 is selected via the CXX_STANDARD properties above and forced per-compiler
# below (-std=c++23 / /std:c++latest) for Mila's OWN compile.
#
# find_package consumers are a separate case: they compile Mila's installed module
# units in a synthetic target derived from the imported Mila target, and that
# target takes its C++ standard ONLY from INTERFACE_COMPILE_FEATURES. CXX_STANDARD
# is a target property, not a usage requirement, so it is not exported -- without
# the line below the consumer's generate step fails with "no C++ standard found".
# Declared INTERFACE-only so it never affects Mila's own compile, and guarded
# because CMake rejects cxx_std_23 when a newer compiler's feature table lags
# (e.g. Clang 21.x); there we fall back to the cxx_std_20 the module support needs.
if(cxx_std_23 IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    target_compile_features(Mila INTERFACE cxx_std_23)
else()
    target_compile_features(Mila INTERFACE cxx_std_20)
endif()

#find_package(miniz REQUIRED)  
#find_package(nlohmann_json REQUIRED)

target_include_directories( Mila
    PRIVATE
        ${miniz_SOURCE_DIR}
        #${sentencepiece_SOURCE_DIR}/src
        $<$<BOOL:${cutlass_ADDED}>:${cutlass_SOURCE_DIR}/include> )

# Kernel headers and the vendored nlohmann header are pulled by file-relative quoted
# includes (#include "Kernels/Foo.cuh", "../../Deps/nlohmann/json.hpp"), which resolve
# relative to the including .ixx. A subproject consumer compiles the module units in
# place against this source tree, so those relative includes resolve unchanged.
target_include_directories( Mila
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Src>
)

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  target_compile_options(Mila PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/std:c++latest /permissive- /Zc:preprocessor /W4>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
  )

elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(Mila PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-fmodules-ts -std=c++23>
  )
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(Mila PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-std=c++23 -fno-implicit-modules -fno-implicit-module-maps>
  )
  find_program(MILA_CLANG_SCAN_DEPS
    NAMES clang-scan-deps clang-scan-deps-21 clang-scan-deps-20 clang-scan-deps-19
    REQUIRED)
  set(CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS "${MILA_CLANG_SCAN_DEPS}")
  # nvcc's host compiler for the (module-free) .cu files. Only default it to the
  # C++ compiler when the caller hasn't chosen one -- the Linux presets and CI set
  # gcc-15 (nvcc 13's max supported host), and overriding that here produces a
  # second, conflicting -ccbin ("incompatible redefinition for option
  # 'compiler-bindir'" on every .cu). Setting CMAKE_CUDA_HOST_COMPILER is
  # sufficient; CMake emits the -ccbin itself, so no manual flag append.
  if(NOT CMAKE_CUDA_HOST_COMPILER)
    set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
  endif()
endif()

# Emit device-side line info under RelWithDebInfo so Nsight Compute/Systems can
# correlate optimized SASS back to source. -lineinfo keeps full device
# optimization, unlike the -G device-debug flag used by the Debug build.
target_compile_options( Mila PRIVATE
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:RelWithDebInfo>>:-lineinfo>
)

if(MILA_ENABLE_OPENMP)
  find_package(OpenMP REQUIRED)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(Mila PRIVATE 
      $<$<COMPILE_LANGUAGE:CXX>:-openmp:llvm -DUSE_OMP>
    )
    target_link_options(Mila PRIVATE -openmp:llvm)
  else()
    target_compile_options(Mila PRIVATE 
      $<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>
    )
    target_link_libraries(Mila PRIVATE OpenMP::OpenMP_CXX)
  endif()
endif()

# =============================================================================
# Mila Core Modules
# =============================================================================

target_sources( Mila
PUBLIC
    FILE_SET module_files
       TYPE CXX_MODULES FILES
        #---------------------------------------------------------------
        # Version and Mila 
        #---------------------------------------------------------------
        "Src/Version.ixx"
        "Src/Mila.ixx"

        #----------------------------------------------------------------------
        # Core
        #----------------------------------------------------------------------
        "Src/Dnn/Core/Component.ixx"
        "Src/Dnn/Core/ComponentType.ixx"
        "Src/Dnn/Core/ModelType.ixx"
        "Src/Dnn/Core/CompositeComponent.ixx"
        "Src/Dnn/Core/ComponentConfig.ixx"
        "Src/Dnn/Core/ComponentFactory.ixx"

        #---------------------------------------------------------------
        # Top-level Utilities
        #---------------------------------------------------------------
        "Src/Logging/Logger.ixx"
        "Src/Logging/ConsoleSink.ixx"
        #"Src/Utils/TrainingLogger.ixx"
        "Src/Utils/RandomGenerator.ixx"
        "Src/Utils/json.ixx"

        #---------------------------------------------------------------
        # Dnn / Tensors (top-level)
        #---------------------------------------------------------------
        "Src/Dnn/Tensors/Tensor.ixx"
        "Src/Dnn/Tensors/TensorBuffer.ixx"
        "Src/Dnn/Tensors/ITensor.ixx"
        "Src/Dnn/Tensors/Tensor.Partitioning.ixx"
        "Src/Dnn/Tensors/Tensor.Types.ixx"

        "Src/Dnn/Tensors/TensorDataType.ixx"
        "Src/Dnn/Tensors/Tensor.Serialization.ixx"

        "Src/Dnn/Tensors/TensorDataTypeMap.ixx"
        "Src/Dnn/Tensors/TensorDataTypeTraits.ixx"
        "Src/Dnn/Tensors/TensorHostTypeMap.ixx"

        # Dnn / Tensors / Operations
        "Src/Dnn/Tensors/Operations/TensorOps.ixx"
        "Src/Dnn/Tensors/Operations/TensorOps-Base.ixx"
        "Src/Dnn/Tensors/Operations/TensorOps.Math.ixx"
        "Src/Dnn/Tensors/Operations/TensorOps.Fill.ixx"
        "Src/Dnn/Tensors/Operations/TensorOps.Transfer.ixx"

        # Dnn /Tensors / Debug Helpers
        "Src/Dnn/Tensors/Tensor.Helpers.ixx"
                
        #---------------------------------------------------------------
        # Dnn / Components
        #---------------------------------------------------------------
        
        # Components - Activations
        "Src/Dnn/Components/Activations/ActivationType.ixx"
        "Src/Dnn/Components/Activations/Gelu/Gelu.ixx"
        "Src/Dnn/Components/Activations/Gelu/Gelu.Config.ixx"
        
        # Unified elementwise activation (added alongside Gelu; folds it in later -- FfnAndMoE.md s5)
        "Src/Dnn/Components/Activations/Activation/Activation.ixx"
        "Src/Dnn/Components/Activations/Activation/Activation.Config.ixx"
        
        # Components - Connections
        "Src/Dnn/Components/Connections/ConnectionType.ixx"
        "Src/Dnn/Components/Connections/Residual.ixx"
        "Src/Dnn/Components/Connections/ResidualConfig.ixx"

        # Components - Convolutions
        "Src/Dnn/Components/Convolutions/CausalConv1d.Config.ixx"
        "Src/Dnn/Components/Convolutions/CausalConv1d.ixx"

        # Components - DeltaNet
        "Src/Dnn/Components/DeltaNet/GatedDeltaRule.Config.ixx"
        "Src/Dnn/Components/DeltaNet/GatedDeltaRule.ixx"

        # Components - Embeddings
        "Src/Dnn/Components/Embeddings/TokenEmbedding.ixx"
        "Src/Dnn/Components/Embeddings/TokenEmbedding.Config.ixx"

        # Components - Positional Encodings
        "Src/Dnn/Components/Encodings/Lpe/Lpe.ixx"
        "Src/Dnn/Components/Encodings/Lpe/Lpe.Config.ixx"
        "Src/Dnn/Components/Encodings/Rope/Rope.ixx"
        "Src/Dnn/Components/Encodings/Rope/Rope.Config.ixx"

        # Components - Attention
        "Src/Dnn/Components/Attention/MHA/MultiHeadAttention.ixx"
        "Src/Dnn/Components/Attention/MHA/MultiHeadAttention.Config.ixx"
        
        "Src/Dnn/Components/Linear/Linear.ixx"
        "Src/Dnn/Components/Linear/LinearConfig.ixx"

        # Components - Normalization / Losses / Regularization / Composites
        "Src/Dnn/Components/Normalization/LayerNorm/LayerNorm.ixx"
        "Src/Dnn/Components/Normalization/LayerNorm/LayerNorm.Config.ixx"
        "Src/Dnn/Components/Normalization/RmsNorm/RmsNorm.ixx"
        "Src/Dnn/Components/Normalization/RmsNorm/RmsNorm.Config.ixx"
        "Src/Dnn/Components/Normalization/Softmax.ixx"
        "Src/Dnn/Components/Normalization/SoftmaxConfig.ixx"
        
        "Src/Dnn/Components/Losses/SoftmaxCrossEntropy.ixx"
        "Src/Dnn/Components/Losses/CrossEntropyConfig.ixx"
        
        # Dropout parked at Dev/Components/Regularization/ pending Alpha.8 re-authoring (see BACKLOG).

        "Src/Dnn/Components/FFN/MLP/MLP.ixx"
        "Src/Dnn/Components/FFN/MLP/MLP.Config.ixx"
        # MLP.Dispatch.ixx RETIRED: the runtime activation dispatch (mlp_activation_impl)
        # is gone now that MLP is pinned to a concrete Gelu child. Kept on disk for
        # reference until the generalized Activation component lands (see FfnAndMoE.md).
        #"Src/Dnn/Components/FFN/MLP/MLP.Dispatch.ixx"
        "Src/Dnn/Components/FFN/Swiglu/Swiglu.ixx"
        "Src/Dnn/Components/FFN/Swiglu/Swiglu.Config.ixx"
        "Src/Dnn/Components/FFN/GatedMLP/GatedMLP.ixx"
        "Src/Dnn/Components/FFN/GatedMLP/GatedMLP.Config.ixx"
        "Src/Dnn/Components/Transformers/Gpt/GptBlock.ixx"
        "Src/Dnn/Components/Transformers/Gpt/GptBlock.Config.ixx"

        #---------------------------------------------------------------
        # Dnn / Compute (core)
        #---------------------------------------------------------------
        "Src/Dnn/Compute/Device.ixx"
        "Src/Dnn/Compute/DeviceType.ixx"
        "Src/Dnn/Compute/DeviceId.ixx"

        "Src/Dnn/Compute/ExecutionContext.Template.ixx"
        "Src/Dnn/Compute/Observation.ixx"
        "Src/Dnn/Compute/ExecutionContext.ixx"
        "Src/Dnn/Compute/IExecutionContext.ixx"
        "Src/Dnn/Compute/ExecutionContextFactory.ixx"

        
        "Src/Dnn/Compute/DeviceTypeTraits.ixx"
        

        # Dnn / Compute / Memory Resources
        "Src/Dnn/Compute/MemoryResource.ixx"
        "Src/Dnn/Compute/MemoryResourceTraits.ixx"
        "Src/Dnn/Compute/MemoryResourceProperties.ixx"
        "Src/Dnn/Compute/MemoryResourceTracker.ixx"

        #----------------------------------------------------------------------
        # Dnn / Compute / Optimizers
        #----------------------------------------------------------------------
        "Src/Dnn/Core/OptimizerBase.ixx"

        #---------------------------------------------------------------
        # Dnn / Compute / Operations
        #---------------------------------------------------------------
        "Src/Dnn/Compute/Operations/OperationBase.ixx"
        "Src/Dnn/Compute/Operations/OperationType.ixx"
        "Src/Dnn/Compute/Operations/OperationTraits.ixx"
        "Src/Dnn/Compute/Operations/OperationTraits.Template.ixx"
        
        #---------------------------------------------------------------
        # Dnn / Compute / Devices (general)
        #---------------------------------------------------------------
        "Src/Dnn/Compute/Registry/DeviceRegistrar.ixx"
        "Src/Dnn/Compute/Registry/DeviceRegistry.ixx"
        "Src/Dnn/Compute/Registry/DeviceRegistryHelpers.ixx"
        
        #----------------------------------------------------------------------
        # Dnn / Data
        #----------------------------------------------------------------------
        "Src/Data/Loaders/DataLoader.ixx"
        "Src/Data/Loaders/TokenSequenceLoader.ixx"
        "Src/Data/Loaders/TokenSequenceLoader.Config.ixx"

        #---------------------------------------------------------------
        # Dnn / Serialization
        #---------------------------------------------------------------
        "Src/Dnn/Serialization/SerializationMode.ixx"
        "Src/Dnn/Serialization/OpenMode.ixx"
        "Src/Dnn/Serialization/ModelArchive.ixx"
        "Src/Dnn/Serialization/ArchiveSerializer.ixx"
        "Src/Dnn/Serialization/ZipSerializer.ixx"

        #---------------------------------------------------------------
        # Dnn / Visualization (FUTURE: moved to Dev/Visualization; see BACKLOG)
        #---------------------------------------------------------------
        #"Src/Dnn/Visualization/Components/Block.Visualizer.ixx"
        #"Src/Dnn/Visualization/Core/ComponentVisualizer.ixx"
        #"Src/Dnn/Visualization/Components/LayerNorm.Visualizer.ixx"
        #"Src/Dnn/Visualization/Core/VisualizerContext.ixx"
        #"Src/Dnn/Visualization/Rendering/FrameBuffer.ixx"
        #"Src/Dnn/Visualization/Rendering/ColorLUT.ixx"
        #"Src/Dnn/Visualization/Rendering/HeatMapRenderer.ixx"
        #"Src/Dnn/Visualization/Components/Block.Visualizer.ixx"
        #"Src/Dnn/Visualization/Components/MLP.Visualizer.ixx"
        
        #----------------------------------------------------------------------
        # Dnn / Compute / Operation Registry Helpers
        #----------------------------------------------------------------------
        "Src/Dnn/Optimizers/AdamW.ixx"
        "Src/Dnn/Optimizers/AdamWConfig.ixx"

        #----------------------------------------------------------------------
        # Network
        #----------------------------------------------------------------------
        "Src/Dnn/Core/Network.ixx"
        "Src/Dnn/Core/NetworkFactory.ixx"

        #----------------------------------------------------------------------
        # Dnn / Modeling 
        # NOTE: Modeling has been changed to Training. A Training loop abstraction
        # will be a future addition to Mila. Moved to Dev/Training for now.
        #----------------------------------------------------------------------
        #"Src/Dnn/Modeling/Model.ixx"
        #"Src/Dnn/Modeling/ModelConfig.ixx"
        #"Src/Dnn/Modeling/ModelMetadata.ixx"
        #
        #"Src/Dnn/Modeling/Checkpoints/CheckpointManager.ixx"
        #"Src/Dnn/Modeling/Checkpoints/CheckpointMetaData.ixx"
        
        #"Src/Dnn/Modeling/TrainingHistory.ixx"
        #"Src/Dnn/Modeling/ModelExporter.ixx"

        #----------------------------------------------------------------------
        # Dnn / Inference
        #----------------------------------------------------------------------
        #"Src/Dnn/Inference/InferenceEngine.ixx"
        
        #----------------------------------------------------------------------
        # Dnn / Extensibility
        #----------------------------------------------------------------------
        #"Src/Dnn/Extensibility/PluginManager.ixx"
        #"Src/Dnn/Extensibility/IModulePlugin.ixx"
        #"Src/Dnn/Extensibility/PluginInfo.ixx"
        
        "Src/Dnn/Components/MilaComponents.ixx"
        "Src/Dnn/Core/Loss.ixx"
        "Src/Dnn/Serialization/SerializationMetadata.ixx"
        
        "Src/Dnn/Tensors/Operations/TensorOps.Zero.ixx"

        "Src/Dnn/Core/Component.BuildContext.ixx"
        "Src/Dnn/Core/LearningRateScheduler.ixx"
        
        "Src/Dnn/Components/Activations/ApproximationMethod.ixx"
        "Src/Dnn/Components/Transformers/Gpt/Gpt.Presets.ixx"
        "Src/Dnn/Components/Attention/AttentionType.ixx"
        "Src/Dnn/Components/Normalization/NormType.ixx"
        "Src/Dnn/Components/Encodings/EncodingType.ixx"
        
        "Src/Dnn/Components/Transformers/ITransformerBlock.ixx"

        "Src/Dnn/Components/Transformers/Gpt/GptTransformer.ixx"
        
        "Src/Dnn/Components/Transformers/Gpt/Gpt.Config.ixx"
        "Src/Dnn/Components/Transformers/LlaMa/Llama.ixx"
        "Src/Dnn/Components/Transformers/LlaMa/Llama.Config.ixx"
        
        "Src/Dnn/Components/Transformers/LlaMa/Llama.Presets.ixx"


        "Src/Dnn/Components/Transformers/Gemma/Gemma.Config.ixx"
        "Src/Dnn/Components/Transformers/Gemma/Gemma.Block.Workspace.ixx"
        "Src/Dnn/Components/Transformers/Gemma/Gemma.Block.ixx"
        "Src/Dnn/Components/Transformers/Gemma/Gemma.ixx"

        "Src/Dnn/Components/Transformers/Qwen/Qwen.PrecisionPlan.ixx"
        "Src/Dnn/Components/Transformers/Qwen/Qwen.Config.ixx"
        "Src/Dnn/Components/Transformers/Qwen/Qwen.AttentionBlock.ixx"
        "Src/Dnn/Components/Transformers/Qwen/Qwen.DeltaNetBlock.ixx"
        "Src/Dnn/Components/Transformers/Qwen/Qwen.ixx"

        "Src/Data/Tokenizers/Tokenizer.ixx"
        "Src/Data/Tokenizers/TokenizerVocabulary.ixx"
        "Src/Data/Tokenizers/TokenizerType.ixx"

        "Src/Data/Core/FileHeader.ixx"        
        "Src/Data/Core/TokenizerTrainer.ixx"
        "Src/Data/Core/TrainerFactory.ixx"

        "Src/Data/Tokenizers/Char/CharTokenizer.ixx"
        "Src/Data/Tokenizers/Char/CharVocabularyConfig.ixx"
        "Src/Data/Tokenizers/Char/CharTrainer.ixx"
        "Src/Data/Tokenizers/Char/CharVocabulary.ixx"

        "Src/Data/Tokenizers/Bpe/BpeVocabularyConfig.ixx"
        "Src/Data/Tokenizers/Bpe/BpeVocabulary.ixx"
        "Src/Data/Tokenizers/Bpe/BpeTokenizer.ixx"
        "Src/Data/Tokenizers/Bpe/BpeTrainer.ixx"

        "Src/Data/Tokenizers/Bpe/BpePreTokenizationMode.ixx"

        "Src/Dnn/Components/Transformers/SamplingParams.ixx"
        "Src/Dnn/Components/Transformers/GenerateParams.ixx"
        "Src/Data/Tokenizers/SpecialTokens.ixx"
        "Src/Dnn/Serialization/SafeTensors.ixx"
        "Src/Dnn/Serialization/PretrainedReader.ixx"
        "Src/Dnn/Serialization/Serializer.ixx"
        
        "Src/Dnn/Compute/Operations/IKvCacheLifecycle.ixx"
        
        # The one runtime-to-compile-time quantization bridge, imported by the models below.
        "Src/Dnn/Models/Conversation.ixx"
        "Src/Dnn/Models/QuantizationDispatch.ixx"
        "Src/Dnn/Models/Gpt/GptModel.ixx"
        "Src/Dnn/Models/LlaMa/LlamaModel.ixx"
        "Src/Dnn/Models/Gemma/GemmaModel.ixx"
        "Src/Dnn/Models/Gemma/Gemma.Protocol.ixx"
        "Src/Dnn/Models/Qwen/QwenModel.ixx"
        "Src/Dnn/Models/Qwen/Qwen.Protocol.ixx"

        "Src/Dnn/Components/Transformers/LlaMa/Llama.Block.ixx"
        
        "Src/Dnn/Components/Attention/GQA/GroupedQueryAttention.ixx"
        "Src/Dnn/Components/Attention/GQA/GroupedQueryAttention.Config.ixx"

        "Src/Dnn/Components/Attention/OutputGate/AttentionOutputGate.Config.ixx"
        "Src/Dnn/Components/Attention/OutputGate/AttentionOutputGate.ixx"
        
        "Src/Dnn/Core/Component.MemoryStats.ixx"
        "Src/Dnn/Core/Model.RuntimeMode.ixx"
        "Src/Dnn/Core/Model.ixx"
        "Src/Dnn/Core/LanguageModelNetwork.ixx"
        "Src/Dnn/Core/LanguageModel.ixx"
        "Src/Dnn/Core/Component.TrainingMode.ixx"
        "Src/Dnn/Compute/Operations/IPositionalPairedOp.ixx"
        "Src/Dnn/Tensors/Operations/TensorOps.Random.ixx"
        "Src/Dnn/Tensors/Operations/TensorOps.Structural.ixx"
        "Src/Dnn/Compute/Operations/IPositionalDecode.ixx"
        "Src/Dnn/Compute/Operations/IKvInference.ixx"
        "Src/Dnn/Compute/Operations/IPackedKvInference.ixx"
        "Src/Dnn/Core/TokenStreamer.ixx"
        #"Src/Dnn/Compute/Operations/LinearOpTypeMap.Template.ixx"
        #"Src/Dnn/Compute/Operations/LinearOpTypeMap.ixx"
        #"Src/Dnn/Core/ModelQuantizationConfig.ixx"
        "Src/Dnn/Core/ModelConfig.ixx"
        "Src/Dnn/Models/LlaMa/LlamaModelConfig.ixx"
        "Src/Dnn/Models/Gemma/GemmaModelConfig.ixx"
        "Src/Dnn/Models/Qwen/QwenModelConfig.ixx"
        #"Src/Dnn/Compute/Operations/GqaOpTypeMap.ixx"
        #"Src/Dnn/Compute/Operations/GqaOpTypeMap.Template.ixx"
        "Src/Dnn/Compute/Operations/GqaState.ixx"
        
        "Src/Logging/NullSink.ixx"
        "Src/Logging/FileSink.ixx"
        
        "Src/Dnn/Quantization/Weight/Policies.ixx"
        "Src/Dnn/Quantization/Weight/PrecisionPlan.ixx"
        "Src/Dnn/Quantization/KvCache/Policy.ixx"
        "Src/Dnn/Quantization/KvCache/QuantPolicy.ixx"
        "Src/Dnn/Quantization/Quantization.ixx"

        "Src/Dnn/Quantization/Weight/CodebookPacking.ixx"
        "Src/Dnn/Core/LanguageModelConfig.ixx"
        
        "Src/Dnn/Core/SamplerBase.ixx"
        "Src/Dnn/Samplers/SamplingConfig.ixx"
        "Src/Dnn/Samplers/TokenSampler.ixx"
 "Src/Dnn/Models/Gpt/GptModelConfig.ixx" "Src/Dnn/Core/GenerateStatus.ixx")

# ----------------------------------------------------------------------------
# CPU backend modules
# ----------------------------------------------------------------------------
if(MILA_HAS_CPU)
    target_sources( Mila
        PUBLIC
            FILE_SET cpu_modules
               TYPE CXX_MODULES FILES
                "Src/Dnn/Compute/Devices/Cpu/CpuDevice.ixx"
                "Src/Dnn/Compute/Devices/Cpu/CpuMemoryResource.ixx"
                "Src/Dnn/Compute/Devices/Cpu/CpuExecutionContext.ixx"
                "Src/Dnn/Compute/Devices/Cpu/CpuDeviceTypeTraits.ixx"
                
                "Src/Dnn/Compute/Devices/Cpu/Tensors/CpuTensorDataTypeTraits.ixx"
                "Src/Dnn/Compute/Devices/Cpu/Tensors/Operations/CpuTensorOps.ixx"
                "Src/Dnn/Compute/Devices/Cpu/Tensors/Operations/CpuTensorOps.Fill.ixx"
                "Src/Dnn/Compute/Devices/Cpu/Tensors/Operations/CpuTensorOps.Math.ixx"
                "Src/Dnn/Compute/Devices/Cpu/Tensors/Operations/CpuTensorOps.Transfer.ixx"
                "Src/Dnn/Compute/Devices/Cpu/Tensors/Operations/CpuTensorOps.Random.ixx"
                
                "Src/Dnn/Compute/Devices/Cpu/Operations/CpuOperations.ixx"
                "Src/Dnn/Compute/Devices/Cpu/Operations/OperationTraits.Cpu.ixx"
                "Src/Dnn/Compute/Devices/Cpu/Operations/CpuEncoderOp.ixx"
                "Src/Dnn/Compute/Devices/Cpu/Operations/CpuGeluOp.ixx"
                "Src/Dnn/Compute/Devices/Cpu/Operations/CpuElementwiseActivationOp.ixx"
                "Src/Dnn/Compute/Devices/Cpu/Operations/CpuLayerNormOp.ixx"
                "Src/Dnn/Compute/Devices/Cpu/Operations/CpuLinearOp.ixx"
                "Src/Dnn/Compute/Devices/Cpu/Operations/CpuAttentionOp.ixx"
                "Src/Dnn/Compute/Devices/Cpu/Operations/CpuResidualOp.ixx"
                #"Src/Dnn/Compute/Devices/Cpu/Operations/CpuSoftmaxCrossEntropyOp.ixx"
                "Src/Dnn/Compute/Devices/Cpu/Operations/CpuSoftmaxOp.ixx"
                "Src/Dnn/Compute/Devices/Cpu/Operations/CpuSamplingOp.ixx"
                "Src/Dnn/Compute/Devices/Cpu/Optimizers/CpuAdamWOptimizer.ixx"
                "Src/Dnn/Compute/Devices/Cpu/Tensors/Operations/CpuTensorOps.Zero.ixx"

                #"Src/Dnn/Compute/Devices/Cpu/Operations/CpuLinearOpTypeMap.ixx"
                #"Src/Dnn/Compute/Devices/Cuda/Operations/Attention/GQA/CudaGqaOpTypeMap.ixx"
    )

    target_compile_definitions( Mila PUBLIC MILA_HAS_CPU)

    message(STATUS "Added CPU backend modules")
endif()

# ----------------------------------------------------------------------------
# CUDA backend modules
# ----------------------------------------------------------------------------
if(MILA_HAS_CUDA)
    # Compiled kernel implementations: private to the static archive, never shipped.
    # A consumer links these symbols out of libMila; a .cu is not a header and has no
    # place in the install tree. Explicit list (not a glob) so intentionally excluded
    # files - e.g. the superseded Swiglu.Fp32.v1.cu - stay out by omission.
    target_sources(Mila
        PRIVATE
            "Src/Dnn/Compute/Devices/Cuda/Operations/Embeddings/Kernels/TokenEmbedding.Fp32.cu"
            "Src/Dnn/Compute/Devices/Cuda/Operations/Embeddings/Kernels/TokenEmbedding.Bf16.cu"
            "Src/Dnn/Compute/Devices/Cuda/Operations/Embeddings/Kernels/TokenEmbedding.Fp8.cu"
            "Src/Dnn/Compute/Devices/Cuda/Operations/Encodings/Lpe/Kernels/Lpe.Fp32.cu"
            "Src/Dnn/Compute/Devices/Cuda/Operations/Encodings/Lpe/Kernels/Lpe.Fp16.cu"
            "Src/Dnn/Compute/Devices/Cuda/Operations/Encodings/Rope/Kernels/Rope.Fp32.cu"
            "Src/Dnn/Compute/Devices/Cuda/Operations/Encodings/Rope/Kernels/Rope.Bf16.cu"
            "Src/Dnn/Compute/Devices/Cuda/Operations/Attention/Common/Kernels/CudaAttention.Unpermute.Fp32.cu"
            "Src/Dnn/Compute/Devices/Cuda/Operations/Attention/Common/Kernels/CudaAttention.Unpermute.Bf16.cu"
            "Src/Dnn/Compute/Devices/Cuda/Operations/Attention/Common/Kernels/CudaAttention.Softmax.Fp32.cu"
            "Src/Dnn/Compute/Devices/Cuda/Operations/Attention/Common/Kernels/CudaAttention.Softmax.Bf16.cu"
            "Src/Dnn/Compute/Devices/Cuda/Operations/Attention/MHA/Kernels/CudaMha.Permute.cu"
            "Src/Dnn/Compute/Devices/Cuda/Operations/Attention/MHA/Kernels/CudaMha.Softmax.cu"
            "Src/Dnn/Compute/Devices/Cuda/Operations/Attention/GQA/Kernels/CudaGqa.Permute.cu"
            "Src/Dnn/Compute/Devices/Cuda/Operations/Attention/GQA/Kernels/Gqa.Cache.Fp32.cu"
            "Src/Dnn/Compute/Devices/Cuda/Operations/Attention/GQA/Kernels/Gqa.Cache.Bf16.cu"
            "Src/Dnn/Compute/Devices/Cuda/Operations/Attention/GQA/Kernels/Gqa.Decode.Fp32.cu"
            "Src/Dnn/Compute/Devices/Cuda/Operations/Attention/GQA/Kernels/Gqa.Decode.Bf16.cu"
            "Src/Dnn/Compute/Devices/Cuda/Operations/Attention/GQA/Kernels/Gqa.Prefill.Fp32.cu"
            "Src/Dnn/Compute/Devices/Cuda/Operations/Attention/GQA/Kernels/Gqa.Prefill.Bf16.cu"
            # Gqa.Flash.Bf16.cu (scalar Iteration 2) retired from the build -- profiled a
            # non-fix (LSU-bound, GqaFlashAttention.md 5.2.1). Superseded by the WMMA path
            # below; file kept in-tree for reference. Both define cuda_gqa_flash_prefill_bf16,
            # so exactly one is compiled.
            "Src/Dnn/Compute/Devices/Cuda/Operations/Attention/GQA/Kernels/Gqa.Flash.Wmma.cu"
            # Gqa.Flash.Fa2.cu owns cuda_gqa_flash_prefill_ring_bf16 (the HS=256 local
            # sliding layers) as a row-split FA-2 kernel; Wmma.cu keeps only the HS=512
            # unbounded global symbol. Exactly one TU defines each flash entry point.
            "Src/Dnn/Compute/Devices/Cuda/Operations/Attention/GQA/Kernels/Gqa.Flash.Fa2.cu"
            "Src/Dnn/Compute/Devices/Cuda/Operations/Attention/GQA/Kernels/Gqa.Train.Fp32.cu"
            "Src/Dnn/Compute/Devices/Cuda/Operations/Attention/GQA/Kernels/Gqa.Train.Bf16.cu"
            "Src/Dnn/Compute/Devices/Cuda/Operations/Linear/Kernels/Fp8Prefill/CudaFp8Prefill.cu"
            "Src/Dnn/Compute/Devices/Cuda/Operations/Linear/Kernels/MatMul/CudaMatMulFp32.cu"
            "Src/Dnn/Compute/Devices/Cuda/Operations/Linear/Kernels/MatMul/CudaMatMulFp16.cu"
            "Src/Dnn/Compute/Devices/Cuda/Operations/Linear/Kernels/MatMul/CudaLinearGelu.cu"
            "Src/Dnn/Compute/Devices/Cuda/Operations/Linear/Kernels/MatMul/CudaReduction.cu"
            "Src/Dnn/Compute/Devices/Cuda/Operations/Linear/Kernels/MatVec/CudaMatVecBias.Fp32.cu"
            "Src/Dnn/Compute/Devices/Cuda/Operations/Linear/Kernels/MatVec/CudaMatVecBias.Bf16.cu"
            "Src/Dnn/Compute/Devices/Cuda/Operations/Linear/Kernels/W8A16Gemm/CudaW8A16Gemm.cu"
            "Src/Dnn/Compute/Devices/Cuda/Operations/Linear/Kernels/W4A16Gemm/CudaW4A16Gemm.cu"
            "Src/Dnn/Compute/Devices/Cuda/Operations/Linear/Kernels/W4A16Gemm/CudaW4A16Gemm.Wmma.cu"
            "Src/Dnn/Compute/Devices/Cuda/Operations/Linear/Kernels/Quantization/CudaFp8WeightQuantization.cu"
            "Src/Dnn/Compute/Devices/Cuda/Operations/Linear/Kernels/Quantization/CudaFp4WeightQuantization.cu"
            "Src/Dnn/Compute/Devices/Cuda/Operations/Linear/Kernels/Codebook/CodebookGemv.cu"
            "Src/Dnn/Compute/Devices/Cuda/Operations/Linear/Kernels/Codebook/CodebookDequantize.cu"
            "Src/Dnn/Compute/Devices/Cuda/Operations/Activations/Gelu/Kernels/Gelu.Fp16.cu"
            "Src/Dnn/Compute/Devices/Cuda/Operations/Activations/Gelu/Kernels/Gelu.Fp32.cu"
            "Src/Dnn/Compute/Devices/Cuda/Operations/Activations/Swiglu/Kernels/Swiglu.Fp32.cu"
            "Src/Dnn/Compute/Devices/Cuda/Operations/Activations/Swiglu/Kernels/Swiglu.Bf16.cu"
            "Src/Dnn/Compute/Devices/Cuda/Operations/Activations/Geglu/Kernels/Geglu.cu"
            "Src/Dnn/Compute/Devices/Cuda/Operations/Activations/Elementwise/Kernels/ElementwiseActivation.cu"
            "Src/Dnn/Compute/Devices/Cuda/Operations/Normalizations/LayerNorm/Kernels/LayerNorm.Fp32.cu"
            "Src/Dnn/Compute/Devices/Cuda/Operations/Normalizations/LayerNorm/Kernels/LayerNorm.Fp16.cu"
            "Src/Dnn/Compute/Devices/Cuda/Operations/Normalizations/RmsNorm/Kernels/RmsNorm.Fp32.cu"
            "Src/Dnn/Compute/Devices/Cuda/Operations/Normalizations/RmsNorm/Kernels/RmsNorm.Bf16.cu"
            "Src/Dnn/Compute/Devices/Cuda/Operations/Normalizations/Softmax/Kernels/Softmax.Fp32.cu"
            "Src/Dnn/Compute/Devices/Cuda/Operations/Normalizations/Softmax/Kernels/Softmax.Fp16.cu"
            "Src/Dnn/Compute/Devices/Cuda/Operations/Sampling/Kernels/Sampling.cu"
            "Src/Dnn/Compute/Devices/Cuda/Operations/Residual/Kernels/Residual.Fp32.cu"
            "Src/Dnn/Compute/Devices/Cuda/Operations/Residual/Kernels/Residual.Bf16.cu"
            "Src/Dnn/Compute/Devices/Cuda/Operations/Convolutions/Kernels/CausalConv1d.Fp32.cu"
            "Src/Dnn/Compute/Devices/Cuda/Operations/Convolutions/Kernels/CausalConv1d.Bf16.cu"
            "Src/Dnn/Compute/Devices/Cuda/Operations/DeltaNet/Kernels/GatedDeltaRule.cu"
            "Src/Dnn/Compute/Devices/Cuda/Operations/Loss/Kernels/SoftmaxCrossEntropy.Fp32.warp.cu"
            "Src/Dnn/Compute/Devices/Cuda/Operations/Loss/Kernels/SoftmaxCrossEntropy.Fp32.block.cu"
            "Src/Dnn/Compute/Devices/Cuda/Operations/Loss/Kernels/SoftmaxCrossEntropy.cu"
            "Src/Dnn/Compute/Devices/Cuda/Tensors/Operations/Kernels/TensorCopy.cu"
            "Src/Dnn/Compute/Devices/Cuda/Tensors/Operations/Kernels/TensorOps.Fill.cu"
            "Src/Dnn/Compute/Devices/Cuda/Tensors/Operations/Kernels/Math.Elementwise.cu"
            "Src/Dnn/Compute/Devices/Cuda/Tensors/Operations/Kernels/Structural.cu"
            "Src/Dnn/Compute/Devices/Cuda/Tensors/Operations/Kernels/Random.cu"
            "Src/Dnn/Compute/Devices/Cuda/Optimizers/Kernels/CudaAdamW.cu"
    )

    # Kernel declaration headers: shipped, because the module units #include them with
    # file-relative quoted paths and the consumer recompiles those modules from source
    # (C++23 BMIs are not portable). A HEADERS file set gives them a BASE_DIRS and real
    # install semantics. BASE_DIRS matches the module file sets so headers install at the
    # same Src-relative offset and the quoted includes keep resolving post-install. The
    # full set is shipped (robust over minimal); new headers are placed here via the VS
    # add-file dialog.
    target_sources(Mila
        PUBLIC
            FILE_SET cuda_headers
               TYPE HEADERS
               BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} FILES
                "Src/Dnn/Compute/Devices/Cuda/Helpers/CudaUtils.h"
                "Src/Dnn/Compute/Devices/Cuda/Operations/CudaOps.h"
                "Src/Dnn/Compute/Devices/Cuda/Operations/Embeddings/Kernels/TokenEmbedding.cuh"
                "Src/Dnn/Compute/Devices/Cuda/Operations/Encodings/Lpe/Kernels/Lpe.cuh"
                "Src/Dnn/Compute/Devices/Cuda/Operations/Encodings/Rope/Kernels/Rope.cuh"
                "Src/Dnn/Compute/Devices/Cuda/Operations/Attention/Common/Kernels/CudaAttention.cuh"
                "Src/Dnn/Compute/Devices/Cuda/Operations/Attention/MHA/Kernels/CudaMha.cuh"
                "Src/Dnn/Compute/Devices/Cuda/Operations/Attention/GQA/Kernels/CudaGqa.cuh"
                "Src/Dnn/Compute/Devices/Cuda/Operations/Linear/Kernels/Linear.cuh"
                "Src/Dnn/Compute/Devices/Cuda/Operations/Linear/Kernels/Fp8Prefill/CudaFp8Prefill.cuh"
                "Src/Dnn/Compute/Devices/Cuda/Operations/Linear/Kernels/W8A16Gemm/CudaW8A16Gemm.cuh"
                "Src/Dnn/Compute/Devices/Cuda/Operations/Linear/Kernels/W4A16Gemm/CudaW4A16Gemm.cuh"
                "Src/Dnn/Compute/Devices/Cuda/Operations/Linear/Kernels/W4A16Gemm/CudaW4A16Gemm.Wmma.cuh"
                "Src/Dnn/Compute/Devices/Cuda/Operations/Linear/Kernels/Quantization/CudaFp8WeightQuantization.cuh"
                "Src/Dnn/Compute/Devices/Cuda/Operations/Linear/Kernels/Quantization/CudaFp4WeightQuantization.cuh"
                "Src/Dnn/Compute/Devices/Cuda/Operations/Linear/Kernels/Codebook/CodebookGemv.cuh"
                "Src/Dnn/Compute/Devices/Cuda/Operations/Linear/Kernels/Codebook/CodebookDequantize.cuh"
                "Src/Dnn/Compute/Devices/Cuda/Operations/Activations/Gelu/Kernels/Gelu.cuh"
                "Src/Dnn/Compute/Devices/Cuda/Operations/Activations/Swiglu/Kernels/Swiglu.cuh"
                "Src/Dnn/Compute/Devices/Cuda/Operations/Activations/Geglu/Kernels/Geglu.cuh"
                "Src/Dnn/Compute/Devices/Cuda/Operations/Activations/Elementwise/Kernels/ElementwiseActivation.cuh"
                # Shared elementwise-activation functor library (consumed by the CPU op, the
                # Activation component, and the CUDA kernel; shipped here for CUDA consumers).
                "Src/Dnn/Components/Activations/Activation/Kernels/ElementwiseActivation.h"
                "Src/Dnn/Compute/Devices/Cuda/Operations/Normalizations/LayerNorm/Kernels/LayerNorm.cuh"
                "Src/Dnn/Compute/Devices/Cuda/Operations/Normalizations/RmsNorm/Kernels/RmsNorm.cuh"
                "Src/Dnn/Compute/Devices/Cuda/Operations/Normalizations/Softmax/Kernels/Softmax.cuh"
                "Src/Dnn/Compute/Devices/Cuda/Operations/Sampling/Kernels/Sampling.cuh"
                "Src/Dnn/Compute/Devices/Cuda/Operations/Residual/Kernels/Residual.cuh"
                "Src/Dnn/Compute/Devices/Cuda/Operations/Convolutions/Kernels/CausalConv1d.cuh"
                "Src/Dnn/Compute/Devices/Cuda/Operations/DeltaNet/Kernels/GatedDeltaRule.cuh"
                "Src/Dnn/Compute/Devices/Cuda/Operations/Loss/Kernels/SoftmaxCrossEntropy.cuh"
                "Src/Dnn/Compute/Devices/Cuda/Tensors/Operations/Kernels/Transfer.Copy.h"
                "Src/Dnn/Compute/Devices/Cuda/Tensors/Operations/Kernels/TensorOps.Fill.h"
                "Src/Dnn/Compute/Devices/Cuda/Tensors/Operations/Kernels/CudaTensorOps.h"
                "Src/Dnn/Compute/Devices/Cuda/Tensors/Operations/Kernels/Math.Elementwise.h"
                "Src/Dnn/Compute/Devices/Cuda/Tensors/Operations/Kernels/Math.Reduction.h"
                "Src/Dnn/Compute/Devices/Cuda/Tensors/Operations/Kernels/Structural.h"
                "Src/Dnn/Compute/Devices/Cuda/Tensors/Operations/Kernels/Random.h"
                "Src/Dnn/Compute/Devices/Cuda/Optimizers/Kernels/CudaOptimizers.h"
    )

    target_sources(Mila
        PUBLIC
            FILE_SET cuda_modules
               TYPE CXX_MODULES FILES
                "Src/Dnn/Compute/Devices/Cuda/CudaDeviceTypeTraits.ixx"
                "Src/Dnn/Compute/Devices/Cuda/Helpers/CudaError.ixx"
                "Src/Dnn/Compute/Devices/Cuda/Helpers/CudaHelpers.ixx"
                "Src/Dnn/Compute/Devices/Cuda/Helpers/CudaBadAlloc.ixx"
                "Src/Dnn/Compute/Devices/Cuda/Helpers/CublasLtError.ixx"
                
                "Src/Dnn/Compute/Devices/Cuda/CudaDeviceProps.ixx"
                "Src/Dnn/Compute/Devices/Cuda/CudaDevice.ixx"

                "Src/Dnn/Compute/Devices/Cuda/CudaExecutionContext.ixx"
                "Src/Dnn/Compute/Devices/Cuda/Profiling/CudaTimer.ixx"

                "Src/Dnn/Compute/Devices/Cuda/CudaDeviceMemoryResource.ixx"
                "Src/Dnn/Compute/Devices/Cuda/CudaManagedMemoryResource.ixx"
                "Src/Dnn/Compute/Devices/Cuda/CudaPinnedMemoryResource.ixx"
                
                "Src/Dnn/Compute/Devices/Cuda/Tensors/CudaTensorDataType.ixx"
                "Src/Dnn/Compute/Devices/Cuda/Tensors/CudaTensorDataType-Specializations.ixx"
                "Src/Dnn/Compute/Devices/Cuda/Tensors/CudaTensorDataType-Maps.ixx"
                "Src/Dnn/Compute/Devices/Cuda/Tensors/Operations/CudaTensorOps.ixx"
                "Src/Dnn/Compute/Devices/Cuda/Tensors/Operations/CudaTensorOps.Math.ixx"
                "Src/Dnn/Compute/Devices/Cuda/Tensors/Operations/CudaTensorOps.Fill.ixx"
                "Src/Dnn/Compute/Devices/Cuda/Tensors/Operations/CudaTensorOps.Transfer.ixx"
                
                "Src/Dnn/Compute/Devices/Cuda/Operations/OperationTraits.Cuda.ixx"

                # CUDA Attention Ops
                "Src/Dnn/Compute/Devices/Cuda/Operations/Attention/GQA/CudaGqaOp.ixx"
                "Src/Dnn/Compute/Devices/Cuda/Operations/Attention/GQA/CudaGqa.Plans.ixx"
                "Src/Dnn/Compute/Devices/Cuda/Operations/Attention/GQA/CudaGqa.Dispatch.ixx"
    
                "Src/Dnn/Compute/Devices/Cuda/Operations/Attention/MHA/CudaMhaOp.ixx"
                "Src/Dnn/Compute/Devices/Cuda/Operations/Attention/MHA/CudaMhaOp.Plans.ixx"
                "Src/Dnn/Compute/Devices/Cuda/Operations/Attention/MHA/CudaMhaOp.Dispatch.ixx"

                "Src/Dnn/Compute/Devices/Cuda/Operations/Embeddings/CudaTokenEmbeddingOp.ixx"
                "Src/Dnn/Compute/Devices/Cuda/Operations/Embeddings/CudaTokenEmbeddingOp.Dispatch.ixx"
                "Src/Dnn/Compute/Devices/Cuda/Operations/Embeddings/CudaTokenEmbeddingOp.Quantize.ixx"
                
                "Src/Dnn/Compute/Devices/Cuda/Operations/Encodings/Lpe/CudaLpeOp.ixx"
                "Src/Dnn/Compute/Devices/Cuda/Operations/Encodings/Lpe/CudaLpeOp.Dispatch.ixx"
                
                "Src/Dnn/Compute/Devices/Cuda/Operations/Encodings/Rope/CudaRopeOp.ixx"
                "Src/Dnn/Compute/Devices/Cuda/Operations/Encodings/Rope/CudaRopeOp.Dispatch.ixx"
                "Src/Dnn/Compute/Devices/Cuda/Operations/Encodings/Rope/CudaRopeOp.Cache.ixx"

                "Src/Dnn/Compute/Devices/Cuda/Operations/Linear/CudaLinearOp.ixx"
                "Src/Dnn/Compute/Devices/Cuda/Operations/Linear/CodebookGemv.ixx"
                "Src/Dnn/Compute/Devices/Cuda/Operations/Activations/Gelu/CudaGeluOp.ixx"
                "Src/Dnn/Compute/Devices/Cuda/Operations/Activations/Swiglu/CudaSwigluOp.ixx"
                "Src/Dnn/Compute/Devices/Cuda/Operations/Activations/Geglu/CudaGegluOp.ixx"
                "Src/Dnn/Compute/Devices/Cuda/Operations/Activations/Elementwise/CudaElementwiseActivationOp.ixx"
                "Src/Dnn/Compute/Devices/Cuda/Operations/Normalizations/LayerNorm/LayerNormOp.ixx"
                "Src/Dnn/Compute/Devices/Cuda/Operations/Normalizations/LayerNorm/LayerNormOp.Dispatch.ixx"
                "Src/Dnn/Compute/Devices/Cuda/Operations/Normalizations/Softmax/CudaSoftmaxOp.ixx"
                "Src/Dnn/Compute/Devices/Cuda/Operations/Sampling/CudaSamplingOp.ixx"
                "Src/Dnn/Compute/Devices/Cuda/Operations/Residual/CudaResidualOp.ixx"
                "Src/Dnn/Compute/Devices/Cuda/Operations/Convolutions/CudaCausalConv1dOp.ixx"
                "Src/Dnn/Compute/Devices/Cuda/Operations/DeltaNet/CudaGatedDeltaRuleOp.ixx"
                
                "Src/Dnn/Compute/Devices/Cuda/Operations/Loss/CudaSoftmaxCrossEntropyOp.ixx"
                "Src/Dnn/Compute/Devices/Cuda/Operations/Loss/CudaSoftmaxCrossEntropyOp.Dispatch.ixx"

                "Src/Dnn/Compute/Devices/Cuda/Optimizers/CudaAdamWOptimizer.ixx"
                "Src/Dnn/Compute/Devices/Cuda/Tensors/Operations/CudaTensorOps.Zero.ixx"
                "Src/Dnn/Compute/Devices/Cuda/Helpers/CudaDebug.ixx"
                
                "Src/Dnn/Compute/Devices/Cuda/Operations/Normalizations/RmsNorm/RmsNormOp.ixx"
                "Src/Dnn/Compute/Devices/Cuda/Operations/Normalizations/RmsNorm/RmsNormOp.Dispatch.ixx"
                "Src/Dnn/Compute/Devices/Cuda/Operations/Activations/Swiglu/CudaSwigluOp.Dispatch.ixx"
                "Src/Dnn/Compute/Devices/Cuda/Tensors/Operations/CudaTensorOps.Random.ixx"
                "Src/Dnn/Compute/Devices/Cuda/Tensors/Operations/CudaTensorOps.Structural.ixx"
                "Src/Dnn/Compute/Devices/Cuda/Helpers/CublasLt.Utils.ixx"
                #"Src/Dnn/Compute/Devices/Cuda/Tensors/Operations/Kernels/Random.h"
                #"Src/Dnn/Compute/Devices/Cuda/Operations/Linear/CudaLinearOpTypeMap.ixx"
                "Src/Dnn/Compute/Devices/Cuda/Operations/Common/CublasLtLinearPlan.ixx"

                # Relocated out of the core module set (Mila core file set): these are
                # CUDA modules that were erroneously listed among the always-compiled
                # core modules, which broke a CPU-only (MILA_ENABLE_CUDA=OFF) configure.
                # They belong with the rest of the CUDA backend and only build when CUDA
                # is enabled.
                "Src/Dnn/Compute/Devices/Cuda/Operations/Common/CublasLtPlan.ixx"
                "Src/Dnn/Compute/Devices/Cuda/Operations/Common/CublasLtPlanCache.ixx"
                "Src/Dnn/Compute/Devices/Cuda/Operations/Residual/CudaResidualOp.Dispatch.ixx"
                "Src/Dnn/Compute/Devices/Cuda/Operations/Convolutions/CudaCausalConv1dOp.Dispatch.ixx"
                "Src/Dnn/Compute/Devices/Cuda/Operations/DeltaNet/CudaGatedDeltaRuleOp.Dispatch.ixx"
                "Src/Dnn/Compute/Devices/Cuda/Operations/Activations/Gelu/CudaGeluOp.Dispatch.ixx"
                "Src/Dnn/Compute/Devices/Cuda/Operations/Linear/CudaLinearOp.Dispatch.ixx"
                "Src/Dnn/Compute/Devices/Cuda/Operations/Linear/CudaLinearOp.Plans.ixx"
                "Src/Dnn/Compute/Devices/Cuda/Operations/Linear/CudaLinearOp.Quantize.ixx"
                "Src/Dnn/Compute/Devices/Cuda/Profiling/NvtxRange.ixx"
                "Src/Dnn/Compute/Devices/Cuda/Tensors/CudaTensorDataTypes-CublasLtTypes.ixx"
    )

    # PUBLIC (like the Metal/ROCm backends below), not PRIVATE: Mila is a STATIC
    # library, so a consumer must (1) resolve cudart/cublas/curand symbols from
    # Mila's .cu objects at its own final link, and (2) get the CUDA toolkit include
    # dir as a usage requirement so the target that compiles Mila's CUDA module
    # units can find <cuda_fp16.h> etc.
    target_link_libraries( Mila PUBLIC
        CUDA::cudart
        CUDA::cublas
        CUDA::curand
        CUDA::nvtx3
    )

    target_compile_definitions(Mila PUBLIC MILA_HAS_CUDA)

    target_include_directories(Mila PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/Src/Dnn/Compute/Devices/Cuda/Helpers"
    )

    message(STATUS "Added CUDA backend modules")
endif()

# ----------------------------------------------------------------------------
# Metal backend modules
# ----------------------------------------------------------------------------
if(MILA_HAS_METAL)
    target_sources(Mila
        PUBLIC
            FILE_SET metal_modules
               TYPE CXX_MODULES FILES
                "Src/Dnn/Compute/Devices/Metal/MetalDevice.ixx"
                "Src/Dnn/Compute/Devices/Metal/MetalExecutionContext.ixx"
                "Src/Dnn/Compute/Devices/Metal/MetalDevicePlugin.ixx"
                "Src/Dnn/Compute/Devices/Metal/MetalMemoryResource.ixx"
                "Src/Dnn/Compute/Devices/Metal/Tensors/MetalTensorTraits.ixx"
    )
    
    target_link_libraries(Mila PUBLIC 
        ${METAL_FRAMEWORK}
        ${METALPERFORMANCESHADERS_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
    )

    target_compile_definitions(Mila PUBLIC MILA_HAS_METAL )

    message(STATUS "Added Metal backend module partition (STUB)")
endif()

# ----------------------------------------------------------------------------
# Model distribution -- everything that does not move bytes
#
# The split is by dependency, not by theme, and the only optional thing is the
# transport. The store, schema, packaging and install are filesystem-only;
# ModelResolver::pull composes an IModelHub with the store; and HuggingFaceHub is
# here too, because URL shapes, the token file, the listing quirks and the meaning
# of a 403 are knowledge, not I/O. A build with no libcurl therefore still knows
# everything about HuggingFace and needs only bytes supplied from elsewhere -- which
# is exactly what the Python wheel does. See Specifications/ModelDistribution.md.
# ----------------------------------------------------------------------------
target_sources(Mila
    PUBLIC
        FILE_SET module_files
        TYPE CXX_MODULES FILES
            "Src/Distribution/Sha256.ixx"
            "Src/Distribution/Environment.ixx"
            "Src/Distribution/ModelCoordinate.ixx"
            "Src/Distribution/ModelManifest.ixx"
            "Src/Distribution/ModelPackage.ixx"
            "Src/Distribution/ModelStore.ixx"
            "Src/Distribution/ModelHub.ixx"
            "Src/Distribution/ModelResolver.ixx"
            "Src/Distribution/HttpTransport.ixx"
            "Src/Distribution/HttpClient.ixx"
            "Src/Distribution/HuggingFaceHub.ixx"
)

# ----------------------------------------------------------------------------
# Model distribution -- the HTTP transport
#
# Distribution.HttpTransportBackend has two candidate source files and exactly one
# is compiled. Selecting by source file rather than by #ifdef means the OFF path is
# compiled code that a build actually checks, and `import Mila;` never varies by a
# preprocessor define. This is the whole of what libcurl is needed for.
# ----------------------------------------------------------------------------
if(MILA_ENABLE_LIBCURL)
    target_sources(Mila
        PUBLIC
            FILE_SET module_files
            TYPE CXX_MODULES FILES
                "Src/Distribution/CurlHttpTransport.ixx"
                "Src/Distribution/HttpTransportBackend.Curl.ixx"
    )

    # PRIVATE: libcurl is an implementation detail. Its headers appear only in a global
    # module fragment, so no consumer needs them.
    target_link_libraries(Mila PRIVATE CURL::libcurl)

    # No MILA_HAS_* define: a PUBLIC macro that decides what a module exports makes the BMI
    # depend on the preprocessor. Consumers ask Distribution::kHttpTransportAvailable
    # instead, which is an exported constant the compiler checks.
    message(STATUS "Added HTTP transport: libcurl")
else()
    target_sources(Mila
        PUBLIC
            FILE_SET module_files
            TYPE CXX_MODULES FILES
                "Src/Distribution/HttpTransportBackend.Null.ixx"
    )

    message(STATUS "Added HTTP transport: none (caller supplies one, or install from disk)")
endif()

# ----------------------------------------------------------------------------
# ROCm backend Modules
# ----------------------------------------------------------------------------
if(MILA_HAS_ROCM)
    target_sources(Mila
        PUBLIC
            FILE_SET rocm_modules
               TYPE CXX_MODULES FILES
                "Src/Dnn/Compute/Devices/Rocm/RocmExecutionContext.ixx"
                "Src/Dnn/Compute/Devices/Rocm/RocmMemoryResource.ixx"
                "Src/Dnn/Compute/Devices/Rocm/Tensors/RocmTensorTraits.ixx"
    )
    
    target_link_libraries( Mila PUBLIC 
        hip::host
        roc::rocblas
    )
    
    target_compile_definitions( Mila PUBLIC MILA_HAS_ROCM )

    message(STATUS "Added ROCm backend modules")
endif()

# ============================================================================
# Interface Component Targets (for user convenience)
# ============================================================================

# Convenience "everything" target
add_library(Mila::Mila ALIAS Mila)

# CPU component
if(MILA_HAS_CPU)
    add_library(Mila_Cpu INTERFACE)
    target_link_libraries(Mila_Cpu INTERFACE Mila)
    add_library(Mila::Cpu ALIAS Mila_Cpu)
endif()

# CUDA component
if(MILA_HAS_CUDA)
    add_library(Mila_Cuda INTERFACE)
    target_link_libraries(Mila_Cuda INTERFACE Mila)
    add_library(Mila::Cuda ALIAS Mila_Cuda)
endif()

# Metal component
if(MILA_HAS_METAL)
    add_library(Mila_Metal INTERFACE)
    target_link_libraries(Mila_Metal INTERFACE Mila)
    add_library(Mila::Metal ALIAS Mila_Metal)
endif()

# ROCm component
if(MILA_HAS_ROCM)
    add_library(Mila_Rocm INTERFACE)
    target_link_libraries(Mila_Rocm INTERFACE Mila)
    add_library(Mila::Rocm ALIAS Mila_Rocm)
endif()

# PUBLIC, not PRIVATE: Mila is a STATIC library, so a find_package consumer must
# resolve miniz's mz_zip_* symbols (referenced by ZipSerializer) at its own final
# link. Header propagation for the recompiled module is set up just below.
target_link_libraries( Mila
  PUBLIC
    miniz
    #sentencepiece-static
)

# ZipSerializer.ixx is re-exported by Mila.ixx, so every consumer recompiles it; it
# includes <miniz.h>, which is multi-header (pulls miniz_common/_tdef/_tinfl/_zip.h)
# plus a generated miniz_export.h in miniz's build dir. Propagate both the source and
# build include dirs so a subproject consumer's recompile resolves them.
target_include_directories( Mila INTERFACE
    $<BUILD_INTERFACE:${miniz_SOURCE_DIR}>
    $<BUILD_INTERFACE:${miniz_BINARY_DIR}>
)

# =============================================================================
# Versioning
# =============================================================================

# The version is parsed from Version.txt into project(VERSION ...) at the repo
# root, which populates PROJECT_VERSION_* here. The prerelease label cannot live
# in project(VERSION) (CMake accepts only numeric major.minor.patch), so it is
# carried in MILA_PRERELEASE_TAG (also set at the root).
#
# Both are surfaced to source as PUBLIC compile definitions rather than a
# generated, install-mirrored Version.h. They flow to find_package consumers --
# who recompile the module units -- through the exported target's interface
# compile definitions, so Mila::getAPIVersion() works without any header to ship.
set(MILA_SEMVER "${PROJECT_VERSION}")

target_compile_definitions(Mila PUBLIC
    MILA_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
    MILA_VERSION_MINOR=${PROJECT_VERSION_MINOR}
    MILA_VERSION_PATCH=${PROJECT_VERSION_PATCH}
    MILA_VERSION_PRERELEASE_TAG="${MILA_PRERELEASE_TAG}"
)

# =============================================================================
# Additional Targets and Subdirectories
# =============================================================================

# Define the data directory path (project-wide resource). PROJECT_SOURCE_DIR (not
# CMAKE_SOURCE_DIR) keeps this pointing at Mila's own Data/ when embedded.
# Mila project benchmarks, samples, tests, and tools can reference this variable to access data files.
set(MILA_DATA_DIR "${PROJECT_SOURCE_DIR}/Data")

# Generate documentation with Doxygen (opt-in: Doxygen is a hard requirement of the Docs target)
if(MILA_ENABLE_DOCS)
    add_subdirectory( Docs )
endif()

# Developer tooling (tokenize.exe, etc.) -- top-level builds only; an embedding
# consumer just wants the library.
if(PROJECT_IS_TOP_LEVEL)
    add_subdirectory( Tools )
endif()

# The runtime's Python binding surface: a runtime-adjacent projection of Mila into
# Python (module `Mila.Bindings`), consumer-blind -- consumed by the MIS server and by
# the parity/converter tooling alike, so it is a peer of the runtime, not an adaptor.
if(MILA_ENABLE_PYTHON_BINDINGS)
    add_subdirectory( Bindings )
endif()

# First-class adaptors over the runtime. Chat is gated inside Adaptors/CMakeLists.txt
# (by MILA_ENABLE_ADAPTORS+CUDA); MIS's Python server is not a CMake target.
add_subdirectory( Adaptors )

if(MILA_ENABLE_PROFILING)
    add_subdirectory(Profiling)
endif()

if(MILA_ENABLE_TESTING)
    add_subdirectory(Tests)
endif()

if(MILA_ENABLE_SAMPLES)
     message(STATUS "MILA_ENABLE_SAMPLES is ON - adding Samples subdirectory")
     add_subdirectory(Samples)
else()
    message(STATUS "MILA_ENABLE_SAMPLES is OFF - skipping Samples")
endif()
