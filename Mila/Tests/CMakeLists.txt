# CMake project for testing the Mila Library
#
# enable_testing() is NOT called here: the top-level CMakeLists already enables CTest
# (gated on PROJECT_IS_TOP_LEVEL), and Mila/CMakeLists.txt enables it for the embedded
# case before adding this directory. A third call here would be pure redundancy.
#
# Test source organization (see Specifications/Testing.md):
#   Section 1 - revised to the current test methodology.
#   Section 2 - active, not yet revisited (pre-methodology).
#   Section 3 - commented, awaiting revival (retired in place, not deleted).
# Within each section, entries are sorted by their Src module path. CPU-agnostic
# units live in the add_executable() list below; CUDA-only units (*.Cuda.cpp /
# Cuda*.cpp) live in the MILA_ENABLE_CUDA block at the bottom, organized the same
# way, so the MILA_ENABLE_CUDA=OFF build never compiles a CUDA translation unit.

include(FetchContent)
   FetchContent_Declare(
   googletest
     URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set( gtest_force_shared_crt ON CACHE BOOL "" FORCE )

FetchContent_MakeAvailable( googletest )
include( GoogleTest )

# Set global C++ and CUDA standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(MILA_ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)

    # Default to building for the GPU on the current system
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
endif()

add_executable( MilaTests
    "main.cpp"

    # ==============================================================
    # Section 1 - Current test methodology (Specifications/Testing.md)
    # ==============================================================
    "Data/Loaders/DataLoader.cpp"
    "Dnn/Components/Activations/Activation/Activation.Cpu.cpp"
    "Dnn/Components/Activations/Activation/ActivationConfig.cpp"
    "Dnn/Components/Activations/Gelu/Gelu.Cpu.cpp"
    "Dnn/Components/Activations/Gelu/GeluConfig.cpp"
    "Dnn/Components/Attention/GQA/GqaConfig.cpp"
    "Dnn/Components/Attention/MHA/MultiHeadAttention.Cpu.cpp"
    "Dnn/Components/Attention/MHA/MultiHeadAttentionConfig.cpp"
    "Dnn/Components/Connections/Residual.Cpu.cpp"
    "Dnn/Components/Connections/ResidualConfig.cpp"
    "Dnn/Components/Embeddings/TokenEmbeddingConfig.cpp"
    "Dnn/Components/Encodings/Lpe/Lpe.Cpu.cpp"
    "Dnn/Components/Encodings/Lpe/LpeConfig.cpp"
    "Dnn/Components/Encodings/Rope/RopeConfig.cpp"
    "Dnn/Components/FFN/GatedMLP/GatedMLPConfig.cpp"
    "Dnn/Components/FFN/MLP/MLP.Cpu.cpp"
    "Dnn/Components/FFN/MLP/MLPConfig.cpp"
    "Dnn/Components/FFN/Swiglu/SwigluConfig.cpp"
    "Dnn/Components/Linear/Linear.Cpu.cpp"
    "Dnn/Components/Linear/LinearConfig.cpp"
    "Dnn/Components/Normalization/LayerNorm/LayerNorm.Cpu.cpp"
    "Dnn/Components/Normalization/LayerNorm/LayerNormConfig.cpp"
    "Dnn/Components/Normalization/RmsNorm/RmsNormConfig.cpp"
    "Dnn/Components/Normalization/Softmax.Cpu.cpp"
    "Dnn/Components/Normalization/SoftmaxConfig.cpp"
    "Dnn/Components/Transformers/Gpt/Gpt.Presets.cpp"
    "Dnn/Components/Transformers/Gpt/GptBlock.Config.cpp"
    "Dnn/Components/Transformers/Gpt/GptBlock.Cpu.cpp"
    "Dnn/Components/Transformers/Gpt/GptTransformer.Cpu.cpp"
    "Dnn/Components/Transformers/Llama/Llama.Presets.cpp"
    "Dnn/Components/Transformers/Gemma/Gemma.Config.cpp"
    "Dnn/Components/Transformers/Gemma/Gemma.Protocol.cpp"
    "Dnn/Compute/Operations/OperationTraits.cpp"
    "Dnn/Core/Component.cpp"
    "Dnn/Core/ComponentConfig.cpp"
    "Dnn/Core/CompositeComponent.cpp"
    # The only GptModel factory reachable without a converted .bin: a checkpoint
    # carries its own config, so fromCheckpoint needs nothing but an archive.
    "Dnn/Models/GptModel.Cpu.cpp"
    "Dnn/Modeling/Network.Cpu.cpp"
    "Dnn/Optimizers/AdamW.Cpu.cpp"
    # Covers both containers PretrainedModelReader accepts, including a hand-built
    # MILA file that guards the legacy .bin path no fixture-free test could reach.
    "Dnn/Serialization/SafeTensors.Cpu.cpp"
    "Dnn/Tensors/Tensor.Constructors.cpp"
    "Dnn/Tensors/Tensor.DataAccess.cpp"
    "Dnn/Tensors/Tensor.DataPointers.cpp"
    "Dnn/Tensors/Tensor.Identity.cpp"
    "Dnn/Tensors/Tensor.Io.cpp"
    "Dnn/Tensors/Tensor.MemoryProperties.cpp"
    "Dnn/Tensors/Tensor.Properties.cpp"
    "Dnn/Tensors/Tensor.ShapeTransform.cpp"
    "Dnn/Tensors/TensorOps/Fill.Cpu.cpp"
    "Dnn/Tensors/TensorOps/Math.Cpu.cpp"
    "Dnn/Tensors/TensorOps/Random.Cpu.cpp"
    "Dnn/Tensors/TensorOps/Zero.Cpu.cpp"
    "Dnn/Quantization/CodebookOracle.Cpu.cpp"
    "Dnn/Quantization/CodebookPacking.Cpu.cpp"
    "Dnn/Quantization/PrecisionPlan.Cpu.cpp"

    # ==============================================================
    # Section 2 - Active, not yet revisited (pre-methodology)
    # ==============================================================
    "Data/Loaders/TokenSequenceLoader.cpp"
    "Data/Tokenizers/Bpe/BpeTokenizer.cpp"
    "Data/Tokenizers/Bpe/BpeTokenizer.Gpt2.cpp"
    "Data/Tokenizers/Bpe/BpeTokenizer.Llama32.cpp"
    "Data/Tokenizers/Bpe/BpeTokenizer.Gemma.cpp"
    "Data/Tokenizers/Bpe/BpeTrainer.cpp"
    "Data/Tokenizers/Bpe/BpeVocabulary.cpp"
    "Data/Tokenizers/Char/CharTokenizer.cpp"
    "Data/Tokenizers/Char/CharTrainer.cpp"
    "Data/Tokenizers/Char/CharVocabulary.cpp"
    "Dnn/Components/Transformers/Llama/Llama.Config.cpp"
    "Dnn/Compute/DeviceId.cpp"
    "Dnn/Compute/DeviceTypes.cpp"
    "Dnn/Compute/Devices/Cpu/CpuDevice.cpp"
    "Dnn/Compute/Devices/Registry/DeviceRegistrar.cpp"
    "Dnn/Compute/Devices/Registry/DeviceRegistry.cpp"
    "Dnn/Compute/Devices/Registry/DeviceRegistryHelpers.cpp"
    "Dnn/Compute/ExecutionContext/CpuExecutionContext.cpp"
    "Dnn/Compute/ExecutionContext/ExecutionContext.cpp"
    "Dnn/Compute/Operations/OperationBase.cpp"
    "Dnn/Tensors/TensorBuffer.cpp"
    "Dnn/Tensors/TensorOps/TensorOps.Transfer.cpp"

    # ==============================================================
    # Section 3 - Disabled, by disposition (see Specifications/Testing.md s1)
    # ==============================================================
    # -- DELETE (redundant op-layer mirror): backend ops are implementation detail
    #    (not in import Mila;) and their numerics are reachable through the component,
    #    so these op tests duplicate the component suite. The only legitimate op-layer
    #    coverage is the quantization white-box, which is net-new (the "quantization
    #    backfill" BACKLOG item), not a revival of these. Files kept on disk pending
    #    an explicit delete.
    #
    # -- REVIVE later (MNIST/training spine): thin Core/Network delta beyond the
    #    container coverage in CompositeComponent.cpp.
    #"Dnn/Core/Network.cpp"
    #
    # -- PARK (loss-on-device milestone): both samples compute loss host-side.
    #"Dnn/Components/Losses/CrossEntropy.Config.cpp"
    #"Dnn/Components/Losses/CrossEntropy.Cpu.cpp"
    #
    # -- PARK (serialization WIP; Model-abstraction target source not current):
    #"Dnn/Model.cpp"
    #"Dnn/Modeling/Model.Cpu.cpp"
    #"Dnn/Serialization/ModelArchive.cpp"
    #"Dnn/Serialization/ZipSerializer.cpp"
    #"Dnn/Tensors/TensorBuffer.Tracking.cpp"
)

# Everything that does not move bytes, which is nearly all of distribution: the store, pull,
# packaging, and HuggingFace's own parsing. Each takes its transport as an injected callable --
# the store a fetcher, the resolver an IModelHub, the hub an IHttpTransport -- so these run
# offline, and they compile whether or not this build has an HTTP client.
target_sources( MilaTests PRIVATE
    "Distribution/ModelStore.Cpu.cpp"
    "Distribution/ModelPackage.Cpu.cpp"
    "Distribution/ModelResolver.Cpu.cpp"
    "Distribution/ModelHub.Cpu.cpp"
    "Distribution/HttpClient.Cpu.cpp"
)

# Nothing under Distribution is gated any more. HttpClient's policy -- redirect hops, the
# cross-host token drop, Range and the 200-versus-206 rule -- runs against a scripted
# transport, so the rules that matter are pinned in every build rather than only where
# libcurl happens to be compiled.

# CUDA-only test translation units. These exercise CUDA memory resources,
# execution contexts, and kernels directly and have no CPU-only meaning, so they
# are compiled only when the CUDA backend is enabled. New *.Cuda.cpp / Cuda*.cpp
# test files belong in this block, not in the add_executable list above. The same
# three-section organization applies here.
if(MILA_ENABLE_CUDA)
    target_sources( MilaTests PRIVATE
        # ----------------------------------------------------------
        # Section 1 - Current test methodology
        # ----------------------------------------------------------
        "Data/Loaders/DataLoader.Cuda.cpp"
        "Dnn/Components/Activations/Activation/Activation.Cuda.cpp"
        "Dnn/Components/Activations/Gelu/Gelu.Cuda.cpp"
        "Dnn/Components/Attention/GQA/GroupedQueryAttention.Cuda.cpp"
        "Dnn/Components/Attention/GQA/CudaGqaOp.Cuda.cpp"
        "Dnn/Components/Attention/MHA/MultiHeadAttention.Cuda.cpp"
        "Dnn/Components/Connections/Residual.Cuda.cpp"
        "Dnn/Components/Embeddings/TokenEmbedding.Cuda.cpp"
        "Dnn/Components/Encodings/Lpe/Lpe.Cuda.cpp"
        "Dnn/Components/Encodings/Rope/Rope.Cuda.cpp"
        "Dnn/Components/FFN/GatedMLP/GatedMLP.Cuda.cpp"
        "Dnn/Components/FFN/MLP/MLP.Cuda.cpp"
        "Dnn/Components/FFN/Swiglu/Swiglu.Cuda.cpp"
        "Dnn/Components/Linear/Linear.Cuda.cpp"
        "Dnn/Components/Normalization/LayerNorm/LayerNorm.Cuda.cpp"
        "Dnn/Components/Normalization/RmsNorm/RmsNorm.Cuda.cpp"
        "Dnn/Components/Normalization/Softmax.Cuda.cpp"
        "Dnn/Components/Transformers/Gpt/GptBlock.Cuda.cpp"
        "Dnn/Components/Transformers/Gpt/GptTransformer.Cuda.cpp"
        "Dnn/Components/Transformers/Llama/Llama.Cuda.cpp"
        "Dnn/Components/Transformers/Gemma/Gemma.Block.Cuda.cpp"
        "Dnn/Components/Transformers/Gemma/Gemma.Cuda.cpp"
        "Dnn/Compute/Operations/OperationTraits.Cuda.cpp"
        "Dnn/Components/Linear/CodebookGemv.Cuda.cpp"
        "Dnn/Components/Linear/CudaLinearOp.Codebook.Cuda.cpp"
        # The saveParameterToArchive staging branch needs a device-only dtype (BF16),
        # so it is unreachable from the CPU companion in Dnn/Core/Component.cpp.
        "Dnn/Core/Component.Cuda.cpp"
        "Dnn/Models/GemmaModel.Parity.Cuda.cpp"
        # Gate B: the reported footprint against cudaMemGetInfo. Both skip without a real
        # checkpoint, so neither runs in CI -- see Specifications/MemoryFootprint.md 7.
        "Dnn/Models/GemmaModel.Footprint.Cuda.cpp"
        "Dnn/Models/LlamaModel.Footprint.Cuda.cpp"
        # The generation context bound. CUDA rather than CPU because the CPU operation
        # layer takes its extents from build(), so a CPU model cannot run a prompt
        # shorter than its built context and never reaches the loop under test.
        "Dnn/Models/GptModel.Cuda.cpp"
        # Net-new: the FP32 master-parameter path has no coverage in AdamW.Cuda.cpp.
        "Dnn/Optimizers/AdamW.MixedPrecision.Cuda.cpp"
        "Dnn/Samplers/Sampling.Cuda.cpp"
        "Dnn/Tensors/Tensor.Constructors.Cuda.cpp"
        "Dnn/Tensors/Tensor.DataAccess.Cuda.cpp"
        "Dnn/Tensors/Tensor.MemoryProperties.Cuda.cpp"
        "Dnn/Tensors/Tensor.ShapeTransform.Cuda.cpp"
        "Dnn/Tensors/TensorOps/Fill.Cuda.cpp"
        "Dnn/Tensors/TensorOps/Math.Cuda.cpp"
        "Dnn/Tensors/TensorOps/Random.Cuda.cpp"
        # split() is CUDA-only (no CpuTensorOps::split), so there is no .Cpu.cpp peer.
        "Dnn/Tensors/TensorOps/Structural.Cuda.cpp"
        "Dnn/Tensors/TensorOps/Zero.Cuda.cpp"

        # ----------------------------------------------------------
        # Section 2 - Active, not yet revisited (pre-methodology)
        # ----------------------------------------------------------
        "Dnn/Compute/Devices/Cuda/CudaDeviceRegistrar.cpp"
        "Dnn/Compute/ExecutionContext/CudaExecutionContext.cpp"

        # ----------------------------------------------------------
        # Section 3 - Disabled, by disposition (see Specifications/Testing.md s1)
        # ----------------------------------------------------------
        # -- DELETE (redundant op-layer mirror; see the CPU Section 3 note): these
        #    re-check GEMM / attention numerics reachable through Linear.Cuda /
        #    MultiHeadAttention.Cuda. They also carry the stale pre-move path
        #    "Dnn/Compute/Operations/Cuda/..." (files now live at
        #    "Dnn/Compute/Devices/Cuda/Operations/..."). The one op test that SHOULD
        #    exist is a newly-written CudaLinearOp quantization white-box (FP8/FP4
        #    scales + nibble packing), tracked as the quantization backfill BACKLOG
        #    item -- not a revival of these.
        #
        # -- REVIVE later (MNIST/training spine GPU companions):
        #"Dnn/Modeling/Network.Cuda.cpp"
        #"Dnn/Optimizers/AdamW.Cuda.cpp"
        #
        # -- PARK (loss-on-device milestone):
        #"Dnn/Components/Losses/SoftmaxCrossEntropy.Cuda.cpp"
        #
        # -- PARK (infra value-type test; group with the Section 2 infra tests):
        #"Dnn/Compute/Devices/Cuda/CudaDevice.cpp"
    )
endif()

target_compile_definitions( MilaTests PRIVATE TEST_DATA_DIR="${MILA_DATA_DIR}" )

# Shared, non-module test helpers (e.g. Common/GradientCheck.h) are included from
# the Tests root so any test file can reference them as "Common/<helper>.h".
target_include_directories( MilaTests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" )

target_compile_features( MilaTests PUBLIC cxx_std_23 )

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  # Test translation units cannot use <fstream>: MSVC C++23 raises C2079 on
  # basic_istream::sentry when stream I/O meets `import Mila;` in a .cpp, so the four test
  # files doing file I/O are on std::fopen by necessity rather than by choice. The library
  # itself needs no such exemption -- its module units use <fstream>. Revisit when the
  # C2079 defect is fixed; the test files can then follow.
  target_compile_definitions( MilaTests PRIVATE _CRT_SECURE_NO_WARNINGS )
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(MilaTests PRIVATE
    -fno-implicit-modules
    -fno-implicit-module-maps
    -fmodules
  )
endif()

target_link_libraries( MilaTests PRIVATE Mila gtest )

if(MILA_ENABLE_CUDA)
    target_link_libraries( MilaTests PRIVATE CUDA::cudart )
endif()

# Visual Studio native C++ code coverage instruments only binaries linked with
# /PROFILE. Mila is a STATIC library linked into MilaTests, so instrumenting the
# test exe covers the Mila code the tests exercise -- the library target needs no
# change. Gated (default OFF) because /PROFILE forces /INCREMENTAL:NO, which slows
# the relink in the normal TDD loop. Turn ON in a symbol-bearing config
# (x64-debug or x64-profile), build, then Test > Analyze Code Coverage.
option(MILA_ENABLE_COVERAGE "Link MilaTests with /PROFILE for Visual Studio native code coverage." OFF)

if(MILA_ENABLE_COVERAGE AND MSVC)
    target_link_options( MilaTests PRIVATE /PROFILE )
endif()

gtest_discover_tests(MilaTests
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"  # Set working directory to project root
    PROPERTIES
        VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

# ---------------------------------------------------------------------------
# Chat adaptor: the rich-text formatter.
#
# Its own executable rather than a file in MilaTests, because Chat.RichText imports
# nothing -- no Mila, no CUDA -- and linking it against the library would give the
# cheapest test surface in the tree the slowest build in the tree. It runs anywhere,
# including a CPU-only configure.
#
# The target lives here rather than under Adaptors/Chat because add_subdirectory(Tests)
# runs after Adaptors, so the GoogleTest targets do not exist yet at that point.
# ---------------------------------------------------------------------------
add_executable( ChatRichTextTests
    "Adaptors/Chat/RichText.Cpu.cpp"
)

# BASE_DIRS is explicit because a module file set otherwise roots itself at this
# directory, and the module being tested lives in the adaptor it belongs to.
target_sources( ChatRichTextTests
PUBLIC
    FILE_SET module_files
    TYPE CXX_MODULES
    BASE_DIRS "${CMAKE_SOURCE_DIR}/Mila/Adaptors/Chat/Src"
    FILES
        "${CMAKE_SOURCE_DIR}/Mila/Adaptors/Chat/Src/Chat.RichText.ixx"
)

set_target_properties( ChatRichTextTests PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

target_link_libraries( ChatRichTextTests PRIVATE gtest gtest_main )

gtest_discover_tests( ChatRichTextTests
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    PROPERTIES
        VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

# Packaging gates: the FetchContent subproject consumer against the local tree, and the
# opt-in CPM-by-tag consumer against a published release. Defined here, with the tests,
# rather than at the root: it is meaningful only when MILA_ENABLE_TESTING is on, and that
# dependency is now structural -- this file is reached only through the
# MILA_ENABLE_TESTING-gated add_subdirectory(Tests). Default OFF because each gate is a
# full subproject reconfigure + rebuild (up to 1800s); unit-test runs should not pay that.
# The x64-validate preset (and CI) turns it on.
option(MILA_ENABLE_PACKAGING_TEST "Build the packaging gates (embed Mila via FetchContent and build a consumer against it)." OFF)

if(MILA_ENABLE_PACKAGING_TEST)
    add_subdirectory(Packaging)
endif()
