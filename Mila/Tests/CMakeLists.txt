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
    "Dnn/Components/Activations/Gelu/Gelu.Cpu.cpp"
    "Dnn/Components/Activations/Gelu/GeluConfig.cpp"
    "Dnn/Core/Component.cpp"
    "Dnn/Core/ComponentConfig.cpp"

    # ==============================================================
    # Section 2 - Active, not yet revisited (pre-methodology)
    # ==============================================================
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
    "Dnn/Tensors/Tensor.Constructors.cpp"
    "Dnn/Tensors/Tensor.DataAccess.cpp"
    "Dnn/Tensors/Tensor.DataPointers.cpp"
    "Dnn/Tensors/Tensor.MemoryProperties.cpp"
    "Dnn/Tensors/TensorBuffer.cpp"
    "Dnn/Tensors/TensorOps/Fill.Cpu.cpp"
    "Dnn/Tensors/TensorOps/Math.Cpu.cpp"
    "Dnn/Tensors/TensorOps/TensorOps.Transfer.cpp"

    # ==============================================================
    # Section 3 - Commented, awaiting revival (retired in place)
    # ==============================================================
    #"Data/Tokenizers/Bpe/BpeTokenizer.cpp"
    #"Data/Tokenizers/Bpe/BpeTokenizer.Gpt2.cpp"
    #"Data/Tokenizers/Bpe/BpeTokenizer.Llama32.cpp"
    #"Data/Tokenizers/Bpe/BpeTrainer.cpp"
    #"Data/Tokenizers/Bpe/BpeVocabulary.cpp"
    #"Data/Tokenizers/Char/CharTokenizer.cpp"
    #"Data/Tokenizers/Char/CharTrainer.cpp"
    #"Data/Tokenizers/Char/CharVocabulary.cpp"
    #"Dnn/Components/Attention/GQA/GQA.Config.cpp"
    #"Dnn/Components/Attention/MHA/MHA.Config.cpp"
    #"Dnn/Components/Attention/MHA/MHA.Cpu.cpp"
    #"Dnn/Components/Composites/GptBlock.Config.cpp"
    #"Dnn/Components/Composites/GptBlock.Cpu.cpp"
    #"Dnn/Components/Composites/MLP.Cpu.cpp"
    #"Dnn/Components/Composites/MLPConfig.cpp"
    #"Dnn/Components/Connections/Residual.Config.cpp"
    #"Dnn/Components/Connections/Residual.Cpu.cpp"
    #"Dnn/Components/Encodings/Lpe/Encoder.Cpu.cpp"
    #"Dnn/Components/Encodings/Lpe/EncoderConfig.cpp"
    #"Dnn/Components/Linear/Linear.Cpu.cpp"
    #"Dnn/Components/Linear/LinearConfig.cpp"
    #"Dnn/Components/Losses/CrossEntropy.Config.cpp"
    #"Dnn/Components/Losses/CrossEntropy.Cpu.cpp"
    #"Dnn/Components/Normalization/LayerNorm.Cpu.cpp"
    #"Dnn/Components/Normalization/LayerNormConfig.cpp"
    #"Dnn/Components/Normalization/Softmax.Cpu.cpp"
    #"Dnn/Components/Normalization/SoftmaxConfig.cpp"
    #"Dnn/Components/Transformers/Gpt/Gpt.Presets.cpp"
    #"Dnn/Components/Transformers/Gpt/GptTransformer.Cpu.cpp"
    #"Dnn/Components/Transformers/Llama/Llama.Presets.cpp"
    #"Dnn/Compute/Devices/Cpu/Operations/CpuEncoderOpTests.cpp"
    #"Dnn/Compute/Devices/Cpu/Operations/CpuGeluOpTests.cpp"
    #"Dnn/Compute/Devices/Cpu/Operations/CpuLayerNormOpTests.cpp"
    #"Dnn/Compute/Devices/Cpu/Operations/CpuLinearOpTests.cpp"
    #"Dnn/Compute/Devices/Cpu/Operations/CpuResidualOpTests.cpp"
    #"Dnn/Compute/Devices/Cpu/Operations/CpuSoftmaxOpTests.cpp"
    #"Dnn/Compute/Operations/BinaryOperation.cpp"
    #"Dnn/Compute/Operations/UnaryOperation.cpp"
    #"Dnn/Core/CompositeComponent.cpp"
    #"Dnn/Core/Network.cpp"
    #"Dnn/Data/DataLoader.cpp"
    #"Dnn/Data/TokenSequenceLoader.cpp"
    #"Dnn/Model.cpp"
    #"Dnn/Modeling/Model.Cpu.cpp"
    #"Dnn/Modeling/Network.Cpu.cpp"
    #"Dnn/Optimizers/AdamW.Cpu.cpp"
    #"Dnn/Serialization/ModelArchive.cpp"
    #"Dnn/Serialization/ZipSerializer.cpp"
    #"Dnn/Tensors/Tensor.Identity.cpp"
    #"Dnn/Tensors/Tensor.Initializers.cpp"
    #"Dnn/Tensors/Tensor.Io.cpp"
    #"Dnn/Tensors/Tensor.Properties.cpp"
    #"Dnn/Tensors/TensorBuffer.Tracking.cpp"
)

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
        "Dnn/Components/Activations/Gelu/Gelu.Cuda.cpp"

        # ----------------------------------------------------------
        # Section 2 - Active, not yet revisited (pre-methodology)
        # ----------------------------------------------------------
        "Dnn/Compute/Devices/Cuda/CudaDeviceRegistrar.cpp"
        "Dnn/Compute/ExecutionContext/CudaExecutionContext.cpp"
        "Dnn/Tensors/TensorOps/Fill.Cuda.cpp"
        "Dnn/Tensors/TensorOps/Math.Cuda.cpp"

        # ----------------------------------------------------------
        # Section 3 - Commented, awaiting revival (retired in place)
        # ----------------------------------------------------------
        #"Dnn/Components/Activations/Gelu/Gelu.Cuda.cpp"
        #"Dnn/Components/Attention/GQA/GQA.Cuda.cpp"
        #"Dnn/Components/Attention/MHA/MHA.Cuda.cpp"
        #"Dnn/Components/Composites/GptBlock.Cuda.cpp"
        #"Dnn/Components/Composites/MLP.Cuda.cpp"
        #"Dnn/Components/Connections/Residual.Cuda.cpp"
        #"Dnn/Components/Encodings/Lpe/Encoder.Cuda.cpp"
        #"Dnn/Components/Linear/Linear.Cuda.cpp"
        #"Dnn/Components/Losses/SoftmaxCrossEntropy.Cuda.cpp"
        #"Dnn/Components/Normalization/LayerNorm.Cuda.cpp"
        #"Dnn/Components/Normalization/RmsNorm/RmsNorm.Cuda.cpp"
        #"Dnn/Components/Normalization/Softmax.Cuda.cpp"
        #"Dnn/Components/Transformers/Gpt/GptTransformer.Cuda.cpp"
        #"Dnn/Components/Transformers/Llama/Llama.Cuda.cpp"
        #"Dnn/Compute/Devices/Cuda/CudaDevice.cpp"
        #"Dnn/Modeling/Network.Cuda.cpp"
        #"Dnn/Optimizers/AdamW.Cuda.cpp"
        # NOTE: the commented CUDA operation tests below carry their original
        # (pre-move) path "Dnn/Compute/Operations/Cuda/...". The files now live at
        # "Dnn/Compute/Devices/Cuda/Operations/..." -- correct the path on revival.
        #"Dnn/Compute/Operations/Cuda/CudaEncoderOpTests.cpp"
        #"Dnn/Compute/Operations/Cuda/CudaGeluOpTests.cpp"
        #"Dnn/Compute/Operations/Cuda/CudaLayerNormOpTests.cpp"
        #"Dnn/Compute/Operations/Cuda/CudaMatMulBiasOpTests.cpp"
        #"Dnn/Compute/Operations/Cuda/CudaMultHeadAttentionOpTests.cpp"
        #"Dnn/Compute/Operations/Cuda/CudaResidualOpTests.cpp"
        #"Dnn/Compute/Operations/Cuda/CudaSoftmaxOpTests.cpp"
    )
endif()

target_compile_definitions( MilaTests PRIVATE TEST_DATA_DIR="${MILA_DATA_DIR}" )

target_compile_features( MilaTests PUBLIC cxx_std_23 )

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

# Packaging gates: the find_package(Mila) consumer (installs Mila, builds
# Samples/QuickStart against the install) and the FetchContent subproject consumer.
# Defined here, with the tests, rather than at the root: it is meaningful only when
# MILA_ENABLE_TESTING is on, and that dependency is now structural -- this file is
# reached only through the MILA_ENABLE_TESTING-gated add_subdirectory(Tests). Default
# OFF because each gate is a full install/subproject reconfigure + rebuild (900-1800s);
# unit-test runs should not pay that. The x64-validate preset (and CI) turns it on.
option(MILA_ENABLE_PACKAGING_TEST "Build the find_package + FetchContent packaging gates (install/embed Mila and build a consumer against it)." OFF)

if(MILA_ENABLE_PACKAGING_TEST)
    add_subdirectory(Packaging)
endif()
