# CMakeList.txt : CMake project for Mila source
#
cmake_minimum_required (VERSION 3.20)

find_package( CUDA 11.5 REQUIRED )

# Add source to this project's executable.
add_library( MilaLib STATIC

    "Dnn/Cuda/CudaUniqueHandle.h"

    "Dnn/CuDNN/CudnnContext.h"

    "Dnn/Cuda/CudaDevice.h"
    "Dnn/Cuda/CudaDeviceProps.h"
    "Dnn/Cuda/CudaDeviceProps.cpp"
    "Dnn/Cuda/CudaEnv.h"
    "Dnn/Cuda/CudaEnv.cpp"
    "Dnn/Cuda/CudaStream.cpp"
    "Dnn/Cuda/CudaStream.h"
    "Dnn/Cuda/CudaError.h"
    "Dnn/Cuda/CudaHelper.h"
    "Dnn/Cuda/CudaProfiler.h"

    "Dnn/CuDNN/init_data.cu"
    "Dnn/CuDNN/init_data.h"
    "Dnn/CuDNN/fp16_emu.cpp"
    "Dnn/CuDNN/fp16_emu.h"

    "Dnn/CuDNN/Descriptor.h"
    "Dnn/CuDNN/Status.h"
    "Dnn/CuDNN/OpaqueDescriptorType.h"
    "Dnn/CuDNN/OpaqueDescriptor.h"
    "Dnn/CuDNN/OpaqueHandle.h"
    "Dnn/CuDNN/Utils.h"
    "Dnn/CuDNN/Error.h"
    "Dnn/CuDNN/Version.h"

    "Dnn/DnnModel.h"
    "Dnn/DnnModelBuilder.h"
    "Dnn/DnnModelOptions.h"
    "Dnn/NeuralNetType.h" 
    "Dnn/RnnModelOptions.h"

    "Dnn/RNNOperation.h"
    "Dnn/RNNDataSet.h"
    "Dnn/Dropout.h"
    "Dnn/Tensor.h"
    "Dnn/StateTensor.h"
      
    "Dnn/RnnLinearLayer.h"
    "Dnn/RnnLayerCollection.h"
 "Dnn/Cuda/CudaMemory.h")

configure_file( MilaConfig.h.in MilaConfig.h)

# Cuda and CuDNN includes
target_include_directories(
    MilaLib
    PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} )


target_link_libraries( 
    MilaLib
    ${CUDNN_LIBRARY} ${CUDA_LIBRARIES}  )

target_include_directories(
    MilaLib INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR} )
