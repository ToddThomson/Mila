# Mila
Mila Deep Neural Net library

## Prerelease Notice
Mila, version 0.9.64-alpha is currently an early, preview release.

## Description
Achilles Mila Deep Neural Network library provides an API to model, train and evaluate
Deep Neural Networks. Mila utilizes the NVIDIA CUDA runtime for high-performance GPU acceleration.

## Top Features
1 Deep Neural Nets
  * GPT2, Recurrent Neural Networks
  * GPU acceleration using CUDA runtime

2 Datasets
  * Batch sequence loader
  *
 
## What's New
Update Mila library and enhance logging mechanisms

- Updated CMakeLists.txt to include new source files, libraries, and changed project version to 0.9.64-alpha.1.
- Replaced Logger with StepLogger in Gpt2.cpp for improved logging.
- Enhanced error handling in CudaPinnedMemoryResource.ixx.
- Added OperationAttributes.ixx for common DNN operation properties.
- Removed Gpt2DataLoader.ixx as part of a refactor.
- Introduced DatasetReader.ixx for efficient data loading with multi-threading.
- Added DefaultLogger.ixx and TrainingLogger.ixx for flexible logging strategies.
- Created DatasetReader.cpp with unit tests to ensure functionality.



## Mila Build Instructions
Mila uses CMake build. To build Mila, follow the steps below:

1. Clone the Mila repository
2. Open the Mila directory using Visual Studio Code or any other IDE

## Required Components
* C++ 20 Module API
* NVIDIA Cuda Runtime, 12.8
* CMake 3.31