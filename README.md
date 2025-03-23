# Mila
Mila Deep Neural Net library

## Prerelease Notice
Mila, version 0.9.61-alpha is currently an early, experimental, preview release.

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

Mila, Version 0.9.61-alpha.1 changes:
Refactor GPT-2 components and update device handling

- Updated CMakeLists.txt to include new source files for Gpt2App and removed CudaSoftmaxOp.
- Moved Gpt2DataLoader class to a new file and removed content from DataLoader.ixx.
- Changed module exports in Gpt2Model and Tokenizer to Gpt2App namespace.
- Created ModelConfig.ixx for model configuration parameters.
- Restructured TrainGpt2.cpp for command-line parsing and model initialization.
- Introduced OperationProperties.ixx for common operation properties.
- Updated README.md for version 0.9.61-alpha and recent changes.
- Replaced Compute::CpuDevice and Compute::CudaDevice with Compute::DeviceType for better device management.
- Updated tests in multiple files to ensure compatibility with the new DeviceType system.
* 

## Mila Build Instructions
Mila uses CMake build. To build Mila, follow the steps below:

1. Clone the Mila repository
2. Open the Mila directory using Visual Studio Code or any other IDE

## Required Components
* C++ 20 Module API
* NVIDIA Cuda Runtime, 12.8
* CMake 3.31