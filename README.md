# Mila
Mila Deep Neural Net library

## Prerelease Notice
Mila, version 0.9.62-alpha is currently an early, experimental, preview release.

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
Mila, Version 0.9.62-alpha.1 changes:
- Added new CUDA kernel files for layer normalization, residual operations, and softmax operations.
- Introduced `OperationAttributes` to replace `OperationProperties` for better attribute management.
- Refactored `Gpt2.cpp` to streamline device management and model initialization.
- Enhanced `Gpt2Model.ixx` with improved tensor data handling methods.
- Updated `Tensor` class with new methods for memory resource accessibility and data copying.
- Revised `README.md` to reflect version 0.9.62-alpha and document recent changes.
- Improved performance in CUDA kernels using cooperative groups.
- Added and modified tests to ensure compatibility with new features.


## Mila Build Instructions
Mila uses CMake build. To build Mila, follow the steps below:

1. Clone the Mila repository
2. Open the Mila directory using Visual Studio Code or any other IDE

## Required Components
* C++ 20 Module API
* NVIDIA Cuda Runtime, 12.8
* CMake 3.31