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
Update build configuration and enhance logging system

- Disabled `MILA_ENABLE_SAMPLES` in CMake and added version parsing from `Version.txt`.
- Commented out `Mila/Samples` and `Benches` in `enable_testing()`.
- Modified `add_library` for `Mila` to include additional source files and set C++ standards.
- Commented out model creation and inference steps in `Gpt2.cpp` for debugging.
- Changed template parameters in `Gpt2Model.ixx` from `TCompute` to `TPrecision`.
- Refactored `CpuAttentionOp.ixx` and `CpuCrossEntropyOp.ixx` for generic template structures.
- Enhanced `OperationRegistry` for flexible operation registration based on input/output types.
- Introduced `DefaultLogger` class for improved logging with timestamps and source location.
- Created `Version.h.in` and `Version.ixx` for better version management.
- Updated `Encoder` and `FullyConnected` classes to use template parameters for type safety.
- Refactored test files to support new template structures for comprehensive testing.
- Organized `Cuda` and `Cpu` operations into separate registration classes.
- Updated `Mila` module with new initialization functions and logging setup.
- Incremented version number in `Version.txt` to `0.9.65-alpha.1`.
- 
## Mila Build Instructions
Mila uses CMake build. To build Mila, follow the steps below:

1. Clone the Mila repository
2. Open the Mila directory using Visual Studio Code or any other IDE

## Required Components
* C++ 20 Module API
* NVIDIA Cuda Runtime, 12.8
* CMake 3.31