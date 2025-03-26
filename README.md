# Mila
Mila Deep Neural Net library

## Prerelease Notice
Mila, version 0.9.63-alpha is currently an early, preview release.

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
- Refactor memory management and add Gpt2DataLoader
- Updated memory resource types across the codebase to unify CPU and GPU handling. Introduced `Gpt2DataLoader` with logging and multi-threaded data reading capabilities. Adjusted related files and tests to ensure compatibility with new resource types. Incremented version to 0.9.63-alpha.1.

## Mila Build Instructions
Mila uses CMake build. To build Mila, follow the steps below:

1. Clone the Mila repository
2. Open the Mila directory using Visual Studio Code or any other IDE

## Required Components
* C++ 20 Module API
* NVIDIA Cuda Runtime, 12.8
* CMake 3.31