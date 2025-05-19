# Mila Benchmarks

This directory contains performance benchmarks for the Mila library. These benchmarks help measure and track the performance of various components and algorithms within Mila.

## Requirements

- CMake 3.14 or later
- A C++ compiler with C++23 support
- NVIDIA CUDA Toolkit 12.9 (for GPU benchmarks)
- Ninja build system (recommended)

## Overview

The Mila benchmarks provide performance measurements for:

- **Module Benchmarks**: Neural network module performance (GELU, Residual, MLP)
- **Operation Benchmarks**: Low-level operations performance (e.g., matrix multiplication, activation functions)
- **Precision Modes**: Different compute precision configurations (Auto, Performance, Accuracy, Disabled)
- **Device Support**: CPU and GPU (CUDA) benchmarks

## Contact

For questions or support, please open an issue or contact the Mila maintainers.
