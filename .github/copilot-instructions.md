# Mila Project - Copilot Instructions

## Code Generation Policy
**Important**: Only generate code when explicitly requested. Focus on providing analysis, architectural guidance, problem identification, and suggestions unless the user specifically asks for code implementation.

## Current Task: Tensor Component Development

### Priority Focus
**Tensor System Redesign** - This is our highest priority development task. All other work should be deprioritized until the tensor foundation is solid.

### Tensor Component Goals
- **API Consistency**: Standardize tensor creation, manipulation, and device transfer APIs
- **Memory Management**: Optimize memory resource abstraction for CPU/CUDA devices
- **Type Safety**: Enhance compile-time type checking and device compatibility validation
- **Performance**: Minimize memory copies and optimize for GPU acceleration patterns
- **Testing**: Comprehensive test coverage for all tensor operations across devices

### Current Tensor Issues to Address
- [ ] Tensor construction and initialization patterns
- [ ] Device transfer and memory resource management
- [ ] Shape manipulation and broadcasting semantics
- [ ] Indexing and slicing operations
- [ ] Serialization and deserialization support
- [ ] Memory layout optimization for GPU operations

### Tensor Development Guidelines
- Focus on CPU and CUDA device compatibility
- Prioritize zero-copy operations where possible
- Ensure all tensor operations have equivalent CPU/CUDA implementations
- Test memory safety and resource cleanup thoroughly
- Document performance characteristics of each operation
- **Design Documentation**: See [Tensor Design Decisions](.github/tensor-design.md) for current architectural decisions and rationale

## General Programming Instructions

1. **Alpha Stage Development - No Backward Compatibility Required**
   - **CRITICAL**: The project is in alpha stage - backward compatibility is NOT required
   - Implement clean, forward-looking solutions without legacy support or migration paths
   - Breaking changes are expected and acceptable during alpha development
   - Focus on optimal architecture rather than compatibility layers
   - Remove deprecated patterns rather than maintaining dual systems
   - Avoid "deprecated" markers - simply implement the target architecture directly

2. **Code Generation Policy**
   - Generate code only when explicitly requested with phrases like "implement", "write code", "generate", or "create code"
   - When not generating code, provide analysis, suggestions, architectural guidance, and problem identification
   - Offer code snippets only when specifically asked or when demonstrating a concept that requires code examples

3. **Code Quality**
   - Avoid simple or obvious code comments - focus on explaining complex logic, algorithms, or non-obvious design decisions
   - Follow modern C++23 standards and best practices
   - Use RAII principles and smart pointers where appropriate
   - Prefer STL containers and algorithms over raw arrays and manual loops
   - Write exception-safe code with proper resource management

4. **Testing & Documentation**
   - All new features must include comprehensive unit tests using GTest
   - Test both CPU and CUDA implementations where applicable
   - Include edge cases, boundary conditions, and numerical stability tests
   - Update Doxygen comments for API changes
   - Follow existing test patterns in `Tests/Dnn/Modules` directory

5. **Performance Considerations**
   - Optimize for GPU acceleration using CUDA runtime
   - Consider memory layout and access patterns for performance
   - Use appropriate precision modes (Auto, Performance, Accuracy, Disabled)
   - Profile performance-critical code paths

## Project Overview

**Mila** is a modern Deep Neural Network library (version 0.9.8XX-alpha) designed for both research and production environments. The library provides:

### Core Features
- **Neural Network Architectures**: GPT2, RNNs, Transformers, CNNs, and recurrent models
- **GPU Acceleration**: NVIDIA CUDA runtime with CUDA Toolkit 13.0 support
- **Distributed Training**: Multi-GPU and multi-node support with automatic hardware optimization
- **Mixed Precision**: ComputePrecision policy framework for automatic mixed precision operations
- **Data Processing**: Optimized batch sequence loaders and data pipelines

### Technical Architecture
- **Build System**: CMake 3.31+ with Ninja generator
- **Language Standards**: 
	- C++23 for host code, CUDA 20 for device code
	- Uses modern C++ features (C++ 20 modules, templates, concepts)
- **Dependencies**: 
  - CUDA Toolkit 13.0
  - GTest (testing framework)
  - nlohmann/json (JSON handling)
  - miniz (compression)
- **Supported Platforms**:
	- Windows (Visual Studio 2022+),
	- WSL2 (Ubuntu 25.04+) with Visual Studio Code,
	- Linux (Docker support)

### Key Components
- **Compute Backend**: Abstracted device interface supporting CPU and CUDA devices with automatic device discovery, registration, and context management
- **Tensors**: Multi-dimensional arrays with GPU/CPU support and memory resource abstraction
- **Modules**: Neural network building blocks with fluent configuration interface
- **Operations**: Low-level optimized operations (GELU, matrix multiplication, etc.) with device-specific implementations
- **Models**: High-level model composition and training orchestration
- **Datasets**: Efficient data loading and preprocessing pipelines

## Current Development Focus

### Active Areas
1. **API Stabilization**: Working toward stable 1.0 beta release - current alpha API is still immature for end-user development

### Development Workflow
- **Main Branch**: Target for all PRs
- **Testing**: Required for all changes with CPU/CUDA equivalence testing
- **Code Review**: At least one maintainer approval required
- **Documentation**: Auto-generated and hosted at https://toddthomson.github.io/Mila
- **Build Badges**: CI pipeline status tracked via GitHub Actions

## Notes for AI Assistant
- **The project is in alpha stage** - the API is not stable and will undergo significant changes
- **Backward compatibility is NOT needed** - implement clean, modern solutions without legacy support
- **Breaking changes are acceptable and expected** - prioritize optimal design over compatibility
- **Avoid deprecated/legacy code patterns** - implement the target architecture directly
- **No migration paths required** - focus on the production-ready architecture
- When suggesting code changes, consider both CPU and CUDA implementations
- Always include appropriate testing suggestions
- Be mindful of performance implications in GPU-accelerated contexts
- Follow the established patterns in the existing codebase
- Please do not add simple comments that state the obvious; focus on complex logic and design decisions