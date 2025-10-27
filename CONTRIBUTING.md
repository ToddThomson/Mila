# Contributing to Mila

Thank you for your interest in contributing to Mila! This document provides guidelines and instructions for contributing to the project.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Coding Standards](#coding-standards)
- [Pull Request Process](#pull-request-process)
- [Testing Requirements](#testing-requirements)
- [Documentation](#documentation)
- [Issue Reporting](#issue-reporting)

## Code of Conduct

We expect all contributors to be respectful and constructive. By participating in this project, you agree to maintain a welcoming, inclusive, and harassment-free environment for everyone.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Add the original repository as a remote named "upstream"
4. Create a new branch for your feature or bug fix
5. Make your changes and commit them
6. Push to your fork and submit a pull request

## Development Environment

### Required Components
* C++23 support
* NVIDIA CUDA Runtime, 12.8
* CMake 3.31
* GTest framework for unit testing

### Build Instructions

#### Using Visual Studio

1. **Prerequisites**
   - Visual Studio 2022 or newer with "Desktop development with C++" workload
   - CUDA Toolkit 12.8
   - CMake 3.31 or newer (included with Visual Studio)

2. **Open the Project**
   - Launch Visual Studio
   - Select "Open a local folder" and navigate to your cloned Mila repository
   - Visual Studio will automatically detect the CMakeLists.txt file

3. **Configure Project**
   - Visual Studio will automatically generate CMake cache
   - To customize build settings, right-click on CMakeLists.txt and select "CMake Settings for MilaProject"
   - Under "Configuration type", select "Release" for optimal performance

4. **Build the Project**
   - Right-click on CMakeLists.txt and select "Build All"
   - Alternatively, use the Build menu or press F7

5. **Run Tests**
   - In the Solution Explorer, expand the "Tests" folder
   - Right-click on a test project and select "Run Tests"

#### Using Visual Studio Code

1. **Prerequisites**
   - Visual Studio Code
   - C/C++ extension
   - CMake Tools extension
   - CUDA Toolkit 12.8
   - CMake 3.31 or newer

2. **Open the Project**
   - Launch VS Code
   - Open the folder containing your cloned Mila repository
   - VS Code should detect the CMake project automatically

3. **Configure Project**
   - Press Ctrl+Shift+P to open the command palette
   - Type "CMake: Configure" and select it
   - Choose your preferred generator (Ninja is recommended for faster builds)
   - Select the build variant (Debug/Release)

4. **Build the Project**
   - Press Ctrl+Shift+P to open the command palette
   - Type "CMake: Build" and select it, or use the build button in the status bar

5. **Run Tests**
   - Press Ctrl+Shift+P to open the command palette
   - Type "CMake: Run Tests" and select it
   - Alternatively, use the Test Explorer extension to browse and run tests

## Coding Standards

### Naming Conventions
* Use `PascalCase` for class, struct, and enum names
* Use `camelCase` for method and variable names
* Use `UPPER_CASE` for constants and macros
* Private member variables should be suffixed with underscore (e.g., `private_member_`)

### Code Formatting
* Use 4 spaces for indentation (no tabs)
* Line length should not exceed 100 characters
* Opening braces should be on the same line for functions and control statements
* Each file should end with a newline

### C++ Specific Guidelines
* Use modern C++23 features where appropriate
* Prefer `nullptr` to `NULL` or `0`
* Use smart pointers instead of raw pointers when possible
* Use `const` whenever applicable
* Avoid using exceptions in performance-critical code paths
* Template parameters should use `TFoo` naming convention (e.g., `TInput`, `TPrecision`)
* Use `#pragma once` at the beginning of header files

### Documentation
* Use Doxygen-style comments for classes, functions, and non-trivial code sections
* Document parameters, return values, exceptions, and any side effects
* Include examples where useful for complex functions or APIs

## Pull Request Process

1. Ensure your code follows the coding standards and passes all tests
2. Update the documentation, including the README.md if necessary
3. Include relevant tests for your changes:
   - Unit tests for new features
   - Regression tests for bug fixes
4. Title your PR with a concise description of the changes
5. Fill out the PR template completely
6. Your PR should target the `main` branch
7. PRs require review and approval from at least one maintainer

## Testing Requirements

### Unit Tests
* All new features must have corresponding unit tests
* Follow existing test patterns in the codebase (see `Tests/Dnn/Modules` directory)
* Include tests for both CPU and CUDA implementations where applicable
* Include tests for edge cases and error conditions
* Test numerical stability for floating-point operations

### Test Patterns
For modules, include tests for:
1. Basic functionality (forward pass, parameter counts)
2. Edge cases (minimal dimensions, boundary conditions)
3. Numerical stability (large and small input values)
4. CPU/CUDA equivalence where applicable
5. Training/inference mode behavior

## Documentation

* Update Doxygen comments in code for API changes
* For significant changes, update relevant documentation in the `Docs` folder
* Document any new dependencies or system requirements

## Issue Reporting

When reporting issues, please include:
1. Description of the issue
2. Steps to reproduce
3. Expected behavior
4. Actual behavior
5. Environment details (OS, compiler version, CUDA version)
6. Any relevant logs or screenshots

## License

By contributing to Mila, you agree that your contributions will be licensed under the project's [Apache License 2.0](LICENSE).

---

Thank you for contributing to Mila!
