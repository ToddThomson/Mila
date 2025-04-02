# Mila
Mila Deep Neural Network Library

## Prerelease Notice
Mila, version 0.9.69-alpha is currently an early, preview release.

## Description
Achilles Mila Deep Neural Network library provides a comprehensive API to model, train and evaluate 
Deep Neural Networks for both research and production environments. The library implements
state-of-the-art architectures including transformers, convolutional networks, and recurrent models.
Mila utilizes the NVIDIA CUDA runtime for high-performance GPU acceleration, enabling efficient
training and inference on large-scale datasets. The library also supports distributed training
across multiple GPUs and compute nodes, with automatic optimization for various hardware configurations.

## Top Features
1. Deep Neural Nets
   * GPT2, Recurrent Neural Networks
   * GPU acceleration using CUDA runtime

2. Datasets
   * Batch sequence loader
   * Optimized data processing pipelines

## What's New
- Enhanced unit testing framework with comprehensive test coverage for neural network modules:
  - Added extensive tests for FullyConnected layers including bias/no-bias configurations
  - Implemented edge case testing for LayerNorm module with various input dimensions
  - Added numerical stability tests for all modules to ensure robust training
  - Implemented CPU/CUDA equivalence testing to guarantee consistent results across hardware
  
- Improved model architecture components:
  - Added support for training/inference mode switching for all layer types
  - Implemented parameter save/load functionality for model persistence
  - Enhanced type safety with template parameters across module implementations
  
- Optimized performance for GPU computing:
  - Added deterministic operation guarantees for CUDA implementations
  - Improved memory management in tensor operations
  - Enhanced numerical stability for large models with potentially extreme values
  
- Documentation and code quality improvements:
  - Updated Doxygen comments throughout codebase for better API documentation
  - Unified test structure across all modules for maintainability
  - Version updated to 0.9.68-alpha.1
  - 
## Mila Build Instructions
Mila uses CMake build. To build Mila, follow the steps below:  

1. Clone the Mila repository
- git clone https://github.com/toddthomson/mila.git cd mila

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
   
## Required Components
* C++23 support
* NVIDIA CUDA Runtime, 12.8
* CMake 3.31
* GTest framework for unit testing

## Documentation
Comprehensive documentation is available in the Docs folder, generated from Doxygen comments.

## License
Mila is licensed under the Apache License 2.0. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

## Contributing
We welcome contributions from the community. If you are interested in contributing to Mila, please follow these steps:

1. Fork the repository on GitHub.
2. Create a new branch from the `main` branch.
3. Make your changes and commit them with clear and concise messages.
4. Push your changes to your forked repository.
5. Create a pull request to the `main` branch of the original repository.

Please ensure that your code adheres to the project's coding standards and includes appropriate tests. For more detailed guidelines, refer to the [contribution guidelines](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.
