[![Mila CI Pipeline](https://github.com/ToddThomson/Mila/actions/workflows/build-pipeline.yml/badge.svg?branch=master)](https://github.com/ToddThomson/Mila/actions/workflows/build-pipeline.yml)
# Mila
Mila Deep Neural Network Library

## Prerelease Notice
Mila, version 0.9.857-alpha is currently an early preview release. 

We are currently working on the core Tensor component and the various compute devices which provide specific tensor operations. 
There is a work in prgress, but I wanted to share the library and its features to show where we are headed and to update the github repository 
so as to track our changes.

The Mila API is still too immature for any end-user development. I am working on the API and will be releasing a stable version as soon as possible.

## Description
Achilles Mila Deep Neural Network library provides a comprehensive API to model, train and evaluate 
Deep Neural Networks for both research and production environments. The library implements
state-of-the-art architectures including transformers, convolutional networks, and recurrent models.
Mila utilizes the NVIDIA CUDA runtime for high-performance GPU acceleration, enabling efficient
training and inference on large-scale datasets. The library also supports distributed training
across multiple GPUs and compute nodes, with automatic optimization for various hardware configurations.

## Documentation
Comprehensive Online documentation is available:

- **Online Documentation**: The complete API reference is hosted on GitHub Pages at [https://toddthomson.github.io/Mila](https://toddthomson.github.io/Mila)

The documentation includes class references, usage examples, and architecture guides. It is automatically updated through our GitHub Actions workflow whenever changes are pushed to the master branch.

## Top Features
1. Deep Neural Nets
   * GPT2, Recurrent Neural Networks
   * GPU acceleration using CUDA runtime

2. Datasets
   * Batch sequence loader
   * Optimized data processing pipelines

## What's New

* Modernized ConfigurationBase with deduced `this` parameters for a more elegant fluent interface
* Refactored Module class to use ConfigurationBase for all configuration options
* Simplified module creation with consistent validation and configuration approach
* Eliminated redundant configuration code with unified parameter handling
* Improved type safety throughout the framework with more comprehensive constraints

* Added Docker support (WIP) for Linux Ubuntu builds
  * Ready-to-use container with all dependencies pre-installed
  * Optimized environment for CI/CD pipelines and development
  * Multi-stage build process for smaller production images
  * Includes CUDA runtime for GPU acceleration on Linux systems
  * Simplified development setup across different Linux distributions

* Added ComputePrecision policy framework to support automatic mixed precision operations
* Enhanced GELU operations with precision-aware implementations for both CPU and CUDA
* Implemented mixed precision support in operation registry with type-specific operation creation


## Mila Build Instructions
Mila uses CMake build. To build Mila, follow the steps below:  

1. Clone the Mila repository
- git clone https://github.com/toddthomson/mila.git cd mila

#### Using Visual Studio

1. **Prerequisites**
   - Visual Studio 2022 or newer with "Desktop development with C++" workload
   - CUDA Toolkit 13.0 latest
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
   - CUDA Toolkit 13.0
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

#### Using Docker on Linux

1. **Prerequisites**
   - Docker installed on your system
   - NVIDIA Docker runtime (for GPU support)

2. **Pull the Docker Image**

3. **Run the Container**
- For CPU-only usage:
  ```bash
  docker run -it --rm toddthomson/mila:latest
  ```
- For GPU support:
  ```bash
  docker run -it --rm --gpus all toddthomson/mila:latest
  ```

4. **Build from Dockerfile**
- Clone the repository and build locally:
  ```bash
  git clone https://github.com/toddthomson/mila.git
  cd mila
  docker build -t mila:local .
  ```

5. **Development Workflow**
- Mount your local source directory for development:
  ```bash
  docker run -it --rm -v $(pwd):/mila/src toddthomson/mila:latest
     ```
   - Build inside the container:
     ```bash
     mkdir -p build && cd build
     cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release
     ninja
     ```
   
## Required Components
* C++23 support
* NVIDIA CUDA Runtime, 13.0 latest
* CMake 3.31 or later
* GTest framework for unit testing, 1.17.0

## License
Mila is licensed under the Apache License 2.0. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

## Contributing
We welcome contributions from the community. If you are interested in contributing to Mila, please follow these steps:

1. Fork the repository on GitHub.
2. Create a new branch from the `master` branch.
3. Make your changes and commit them with clear and concise messages.
4. Push your changes to your forked repository.
5. Create a pull request to the `master` branch of the original repository.

Please ensure that your code adheres to the project's coding standards and includes appropriate tests. For more detailed guidelines, refer to the [contribution guidelines](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.
