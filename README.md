# Mila
Mila Deep Neural Network Library

## Prerelease Notice
Mila, version 0.9.67-alpha is currently an early, preview release.

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
- Added Docs folder from Doxygen comments
- Updated Op registration mechanism
- Updated Doxygen comments for most classes
- Performance improvements for large model training

## Mila Build Instructions
Mila uses CMake build. To build Mila, follow the steps below:  

1. Clone the Mila repository
- git clone https://github.com/toddthomson/mila.git cd mila

2. Configure with CMake:
- mkdir build cd build cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ..

3. Build the project:
4. Run tests:
   
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

Please ensure that your code adheres to the project's coding standards and includes appropriate tests. For more detailed guidelines, refer to the [Contribution guidelines placeholder].