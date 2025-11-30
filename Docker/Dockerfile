FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# Install core packages, GPG tools and lsb-release (used for distro codename)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ninja-build \
        curl \
        wget \
        git \
        gdb \
        doxygen \
        graphviz \
        software-properties-common \
        lsb-release \
        gnupg2 \
        ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Update package lists, install wget and gnupg2 (necessary dependencies for the script)
RUN apt-get update && apt-get install -y wget gnupg2 software-properties-common

# Download and run the official LLVM installation script for Clang 19
# The script handles adding the repository and public key
RUN wget https://apt.llvm.org/llvm.sh && chmod +x llvm.sh && ./llvm.sh 19

# (Optional) Clean up to reduce image size
RUN apt-get autoremove -y && apt-get clean && rm -rf /var/lib/apt/lists/* /llvm.sh

# Set default command to verify the installation
CMD ["clang-19", "--version"]

# Install latest Ninja build system from source (required for c++20 modules support)
RUN git clone https://github.com/ninja-build/ninja.git /tmp/ninja \
    && cd /tmp/ninja \
    && git checkout release \
    && cmake -Bbuild -H. -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --target install \
    && rm -rf /tmp/ninja \
    && ninja --version

# Set Clang 19 as the default compiler
RUN update-alternatives --install /usr/bin/clang clang /usr/bin/clang-19 100 \
    && update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-19 100 \
    && update-alternatives --install /usr/bin/cc cc /usr/bin/clang-19 100 \
    && update-alternatives --install /usr/bin/c++ c++ /usr/bin/clang++-19 100 \
    && update-alternatives --set clang /usr/bin/clang-19 \
    && update-alternatives --set clang++ /usr/bin/clang++-19 \
    && update-alternatives --set cc /usr/bin/clang-19 \
    && update-alternatives --set c++ /usr/bin/clang++-19

RUN clang --version && clang++ --version

# Install CMake 3.31 (project requires 3.31). Consider adding checksum validation if desired.
RUN wget -qO- "https://cmake.org/files/v3.31/cmake-3.31.0-linux-x86_64.tar.gz" \
    | tar --strip-components=1 -xz -C /usr/local

# Clone and install GoogleTest for testing support
RUN git clone https://github.com/google/googletest.git /tmp/googletest \
    && cd /tmp/googletest \
    && mkdir build \
    && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang-19 -DCMAKE_CXX_COMPILER=clang++-19 \
    && cmake --build . --target install \
    && rm -rf /tmp/googletest

# Set work directory
WORKDIR /mila

# Set environment variables
ENV CUDA_HOME=/usr/local/cuda-13.0
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV CC=clang-19
ENV CXX=clang++-19