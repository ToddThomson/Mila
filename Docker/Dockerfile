FROM nvidia/cuda:13.0.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
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
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install latest Ninja build system from source (required for c++20 modules support)
RUN git clone https://github.com/ninja-build/ninja.git /tmp/ninja \
    && cd /tmp/ninja \
    && git checkout release \
    && cmake -Bbuild -H. -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --target install \
    && rm -rf /tmp/ninja \
    && ninja --version

# Install Clang 19 and related tools (See issue #2 ).
RUN wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - \
    && echo "deb http://apt.llvm.org/noble/ llvm-toolchain-noble-19 main" > /etc/apt/sources.list.d/llvm.list \
    && apt-get update \
    && apt-get install -y clang-19 clang++-19 lldb-19 lld-19 clang-tools-19 clang-tidy-19 clang-format-19 \

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

# Install CMake 3.31 or newer (since your project requires 3.31)
RUN wget -qO- "https://cmake.org/files/v3.31/cmake-3.31.0-linux-x86_64.tar.gz" | tar --strip-components=1 -xz -C /usr/local

# Clone and build Google Test for testing support
RUN git clone https://github.com/google/googletest.git /tmp/googletest \
    && cd /tmp/googletest \
    && mkdir build \
    && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang-19 -DCMAKE_CXX_COMPILER=clang++-19 \
    && make install \
    && cd / \
    && rm -rf /tmp/googletest

# Set work directory
WORKDIR /mila

# Set environment variables
ENV CUDA_HOME=/usr/local/cuda-12.9
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV CC=clang-19
ENV CXX=clang++-19