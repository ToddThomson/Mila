# Mila development container.
# CUDA + Clang 19 + CMake 4.x + Ninja toolchain for building and testing the C++23
# library. Model weight conversion is an offline, host-side step (see
# Mila/Tools/Converters); converted weights are bind-mounted in via docker-compose.
FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# Core toolchain and tools. Ninja 1.11+ (Ubuntu 24.04) supports C++23 module dyndep and is
# the generator CI builds with. CMake is installed separately below (the distro package
# predates the 4.0 floor). Doxygen/Graphviz are only needed for -DMILA_ENABLE_DOCS=ON.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
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

# Clang 19 via the official LLVM script (adds the repository and key, installs the toolchain).
RUN wget -q https://apt.llvm.org/llvm.sh \
    && chmod +x llvm.sh \
    && ./llvm.sh 19 \
    && rm -f llvm.sh \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Make Clang 19 the default C/C++ compiler.
RUN update-alternatives --install /usr/bin/clang clang /usr/bin/clang-19 100 \
    && update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-19 100 \
    && update-alternatives --install /usr/bin/cc cc /usr/bin/clang-19 100 \
    && update-alternatives --install /usr/bin/c++ c++ /usr/bin/clang++-19 100 \
    && update-alternatives --set clang /usr/bin/clang-19 \
    && update-alternatives --set clang++ /usr/bin/clang++-19 \
    && update-alternatives --set cc /usr/bin/clang-19 \
    && update-alternatives --set c++ /usr/bin/clang++-19

# CMake 4.2.3 (the project requires 4.0+; the Ubuntu package is older). Matches CI.
RUN wget -qO- "https://cmake.org/files/v4.2/cmake-4.2.3-linux-x86_64.tar.gz" \
    | tar --strip-components=1 -xz -C /usr/local

WORKDIR /mila

ENV CUDA_HOME=/usr/local/cuda-13.0
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV CC=clang-19
ENV CXX=clang++-19

# Dependencies (GoogleTest, nlohmann_json, miniz, CUTLASS, ...) are fetched by the build
# via CPM/FetchContent at CMake configure time, which requires git and network access, so
# none are installed here.

CMD ["/bin/bash"]
