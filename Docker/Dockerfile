# Mila build + run container.
#
# A single CUDA `-devel` image that both BUILDS the C++23 module library and RUNS
# the Chat app: the devel image already carries the CUDA runtime libraries Chat
# links (cudart / cuBLAS / cuBLASLt / cuRAND / nvtx), so no separate runtime image
# is needed for the interactive Chat workflow. A slim multi-stage runtime image for
# headless deployment is a separate, later deliverable (see ROADMAP).
#
# Toolchain is pinned to the validated CI/WSL matrix: Ubuntu 26.04 + CUDA 13.3
# (glibc 2.43 -- the path that broke under CUDA 13.0's rsqrt declarations), clang-21
# for the C++23 module units, gcc-15 as nvcc's (module-free) host compiler. No cuDNN
# variant exists for 26.04 and none is needed while USE_CUDNN is unset, so the plain
# `devel` image is correct.
#
# Model weights are converted offline (Mila/Tools/Converters) and bind-mounted at
# run time under Data/Models -- never baked into the image (they are gigabytes and
# separately licensed). See Docker/README.md.
FROM nvidia/cuda:13.3.0-devel-ubuntu26.04

ENV DEBIAN_FRONTEND=noninteractive

# Build toolchain, mirroring .github/workflows/build-pipeline.yml so the container
# and CI stay byte-comparable:
#   clang-21   -- compiles the C++23 module units
#   gcc-15/g++ -- nvcc's host compiler for the .cu files (no modules there)
#   ninja      -- the C++23 module-dyndep generator the build requires
#   ccache     -- persists .cu object codegen across runs (CUDA launcher only;
#                 ccache's module-unit support is unreliable, so C++ is unwrapped)
#   python3    -- CUTLASS's configure needs find_package(Python3 REQUIRED)
# gdb/less are dev-shell niceties. Dependencies (GoogleTest, nlohmann_json, miniz,
# CUTLASS) are fetched by CPM/FetchContent at configure time and are not installed.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        wget \
        ca-certificates \
        ninja-build \
        ccache \
        gcc-15 \
        g++-15 \
        clang-21 \
        python3 \
        python3-dev \
        gdb \
        less \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# CMake 4.2.3 (the project requires 4.0+; the Ubuntu package predates it). Matches CI.
RUN wget -q https://github.com/Kitware/CMake/releases/download/v4.2.3/cmake-4.2.3-linux-x86_64.tar.gz \
    && tar --strip-components=1 -xz -C /usr/local -f cmake-4.2.3-linux-x86_64.tar.gz \
    && rm -f cmake-4.2.3-linux-x86_64.tar.gz

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV CCACHE_DIR=/ccache

# Convenience entry points baked into the image so the exact configure flags live in
# one versioned place (see the scripts for the flag rationale):
#   mila-build-chat  -- configure + build the ChatApp target
#   mila-chat        -- run the built ChatApp
COPY Docker/build-chat.sh /usr/local/bin/mila-build-chat
COPY Docker/run-chat.sh   /usr/local/bin/mila-chat
RUN chmod +x /usr/local/bin/mila-build-chat /usr/local/bin/mila-chat

WORKDIR /mila

CMD ["/bin/bash"]
