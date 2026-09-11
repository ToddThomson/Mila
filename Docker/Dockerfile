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
#   python3    -- CUTLASS's configure needs find_package(Python3 REQUIRED); also the
#                 interpreter the mila binding is built against + MIS runs under
#   python3-venv -- so mila-build-mis can create the MIS server venv (Debian/Ubuntu
#                 split venv/ensurepip out of the base python3 package)
#   libssl-dev -- curl's configure does find_package(OpenSSL) and fails the whole build
#                 without it. This image builds Chat at the MILA_ENABLE_LIBCURL default
#                 (ON), unlike Dockerfile.wheel which omits OpenSSL deliberately because
#                 the wheel forces it OFF. Chat in the container needs a real transport:
#                 /install pulls into the store on the bind mount.
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
        libssl-dev \
        python3 \
        python3-dev \
        python3-venv \
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

# The Mila model store, on the bind mount rather than in the container's own home. Chat
# and MIS both read it and neither downloads on load, so a store on a container-local
# layer would be lost on `run --rm` and have to be re-pulled -- gigabytes each time. Here
# it is the same directory the host sees under Data/Models/Store.
ENV MILA_CACHE_DIR=/mila/Data/Models/Store

# Cap concurrent compile jobs so an IDE-driven build (VS 2026 / VSCode CMake Tools,
# which do not read the mila-build-* scripts' -j) cannot starve this container's RAM on
# the several-GB module TUs (OperationTraits.ixx et al.) and hang. The root CMakeLists
# turns this into a Ninja compile job pool; 4 matches the scripts' default MILA_BUILD_JOBS.
# A RAM-rich WSL2 VM can raise it (and %USERPROFILE%\.wslconfig memory=).
ENV MILA_COMPILE_JOB_POOL_SIZE=4

# Convenience entry points baked into the image so the exact configure flags live in
# one versioned place (see the scripts for the flag rationale):
#   mila-build-chat  -- configure + build just the mila-chat target (fast path)
#   mila-build-all   -- configure + build the full product set (library, samples,
#                       Chat, Python binding); single-arch for a fast one-GPU build
#   mila-build-mis   -- build the mila binding + the MIS Python server venv
#   mila-chat        -- run the built mila-chat
#   mila-mis         -- run the Mila Inference Server (HTTP wire adaptor)
COPY Docker/build-chat.sh /usr/local/bin/mila-build-chat
COPY Docker/build-all.sh  /usr/local/bin/mila-build-all
COPY Docker/build-mis.sh  /usr/local/bin/mila-build-mis
COPY Docker/run-chat.sh   /usr/local/bin/mila-chat
COPY Docker/run-mis.sh    /usr/local/bin/mila-mis
RUN chmod +x /usr/local/bin/mila-build-chat /usr/local/bin/mila-build-all \
             /usr/local/bin/mila-build-mis /usr/local/bin/mila-chat /usr/local/bin/mila-mis

WORKDIR /mila

CMD ["/bin/bash"]
