
#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace Mila::Dnn::Compute::Cuda::Activation
{
    // Functor-templated launchers. The kernel is templated on the native element
    // type (float or __nv_bfloat16) and the elementwise functor; nvcc inlines the
    // functor so every thread runs identical, branch-free code. Definitions and the
    // explicit instantiations for each (native, functor) pair live in the matching
    // .cu. These declarations are functor-agnostic so the op module unit (compiled by
    // the host C++ compiler, not nvcc) can name the launcher without seeing any
    // CUDA kernel syntax; the symbol resolves at link from the .cu instantiations.

    template<typename TNative, typename TFunctor>
    void launch_elementwise_forward(
        TNative* Y,
        const TNative* X,
        int N,
        TFunctor functor,
        cudaStream_t stream );

    template<typename TNative, typename TFunctor>
    void launch_elementwise_backward(
        TNative* dX,
        const TNative* X,
        const TNative* dY,
        int N,
        TFunctor functor,
        cudaStream_t stream );
}
