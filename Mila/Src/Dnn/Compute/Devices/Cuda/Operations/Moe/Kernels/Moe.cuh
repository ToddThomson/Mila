#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace Mila::Dnn::Compute::Cuda::Moe
{
    // Mixture-of-experts bank, two passes over the stacked tensors in place (see Gemma4MoE.md Phase 6).
    //
    // The launchers are templates so the module side (compiled by the host C++ compiler, not nvcc)
    // can name them without seeing kernel syntax; Moe.cu instantiates them for every supported
    // (native type, gate functor) pair and the symbols resolve at link.

    /**
     * @brief Pass 1: gated[token, slot, i] = gate_functor( gate ) * up, in FP32.
     *
     * gate and up are rows i and I + i of gate_up[ expert ] applied to input[ token ]. One thread per
     * (token, slot, i), nothing shared. An out-of-range expert index writes NaN.
     */
    template<typename TNative, typename TFunctor>
    void launch_moe_gated_forward(
        const TNative* input, const TNative* gate_up, const int32_t* indices, float* gated,
        int tokens, int hidden, int intermediate, int experts, int top_k,
        TFunctor functor, cudaStream_t stream );

    /**
     * @brief Pass 2: output[token, j] = sum over slots of weight * ( down[ expert ][ j ] . gated ).
     *
     * One thread per (token, j), nothing shared. An out-of-range expert index writes NaN.
     */
    template<typename TNative>
    void launch_moe_combine_forward(
        const float* gated, const TNative* down, const TNative* weights, const int32_t* indices, TNative* output,
        int tokens, int hidden, int intermediate, int experts, int top_k,
        cudaStream_t stream );
}
