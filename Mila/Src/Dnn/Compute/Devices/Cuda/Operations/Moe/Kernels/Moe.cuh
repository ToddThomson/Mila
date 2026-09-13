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

    /**
     * @brief Pass 1 over a per-group FP4 bank, BF16 activations.
     *
     * gate_up rows are packed [.., hidden / 2] (low nibble = even column) with FP32 scales
     * [.., hidden / group_size]. Each weight is its decoded nibble times its group scale, accumulated in
     * launch_moe_gated_forward's order, so weights FP4 represents exactly give that pass's bits.
     */
    template<typename TFunctor>
    void launch_moe_gated_forward_fp4(
        const __nv_bfloat16* input, const uint8_t* gate_up, const float* gate_up_scales, const int32_t* indices,
        float* gated, int tokens, int hidden, int intermediate, int experts, int top_k, int group_size,
        TFunctor functor, cudaStream_t stream );

    /**
     * @brief Pass 2 over a per-group FP4 bank: down rows packed [.., intermediate / 2], scales
     *        [.., intermediate / group_size], accumulated in launch_moe_combine_forward's order.
     */
    void launch_moe_combine_forward_fp4(
        const float* gated, const uint8_t* down, const float* down_scales, const __nv_bfloat16* weights,
        const int32_t* indices, __nv_bfloat16* output,
        int tokens, int hidden, int intermediate, int experts, int top_k, int group_size,
        cudaStream_t stream );
}
