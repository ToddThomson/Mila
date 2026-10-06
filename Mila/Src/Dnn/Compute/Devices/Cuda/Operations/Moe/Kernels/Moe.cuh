#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstddef>
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

    /**
     * @brief Pass 1 over a Q4_0 bank as a gather-matvec: one warp per (token, slot, i), each lane a whole 32-code
     *        group of the gate and up rows at a time. The decode path (MixtureOfExperts.md section 6).
     *
     * Each group sums x times ( code - 8 ) in FP32, then scales, so a weight is never formed. hidden must be a
     * multiple of 32. Every token's values are the ones a one-token launch gives it, bit for bit; several tokens'
     * blocks for the same rows run together, so an expert they share comes from memory once (Gemma4Mtp.md 4.7).
     */
    template<typename TFunctor>
    void launch_moe_gated_gather_int4(
        const __nv_bfloat16* input, const uint8_t* gate_up, const __half* gate_up_scales, const int32_t* indices,
        float* gated, int tokens, int hidden, int intermediate, int experts, int top_k,
        TFunctor functor, cudaStream_t stream );

    /**
     * @brief Pass 2 over a Q4_0 bank as a gather-matvec: one warp per (token, j) over every slot's down row, the
     *        token's gated values staged in shared memory. intermediate must be a multiple of 32. Several tokens run
     *        as launch_moe_gated_gather_int4's do: each one's output is its one-token launch's.
     */
    void launch_moe_combine_gather_int4(
        const float* gated, const uint8_t* down, const __half* down_scales, const __nv_bfloat16* weights,
        const int32_t* indices, __nv_bfloat16* output,
        int tokens, int hidden, int intermediate, int experts, int top_k,
        cudaStream_t stream );

    /// Most experts the grouped prefill's routing kernel counts in shared memory.
    inline constexpr int kGroupedMaximumExperts = 256;

    /**
     * @brief Bytes of scratch the grouped prefill needs for @p tokens tokens: the routing, the tile table and the
     *        INT8 activations, which the input and then the gated values occupy in turn.
     *
     * When @p gated_capacity cannot hold one token's top_k combine rows (fewer than hidden / intermediate built
     * tokens), the scratch holds them too.
     */
    size_t moe_grouped_scratch_bytes(
        int tokens, int hidden, int intermediate, int experts, int top_k, int64_t gated_capacity );

    /**
     * @brief Prefill over a Q4_0 bank as two grouped INT8 GEMMs (MixtureOfExperts.md section 6).
     *
     * The rows are permuted into expert segments, token-ascending within each. The input is quantized to INT8
     * per 32-element block, as Linear's Q4_0 prefill; the gated pass multiplies each segment against its
     * expert's gate and up rows interleaved, so the activation runs on registers, and writes FP32 gated values.
     * Those are quantized the same way, and the combine pass writes each (token, slot)'s down projection in FP32
     * over the gated buffer, now dead, as many tokens at a time as it holds; each token's rows are then summed
     * in slot order, weighted, and stored in BF16. Every output depends on its own row alone, so the result is
     * the same however tokens are batched. An out-of-range expert index writes NaN for its token.
     *
     * @param gated          FP32 scratch of gated_capacity elements, at least tokens x top_k x intermediate.
     * @param scratch        moe_grouped_scratch_bytes( ... ) bytes, 16-byte aligned.
     * @param hidden         A multiple of 64, as is intermediate.
     */
    template<typename TFunctor>
    void launch_moe_grouped_prefill_int4(
        const __nv_bfloat16* input, const uint8_t* gate_up, const __half* gate_up_scales,
        const uint8_t* down, const __half* down_scales, const __nv_bfloat16* weights, const int32_t* indices,
        float* gated, int64_t gated_capacity, __nv_bfloat16* output, void* scratch,
        int tokens, int hidden, int intermediate, int experts, int top_k,
        TFunctor functor, cudaStream_t stream );
}
