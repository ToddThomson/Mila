/**
 * @file GqaState.ixx
 * @brief Non-owning transient scratch state for CudaGqaOp inference paths.
 *
 * Tensors are owned by the caller (LlamaTransformer) and shared across all GQA
 * layers. Each layer is called sequentially, so a single allocation suffices.
 */

export module Compute.GqaState;

import Dnn.ITensor;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Non-owning pointers to shared transient GQA scratch buffers.
     *
     * All slots are nullable. CudaGqaOp::setState() accepts a partially populated
     * state — only non-null slots replace the previously wired pointers.
     *
     * Prefill slots:
     *   q_permute     [B, NH, chunk, HS]
     *   preatt        [B, NH, chunk, T]
     *   att           [B, NH, chunk, T]
     *   v_out         [B, NH, chunk, HS]
     *
     * Decode slots:
     *   preatt_decode [B, NH, 1, T]
     *   att_decode    [B, NH, 1, T]
     *   v_out_decode  [B, NH, 1, HS]
     *
     * Ownership: caller retains ownership and must ensure tensors outlive all GQA
     * layers that reference them.
     */
    export struct GqaState
    {
        ITensor* q_permute{ nullptr };
        ITensor* preatt{ nullptr };
        ITensor* att{ nullptr };
        ITensor* v_out{ nullptr };

        ITensor* preatt_decode{ nullptr };
        ITensor* att_decode{ nullptr };
        ITensor* v_out_decode{ nullptr };
    };
}