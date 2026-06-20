// Gqa.Prefill.Bf16.cu

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include "CudaUtils.h"
#include "CudaGqa.cuh"

namespace Mila::Dnn::Compute::Cuda::Gqa
{
    /**
     * @brief Per-row causal softmax over BF16 preattention scores.
     *
     * Mirrors prefill_softmax_fp32_kernel_v2. All arithmetic is performed in
     * float32; BF16 inputs are widened on load and narrowed on store to
     * preserve numerical stability across the exp/normalize pass.
     *
     * Each thread owns one query row (b, nh, t), iterates over key positions
     * [0, abs_t], and zeros positions (abs_t, T_stride).
     *
     * @param att            Output attention weights [B, NH, chunk_stride, T_stride].
     * @param preatt         Input pre-attention logits [B, NH, chunk_stride, T_stride].
     * @param B              Batch size.
     * @param NH             Number of query heads.
     * @param T_stride       Maximum sequence length (row width in memory).
     * @param chunk_stride   Allocated chunk capacity (row pitch for the query axis).
     * @param chunk_len      Number of active query tokens in this chunk (<= chunk_stride).
     * @param position_offset Absolute position of the first token in this chunk.
     */
    __global__ void prefill_softmax_bf16_kernel(
        __nv_bfloat16* att,
        const __nv_bfloat16* preatt,
        int B,
        int NH,
        int T_stride,
        int chunk_stride,
        int chunk_len,
        int position_offset,
        int window )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total_rows = B * NH * chunk_len;

        if ( idx >= total_rows )
            return;

        int b_nh = idx / chunk_len;
        int t = idx % chunk_len;

        int b = b_nh / NH;
        int nh = b_nh % NH;

        // row_offset must use chunk_len (the actual M dimension cuBLASLt used when writing
        // preatt_), not chunk_stride (kPrefillChunkSize). They are equal for full chunks,
        // so single-chunk and full-chunk multi-chunk prefill are unaffected. On a partial
        // final chunk (chunk_len < kPrefillChunkSize), using chunk_stride skips past the
        // actual data for every head beyond the first (b_nh > 0), producing garbage softmax
        // input and therefore garbage attention output for the remainder of the sequence.
        int row_offset = ((b * NH + nh) * chunk_len + t) * T_stride;
        // WAS: int row_offset = ((b * NH + nh) * chunk_stride + t) * T_stride;

        const __nv_bfloat16* preatt_row = preatt + row_offset;
        __nv_bfloat16* att_row = att + row_offset;

        int abs_t = position_offset + t;
        int max_t2 = min( abs_t, T_stride - 1 );

        // Sliding-window lower bound. window <= 0 means global causal (window_start
        // = 0), which reproduces the unbounded behavior exactly.
        int window_start = ( window > 0 ) ? max( 0, abs_t - window + 1 ) : 0;

        // Step 1: find max for numerical stability, promoting to float
        float max_val = -CUDART_INF_F;
        for ( int t2 = window_start; t2 <= max_t2; ++t2 )
            max_val = fmaxf( max_val, __bfloat162float( preatt_row[ t2 ] ) );

        // Step 2: exponentiate and accumulate sum
        float sum = 0.0f;
        for ( int t2 = window_start; t2 <= max_t2; ++t2 )
        {
            float val = expf( __bfloat162float( preatt_row[ t2 ] ) - max_val );
            sum += val;
            att_row[ t2 ] = __float2bfloat16( val );
        }

        // Step 3: normalize
        float inv_sum = 1.0f / sum;
        for ( int t2 = window_start; t2 <= max_t2; ++t2 )
            att_row[ t2 ] = __float2bfloat16( __bfloat162float( att_row[ t2 ] ) * inv_sum );

        // Step 4: zero out positions outside the attended window — below the
        // window [0, window_start) and the future [max_t2+1, T_stride).
        for ( int t2 = 0; t2 < window_start; ++t2 )
            att_row[ t2 ] = __float2bfloat16( 0.0f );

        for ( int t2 = max_t2 + 1; t2 < T_stride; ++t2 )
            att_row[ t2 ] = __float2bfloat16( 0.0f );
    }

    /**
     * @brief Unpack vaccum [B, NQH, padded_T, HS] → out [B, actual_T, NQH*HS].
     *
     * BF16 equivalent of gqa_prefill_unpermute_output_padded_fp32_kernel.
     * No arithmetic; only index permutation — elements are copied as BF16.
     *
     * @param vaccum   Input  [B * NQH * padded_T * HS], head-major layout.
     * @param out      Output [B * actual_T * NQH * HS], token-major layout.
     * @param B        Batch size.
     * @param actual_T Number of valid query tokens in this chunk.
     * @param padded_T Padded chunk capacity (row pitch in vaccum).
     * @param NQH      Number of query heads.
     * @param HS       Head dimension.
     */
    __global__ void gqa_prefill_unpermute_output_padded_bf16_kernel(
        const __nv_bfloat16* __restrict__ vaccum,
        __nv_bfloat16* __restrict__ out,
        int B, int actual_T, int padded_T,
        int NQH, int HS )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int C = NQH * HS;

        int total = B * actual_T * C;

        if ( idx >= total ) return;

        const int b = idx / (actual_T * C);
        int rest = idx % (actual_T * C);
        const int t = rest / C;
        const int c = rest % C;
        const int nh = c / HS;
        const int hs = c % HS;

        const int in_idx =
            b * (NQH * padded_T * HS)
            + nh * (padded_T * HS)
            + t * HS
            + hs;

        out[ idx ] = vaccum[ in_idx ];
    }

    // -------------------------------------------------------------------------
    // Host launchers
    // -------------------------------------------------------------------------

    void cuda_gqa_prefill_softmax_bf16(
        __nv_bfloat16* att, const __nv_bfloat16* preatt,
        int B, int NH, int T_stride, int chunk_stride,
        int chunk_len, int position_offset, int window,
        cudaStream_t stream )
    {
        const int total_rows = B * NH * chunk_len;
        const int block_size = 256;
        const int grid_size = ceil_div( total_rows, block_size );

        prefill_softmax_bf16_kernel <<< grid_size, block_size, 0, stream >>> (
            att, preatt,
            B, NH, T_stride, chunk_stride,
            chunk_len, position_offset, window);

        cudaCheck( cudaGetLastError() );
    }

    void cuda_gqa_prefill_unpermute_output_padded_bf16(
        const __nv_bfloat16* vaccum, __nv_bfloat16* out,
        int B, int actual_T, int padded_T,
        int NQH, int HS,
        cudaStream_t stream )
    {
        const int block_size = 256;
        const int total = B * actual_T * NQH * HS;
        const int num_blocks = ceil_div( total, block_size );

        gqa_prefill_unpermute_output_padded_bf16_kernel << < num_blocks, block_size, 0, stream >> > (
            vaccum, out, B, actual_T, padded_T, NQH, HS);

        cudaCheck( cudaGetLastError() );
    }
}