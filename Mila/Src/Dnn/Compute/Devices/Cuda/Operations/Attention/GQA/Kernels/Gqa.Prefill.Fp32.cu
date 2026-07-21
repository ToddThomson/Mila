// CudaGqa.Prefill.Fp32.cu

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include "CudaUtils.h"
#include "CudaGqa.cuh"

namespace Mila::Dnn::Compute::Cuda::Gqa
{
    // T_stride is the PHYSICAL row width (KV cache capacity) used for addressing;
    // attended_len is the LOGICAL number of valid keys for this chunk
    // (position_offset + chunk_len). Decoupling them lets a short prompt over a large
    // allocated context touch only attended_len columns. The AV GEMM reads columns
    // [0, attended_len), so [attended_len, T_stride) are never consumed and need no
    // zeroing. See GqaAttentionExtent.md.
    __global__ void prefill_softmax_fp32_kernel_v2(
        float* att,
        const float* preatt,
        int B,
        int NH,
        int T_stride,
        int attended_len,
        int chunk_stride,
        int chunk_len,
        int position_offset,
        int window )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total_rows = B * NH * chunk_len;

        if ( idx >= total_rows )
            return;

        // Decode batch, head, and query index within chunk
        int b_nh = idx / chunk_len;   // flattened batch * heads
        int t = idx % chunk_len;  // query index in this chunk

        int b = b_nh / NH;
        int nh = b_nh % NH;

        // Compute row start for this query
        int row_offset = ((b * NH + nh) * chunk_len + t) * T_stride;

        const float* preatt_row = preatt + row_offset;
        float* att_row = att + row_offset;

        // Compute causal mask limit: cannot attend to future tokens
        int abs_t = position_offset + t;        // global query index
        int max_t2 = min( abs_t, attended_len - 1 );  // last key index to attend

        // Sliding-window lower bound. window <= 0 means global causal (window_start
        // = 0), which reproduces the unbounded behavior exactly.
        int window_start = ( window > 0 ) ? max( 0, abs_t - window + 1 ) : 0;

        // Step 1: find max for numerical stability
        float max_val = -CUDART_INF_F;
        for ( int t2 = window_start; t2 <= max_t2; ++t2 )
            max_val = fmaxf( max_val, preatt_row[ t2 ] );

        // Step 2: exponentiate & sum
        float sum = 0.0f;
        for ( int t2 = window_start; t2 <= max_t2; ++t2 )
        {
            float val = expf( preatt_row[ t2 ] - max_val );
            sum += val;
            att_row[ t2 ] = val;
        }

        // Step 3: normalize
        float inv_sum = 1.0f / sum;
        for ( int t2 = window_start; t2 <= max_t2; ++t2 )
            att_row[ t2 ] *= inv_sum;

        // Step 4: zero out positions the AV GEMM will read but this row does not
        // attend — below the window [0, window_start) and the causal future
        // [max_t2+1, attended_len). Columns [attended_len, T_stride) are outside the
        // AV GEMM's K extent, so they are never read and are left untouched.
        for ( int t2 = 0; t2 < window_start; ++t2 )
            att_row[ t2 ] = 0.0f;

        for ( int t2 = max_t2 + 1; t2 < attended_len; ++t2 )
            att_row[ t2 ] = 0.0f;
    }


    // Bounded sliding-window ring prefill softmax. preatt/att rows have `capacity`
    // columns, where column j is RING SLOT j. Slot j holds the key at absolute
    // position p_j = end - ((r - j + capacity) % capacity), where end is the cache's
    // newest position (position_offset + chunk_len - 1) and r = end % capacity. A
    // query at abs_t keeps slot j iff window_start <= p_j <= abs_t — the causal upper
    // bound excludes same-chunk-future keys already written into the ring. One thread
    // owns a row, so ring (rotated) slot order is irrelevant. See SlidingWindowKvCache.md D6.
    __global__ void prefill_softmax_ring_fp32_kernel(
        float* att,
        const float* preatt,
        int B,
        int NH,
        int capacity,
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

        int row_offset = ( b_nh * chunk_len + t ) * capacity;

        const float* preatt_row = preatt + row_offset;
        float* att_row = att + row_offset;

        const int abs_t = position_offset + t;
        const int window_start = ( window > 0 ) ? max( 0, abs_t - window + 1 ) : 0;
        const int end = position_offset + chunk_len - 1;
        const int r = end % capacity;

        float max_val = -CUDART_INF_F;
        for ( int j = 0; j < capacity; ++j )
        {
            const int p = end - ( ( r - j + capacity ) % capacity );

            if ( p >= window_start && p <= abs_t )
                max_val = fmaxf( max_val, preatt_row[ j ] );
        }

        float sum = 0.0f;
        for ( int j = 0; j < capacity; ++j )
        {
            const int p = end - ( ( r - j + capacity ) % capacity );

            if ( p >= window_start && p <= abs_t )
            {
                float val = expf( preatt_row[ j ] - max_val );
                sum += val;
                att_row[ j ] = val;
            }
            else
            {
                att_row[ j ] = 0.0f;
            }
        }

        float inv_sum = 1.0f / sum;
        for ( int j = 0; j < capacity; ++j )
        {
            const int p = end - ( ( r - j + capacity ) % capacity );

            if ( p >= window_start && p <= abs_t )
                att_row[ j ] *= inv_sum;
        }
    }


    __global__ void gqa_prefill_unpermute_output_padded_fp32_kernel(
        const float* __restrict__ vaccum,
        float* __restrict__ out,
        int B, int actual_T, int padded_T,
        int NQH, int HS )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int C = NQH * HS; // channels per token

        // Total number of output elements: B * actual_T * NQH * HS
        int total = B * actual_T * C;

        if ( idx >= total ) return;

        // Decode flat index → (b, t, nh, hs)
        const int b = idx / (actual_T * C);
        int rest = idx % (actual_T * C);
        const int t = rest / C;
        const int c = rest % C;
        const int nh = c / HS;
        const int hs = c % HS;

        // Read from vaccum: [B][NQH][padded_T][HS]
        const int in_idx =
            b * (NQH * padded_T * HS)
            + nh * (padded_T * HS)
            + t * HS
            + hs;

        // Write to out: [B][actual_T][NQH][HS]
        out[ idx ] = vaccum[ in_idx ];
    }

    // -----------------------------------------------------------------------
    // Prefill launchers
    // -----------------------------------------------------------------------

    void cuda_gqa_prefill_softmax_fp32(
        float* att, const float* preatt,
        int B, int NH, int T_stride, int attended_len, int chunk_stride,
        int chunk_len, int position_offset, int window,
        cudaStream_t stream )
    {
        int total_rows = B * NH * chunk_len;
        int block_size = 256;
        int grid_size = (total_rows + block_size - 1) / block_size;

        prefill_softmax_fp32_kernel_v2 <<<grid_size, block_size, 0, stream >>> (
            att, preatt,
            B, NH, T_stride, attended_len, chunk_stride,
            chunk_len, position_offset, window);

        cudaCheck( cudaGetLastError() );
    }

    void cuda_gqa_prefill_softmax_ring_fp32(
        float* att, const float* preatt,
        int B, int NH, int capacity,
        int chunk_len, int position_offset, int window,
        cudaStream_t stream )
    {
        int total_rows = B * NH * chunk_len;
        int block_size = 256;
        int grid_size = (total_rows + block_size - 1) / block_size;

        prefill_softmax_ring_fp32_kernel <<<grid_size, block_size, 0, stream >>> (
            att, preatt,
            B, NH, capacity,
            chunk_len, position_offset, window);

        cudaCheck( cudaGetLastError() );
    }

    void cuda_gqa_prefill_unpermute_output_padded_fp32(
        const float* vaccum, float* out,
        int B, int actual_T, int padded_T,
        int NQH, int HS,
        cudaStream_t stream )
    {
        const int block_size = 256;
        const int total = B * actual_T * NQH * HS;
        const int num_blocks = (total + block_size - 1) / block_size;

        gqa_prefill_unpermute_output_padded_fp32_kernel <<< num_blocks, block_size, 0, stream >>> (
                vaccum, out, B, actual_T, padded_T, NQH, HS );

        cudaCheck( cudaGetLastError() );
    }

}