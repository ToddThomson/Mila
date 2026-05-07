// CudaGqa.Prefill.Fp32.cu

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include "CudaUtils.h"
#include "CudaGqa.cuh"

namespace Mila::Dnn::Compute::Cuda::Gqa
{
    __global__ void prefill_softmax_fp32_kernel_v2(
        float* att,
        const float* preatt,
        int B,
        int NH,
        int T_stride,
        int chunk_stride,
        int chunk_len,
        int position_offset )
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
        int max_t2 = min( abs_t, T_stride - 1 );  // last key index to attend

        // Step 1: find max for numerical stability
        float max_val = -CUDART_INF_F;
        for ( int t2 = 0; t2 <= max_t2; ++t2 )
            max_val = fmaxf( max_val, preatt_row[ t2 ] );

        // Step 2: exponentiate & sum
        float sum = 0.0f;
        for ( int t2 = 0; t2 <= max_t2; ++t2 )
        {
            float val = expf( preatt_row[ t2 ] - max_val );
            sum += val;
            att_row[ t2 ] = val;
        }

        // Step 3: normalize
        float inv_sum = 1.0f / sum;
        for ( int t2 = 0; t2 <= max_t2; ++t2 )
            att_row[ t2 ] *= inv_sum;

        // Step 4: zero out future tokens
        for ( int t2 = max_t2 + 1; t2 < T_stride; ++t2 )
            att_row[ t2 ] = 0.0f;
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
        int B, int NH, int T_stride, int chunk_stride,
        int chunk_len, int position_offset,
        cudaStream_t stream )
    {
        int total_rows = B * NH * chunk_len;
        int block_size = 256;
        int grid_size = (total_rows + block_size - 1) / block_size;

        prefill_softmax_fp32_kernel_v2 <<<grid_size, block_size, 0, stream >>> (
            att, preatt,
            B, NH, T_stride, chunk_stride,
            chunk_len, position_offset);

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