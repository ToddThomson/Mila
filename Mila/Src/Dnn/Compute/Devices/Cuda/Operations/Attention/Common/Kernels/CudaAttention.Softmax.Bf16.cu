/**
 * @file CudaAttention.Softmax.Bf16.cu
 * @brief BF16 softmax kernels for attention operations.
 *
 * Implements the BF16 softmax family declared in CudaAttention.cuh. All
 * arithmetic is promoted to float32; BF16 values are widened on load and
 * narrowed on store. These implementations are shared across MHA and GQA.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "device_launch_parameters.h"
#include "CudaUtils.h"
#include "CudaAttention.cuh"

namespace Mila::Dnn::Compute::Cuda::Attention::Common
{
    // ========================================================================
    // BF16 Softmax Kernels
    // ========================================================================

    __global__ void softmax_forward_bf16_kernel(
        __nv_bfloat16* att, float scale, const __nv_bfloat16* preatt,
        int B_NH, int T )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total_rows = B_NH * T;

        if ( idx < total_rows )
        {
            int b_nh = idx / T;
            int t = idx % T;

            const __nv_bfloat16* preatt_row = preatt + b_nh * (T * T) + t * T;
            __nv_bfloat16* att_row = att + b_nh * (T * T) + t * T;

            float max_val = -INFINITY;

            for ( int t2 = 0; t2 <= t; ++t2 )
                max_val = fmaxf( max_val, __bfloat162float( preatt_row[ t2 ] ) );

            float sum = 0.0f;

            for ( int t2 = 0; t2 <= t; ++t2 )
            {
                float val = expf( (__bfloat162float( preatt_row[ t2 ] ) - max_val) * scale );
                sum += val;
                att_row[ t2 ] = __float2bfloat16( val );
            }

            float inv_sum = 1.0f / sum;

            for ( int t2 = 0; t2 <= t; ++t2 )
                att_row[ t2 ] = __float2bfloat16( __bfloat162float( att_row[ t2 ] ) * inv_sum );

            for ( int t2 = t + 1; t2 < T; ++t2 )
                att_row[ t2 ] = __float2bfloat16( 0.0f );
        }
    }

    __global__ void softmax_padded_forward_bf16_kernel(
        __nv_bfloat16* att, float scale, const __nv_bfloat16* preatt,
        int B_NH, int T, int actual_T )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total_rows = B_NH * T;

        if ( idx < total_rows )
        {
            int b_nh = idx / T;
            int t = idx % T;

            __nv_bfloat16* att_row = att + b_nh * (T * T) + t * T;

            if ( t >= actual_T )
            {
                for ( int t2 = 0; t2 < T; ++t2 )
                    att_row[ t2 ] = __float2bfloat16( 0.0f );

                return;
            }

            const __nv_bfloat16* preatt_row = preatt + b_nh * (T * T) + t * T;

            int max_t2 = min( t, actual_T - 1 );
            float max_val = -INFINITY;

            for ( int t2 = 0; t2 <= max_t2; ++t2 )
                max_val = fmaxf( max_val, __bfloat162float( preatt_row[ t2 ] ) );

            float sum = 0.0f;

            for ( int t2 = 0; t2 <= max_t2; ++t2 )
            {
                float val = expf( (__bfloat162float( preatt_row[ t2 ] ) - max_val) * scale );
                sum += val;
                att_row[ t2 ] = __float2bfloat16( val );
            }

            float inv_sum = 1.0f / sum;

            for ( int t2 = 0; t2 <= max_t2; ++t2 )
                att_row[ t2 ] = __float2bfloat16( __bfloat162float( att_row[ t2 ] ) * inv_sum );

            for ( int t2 = max_t2 + 1; t2 < T; ++t2 )
                att_row[ t2 ] = __float2bfloat16( 0.0f );
        }
    }

    // Decode softmax: one row per warp. Each warp strides its row in fp32 with
    // butterfly reductions; att is written once (narrowed to bf16 at store), so
    // there is no unnormalized-exp round-trip through global memory. The scale
    // factor is omitted -- at decode the 1/sqrt(head_size) term is folded into
    // the QK GEMM alpha, so it is always 1.0 here.
    __global__ void softmax_decode_forward_bf16_kernel(
        __nv_bfloat16* att, const __nv_bfloat16* preatt,
        int B_NH, int max_len, int actual_len, int window )
    {
        const int lane = threadIdx.x % warpSize;
        const int warp_id = threadIdx.x / warpSize;
        const int row = blockIdx.x * (blockDim.x / warpSize) + warp_id;

        // Uniform across the warp (row depends only on warp_id), so the full mask
        // stays valid for the reductions below.
        if ( row >= B_NH )
            return;

        const __nv_bfloat16* preatt_row = preatt + row * max_len;
        __nv_bfloat16* att_row = att + row * max_len;

        // Sliding-window lower bound (uniform across the warp). window <= 0 means
        // global (window_start = 0), reproducing the unbounded decode exactly.
        const int window_start = ( window > 0 ) ? max( 0, actual_len - window ) : 0;

        // Pass 1: row max.
        float thread_max = -INFINITY;

        for ( int t2 = window_start + lane; t2 < actual_len; t2 += warpSize )
            thread_max = fmaxf( thread_max, __bfloat162float( preatt_row[ t2 ] ) );

        for ( int offset = warpSize / 2; offset > 0; offset >>= 1 )
            thread_max = fmaxf( thread_max, __shfl_xor_sync( 0xffffffffu, thread_max, offset ) );

        const float max_val = thread_max;

        // Pass 2: sum of exp, fp32 accumulation.
        float thread_sum = 0.0f;

        for ( int t2 = window_start + lane; t2 < actual_len; t2 += warpSize )
            thread_sum += expf( __bfloat162float( preatt_row[ t2 ] ) - max_val );

        for ( int offset = warpSize / 2; offset > 0; offset >>= 1 )
            thread_sum += __shfl_xor_sync( 0xffffffffu, thread_sum, offset );

        const float inv_sum = 1.0f / thread_sum;

        // Pass 3: normalize and store.
        for ( int t2 = window_start + lane; t2 < actual_len; t2 += warpSize )
            att_row[ t2 ] = __float2bfloat16( expf( __bfloat162float( preatt_row[ t2 ] ) - max_val ) * inv_sum );

        // Zero the positions outside the window: below [0, window_start) and the
        // unused tail [actual_len, max_len).
        for ( int t2 = lane; t2 < window_start; t2 += warpSize )
            att_row[ t2 ] = __float2bfloat16( 0.0f );

        for ( int t2 = actual_len + lane; t2 < max_len; t2 += warpSize )
            att_row[ t2 ] = __float2bfloat16( 0.0f );
    }

    // Bounded sliding-window ring decode softmax. preatt/att rows have `capacity`
    // columns, where column j is RING SLOT j (not an absolute position). Slot j
    // holds the key at absolute position
    //   p_j = end - ((r - j + capacity) % capacity),   end = actual_len - 1, r = end % capacity,
    // which lies in (end - capacity, end]. The slot is in-window (and therefore
    // resident) iff p_j >= window_start. Each slot is read/written by exactly one
    // lane, so there is no cross-lane race and ring (rotated) order is irrelevant
    // to the result. See SlidingWindowKvCache.md D6.
    __global__ void softmax_decode_ring_forward_bf16_kernel(
        __nv_bfloat16* att, const __nv_bfloat16* preatt,
        int B_NH, int capacity, int actual_len, int window )
    {
        const int lane = threadIdx.x % warpSize;
        const int warp_id = threadIdx.x / warpSize;
        const int row = blockIdx.x * (blockDim.x / warpSize) + warp_id;

        if ( row >= B_NH )
            return;

        const __nv_bfloat16* preatt_row = preatt + row * capacity;
        __nv_bfloat16* att_row = att + row * capacity;

        const int end = actual_len - 1;
        const int window_start = ( window > 0 ) ? max( 0, actual_len - window ) : 0;
        const int r = end % capacity;

        // Pass 1: row max over the in-window slots.
        float thread_max = -INFINITY;

        for ( int j = lane; j < capacity; j += warpSize )
        {
            const int p = end - ( ( r - j + capacity ) % capacity );

            if ( p >= window_start )
                thread_max = fmaxf( thread_max, __bfloat162float( preatt_row[ j ] ) );
        }

        for ( int offset = warpSize / 2; offset > 0; offset >>= 1 )
            thread_max = fmaxf( thread_max, __shfl_xor_sync( 0xffffffffu, thread_max, offset ) );

        const float max_val = thread_max;

        // Pass 2: sum of exp over the in-window slots.
        float thread_sum = 0.0f;

        for ( int j = lane; j < capacity; j += warpSize )
        {
            const int p = end - ( ( r - j + capacity ) % capacity );

            if ( p >= window_start )
                thread_sum += expf( __bfloat162float( preatt_row[ j ] ) - max_val );
        }

        for ( int offset = warpSize / 2; offset > 0; offset >>= 1 )
            thread_sum += __shfl_xor_sync( 0xffffffffu, thread_sum, offset );

        const float inv_sum = 1.0f / thread_sum;

        // Pass 3: normalize in-window slots, zero the rest. One writer per slot.
        for ( int j = lane; j < capacity; j += warpSize )
        {
            const int p = end - ( ( r - j + capacity ) % capacity );

            att_row[ j ] = ( p >= window_start )
                ? __float2bfloat16( expf( __bfloat162float( preatt_row[ j ] ) - max_val ) * inv_sum )
                : __float2bfloat16( 0.0f );
        }
    }

    __global__ void softmax_backward_bf16_kernel(
        __nv_bfloat16* dpreatt, const __nv_bfloat16* datt, const __nv_bfloat16* att,
        float scale,
        int B_NH, int T )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total_rows = B_NH * T;

        if ( idx < total_rows )
        {
            int b_nh = idx / T;
            int t = idx % T;

            const __nv_bfloat16* att_row = att + b_nh * (T * T) + t * T;
            const __nv_bfloat16* datt_row = datt + b_nh * (T * T) + t * T;
            __nv_bfloat16* dpreatt_row = dpreatt + b_nh * (T * T) + t * T;

            float sum = 0.0f;

            for ( int t2 = 0; t2 <= t; ++t2 )
                sum += __bfloat162float( datt_row[ t2 ] ) * __bfloat162float( att_row[ t2 ] );

            for ( int t2 = 0; t2 <= t; ++t2 )
            {
                float grad = scale * __bfloat162float( att_row[ t2 ] ) *
                    (__bfloat162float( datt_row[ t2 ] ) - sum);
                dpreatt_row[ t2 ] = __float2bfloat16( grad );
            }

            for ( int t2 = t + 1; t2 < T; ++t2 )
                dpreatt_row[ t2 ] = __float2bfloat16( 0.0f );
        }
    }

    // ========================================================================
    // Host launchers
    // ========================================================================

    void cuda_attention_softmax_forward_bf16(
        __nv_bfloat16* att, float scale, const __nv_bfloat16* preatt,
        int B, int NH, int T,
        cudaStream_t stream )
    {
        const int block_size = 256;
        const int B_NH = B * NH;
        const int num_blocks = ceil_div( B_NH * T, block_size );

        softmax_forward_bf16_kernel << < num_blocks, block_size, 0, stream >> > (
            att, scale, preatt, B_NH, T);

        cudaCheck( cudaGetLastError() );
    }

    void cuda_attention_softmax_padded_forward_bf16(
        __nv_bfloat16* att, float scale, const __nv_bfloat16* preatt,
        int B, int NH, int max_T, int actual_T,
        cudaStream_t stream )
    {
        const int block_size = 256;
        const int B_NH = B * NH;
        const int num_blocks = ceil_div( B_NH * max_T, block_size );

        softmax_padded_forward_bf16_kernel << < num_blocks, block_size, 0, stream >> > (
            att, scale, preatt, B_NH, max_T, actual_T);

        cudaCheck( cudaGetLastError() );
    }

    void cuda_attention_softmax_decode_forward_bf16(
        __nv_bfloat16* att, float scale, const __nv_bfloat16* preatt,
        int B, int NH, int max_len, int actual_len,
        cudaStream_t stream, int window )
    {
        // scale is kept for signature symmetry with the fp32/fp16 launchers but
        // is unused: the decode QK GEMM already folds 1/sqrt(head_size) into its
        // alpha, so the softmax operand is pre-scaled.
        (void) scale;

        constexpr int warps_per_block = 8;
        const int block_size = warps_per_block * 32;
        const int B_NH = B * NH;
        const int num_blocks = ceil_div( B_NH, warps_per_block );

        softmax_decode_forward_bf16_kernel <<< num_blocks, block_size, 0, stream >>> (
            att, preatt, B_NH, max_len, actual_len, window);

        cudaCheck( cudaGetLastError() );
    }

    void cuda_attention_softmax_decode_ring_forward_bf16(
        __nv_bfloat16* att, float scale, const __nv_bfloat16* preatt,
        int B, int NH, int capacity, int actual_len,
        cudaStream_t stream, int window )
    {
        // scale unused: decode folds 1/sqrt(head_size) into the QK GEMM alpha.
        (void) scale;

        constexpr int warps_per_block = 8;
        const int block_size = warps_per_block * 32;
        const int B_NH = B * NH;
        const int num_blocks = ceil_div( B_NH, warps_per_block );

        softmax_decode_ring_forward_bf16_kernel <<< num_blocks, block_size, 0, stream >>> (
            att, preatt, B_NH, capacity, actual_len, window);

        cudaCheck( cudaGetLastError() );
    }

    void cuda_attention_softmax_backward_bf16(
        __nv_bfloat16* dpreatt, const __nv_bfloat16* datt, const __nv_bfloat16* att,
        float scale,
        int B, int NH, int T,
        cudaStream_t stream )
    {
        const int block_size = 256;
        const int B_NH = B * NH;
        const int num_blocks = ceil_div( B_NH * T, block_size );

        softmax_backward_bf16_kernel << < num_blocks, block_size, 0, stream >> > (
            dpreatt, datt, att, scale, B_NH, T);

        cudaCheck( cudaGetLastError() );
    }
}