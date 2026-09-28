// Gqa.Cache.Fp8.cu
//
// The KV-cache write for PerTokenKvFp8: each row -- one KV head at one token -- stored as E4M3 codes with one FP32
// scale, absmax / 448, computed from the row as it is written. Quantization.md, Part III.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include "CudaUtils.h"
#include "CudaGqa.cuh"

namespace Mila::Dnn::Compute::Cuda::Gqa
{
    namespace
    {
        constexpr int kWarpsPerBlock = 8;

        // One warp per row of K and of V. Each lane holds HS / 32 values of the row, so the absmax is a warp reduction
        // and the row is read once.
        template<int kHeadSize>
        __global__ void kvcache_write_kv_fp8_kernel(
            __nv_fp8_e4m3* __restrict__ K, __nv_fp8_e4m3* __restrict__ V,
            float* __restrict__ k_scales, float* __restrict__ v_scales,
            const __nv_bfloat16* __restrict__ Xk, const __nv_bfloat16* __restrict__ Xv,
            int rows, int chunk_len, int NKV, int start_pos, int capacity )
        {
            constexpr int kPerLane = kHeadSize / 32;

            const int lane = threadIdx.x & 31;
            const int warp_row = blockIdx.x * kWarpsPerBlock + ( threadIdx.x >> 5 );

            // Rows 0..rows-1 are K rows, rows..2*rows-1 are V rows; both follow the [B, chunk, NKV] order of the input.
            if ( warp_row >= 2 * rows )
            {
                return;
            }

            const bool is_value = warp_row >= rows;
            const int row = is_value ? warp_row - rows : warp_row;

            const int b = row / ( chunk_len * NKV );
            const int rest = row % ( chunk_len * NKV );
            const int t = rest / NKV;
            const int nkv = rest % NKV;

            const __nv_bfloat16* source = ( is_value ? Xv : Xk ) + static_cast<size_t>( row ) * kHeadSize;

            float values[ kPerLane ];
            float absmax = 0.0f;

#pragma unroll
            for ( int i = 0; i < kPerLane; ++i )
            {
                values[ i ] = __bfloat162float( source[ lane + i * 32 ] );
                absmax = fmaxf( absmax, fabsf( values[ i ] ) );
            }

#pragma unroll
            for ( int offset = 16; offset > 0; offset >>= 1 )
            {
                absmax = fmaxf( absmax, __shfl_xor_sync( 0xffffffffu, absmax, offset ) );
            }

            // The ring wrap is the identity for the unbounded cache, as in the BF16 write.
            const int slot = ( start_pos + t ) % capacity;
            const size_t cache_row = ( static_cast<size_t>( b ) * NKV + nkv ) * capacity + slot;

            const float scale = absmax / 448.0f;
            const float inverse = absmax > 0.0f ? 1.0f / scale : 0.0f;

            __nv_fp8_e4m3* destination = ( is_value ? V : K ) + cache_row * kHeadSize;

#pragma unroll
            for ( int i = 0; i < kPerLane; ++i )
            {
                destination[ lane + i * 32 ].__x = __nv_cvt_float_to_fp8( values[ i ] * inverse, __NV_SATFINITE, __NV_E4M3 );
            }

            if ( lane == 0 )
            {
                ( is_value ? v_scales : k_scales )[ cache_row ] = scale;
            }
        }

        template<int kHeadSize>
        void launch( __nv_fp8_e4m3* K, __nv_fp8_e4m3* V, float* k_scales, float* v_scales,
            const __nv_bfloat16* Xk, const __nv_bfloat16* Xv,
            int batch, int chunk_len, int NKV, int start_pos, int capacity, cudaStream_t stream )
        {
            const int rows = batch * chunk_len * NKV;
            const int blocks = ceil_div( 2 * rows, kWarpsPerBlock );

            kvcache_write_kv_fp8_kernel<kHeadSize><<<blocks, kWarpsPerBlock * 32, 0, stream>>>(
                K, V, k_scales, v_scales, Xk, Xv, rows, chunk_len, NKV, start_pos, capacity );
        }
    }

    bool cuda_gqa_kvcache_write_kv_fp8_supported( int head_size )
    {
        return head_size == 128 || head_size == 256 || head_size == 512;
    }

    void cuda_gqa_kvcache_write_kv_fp8(
        __nv_fp8_e4m3* K, __nv_fp8_e4m3* V,
        float* k_scales, float* v_scales,
        const __nv_bfloat16* Xk, const __nv_bfloat16* Xv,
        int batch, int chunk_len,
        int NKV, int HS,
        int start_pos, int capacity,
        cudaStream_t stream )
    {
        switch ( HS )
        {
            case 128:
                launch<128>( K, V, k_scales, v_scales, Xk, Xv, batch, chunk_len, NKV, start_pos, capacity, stream );
                break;

            case 256:
                launch<256>( K, V, k_scales, v_scales, Xk, Xv, batch, chunk_len, NKV, start_pos, capacity, stream );
                break;

            case 512:
                launch<512>( K, V, k_scales, v_scales, Xk, Xv, batch, chunk_len, NKV, start_pos, capacity, stream );
                break;

            default:
                return;
        }

        cudaCheck( cudaGetLastError() );
    }
}
