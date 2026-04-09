/**
 * @file CudaMatVecBias.Bf16.cu
 * @brief BF16 matrix-vector multiply for the M=1 decode path.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "device_launch_parameters.h"
#include <cassert>

namespace Mila::Dnn::Compute::Cuda::Linear
{
    // Loads 4 BF16 elements as two __nv_bfloat162 pairs via an int2 (8-byte) load.
    // Requires ptr to be 8-byte aligned, guaranteed when C % 4 == 0.
    __device__ inline void ld_bf16x4(
        __nv_bfloat162& lo,
        __nv_bfloat162& hi,
        const __nv_bfloat16* ptr )
    {
        int2 raw = *reinterpret_cast<const int2*>(ptr);
        lo = *reinterpret_cast<const __nv_bfloat162*>(&raw.x);
        hi = *reinterpret_cast<const __nv_bfloat162*>(&raw.y);
    }

    /**
     * @brief Optimized CUDA kernel for BF16 matrix-vector multiply (M=1 decode path).
     *
     * Computes y[oc] = sum(x[c] * weight[oc, c]) + bias[oc] for each oc independently.
     * Dot-product accumulation is performed in float32 to preserve precision.
     *
     * Grid:  (ceil(OC / kMatvecBlockOC), 1)
     * Block: (kMatvecThreadsPerOC, kMatvecBlockOC) — 32 threads collaborate per output element
     *
     * OC need not be a multiple of kMatvecBlockOC; threads with oc >= OC exit early.
     *
     * Requirements:
     *   - C must be divisible by 4 (int2 loads require 8-byte alignment on the weight row)
     *   - kMatvecThreadsPerOC must be a power of 2 <= 32 (warp shuffle)
     */
    static constexpr int kMatvecThreadsPerOC = 32;
    static constexpr int kMatvecBlockOC = 8;

    __global__ void __launch_bounds__( kMatvecThreadsPerOC* kMatvecBlockOC )
        matvec_decode_bf16_kernel(
            __nv_bfloat16* __restrict__       y,
            const __nv_bfloat16* __restrict__ x,
            const __nv_bfloat16* __restrict__ weight,
            const __nv_bfloat16* __restrict__ bias,
            int C,
            int OC )
    {
        const int oc_base = blockIdx.x * kMatvecBlockOC;
        const int oc = oc_base + threadIdx.y;

        if ( oc >= OC ) return;

        const __nv_bfloat16* w_row = weight + oc * C;

        float acc = 0.0f;
        const int c_start = threadIdx.x * 4;
        const int c_step = kMatvecThreadsPerOC * 4;

        for ( int c = c_start; c < C; c += c_step )
        {
            __nv_bfloat162 x_lo, x_hi, w_lo, w_hi;
            ld_bf16x4( x_lo, x_hi, x + c );
            ld_bf16x4( w_lo, w_hi, w_row + c );

            float2 x_lo_f = __bfloat1622float2( x_lo );
            float2 w_lo_f = __bfloat1622float2( w_lo );
            float2 x_hi_f = __bfloat1622float2( x_hi );
            float2 w_hi_f = __bfloat1622float2( w_hi );

            acc += x_lo_f.x * w_lo_f.x + x_lo_f.y * w_lo_f.y
                + x_hi_f.x * w_hi_f.x + x_hi_f.y * w_hi_f.y;
        }

    #pragma unroll
        for ( int offset = kMatvecThreadsPerOC / 2; offset > 0; offset >>= 1 )
        {
            acc += __shfl_down_sync( 0xffffffff, acc, offset );
        }

        if ( threadIdx.x == 0 )
        {
            float bias_val = (bias != nullptr) ? __bfloat162float( bias[ oc ] ) : 0.0f;
            y[ oc ] = __float2bfloat16( acc + bias_val );
        }
    }

    void cuda_matvec_decode_bf16(
        __nv_bfloat16* y,
        const __nv_bfloat16* x,
        const __nv_bfloat16* weight,
        const __nv_bfloat16* bias,
        int C,
        int OC,
        cudaStream_t stream )
    {
        assert( C % 4 == 0 && "cuda_matvec_decode_bf16: C must be divisible by 4 for int2/bfloat162 loads" );

        const dim3 block( kMatvecThreadsPerOC, kMatvecBlockOC );
        const dim3 grid( (OC + kMatvecBlockOC - 1) / kMatvecBlockOC );

        matvec_decode_bf16_kernel << <grid, block, 0, stream >> > (y, x, weight, bias, C, OC);
    }
}