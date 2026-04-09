// CudaGqa.Prefill.Fp16.cu

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include "CudaUtils.h"
#include "CudaGqa.cuh"

namespace Mila::Dnn::Compute::Cuda::GroupedQueryAttention
{
    __global__ void gqa_prefill_permute_q_bf16_kernel(
        nv_bfloat16* Q,
        const nv_bfloat16* X,
        int B,
        int chunk_len,
        int start_pos,
        int T_max,
        int NH,
        int NKV,
        int HS )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        // operate on half2 pairs (2 elements)
        int HS2 = HS >> 1;
        const int total2 = B * NH * chunk_len * HS2;
        if ( idx >= total2 ) return;

        int b = idx / (NH * chunk_len * HS2);
        int r1 = idx % (NH * chunk_len * HS2);

        int nh = r1 / (chunk_len * HS2);
        r1 %= (chunk_len * HS2);

        int t = r1 / HS2;
        int h2 = r1 % HS2;

        int kv_pos = start_pos + t;

        int out2 =
            b * (NH * T_max * HS)
            + nh * (T_max * HS)
            + kv_pos * HS
            + (h2 << 1);

        int src2 =
            b * chunk_len * (NH + 2 * NKV) * HS +
            t * (NH + 2 * NKV) * HS +
            nh * HS + (h2 << 1);

        reinterpret_cast<half2*>(Q)[ out2 >> 1 ] =
            reinterpret_cast<const half2*>(X)[ src2 >> 1 ];
    }

    __global__ void gqa_prefill_permute_kv_bf16_kernel(
        nv_bfloat16* K, nv_bfloat16* V,
        const nv_bfloat16* X,
        int B,
        int chunk_len,
        int start_pos,
        int T_max,
        int NH,
        int NKV,
        int HS )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        int HS2 = HS >> 1;
        const int total2 = B * NKV * chunk_len * HS2;
        if ( idx >= total2 ) return;

        int b = idx / (NKV * chunk_len * HS2);
        int r1 = idx % (NKV * chunk_len * HS2);

        int nkv = r1 / (chunk_len * HS2);
        r1 %= (chunk_len * HS2);

        int t = r1 / HS2;
        int h2 = r1 % HS2;

        int kv_pos = start_pos + t;

        int out2 =
            b * (NKV * T_max * HS)
            + nkv * (T_max * HS)
            + kv_pos * HS
            + (h2 << 1);

        int base =
            b * chunk_len * (NH + 2 * NKV) * HS +
            t * (NH + 2 * NKV) * HS;

        int k_src = base + NH * HS + (nkv * HS) + (h2 << 1);
        int v_src = base + (NH + NKV) * HS + (nkv * HS) + (h2 << 1);

        auto K2 = reinterpret_cast<half2*>(K);
        auto V2 = reinterpret_cast<half2*>(V);

        const auto* X2 = reinterpret_cast<const half2*>(X);

        K2[ out2 >> 1 ] = X2[ k_src >> 1 ];
        V2[ out2 >> 1 ] = X2[ v_src >> 1 ];
    }

    // ------------------------------------------------------------------------
    // Host function to launch the kernels
    // ------------------------------------------------------------------------

    void cuda_gqa_prefill_permute_qkv_bf16(
        nv_bfloat16* Q, nv_bfloat16* K, nv_bfloat16* V,
        const nv_bfloat16* X,
        int B, int chunk_len, int start_pos, int T_max,
        int NH, int NKV, int HS,
        cudaStream_t stream )
    {
        int HS2 = HS >> 1;
        dim3 block( 256 );

        gqa_prefill_permute_q_bf16_kernel
            << < ceil_div( B * NH * chunk_len * HS2, 256 ), block, 0, stream >> > (
                Q, X, B, chunk_len, start_pos, T_max, NH, NKV, HS);

        gqa_prefill_permute_kv_bf16_kernel
            << < ceil_div( B * NKV * chunk_len * HS2, 256 ), block, 0, stream >> > (
                K, V, X, B, chunk_len, start_pos, T_max, NH, NKV, HS);
    }
}