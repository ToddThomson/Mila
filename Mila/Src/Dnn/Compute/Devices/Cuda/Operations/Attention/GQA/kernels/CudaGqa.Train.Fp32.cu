// CudaGqa.Train.cu

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include "CudaUtils.h"
#include "CudaGqa.cuh"

namespace Mila::Dnn::Compute::Cuda::GroupedQueryAttention
{
    /**
     * @brief Broadcast each KV head to its group of Q heads (FP32).
     *
     * For Q head nh: kv_head = nh / GS  where GS = NH / NKV.
     * Reads k_compact/v_compact[b, kv_head, t, hs] and writes
     * k_exp/v_exp[b, nh, t, hs].
     *
     * Total threads: B * NH * T * HS (indexed over the expanded layout).
     */
    __global__ void expand_kv_fp32_kernel(
        float* k_exp, float* v_exp,
        const float* k_compact, const float* v_compact,
        int B, int T, int NH, int NKV, int HS )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if ( idx < B * NH * T * HS )
        {
            const int b = idx / (NH * T * HS);
            int rest = idx % (NH * T * HS);
            const int nh = rest / (T * HS);
            rest = rest % (T * HS);
            const int t = rest / HS;
            const int hs = rest % HS;

            // Map Q head → KV head (integer division, GS = NH/NKV).
            const int nkv = nh / (NH / NKV);

            const int src_idx = b * (NKV * T * HS) + nkv * (T * HS) + t * HS + hs;

            k_exp[ idx ] = k_compact[ src_idx ];
            v_exp[ idx ] = v_compact[ src_idx ];
        }
    }

    // ------------------------------------------------------------------------
    // Host Launchers
    // ------------------------------------------------------------------------

    void cuda_gqa_expand_kv_fp32(
        float* k_exp, float* v_exp,
        const float* k_compact, const float* v_compact,
        int B, int T, int NH, int NKV, int HS,
        cudaStream_t stream )
    {
        const int block_size = 256;
        const int num_blocks = ceil_div( B * NH * T * HS, block_size );

        expand_kv_fp32_kernel <<<num_blocks, block_size, 0, stream >>> (
            k_exp, v_exp, k_compact, v_compact, B, T, NH, NKV, HS);
        
        cudaCheck( cudaGetLastError() );
    }
}
