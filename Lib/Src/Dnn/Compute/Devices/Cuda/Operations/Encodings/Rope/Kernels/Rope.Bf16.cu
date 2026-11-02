#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "CudaUtils.h"
#include "Rope.cuh"
#include "Rope.Rotation.cuh"

namespace Mila::Dnn::Compute::Cuda::Rope
{
    // Q and K load and store as BF16; every angle and rotation is computed in FP32.

    void cuda_rope_forward_bf16(
        __nv_bfloat16* Q_out,
        __nv_bfloat16* K_out,
        const __nv_bfloat16* Q_in,
        const __nv_bfloat16* K_in,
        const RopeAngleParameters& angles,
        int B, int T,
        int n_heads, int n_kv_heads, int head_dim,
        int rotary_dim, int rotary_layout,
        int position_offset,
        cudaStream_t stream )
    {
        launch_rope_rotation<false>( Q_out, K_out, Q_in, K_in, angles,
            B * T, T, n_heads, n_kv_heads, head_dim, rotary_dim, rotary_layout, position_offset, nullptr, stream );
    }

    void cuda_rope_backward_bf16(
        __nv_bfloat16* dQ_in,
        __nv_bfloat16* dK_in,
        const __nv_bfloat16* dQ_out,
        const __nv_bfloat16* dK_out,
        const RopeAngleParameters& angles,
        int B, int T,
        int n_heads, int n_kv_heads, int head_dim,
        int rotary_dim, int rotary_layout,
        cudaStream_t stream )
    {
        launch_rope_rotation<true>( dQ_in, dK_in, dQ_out, dK_out, angles,
            B * T, T, n_heads, n_kv_heads, head_dim, rotary_dim, rotary_layout, 0, nullptr, stream );
    }

    void cuda_rope_decode_bf16(
        __nv_bfloat16* Q_out,
        __nv_bfloat16* K_out,
        const __nv_bfloat16* Q_in,
        const __nv_bfloat16* K_in,
        const RopeAngleParameters& angles,
        int B, int T, const int* position,
        int n_heads, int n_kv_heads, int head_dim,
        int rotary_dim, int rotary_layout,
        cudaStream_t stream )
    {
        launch_rope_rotation<false>( Q_out, K_out, Q_in, K_in, angles,
            B * T, T, n_heads, n_kv_heads, head_dim, rotary_dim, rotary_layout, 0, position, stream );
    }
}
