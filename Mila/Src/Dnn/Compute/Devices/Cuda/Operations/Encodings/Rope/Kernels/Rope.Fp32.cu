#include <cuda_runtime.h>
#include "CudaUtils.h"
#include "Rope.cuh"
#include "Rope.Rotation.cuh"

namespace Mila::Dnn::Compute::Cuda::Rope
{
    void cuda_rope_forward_fp32(
        float* Q_out,
        float* K_out,
        const float* Q_in,
        const float* K_in,
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

    void cuda_rope_backward_fp32(
        float* dQ_in,
        float* dK_in,
        const float* dQ_out,
        const float* dK_out,
        const RopeAngleParameters& angles,
        int B, int T,
        int n_heads, int n_kv_heads, int head_dim,
        int rotary_dim, int rotary_layout,
        cudaStream_t stream )
    {
        launch_rope_rotation<true>( dQ_in, dK_in, dQ_out, dK_out, angles,
            B * T, T, n_heads, n_kv_heads, head_dim, rotary_dim, rotary_layout, 0, nullptr, stream );
    }

    void cuda_rope_decode_fp32(
        float* Q_out,
        float* K_out,
        const float* Q_in,
        const float* K_in,
        const RopeAngleParameters& angles,
        int B, const int* position,
        int n_heads, int n_kv_heads, int head_dim,
        int rotary_dim, int rotary_layout,
        cudaStream_t stream )
    {
        launch_rope_rotation<false>( Q_out, K_out, Q_in, K_in, angles,
            B, 1, n_heads, n_kv_heads, head_dim, rotary_dim, rotary_layout, 0, position, stream );
    }
}
