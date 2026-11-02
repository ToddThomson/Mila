#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace Mila::Dnn::Compute::Cuda::Residual
{
    /**
     * @brief The gated residual's elementwise steps over n streams of width H, overloaded on storage type.
     *
     *   activateDown:      a = silu( down / n )                              [rows, rank]
     *   mix:               x[h] = mean_i( sigmoid( up[i, h] ) * N[i, h] )    [rows, H]
     *   injectionWeights:  g[i] = 2 * sigmoid( inject[i] / n )               [rows, n]
     *   inject:            S'[i, h] = S[i, h] + g[i] * y[h]                  [rows, n * H]
     *
     * Arithmetic is in float at either storage type. Every extent is a launch argument that a decode step
     * keeps fixed, so a recorded step replays unchanged.
     */
    void cuda_gated_residual_activate_down( float* activated, const float* down, int count, int streams, cudaStream_t stream );
    void cuda_gated_residual_activate_down( __nv_bfloat16* activated, const __nv_bfloat16* down, int count, int streams, cudaStream_t stream );

    void cuda_gated_residual_mix( float* mixed, const float* normed, const float* up,
        int rows, int streams, int model_dim, cudaStream_t stream );
    void cuda_gated_residual_mix( __nv_bfloat16* mixed, const __nv_bfloat16* normed, const __nv_bfloat16* up,
        int rows, int streams, int model_dim, cudaStream_t stream );

    void cuda_gated_residual_injection_weights( float* weights, const float* inject, int count, int streams, cudaStream_t stream );
    void cuda_gated_residual_injection_weights( __nv_bfloat16* weights, const __nv_bfloat16* inject, int count, int streams, cudaStream_t stream );

    void cuda_gated_residual_inject( float* output, const float* residual, const float* weights, const float* sublayer_output,
        int rows, int streams, int model_dim, cudaStream_t stream );
    void cuda_gated_residual_inject( __nv_bfloat16* output, const __nv_bfloat16* residual, const __nv_bfloat16* weights,
        const __nv_bfloat16* sublayer_output, int rows, int streams, int model_dim, cudaStream_t stream );
}
