#define _USE_MATH_DEFINES
#include <math.h>
#include <cassert>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "device_launch_parameters.h"
#include "CudaUtils.h"

namespace Mila::Dnn::Compute::Cuda::Rope
{
    // ------------------------------------------------------------------------
    // RoPE forward/backward kernels (scalar pair-based implementation)
    //
    // Implementation notes:
    // - Rotary embedding is applied to the first `rotary_dim` channels (must be even).
    // - Channels are processed as 2-element rotation blocks:
    //     for pair p: c0 = 2*p, c1 = c0+1
    // - angle = position * inv_freq[p]; inv_freq[p] = base^{-2p/rotary_dim}
    // - Forward rotation:
    //     y0 = cos* x0 - sin* x1
    //     y1 = sin* x0 + cos* x1
    // - Backward / gradients:
    //     dx0 = cos* dy0 + sin* dy1
    //     dx1 = -sin* dy0 + cos* dy1
    //
    // This implementation favors clarity and correctness over maximal throughput.
    // Consider vectorized / precomputed-cos/sin kernels for high-performance paths.
    // ------------------------------------------------------------------------

    __global__ void rope_forward_fp32_kernel(
        float* __restrict__ out,           // [B*T*C]
        const int* __restrict__ inp,       // [B*T]
        const float* __restrict__ wte,     // [vocab_size * C]
        int B, int T, int C,
        int rotary_dim, float base )
    {
        const int bt = blockIdx.x;
        const int tid = threadIdx.x;
        const int block_threads = blockDim.x;

        if ( bt >= B * T ) return;

        const int t = bt % T;
        const int ix = inp[ bt ];

        const int pairs = rotary_dim / 2;

        // Process pairs (rotary region)
        for ( int p = tid; p < pairs; p += block_threads )
        {
            const int c0 = 2 * p;
            const int c1 = c0 + 1;

            const float x0 = wte[ static_cast<size_t>( ix ) * C + c0 ];
            const float x1 = wte[ static_cast<size_t>( ix ) * C + c1 ];

            // Compute frequency: inv_freq = base^(-2p / rotary_dim)
            const float exponent = (2.0f * static_cast<float>( p )) / static_cast<float>( rotary_dim );
            const float inv_freq = powf( base, -exponent );

            // Compute angle and apply rotation
            const float angle = static_cast<float>(t) * inv_freq;
            const float s = sinf( angle );
            const float c = cosf( angle );

            // Rotation matrix: [cos -sin; sin cos]
            const float y0 = c * x0 - s * x1;
            const float y1 = s * x0 + c * x1;

            out[ static_cast<size_t>(bt) * C + c0 ] = y0;
            out[ static_cast<size_t>(bt) * C + c1 ] = y1;
        }

        // Copy remaining channels (non-rotary region)
        for ( int c = rotary_dim + tid; c < C; c += block_threads )
        {
            out[ static_cast<size_t>( bt ) * C + c ] = wte[ static_cast<size_t>( ix ) * C + c ];
        }
    }

    __global__ void rope_backward_fp32_kernel(
        float* __restrict__ wte_grad,      // [vocab_size * C] (accumulated)
        const float* __restrict__ dY,      // [B*T*C]
        const int* __restrict__ inp,       // [B*T]
        int B, int T, int C,
        int rotary_dim, float base )
    {
        const int bt = blockIdx.x;
        const int tid = threadIdx.x;
        const int block_threads = blockDim.x;

        if ( bt >= B * T ) return;

        const int t = bt % T;
        const int ix = inp[ bt ];

        const int pairs = rotary_dim / 2;

        // Process pairs (rotary region)
        for ( int p = tid; p < pairs; p += block_threads )
        {
            const int c0 = 2 * p;
            const int c1 = c0 + 1;

            const float dy0 = dY[ static_cast<size_t>( bt ) * C + c0 ];
            const float dy1 = dY[ static_cast<size_t>( bt ) * C + c1 ];

            // Compute frequency: inv_freq = base^(-2p / rotary_dim)
            const float exponent = (2.0f * static_cast<float>( p )) / static_cast<float>( rotary_dim );
            const float inv_freq = powf( base, -exponent );

            // Compute angle and apply rotation transpose
            const float angle = static_cast<float>(t) * inv_freq;
            const float s = sinf( angle );
            const float c = cosf( angle );

            // Rotation matrix transpose: [cos sin; -sin cos]
            const float dx0 = c * dy0 + s * dy1;
            const float dx1 = -s * dy0 + c * dy1;

            // Atomic accumulate into wte_grad
            atomicAdd( &wte_grad[ static_cast<size_t>(ix) * C + c0 ], dx0 );
            atomicAdd( &wte_grad[ static_cast<size_t>(ix) * C + c1 ], dx1 );
        }

        // Remaining channels: gradient is direct copy of dY
        for ( int c = rotary_dim + tid; c < C; c += block_threads )
        {
            const float dy = dY[ static_cast<size_t>( bt ) * C + c ];
            atomicAdd( &wte_grad[ static_cast<size_t>( ix ) * C + c ], dy );
        }
    }

    // ------------------------------------------------------------------------
    // Host launchers
    // ------------------------------------------------------------------------

    void cuda_rope_forward_fp32(
        float* Y,
        const int* X,
        const float* wte,
        int B, int T, int C,
        cudaStream_t stream,
        int rotary_dim, float base )
    {
        // Validate parameters
        assert( rotary_dim >= 0 && rotary_dim <= C );
        assert( rotary_dim % 2 == 0 );

        const int bt_count = B * T;
        const int threads = 128;
        const int blocks = bt_count;

        rope_forward_fp32_kernel << < blocks, threads, 0, stream >> > (
            Y, X, wte, B, T, C, rotary_dim, base);

        cudaCheck( cudaGetLastError() );
    }

    void cuda_rope_backward_fp32(
        float* wte_grad,
        const float* dY,
        const int* X,
        int B, int T, int C,
        cudaStream_t stream,
        int rotary_dim, float base )
    {
        // Validate parameters
        assert( rotary_dim >= 0 && rotary_dim <= C );
        assert( rotary_dim % 2 == 0 );

        const int bt_count = B * T;
        const int threads = 128;
        const int blocks = bt_count;

        // Note: wte_grad must be zeroed by caller before this call
        rope_backward_fp32_kernel << < blocks, threads, 0, stream >> > (
            wte_grad, dY, X, B, T, C, rotary_dim, base);

        cudaCheck( cudaGetLastError() );
    }
}