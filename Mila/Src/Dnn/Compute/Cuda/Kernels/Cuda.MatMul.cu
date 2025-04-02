#include <cuda_runtime.h>
#include "Cuda.Utils.h"

__device__ float4 ld_vec( const float* address ) {
    return *reinterpret_cast<const float4*>(address);
}

__device__ void st_vec( float* address, float4 val ) {
    *reinterpret_cast<float4*>(address) = val;
}

__global__ void __launch_bounds__( 16 * 16 ) matmul_forward_kernel4(
    float* out,
    const float* inp, const float* weight, const float* bias,
    int C, int OC ) {
    // Each thread handles 8x8 elements; each block 128 by 128 elements.
    int oc = 8 * (blockIdx.y * blockDim.y + threadIdx.y);

    // buffers to cache chunks of the input matrices
    __shared__ float lhs_s[ 128 ][ 32 ];
    __shared__ float rhs_s[ 128 ][ 32 ];

    // adjust our pointers for the current block
    inp += 128 * blockIdx.x * C;
    weight += 128 * blockIdx.y * C;
    out += 128 * blockIdx.x * OC + 128 * blockIdx.y;

    float vals[ 8 ][ 8 ] = {};
    if ( bias != NULL ) {
        for ( int i = 0; i < 8; i++ ) {
            for ( int j = 0; j < 8; j += 4 ) {
                float4 b = ld_vec( bias + oc + j );
                vals[ i ][ j + 0 ] = b.x;
                vals[ i ][ j + 1 ] = b.y;
                vals[ i ][ j + 2 ] = b.z;
                vals[ i ][ j + 3 ] = b.w;
            }
        }
    }

    int si_start = 4 * (16 * threadIdx.y + threadIdx.x);
    for ( int so = 0; so < C; so += 32 ) {
        __syncthreads();
        int xmod8 = threadIdx.x % 8;
        int xby8 = threadIdx.x / 8;
        int xo = 4 * xmod8;
        for ( int y = 2 * threadIdx.y + xby8; y < 128; y += 32 ) {
            st_vec( &lhs_s[ y ][ xo ], ld_vec( inp + y * C + so + xo ) );
            st_vec( &rhs_s[ y ][ xo ], ld_vec( weight + y * C + so + xo ) );
        }
        __syncthreads();

        for ( int si = si_start; si < si_start + 32; si += 4 ) {
            float4 rhs[ 8 ];
            for ( int u = 0; u < 8; ++u ) {
                rhs[ u ] = ld_vec( &rhs_s[ u + 8 * threadIdx.y ][ si % 32 ] );
            }

            for ( int ii = 0; ii < 8; ++ii ) {
                float4 lhs = ld_vec( &lhs_s[ ii + 8 * threadIdx.x ][ si % 32 ] );
                for ( int ji = 0; ji < 8; ++ji ) {
                    vals[ ii ][ ji ] += lhs.x * rhs[ ji ].x;
                    vals[ ii ][ ji ] += lhs.y * rhs[ ji ].y;
                    vals[ ii ][ ji ] += lhs.z * rhs[ ji ].z;
                    vals[ ii ][ ji ] += lhs.w * rhs[ ji ].w;
                }
            }
        }
    }

    for ( int i = 0; i < 8; ++i ) {
        for ( int j = 0; j < 8; j += 4 ) {
            float4 result;
            result.x = vals[ i ][ j + 0 ];
            result.y = vals[ i ][ j + 1 ];
            result.z = vals[ i ][ j + 2 ];
            result.w = vals[ i ][ j + 3 ];
            st_vec( out + (8 * threadIdx.x + i) * OC + 8 * threadIdx.y + j, result );
        }
    }
};

//class Cuda_MatMul_kernel {
//public:
    /**
     * @brief Performs the forward pass of the matrix multiplication operation.
     * 
     * @param out Pointer to the output matrix (B, TElementType, OC).
     * @param inp Pointer to the input matrix (B, TElementType, C).
     * @param weight Pointer to the weight matrix (OC, C).
     * @param bias Pointer to the bias vector (OC).
     * @param B Batch size.
     * @param TElementType Sequence length.
     * @param C Number of input channels.
     * @param OC Number of output channels (N * C.
     */
    void cuda_matmul_forward( float* out,
        const float* inp, const float* weight, const float* bias,
        int B, int T, int C, int OC ) {
        
        int sqrt_block_size = 16;

        dim3 gridDim( ceil_div( B * T, 8 * sqrt_block_size ), ceil_div( OC, 8 * sqrt_block_size ) );
        dim3 blockDim( sqrt_block_size, sqrt_block_size );
        
        matmul_forward_kernel4 << <gridDim, blockDim >> > (out, inp, weight, bias, C, OC);
        
        cudaCheck( cudaGetLastError() );
    }