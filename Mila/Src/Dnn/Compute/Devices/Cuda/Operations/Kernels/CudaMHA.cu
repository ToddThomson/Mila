#include <cuda_fp16.h>
#include "device_launch_parameters.h"
#include "CudaUtils.h"

namespace Mila::Dnn::Compute
{
    __global__ void permute_fp32_kernel( float* q, float* k, float* v,
        const float* inp, int B, int N, int NH, int d ) {
        // okay so now, this kernel wants Q,K,V to all be of shape (B, NH, N, d)
        // but instead, we have a single tensor QKV (inp) of shape (B, N, 3, NH, d)
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        // Q[b][nh_][n][d_] = inp[b][n][0][nh_][d_]
        if ( idx < B * NH * N * d ) {
            int b = idx / (NH * N * d);
            int rest = idx % (NH * N * d);
            int nh_ = rest / (N * d);
            rest = rest % (N * d);
            int n = rest / d;
            int d_ = rest % d;
            int inp_idx = (b * N * 3 * NH * d) + (n * 3 * NH * d) + (0 * NH * d) + (nh_ * d) + d_;
            q[ idx ] = __ldcs( &inp[ inp_idx ] );
            k[ idx ] = __ldcs( &inp[ inp_idx + NH * d ] );
            v[ idx ] = __ldcs( &inp[ inp_idx + 2 * (NH * d) ] );
        }
    }

    void cuda_mha_forward_fp32(
        float* out,
        float* qkvr, float* att,
        const float* inp,
        int B, int T, int C, int NH,
        cudaStream_t stream ) {
        // Note: `inp` is not needed for backward pass, so we re-use it as a scratch buffer.
        // Its contents will be overwritten by this function.
        const int block_size = 256;
        const int softmax_block_size = 256;

        // inp is (B, T, 3C) QKV
        // preatt, att are (B, NH, T, T)
        // output is (B, T, C)
        int HS = C / NH; // head size

        // permute and separate inp from (B, T, 3, NH, HS) to 3X (B, NH, T, HS)
        float* q, * k, * v;
        q = qkvr + 0 * B * T * C;
        k = qkvr + 1 * B * T * C;
        v = qkvr + 2 * B * T * C;
        int total_threads = B * NH * T * HS;
        int num_blocks = ceil_div( total_threads, block_size );

        permute_fp32_kernel <<<num_blocks, block_size>>> (q, k, v, inp, B, T, NH, HS);

        cudaCheck( cudaGetLastError() );

        // batched matrix multiply with cuBLAS
        const float alpha = 1.0f;
        const float beta = 0.0f;
        float* preatt = inp;
        cublasCheck( cublasSgemmStridedBatched( cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, T, T, HS, &alpha, k, HS, T * HS, q, HS, T * HS, &beta, preatt, T, T * T, B * NH ) );

        // multiply all elements of preatt elementwise by scale
        float scale = 1.0 / sqrtf( HS );
        int grid_size = CEIL_DIV( B * NH * T * 32, softmax_block_size );
        softmax_forward_kernel5 << <grid_size, softmax_block_size >> > (att, scale, preatt, B * NH, T);
        cudaCheck( cudaGetLastError() );

        // new approach: first cuBLAS another batched matmul
        float* vaccum = inp;
        // y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        cublasCheck( cublasSgemmStridedBatched( cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, HS, T, T, &alpha, v, HS, T * HS, att, T, T * T, &beta, vaccum, HS, T * HS, B * NH ) );

        // now unpermute
        // y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        num_blocks = CEIL_DIV( B * T * C, block_size );
        unpermute_fp_32_kernel<<<num_blocks, block_size >>>(vaccum, out, B, T, NH, HS);

        cudaCheck( cudaGetLastError() );
    }
}