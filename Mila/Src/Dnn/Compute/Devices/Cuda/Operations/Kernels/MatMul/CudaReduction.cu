#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "../../../Helpers/CudaUtils.h"

namespace Mila::Dnn::Compute
{
    // this kernel performs a column-wise reduction over dout, in PyTorch equivalent to:
    // dbias = dout.sum((0,1))
    // the idea is to employ one block to reduce along several columns,
    // where each block has a width of 32 columns to ensure coalesced access.
    // at the end we accumulate the reductions performed by the warps in each block via shared memory
    __global__ void matmul_backward_bias_kernel4( float* dbias, const float* dout, int outer_size, int OC )
    {
        // this kernel is launched with 1D grid_dim of OC/32
        // for example let's say block_size is 128
        extern __shared__ float smem[]; // of size block_size (128)
        const int warp_id = threadIdx.x / warpSize; // warp index in the block, 0,1,2,3
        const int lane_id = threadIdx.x % warpSize; // thread index in the warp, 0,1,2,...,31
        const int tl = blockIdx.x * warpSize; // pointer to the start column for this block
        const int vstep = blockDim.x / warpSize; // number of warps in a block, e.g. 4

        // pointer to the start of the column for one lane of threads
        // so e.g. 4 threads (of the same lane_id) will reduce this one column
        const float* dout_col = dout + tl + lane_id;

        // column reductions by looping through the rows
        // each of the 4 threads offsets by its warp_id and then skips by vstep
        // together these 4 threads cover all B*T rows of this (lane_id) column
        // importantly, consecutive threads (in threadId) are processing adjacent columns,
        // leading to a coalesced memory access pattern
        float dout_sum = 0.0f;
        for (int row = warp_id; row < outer_size; row += vstep)
        {
            dout_sum += dout_col[row * OC];
        }
        smem[lane_id + warp_id * warpSize] = dout_sum;
        __syncthreads();

        // warp_id 0 reduces the shared memory column-wise, linearly
        dout_sum = 0.0f;
        if (warp_id == 0)
        {
            for (int j = 0; j < vstep; j++)
            {
                dout_sum += smem[lane_id + j * warpSize];
            }
            
            dbias[tl + lane_id] += dout_sum;
        }
    }

    void cuda_reduce_sum_batch_fp32(
        float* bias_grad,
        const float* output_grad,
        int batch_size,
        int out_features,
        cudaStream_t stream )
    {
        const int block_size = 1024;
        const int grid_size = out_features / 32; // for now, OC must be divisible by 32 for this kernel to work

        matmul_backward_bias_kernel4 <<<grid_size, block_size, block_size * sizeof( float ) >>> ( bias_grad, output_grad, batch_size, out_features );

        cudaCheck( cudaGetLastError() );
    }
}