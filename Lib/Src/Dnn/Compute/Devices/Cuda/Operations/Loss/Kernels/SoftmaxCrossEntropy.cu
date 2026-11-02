#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "CudaUtils.h"
#include "SoftmaxCrossEntropy.cuh"

namespace Mila::Dnn::Compute::Cuda::SoftmaxCrossEntropy
{
    void cuda_softmax_crossentropy_forward_fp32(
        float* losses,
        const float* logits,
        const int* targets,
        int batch_size,
        int seq_len,
        int vocab_size,
        cudaStream_t stream )
    {
        int num_positions = batch_size * seq_len;

        if ( vocab_size < WARP_VOCAB_THRESHOLD ) {
            // Warp-level approach: multiple warps per block
            int block_size = 256;  // 8 warps per block
            int num_warps = num_positions;
            int grid_size = (num_warps * WARP_SIZE + block_size - 1) / block_size;

            softmax_crossentropy_forward_fp32_warp_kernel << <grid_size, block_size, 0, stream >> > (
                losses, logits, targets, batch_size, seq_len, vocab_size);
        }
        else {
            // Block-level approach: one block per position
            int block_size = 256;  // Tune based on vocab_size

            softmax_crossentropy_forward_fp32_block_kernel << <num_positions, block_size, 0, stream >> > (
                losses, logits, targets, batch_size, seq_len, vocab_size);
        }

        cudaCheck( cudaGetLastError() );
    }

    void cuda_softmax_crossentropy_backward_fp32(
        float* dlogits,
        const float* dlosses,
        const float* logits,
        const int* targets,
        int batch_size,
        int seq_len,
        int vocab_size,
        cudaStream_t stream )
    {
        int num_positions = batch_size * seq_len;

        if ( vocab_size < WARP_VOCAB_THRESHOLD ) {
            int block_size = 256;
            int num_warps = num_positions;
            int grid_size = (num_warps * WARP_SIZE + block_size - 1) / block_size;

            softmax_crossentropy_backward_fp32_warp_kernel << <grid_size, block_size, 0, stream >> > (
                dlogits, dlosses, logits, targets, batch_size, seq_len, vocab_size);
        }
        else {
            int block_size = 256;

            softmax_crossentropy_backward_fp32_block_kernel << <num_positions, block_size, 0, stream >> > (
                dlogits, dlosses, logits, targets, batch_size, seq_len, vocab_size);
        }

        cudaCheck( cudaGetLastError() );
    }
}