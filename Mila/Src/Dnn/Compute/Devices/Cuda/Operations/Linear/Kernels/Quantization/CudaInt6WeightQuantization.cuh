/**
 * @file CudaInt6WeightQuantization.cuh
 * @brief Host-side per-group BF16->INT6 weight quantization for CudaLinearOp and CudaTokenEmbeddingOp.
 *
 * Produces exactly the codes and scales of the CPU reference codec in Int6Packing.ixx.
 */

#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace Mila::Dnn::Compute::Cuda::Linear
{
    /**
     * @brief Per-group BF16->INT6 quantization with async device upload.
     *
     * All device operations are issued on stream; nothing synchronizes.
     *
     * @param src_bf16      Host pointer to BF16 weights [out_features * in_features], pinned for async DMA.
     * @param dst_packed    Device uint8 codes [out_features * 3 * in_features / 4], layout of Int6Packing.ixx.
     * @param dst_scales    Device IEEE half scales [out_features * in_features / group_size].
     * @param out_features  Rows.
     * @param in_features   Columns; a multiple of group_size.
     * @param group_size    Must be 32.
     * @param dev_staging   Device buffer of at least one BF16 row; rows are quantized in blocks that fit.
     * @param staging_bytes Capacity of dev_staging.
     * @param stream        CUDA stream.
     *
     * @throws std::runtime_error on an unsupported group size, a staging buffer smaller than one row, or a
     *         failed CUDA call.
     */
    void cuda_quantize_int6_per_group(
        const void*  src_bf16,
        void*        dst_packed,
        void*        dst_scales,
        int64_t      out_features,
        int64_t      in_features,
        int          group_size,
        void*        dev_staging,
        size_t       staging_bytes,
        cudaStream_t stream );

} // namespace Mila::Dnn::Compute::Cuda::Linear
