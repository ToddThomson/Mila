/**
 * @file CudaTokenEmbeddingOp.Quantize.ixx
 * @brief Quantize partition of CudaTokenEmbeddingOp (D4 Design B).
 *
 * Exports Detail::quantize_table_fp8_per_row() and quantize_table_int6_per_group()
 * as non-template functions so
 * that CudaTokenEmbeddingOp::quantize() (a class-template member instantiated
 * by cl.exe) never needs the NVCC-compiled kernel body -- the same module
 * boundary crossing pattern as CudaLinearOp:Quantize.
 *
 * Per-vocab-row table quantization IS per-channel quantization with
 * out_features = vocab_size and in_features = embedding_dim, so this
 * partition delegates to the Linear FP8 quantization kernel rather than
 * duplicating it: scale[v] = max(|W[v,:]|) / 448.0f.
 */

module;
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <format>
#include "../Linear/Kernels/Quantization/CudaFp8WeightQuantization.cuh"
#include "../Linear/Kernels/Quantization/CudaInt6WeightQuantization.cuh"

export module Compute.CudaTokenEmbeddingOp:Quantize;

import Dnn.ITensor;
import Dnn.TensorTypes;
import Serialization.Tensor;

namespace Mila::Dnn::Compute::Cuda::TokenEmbedding
{
    namespace Detail
    {
        /**
         * @brief Validate, quantize and upload a BF16 embedding table blob to
         *        FP8_E4M3 with one float32 absmax scale per vocabulary row.
         *
         * The table is processed in row chunks sized to staging_bytes rather than
         * staged whole: the full BF16 table (262144 x 3840 x 2B ~= 2 GB on the 12B
         * build) would permanently grow the grow-only shared scratch buffer past
         * the ~470 MB the prefill dequant path already forces, inflating
         * steady-state VRAM by more than the FP8 table saves. Row scales are
         * row-local, so chunking on rows is exact -- the same kernel runs per
         * chunk with offset pointers. Stream ordering serializes each chunk's
         * upload against the previous chunk's kernel, so one staging buffer is
         * safely reused.
         *
         * All device work is issued on the caller's stream; the caller must
         * synchronize before the blob memory is released (the H2D uploads are
         * asynchronous).
         *
         * @param blob           Host BF16 table blob [vocab_size, embedding_dim].
         * @param table_out      Device FP8_E4M3 tensor [vocab_size, embedding_dim].
         * @param scales_out     Device float32 tensor [vocab_size].
         * @param expected_shape Expected table shape for validation.
         * @param dev_staging    Device staging buffer of at least staging_bytes.
         * @param staging_bytes  Usable staging size; must hold at least one BF16 row.
         * @param stream         CUDA stream for all async operations.
         *
         * @throws std::invalid_argument if the blob shape does not match expected_shape
         *         or staging_bytes cannot hold one table row.
         * @throws std::runtime_error    if a CUDA call fails.
         */
        export void quantize_table_fp8_per_row(
            const Mila::Dnn::Serialization::ITensorBlob& blob,
            Mila::Dnn::ITensor&                          table_out,
            Mila::Dnn::ITensor&                          scales_out,
            const Mila::Dnn::shape_t&                    expected_shape,
            void*                                        dev_staging,
            size_t                                       staging_bytes,
            cudaStream_t                                 stream )
        {
            const auto& meta = blob.getMetadata();

            if ( meta.shape != expected_shape )
            {
                throw std::invalid_argument( std::format(
                    "quantize_table_fp8_per_row - shape mismatch: expected [{},{}], got [{},{}]",
                    expected_shape[ 0 ], expected_shape[ 1 ],
                    meta.shape[ 0 ],     meta.shape[ 1 ] ) );
            }

            const int64_t vocab_size = static_cast<int64_t>( expected_shape[ 0 ] );
            const int64_t embedding_dim = static_cast<int64_t>( expected_shape[ 1 ] );

            const int64_t source_row_bytes = embedding_dim * static_cast<int64_t>( sizeof( uint16_t ) );
            const int64_t rows_per_chunk = static_cast<int64_t>( staging_bytes ) / source_row_bytes;

            if ( rows_per_chunk < 1 )
            {
                throw std::invalid_argument( std::format(
                    "quantize_table_fp8_per_row - staging ({} bytes) cannot hold one {}-byte table row",
                    staging_bytes, source_row_bytes ) );
            }

            const auto* source = static_cast<const char*>( blob.data() );
            auto* table = static_cast<char*>( table_out.rawData() );
            auto* scales = static_cast<float*>( scales_out.rawData() );

            for ( int64_t row = 0; row < vocab_size; row += rows_per_chunk )
            {
                const int64_t chunk_rows = std::min( rows_per_chunk, vocab_size - row );

                Mila::Dnn::Compute::Cuda::Linear::cuda_quantize_fp8_per_channel(
                    source + row * source_row_bytes,
                    table + row * embedding_dim,
                    scales + row,
                    chunk_rows,
                    embedding_dim,
                    dev_staging,
                    staging_bytes,
                    stream );
            }
        }

        /**
         * @brief Validate, quantize and upload a BF16 embedding table blob to six-bit codes with one IEEE half
         *        scale per 32 elements of a row, by the rule in Int6Packing.ixx.
         *
         * A table row is a weight row with out_features = vocab_size, so this delegates to the Linear INT6
         * quantizer, which already walks the rows in blocks that fit the staging buffer.
         *
         * @param blob           Host BF16 table blob [vocab_size, embedding_dim].
         * @param table_out      Device UINT8 tensor [vocab_size, 3 * embedding_dim / 4].
         * @param scales_out     Device FP16 tensor [vocab_size, embedding_dim / 32].
         * @param expected_shape Logical table shape [vocab_size, embedding_dim], for validation.
         * @param group_size     Elements per scale (32).
         * @param dev_staging    Device staging buffer of at least staging_bytes.
         * @param staging_bytes  Usable staging size; must hold at least one BF16 row.
         * @param stream         CUDA stream for all async operations.
         *
         * @throws std::invalid_argument if the blob shape does not match expected_shape.
         * @throws std::runtime_error    if staging cannot hold one row or a CUDA call fails.
         */
        export void quantize_table_int6_per_group(
            const Mila::Dnn::Serialization::ITensorBlob& blob,
            Mila::Dnn::ITensor&                          table_out,
            Mila::Dnn::ITensor&                          scales_out,
            const Mila::Dnn::shape_t&                    expected_shape,
            int                                          group_size,
            void*                                        dev_staging,
            size_t                                       staging_bytes,
            cudaStream_t                                 stream )
        {
            const auto& meta = blob.getMetadata();

            if ( meta.shape != expected_shape )
            {
                throw std::invalid_argument( std::format(
                    "quantize_table_int6_per_group - shape mismatch: expected [{},{}], got [{},{}]",
                    expected_shape[ 0 ], expected_shape[ 1 ],
                    meta.shape[ 0 ], meta.shape[ 1 ] ) );
            }

            Mila::Dnn::Compute::Cuda::Linear::cuda_quantize_int6_per_group(
                blob.data(),
                table_out.rawData(),
                scales_out.rawData(),
                static_cast<int64_t>( expected_shape[ 0 ] ),
                static_cast<int64_t>( expected_shape[ 1 ] ),
                group_size,
                dev_staging,
                staging_bytes,
                stream );
        }

    } // namespace Detail

} // namespace Mila::Dnn::Compute::Cuda::TokenEmbedding
