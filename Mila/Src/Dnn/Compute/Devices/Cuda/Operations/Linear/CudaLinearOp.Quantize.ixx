/**
 * @file CudaLinearOp.Quantize.ixx
 * @brief Quantize partition of CudaLinearOp.
 *
 * Exports Detail::quantize_fp8_per_channel() as a non-template function compiled
 * by NVCC. This is the critical module-boundary crossing point for the FP8
 * quantize-on-load path:
 *
 *   Linear::loadParameter          (cl.exe, unchanged)
 *     → CudaLinearOp::quantize()   (class template body, cl.exe instantiation)
 *       → Detail::quantize_fp8_per_channel()  (non-template; pre-compiled by NVCC)
 *         → cuda_quantize_fp8_per_channel()   (plain .cu, NVCC)
 *
 * Because Detail::quantize_fp8_per_channel() is a non-template function, its
 * body is pre-compiled into the NVCC-generated BMI. When cl.exe instantiates
 * CudaLinearOp::quantize() it only needs the declaration of this function —
 * not the body — so no CUDA headers or intrinsics reach cl.exe.
 */

module;
#include <cstdint>
#include <stdexcept>
#include <format>
#include "Kernels/Quantization/CudaFp8WeightQuantization.cuh"
#include "Kernels/Quantization/CudaFp4WeightQuantization.cuh"

export module Compute.CudaLinearOp:Quantize;

import Dnn.ITensor;
import Dnn.TensorTypes;
import Serialization.Tensor;

namespace Mila::Dnn::Compute::Cuda::Linear
{
    namespace Detail
    {
        /**
         * @brief Validate, quantize and upload a BF16 weight blob to FP8_E4M3 on device.
         *
         * Validates the incoming blob shape against expected_shape, then delegates to
         * cuda_quantize_fp8_per_channel() for per-channel absmax quantization and device
         * upload. See cuda_quantize_fp8_per_channel() in CudaFp8WeightQuantization.cu for
         * the quantization algorithm.
         *
         * This function is the non-template bridge that keeps all CUDA host code inside
         * NVCC-compiled TUs. CudaLinearOp::quantize() (a template member body compiled
         * by NVCC) is the sole caller.
         *
         * @param blob           Host BF16 weight blob from the model archive.
         * @param weight_out     Device FP8_E4M3 tensor of shape [out_features, in_features].
         * @param scales_out     Device float32 tensor of shape [out_features].
         * @param expected_shape Expected weight shape for validation.
         *
         * @throws std::invalid_argument if the blob shape does not match expected_shape.
         * @throws std::runtime_error    if a cudaMemcpy device upload fails.
         */
        export void quantize_fp8_per_channel(
            const Mila::Dnn::Serialization::ITensorBlob& blob,
            Mila::Dnn::ITensor&                          weight_out,
            Mila::Dnn::ITensor&                          scales_out,
            const Mila::Dnn::shape_t&                    expected_shape,
            void*                                        dev_staging,
            cudaStream_t                                 stream )
        {
            const auto& meta = blob.getMetadata();

            if ( meta.shape != expected_shape )
            {
                throw std::invalid_argument( std::format(
                    "quantize_fp8_per_channel - shape mismatch: expected [{},{}], got [{},{}]",
                    expected_shape[ 0 ], expected_shape[ 1 ],
                    meta.shape[ 0 ],     meta.shape[ 1 ] ) );
            }

            const int64_t out_features = static_cast<int64_t>( expected_shape[ 0 ] );
            const int64_t in_features  = static_cast<int64_t>( expected_shape[ 1 ] );

            cuda_quantize_fp8_per_channel(
                blob.data(),
                weight_out.rawData(),
                static_cast<float*>( scales_out.rawData() ),
                out_features,
                in_features,
                dev_staging,
                stream );
        }

        /**
         * @brief Validate, quantize and upload a BF16 weight blob to FP8_E4M3 with a
         *        single per-tensor scale — for the Ada (SM 8.9+) cuBLASLt TN path.
         *
         * Folds all per-channel scales into one global scale:
         *   global_scale = max(|W[o, i]|, for all o, i) / 448.0f
         *
         * Every slot of scales_out is filled with global_scale so that scales_out[0]
         * can be passed directly as CUBLASLT_MATMUL_DESC_A_SCALE_POINTER.
         *
         * @param blob           Host BF16 weight blob from the model archive.
         * @param weight_out     Device FP8_E4M3 tensor of shape [out_features, in_features].
         * @param scales_out     Device float32 tensor of shape [out_features].
         * @param expected_shape Expected weight shape for validation.
         *
         * @throws std::invalid_argument if the blob shape does not match expected_shape.
         * @throws std::runtime_error    if any CUDA call fails.
         */
        export void quantize_fp8_per_tensor(
            const Mila::Dnn::Serialization::ITensorBlob& blob,
            Mila::Dnn::ITensor&                          weight_out,
            Mila::Dnn::ITensor&                          scales_out,
            const Mila::Dnn::shape_t&                    expected_shape,
            void*                                        dev_staging,
            cudaStream_t                                 stream )
        {
            const auto& meta = blob.getMetadata();

            if ( meta.shape != expected_shape )
            {
                throw std::invalid_argument( std::format(
                    "quantize_fp8_per_tensor - shape mismatch: expected [{},{}], got [{},{}]",
                    expected_shape[ 0 ], expected_shape[ 1 ],
                    meta.shape[ 0 ],     meta.shape[ 1 ] ) );
            }

            const int64_t out_features = static_cast<int64_t>( expected_shape[ 0 ] );
            const int64_t in_features  = static_cast<int64_t>( expected_shape[ 1 ] );

            cuda_quantize_fp8_per_tensor(
                blob.data(),
                weight_out.rawData(),
                static_cast<float*>( scales_out.rawData() ),
                out_features,
                in_features,
                dev_staging,
                stream );
        }

        /**
         * @brief Validate, quantize and upload a BF16 weight blob to packed FP4_E2M1
         *        with per-group float32 scales.
         *
         * For each output channel and each group of group_size input channels:
         *   scale[n, g] = max(|W[n, g*gs..(g+1)*gs)|) / 6.0f
         *   packed[n, k/2] = fp4_e2m1(W[n,k]/scale) nibble-packed (low=even, high=odd)
         *
         * This function is the non-template CL.EXE/NVCC boundary crossing point for
         * the FP4 quantize-on-load path. CudaLinearOp::quantize() (template body,
         * CL.EXE) passes the compile-time group_size as a runtime int; this function
         * dispatches to the correct NVCC-compiled kernel instantiation.
         *
         * @param blob           Host BF16 weight blob [out_features, in_features].
         * @param weight_out     Device UINT8 tensor [out_features, in_features/2].
         * @param scales_out     Device FP32 tensor [out_features, in_features/group_size].
         * @param expected_shape Expected weight shape [out_features, in_features].
         * @param group_size     Quantization group size (64 or 128).
         */
        export void quantize_fp4_per_group(
            const Mila::Dnn::Serialization::ITensorBlob& blob,
            Mila::Dnn::ITensor&                          weight_out,
            Mila::Dnn::ITensor&                          scales_out,
            const Mila::Dnn::shape_t&                    expected_shape,
            int                                          group_size,
            void*                                        dev_staging,
            cudaStream_t                                 stream )
        {
            const auto& meta = blob.getMetadata();

            if ( meta.shape != expected_shape )
            {
                throw std::invalid_argument( std::format(
                    "quantize_fp4_per_group - shape mismatch: expected [{},{}], got [{},{}]",
                    expected_shape[ 0 ], expected_shape[ 1 ],
                    meta.shape[ 0 ],     meta.shape[ 1 ] ) );
            }

            const int64_t out_features = static_cast<int64_t>( expected_shape[ 0 ] );
            const int64_t in_features  = static_cast<int64_t>( expected_shape[ 1 ] );

            cuda_quantize_fp4_per_group(
                blob.data(),
                weight_out.rawData(),
                static_cast<float*>( scales_out.rawData() ),
                out_features,
                in_features,
                group_size,
                dev_staging,
                stream );
        }

    } // namespace Detail

} // namespace Mila::Dnn::Compute::Cuda::Linear
