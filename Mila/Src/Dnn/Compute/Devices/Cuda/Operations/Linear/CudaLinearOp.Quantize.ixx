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
            const Mila::Dnn::shape_t&                    expected_shape )
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
                in_features );
        }

    } // namespace Detail

} // namespace Mila::Dnn::Compute::Cuda::Linear
