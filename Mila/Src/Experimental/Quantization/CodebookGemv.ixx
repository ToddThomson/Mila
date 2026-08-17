/**
 * @file CodebookGemv.ixx
 * @brief Module surface over the codebook GEMV kernels: plain-type host entries so
 * consumers and tests need no CUDA headers. Operations include Kernels/CodebookGemv.cuh
 * directly for the stream-based launchers.
 */

module;
#include <cstdint>
#include "Kernels/CodebookGemv.cuh"

export module Experimental.Quantization.CodebookGemv;

namespace Mila::Dnn::Experimental::Quantization
{
    // Returns 0 on success, else the first cudaError_t as int. Activations, bias
    // and the returned outputs pass through BF16 exactly as inference will see them.
    export inline int runCodebook2MatVecDecodeOnDevice(
        const float* activations, const std::uint8_t* packedCodes,
        const std::uint16_t* scaleBits, const float* codebook, const float* bias,
        float* output, int columns, int rows, int groupSize )
    {
        return Kernels::run_codebook2_matvec_decode_host(
            activations, packedCodes, scaleBits, codebook, bias, output,
            columns, rows, groupSize );
    }

    export inline int runCodebook3MatVecDecodeOnDevice(
        const float* activations, const std::uint8_t* planeTwoBits,
        const std::uint8_t* planeOneBit, const std::uint16_t* scaleBits,
        const float* codebook, const float* bias, float* output,
        int columns, int rows, int groupSize )
    {
        return Kernels::run_codebook3_matvec_decode_host(
            activations, planeTwoBits, planeOneBit, scaleBits, codebook, bias, output,
            columns, rows, groupSize );
    }

    export inline bool codebookGemvDeviceAvailable()
    {
        return Kernels::codebook_gemv_device_available() != 0;
    }
}
