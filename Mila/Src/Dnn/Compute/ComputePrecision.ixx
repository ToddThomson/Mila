module;

export module Compute.Precision;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Enumeration of supported compute precision types.
     *
     * Defines the available numeric precisions for internal computation in operations.
     * This allows runtime selection of computation precision independently from
     * input/output tensor data types.
     */
    export enum class ComputePrecision {
        FP8,    ///< 8-bit floating point (NVIDIA FP8)
        FP16,   ///< Half precision (16-bit floating point)
        BF16,   ///< Brain floating point (16-bit, different exponent/mantissa than FP16)
        FP32,   ///< Single precision (32-bit floating point)
        Default ///< Use the same precision as the output tensor type
    };
}