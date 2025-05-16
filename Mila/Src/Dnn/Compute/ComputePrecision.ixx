module;
#include <string_view>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

export module Compute.Precision;

import Dnn.TensorTraits;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Traits for compute precision types used in neural network operations.
     *
     * This trait structure provides metadata and utilities for precision types
     * used in tensor computations, particularly for mixed-precision operations
     * on CUDA devices.
     *
     * @tparam T The precision type
     */
    export template <typename T>
        struct PrecisionTraits {
        static constexpr bool IsSupported = ValidFloatTensorType<T>;
        using Type = T;

        static constexpr std::string_view type_name = Mila::Dnn::TensorTrait<T>::type_name;
        static constexpr size_t size_in_bytes = Mila::Dnn::TensorTrait<T>::size_in_bytes;
    };

    /**
     * @brief Helper to select appropriate precision based on hardware capabilities.
     *
     * This template provides a mechanism to select the most efficient precision
     * type based on the available hardware, when "auto" precision is requested.
     *
     * @tparam TOutput The output tensor type to consider when selecting precision
     */
    export template <typename TOutput>
        struct AutoPrecisionSelector {
        // On modern CUDA hardware with tensor cores, could select FP16 for better performance
        // For now, default to the same type as output
        using Type = TOutput;

        // Helper method to determine best precision based on device capabilities
        template <typename TDevice>
        static auto selectBestPrecision() {
            // Logic to determine most efficient precision based on device capabilities
            // Could check for tensor cores, compute capability, etc.
            if constexpr ( std::is_same_v<TDevice, float> ) {
                // For example, if device has good tensor core support, return half
                // Otherwise return float
                return float{};
            }
            else {
                return TOutput{};
            }
        }
    };

    /**
     * @brief Concept that verifies a type is a valid precision type for computation.
     *
     * This concept ensures that a type has PrecisionTraits support and
     * is a valid floating-point tensor type.
     *
     * @tparam T The type to check for valid precision support
     */
    export template <typename T>
        concept ValidPrecisionType = ValidFloatTensorType<T> && PrecisionTraits<T>::IsSupported;

    /**
     * @brief Type trait to resolve the actual precision type to use.
     *
     * This type trait handles special cases when TPrecision is meant to be automatically
     * determined from the output type.
     *
     * @tparam TPrecision The requested precision type
     * @tparam TOutput The output tensor type (used for default precision)
     */
    export template <typename TPrecision, typename TOutput>
        struct ResolvePrecisionType {
        // Regular case: use the requested precision directly
        using Type = TPrecision;
    };

    /**
     * @brief Helper function to get a human-readable precision type name.
     *
     * @tparam T The precision type
     * @return constexpr std::string_view The type name
     */
    export template <typename T>
        constexpr std::string_view precision_type_name() {
        return PrecisionTraits<T>::type_name;
    }
}