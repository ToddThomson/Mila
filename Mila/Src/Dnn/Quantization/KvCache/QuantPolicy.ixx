/**
 * @file QuantPolicy.ixx
 * @brief Quantization-specific KV cache compression policies.
 *
 * Refines @c KvCachePolicy with the dtype and scale granularity fields
 * required by the FP8 cache write and read kernels in @c CudaGqaOp.
 *
 * A future @c SlidingWindowPolicy would import @c Dnn.Quantization.KvCache.Policy
 * directly and would @b not import this module, as sliding window eviction
 * carries no dtype fields.
 *
 * @note Alpha.6 target: @c PerChannelKvFp8<> -- symmetric per-head per-token
 *       FP8 compression applied identically to K and V.
 */

 module;
 #include <concepts>

export module Dnn.Quantization.KvCache.QuantPolicy;

import Dnn.Quantization.KvCache.Policy;
import Dnn.TensorDataType;

namespace Mila::Dnn::Quant::KvCache
{
    /**
     * @brief Concept for quantization-based KV cache compression policies.
     *
     * Refinement of @c KvCachePolicy requiring the additional dtype and
     * granularity fields consumed by the FP8 cache kernels in @c CudaGqaOp.
     * Used where the operation needs to inspect storage and scale dtypes,
     * for example during kernel selection and scale tensor allocation.
     *
     * @note @c NoKvCompression satisfies @c KvCachePolicy but does @b not
     *       satisfy @c QuantKvPolicy. @c CudaGqaOp guards all
     *       @c QuantKvPolicy-specific paths with
     *       @c if constexpr (kKvCompressed) before accessing these fields.
     *
     * @tparam T Candidate policy type.
     */
    export template<typename T>
    concept QuantKvPolicy = KvCachePolicy<T> && requires
    {
        { T::kStorageDtype    } -> std::convertible_to<TensorDataType>;
        {
            T::kScaleDtype
        } -> std::convertible_to<TensorDataType>;
        {
            T::kPerHeadPerToken
        } -> std::convertible_to<bool>;
        {
            T::kSymmetric
        } -> std::convertible_to<bool>;
    };

    /**
     * @brief Symmetric per-head per-token FP8 KV cache compression policy.
     *
     * Compresses K and V cache tensors from BF16 to FP8_E4M3 on every prefill
     * chunk write and every decode append. Only the FP8 representation is
     * stored in the cache.
     *
     * Scale granularity is per-head per-token: one @c float32 scale per KV
     * head per cached token position, computed as
     * @c max(abs(x[head,token,:])) / 448.0f. Scale tensor shape is
     * @c [num_kv_heads, max_seq_len].
     *
     * K and V use this policy symmetrically (@c kSymmetric = @c true).
     * Asymmetric K/V compression is not a current Mila target.
     *
     * On the read path, FP8 K and V are dequantized to transient BF16 buffers
     * immediately before attention score and weighted-sum computation.
     * Dequantized values are never written back to the cache.
     *
     * @tparam TStorage Storage dtype for compressed cache values. Defaults to
     *         @c FP8_E4M3 (4 exponent bits, 3 mantissa bits), consistent with
     *         weight quantization. @c FP8_E5M2 is reserved for gradients and
     *         is not a current Mila target.
     */
    export template<TensorDataType TStorage = TensorDataType::FP8_E4M3>
    struct PerChannelKvFp8
    {
        static constexpr bool kIsActive = true;  ///< Activates KV cache compression in @c CudaGqaOp.
        static constexpr TensorDataType kStorageDtype = TStorage;                ///< Storage dtype for compressed K and V values.
        static constexpr TensorDataType kScaleDtype = TensorDataType::FP32;   ///< Dtype for per-head per-token scale factors.
        static constexpr bool kPerHeadPerToken = true;  ///< One scale per KV head per token position.
        static constexpr bool kSymmetric = true;  ///< K and V share the same compression policy.
    };

    static_assert(KvCachePolicy<PerChannelKvFp8<>>);
    static_assert(QuantKvPolicy<PerChannelKvFp8<>>);

}
