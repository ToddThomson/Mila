/**
 * @file Quantization.ixx
 * @brief Umbrella module for the quantization policies: weights and the KV cache.
 */

export module Dnn.Quantization;

/// @brief Weight quantization policies -- consumed by @c Linear and @c CudaLinearOp.
export import Dnn.Quantization.Weight.Policies;

/// @brief KV cache policy concept and the uncompressed policy -- consumed by @c GroupedQueryAttention and @c CudaGqaOp.
export import Dnn.Quantization.KvCache.Policy;

/// @brief The concept a quantizing KV cache policy satisfies.
export import Dnn.Quantization.KvCache.QuantPolicy;

/// @brief FP8 KV cache compression, one scale per KV head per token.
export import Dnn.Quantization.KvCache.PerTokenKvFp8;
export import Dnn.Quantization.KvCache.SlidingWindowKvFp8;
