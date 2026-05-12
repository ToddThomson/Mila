/**
 * @file Quantization.ixx
 * @brief Umbrella module for the Mila quantization subsystem.
 *
 * Re-exports the full quantization policy surface. Consumers that need
 * all policy types in a single import (model builders, test harnesses)
 * use this module. Components that need only one subsystem should import
 * the targeted submodule directly to keep module dependency graphs tight:
 *
 * @code
 * // Linear component — weight quantization only
 * import Dnn.Quantization.WeightQuant.Policies;
 *
 * // GroupedQueryAttention — KV cache policy only (Alpha.5)
 * import Dnn.Quantization.KvCache.Policy;
 *
 * // GroupedQueryAttention — KV cache quantization (Alpha.6+)
 * import Dnn.Quantization.KvCache.QuantPolicy;
 * @endcode
 *
 * Module dependency graph (no cycles):
 * @code
 * Dnn.Quantization
 *   ??? Dnn.Quantization.WeightQuant.Policies
 *   ??? Dnn.Quantization.KvCache.Policy
 *         ??? Dnn.Quantization.KvCache.QuantPolicy
 * @endcode
 */

export module Dnn.Quantization;

/// @brief Weight quantization policy surface — consumed by @c Linear and @c CudaLinearOp.
export import Dnn.Quantization.Weight.Policies;

/// @brief KV cache compression base concept and identity — consumed by @c GroupedQueryAttention and @c CudaGqaOp.
export import Dnn.Quantization.KvCache.Policy;

/// @brief KV cache quantization policies — consumed by @c GroupedQueryAttention and @c CudaGqaOp (Alpha.6+).
export import Dnn.Quantization.KvCache.QuantPolicy;
