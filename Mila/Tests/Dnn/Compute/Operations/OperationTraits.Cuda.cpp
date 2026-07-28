/**
 * @file OperationTraits.Cuda.cpp
 * @brief Contract tests for the CUDA half of the OperationTraits dispatch table.
 *
 * Device companion to OperationTraits.cpp. Compile-time only: these are
 * static_asserts and need no GPU at runtime, but they need the CUDA
 * specializations to exist, so the file is gated on MILA_ENABLE_CUDA.
 *
 * The quantization rows here are the load-time-quantization half of the
 * inference-drought backfill: they pin which (precision, weight policy) pairs
 * exist, which is the decision a model's fromPretrained dispatch depends on.
 */

#include <gtest/gtest.h>
#include <type_traits>

import Mila;

namespace Mila::Tests::Dnn::Compute::Operations
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Quant::Weight;
    using namespace Mila::Dnn::Quant::KvCache;

    // ====================================================================
    // A. Linear -- the reference component for the full dispatch pattern
    // ====================================================================

    // Unquantized: FP32 (reference/validation) and BF16 (the primary target).
    static_assert( OperationSupported<OperationType::LinearOp, DeviceType::Cuda,
        TensorDataType::FP32, NoWeightQuant> );
    static_assert( OperationSupported<OperationType::LinearOp, DeviceType::Cuda,
        TensorDataType::BF16, NoWeightQuant> );

    // Every quantized weight policy is registered at BF16 only. This is the
    // design rule from CLAUDE.md -- quantization rides the BF16 activation path --
    // and pinning it means an accidental FP32 quantized row cannot appear unnoticed.
    static_assert( OperationSupported<OperationType::LinearOp, DeviceType::Cuda,
        TensorDataType::BF16, PerChannelFp8<>> );
    static_assert( OperationSupported<OperationType::LinearOp, DeviceType::Cuda,
        TensorDataType::BF16, PerGroupFp4<128>> );
    static_assert( OperationSupported<OperationType::LinearOp, DeviceType::Cuda,
        TensorDataType::BF16, PerGroupFp4<64>> );
    static_assert( OperationSupported<OperationType::LinearOp, DeviceType::Cuda,
        TensorDataType::BF16, PerGroupInt4<128>> );
    static_assert( OperationSupported<OperationType::LinearOp, DeviceType::Cuda,
        TensorDataType::BF16, PerGroupInt4<64>> );

    static_assert( !OperationSupported<OperationType::LinearOp, DeviceType::Cuda,
        TensorDataType::FP32, PerChannelFp8<>> );
    static_assert( !OperationSupported<OperationType::LinearOp, DeviceType::Cuda,
        TensorDataType::FP32, PerGroupFp4<128>> );

    // The group size is part of the policy type, so it is part of the dispatch key:
    // an unregistered group size must fail rather than silently reuse another.
    static_assert( !OperationSupported<OperationType::LinearOp, DeviceType::Cuda,
        TensorDataType::BF16, PerGroupFp4<32>> );

    // Distinct policies must resolve to distinct op types. Without this, a policy
    // could be accepted and then quietly dispatch to the wrong kernel -- exactly the
    // failure the compile-time table exists to make impossible.
    static_assert( !std::is_same_v<
        OperationTraits<OperationType::LinearOp, DeviceType::Cuda, TensorDataType::BF16, NoWeightQuant>::type,
        OperationTraits<OperationType::LinearOp, DeviceType::Cuda, TensorDataType::BF16, PerChannelFp8<>>::type> );
    static_assert( !std::is_same_v<
        OperationTraits<OperationType::LinearOp, DeviceType::Cuda, TensorDataType::BF16, PerChannelFp8<>>::type,
        OperationTraits<OperationType::LinearOp, DeviceType::Cuda, TensorDataType::BF16, PerGroupFp4<128>>::type> );
    static_assert( !std::is_same_v<
        OperationTraits<OperationType::LinearOp, DeviceType::Cuda, TensorDataType::BF16, PerGroupFp4<64>>::type,
        OperationTraits<OperationType::LinearOp, DeviceType::Cuda, TensorDataType::BF16, PerGroupFp4<128>>::type> );

    // Precision is likewise part of the key.
    static_assert( !std::is_same_v<
        OperationTraits<OperationType::LinearOp, DeviceType::Cuda, TensorDataType::FP32, NoWeightQuant>::type,
        OperationTraits<OperationType::LinearOp, DeviceType::Cuda, TensorDataType::BF16, NoWeightQuant>::type> );

    // ====================================================================
    // B. GQA -- the KV-cache policy axis
    // ====================================================================

    static_assert( OperationSupported<OperationType::GroupedQueryAttentionOp, DeviceType::Cuda,
        TensorDataType::FP32, NoKvCompression> );
    static_assert( OperationSupported<OperationType::GroupedQueryAttentionOp, DeviceType::Cuda,
        TensorDataType::BF16, NoKvCompression> );
    static_assert( OperationSupported<OperationType::GroupedQueryAttentionOp, DeviceType::Cuda,
        TensorDataType::BF16, SlidingWindowKvCache> );

    // Sliding-window vs unbounded must be distinct types: they are the bounded-ring
    // and full-history kernels, and Gemma runs both in one model.
    static_assert( !std::is_same_v<
        OperationTraits<OperationType::GroupedQueryAttentionOp, DeviceType::Cuda,
            TensorDataType::BF16, NoKvCompression>::type,
        OperationTraits<OperationType::GroupedQueryAttentionOp, DeviceType::Cuda,
            TensorDataType::BF16, SlidingWindowKvCache>::type> );

    // FP8 KV compression is the Qwen 3 milestone. There is no PerChannelKvFp8 type to
    // assert against yet -- KvCache/Policy.ixx defines only NoKvCompression and
    // SlidingWindowKvCache -- so the absence is pinned one level up instead: both
    // existing policies are inactive-or-window, and neither carries a storage dtype.
    static_assert( KvCachePolicy<NoKvCompression> );
    static_assert( KvCachePolicy<SlidingWindowKvCache> );
    static_assert( !NoKvCompression::kIsActive );
    static_assert( SlidingWindowKvCache::kIsActive );

    // A weight policy must not satisfy the KV-policy axis.
    static_assert( !OperationSupported<OperationType::GroupedQueryAttentionOp, DeviceType::Cuda,
        TensorDataType::BF16, NoWeightQuant> );

    // ====================================================================
    // C. Token embedding -- the tied-table quantization row
    // ====================================================================

    static_assert( OperationSupported<OperationType::TokenEmbeddingOp, DeviceType::Cuda,
        TensorDataType::FP32, NoWeightQuant> );
    static_assert( OperationSupported<OperationType::TokenEmbeddingOp, DeviceType::Cuda,
        TensorDataType::BF16, NoWeightQuant> );

    // PerChannelFp8 exists because the Gemma tied embedding/lm_head table is FP8;
    // PerGroupFp4 deliberately does not. Pinned so the asymmetry is a decision.
    static_assert( OperationSupported<OperationType::TokenEmbeddingOp, DeviceType::Cuda,
        TensorDataType::BF16, PerChannelFp8<>> );
    static_assert( !OperationSupported<OperationType::TokenEmbeddingOp, DeviceType::Cuda,
        TensorDataType::BF16, PerGroupFp4<128>> );

    // ====================================================================
    // D. Policy-free ops -- precision coverage as it actually stands
    // ====================================================================

    static_assert( OperationSupported<OperationType::RmsNormOp, DeviceType::Cuda, TensorDataType::FP32, void> );
    static_assert( OperationSupported<OperationType::RmsNormOp, DeviceType::Cuda, TensorDataType::BF16, void> );
    static_assert( OperationSupported<OperationType::RopeOp, DeviceType::Cuda, TensorDataType::FP32, void> );
    static_assert( OperationSupported<OperationType::RopeOp, DeviceType::Cuda, TensorDataType::BF16, void> );
    static_assert( OperationSupported<OperationType::SwigluOp, DeviceType::Cuda, TensorDataType::FP32, void> );
    static_assert( OperationSupported<OperationType::SwigluOp, DeviceType::Cuda, TensorDataType::BF16, void> );
    static_assert( OperationSupported<OperationType::GegluOp, DeviceType::Cuda, TensorDataType::BF16, void> );
    static_assert( OperationSupported<OperationType::ResidualOp, DeviceType::Cuda, TensorDataType::BF16, void> );
    static_assert( OperationSupported<OperationType::SamplingOp, DeviceType::Cuda, TensorDataType::BF16, void> );
    static_assert( OperationSupported<OperationType::CrossEntropyOp, DeviceType::Cuda, TensorDataType::BF16, void> );

    // These three are FP32-only today. Recorded as the current contract rather than
    // as an aspiration: Softmax and MHA are on the GPT-2 lineage, and the Llama/Gemma
    // path reaches neither at BF16.
    static_assert( !OperationSupported<OperationType::SoftmaxOp, DeviceType::Cuda, TensorDataType::BF16, void> );
    static_assert( !OperationSupported<OperationType::MultiHeadAttentionOp, DeviceType::Cuda,
        TensorDataType::BF16, void> );
    static_assert( !OperationSupported<OperationType::GeluOp, DeviceType::Cuda, TensorDataType::BF16, void> );

    // LayerNorm is registered at FP32 and FP16 -- and NOT BF16, which is the odd row
    // in the whole table given BF16 is the primary target and FP16 is slated for
    // removal (Production Hardening, "Remove FP16"). Pinned so that removal has to
    // confront this line rather than discover it: deleting the FP16 row leaves
    // LayerNorm CUDA at FP32 only.
    static_assert( OperationSupported<OperationType::LayerNormOp, DeviceType::Cuda, TensorDataType::FP32, void> );
    static_assert( OperationSupported<OperationType::LayerNormOp, DeviceType::Cuda, TensorDataType::FP16, void> );
    static_assert( !OperationSupported<OperationType::LayerNormOp, DeviceType::Cuda, TensorDataType::BF16, void> );

    TEST( OperationTraitsCudaTests, DispatchTableContractsHoldAtCompileTime )
    {
        SUCCEED() << "OperationTraits CUDA dispatch contracts are static_asserts; "
                     "compiling this file is the assertion.";
    }
}
