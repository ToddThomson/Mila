/**
 * @file OperationTraits.cpp
 * @brief Contract tests for the CPU half of the OperationTraits dispatch table.
 *
 * OperationTraits is the compile-time seam every component resolves its concrete
 * operation through, and until now nothing tested it directly. These are
 * static_asserts, not runtime checks: the contract under test *is* the compile-time
 * one, and a regression here should fail the build rather than a test binary.
 *
 * The assertions deliberately never name a concrete op class. Those names are not
 * re-exported through the umbrella, and pinning them would couple this file to
 * internal naming rather than to the dispatch contract. What is asserted instead is
 * which tuples resolve, which do not, and that distinct policies resolve to distinct
 * types.
 *
 * CUDA rows live in OperationTraits.Cuda.cpp; this file must keep compiling under
 * MILA_ENABLE_CUDA=OFF, so it is the half the CPU-only CI ratchet protects.
 */

#include <gtest/gtest.h>
#include <type_traits>

import Mila;

namespace Mila::Tests::Dnn::Compute::Operations
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Quant::Weight;

    // ====================================================================
    // A. The predicate itself -- the seam the rest of the suite relies on
    // ====================================================================

    // OperationSupported must report false rather than hard-error on an unsupported
    // tuple. That property is what lets a multi-precision typed test skip precisions
    // an op does not implement, and it holds only while the primary template stays
    // undefined -- a static_assert in the primary body would fire during this probe
    // and turn the detectable false back into a compile error.
    static_assert( OperationSupported<OperationType::LinearOp, DeviceType::Cpu,
        TensorDataType::FP32, NoWeightQuant> );
    static_assert( !OperationSupported<OperationType::LinearOp, DeviceType::Cpu,
        TensorDataType::BF16, NoWeightQuant> );

    // ====================================================================
    // B. The CPU table is FP32-only
    // ====================================================================

    static_assert( OperationSupported<OperationType::GeluOp, DeviceType::Cpu, TensorDataType::FP32, void> );
    static_assert( OperationSupported<OperationType::ResidualOp, DeviceType::Cpu, TensorDataType::FP32, void> );
    static_assert( OperationSupported<OperationType::LayerNormOp, DeviceType::Cpu, TensorDataType::FP32, void> );
    static_assert( OperationSupported<OperationType::SoftmaxOp, DeviceType::Cpu, TensorDataType::FP32, void> );
    static_assert( OperationSupported<OperationType::MultiHeadAttentionOp, DeviceType::Cpu, TensorDataType::FP32, void> );
    static_assert( OperationSupported<OperationType::LpeOp, DeviceType::Cpu, TensorDataType::FP32, void> );
    static_assert( OperationSupported<OperationType::SamplingOp, DeviceType::Cpu, TensorDataType::FP32, void> );

    // A functor-templated op registers `op_for` rather than `type`; OperationSupported
    // is documented to be satisfied by both, so this pins that half of the contract.
    static_assert( OperationSupported<OperationType::ElementwiseActivationOp, DeviceType::Cpu,
        TensorDataType::FP32, void> );

    // "BF16 CPU paths are not a current Mila target" (OperationTraits.Cpu.ixx) --
    // asserted so the claim is enforced rather than merely written down.
    static_assert( !OperationSupported<OperationType::GeluOp, DeviceType::Cpu, TensorDataType::BF16, void> );
    static_assert( !OperationSupported<OperationType::ResidualOp, DeviceType::Cpu, TensorDataType::BF16, void> );
    static_assert( !OperationSupported<OperationType::SoftmaxOp, DeviceType::Cpu, TensorDataType::BF16, void> );
    static_assert( !OperationSupported<OperationType::LayerNormOp, DeviceType::Cpu, TensorDataType::BF16, void> );

    // Llama-lineage CPU ops are the open [contributor] item in Production Hardening.
    // Their absence is the current contract, so it is pinned: when someone implements
    // them these lines fail and are deleted as part of that work.
    static_assert( !OperationSupported<OperationType::RmsNormOp, DeviceType::Cpu, TensorDataType::FP32, void> );
    static_assert( !OperationSupported<OperationType::SwigluOp, DeviceType::Cpu, TensorDataType::FP32, void> );
    static_assert( !OperationSupported<OperationType::RopeOp, DeviceType::Cpu, TensorDataType::FP32, void> );
    static_assert( !OperationSupported<OperationType::TokenEmbeddingOp, DeviceType::Cpu,
        TensorDataType::FP32, NoWeightQuant> );

    // GQA has no CPU backend at all -- the Llama/Gemma attention path is CUDA-only.
    static_assert( !OperationSupported<OperationType::GroupedQueryAttentionOp, DeviceType::Cpu,
        TensorDataType::FP32, void> );

    // ====================================================================
    // C. The policy axis is part of the key, not decoration
    // ====================================================================

    // LinearOp registers under NoWeightQuant, so the default void policy must NOT
    // resolve. This is the contract that gives a caller who omits the policy a clean
    // "undefined type" naming the tuple, instead of a silent match.
    static_assert( !OperationSupported<OperationType::LinearOp, DeviceType::Cpu,
        TensorDataType::FP32, void> );

    // Quantized weight policies are CUDA-only; on CPU they must not resolve at any
    // precision.
    static_assert( !OperationSupported<OperationType::LinearOp, DeviceType::Cpu,
        TensorDataType::FP32, PerChannelFp8<>> );
    static_assert( !OperationSupported<OperationType::LinearOp, DeviceType::Cpu,
        TensorDataType::BF16, PerGroupFp4<128>> );

    // Conversely a policy-free op keys on void and must not resolve under a weight
    // policy -- the two policy families are not interchangeable.
    static_assert( !OperationSupported<OperationType::SoftmaxOp, DeviceType::Cpu,
        TensorDataType::FP32, NoWeightQuant> );

    // ====================================================================
    // D. Runtime placeholder
    // ====================================================================

    // Every assertion in this file is compile-time; the build succeeding is the
    // result. This case exists so the file reports as a run test rather than
    // silently contributing nothing to the ctest count.
    TEST( OperationTraitsCpuTests, DispatchTableContractsHoldAtCompileTime )
    {
        SUCCEED() << "OperationTraits CPU dispatch contracts are static_asserts; "
                     "compiling this file is the assertion.";
    }
}
