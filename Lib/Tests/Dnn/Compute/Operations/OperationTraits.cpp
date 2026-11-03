/**
 * @file OperationTraits.cpp
 * @brief Contract tests for the OperationTraits dispatch table that hold without CUDA.
 *
 * These are static_asserts, not runtime checks: the contract under test is the
 * compile-time one, and a regression should fail the build rather than a test binary.
 * This file must keep compiling under MILA_ENABLE_CUDA=OFF; the CUDA rows live in
 * OperationTraits.Cuda.cpp.
 */

#include <gtest/gtest.h>

import Mila;

namespace Mila::Tests::Dnn::Compute::Operations
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Quant::Weight;

    // OperationSupported must report false rather than hard-error on an unsupported
    // tuple. That property is what lets a multi-precision typed test skip precisions
    // an op does not implement, and it holds only while the primary template stays
    // undefined -- a static_assert in the primary body would fire during this probe
    // and turn the detectable false back into a compile error.
    //
    // The CPU device has no compute operations, so every CPU tuple probes false.
    static_assert( !OperationSupported<OperationType::LinearOp, DeviceType::Cpu,
        TensorDataType::FP32, NoWeightQuant> );
    static_assert( !OperationSupported<OperationType::RmsNormOp, DeviceType::Cpu,
        TensorDataType::FP32, void> );
    static_assert( !OperationSupported<OperationType::SamplingOp, DeviceType::Cpu,
        TensorDataType::FP32, void> );
    static_assert( !OperationSupported<OperationType::MoeOp, DeviceType::Cpu,
        TensorDataType::FP32, void> );

    // Every assertion in this file is compile-time; the build succeeding is the
    // result. This case exists so the file reports as a run test rather than
    // silently contributing nothing to the ctest count.
    TEST( OperationTraitsTests, DispatchTableContractsHoldAtCompileTime )
    {
        SUCCEED() << "OperationTraits dispatch contracts are static_asserts; "
                     "compiling this file is the assertion.";
    }
}
