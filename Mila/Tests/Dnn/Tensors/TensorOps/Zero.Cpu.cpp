/**
 * @file Zero.Cpu.cpp
 * @brief CPU tests for TensorOps zero() (Dnn.TensorOps:Zero).
 *
 * Operation-layer area file of the Tensor value-type archetype (see
 * Specifications/Testing.Tensors.md). zero() sets every element of a tensor to 0
 * via the device-dispatched TensorOps<Cpu>::zero. CPU memory resource only; the
 * device path is in Zero.Cuda.cpp. Behavior is dtype-independent, so a float and an
 * integer case suffice.
 */

#include <gtest/gtest.h>

import Mila;

namespace Mila::Tests::Dnn::Tensors::TensorOps
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    class TensorOpsZeroCpuTests : public testing::Test {};

    TEST_F( TensorOpsZeroCpuTests, Zero_SetsAllElementsToZero_Float ) {
        Tensor<TensorDataType::FP32, CpuMemoryResource> t( Device::Cpu(), shape_t{ 2, 3 } );

        auto* data_ptr = t.data();
        for ( size_t i = 0; i < t.size(); ++i ) {
            data_ptr[ i ] = 5.0f;
        }

        zero( t );

        for ( size_t i = 0; i < t.size(); ++i ) {
            EXPECT_FLOAT_EQ( data_ptr[ i ], 0.0f );
        }
    }

    TEST_F( TensorOpsZeroCpuTests, Zero_SetsAllElementsToZero_Integer ) {
        Tensor<TensorDataType::INT32, CpuMemoryResource> t( Device::Cpu(), shape_t{ 4 } );

        auto* data_ptr = t.data();
        for ( size_t i = 0; i < t.size(); ++i ) {
            data_ptr[ i ] = 7;
        }

        zero( t );

        for ( size_t i = 0; i < t.size(); ++i ) {
            EXPECT_EQ( data_ptr[ i ], 0 );
        }
    }

    TEST_F( TensorOpsZeroCpuTests, Zero_ScalarTensor ) {
        Tensor<TensorDataType::FP32, CpuMemoryResource> scalar( Device::Cpu(), shape_t{} );
        scalar.item() = 9.0f;

        zero( scalar );

        EXPECT_FLOAT_EQ( scalar.item(), 0.0f );
    }
}
