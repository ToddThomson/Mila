/**
 * @file Tensor.Properties.cpp
 * @brief Introspection tests for Tensor.ixx — shape/strides/size/rank/empty/id.
 *
 * Area file of the Tensor value-type archetype (see Specifications/Testing.Tensors.md).
 * These accessors are dtype- and device-independent, so they are exercised once on
 * a CPU FP32 tensor. Covers shape, strides, size, rank, empty vs scalar, isScalar,
 * isValid, and getDeviceId.
 */

#include <gtest/gtest.h>

import Mila;

namespace Mila::Tests::Dnn::Tensors
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    class TensorPropertiesTests : public testing::Test {};

    TEST_F( TensorPropertiesTests, Shape_MatchesConstructionForEachRank ) {
        HostTensor<TensorDataType::FP32> t1( Device::Cpu(), shape_t{ 10 } );
        HostTensor<TensorDataType::FP32> t2( Device::Cpu(), shape_t{ 2, 3 } );
        HostTensor<TensorDataType::FP32> t3( Device::Cpu(), shape_t{ 4, 5, 6 } );

        EXPECT_EQ( t1.shape(), (shape_t{ 10 }) );
        EXPECT_EQ( t2.shape(), (shape_t{ 2, 3 }) );
        EXPECT_EQ( t3.shape(), (shape_t{ 4, 5, 6 }) );

        EXPECT_EQ( t1.rank(), 1 );
        EXPECT_EQ( t2.rank(), 2 );
        EXPECT_EQ( t3.rank(), 3 );
    }

    TEST_F( TensorPropertiesTests, Strides_AreRowMajor ) {
        HostTensor<TensorDataType::FP32> t( Device::Cpu(), shape_t{ 2, 3, 4 } );

        // Row-major strides for {2,3,4} are {12, 4, 1}.
        ASSERT_EQ( t.strides().size(), 3u );
        EXPECT_EQ( t.strides()[ 0 ], 12 );
        EXPECT_EQ( t.strides()[ 1 ], 4 );
        EXPECT_EQ( t.strides()[ 2 ], 1 );
    }

    TEST_F( TensorPropertiesTests, Size_IsProductOfShape ) {
        HostTensor<TensorDataType::FP32> t( Device::Cpu(), shape_t{ 4, 5, 6 } );
        EXPECT_EQ( t.size(), 120u );
    }

    TEST_F( TensorPropertiesTests, Empty_ScalarIsNotEmpty ) {
        HostTensor<TensorDataType::FP32> scalar( Device::Cpu(), shape_t{} );
        EXPECT_FALSE( scalar.empty() );   // size == 1
        EXPECT_TRUE( scalar.isScalar() );

        HostTensor<TensorDataType::FP32> zero( Device::Cpu(), shape_t{ 0 } );
        EXPECT_TRUE( zero.empty() );       // a zero in the shape is empty
        EXPECT_FALSE( zero.isScalar() );

        HostTensor<TensorDataType::FP32> normal( Device::Cpu(), shape_t{ 2, 2 } );
        EXPECT_FALSE( normal.empty() );
        EXPECT_FALSE( normal.isScalar() );
    }

    TEST_F( TensorPropertiesTests, IsValid_TrueForConstructedTensor ) {
        // isValid() currently reports the constructed contract (see Tensor.ixx FIXME
        // on a future moved-from state); assert today's behavior so a later change
        // has a green oracle.
        HostTensor<TensorDataType::FP32> t( Device::Cpu(), shape_t{ 2, 2 } );
        EXPECT_TRUE( t.isValid() );
    }

    TEST_F( TensorPropertiesTests, DeviceId_ReportsCpu ) {
        HostTensor<TensorDataType::FP32> t( Device::Cpu(), shape_t{ 2, 2 } );
        EXPECT_EQ( t.getDeviceId().type, DeviceType::Cpu );
    }
}
