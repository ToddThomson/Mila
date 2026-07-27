/**
 * @file Tensor.ShapeTransform.cpp
 * @brief Shape-transformation tests for Tensor.ixx — view, reshape, flatten.
 *
 * Area file of the Tensor value-type archetype (see Specifications/Testing.Tensors.md).
 * This area had no test file before the Alpha.7 Tensor revival. Shape transforms are
 * dtype/device-independent metadata operations, exercised on CPU here; the device
 * companion (Tensor.ShapeTransform.Cuda.cpp) confirms they work without host access.
 * Covers reshape (preserve-count + grow-from-empty), flatten, view (buffer sharing +
 * offset), isView, and every documented throw.
 */

#include <gtest/gtest.h>
#include <stdexcept>

import Mila;

namespace Mila::Tests::Dnn::Tensors
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    class TensorShapeTransformTests : public testing::Test {};

    // ====================================================================
    // reshape
    // ====================================================================

    TEST_F( TensorShapeTransformTests, Reshape_PreservesElementCount ) {
        HostTensor<TensorDataType::FP32> t( Device::Cpu(), shape_t{ 2, 3 } );

        t.reshape( shape_t{ 6 } );
        EXPECT_EQ( t.shape(), (shape_t{ 6 }) );
        EXPECT_EQ( t.size(), 6u );
        EXPECT_EQ( t.rank(), 1u );
    }

    TEST_F( TensorShapeTransformTests, Reshape_GrowsEmptyTensorAndAllocates ) {
        HostTensor<TensorDataType::FP32> t( Device::Cpu(), shape_t{ 0 } );
        ASSERT_TRUE( t.empty() );

        t.reshape( shape_t{ 2, 2 } );
        EXPECT_EQ( t.size(), 4u );
        EXPECT_FALSE( t.empty() );
        EXPECT_NE( t.data(), nullptr );
    }

    TEST_F( TensorShapeTransformTests, Reshape_ThrowsWhenElementCountChanges ) {
        HostTensor<TensorDataType::FP32> t( Device::Cpu(), shape_t{ 2, 3 } );
        EXPECT_THROW( t.reshape( shape_t{ 4 } ), std::runtime_error );
    }

    // ====================================================================
    // flatten
    // ====================================================================

    TEST_F( TensorShapeTransformTests, Flatten_CollapsesToTwoDimensions ) {
        HostTensor<TensorDataType::FP32> t( Device::Cpu(), shape_t{ 2, 3, 4 } );

        t.flatten();
        EXPECT_EQ( t.shape(), (shape_t{ 6, 4 }) );
    }

    TEST_F( TensorShapeTransformTests, Flatten_NoOpForRankAtMostOne ) {
        HostTensor<TensorDataType::FP32> vec( Device::Cpu(), shape_t{ 5 } );
        vec.flatten();
        EXPECT_EQ( vec.shape(), (shape_t{ 5 }) );

        HostTensor<TensorDataType::FP32> scalar( Device::Cpu(), shape_t{} );
        scalar.flatten();
        EXPECT_EQ( scalar.rank(), 0u );
    }

    // ====================================================================
    // view
    // ====================================================================

    TEST_F( TensorShapeTransformTests, View_SharesBufferAndFlagsView ) {
        HostTensor<TensorDataType::FP32> parent( Device::Cpu(), shape_t{ 2, 3 } );
        parent[ 0, 0 ] = 7.0f;

        auto v = parent.view( shape_t{ 6 } );

        EXPECT_TRUE( v.isView() );
        EXPECT_FALSE( parent.isView() );
        EXPECT_EQ( v.size(), 6u );
        // Shared buffer: writing through the parent is visible through the view.
        EXPECT_EQ( v.data()[ 0 ], 7.0f );
    }

    TEST_F( TensorShapeTransformTests, View_AppliesOffset ) {
        HostTensor<TensorDataType::FP32> parent( Device::Cpu(), shape_t{ 6 } );
        parent[ 2 ] = 3.0f;

        auto v = parent.view( shape_t{ 4 }, 2 );
        EXPECT_EQ( v.data()[ 0 ], 3.0f );
    }

    TEST_F( TensorShapeTransformTests, View_ThrowsWhenOffsetExceedsBuffer ) {
        HostTensor<TensorDataType::FP32> parent( Device::Cpu(), shape_t{ 6 } );
        EXPECT_THROW( (void)parent.view( shape_t{ 6 }, 1 ), std::invalid_argument );
    }

    TEST_F( TensorShapeTransformTests, View_ThrowsWhenNoBuffer ) {
        HostTensor<TensorDataType::FP32> empty( Device::Cpu(), shape_t{ 0 } );
        EXPECT_THROW( (void)empty.view( shape_t{ 1 } ), std::runtime_error );
    }
}
