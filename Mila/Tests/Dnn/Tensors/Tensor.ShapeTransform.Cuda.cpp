/**
 * @file Tensor.ShapeTransform.Cuda.cpp
 * @brief CUDA shape-transformation tests for device-resource tensors.
 *
 * Device companion to Tensor.ShapeTransform.cpp. Shape transforms are metadata-only
 * (no host access), so they apply equally to device tensors; this confirms reshape,
 * flatten, and view operate on a non-host-accessible resource. GPU-local: compiled
 * only under MILA_ENABLE_CUDA, skipped when no device is present.
 */

#include <gtest/gtest.h>

import Mila;

namespace Mila::Tests::Dnn::Tensors
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    class TensorShapeTransformCudaTests : public testing::Test {
    protected:
        void SetUp() override {
            has_cuda_ = DeviceRegistry::instance().hasDeviceType( DeviceType::Cuda );
        }

        bool has_cuda_ = false;
    };

    TEST_F( TensorShapeTransformCudaTests, ReshapeAndFlatten_OnDeviceTensor ) {
        if ( !has_cuda_ ) {
            GTEST_SKIP() << "CUDA device not available.";
        }

        DeviceTensor<TensorDataType::FP16> t( Device::Cuda( 0 ), shape_t{ 2, 3, 4 } );

        t.reshape( shape_t{ 24 } );
        EXPECT_EQ( t.size(), 24u );

        t.reshape( shape_t{ 2, 3, 4 } );
        t.flatten();
        EXPECT_EQ( t.shape(), (shape_t{ 6, 4 }) );
    }

    TEST_F( TensorShapeTransformCudaTests, View_SharesDeviceBuffer ) {
        if ( !has_cuda_ ) {
            GTEST_SKIP() << "CUDA device not available.";
        }

        DeviceTensor<TensorDataType::FP16> parent( Device::Cuda( 0 ), shape_t{ 2, 3 } );

        auto v = parent.view( shape_t{ 6 } );
        EXPECT_TRUE( v.isView() );
        EXPECT_EQ( v.size(), 6u );
    }
}
