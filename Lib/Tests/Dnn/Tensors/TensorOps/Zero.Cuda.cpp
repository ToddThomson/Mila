/**
 * @file Zero.Cuda.cpp
 * @brief CUDA tests for TensorOps zero() (Dnn.TensorOps:Zero).
 *
 * Device companion to Zero.Cpu.cpp. Verifies TensorOps<Cuda>::zero on device memory
 * via a host-accessible managed tensor (so the result is readable on the host).
 * GPU-local: compiled only under MILA_ENABLE_CUDA, skipped when no device present.
 */

#include <gtest/gtest.h>

import Mila;

namespace Mila::Tests::Dnn::Tensors::TensorOps
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    class TensorOpsZeroCudaTests : public testing::Test {
    protected:
        void SetUp() override {
            has_cuda_ = DeviceRegistry::instance().hasDeviceType( DeviceType::Cuda );
        }

        bool has_cuda_ = false;
    };

    TEST_F( TensorOpsZeroCudaTests, Zero_SetsManagedTensorToZero ) {
        if ( !has_cuda_ ) {
            GTEST_SKIP() << "CUDA device not available.";
        }

        UniversalTensor<TensorDataType::FP32> t( Device::Cuda( 0 ), shape_t{ 2, 3 } );

        auto* data_ptr = t.data();
        for ( size_t i = 0; i < t.size(); ++i ) {
            data_ptr[ i ] = 5.0f;
        }

        zero( t );

        for ( size_t i = 0; i < t.size(); ++i ) {
            EXPECT_FLOAT_EQ( data_ptr[ i ], 0.0f );
        }
    }
}
