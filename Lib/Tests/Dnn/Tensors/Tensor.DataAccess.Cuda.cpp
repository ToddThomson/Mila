/**
 * @file Tensor.DataAccess.Cuda.cpp
 * @brief CUDA element-access tests — host-visible resources + host-only contract.
 *
 * Device companion to Tensor.DataAccess.cpp / Tensor.DataPointers.cpp. Two concerns:
 * (1) pinned and managed memory are host-accessible, so item()/operator[]/data()
 * work at runtime on them; (2) a device-only resource (CudaDeviceMemoryResource) is
 * NOT host-accessible, so those same members are constrained away
 * (requires is_host_accessible) and must be non-callable — verified at compile time
 * with requires-expressions, no GPU needed. GPU-local: compiled only under
 * MILA_ENABLE_CUDA; the runtime cases skip when no device is present.
 */

#include <gtest/gtest.h>

import Mila;

namespace Mila::Tests::Dnn::Tensors
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    class TensorDataAccessCudaTests : public testing::Test {
    protected:
        void SetUp() override {
            has_cuda_ = DeviceRegistry::instance().hasDeviceType( DeviceType::Cuda );
        }

        bool has_cuda_ = false;
    };

    // ====================================================================
    // Host-only access contract (compile-time, no device required)
    // ====================================================================

    // The host accessors are constrained on TMemoryResource::is_host_accessible.
    // Detection must route through a parameterized concept, not an inline
    // requires-expression on a concrete type: MSVC treats a non-dependent
    // requires-expression whose only candidates fail their constraints as a hard
    // error (C7500) instead of yielding false. A template parameter puts the
    // constraint check back into a SFINAE context, where the failure is merely an
    // unsatisfied requirement.
    template <typename TensorT>
    concept SupportsData = requires ( TensorT t ) { t.data(); };

    template <typename TensorT>
    concept SupportsItem = requires ( TensorT t ) { t.item(); };

    template <typename TensorT>
    concept SupportsIndexing = requires ( TensorT t, index_t i ) { t[ i ]; };

    TEST_F( TensorDataAccessCudaTests, HostOnlyContract_DeviceTensorAccessorsAreNotCallable ) {
        // A device-only resource is not host-accessible, so the host accessors are
        // constrained away and must not be callable.
        static_assert( !SupportsData<DeviceTensor<TensorDataType::FP32>> );
        static_assert( !SupportsItem<DeviceTensor<TensorDataType::FP32>> );
        static_assert( !SupportsIndexing<DeviceTensor<TensorDataType::FP32>> );

        // Host-accessible CUDA resources keep them.
        static_assert( SupportsData<PinnedTensor<TensorDataType::FP32>> );
        static_assert( SupportsData<UniversalTensor<TensorDataType::FP32>> );

        SUCCEED();
    }

    // ====================================================================
    // Runtime access on host-visible device memory
    // ====================================================================

    TEST_F( TensorDataAccessCudaTests, PinnedMemory_OperatorIndexReadWrite ) {
        if ( !has_cuda_ ) {
            GTEST_SKIP() << "CUDA device not available.";
        }

        PinnedTensor<TensorDataType::FP32> tensor( Device::Cuda( 0 ), shape_t{ 2, 3 } );

        tensor[ 0, 0 ] = 1.1f;
        tensor[ 1, 2 ] = 2.2f;

        EXPECT_FLOAT_EQ( (tensor[ 0, 0 ]), 1.1f );
        EXPECT_FLOAT_EQ( (tensor[ 1, 2 ]), 2.2f );
    }

    TEST_F( TensorDataAccessCudaTests, ManagedMemory_DataPointerReadWrite ) {
        if ( !has_cuda_ ) {
            GTEST_SKIP() << "CUDA device not available.";
        }

        UniversalTensor<TensorDataType::FP32> tensor( Device::Cuda( 0 ), shape_t{ 2, 2 } );

        auto* data_ptr = tensor.data();
        data_ptr[ 0 ] = 4.4f;
        data_ptr[ 3 ] = 5.5f;

        EXPECT_FLOAT_EQ( (tensor[ 0, 0 ]), 4.4f );
        EXPECT_FLOAT_EQ( (tensor[ 1, 1 ]), 5.5f );
    }
}
