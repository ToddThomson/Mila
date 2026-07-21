/**
 * @file Tensor.Identity.cpp
 * @brief Identity/metadata tests for Tensor.ixx — uid, name, setName.
 *
 * Area file of the Tensor value-type archetype (see Specifications/Testing.Tensors.md).
 * Identity is dtype/device-independent; exercised once on CPU. Covers unique-id
 * generation, name round-trip, and the setName empty-string throw.
 */

#include <gtest/gtest.h>
#include <stdexcept>
#include <string>

import Mila;

namespace Mila::Tests::Dnn::Tensors
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    class TensorIdentityTests : public testing::Test {};

    TEST_F( TensorIdentityTests, Uid_NonEmptyAndUnique ) {
        HostTensor<TensorDataType::FP32> a( Device::Cpu(), shape_t{ 2 } );
        HostTensor<TensorDataType::FP32> b( Device::Cpu(), shape_t{ 2 } );

        EXPECT_FALSE( a.getUId().empty() );
        EXPECT_NE( a.getUId(), b.getUId() );
    }

    TEST_F( TensorIdentityTests, Name_DefaultsEmptyThenRoundTrips ) {
        HostTensor<TensorDataType::FP32> t( Device::Cpu(), shape_t{ 2 } );
        EXPECT_TRUE( t.getName().empty() );

        t.setName( "weights" );
        EXPECT_EQ( t.getName(), "weights" );
    }

    TEST_F( TensorIdentityTests, Name_ConstructorArgumentIsStored ) {
        HostTensor<TensorDataType::FP32> t( Device::Cpu(), shape_t{ 2 }, "bias" );
        EXPECT_EQ( t.getName(), "bias" );
    }

    TEST_F( TensorIdentityTests, SetName_ThrowsWhenEmpty ) {
        HostTensor<TensorDataType::FP32> t( Device::Cpu(), shape_t{ 2 } );
        EXPECT_THROW( t.setName( "" ), std::invalid_argument );
    }
}
