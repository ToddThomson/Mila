/**
 * @file Tensor.Io.cpp
 * @brief String-representation tests for Tensor.ixx — toString, operator<<.
 *
 * Area file of the Tensor value-type archetype (see Specifications/Testing.Tensors.md).
 * Diagnostics surface; exercised on a host-accessible CPU tensor so buffer content
 * is printable. Covers toString header fields, the showBuffer path, and the
 * stream-insertion operator delegating to toString.
 */

#include <gtest/gtest.h>
#include <sstream>
#include <string>

import Mila;

namespace Mila::Tests::Dnn::Tensors
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    class TensorIoTests : public testing::Test {};

    TEST_F( TensorIoTests, ToString_ContainsIdentityShapeTypeDevice ) {
        HostTensor<TensorDataType::FP32> t( Device::Cpu(), shape_t{ 2, 3 }, "activations" );

        std::string s = t.toString();

        EXPECT_NE( s.find( t.getUId() ), std::string::npos );
        EXPECT_NE( s.find( "activations" ), std::string::npos );
        EXPECT_NE( s.find( "FP32" ), std::string::npos );
        EXPECT_NE( s.find( "shape=[2,3]" ), std::string::npos );
    }

    TEST_F( TensorIoTests, ToString_ShowBufferAppendsContent ) {
        HostTensor<TensorDataType::FP32> t( Device::Cpu(), shape_t{ 2, 2 } );
        t[ 0, 0 ] = 1.0f;

        std::string with_buffer = t.toString( true );
        std::string header_only = t.toString( false );

        EXPECT_GT( with_buffer.size(), header_only.size() );
    }

    TEST_F( TensorIoTests, StreamInsertion_DelegatesToToString ) {
        HostTensor<TensorDataType::FP32> t( Device::Cpu(), shape_t{ 2, 3 }, "x" );

        std::ostringstream oss;
        oss << t;

        EXPECT_EQ( oss.str(), t.toString() );
    }
}
