#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <memory>

import Mila;

using namespace Mila::Dnn::Modules;
using namespace Mila::Dnn;
using namespace Mila::Dnn::Compute;

namespace Dnn::Modules::Tests
{
    class SoftmaxTest : public ::testing::Test {
    protected:
        void SetUp() override {
            input_shape = { 5, 3, 8 };
            softmax = std::make_shared<Softmax<float, CpuMemoryResource>>( "softmax_test", input_shape );
        }

        std::vector<size_t> input_shape;
        std::shared_ptr<Softmax<float, CpuMemoryResource>> softmax;
    };

    TEST_F( SoftmaxTest, TestName ) {
        EXPECT_EQ( softmax->name(), "softmax_test" );
    }

    TEST_F( SoftmaxTest, TestParameters ) {
        EXPECT_EQ( softmax->parameters(), 0 );
    }

    TEST_F( SoftmaxTest, Cpu_TestForward ) {
        auto input = Tensor<float, CpuMemoryResource>( input_shape );
        random<float, Compute::CpuMemoryResource>( input, -5.0f, 5.0f );
        auto output = softmax->forward( input );
        ASSERT_EQ( output.shape(), input_shape );

		// Check if all values in the output sum to a value close to 1
		auto B = output.shape()[ 0 ];
		auto T = output.shape()[ 1 ];
		auto V = output.shape()[ 2 ];
		
        for ( size_t i = 0; i < B; ++i ) {
			for ( size_t j = 0; j < T; ++j ) {
				auto sum = 0.0f;
				for ( size_t v = 0; v < V; ++v ) {
					sum += output[ i, j, v ];
				}
                EXPECT_NEAR( sum, 1.0f, 1e-6 );
			}
        }
    }

    TEST_F( SoftmaxTest, TestPrint ) {
        testing::internal::CaptureStdout();
        softmax->print();
        std::string output = testing::internal::GetCapturedStdout();
        EXPECT_NE( output.find( "Module: softmax_test" ), std::string::npos );
        EXPECT_NE( output.find( "Parameters: 0" ), std::string::npos );
    }
}