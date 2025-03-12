#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <memory>

import Mila;

namespace Modules::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    class SoftmaxTest : public ::testing::Test {
    protected:
        void SetUp() override {
            io_shape = { 5, 3, 8 };
            softmax_dim = 2;
            
            softmax = std::make_unique<Softmax<float, float, CpuDevice>>( "softmax_1", softmax_dim );
        }

        std::vector<size_t> io_shape;
        int64_t softmax_dim;
        std::unique_ptr<Softmax<float, float, CpuDevice>> softmax;
    };

    TEST_F( SoftmaxTest, TestName ) {
        EXPECT_EQ( softmax->name(), "softmax_1" );
    }

    TEST_F( SoftmaxTest, Test_parameterCount ) {
        EXPECT_EQ( softmax->parameterCount(), 0 );
    }

    TEST_F( SoftmaxTest, Cpu_TestForward ) {
        auto input = Tensor<float, CpuMemoryResource>( io_shape );
        auto output = Tensor<float, CpuMemoryResource>( io_shape );
        random<float, Compute::CpuMemoryResource>( input, -5.0f, 5.0f );

        softmax->forward( input, output );

        ASSERT_EQ( output.shape(), io_shape );

		auto B = output.shape()[ 0 ];
		auto T = output.shape()[ 1 ];
		auto V = output.shape()[ 2 ];
		
        for ( size_t i = 0; i < B; ++i ) {
			for ( size_t j = 0; j < T; ++j ) {
				// For each (b,t) position, sum the values across the vocabulary dimension
                // Check if the sum is a value close to 1
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
        EXPECT_NE( output.find( "Module: softmax_1" ), std::string::npos );
        EXPECT_NE( output.find( "Parameter count: 0" ), std::string::npos );
    }
}