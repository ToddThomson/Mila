#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>

import Mila;

namespace Modules::Tests
{
    using namespace Mila::Dnn;

    class ResidualTests : public ::testing::Test {
    protected:
        void SetUp() override {
            batch_size_ = 128;
            cpu_batch_size_ = 4;
            sequence_length_ = 1024;
            channels_ = 768;
            cpu_io_shape_ = { cpu_batch_size_, sequence_length_, channels_ };
			cuda_input_shape_ = { batch_size_, sequence_length_, channels_ };

            cpu_residual = std::make_unique<Residual<float, float, Compute::CpuDevice>>(
                "cpu_residual" );

            //cuda_linear = std::make_unique<MilaDnn::Modules::Linear<float, MilaDnn::Compute::DeviceMemoryResource>>(
            //    "cuda_linear_2", input_shape_ );
        }

        std::unique_ptr<Residual<float, float, Compute::CpuDevice>> cpu_residual;
        //std::unique_ptr<MilaDnn::Modules::Linear<float, MilaDnn::Compute::DeviceMemoryResource>> cuda_linear;

        size_t batch_size_{ 0 };
        size_t cpu_batch_size_{ 0 };
        size_t sequence_length_{ 0 };
        size_t channels_{ 0 };
        std::vector<size_t> cpu_io_shape_;
        std::vector<size_t> cuda_input_shape_;
    };

    TEST_F( ResidualTests, Cpu_TestName ) {
        EXPECT_EQ( cpu_residual->getName(), "cpu_residual" );
    }

    TEST_F( ResidualTests, Cpu_parameterCount ) {
        auto num_parameters = 0;
        EXPECT_EQ( cpu_residual->parameterCount(), num_parameters );
    }

    TEST_F( ResidualTests, Cpu_TestForward ) {
        Tensor<float, Compute::CpuMemoryResource> input( cpu_io_shape_ );
        Tensor<float, Compute::CpuMemoryResource> output( cpu_io_shape_ );
        cpu_residual->forward( input, output );
        EXPECT_EQ( output.size(), input.size() );
    }

    /*TEST_F( ResidualTests, Cuda_TestForward ) {
        MilaDnn::Tensor<float, MilaDnn::Compute::DeviceMemoryResource> input( { batch_size_, sequence_length_, channels_ } );
        auto output = cuda_gelu->forward( std::make_shared<MilaDnn::Tensor<float,MilaDnn::Compute::DeviceMemoryResource>>( input ) );
        EXPECT_EQ( output->size(), batch_size_ * sequence_length_ * output_channels_ );
    }*/

    TEST_F( ResidualTests, Cpu_TestPrint ) {
        testing::internal::CaptureStdout();
        cpu_residual->print();
        std::string output = testing::internal::GetCapturedStdout();
        EXPECT_NE( output.find( "Module: cpu_residual" ), std::string::npos );
        EXPECT_NE( output.find( "Parameter count: " ), std::string::npos );
    }
}