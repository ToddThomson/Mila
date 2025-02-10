#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>

import Mila;

namespace Dnn::Modules::Tests
{
	namespace MilaDnn = Mila::Dnn;

    class GeluTests : public ::testing::Test {
    protected:
        void SetUp() override {
			batch_size_ = 128;
			cpu_batch_size_= 4;
			sequence_length_ = 1024;
			channels_ = 768;
			input_shape_ = { batch_size_, sequence_length_, channels_ };
			
            cpu_gelu = std::make_unique<MilaDnn::Modules::Gelu<float, MilaDnn::Compute::CpuMemoryResource>>(
                "cpu_gelu", input_shape_, output_channels_ );

            //cuda_linear = std::make_unique<MilaDnn::Modules::Linear<float, MilaDnn::Compute::DeviceMemoryResource>>(
            //    "cuda_linear_2", input_shape_, output_channels_ );
        }

        std::unique_ptr<MilaDnn::Modules::Gelu<float, MilaDnn::Compute::CpuMemoryResource>> cpu_gelu;
        //std::unique_ptr<MilaDnn::Modules::Linear<float, MilaDnn::Compute::DeviceMemoryResource>> cuda_linear;
        
        size_t batch_size_{ 0 };
        size_t cpu_batch_size_{ 0 };
        size_t sequence_length_{ 0 };
        size_t channels_{ 0 };
		size_t output_channels_{ 0 };
		std::vector<size_t> input_shape_;
    };

    TEST_F( GeluTests, Cpu_TestName ) {
        EXPECT_EQ( cpu_gelu->name(), "cpu_gelu" );
    }

    TEST_F( GeluTests, Cpu_TestParameters ) {
        auto num_parameters = 0;
        EXPECT_EQ( cpu_gelu->parameters(), num_parameters ) ;
    }

    TEST_F( GeluTests, Cpu_TestForward ) {
        MilaDnn::Tensor<float, MilaDnn::Compute::CpuMemoryResource> input( { cpu_batch_size_, sequence_length_, channels_ } );
        auto output = cpu_gelu->forward( input );
        EXPECT_EQ( output.size(), cpu_batch_size_ * sequence_length_ * channels_ );
    }

    /*TEST_F( GeluTests, Cuda_TestForward ) {
        MilaDnn::Tensor<float, MilaDnn::Compute::DeviceMemoryResource> input( { batch_size_, sequence_length_, channels_ } );
        auto output = cuda_gelu->forward( std::make_shared<MilaDnn::Tensor<float,MilaDnn::Compute::DeviceMemoryResource>>( input ) );
        EXPECT_EQ( output->size(), batch_size_ * sequence_length_ * output_channels_ );
    }*/

    TEST_F( GeluTests, Cpu_TestPrint ) {
        testing::internal::CaptureStdout();
        cpu_gelu->print();
        std::string output = testing::internal::GetCapturedStdout();
        EXPECT_NE( output.find( "Module: cpu_gelu" ), std::string::npos );
        EXPECT_NE( output.find( "Parameters: " ), std::string::npos );
    }
}