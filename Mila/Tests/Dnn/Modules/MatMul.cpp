#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>

import Mila;

namespace Dnn::Modules::Tests
{
	namespace MilaDnn = Mila::Dnn;

    class MatMulTests : public ::testing::Test {
    protected:
        void SetUp() override {
			batch_size_ = 128;
			cpu_batch_size_= 4;
			sequence_length_ = 1024;
			channels_ = 768;
			input_shape_ = { batch_size_, sequence_length_, channels_ };
			output_channels_ = 3 * channels_;

            cpu_matmul = std::make_unique<MilaDnn::Modules::MatMul<float, MilaDnn::Compute::CpuMemoryResource>>( 
                "CpuMatMul_1", input_shape_, output_channels_, true );

            cuda_matmul = std::make_unique<MilaDnn::Modules::MatMul<float, MilaDnn::Compute::DeviceMemoryResource>>(
                "CudaMatMul_2", input_shape_, output_channels_, true );
        }

        std::unique_ptr<MilaDnn::Modules::MatMul<float, MilaDnn::Compute::CpuMemoryResource>> cpu_matmul;
        std::unique_ptr<MilaDnn::Modules::MatMul<float, MilaDnn::Compute::DeviceMemoryResource>> cuda_matmul;
        
        size_t batch_size_{ 0 };
        size_t cpu_batch_size_{ 0 };
        size_t sequence_length_{ 0 };
        size_t channels_{ 0 };
		std::vector<size_t> input_shape_;
        size_t output_channels_{ 0 };
    };

    TEST_F( MatMulTests, Cpu_TestName ) {
        EXPECT_EQ( cpu_matmul->name(), "CpuMatMul_1" );
    }

    TEST_F( MatMulTests, Cpu_TestParameters ) {
        EXPECT_EQ( cpu_matmul->parameters(), 2 * output_channels_ );
    }

    TEST_F( MatMulTests, Cpu_TestWeightInitialization ) {
        auto weight = cpu_matmul->getWeight();
        EXPECT_EQ( weight->size(), output_channels_ * channels_ );
    }

    TEST_F( MatMulTests, Cpu_TestBiasInitialization ) {
        auto bias = cpu_matmul->getBias();
        EXPECT_EQ( bias->size(), output_channels_ );
    }

    TEST_F( MatMulTests, Cpu_TestForward ) {
        MilaDnn::Tensor<float, MilaDnn::Compute::CpuMemoryResource> input( { cpu_batch_size_, sequence_length_, channels_ } );
        auto output = cpu_matmul->forward( std::make_shared<MilaDnn::HostTensor<float>>( input ) );
        EXPECT_EQ( output->size(), cpu_batch_size_ * sequence_length_ * output_channels_ );
    }

    TEST_F( MatMulTests, Cuda_TestForward ) {
        MilaDnn::Tensor<float, MilaDnn::Compute::DeviceMemoryResource> input( { batch_size_, sequence_length_, channels_ } );
        auto output = cuda_matmul->forward( std::make_shared<MilaDnn::Tensor<float,MilaDnn::Compute::DeviceMemoryResource>>( input ) );
        EXPECT_EQ( output->size(), batch_size_ * sequence_length_ * output_channels_ );
    }

    TEST_F( MatMulTests, Cpu_TestPrint ) {
        testing::internal::CaptureStdout();
        cpu_matmul->print();
        std::string output = testing::internal::GetCapturedStdout();
        EXPECT_NE( output.find( "Module: CpuMatMul_1" ), std::string::npos );
        EXPECT_NE( output.find( "Parameters: " ), std::string::npos );
    }
}