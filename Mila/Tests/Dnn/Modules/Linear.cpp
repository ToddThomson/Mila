#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>

import Mila;

namespace Dnn::Modules::Tests
{
	namespace MilaDnn = Mila::Dnn;

    class LinearTests : public ::testing::Test {
    protected:
        void SetUp() override {
			batch_size_ = 128;
			cpu_batch_size_= 4;
			sequence_length_ = 1024;
			channels_ = 768;
			input_shape_ = { batch_size_, sequence_length_, channels_ };
            output_features_ = 3;
			output_channels_ = output_features_ * channels_;
			has_bias_ = true;

            cpu_linear = std::make_unique<MilaDnn::Modules::Linear<float, MilaDnn::Compute::CpuMemoryResource>>(
                "cpu_linear_1", input_shape_, output_channels_ );

            cuda_linear = std::make_unique<MilaDnn::Modules::Linear<float, MilaDnn::Compute::DeviceMemoryResource>>(
                "cuda_linear_2", input_shape_, output_channels_ );
        }

        std::unique_ptr<MilaDnn::Modules::Linear<float, MilaDnn::Compute::CpuMemoryResource>> cpu_linear;
        std::unique_ptr<MilaDnn::Modules::Linear<float, MilaDnn::Compute::DeviceMemoryResource>> cuda_linear;
        
        size_t batch_size_{ 0 };
        size_t cpu_batch_size_{ 0 };
        size_t sequence_length_{ 0 };
        size_t channels_{ 0 };
		size_t output_channels_{ 0 };
		std::vector<size_t> input_shape_;
        size_t output_features_{ 0 };
		bool has_bias_{ true };
    };

    TEST_F( LinearTests, Cpu_TestName ) {
        EXPECT_EQ( cpu_linear->name(), "cpu_linear_1" );
    }

    TEST_F( LinearTests, Cpu_TestParameters ) {
        auto num_parameters = /* weights */(output_channels_ * channels_);
		if ( has_bias_ ) {
			num_parameters += output_channels_;
		}

        EXPECT_EQ( cpu_linear->parameters(), num_parameters ) ;
    }

    TEST_F( LinearTests, Cpu_TestWeightInitialization ) {
        auto weight = cpu_linear->getWeight();
        EXPECT_EQ( weight->size(), output_channels_ * channels_ );
    }

    TEST_F( LinearTests, Cpu_TestBiasInitialization ) {
        auto bias = cpu_linear->getBias();
        EXPECT_EQ( bias->size(), output_channels_ );
    }

    TEST_F( LinearTests, Cpu_TestForward ) {
        MilaDnn::Tensor<float, MilaDnn::Compute::CpuMemoryResource> input( { cpu_batch_size_, sequence_length_, channels_ } );
        auto output = cpu_linear->forward( std::make_shared<MilaDnn::HostTensor<float>>( input ) );
        EXPECT_EQ( output->size(), cpu_batch_size_ * sequence_length_ * output_channels_ );
    }

    TEST_F( LinearTests, Cuda_TestForward ) {
        MilaDnn::Tensor<float, MilaDnn::Compute::DeviceMemoryResource> input( { batch_size_, sequence_length_, channels_ } );
        auto output = cuda_linear->forward( std::make_shared<MilaDnn::Tensor<float,MilaDnn::Compute::DeviceMemoryResource>>( input ) );
        EXPECT_EQ( output->size(), batch_size_ * sequence_length_ * output_channels_ );
    }

    TEST_F( LinearTests, Cpu_TestPrint ) {
        testing::internal::CaptureStdout();
        cpu_linear->print();
        std::string output = testing::internal::GetCapturedStdout();
        EXPECT_NE( output.find( "Module: cpu_linear_1" ), std::string::npos );
        EXPECT_NE( output.find( "Parameters: " ), std::string::npos );
    }
}