#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>

import Mila;

namespace Modules::Tests
{
	using namespace Mila::Dnn;

    class LinearTests : public ::testing::Test {
    protected:
        void SetUp() override {
			batch_size_ = 128;
			cpu_batch_size_= 4;
			sequence_length_ = 1024;
			channels_ = 768;
			cuda_input_shape_ = { batch_size_, sequence_length_, channels_ };
            cpu_input_shape_ = { cpu_batch_size_, sequence_length_, channels_ };
            output_features_ = 4;
			output_channels_ = output_features_ * channels_;
			has_bias_ = true;

            cpu_linear = std::make_unique<Linear<float, float,Compute::CpuDevice>>(
                "cpu_linear", channels_, output_channels_ );

            /*cuda_linear = std::make_unique<Linear<float, float, Compute::CudaDevice>>(
                "cuda_linear", channels_, output_channels_ );*/
        }

        std::unique_ptr<Linear<float, float, Compute::CpuDevice>> cpu_linear;
        //std::unique_ptr<Linear<float, float, Compute::CudaDevice>> cuda_linear;
        
        size_t batch_size_{ 0 };
        size_t cpu_batch_size_{ 0 };
        size_t sequence_length_{ 0 };
        size_t channels_{ 0 };
		size_t output_channels_{ 0 };
		std::vector<size_t> cuda_input_shape_;
        std::vector<size_t> cpu_input_shape_;
        size_t output_features_{ 0 };
		bool has_bias_{ true };
    };

    TEST_F( LinearTests, Cpu_TestName ) {
        EXPECT_EQ( cpu_linear->getName(), "cpu_linear" );
    }

    TEST_F( LinearTests, Cpu_parameterCount ) {
        auto num_parameters = /* weights */(output_channels_ * channels_);
		if ( has_bias_ ) {
			num_parameters += output_channels_;
		}

        EXPECT_EQ( cpu_linear->parameterCount(), num_parameters ) ;
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
        Tensor<float, Compute::CpuMemoryResource> input( { cpu_batch_size_, sequence_length_, channels_ } );
        Tensor<float, Compute::CpuMemoryResource> output( { cpu_batch_size_, sequence_length_, 4 * channels_ } );
        cpu_linear->forward( input, output );
        EXPECT_EQ( output.size(), cpu_batch_size_ * sequence_length_ * output_channels_ );
    }

    /*TEST_F( LinearTests, Cuda_TestForward ) {
        Tensor<float, Compute::CudaMemoryResource> input( { batch_size_, sequence_length_, channels_ } );
        Tensor<float, Compute::CudaMemoryResource> output( { batch_size_, sequence_length_, 4 * channels_ } );
        cuda_linear->forward( input, output );
        EXPECT_EQ( output.size(), batch_size_ * sequence_length_ * output_channels_ );
    }*/

    TEST_F( LinearTests, toString ) {
        std::string output = cpu_linear->toString();
        EXPECT_NE( output.find( "Linear: cpu_linear" ), std::string::npos );
    }
}