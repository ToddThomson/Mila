#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>

import Mila;

namespace Dnn::Modules::Tests
{
    namespace MilaDnn = Mila::Dnn;

    class LayernormTests : public ::testing::Test {
    protected:
        void SetUp() override {
            batch_size_ = 128;
            cpu_batch_size_ = 4;
            sequence_length_ = 1024;
            channels_ = 768;
			cpu_input_shape_ = { cpu_batch_size_, sequence_length_, channels_ };
            cuda_input_shape_ = { batch_size_, sequence_length_, channels_ };
            has_bias_ = true;

            cpu_layernorm = std::make_unique<MilaDnn::Modules::LayerNorm<float, float, MilaDnn::Compute::CpuMemoryResource>>(
                "cpu_ln", cpu_input_shape_ );

            /*cuda_linear = std::make_unique<MilaDnn::Modules::Linear<float, MilaDnn::Compute::DeviceMemoryResource>>(
                "cuda_ln", input_shape_, output_channels_ );*/
        }

        std::unique_ptr<MilaDnn::Modules::LayerNorm<float, float, MilaDnn::Compute::CpuMemoryResource>> cpu_layernorm;
        std::unique_ptr<MilaDnn::Modules::LayerNorm<float, float, MilaDnn::Compute::DeviceMemoryResource>> cuda_layernorm;

        size_t batch_size_{ 0 };
        size_t cpu_batch_size_{ 0 };
        size_t sequence_length_{ 0 };
        size_t channels_{ 0 };
        std::vector<size_t> cpu_input_shape_;
        std::vector<size_t> cuda_input_shape_;
        bool has_bias_{ true };
    };

    TEST_F( LayernormTests, Cpu_TestName ) {
        EXPECT_EQ( cpu_layernorm->name(), "cpu_ln" );
    }

    TEST_F( LayernormTests, Cpu_TestParameters ) {
        auto num_parameters = /* weights */ channels_;
        if ( has_bias_ ) {
            num_parameters += channels_;
        }

        EXPECT_EQ( cpu_layernorm->parameters(), num_parameters );
    }

    TEST_F( LayernormTests, Cpu_TestWeightInitialization ) {
        auto weight = cpu_layernorm->getWeight();
        EXPECT_EQ( weight->size(), channels_ );
    }

    TEST_F( LayernormTests, Cpu_TestBiasInitialization ) {
        auto bias = cpu_layernorm->getBias();
        EXPECT_EQ( bias->size(), channels_ );
    }

    TEST_F( LayernormTests, Cpu_TestForward ) {
        MilaDnn::Tensor<float, MilaDnn::Compute::CpuMemoryResource> input( cpu_input_shape_ );
        auto output = cpu_layernorm->forward( input );
        EXPECT_EQ( output.size(), input.size() );
    }

    /*TEST_F( LayernormTests, Cuda_TestForward ) {
        MilaDnn::Tensor<float, MilaDnn::Compute::DeviceMemoryResource> input( { batch_size_, sequence_length_, channels_ } );
        auto output = cuda_linear->forward( std::make_shared<MilaDnn::Tensor<float, MilaDnn::Compute::DeviceMemoryResource>>( input ) );
        EXPECT_EQ( output->size(), batch_size_ * sequence_length_ * output_channels_ );
    }*/

    TEST_F( LayernormTests, Cpu_TestPrint ) {
        testing::internal::CaptureStdout();
        cpu_layernorm->print();
        std::string output = testing::internal::GetCapturedStdout();
        EXPECT_NE( output.find( "Module: cpu_ln" ), std::string::npos );
        EXPECT_NE( output.find( "Parameters: " ), std::string::npos );
    }
}