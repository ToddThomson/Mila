#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <sstream>

import Mila;

namespace Modules::Tests
{
    using namespace Mila::Dnn;

    class AttentionTests : public ::testing::Test {
    protected:
        void SetUp() override {
            batch_size_ = 64;
            cpu_batch_size_ = 4;
            sequence_length_ = 1024;
            channels_ = 768;
			num_heads_ = 12;
            cpu_input_shape_ = { cpu_batch_size_, sequence_length_, 3 * channels_ };
            cuda_input_shape_ = { batch_size_, sequence_length_, 3 * channels_ };
            has_bias_ = true;

            cpu_attention = std::make_unique<MultiHeadAttention<float, float, Compute::DeviceType::Cpu>>(
                "cpu_attn", cpu_input_shape_, num_heads_ );

            /*cuda_linear = std::make_unique<MilaDnn::Modules::Linear<float, MilaDnn::Compute::DeviceMemoryResource>>(
                "cuda_linear_2", input_shape_, output_channels_ );*/
        }

        std::unique_ptr<MultiHeadAttention<float, float, Compute::DeviceType::Cpu>> cpu_attention;
        //std::unique_ptr<MilaDnn::Modules::Linear<float, MilaDnn::Compute::DeviceMemoryResource>> cuda_linear;

        size_t batch_size_{ 0 };
        size_t cpu_batch_size_{ 0 };
        size_t sequence_length_{ 0 };
        size_t channels_{ 0 };
		size_t num_heads_{ 0 };
        std::vector<size_t> cpu_input_shape_;
        std::vector<size_t> cuda_input_shape_;
        bool has_bias_{ true };
    };

    TEST_F( AttentionTests, Cpu_TestName ) {
        EXPECT_EQ( cpu_attention->getName(), "cpu_attn" );
    }

    //TEST_F( AttentionTests, Cpu_TestParameters ) {
    //    auto num_parameters = /* weights */ channels_;
    //    if ( has_bias_ ) {
    //        num_parameters += channels_;
    //    }

    //    EXPECT_EQ( cpu_attention->parameters(), num_parameters );
    //}

    TEST_F( AttentionTests, Cpu_TestForward ) {
        Tensor<float, Compute::HostMemoryResource> input( { cpu_batch_size_, sequence_length_,  3 * channels_ } );
        Tensor<float, Compute::HostMemoryResource> output( { cpu_batch_size_, sequence_length_,  3 * channels_ } );
        cpu_attention->forward( input, output );
        EXPECT_EQ( output.size(), cpu_batch_size_ * sequence_length_ *  ( 3 * channels_ ) );
    }

    /*TEST_F( AttentionTests, Cuda_TestForward ) {
        MilaDnn::Tensor<float, MilaDnn::Compute::DeviceMemoryResource> input( { batch_size_, sequence_length_, channels_ } );
        auto output = cuda_linear->forward( std::make_shared<MilaDnn::Tensor<float, MilaDnn::Compute::DeviceMemoryResource>>( input ) );
        EXPECT_EQ( output->size(), batch_size_ * sequence_length_ * output_channels_ );
    }*/

    TEST_F( AttentionTests, Test_toString ) {
		std::string output = cpu_attention->toString();
        EXPECT_NE( output.find("MultiHeadAttention: cpu_attn"), std::string::npos);
    }
}