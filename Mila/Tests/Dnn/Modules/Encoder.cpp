#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>

import Mila;

namespace Modules::Tests
{
    using namespace Mila::Dnn;

    class EncoderTests : public ::testing::Test {
    protected:
        void SetUp() override {
            batch_size_ = 64;
            cpu_batch_size_ = 4;
            sequence_length_ = 512;
			max_seq_len_ = 1024;
            channels_ = 128;
			vocab_len_ = 50257;

            cuda_input_shape_ = { batch_size_, sequence_length_ };
            cpu_input_shape_ = { cpu_batch_size_, sequence_length_ };
            cpu_output_shape_ = { cpu_batch_size_, sequence_length_, channels_ };
            
            cpu_encoder = std::make_unique<Encoder<int, float, Compute::CpuDevice>>(
                "cpu_encoder", channels_, max_seq_len_, vocab_len_ );

            //cuda_encoder = std::make_unique<Encoder<int, float, Compute::CudaDevice>>(
            //    "cuda_encoder", channels_, max_seq_len_, vocab_len_ );
        }

        std::unique_ptr<Encoder<int, float, Compute::CpuDevice>> cpu_encoder;
        //std::unique_ptr<Encoder<int, float, Compute::CudaDevice>> cuda_encoder;

        size_t batch_size_{ 0 };
        size_t cpu_batch_size_{ 0 };
        size_t sequence_length_{ 0 };
        size_t channels_{ 0 };
        size_t vocab_len_{ 0 };
		size_t max_seq_len_{ 0 };
        std::vector<size_t> cuda_input_shape_;
        std::vector<size_t> cpu_input_shape_;
        std::vector<size_t> cpu_output_shape_;
    };

    TEST_F( EncoderTests, Cpu_TestName ) {
        EXPECT_EQ( cpu_encoder->getName(), "cpu_encoder" );
    }

    TEST_F( EncoderTests, Cpu_parameterCount ) {
        auto num_parameters = /* wte */ (vocab_len_ * channels_) + /* wpe */ (max_seq_len_ * channels_);

        EXPECT_EQ( cpu_encoder->parameterCount(), num_parameters );
    }

    TEST_F( EncoderTests, Cpu_TestForward ) {
        Tensor<int, Compute::CpuMemoryResource> input( cpu_input_shape_ );
        Tensor<float, Compute::CpuMemoryResource> output( cpu_output_shape_ );
        cpu_encoder->forward( input, output );
        EXPECT_EQ( output.size(), cpu_batch_size_ * sequence_length_ * channels_ );
    }

	// TODO: Uncomment when the CUDA encoder is fully implemented
    //TEST_F( EncoderTests, Cuda_TestForward ) {
    //    Tensor<float, Compute::CudaMemoryResource> input( { batch_size_, sequence_length_ } );
    //    Tensor<float, Compute::CudaMemoryResource> output( { batch_size_, sequence_length_, channels_ } );
    //    cuda_encoder->forward( input, output );
    //    EXPECT_EQ( output.size(), batch_size_ * sequence_length_ * channels_ );
    //}

    TEST_F( EncoderTests, Cpu_toString ) {
        /*std::ostringstream oss;
        oss << cpu_encoder;
        std::string output = oss.str();
        EXPECT_NE( output.find( "Encoder: cpu_encoder" ), std::string::npos );*/

        std::string output = cpu_encoder->toString();
        EXPECT_NE( output.find( "Encoder: cpu_encoder" ), std::string::npos );
    }
}