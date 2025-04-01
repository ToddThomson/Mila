#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>

import Mila;

namespace Modules::Tests
{
	using namespace Mila::Dnn;

	// Use a common test data structure with precise type control
	template<typename TInput, typename TPrecision, Compute::DeviceType TDevice>
	struct EncoderTestData {
		std::vector<size_t> input_shape;
		std::vector<size_t> output_shape;
		std::shared_ptr<Encoder<TInput, TPrecision, TDevice>> encoder;
		size_t channels;
		size_t max_seq_len;
		size_t vocab_len;
	};

	class EncoderTests : public ::testing::Test {
	protected:
		void SetUp() override {
			batch_size_ = 64;
			cpu_batch_size_ = 4;
			sequence_length_ = 512;
			max_seq_len_ = 1024;
			channels_ = 768;
			vocab_len_ = 50257;

			// CPU test data (float precision)
			cpu_uint16_data_.input_shape = { cpu_batch_size_, sequence_length_ };
			cpu_uint16_data_.output_shape = { cpu_batch_size_, sequence_length_, channels_ };
			cpu_uint16_data_.channels = channels_;
			cpu_uint16_data_.max_seq_len = max_seq_len_;
			cpu_uint16_data_.vocab_len = vocab_len_;
			cpu_uint16_data_.encoder = std::make_shared<Encoder<uint16_t, float, Compute::DeviceType::Cpu>>(
				"cpu_encoder_uint16", channels_, max_seq_len_, vocab_len_ );

			// CUDA test data (float precision)
			cuda_uint16_data_.input_shape = { batch_size_, sequence_length_ };
			cuda_uint16_data_.output_shape = { batch_size_, sequence_length_, channels_ };
			cuda_uint16_data_.channels = channels_;
			cuda_uint16_data_.max_seq_len = max_seq_len_;
			cuda_uint16_data_.vocab_len = vocab_len_;
			cuda_uint16_data_.encoder = std::make_shared<Encoder<uint16_t, float, Compute::DeviceType::Cuda>>(
				"cuda_encoder_uint16", channels_, max_seq_len_, vocab_len_ );
		}

		size_t batch_size_{ 0 };
		size_t cpu_batch_size_{ 0 };
		size_t sequence_length_{ 0 };
		size_t channels_{ 0 };
		size_t vocab_len_{ 0 };
		size_t max_seq_len_{ 0 };

		// Structured test data
		EncoderTestData<uint16_t, float, Compute::DeviceType::Cpu> cpu_uint16_data_;
		EncoderTestData<uint16_t, float, Compute::DeviceType::Cuda> cuda_uint16_data_;
	};

	// Common test function templates
	template<typename TInput, typename TPrecision, Compute::DeviceType TDevice, typename TMemResource>
	void TestGetName( const EncoderTestData<TInput, TPrecision, TDevice>& data, const std::string& expected_name ) {
		EXPECT_EQ( data.encoder->getName(), expected_name );
	}

	template<typename TInput, typename TPrecision, Compute::DeviceType TDevice, typename TMemResource>
	void TestParameterCount( const EncoderTestData<TInput, TPrecision, TDevice>& data ) {
		auto num_parameters = /* wte */ (data.vocab_len * data.channels) + /* wpe */ (data.max_seq_len * data.channels);
		EXPECT_EQ( data.encoder->parameterCount(), num_parameters );
	}

	template<typename TInput, typename TPrecision, Compute::DeviceType TDevice, typename TMemResource>
	void TestForward( const EncoderTestData<TInput, TPrecision, TDevice>& data ) {
		Tensor<TInput, TMemResource> input( data.input_shape );
		Tensor<TPrecision, TMemResource> output( data.output_shape );
		data.encoder->forward( input, output );
		EXPECT_EQ( output.size(), input.size() * data.channels );
	}

	template<typename TInput, typename TPrecision, Compute::DeviceType TDevice>
	void TestPrint( const EncoderTestData<TInput, TPrecision, TDevice>& data, const std::string& expected_substring ) {
		std::string output = data.encoder->toString();
		EXPECT_NE( output.find( expected_substring ), std::string::npos );
	}

	// CPU Tests with uint16_t precision

	TEST_F( EncoderTests, Cpu_Uint16_TestName ) {
		TestGetName<uint16_t, float, Compute::DeviceType::Cpu, Compute::HostMemoryResource>(
			cpu_uint16_data_, "cpu_encoder_uint16" );
	}

	TEST_F( EncoderTests, Cpu_Uint16_ParameterCount ) {
		TestParameterCount<uint16_t, float, Compute::DeviceType::Cpu, Compute::HostMemoryResource>(
			cpu_uint16_data_ );
	}

	TEST_F( EncoderTests, Cpu_Uint16_TestForward ) {
		TestForward<uint16_t, float, Compute::DeviceType::Cpu, Compute::HostMemoryResource>( cpu_uint16_data_ );
	}

	TEST_F( EncoderTests, Cpu_Uint16_TestPrint ) {
		TestPrint<uint16_t, float, Compute::DeviceType::Cpu>( cpu_uint16_data_, "Encoder: cpu_encoder_uint16" );
	}

	// CUDA Tests with uint16_t precision

	TEST_F( EncoderTests, Cuda_Uint16_TestName ) {
		TestGetName<uint16_t, float, Compute::DeviceType::Cuda, Compute::DeviceMemoryResource>(
			cuda_uint16_data_, "cuda_encoder_uint16" );
	}

	TEST_F( EncoderTests, Cuda_Uint16_ParameterCount ) {
		TestParameterCount<uint16_t, float, Compute::DeviceType::Cuda, Compute::DeviceMemoryResource>(
			cuda_uint16_data_ );
	}

	TEST_F( EncoderTests, Cuda_Uint16_TestForward ) {
		TestForward<uint16_t, float, Compute::DeviceType::Cuda, Compute::DeviceMemoryResource>( cuda_uint16_data_ );
	}

	TEST_F( EncoderTests, Cuda_Uint16_TestPrint ) {
		TestPrint<uint16_t, float, Compute::DeviceType::Cuda>( cuda_uint16_data_, "Encoder: cuda_encoder_uint16" );
	}
}
