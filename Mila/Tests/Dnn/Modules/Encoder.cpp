#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>

import Mila;

namespace Modules::Tests
{
	using namespace Mila::Dnn;

	// Combined template parameters for both device type and input type
	template <Compute::DeviceType TDeviceType, typename TInput>
	struct TestParams {
		static constexpr Compute::DeviceType device_type = TDeviceType;
		using InputType = TInput;
		using MR = std::conditional_t<TDeviceType == Compute::DeviceType::Cuda,
			Compute::DeviceMemoryResource,
			Compute::HostMemoryResource>;
	};

	// Define the combinations of device types and input types we want to test
	using CpuUint16 = TestParams<Compute::DeviceType::Cpu, uint16_t>;
	using CpuInt = TestParams<Compute::DeviceType::Cpu, int>;
	using CudaUint16 = TestParams<Compute::DeviceType::Cuda, uint16_t>;
	using CudaInt = TestParams<Compute::DeviceType::Cuda, int>;

	template <typename TParams>
	class EncoderTests : public ::testing::Test {
	protected:
		void SetUp() override {
			constexpr Compute::DeviceType TDeviceType = TParams::device_type;
			using TInput = typename TParams::InputType;

			if constexpr ( TDeviceType == DeviceType::Cpu ) {
				batch_size_ = 4;
			}
			else if constexpr ( TDeviceType == DeviceType::Cuda ) {
				batch_size_ = 64;
			}
			sequence_length_ = 512;
			max_seq_len_ = 1024;
			channels_ = 768;
			vocab_len_ = 50257;
			input_shape_ = { batch_size_, sequence_length_ };
			output_shape_ = { batch_size_, sequence_length_, channels_ };

			encoder_ = std::make_unique<Encoder<TInput, float, TDeviceType>>(
				"enc_1", channels_, max_seq_len_, vocab_len_ );
		}

		// Use template parameter for TInput
		std::unique_ptr<Encoder<typename TParams::InputType, float, TParams::device_type>> encoder_;

		size_t batch_size_{ 0 };
		size_t sequence_length_{ 0 };
		size_t channels_{ 0 };
		size_t vocab_len_{ 0 };
		size_t max_seq_len_{ 0 };
		std::vector<size_t> input_shape_;
		std::vector<size_t> output_shape_;
	};

	// Define the test suite with all configurations
	using EncoderTestConfigs = ::testing::Types<CpuUint16, CpuInt, CudaUint16, CudaInt>;
	TYPED_TEST_SUITE( EncoderTests, EncoderTestConfigs );

	TYPED_TEST( EncoderTests, getName ) {
		EXPECT_EQ( this->encoder_->getName(), "enc_1" );
	}

	TYPED_TEST( EncoderTests, parameterCount ) {
		auto num_parameters = /* wte */ (this->vocab_len_ * this->channels_) + /* wpe */ (this->max_seq_len_ * this->channels_);
		EXPECT_EQ( this->encoder_->parameterCount(), num_parameters );
	}

	TYPED_TEST( EncoderTests, forward ) {
		using TInput = typename TypeParam::InputType;
		Tensor<TInput, typename TypeParam::MR> input( this->input_shape_ );
		Tensor<float, typename TypeParam::MR> output( this->output_shape_ );
		this->encoder_->forward( input, output );
		EXPECT_EQ( output.size(), this->batch_size_ * this->sequence_length_ * this->channels_ );
	}

	TYPED_TEST( EncoderTests, toString ) {
		std::string output = this->encoder_->toString();
		EXPECT_NE( output.find( "Encoder: enc_1" ), std::string::npos );
	}

	TYPED_TEST( EncoderTests, initialization_check ) {
		// Verify parameter shapes
		auto params = this->encoder_->getParameterTensors();
		ASSERT_EQ( params.size(), 2u ); // Should have wte_ and wpe_
		auto wteShape = params[ "wte" ]->shape();
		auto wpeShape = params[ "wpe" ]->shape();

		EXPECT_EQ( wteShape.size(), 2u );
		EXPECT_EQ( wteShape[ 0 ], this->vocab_len_ );
		EXPECT_EQ( wteShape[ 1 ], this->channels_ );

		EXPECT_EQ( wpeShape.size(), 2u );
		EXPECT_EQ( wpeShape[ 0 ], this->max_seq_len_ );
		EXPECT_EQ( wpeShape[ 1 ], this->channels_ );
	}

	TYPED_TEST( EncoderTests, training_mode_toggle ) {
		EXPECT_FALSE( this->encoder_->isTraining() );
		this->encoder_->setTraining( true );
		EXPECT_TRUE( this->encoder_->isTraining() );
		this->encoder_->setTraining( false );
		EXPECT_FALSE( this->encoder_->isTraining() );
	}

	TYPED_TEST( EncoderTests, edge_case_zero_dimensions ) {
		using TInput = typename TypeParam::InputType;
		// Build an encoder with zero channels
		auto zeroEncoder = std::make_unique<Encoder<TInput, float, TypeParam::device_type>>(
			"enc_zero", /*channels=*/0,
			this->max_seq_len_,
			this->vocab_len_ );

		// Attempt forward
		Tensor<TInput, typename TypeParam::MR> inputZero( { this->batch_size_, this->sequence_length_ } );
		Tensor<float, typename TypeParam::MR> outputZero( { this->batch_size_, this->sequence_length_, 0 } );
		EXPECT_NO_THROW( zeroEncoder->forward( inputZero, outputZero ) );
		EXPECT_EQ( outputZero.size(), 0u );
	}
}