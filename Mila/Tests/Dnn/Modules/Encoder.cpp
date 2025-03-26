#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>

import Mila;

namespace Modules::Tests
{
    using namespace Mila::Dnn;

    // Create wrapper types for each device type
    template <Compute::DeviceType TDeviceType>
    struct DeviceTypeWrapper {
        static constexpr Compute::DeviceType value = TDeviceType;
        // Add the MR type that's used in the tests
        using MR = std::conditional_t<TDeviceType == Compute::DeviceType::Cuda, Compute::DeviceMemoryResource, Compute::HostMemoryResource>;
    };

    // Specific type definitions for CPU and CUDA
    using CpuDeviceType = DeviceTypeWrapper<Compute::DeviceType::Cpu>;
    using CudaDeviceType = DeviceTypeWrapper<Compute::DeviceType::Cuda>;

    template <typename TDeviceTypeWrapper>
    class EncoderTests : public ::testing::Test {
    protected:
        void SetUp() override {
            constexpr Compute::DeviceType TDeviceType = TDeviceTypeWrapper::value;

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

            encoder_ = std::make_unique<Encoder<int, float, TDeviceType>>(
                "enc_1", channels_, max_seq_len_, vocab_len_ );
        }

        std::unique_ptr<Encoder<int, float, TDeviceTypeWrapper::value>> encoder_;

        size_t batch_size_{ 0 };
        size_t sequence_length_{ 0 };
        size_t channels_{ 0 };
        size_t vocab_len_{ 0 };
        size_t max_seq_len_{ 0 };
        std::vector<size_t> input_shape_;
        std::vector<size_t> output_shape_;
    };

    /*struct GenerateDeviceName {
        template <typename T>
        static std::string GetName( int ) {
            if constexpr ( std::is_same_v<T, CudaDevice> ) {
                return "::Cuda";
            }
            else if constexpr ( std::is_same_v<T, CpuDevice> ) {
                return "::Cpu";
            }

            return "Unknown";
        }
    };*/

    using EncoderTypes = ::testing::Types<CpuDeviceType, CudaDeviceType>;
    TYPED_TEST_SUITE(EncoderTests, EncoderTypes );

    TYPED_TEST( EncoderTests, getName ) {
        EXPECT_EQ( this->encoder_->getName(), "enc_1" );
    }

    TYPED_TEST( EncoderTests, parameterCount ) {
        auto num_parameters = /* wte */ (this->vocab_len_ * this->channels_) + /* wpe */ (this->max_seq_len_ * this->channels_);

        EXPECT_EQ( this->encoder_->parameterCount(), num_parameters );
    }

    TYPED_TEST( EncoderTests, forward_float ) {
        Tensor<int, typename TypeParam::MR> input( this->input_shape_ );
        Tensor<float, typename TypeParam::MR> output( this->output_shape_ );
        this->encoder_->forward( input, output );
        EXPECT_EQ( output.size(), this->batch_size_ * this->sequence_length_ * this->channels_ );
    }

    TYPED_TEST( EncoderTests, toString ) {
        /*std::ostringstream oss;
        oss << cpu_encoder;
        std::string output = oss.str();
        EXPECT_NE( output.find( "Encoder: cpu_encoder" ), std::string::npos );*/

        std::string output = this->encoder_->toString();
        EXPECT_NE( output.find( "Encoder: enc_1" ), std::string::npos );
    }

    TYPED_TEST( EncoderTests, initialization_check ) {
        // Verify parameter shapes
        auto params = this->encoder_->getParameterTensors(); // Or your method for retrieving tensors
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
        // Initially false
        EXPECT_FALSE( this->encoder_->isTraining() );

        // Turn training on
        this->encoder_->setTraining( true );
        EXPECT_TRUE( this->encoder_->isTraining() );

        // Toggle back off
        this->encoder_->setTraining( false );
        EXPECT_FALSE( this->encoder_->isTraining() );
    }

    //TYPED_TEST( EncoderTests, save_load ) {
    //    // Mock or create your zip archive
    //    mz_zip_archive zip{};
    //    memset( &zip, 0, sizeof( zip ) );

    //    // Initialize (this is just a demonstration; configure as needed)
    //    // Example: mz_zip_writer_init_heap(&zip, 0, 0);

    //    // Save
    //    EXPECT_NO_THROW( this->encoder_->save( zip ) );

    //    // Load
    //    EXPECT_NO_THROW( this->encoder_->load( zip ) );

    //    // Cleanup
    //    // Example: mz_zip_writer_end(&zip);
    //}

    TYPED_TEST( EncoderTests, edge_case_zero_dimensions ) {
        // Build an encoder with zero channels to confirm it doesn't crash
        auto zeroEncoder = std::make_unique<Encoder<int, float, TypeParam::value>>( "enc_zero", /*channels=*/0,
            this->max_seq_len_,
            this->vocab_len_ );
        // Attempt forward
        Tensor<int, typename TypeParam::MR> inputZero( { this->batch_size_, this->sequence_length_ } );
        Tensor<float, typename TypeParam::MR> outputZero( { this->batch_size_, this->sequence_length_, 0 } );
        EXPECT_NO_THROW( zeroEncoder->forward( inputZero, outputZero ) );
        // Check output size correctness
        EXPECT_EQ( outputZero.size(), 0u );
    }
}