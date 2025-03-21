#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>

import Mila;

namespace Modules::Tests
{
    using namespace Mila::Dnn;

    template <class TDevice>
    class EncoderTests : public ::testing::Test {
    protected:
        void SetUp() override {
            if constexpr ( std::is_same_v<TDevice, Compute::CpuDevice> ) {
                batch_size_ = 4;
            }
            else if constexpr ( std::is_same_v<TDevice, Compute::CudaDevice> ) {
                batch_size_ = 64;
            }
            sequence_length_ = 512;
            max_seq_len_ = 1024;
            channels_ = 768;
            vocab_len_ = 50257;
            input_shape_ = { batch_size_, sequence_length_ };
            output_shape_ = { batch_size_, sequence_length_, channels_ };

            encoder_ = std::make_unique<Encoder<int, float, TDevice>>(
                "enc_1", channels_, max_seq_len_, vocab_len_ );
        }

        std::unique_ptr<Encoder<int, float, TDevice>> encoder_;

        size_t batch_size_{ 0 };
        size_t sequence_length_{ 0 };
        size_t channels_{ 0 };
        size_t vocab_len_{ 0 };
        size_t max_seq_len_{ 0 };
        std::vector<size_t> input_shape_;
        std::vector<size_t> output_shape_;
    };

    struct GenerateDeviceName {
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
    };

    using EncoderTypes = ::testing::Types<Compute::CpuDevice, Compute::CudaDevice>;
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
}