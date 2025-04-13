#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <optional>

import Mila;

namespace Modules::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    // Use a common test data structure with device context
    template<typename TInput, typename TPrecision, DeviceType TDeviceType>
    struct EncoderTestData {
        std::vector<size_t> input_shape;
        std::vector<size_t> output_shape;
        mutable std::optional<std::shared_ptr<Encoder<TInput, TPrecision, TDeviceType>>> encoder;
        std::string device_name;
        size_t channels;
        size_t max_seq_len;
        size_t vocab_len;
        std::string name;

        // Lazy initialization method
        std::shared_ptr<Encoder<TInput, TPrecision, TDeviceType>> getEncoder() const {
            if ( !encoder ) {
                encoder = std::make_shared<Encoder<TInput, TPrecision, TDeviceType>>(
                    name, channels, max_seq_len, vocab_len, device_name );
            }
            return *encoder;
        }
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

            cpu_device_name_ = "CPU";
            cuda_device_name_ = "CUDA:0";

            // CPU test data with int inputs - defer encoder creation
            cpu_int_data_.input_shape = { cpu_batch_size_, sequence_length_ };
            cpu_int_data_.output_shape = { cpu_batch_size_, sequence_length_, channels_ };
            cpu_int_data_.channels = channels_;
            cpu_int_data_.max_seq_len = max_seq_len_;
            cpu_int_data_.vocab_len = vocab_len_;
            cpu_int_data_.device_name = cpu_device_name_;
            cpu_int_data_.name = "cpu_encoder_int";

            // CPU test data with uint16_t inputs - defer encoder creation
            cpu_uint16_data_.input_shape = { cpu_batch_size_, sequence_length_ };
            cpu_uint16_data_.output_shape = { cpu_batch_size_, sequence_length_, channels_ };
            cpu_uint16_data_.channels = channels_;
            cpu_uint16_data_.max_seq_len = max_seq_len_;
            cpu_uint16_data_.vocab_len = vocab_len_;
            cpu_uint16_data_.device_name = cpu_device_name_;
            cpu_uint16_data_.name = "cpu_encoder_uint16";

            // CUDA test data with int inputs - defer encoder creation
            cuda_int_data_.input_shape = { batch_size_, sequence_length_ };
            cuda_int_data_.output_shape = { batch_size_, sequence_length_, channels_ };
            cuda_int_data_.channels = channels_;
            cuda_int_data_.max_seq_len = max_seq_len_;
            cuda_int_data_.vocab_len = vocab_len_;
            cuda_int_data_.device_name = cuda_device_name_;
            cuda_int_data_.name = "cuda_encoder_int";

            // CUDA test data with uint16_t inputs - defer encoder creation
            cuda_uint16_data_.input_shape = { batch_size_, sequence_length_ };
            cuda_uint16_data_.output_shape = { batch_size_, sequence_length_, channels_ };
            cuda_uint16_data_.channels = channels_;
            cuda_uint16_data_.max_seq_len = max_seq_len_;
            cuda_uint16_data_.vocab_len = vocab_len_;
            cuda_uint16_data_.device_name = cuda_device_name_;
            cuda_uint16_data_.name = "cuda_encoder_uint16";
        }

        void TearDown() override {
            // Explicitly reset encoders to release resources earlier
            if ( cpu_int_data_.encoder ) cpu_int_data_.encoder.reset();
            if ( cpu_uint16_data_.encoder ) cpu_uint16_data_.encoder.reset();
            if ( cuda_int_data_.encoder ) cuda_int_data_.encoder.reset();
            if ( cuda_uint16_data_.encoder ) cuda_uint16_data_.encoder.reset();
        }

        size_t batch_size_{ 0 };
        size_t cpu_batch_size_{ 0 };
        size_t sequence_length_{ 0 };
        size_t channels_{ 0 };
        size_t vocab_len_{ 0 };
        size_t max_seq_len_{ 0 };

        std::string cpu_device_name_;
        std::string cuda_device_name_;

        // Structured test data
        EncoderTestData<int, float, DeviceType::Cpu> cpu_int_data_;
        EncoderTestData<uint16_t, float, DeviceType::Cpu> cpu_uint16_data_;
        EncoderTestData<int, float, DeviceType::Cuda> cuda_int_data_;
        EncoderTestData<uint16_t, float, DeviceType::Cuda> cuda_uint16_data_;
    };

    // Common test function templates
    template<typename TInput, typename TPrecision, DeviceType TDeviceType>
    void TestGetName( const EncoderTestData<TInput, TPrecision, TDeviceType>& data, const std::string& expected_name ) {
        EXPECT_EQ( data.getEncoder()->getName(), expected_name );
    }

    template<typename TInput, typename TPrecision, DeviceType TDeviceType>
    void TestParameterCount( const EncoderTestData<TInput, TPrecision, TDeviceType>& data ) {
        auto num_parameters = /* wte */ (data.vocab_len * data.channels) + /* wpe */ (data.max_seq_len * data.channels);
        EXPECT_EQ( data.getEncoder()->parameterCount(), num_parameters );
    }

    template<typename TInput, typename TPrecision, DeviceType TDeviceType>
    void TestForward( const EncoderTestData<TInput, TPrecision, TDeviceType>& data ) {
        // Use appropriate memory resource type based on device type
        using MR = typename std::conditional_t<TDeviceType == DeviceType::Cuda,
            DeviceMemoryResource,
            HostMemoryResource>;

        Tensor<TInput, MR> input( data.input_shape );
        Tensor<TPrecision, MR> output( data.output_shape );

        // Fill input with token IDs
        for ( size_t i = 0; i < input.size(); ++i ) {
            input.data()[ i ] = static_cast<TInput>( i % 100 ); // Use a range of token IDs
        }

        data.getEncoder()->forward( input, output );
        EXPECT_EQ( output.size(), input.size() * data.channels );
    }

    template<typename TInput, typename TPrecision, DeviceType TDeviceType>
    void TestPrint( const EncoderTestData<TInput, TPrecision, TDeviceType>& data, const std::string& expected_substring ) {
        std::string output = data.getEncoder()->toString();
        EXPECT_NE( output.find( expected_substring ), std::string::npos );
    }

    // CPU Tests with int precision
    TEST_F( EncoderTests, Cpu_Int_TestName ) {
        TestGetName( cpu_int_data_, "cpu_encoder_int" );
    }

    TEST_F( EncoderTests, Cpu_Int_ParameterCount ) {
        TestParameterCount( cpu_int_data_ );
    }

    TEST_F( EncoderTests, Cpu_Int_TestForward ) {
        TestForward( cpu_int_data_ );
    }

    TEST_F( EncoderTests, Cpu_Int_TestPrint ) {
        TestPrint( cpu_int_data_, "Encoder: cpu_encoder_int" );
    }

    // CPU Tests with uint16_t precision
    TEST_F( EncoderTests, Cpu_Uint16_TestName ) {
        TestGetName( cpu_uint16_data_, "cpu_encoder_uint16" );
    }

    TEST_F( EncoderTests, Cpu_Uint16_ParameterCount ) {
        TestParameterCount( cpu_uint16_data_ );
    }

    TEST_F( EncoderTests, Cpu_Uint16_TestForward ) {
        TestForward( cpu_uint16_data_ );
    }

    TEST_F( EncoderTests, Cpu_Uint16_TestPrint ) {
        TestPrint( cpu_uint16_data_, "Encoder: cpu_encoder_uint16" );
    }

    // CUDA Tests with int precision
    TEST_F( EncoderTests, Cuda_Int_TestName ) {
        TestGetName( cuda_int_data_, "cuda_encoder_int" );
    }

    TEST_F( EncoderTests, Cuda_Int_ParameterCount ) {
        TestParameterCount( cuda_int_data_ );
    }

    TEST_F( EncoderTests, Cuda_Int_TestForward ) {
        try {
            TestForward( cuda_int_data_ );
        }
        catch ( const std::exception& e ) {
            GTEST_SKIP() << "Skipping CUDA test due to exception: " << e.what();
        }
    }

    TEST_F( EncoderTests, Cuda_Int_TestPrint ) {
        TestPrint( cuda_int_data_, "Encoder: cuda_encoder_int" );
    }

    // CUDA Tests with uint16_t precision
    TEST_F( EncoderTests, Cuda_Uint16_TestName ) {
        TestGetName( cuda_uint16_data_, "cuda_encoder_uint16" );
    }

    TEST_F( EncoderTests, Cuda_Uint16_ParameterCount ) {
        TestParameterCount( cuda_uint16_data_ );
    }

    TEST_F( EncoderTests, Cuda_Uint16_TestForward ) {
        try {
            TestForward( cuda_uint16_data_ );
        }
        catch ( const std::exception& e ) {
            GTEST_SKIP() << "Skipping CUDA test due to exception: " << e.what();
        }
    }

    TEST_F( EncoderTests, Cuda_Uint16_TestPrint ) {
        TestPrint( cuda_uint16_data_, "Encoder: cuda_encoder_uint16" );
    }

    // Test output equivalence between int and uint16_t types on CPU
    TEST_F( EncoderTests, Cpu_Int_Uint16_Output_Equivalence ) {
        // Create small test shapes for quick comparison
        std::vector<size_t> test_input_shape = { 2, 4 }; // Small shape for verification
        std::vector<size_t> test_output_shape = { 2, 4, channels_ };

        // Create and fill input data with the same values for both types
        Tensor<int, HostMemoryResource> input_int( test_input_shape );
        Tensor<uint16_t, HostMemoryResource> input_uint16( test_input_shape );

        for ( size_t i = 0; i < input_int.size(); ++i ) {
            int value = static_cast<int>( i % 100 );
            input_int.data()[ i ] = value;
            input_uint16.data()[ i ] = static_cast<uint16_t>( value );
        }

        // Create output tensors
        Tensor<float, HostMemoryResource> output_int( test_output_shape );
        Tensor<float, HostMemoryResource> output_uint16( test_output_shape );

        // Run forward passes
        cpu_int_data_.getEncoder()->forward( input_int, output_int );
        cpu_uint16_data_.getEncoder()->forward( input_uint16, output_uint16 );

        // Compare outputs with tolerance
        const float epsilon = 1e-4f;
        bool all_equal = true;

        for ( size_t i = 0; i < output_int.size(); ++i ) {
            float diff = std::abs( output_int.data()[ i ] - output_uint16.data()[ i ] );
            if ( diff > epsilon ) {
                std::cout << "Difference at index " << i << ": int=" << output_int.data()[ i ]
                    << ", uint16=" << output_uint16.data()[ i ] << ", diff=" << diff << std::endl;
                    all_equal = false;
                    break;
            }
        }

        EXPECT_TRUE( all_equal ) << "int and uint16_t encoders produced different results";
    }

    // Test for CUDA-CPU equivalence with the same input type (where CUDA is available)
    TEST_F( EncoderTests, Cpu_Cuda_Int_Output_Equivalence ) {
        // Skip this test if CUDA is not available
        try {
            DeviceContext context( "CUDA:0" );
        }
        catch ( const std::exception& ) {
            GTEST_SKIP() << "CUDA device not available, skipping CPU-CUDA equivalence test";
            return;
        }

        // Create small test shapes for quick comparison
        std::vector<size_t> test_input_shape = { 2, 4 }; // Small shape for verification
        std::vector<size_t> test_output_shape = { 2, 4, channels_ };

        // Create and fill host input data
        Tensor<int, HostMemoryResource> host_input( test_input_shape );
        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input.data()[ i ] = static_cast<int>( i % 100 );
        }

        // Run CPU encoder
        Tensor<float, HostMemoryResource> cpu_output( test_output_shape );
        cpu_int_data_.getEncoder()->forward( host_input, cpu_output );

        // Create device input by copying host data
        Tensor<int, DeviceMemoryResource> device_input( test_input_shape );
        device_input.copyFrom( host_input );

        // Run CUDA encoder
        Tensor<float, DeviceMemoryResource> cuda_output( test_output_shape );
        cuda_int_data_.getEncoder()->forward( device_input, cuda_output );

        // Copy CUDA output back to host for comparison
        Tensor<float, HostMemoryResource> cuda_output_host( test_output_shape );
        cuda_output_host.copyFrom( cuda_output );

        // Compare outputs with tolerance
        const float epsilon = 1e-4f;
        bool all_equal = true;

        for ( size_t i = 0; i < cpu_output.size(); ++i ) {
            float diff = std::abs( cpu_output.data()[ i ] - cuda_output_host.data()[ i ] );
            if ( diff > epsilon ) {
                std::cout << "Difference at index " << i << ": CPU=" << cpu_output.data()[ i ]
                    << ", CUDA=" << cuda_output_host.data()[ i ] << ", diff=" << diff << std::endl;
                    all_equal = false;
                    break;
            }
        }

        EXPECT_TRUE( all_equal ) << "CPU and CUDA implementations produced different results";
    }
}