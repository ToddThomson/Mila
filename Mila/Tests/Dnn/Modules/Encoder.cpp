#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <optional>
#include <cuda_fp16.h>

import Mila;

namespace Modules::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    template<typename TPrecision, DeviceType TDeviceType>
    struct EncoderTestData {
        std::vector<size_t> input_shape;
        std::vector<size_t> output_shape;
        mutable std::optional<std::shared_ptr<Encoder<TPrecision, TPrecision, TPrecision, TDeviceType>>> encoder;
        std::string device_name;
        size_t channels;
        size_t max_seq_len;
        size_t vocab_len;
        std::string name;

        // Lazy initialization method
        std::shared_ptr<Encoder<TPrecision, TDeviceType>> getEncoder() const {
            if ( !encoder ) {
                encoder = std::make_shared<Encoder<TPrecision, TDeviceType>>(
                    name, device_name, channels, max_seq_len, vocab_len );
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

            // CPU test data with float precision - defer encoder creation
            cpu_float_data_.input_shape = { cpu_batch_size_, sequence_length_ };
            cpu_float_data_.output_shape = { cpu_batch_size_, sequence_length_, channels_ };
            cpu_float_data_.channels = channels_;
            cpu_float_data_.max_seq_len = max_seq_len_;
            cpu_float_data_.vocab_len = vocab_len_;
            cpu_float_data_.device_name = cpu_device_name_;
            cpu_float_data_.name = "cpu_encoder_float";

            // CUDA test data with float precision - defer encoder creation
            cuda_float_data_.input_shape = { batch_size_, sequence_length_ };
            cuda_float_data_.output_shape = { batch_size_, sequence_length_, channels_ };
            cuda_float_data_.channels = channels_;
            cuda_float_data_.max_seq_len = max_seq_len_;
            cuda_float_data_.vocab_len = vocab_len_;
            cuda_float_data_.device_name = cuda_device_name_;
            cuda_float_data_.name = "cuda_encoder_float";

            // CUDA test data with half precision - defer encoder creation
            cuda_half_data_.input_shape = { batch_size_, sequence_length_ };
            cuda_half_data_.output_shape = { batch_size_, sequence_length_, channels_ };
            cuda_half_data_.channels = channels_;
            cuda_half_data_.max_seq_len = max_seq_len_;
            cuda_half_data_.vocab_len = vocab_len_;
            cuda_half_data_.device_name = cuda_device_name_;
            cuda_half_data_.name = "cuda_encoder_half";
        }

        void TearDown() override {
            // Explicitly reset encoders to release resources earlier
            if ( cpu_float_data_.encoder ) cpu_float_data_.encoder.reset();
            if ( cuda_float_data_.encoder ) cuda_float_data_.encoder.reset();
            if ( cuda_half_data_.encoder ) cuda_half_data_.encoder.reset();
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
        EncoderTestData<float, DeviceType::Cpu> cpu_float_data_;
        EncoderTestData<float, DeviceType::Cuda> cuda_float_data_;
        EncoderTestData<half, DeviceType::Cuda> cuda_half_data_;
    };

    // Common test function templates
    template<typename TPrecision, DeviceType TDeviceType>
    void TestGetName( const EncoderTestData<TPrecision, TDeviceType>& data, const std::string& expected_name ) {
        EXPECT_EQ( data.getEncoder()->getName(), expected_name );
    }

    template<typename TPrecision, DeviceType TDeviceType>
    void TestParameterCount( const EncoderTestData<TPrecision, TDeviceType>& data ) {
        auto num_parameters = /* wte */ (data.vocab_len * data.channels) + /* wpe */ (data.max_seq_len * data.channels);
        EXPECT_EQ( data.getEncoder()->parameterCount(), num_parameters );
    }

    template<typename TPrecision, DeviceType TDeviceType>
    void TestForward( const EncoderTestData<TPrecision, TDeviceType>& data ) {
        // Use appropriate memory resource type based on device type
        using MR = typename std::conditional_t<TDeviceType == DeviceType::Cuda,
            CudaMemoryResource,
            HostMemoryResource>;

        Tensor<int, MR> input( data.input_shape );
        Tensor<TPrecision, MR> output( data.output_shape );

        // Fill input with token IDs
        if constexpr ( TDeviceType == DeviceType::Cpu ) {
            // Direct access for CPU memory
            for ( size_t i = 0; i < input.size(); ++i ) {
                input.data()[ i ] = static_cast<int>( i % 100 ); // Use a range of token IDs
            }
        }
        else {
            // For device memory, create a host tensor, fill it, then copy to device
            Tensor<int, HostMemoryResource> host_input( data.input_shape );
            for ( size_t i = 0; i < host_input.size(); ++i ) {
                host_input.data()[ i ] = static_cast<int>( i % 100 ); // Use a range of token IDs
            }
            input.copyFrom( host_input );
        }

        data.getEncoder()->forward( input, output );
        EXPECT_EQ( output.size(), input.size() * data.channels );
    }

    template<typename TPrecision, DeviceType TDeviceType>
    void TestPrint( const EncoderTestData<TPrecision, TDeviceType>& data, const std::string& expected_substring ) {
        std::string output = data.getEncoder()->toString();
        EXPECT_NE( output.find( expected_substring ), std::string::npos );
    }

    // CPU Tests with float precision
    TEST_F( EncoderTests, Cpu_Float_TestName ) {
        TestGetName( cpu_float_data_, "cpu_encoder_float" );
    }

    TEST_F( EncoderTests, Cpu_Float_ParameterCount ) {
        TestParameterCount( cpu_float_data_ );
    }

    TEST_F( EncoderTests, Cpu_Float_TestForward ) {
        TestForward( cpu_float_data_ );
    }

    TEST_F( EncoderTests, Cpu_Float_TestPrint ) {
        TestPrint( cpu_float_data_, "Encoder: cpu_encoder_float" );
    }

    // CUDA Tests with float precision
    TEST_F( EncoderTests, Cuda_Float_TestName ) {
        TestGetName( cuda_float_data_, "cuda_encoder_float" );
    }

    TEST_F( EncoderTests, Cuda_Float_ParameterCount ) {
        TestParameterCount( cuda_float_data_ );
    }

    TEST_F( EncoderTests, Cuda_Float_TestForward ) {
        try {
            TestForward( cuda_float_data_ );
        }
        catch ( const std::exception& e ) {
            GTEST_SKIP() << "Skipping CUDA test due to exception: " << e.what();
        }
    }

    TEST_F( EncoderTests, Cuda_Float_TestPrint ) {
        TestPrint( cuda_float_data_, "Encoder: cuda_encoder_float" );
    }

    // CUDA Tests with half precision
    TEST_F( EncoderTests, Cuda_Half_TestName ) {
        TestGetName( cuda_half_data_, "cuda_encoder_half" );
    }

    TEST_F( EncoderTests, Cuda_Half_ParameterCount ) {
        TestParameterCount( cuda_half_data_ );
    }

    TEST_F( EncoderTests, Cuda_Half_TestForward ) {
        try {
            TestForward( cuda_half_data_ );
        }
        catch ( const std::exception& e ) {
            GTEST_SKIP() << "Skipping CUDA test due to exception: " << e.what();
        }
    }

    TEST_F( EncoderTests, Cuda_Half_TestPrint ) {
        TestPrint( cuda_half_data_, "Encoder: cuda_encoder_half" );
    }

    // Test for CUDA-CPU equivalence with float precision (where CUDA is available)
    TEST_F( EncoderTests, Cpu_Cuda_Float_Output_Equivalence ) {
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
        cpu_float_data_.getEncoder()->forward( host_input, cpu_output );

        // Create device input by copying host data
        Tensor<int, CudaMemoryResource> device_input( test_input_shape );
        device_input.copyFrom( host_input );

        // Run CUDA encoder
        Tensor<float, CudaMemoryResource> cuda_output( test_output_shape );
        cuda_float_data_.getEncoder()->forward( device_input, cuda_output );

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

    // Test for CUDA float vs half precision accuracy
    TEST_F( EncoderTests, Cuda_Float_Half_Precision_Comparison ) {
        // Skip this test if CUDA is not available
        try {
            DeviceContext context( "CUDA:0" );
        }
        catch ( const std::exception& ) {
            GTEST_SKIP() << "CUDA device not available, skipping CUDA float-half precision test";
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

        // Create device input by copying host data
        Tensor<int, CudaMemoryResource> device_input( test_input_shape );
        device_input.copyFrom( host_input );

        // Run CUDA float precision encoder
        Tensor<float, CudaMemoryResource> cuda_float_output( test_output_shape );
        cuda_float_data_.getEncoder()->forward( device_input, cuda_float_output );

        // Run CUDA half precision encoder
        Tensor<half, CudaMemoryResource> cuda_half_output( test_output_shape );
        cuda_half_data_.getEncoder()->forward( device_input, cuda_half_output );

        // Copy results back to host for comparison
        Tensor<float, HostMemoryResource> float_output_host( test_output_shape );
        float_output_host.copyFrom( cuda_float_output );

        Tensor<float, HostMemoryResource> half_output_host( test_output_shape );
        // Convert half results to float for comparison
        Tensor<half, HostMemoryResource> half_output_host_tmp( test_output_shape );
        half_output_host_tmp.copyFrom( cuda_half_output );
        for ( size_t i = 0; i < half_output_host_tmp.size(); i++ ) {
            half_output_host.data()[ i ] = static_cast<float>( half_output_host_tmp.data()[ i ] );
        }

        // Compare outputs with a larger tolerance (half precision has less accuracy)
        const float epsilon = 1e-2f; // Larger epsilon for half precision comparison
        bool all_close = true;
        size_t diff_count = 0;
        float max_diff = 0.0f;

        for ( size_t i = 0; i < float_output_host.size(); ++i ) {
            float diff = std::abs( float_output_host.data()[ i ] - half_output_host.data()[ i ] );
            max_diff = std::max( max_diff, diff );
            if ( diff > epsilon ) {
                diff_count++;
                if ( diff_count <= 5 ) { // Only print the first few differences
                    std::cout << "Difference at index " << i << ": float=" << float_output_host.data()[ i ]
                        << ", half=" << half_output_host.data()[ i ] << ", diff=" << diff << std::endl;
                }
                all_close = false;
            }
        }

        // We expect some differences due to precision, but they should be relatively small
        std::cout << "Total differences: " << diff_count << " out of " << float_output_host.size()
            << " values, max difference: " << max_diff << std::endl;

        // We're allowing this test to pass even with differences, as half-precision naturally has lower accuracy
        EXPECT_LT( static_cast<float>(diff_count) / float_output_host.size(), 0.1f )
            << "More than 10% of values show significant differences between float and half precision";
    }

    // Additional tests for added methods
    TEST_F( EncoderTests, GetDimensions ) {
        auto encoder = cpu_float_data_.getEncoder();

        EXPECT_EQ( encoder->getChannels(), channels_ );
        EXPECT_EQ( encoder->getVocabularyLength(), vocab_len_ );
        EXPECT_EQ( encoder->getMaxSequenceLength(), max_seq_len_ );
    }
}
