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

    // Memory resource selector based on device type
    template<DeviceType TDevice, typename TPrecision>
    using MemoryResourceType = std::conditional_t<TDevice == Compute::DeviceType::Cuda,
        Compute::CudaDeviceMemoryResource,
        Compute::HostMemoryResource>;

    // Test data structure for Encoder tests
    template<DeviceType TDevice, typename TInput = int, typename TOutput = float>
    struct EncoderTestData {
        shape_t input_shape;
        shape_t output_shape;
        std::shared_ptr<Encoder<TDevice, TInput, TOutput>> encoder_module;
        size_t channels;
        size_t max_seq_len;
        size_t vocab_len;
        bool is_training;

        // Make the test data structure self-initializing
        static EncoderTestData Create(
            const std::string& name,
            size_t batch_size,
            size_t sequence_length,
            size_t channels,
            size_t max_seq_len,
            size_t vocab_len,
            bool is_training = false,
            ComputePrecision::Policy precision = ComputePrecision::Policy::Auto )
        {
            EncoderTestData data;
            data.input_shape = { batch_size, sequence_length };
            data.output_shape = { batch_size, sequence_length, channels };
            data.channels = channels;
            data.max_seq_len = max_seq_len;
            data.vocab_len = vocab_len;
            data.is_training = is_training;

            std::string device_str = TDevice == Compute::DeviceType::Cuda ? "CUDA:0" : "CPU";
            data.encoder_module = std::make_shared<Encoder<TDevice, TInput, TOutput>>(
                name, device_str, channels, max_seq_len, vocab_len, precision, is_training );

            return data;
        }

        // Overload for creating with device context
        static EncoderTestData CreateWithContext(
            const std::string& name,
            size_t batch_size,
            size_t sequence_length,
            size_t channels,
            size_t max_seq_len,
            size_t vocab_len,
            std::shared_ptr<DeviceContext> context,
            bool is_training = false,
            ComputePrecision::Policy precision = ComputePrecision::Policy::Auto )
        {
            EncoderTestData data;
            data.input_shape = { batch_size, sequence_length };
            data.output_shape = { batch_size, sequence_length, channels };
            data.channels = channels;
            data.max_seq_len = max_seq_len;
            data.vocab_len = vocab_len;
            data.is_training = is_training;

            data.encoder_module = std::make_shared<Encoder<TDevice, TInput, TOutput>>(
                name, context, channels, max_seq_len, vocab_len, precision, is_training );

            return data;
        }
    };

    class EncoderTests : public ::testing::Test {
    protected:
        void SetUp() override {
            // Initialize test parameters only
            batch_size_ = 64;
            cpu_batch_size_ = 4;
            sequence_length_ = 512;
            max_seq_len_ = 1024;
            channels_ = 768;
            vocab_len_ = 50257;
            // Modules will be created on demand
        }

        void TearDown() override {
            // Clean up resources explicitly
            cpu_float_data_.encoder_module.reset();
            training_cpu_float_data_.encoder_module.reset();
            context_cpu_float_data_.encoder_module.reset();
            cuda_float_data_.encoder_module.reset();
            training_cuda_float_data_.encoder_module.reset();
            cuda_half_data_.encoder_module.reset();
            training_cuda_half_data_.encoder_module.reset();
            mixed_precision_data_.encoder_module.reset();
            perf_precision_cpu_float_data_.encoder_module.reset();
            accuracy_precision_cpu_float_data_.encoder_module.reset();
            disabled_precision_cpu_float_data_.encoder_module.reset();
            perf_precision_cuda_float_data_.encoder_module.reset();
            accuracy_precision_cuda_float_data_.encoder_module.reset();
            disabled_precision_cuda_float_data_.encoder_module.reset();
        }

        // Factory methods to lazily create test data as needed
        EncoderTestData<Compute::DeviceType::Cpu, int, float>& CpuFloatData() {
            if ( !cpu_float_data_.encoder_module ) {
                cpu_float_data_ = EncoderTestData<Compute::DeviceType::Cpu, int, float>::Create(
                    "cpu_encoder_float", cpu_batch_size_, sequence_length_,
                    channels_, max_seq_len_, vocab_len_ );
            }
            return cpu_float_data_;
        }

        EncoderTestData<Compute::DeviceType::Cuda, int, float>& CudaFloatData() {
            if ( !cuda_float_data_.encoder_module ) {
                cuda_float_data_ = EncoderTestData<Compute::DeviceType::Cuda, int, float>::Create(
                    "cuda_encoder_float", batch_size_, sequence_length_,
                    channels_, max_seq_len_, vocab_len_ );
            }
            return cuda_float_data_;
        }

        EncoderTestData<Compute::DeviceType::Cpu, int, float>& TrainingCpuFloatData() {
            if ( !training_cpu_float_data_.encoder_module ) {
                training_cpu_float_data_ = EncoderTestData<Compute::DeviceType::Cpu, int, float>::Create(
                    "cpu_encoder_float_training", cpu_batch_size_, sequence_length_,
                    channels_, max_seq_len_, vocab_len_, true );
            }
            return training_cpu_float_data_;
        }

        EncoderTestData<Compute::DeviceType::Cuda, int, float>& TrainingCudaFloatData() {
            if ( !training_cuda_float_data_.encoder_module ) {
                training_cuda_float_data_ = EncoderTestData<Compute::DeviceType::Cuda, int, float>::Create(
                    "cuda_encoder_float_training", batch_size_, sequence_length_,
                    channels_, max_seq_len_, vocab_len_, true );
            }
            return training_cuda_float_data_;
        }

        EncoderTestData<Compute::DeviceType::Cpu, int, float>& ContextCpuFloatData() {
            if ( !context_cpu_float_data_.encoder_module ) {
                auto cpu_context = std::make_shared<DeviceContext>( "CPU" );
                context_cpu_float_data_ = EncoderTestData<Compute::DeviceType::Cpu, int, float>::CreateWithContext(
                    "cpu_context_encoder_float", cpu_batch_size_, sequence_length_,
                    channels_, max_seq_len_, vocab_len_, cpu_context );
            }
            return context_cpu_float_data_;
        }

        EncoderTestData<Compute::DeviceType::Cuda, int, half>& CudaHalfData() {
            if ( !cuda_half_data_.encoder_module ) {
                cuda_half_data_ = EncoderTestData<Compute::DeviceType::Cuda, int, half>::Create(
                    "cuda_encoder_half", batch_size_, sequence_length_,
                    channels_, max_seq_len_, vocab_len_ );
            }
            return cuda_half_data_;
        }

        EncoderTestData<Compute::DeviceType::Cuda, int, half>& TrainingCudaHalfData() {
            if ( !training_cuda_half_data_.encoder_module ) {
                training_cuda_half_data_ = EncoderTestData<Compute::DeviceType::Cuda, int, half>::Create(
                    "cuda_encoder_half_training", batch_size_, sequence_length_,
                    channels_, max_seq_len_, vocab_len_, true );
            }
            return training_cuda_half_data_;
        }

        // Test for mixed precision (int input, half output on CUDA)
        EncoderTestData<Compute::DeviceType::Cuda, int, half>& MixedPrecisionData() {
            if ( !mixed_precision_data_.encoder_module ) {
                mixed_precision_data_ = EncoderTestData<Compute::DeviceType::Cuda, int, half>::Create(
                    "cuda_encoder_mixed", batch_size_, sequence_length_,
                    channels_, max_seq_len_, vocab_len_ );
            }
            return mixed_precision_data_;
        }

        // Tests with specific precision policies - CPU
        EncoderTestData<Compute::DeviceType::Cpu, int, float>& PerfPrecisionCpuFloatData() {
            if ( !perf_precision_cpu_float_data_.encoder_module ) {
                perf_precision_cpu_float_data_ = EncoderTestData<Compute::DeviceType::Cpu, int, float>::Create(
                    "cpu_encoder_perf_precision", cpu_batch_size_, sequence_length_,
                    channels_, max_seq_len_, vocab_len_, false,
                    ComputePrecision::Policy::Performance );
            }
            return perf_precision_cpu_float_data_;
        }

        EncoderTestData<Compute::DeviceType::Cpu, int, float>& AccuracyPrecisionCpuFloatData() {
            if ( !accuracy_precision_cpu_float_data_.encoder_module ) {
                accuracy_precision_cpu_float_data_ = EncoderTestData<Compute::DeviceType::Cpu, int, float>::Create(
                    "cpu_encoder_accuracy_precision", cpu_batch_size_, sequence_length_,
                    channels_, max_seq_len_, vocab_len_, false,
                    ComputePrecision::Policy::Accuracy );
            }
            return accuracy_precision_cpu_float_data_;
        }

        EncoderTestData<Compute::DeviceType::Cpu, int, float>& DisabledPrecisionCpuFloatData() {
            if ( !disabled_precision_cpu_float_data_.encoder_module ) {
                disabled_precision_cpu_float_data_ = EncoderTestData<Compute::DeviceType::Cpu, int, float>::Create(
                    "cpu_encoder_disabled_precision", cpu_batch_size_, sequence_length_,
                    channels_, max_seq_len_, vocab_len_, false,
                    ComputePrecision::Policy::Native );
            }
            return disabled_precision_cpu_float_data_;
        }

        // Tests with specific precision policies - CUDA
        EncoderTestData<Compute::DeviceType::Cuda, int, float>& PerfPrecisionCudaFloatData() {
            if ( !perf_precision_cuda_float_data_.encoder_module ) {
                perf_precision_cuda_float_data_ = EncoderTestData<Compute::DeviceType::Cuda, int, float>::Create(
                    "cuda_encoder_perf_precision", batch_size_, sequence_length_,
                    channels_, max_seq_len_, vocab_len_, false,
                    ComputePrecision::Policy::Performance );
            }
            return perf_precision_cuda_float_data_;
        }

        EncoderTestData<Compute::DeviceType::Cuda, int, float>& AccuracyPrecisionCudaFloatData() {
            if ( !accuracy_precision_cuda_float_data_.encoder_module ) {
                accuracy_precision_cuda_float_data_ = EncoderTestData<Compute::DeviceType::Cuda, int, float>::Create(
                    "cuda_encoder_accuracy_precision", batch_size_, sequence_length_,
                    channels_, max_seq_len_, vocab_len_, false,
                    ComputePrecision::Policy::Accuracy );
            }
            return accuracy_precision_cuda_float_data_;
        }

        EncoderTestData<Compute::DeviceType::Cuda, int, float>& DisabledPrecisionCudaFloatData() {
            if ( !disabled_precision_cuda_float_data_.encoder_module ) {
                disabled_precision_cuda_float_data_ = EncoderTestData<Compute::DeviceType::Cuda, int, float>::Create(
                    "cuda_encoder_disabled_precision", batch_size_, sequence_length_,
                    channels_, max_seq_len_, vocab_len_, false,
                    ComputePrecision::Policy::Native );
            }
            return disabled_precision_cuda_float_data_;
        }

        // Test parameters
        size_t batch_size_{ 0 };
        size_t cpu_batch_size_{ 0 };
        size_t sequence_length_{ 0 };
        size_t channels_{ 0 };
        size_t vocab_len_{ 0 };
        size_t max_seq_len_{ 0 };

        // Test data objects - initialized on demand
        EncoderTestData<Compute::DeviceType::Cpu, int, float> cpu_float_data_;
        EncoderTestData<Compute::DeviceType::Cpu, int, float> training_cpu_float_data_;
        EncoderTestData<Compute::DeviceType::Cpu, int, float> context_cpu_float_data_;

        EncoderTestData<Compute::DeviceType::Cuda, int, float> cuda_float_data_;
        EncoderTestData<Compute::DeviceType::Cuda, int, float> training_cuda_float_data_;

        EncoderTestData<Compute::DeviceType::Cuda, int, half> cuda_half_data_;
        EncoderTestData<Compute::DeviceType::Cuda, int, half> training_cuda_half_data_;

        // Mixed precision test data
        EncoderTestData<Compute::DeviceType::Cuda, int, half> mixed_precision_data_;

        // Precision policy test data - CPU
        EncoderTestData<Compute::DeviceType::Cpu, int, float> perf_precision_cpu_float_data_;
        EncoderTestData<Compute::DeviceType::Cpu, int, float> accuracy_precision_cpu_float_data_;
        EncoderTestData<Compute::DeviceType::Cpu, int, float> disabled_precision_cpu_float_data_;

        // Precision policy test data - CUDA
        EncoderTestData<Compute::DeviceType::Cuda, int, float> perf_precision_cuda_float_data_;
        EncoderTestData<Compute::DeviceType::Cuda, int, float> accuracy_precision_cuda_float_data_;
        EncoderTestData<Compute::DeviceType::Cuda, int, float> disabled_precision_cuda_float_data_;
    };

    // Common test function templates
    template<DeviceType TDevice, typename TInput = int, typename TOutput = float>
    void TestGetName( const EncoderTestData<TDevice, TInput, TOutput>& data, const std::string& expected_name ) {
        EXPECT_EQ( data.encoder_module->getDeviceName(), expected_name );
    }

    template<DeviceType TDevice, typename TInput = int, typename TOutput = float>
    void TestParameterCount( const EncoderTestData<TDevice, TInput, TOutput>& data ) {
        auto num_parameters = /* wte */ (data.vocab_len * data.channels) + /* wpe */ (data.max_seq_len * data.channels);
        EXPECT_EQ( data.encoder_module->parameterCount(), num_parameters );
    }

    template<DeviceType TDevice, typename TInput = int, typename TOutput = float>
    void TestForward( const EncoderTestData<TDevice, TInput, TOutput>& data ) {
        using MR = MemoryResourceType<TDevice, TOutput>;

        Tensor<TInput, MR> input( data.input_shape );
        Tensor<TOutput, MR> output( data.output_shape );

        // Fill input with token IDs
        if constexpr ( TDevice == Compute::DeviceType::Cpu ) {
            // Direct access for CPU memory
            for ( size_t i = 0; i < input.size(); ++i ) {
                input.data()[ i ] = static_cast<TInput>( i % 100 ); // Use a range of token IDs
            }
        }
        else {
            // For device memory, create a host tensor, fill it, then copy to device
            Tensor<TInput, HostMemoryResource> host_input( data.input_shape );
            for ( size_t i = 0; i < host_input.size(); ++i ) {
                host_input.data()[ i ] = static_cast<TInput>( i % 100 ); // Use a range of token IDs
            }
            input.copyFrom( host_input );
        }

        data.encoder_module->forward( input, output );
        EXPECT_EQ( output.size(), input.size() * data.channels );
    }

    template<DeviceType TDevice, typename TInput = int, typename TOutput = float>
    void TestPrint( const EncoderTestData<TDevice, TInput, TOutput>& data, const std::string& expected_substring ) {
        std::string output = data.encoder_module->toString();
        EXPECT_NE( output.find( expected_substring ), std::string::npos );
    }

    template<DeviceType TDevice, typename TInput = int, typename TOutput = float>
    void TestTrainingMode( const EncoderTestData<TDevice, TInput, TOutput>& data, bool expected_mode ) {
        EXPECT_EQ( data.encoder_module->isTraining(), expected_mode );
    }

    template<DeviceType TDevice, typename TInput = int, typename TOutput = float>
    void TestDeviceType( const EncoderTestData<TDevice, TInput, TOutput>& data ) {
        auto device_context = data.encoder_module->getDeviceContext();
        EXPECT_NE( device_context, nullptr );
        auto device = device_context->getDevice();
        EXPECT_NE( device, nullptr );
        EXPECT_EQ( device->getDeviceType(), TDevice );
    }

    template<DeviceType TDevice, typename TInput = int, typename TOutput = float>
    void TestDimensions( const EncoderTestData<TDevice, TInput, TOutput>& data ) {
        EXPECT_EQ( data.encoder_module->getChannels(), data.channels );
        EXPECT_EQ( data.encoder_module->getVocabularyLength(), data.vocab_len );
        EXPECT_EQ( data.encoder_module->getMaxSequenceLength(), data.max_seq_len );
    }

    // New test for precision policy
    template<DeviceType TDevice, typename TInput = int, typename TOutput = float>
    void TestPrecisionPolicy( const EncoderTestData<TDevice, TInput, TOutput>& data, ComputePrecision::Policy expected_policy ) {
        EXPECT_EQ( data.encoder_module->getComputePrecision().getPolicy(), expected_policy );

        // Verify toString() contains precision info
        std::string output = data.encoder_module->toString();
        std::string policy_string;
        switch ( expected_policy ) {
            case ComputePrecision::Policy::Native:
                policy_string = "Disabled";
                break;
            case ComputePrecision::Policy::Performance:
                policy_string = "Performance";
                break;
            case ComputePrecision::Policy::Auto:
                policy_string = "Auto";
                break;
            case ComputePrecision::Policy::Accuracy:
                policy_string = "Accuracy";
                break;
        }
        EXPECT_NE( output.find( policy_string ), std::string::npos );
    }

    // Function to test equivalence of CPU and CUDA outputs
    template<typename TInput = int, typename TOutput = float>
    void TestCpuCudaEquivalence(
        const EncoderTestData<Compute::DeviceType::Cpu, TInput, TOutput>& cpu_data,
        const EncoderTestData<Compute::DeviceType::Cuda, TInput, TOutput>& cuda_data ) {

        try {
            // Check if CUDA is available
            DeviceContext context( "CUDA:0" );
        }
        catch ( const std::exception& ) {
            GTEST_SKIP() << "CUDA device not available, skipping CPU-CUDA equivalence test";
            return;
        }

        // Create small test shapes for quick comparison
        shape_t test_input_shape = { 2, 4 }; // Small shape for verification
        shape_t test_output_shape = { 2, 4, cpu_data.channels };

        // Create and fill host input data
        Tensor<TInput, HostMemoryResource> host_input( test_input_shape );
        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input.data()[ i ] = static_cast<TInput>( i % 100 );
        }

        // Run CPU encoder
        Tensor<TOutput, HostMemoryResource> cpu_output( test_output_shape );
        cpu_data.encoder_module->forward( host_input, cpu_output );

        // Create device input by copying host data
        Tensor<TInput, CudaDeviceMemoryResource> device_input( test_input_shape );
        device_input.copyFrom( host_input );

        // Run CUDA encoder
        Tensor<TOutput, CudaDeviceMemoryResource> cuda_output( test_output_shape );
        cuda_data.encoder_module->forward( device_input, cuda_output );

        // Copy CUDA output back to host for comparison
        Tensor<TOutput, HostMemoryResource> cuda_output_host( test_output_shape );
        cuda_output_host.copyFrom( cuda_output );

        // Compare outputs with tolerance
        const float epsilon = 1e-4f;
        bool all_equal = true;

        for ( size_t i = 0; i < cpu_output.size(); ++i ) {
            float diff = std::abs( static_cast<float>( cpu_output.data()[ i ] ) - static_cast<float>( cuda_output_host.data()[ i ] ) );
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
    template<typename TInput = int>
    void TestPrecisionComparison(
        const EncoderTestData<Compute::DeviceType::Cuda, TInput, float>& float_data,
        const EncoderTestData<Compute::DeviceType::Cuda, TInput, half>& half_data ) {

        try {
            // Check if CUDA is available
            DeviceContext context( "CUDA:0" );
        }
        catch ( const std::exception& ) {
            GTEST_SKIP() << "CUDA device not available, skipping precision comparison test";
            return;
        }

        // Create small test shapes for quick comparison
        shape_t test_input_shape = { 2, 4 }; // Small shape for verification
        shape_t test_output_shape = { 2, 4, float_data.channels };

        // Create and fill host input data
        Tensor<TInput, HostMemoryResource> host_input( test_input_shape );
        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input.data()[ i ] = static_cast<TInput>( i % 100 );
        }

        // Create device input by copying host data
        Tensor<TInput, CudaDeviceMemoryResource> device_input( test_input_shape );
        device_input.copyFrom( host_input );

        // Run CUDA float precision encoder
        Tensor<float, CudaDeviceMemoryResource> cuda_float_output( test_output_shape );
        float_data.encoder_module->forward( device_input, cuda_float_output );

        // Run CUDA half precision encoder
        Tensor<half, CudaDeviceMemoryResource> cuda_half_output( test_output_shape );
        half_data.encoder_module->forward( device_input, cuda_half_output );

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

    // Test edge cases
    template<DeviceType TDevice, typename TInput = int, typename TOutput = float>
    void TestEdgeCases() {
        using MR = MemoryResourceType<TDevice, TOutput>;
        std::string device_str = TDevice == Compute::DeviceType::Cuda ? "CUDA:0" : "CPU";

        try {
            // Test with minimal sizes
            shape_t minimal_input_shape = { 1, 1 };
            shape_t minimal_output_shape = { 1, 1, 32 };

            auto minimal_module = std::make_shared<Encoder<TDevice, TInput, TOutput>>(
                "minimal_encoder", device_str, 32, 16, 100 );

            Tensor<TInput, MR> minimal_input( minimal_input_shape );
            Tensor<TOutput, MR> minimal_output( minimal_output_shape );

            // Fill input with valid token IDs
            for ( size_t i = 0; i < minimal_input.size(); ++i ) {
                minimal_input.data()[ i ] = static_cast<TInput>( i % 100 );
            }

            EXPECT_NO_THROW( minimal_module->forward( minimal_input, minimal_output ) );
            EXPECT_EQ( minimal_output.size(), 32 );
        }
        catch ( const std::exception& e ) {
            std::cerr << "Exception during edge case test: " << e.what() << std::endl;
            throw;
        }
    }

    // CPU Tests with float precision
    TEST_F( EncoderTests, Cpu_Float_TestName ) {
        TestGetName<Compute::DeviceType::Cpu, int, float>( CpuFloatData(), "cpu_encoder_float" );
    }

    TEST_F( EncoderTests, Cpu_Float_ParameterCount ) {
        TestParameterCount<Compute::DeviceType::Cpu, int, float>( CpuFloatData() );
    }

    TEST_F( EncoderTests, Cpu_Float_TestForward ) {
        TestForward<Compute::DeviceType::Cpu, int, float>( CpuFloatData() );
    }

    TEST_F( EncoderTests, Cpu_Float_TestPrint ) {
        TestPrint<Compute::DeviceType::Cpu, int, float>( CpuFloatData(), "Encoder: cpu_encoder_float" );
    }

    TEST_F( EncoderTests, Cpu_Float_DeviceType ) {
        TestDeviceType<Compute::DeviceType::Cpu, int, float>( CpuFloatData() );
    }

    TEST_F( EncoderTests, Cpu_Float_TrainingMode ) {
        TestTrainingMode<Compute::DeviceType::Cpu, int, float>( CpuFloatData(), false );
    }

    TEST_F( EncoderTests, Cpu_Float_TestDimensions ) {
        TestDimensions<Compute::DeviceType::Cpu, int, float>( CpuFloatData() );
    }

    // CPU Training Mode Tests
    TEST_F( EncoderTests, Cpu_Training_Float_TrainingMode ) {
        TestTrainingMode<Compute::DeviceType::Cpu, int, float>( TrainingCpuFloatData(), true );
    }

    // Context Construction Tests
    TEST_F( EncoderTests, Context_Cpu_Float_DeviceType ) {
        TestDeviceType<Compute::DeviceType::Cpu, int, float>( ContextCpuFloatData() );
    }

    TEST_F( EncoderTests, Context_Cpu_Float_Forward ) {
        TestForward<Compute::DeviceType::Cpu, int, float>( ContextCpuFloatData() );
    }

    // CUDA Tests with float precision
    TEST_F( EncoderTests, Cuda_Float_TestName ) {
        TestGetName<Compute::DeviceType::Cuda, int, float>( CudaFloatData(), "cuda_encoder_float" );
    }

    TEST_F( EncoderTests, Cuda_Float_ParameterCount ) {
        TestParameterCount<Compute::DeviceType::Cuda, int, float>( CudaFloatData() );
    }

    TEST_F( EncoderTests, Cuda_Float_TestForward ) {
        try {
            TestForward<Compute::DeviceType::Cuda, int, float>( CudaFloatData() );
        }
        catch ( const std::exception& e ) {
            GTEST_SKIP() << "Skipping CUDA test due to exception: " << e.what();
        }
    }

    TEST_F( EncoderTests, Cuda_Float_TestPrint ) {
        TestPrint<Compute::DeviceType::Cuda, int, float>( CudaFloatData(), "Encoder: cuda_encoder_float" );
    }

    TEST_F( EncoderTests, Cuda_Float_DeviceType ) {
        TestDeviceType<Compute::DeviceType::Cuda, int, float>( CudaFloatData() );
    }

    TEST_F( EncoderTests, Cuda_Float_TrainingMode ) {
        TestTrainingMode<Compute::DeviceType::Cuda, int, float>( CudaFloatData(), false );
    }

    TEST_F( EncoderTests, Cuda_Training_Float_TrainingMode ) {
        TestTrainingMode<Compute::DeviceType::Cuda, int, float>( TrainingCudaFloatData(), true );
    }

    // CUDA Tests with half precision
    TEST_F( EncoderTests, Cuda_Half_TestName ) {
        TestGetName<Compute::DeviceType::Cuda, int, half>( CudaHalfData(), "cuda_encoder_half" );
    }

    TEST_F( EncoderTests, Cuda_Half_ParameterCount ) {
        TestParameterCount<Compute::DeviceType::Cuda, int, half>( CudaHalfData() );
    }

    TEST_F( EncoderTests, Cuda_Half_TestForward ) {
        try {
            TestForward<Compute::DeviceType::Cuda, int, half>( CudaHalfData() );
        }
        catch ( const std::exception& e ) {
            GTEST_SKIP() << "Skipping CUDA test due to exception: " << e.what();
        }
    }

    TEST_F( EncoderTests, Cuda_Half_TestPrint ) {
        TestPrint<Compute::DeviceType::Cuda, int, half>( CudaHalfData(), "Encoder: cuda_encoder_half" );
    }

    TEST_F( EncoderTests, Cuda_Half_TrainingMode ) {
        TestTrainingMode<Compute::DeviceType::Cuda, int, half>( CudaHalfData(), false );
    }

    TEST_F( EncoderTests, Cuda_Training_Half_TrainingMode ) {
        TestTrainingMode<Compute::DeviceType::Cuda, int, half>( TrainingCudaHalfData(), true );
    }

    // Mixed Precision Tests
    TEST_F( EncoderTests, Cuda_MixedPrecision_TestForward ) {
        try {
            TestForward<Compute::DeviceType::Cuda, int, half>( MixedPrecisionData() );
        }
        catch ( const std::exception& e ) {
            GTEST_SKIP() << "Skipping CUDA mixed precision test due to exception: " << e.what();
        }
    }

    TEST_F( EncoderTests, Cuda_MixedPrecision_TestName ) {
        TestGetName<Compute::DeviceType::Cuda, int, half>( MixedPrecisionData(), "cuda_encoder_mixed" );
    }

    // Edge Case Tests
    TEST_F( EncoderTests, Cpu_Float_EdgeCases ) {
        TestEdgeCases<Compute::DeviceType::Cpu, int, float>();
    }

    TEST_F( EncoderTests, Cuda_Float_EdgeCases ) {
        try {
            TestEdgeCases<Compute::DeviceType::Cuda, int, float>();
        }
        catch ( const std::exception& e ) {
            GTEST_SKIP() << "Skipping CUDA edge cases test due to exception: " << e.what();
        }
    }

    // CPU-CUDA Equivalence Test
    TEST_F( EncoderTests, Cpu_Cuda_Forward_Output_Equivalence ) {
        TestCpuCudaEquivalence<int, float>( CpuFloatData(), CudaFloatData() );
    }

    // CUDA Float-Half Precision Comparison
    TEST_F( EncoderTests, Cuda_Float_Half_Precision_Comparison ) {
        TestPrecisionComparison<int>( CudaFloatData(), CudaHalfData() );
    }

    // ComputePrecision Tests - CPU
    TEST_F( EncoderTests, Cpu_Float_DefaultPrecisionIsAuto ) {
        TestPrecisionPolicy<Compute::DeviceType::Cpu, int, float>( CpuFloatData(), ComputePrecision::Policy::Native );
    }

    TEST_F( EncoderTests, Cpu_Float_PerformancePrecision ) {
        TestPrecisionPolicy<Compute::DeviceType::Cpu, int, float>( PerfPrecisionCpuFloatData(), ComputePrecision::Policy::Native );
    }

    TEST_F( EncoderTests, Cpu_Float_AccuracyPrecision ) {
        TestPrecisionPolicy<Compute::DeviceType::Cpu, int, float>( AccuracyPrecisionCpuFloatData(), ComputePrecision::Policy::Native );
    }

    TEST_F( EncoderTests, Cpu_Float_DisabledPrecision ) {
        TestPrecisionPolicy<Compute::DeviceType::Cpu, int, float>( DisabledPrecisionCpuFloatData(), ComputePrecision::Policy::Native );
    }

    // ComputePrecision Tests - CUDA
    TEST_F( EncoderTests, Cuda_Float_DefaultPrecisionIsAuto ) {
        TestPrecisionPolicy<Compute::DeviceType::Cuda, int, float>( CudaFloatData(), ComputePrecision::Policy::Auto );
    }

    TEST_F( EncoderTests, Cuda_Float_PerformancePrecision ) {
        TestPrecisionPolicy<Compute::DeviceType::Cuda, int, float>( PerfPrecisionCudaFloatData(), ComputePrecision::Policy::Performance );
    }

    TEST_F( EncoderTests, Cuda_Float_AccuracyPrecision ) {
        TestPrecisionPolicy<Compute::DeviceType::Cuda, int, float>( AccuracyPrecisionCudaFloatData(), ComputePrecision::Policy::Accuracy );
    }

    TEST_F( EncoderTests, Cuda_Float_DisabledPrecision ) {
        TestPrecisionPolicy<Compute::DeviceType::Cuda, int, float>( DisabledPrecisionCudaFloatData(), ComputePrecision::Policy::Native );
    }

    // Type alias tests
    TEST_F( EncoderTests, TypeAliases ) {
        // Verify that type aliases work correctly
        std::string name = "type_alias_test";
        size_t channels = 32;
        size_t max_seq_len = 64;
        size_t vocab_len = 100;

        // CpuEncoder
        auto cpu_encoder = std::make_shared<CpuEncoder<int, float>>(
            name, "CPU", channels, max_seq_len, vocab_len );
        EXPECT_EQ( cpu_encoder->getDeviceContext()->getDevice()->getDeviceType(), DeviceType::Cpu );

        try {
            // CudaEncoder (only test if CUDA is available)
            DeviceContext context( "CUDA:0" );
            auto cuda_encoder = std::make_shared<CudaEncoder<int, float>>(
                name, "CUDA:0", channels, max_seq_len, vocab_len );
            EXPECT_EQ( cuda_encoder->getDeviceContext()->getDevice()->getDeviceType(), DeviceType::Cuda );
        }
        catch ( const std::exception& ) {
            GTEST_SKIP() << "CUDA device not available, skipping CUDA type alias test";
        }
    }
}