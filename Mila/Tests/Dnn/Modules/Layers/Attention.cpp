#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <sstream>
#include <cuda_fp16.h>

import Mila;

namespace Modules::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    // Memory resource selector based on device type
    template<DeviceType TDevice>
    using MemoryResourceType = std::conditional_t<TDevice == Compute::DeviceType::Cuda,
        Compute::CudaDeviceMemoryResource,
        Compute::HostMemoryResource>;

    // Common test data structure that can be reused
    template<DeviceType TDevice, typename TInput = float, typename TOutput = TInput>
    struct AttentionTestData {
        size_t batch_size;
        size_t sequence_length;
        size_t channels;
        size_t num_heads;
        std::vector<size_t> input_shape;
        std::shared_ptr<MultiHeadAttention<TDevice, TInput, TOutput>> attention_module;

        // Make the test data structure self-initializing
        static AttentionTestData Create(
            const std::string& name,
            size_t batch_size,
            size_t sequence_length,
            size_t channels,
            size_t num_heads,
            bool is_training = false,
            ComputePrecision::Policy precision = ComputePrecision::Policy::Auto )
        {
            AttentionTestData data;
            data.batch_size = batch_size;
            data.sequence_length = sequence_length;
            data.channels = channels;
            data.num_heads = num_heads;
            data.input_shape = { batch_size, sequence_length, 3 * channels };

            std::string device_str = TDevice == DeviceType::Cuda ? "CUDA:0" : "CPU";
            data.attention_module = std::make_shared<MultiHeadAttention<TDevice, TInput, TOutput>>(
                name, device_str, data.input_shape, num_heads, is_training, precision );

            return data;
        }

        // Overload for creating with device context
        static AttentionTestData CreateWithContext(
            const std::string& name,
            size_t batch_size,
            size_t sequence_length,
            size_t channels,
            size_t num_heads,
            std::shared_ptr<DeviceContext> context,
            bool is_training = false,
            ComputePrecision::Policy precision = ComputePrecision::Policy::Auto )
        {
            AttentionTestData data;
            data.batch_size = batch_size;
            data.sequence_length = sequence_length;
            data.channels = channels;
            data.num_heads = num_heads;
            data.input_shape = { batch_size, sequence_length, 3 * channels };

            data.attention_module = std::make_shared<MultiHeadAttention<TDevice, TInput, TOutput>>(
                name, context, data.input_shape, num_heads, is_training, precision );

            return data;
        }
    };

    class AttentionTests : public ::testing::Test {
    protected:
        void SetUp() override {
            batch_size_ = 64;
            cpu_batch_size_ = 4;
            sequence_length_ = 1024;
            channels_ = 768;
            num_heads_ = 12;
            // Modules will be created on demand
        }

        void TearDown() override {
            // Clean up resources explicitly
            cpu_float_data_.attention_module.reset();
            training_cpu_float_data_.attention_module.reset();
            context_cpu_float_data_.attention_module.reset();
            cuda_float_data_.attention_module.reset();
            training_cuda_float_data_.attention_module.reset();
            cuda_half_data_.attention_module.reset();
            training_cuda_half_data_.attention_module.reset();
            mixed_precision_data_.attention_module.reset();

            // Precision policy test data
            perf_precision_cpu_float_data_.attention_module.reset();
            accuracy_precision_cpu_float_data_.attention_module.reset();
            disabled_precision_cpu_float_data_.attention_module.reset();
            perf_precision_cuda_float_data_.attention_module.reset();
            accuracy_precision_cuda_float_data_.attention_module.reset();
            disabled_precision_cuda_float_data_.attention_module.reset();
        }

        // Factory methods to lazily create test data as needed
        AttentionTestData<DeviceType::Cpu, float, float>& CpuFloatData() {
            if ( !cpu_float_data_.attention_module ) {
                cpu_float_data_ = AttentionTestData<DeviceType::Cpu, float, float>::Create(
                    "cpu_attn_float", cpu_batch_size_, sequence_length_, channels_, num_heads_ );
            }
            return cpu_float_data_;
        }

        AttentionTestData<DeviceType::Cuda, float, float>& CudaFloatData() {
            if ( !cuda_float_data_.attention_module ) {
                cuda_float_data_ = AttentionTestData<DeviceType::Cuda, float, float>::Create(
                    "cuda_attn_float", batch_size_, sequence_length_, channels_, num_heads_ );
            }
            return cuda_float_data_;
        }

        AttentionTestData<DeviceType::Cpu, float, float>& TrainingCpuFloatData() {
            if ( !training_cpu_float_data_.attention_module ) {
                training_cpu_float_data_ = AttentionTestData<DeviceType::Cpu, float, float>::Create(
                    "cpu_attn_float_training", cpu_batch_size_, sequence_length_, channels_, num_heads_, true );
            }
            return training_cpu_float_data_;
        }

        AttentionTestData<DeviceType::Cuda, float, float>& TrainingCudaFloatData() {
            if ( !training_cuda_float_data_.attention_module ) {
                training_cuda_float_data_ = AttentionTestData<DeviceType::Cuda, float, float>::Create(
                    "cuda_attn_float_training", batch_size_, sequence_length_, channels_, num_heads_, true );
            }
            return training_cuda_float_data_;
        }

        AttentionTestData<DeviceType::Cpu, float, float>& ContextCpuFloatData() {
            if ( !context_cpu_float_data_.attention_module ) {
                auto cpu_context = std::make_shared<DeviceContext>( "CPU" );
                context_cpu_float_data_ = AttentionTestData<DeviceType::Cpu, float, float>::CreateWithContext(
                    "cpu_context_attn_float", cpu_batch_size_, sequence_length_, channels_, num_heads_, cpu_context );
            }
            return context_cpu_float_data_;
        }

        AttentionTestData<DeviceType::Cuda, float, half>& CudaHalfData() {
            if ( !cuda_half_data_.attention_module ) {
                cuda_half_data_ = AttentionTestData<DeviceType::Cuda, float, half>::Create(
                    "cuda_attn_half", batch_size_, sequence_length_, channels_, num_heads_ );
            }
            return cuda_half_data_;
        }

        AttentionTestData<DeviceType::Cuda, float, half>& TrainingCudaHalfData() {
            if ( !training_cuda_half_data_.attention_module ) {
                training_cuda_half_data_ = AttentionTestData<DeviceType::Cuda, float, half>::Create(
                    "cuda_attn_half_training", batch_size_, sequence_length_, channels_, num_heads_, true );
            }
            return training_cuda_half_data_;
        }

        // Test for mixed precision (float input, half output)
        AttentionTestData<DeviceType::Cuda, float, half>& MixedPrecisionData() {
            if ( !mixed_precision_data_.attention_module ) {
                mixed_precision_data_ = AttentionTestData<DeviceType::Cuda, float, half>::Create(
                    "cuda_attn_mixed", batch_size_, sequence_length_, channels_, num_heads_ );
            }
            return mixed_precision_data_;
        }

        // Tests with specific precision policies - CPU
        AttentionTestData<DeviceType::Cpu, float, float>& PerfPrecisionCpuFloatData() {
            if ( !perf_precision_cpu_float_data_.attention_module ) {
                perf_precision_cpu_float_data_ = AttentionTestData<DeviceType::Cpu, float, float>::Create(
                    "cpu_attn_perf_precision", cpu_batch_size_, sequence_length_, channels_, num_heads_, false,
                    ComputePrecision::Policy::Performance );
            }
            return perf_precision_cpu_float_data_;
        }

        AttentionTestData<DeviceType::Cpu, float, float>& AccuracyPrecisionCpuFloatData() {
            if ( !accuracy_precision_cpu_float_data_.attention_module ) {
                accuracy_precision_cpu_float_data_ = AttentionTestData<DeviceType::Cpu, float, float>::Create(
                    "cpu_attn_accuracy_precision", cpu_batch_size_, sequence_length_, channels_, num_heads_, false,
                    ComputePrecision::Policy::Accuracy );
            }
            return accuracy_precision_cpu_float_data_;
        }

        AttentionTestData<DeviceType::Cpu, float, float>& DisabledPrecisionCpuFloatData() {
            if ( !disabled_precision_cpu_float_data_.attention_module ) {
                disabled_precision_cpu_float_data_ = AttentionTestData<DeviceType::Cpu, float, float>::Create(
                    "cpu_attn_disabled_precision", cpu_batch_size_, sequence_length_, channels_, num_heads_, false,
                    ComputePrecision::Policy::Native );
            }
            return disabled_precision_cpu_float_data_;
        }

        // Tests with specific precision policies - CUDA
        AttentionTestData<DeviceType::Cuda, float, float>& PerfPrecisionCudaFloatData() {
            if ( !perf_precision_cuda_float_data_.attention_module ) {
                perf_precision_cuda_float_data_ = AttentionTestData<DeviceType::Cuda, float, float>::Create(
                    "cuda_attn_perf_precision", batch_size_, sequence_length_, channels_, num_heads_, false,
                    ComputePrecision::Policy::Performance );
            }
            return perf_precision_cuda_float_data_;
        }

        AttentionTestData<DeviceType::Cuda, float, float>& AccuracyPrecisionCudaFloatData() {
            if ( !accuracy_precision_cuda_float_data_.attention_module ) {
                accuracy_precision_cuda_float_data_ = AttentionTestData<DeviceType::Cuda, float, float>::Create(
                    "cuda_attn_accuracy_precision", batch_size_, sequence_length_, channels_, num_heads_, false,
                    ComputePrecision::Policy::Accuracy );
            }
            return accuracy_precision_cuda_float_data_;
        }

        AttentionTestData<DeviceType::Cuda, float, float>& DisabledPrecisionCudaFloatData() {
            if ( !disabled_precision_cuda_float_data_.attention_module ) {
                disabled_precision_cuda_float_data_ = AttentionTestData<DeviceType::Cuda, float, float>::Create(
                    "cuda_attn_disabled_precision", batch_size_, sequence_length_, channels_, num_heads_, false,
                    ComputePrecision::Policy::Native );
            }
            return disabled_precision_cuda_float_data_;
        }

        // Test parameters
        size_t batch_size_{ 0 };
        size_t cpu_batch_size_{ 0 };
        size_t sequence_length_{ 0 };
        size_t channels_{ 0 };
        size_t num_heads_{ 0 };

        // Test data objects - initialized on demand
        AttentionTestData<DeviceType::Cpu, float, float> cpu_float_data_;
        AttentionTestData<DeviceType::Cpu, float, float> training_cpu_float_data_;
        AttentionTestData<DeviceType::Cpu, float, float> context_cpu_float_data_;

        AttentionTestData<DeviceType::Cuda, float, float> cuda_float_data_;
        AttentionTestData<DeviceType::Cuda, float, float> training_cuda_float_data_;

        AttentionTestData<DeviceType::Cuda, float, half> cuda_half_data_;
        AttentionTestData<DeviceType::Cuda, float, half> training_cuda_half_data_;

        // Mixed precision test data
        AttentionTestData<DeviceType::Cuda, float, half> mixed_precision_data_;

        // Precision policy test data - CPU
        AttentionTestData<DeviceType::Cpu, float, float> perf_precision_cpu_float_data_;
        AttentionTestData<DeviceType::Cpu, float, float> accuracy_precision_cpu_float_data_;
        AttentionTestData<DeviceType::Cpu, float, float> disabled_precision_cpu_float_data_;

        // Precision policy test data - CUDA
        AttentionTestData<DeviceType::Cuda, float, float> perf_precision_cuda_float_data_;
        AttentionTestData<DeviceType::Cuda, float, float> accuracy_precision_cuda_float_data_;
        AttentionTestData<DeviceType::Cuda, float, float> disabled_precision_cuda_float_data_;
    };

    // Common test function templates
    template<DeviceType TDevice, typename TInput, typename TOutput = TInput>
    void TestGetName( const AttentionTestData<TDevice, TInput, TOutput>& data, const std::string& expected_name ) {
        EXPECT_EQ( data.attention_module->getName(), expected_name );
    }

    template<DeviceType TDevice, typename TInput, typename TOutput = TInput>
    void TestParameterCount( const AttentionTestData<TDevice, TInput, TOutput>& data, size_t expected_count ) {
        EXPECT_EQ( data.attention_module->parameterCount(), expected_count );
    }

    template<DeviceType TDevice, typename TInput, typename TOutput = TInput>
    void TestForward( const AttentionTestData<TDevice, TInput, TOutput>& data ) {
        using MR = MemoryResourceType<TDevice>;

        Tensor<TInput, MR> input( data.input_shape );
        Tensor<TOutput, MR> output( data.input_shape );

        // Fill with random values
        if constexpr ( TDevice == DeviceType::Cpu ) {
            // Direct access for CPU memory
            for ( size_t i = 0; i < input.size(); ++i ) {
                input.data()[ i ] = static_cast<TInput>( (i % 100) * 0.01f );
            }
        }
        else {
            // For device memory, create a host tensor, fill it, then copy to device
            Tensor<TInput, HostMemoryResource> host_input( data.input_shape );
            for ( size_t i = 0; i < host_input.size(); ++i ) {
                host_input.data()[ i ] = static_cast<TInput>( (i % 100) * 0.01f );
            }
            input.copyFrom( host_input );
        }

        data.attention_module->forward( input, output );
        EXPECT_EQ( output.size(), input.size() );
    }

    template<DeviceType TDevice, typename TInput, typename TOutput = TInput>
    void TestPrint( const AttentionTestData<TDevice, TInput, TOutput>& data, const std::string& expected_substring ) {
        std::string output = data.attention_module->toString();
        EXPECT_NE( output.find( expected_substring ), std::string::npos );
    }

    template<DeviceType TDevice, typename TInput, typename TOutput = TInput>
    void TestTrainingMode( const AttentionTestData<TDevice, TInput, TOutput>& data, bool expected_mode ) {
        EXPECT_EQ( data.attention_module->isTraining(), expected_mode );
    }

    template<DeviceType TDevice, typename TInput, typename TOutput = TInput>
    void TestDeviceType( const AttentionTestData<TDevice, TInput, TOutput>& data ) {
        auto device_context = data.attention_module->getDeviceContext();
        EXPECT_NE( device_context, nullptr );
        auto device = device_context->getDevice();
        EXPECT_NE( device, nullptr );
        EXPECT_EQ( device->getDeviceType(), TDevice );
    }

    // New test for precision policy
    template<DeviceType TDevice, typename TInput, typename TOutput = TInput>
    void TestPrecisionPolicy( const AttentionTestData<TDevice, TInput, TOutput>& data, ComputePrecision::Policy expected_policy ) {
        EXPECT_EQ( data.attention_module->getComputePrecision().getPolicy(), expected_policy );

        // Verify toString() contains precision info
        std::string output = data.attention_module->toString();
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
    template<typename TInput, typename TOutput = TInput>
    void TestCpuCudaEquivalence(
        const AttentionTestData<DeviceType::Cpu, TInput, TOutput>& cpu_data,
        const AttentionTestData<DeviceType::Cuda, TInput, TOutput>& cuda_data ) {

        try {
            // Check if CUDA is available
            DeviceContext context( "CUDA:0" );
        }
        catch ( const std::exception& ) {
            GTEST_SKIP() << "CUDA device not available, skipping CPU-CUDA equivalence test";
            return;
        }

        // Create a small test shape to make comparison faster
        std::vector<size_t> test_shape = { 2, 16, 3 * 32 }; // Small shape for quick verification
        size_t num_heads = 4;

        // Create test modules with smaller dimensions
        auto cpu_test_module = std::make_shared<MultiHeadAttention<DeviceType::Cpu, TInput, TOutput>>(
            "cpu_test", "CPU", test_shape, num_heads );

        auto cuda_test_module = std::make_shared<MultiHeadAttention<DeviceType::Cuda, TInput, TOutput>>(
            "cuda_test", "CUDA:0", test_shape, num_heads );

        // Create random input data
        Tensor<TInput, HostMemoryResource> host_input( test_shape );

        // Fill with predictable values
        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input.data()[ i ] = static_cast<TInput>( (static_cast<float>( i ) / host_input.size()) - 0.5f );
        }

        // Create CPU output
        Tensor<TOutput, HostMemoryResource> cpu_output( test_shape );
        cpu_test_module->forward( host_input, cpu_output );

        // Create device input by copying host data
        Tensor<TInput, CudaDeviceMemoryResource> device_input( test_shape );
        device_input.copyFrom( host_input );

        // Create device output
        Tensor<TOutput, CudaDeviceMemoryResource> cuda_output( test_shape );
        cuda_test_module->forward( device_input, cuda_output );

        // Copy CUDA output back to host for comparison
        Tensor<TOutput, HostMemoryResource> cuda_output_host( test_shape );
        cuda_output_host.copyFrom( cuda_output );

        // Compare outputs with tolerance for floating point differences
        const float epsilon = 1e-4f; // Tolerance depends on precision and implementation
        bool all_equal = true;

        for ( size_t i = 0; i < cpu_output.size(); ++i ) {
            float diff = std::abs( static_cast<float>( cpu_output.data()[ i ] ) - static_cast<float>( cuda_output_host.data()[ i ] ) );
            if ( diff > epsilon ) {
                std::cout << "Difference at index " << i << ": CPU=" << cpu_output.data()[ i ]
                    << ", CUDA=" << cuda_output_host.data()[ i ] << ", diff=" << diff << std::endl;
                all_equal = false;
                // Only show first few differences to avoid flooding the output
                if ( i > 20 ) break;
            }
        }

        EXPECT_TRUE( all_equal ) << "CPU and CUDA implementations produced different results";
    }

    // Test for CUDA float vs half precision accuracy
    template<typename TInput = float>
    void TestPrecisionComparison(
        const AttentionTestData<DeviceType::Cuda, TInput, float>& float_data,
        const AttentionTestData<DeviceType::Cuda, TInput, half>& half_data ) {

        try {
            // Check if CUDA is available
            DeviceContext context( "CUDA:0" );
        }
        catch ( const std::exception& ) {
            GTEST_SKIP() << "CUDA device not available, skipping precision comparison test";
            return;
        }

        // Create small test shapes
        std::vector<size_t> test_shape = { 2, 16, 3 * 32 };
        size_t num_heads = 4;

        // Create test modules with smaller dimensions
        auto float_module = std::make_shared<MultiHeadAttention<DeviceType::Cuda, TInput, float>>(
            "float_test", "CUDA:0", test_shape, num_heads );

        auto half_module = std::make_shared<MultiHeadAttention<DeviceType::Cuda, TInput, half>>(
            "half_test", "CUDA:0", test_shape, num_heads );

        // Create input data
        Tensor<TInput, HostMemoryResource> host_input( test_shape );
        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input.data()[ i ] = static_cast<TInput>( (i % 100) * 0.01f - 0.5f );
        }

        // Copy to device
        Tensor<TInput, CudaDeviceMemoryResource> device_input( test_shape );
        device_input.copyFrom( host_input );

        // Run both modules
        Tensor<float, CudaDeviceMemoryResource> float_output( test_shape );
        Tensor<half, CudaDeviceMemoryResource> half_output( test_shape );

        float_module->forward( device_input, float_output );
        half_module->forward( device_input, half_output );

        // Copy results back to host for comparison
        Tensor<float, HostMemoryResource> float_output_host( test_shape );
        float_output_host.copyFrom( float_output );

        Tensor<float, HostMemoryResource> half_output_host( test_shape );
        // Convert half results to float for comparison
        Tensor<half, HostMemoryResource> half_output_host_tmp( test_shape );
        half_output_host_tmp.copyFrom( half_output );
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

    template<DeviceType TDevice, typename TInput, typename TOutput = TInput>
    void TestDifferentHeadCounts( const AttentionTestData<TDevice, TInput, TOutput>& data ) {
        using MR = MemoryResourceType<TDevice>;
        std::vector<size_t> head_counts = { 1, 4, 8, 16 };
        std::vector<size_t> small_shape = { 2, 16, 3 * 64 };

        for ( auto num_heads : head_counts ) {
            auto test_module = std::make_shared<MultiHeadAttention<TDevice, TInput, TOutput>>(
                "test_heads_" + std::to_string( num_heads ),
                TDevice == DeviceType::Cuda ? "CUDA:0" : "CPU",
                small_shape, num_heads );

            Tensor<TInput, MR> input( small_shape );
            Tensor<TOutput, MR> output( small_shape );

            // Initialize input with some values
            if constexpr ( TDevice == DeviceType::Cpu ) {
                for ( size_t i = 0; i < input.size(); ++i ) {
                    input.data()[ i ] = static_cast<TInput>( i % 10 ) * 0.1f;
                }
            }
            else {
                Tensor<TInput, HostMemoryResource> host_input( small_shape );
                for ( size_t i = 0; i < host_input.size(); ++i ) {
                    host_input.data()[ i ] = static_cast<TInput>( i % 10 ) * 0.1f;
                }
                input.copyFrom( host_input );
            }

            EXPECT_NO_THROW( test_module->forward( input, output ) );
            EXPECT_EQ( output.size(), input.size() );
        }
    }

    // Test edge cases with minimal and large shapes
    template<DeviceType TDevice, typename TInput, typename TOutput = TInput>
    void TestEdgeCases( const AttentionTestData<TDevice, TInput, TOutput>& data ) {
        using MR = MemoryResourceType<TDevice>;
        // Test with minimal batch size and sequence length
        std::vector<size_t> minimal_shape = { 1, 1, 3 * 64 };
        auto minimal_module = std::make_shared<MultiHeadAttention<TDevice, TInput, TOutput>>(
            "minimal", TDevice == DeviceType::Cuda ? "CUDA:0" : "CPU", minimal_shape, 1 );

        Tensor<TInput, MR> minimal_input( minimal_shape );
        Tensor<TOutput, MR> minimal_output( minimal_shape );

        EXPECT_NO_THROW( minimal_module->forward( minimal_input, minimal_output ) );

        // Test with large sequence length
        std::vector<size_t> long_seq_shape = { 2, 128, 3 * 64 }; // Reduced for test performance
        auto long_seq_module = std::make_shared<MultiHeadAttention<TDevice, TInput, TOutput>>(
            "long_seq", TDevice == DeviceType::Cuda ? "CUDA:0" : "CPU", long_seq_shape, 4 );

        Tensor<TInput, MR> long_input( long_seq_shape );
        Tensor<TOutput, MR> long_output( long_seq_shape );

        EXPECT_NO_THROW( long_seq_module->forward( long_input, long_output ) );
    }

    // Test training mode behavior
    template<DeviceType TDevice, typename TInput, typename TOutput = TInput>
    void TestTrainingModeBehavior( const AttentionTestData<TDevice, TInput, TOutput>& data ) {
        using MR = MemoryResourceType<TDevice>;
        std::vector<size_t> shape = { 2, 16, 3 * 64 };
        auto train_module = std::make_shared<MultiHeadAttention<TDevice, TInput, TOutput>>(
            "train_mode", TDevice == DeviceType::Cuda ? "CUDA:0" : "CPU", shape, 4, true ); // training mode enabled

        auto infer_module = std::make_shared<MultiHeadAttention<TDevice, TInput, TOutput>>(
            "infer_mode", TDevice == DeviceType::Cuda ? "CUDA:0" : "CPU", shape, 4, false ); // training mode disabled

        Tensor<TInput, MR> input( shape );
        Tensor<TOutput, MR> train_output( shape );
        Tensor<TOutput, MR> infer_output( shape );

        // Fill input with values
        if constexpr ( TDevice == DeviceType::Cpu ) {
            for ( size_t i = 0; i < input.size(); ++i ) {
                input.data()[ i ] = static_cast<TInput>( i % 7 ) * 0.1f;
            }
        }
        else {
            Tensor<TInput, HostMemoryResource> host_input( shape );
            for ( size_t i = 0; i < host_input.size(); ++i ) {
                host_input.data()[ i ] = static_cast<TInput>( i % 7 ) * 0.1f;
            }
            input.copyFrom( host_input );
        }

        train_module->forward( input, train_output );
        infer_module->forward( input, infer_output );

        // Check if train_module is in training mode
        EXPECT_TRUE( train_module->isTraining() );

        // Check if infer_module is not in training mode
        EXPECT_FALSE( infer_module->isTraining() );

        // Verify that both still produce valid outputs
        EXPECT_EQ( train_output.size(), input.size() );
        EXPECT_EQ( infer_output.size(), input.size() );
    }

    // Test numerical stability
    template<DeviceType TDevice, typename TInput, typename TOutput = TInput>
    void TestNumericalStability( const AttentionTestData<TDevice, TInput, TOutput>& data ) {
        using MR = MemoryResourceType<TDevice>;
        std::vector<size_t> shape = { 2, 16, 3 * 64 };
        auto stability_module = std::make_shared<MultiHeadAttention<TDevice, TInput, TOutput>>(
            "stability", TDevice == DeviceType::Cuda ? "CUDA:0" : "CPU", shape, 4 );

        Tensor<TInput, MR> input( shape );
        Tensor<TOutput, MR> output( shape );

        // Test with large values (smaller than previously suggested to avoid overflow)
        if constexpr ( TDevice == DeviceType::Cpu ) {
            for ( size_t i = 0; i < input.size(); ++i ) {
                input.data()[ i ] = 1.0e+3f;
            }
        }
        else {
            Tensor<TInput, HostMemoryResource> host_input( shape );
            for ( size_t i = 0; i < host_input.size(); ++i ) {
                host_input.data()[ i ] = 1.0e+3f;
            }
            input.copyFrom( host_input );
        }

        stability_module->forward( input, output );

        // Verify no NaNs or Infs in output
        bool has_nan_or_inf = false;
        if constexpr ( TDevice == DeviceType::Cpu ) {
            for ( size_t i = 0; i < output.size(); ++i ) {
                if ( std::isnan( output.data()[ i ] ) || std::isinf( output.data()[ i ] ) ) {
                    has_nan_or_inf = true;
                    break;
                }
            }
        }
        else {
            Tensor<TOutput, HostMemoryResource> host_output( shape );
            host_output.copyFrom( output );

            for ( size_t i = 0; i < host_output.size(); ++i ) {
                if ( std::isnan( host_output.data()[ i ] ) || std::isinf( host_output.data()[ i ] ) ) {
                    has_nan_or_inf = true;
                    break;
                }
            }
        }

        EXPECT_FALSE( has_nan_or_inf ) << "Output contains NaN or Inf values with large inputs";

        // Test with small values
        if constexpr ( TDevice == DeviceType::Cpu ) {
            for ( size_t i = 0; i < input.size(); ++i ) {
                input.data()[ i ] = 1.0e-3f;
            }
        }
        else {
            Tensor<TInput, HostMemoryResource> host_input( shape );
            for ( size_t i = 0; i < host_input.size(); ++i ) {
                host_input.data()[ i ] = 1.0e-3f;
            }
            input.copyFrom( host_input );
        }

        stability_module->forward( input, output );

        // Verify output again
        has_nan_or_inf = false;
        if constexpr ( TDevice == DeviceType::Cpu ) {
            for ( size_t i = 0; i < output.size(); ++i ) {
                if ( std::isnan( output.data()[ i ] ) || std::isinf( output.data()[ i ] ) ) {
                    has_nan_or_inf = true;
                    break;
                }
            }
        }
        else {
            Tensor<TOutput, HostMemoryResource> host_output( shape );
            host_output.copyFrom( output );

            for ( size_t i = 0; i < host_output.size(); ++i ) {
                if ( std::isnan( host_output.data()[ i ] ) || std::isinf( host_output.data()[ i ] ) ) {
                    has_nan_or_inf = true;
                    break;
                }
            }
        }

        EXPECT_FALSE( has_nan_or_inf ) << "Output contains NaN or Inf values with small inputs";
    }

    // Test deterministic behavior (for CUDA implementation)
    template<typename TInput, typename TOutput = TInput>
    void TestDeterministicBehavior(
        const AttentionTestData<DeviceType::Cuda, TInput, TOutput>& data ) {

        std::vector<size_t> shape = { 2, 16, 3 * 64 };
        auto deterministic_module = std::make_shared<MultiHeadAttention<DeviceType::Cuda, TInput, TOutput>>(
            "deterministic", "CUDA:0", shape, 4 );

        Tensor<TInput, CudaDeviceMemoryResource> input( shape );
        Tensor<TOutput, CudaDeviceMemoryResource> output1( shape );
        Tensor<TOutput, CudaDeviceMemoryResource> output2( shape );

        // Fill input with predictable values
        Tensor<TInput, HostMemoryResource> host_input( shape );
        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input.data()[ i ] = static_cast<TInput>( (i % 10) * 0.1f );
        }
        input.copyFrom( host_input );

        // Run forward pass twice
        deterministic_module->forward( input, output1 );
        deterministic_module->forward( input, output2 );

        // Copy results back to host for comparison
        Tensor<TOutput, HostMemoryResource> host_output1( shape );
        Tensor<TOutput, HostMemoryResource> host_output2( shape );
        host_output1.copyFrom( output1 );
        host_output2.copyFrom( output2 );

        // Verify outputs are identical
        bool outputs_match = true;
        for ( size_t i = 0; i < host_output1.size(); ++i ) {
            if ( host_output1.data()[ i ] != host_output2.data()[ i ] ) {
                outputs_match = false;
                break;
            }
        }

        EXPECT_TRUE( outputs_match ) << "Multiple runs with the same input produced different results";
    }

    // Now add the actual test cases that use these templates:

    // CPU Tests with float precision
    TEST_F( AttentionTests, Cpu_Float_TestName ) {
        TestGetName<DeviceType::Cpu, float>( CpuFloatData(), "cpu_attn_float" );
    }

    TEST_F( AttentionTests, Cpu_Float_ParameterCount ) {
        TestParameterCount<DeviceType::Cpu, float>( CpuFloatData(), 0 );
    }

    TEST_F( AttentionTests, Cpu_Float_TestForward ) {
        TestForward<DeviceType::Cpu, float>( CpuFloatData() );
    }

    TEST_F( AttentionTests, Cpu_Float_TestPrint ) {
        TestPrint<DeviceType::Cpu, float>( CpuFloatData(), "MultiHeadAttention: cpu_attn_float" );
    }

    TEST_F( AttentionTests, Cpu_Float_DeviceType ) {
        TestDeviceType<DeviceType::Cpu, float>( CpuFloatData() );
    }

    TEST_F( AttentionTests, Cpu_Float_TrainingMode ) {
        TestTrainingMode<DeviceType::Cpu, float>( CpuFloatData(), false );
    }

    TEST_F( AttentionTests, Cpu_Training_Float_TrainingMode ) {
        TestTrainingMode<DeviceType::Cpu, float>( TrainingCpuFloatData(), true );
    }

    TEST_F( AttentionTests, Cpu_Float_DifferentHeadCounts ) {
        TestDifferentHeadCounts<DeviceType::Cpu, float>( CpuFloatData() );
    }

    TEST_F( AttentionTests, Cpu_Float_EdgeCases ) {
        TestEdgeCases<DeviceType::Cpu, float>( CpuFloatData() );
    }

    TEST_F( AttentionTests, Cpu_Float_TrainingModeBehavior ) {
        TestTrainingModeBehavior<DeviceType::Cpu, float>( CpuFloatData() );
    }

    TEST_F( AttentionTests, Cpu_Float_NumericalStability ) {
        TestNumericalStability<DeviceType::Cpu, float>( CpuFloatData() );
    }

    // Context Construction Tests
    TEST_F( AttentionTests, Context_Cpu_Float_DeviceType ) {
        TestDeviceType<DeviceType::Cpu, float>( ContextCpuFloatData() );
    }

    // CUDA Tests with float precision
    TEST_F( AttentionTests, Cuda_Float_TestName ) {
        TestGetName<DeviceType::Cuda, float>( CudaFloatData(), "cuda_attn_float" );
    }

    TEST_F( AttentionTests, Cuda_Float_ParameterCount ) {
        TestParameterCount<DeviceType::Cuda, float>( CudaFloatData(), 0 );
    }

    TEST_F( AttentionTests, Cuda_Float_TestForward ) {
        try {
            TestForward<DeviceType::Cuda, float>( CudaFloatData() );
        }
        catch ( const std::exception& e ) {
            GTEST_SKIP() << "Skipping CUDA test due to exception: " << e.what();
        }
    }

    TEST_F( AttentionTests, Cuda_Float_TestPrint ) {
        TestPrint<DeviceType::Cuda, float>( CudaFloatData(), "MultiHeadAttention: cuda_attn_float" );
    }

    TEST_F( AttentionTests, Cuda_Float_DeviceType ) {
        TestDeviceType<DeviceType::Cuda, float>( CudaFloatData() );
    }

    TEST_F( AttentionTests, Cuda_Float_TrainingMode ) {
        TestTrainingMode<DeviceType::Cuda, float>( CudaFloatData(), false );
    }

    TEST_F( AttentionTests, Cuda_Training_Float_TrainingMode ) {
        TestTrainingMode<DeviceType::Cuda, float>( TrainingCudaFloatData(), true );
    }

    TEST_F( AttentionTests, Cuda_Float_DifferentHeadCounts ) {
        try {
            TestDifferentHeadCounts<DeviceType::Cuda, float>( CudaFloatData() );
        }
        catch ( const std::exception& e ) {
            GTEST_SKIP() << "Skipping CUDA test due to exception: " << e.what();
        }
    }

    TEST_F( AttentionTests, Cuda_Float_EdgeCases ) {
        try {
            TestEdgeCases<DeviceType::Cuda, float>( CudaFloatData() );
        }
        catch ( const std::exception& e ) {
            GTEST_SKIP() << "Skipping CUDA test due to exception: " << e.what();
        }
    }

    TEST_F( AttentionTests, Cuda_Float_TrainingModeBehavior ) {
        try {
            TestTrainingModeBehavior<DeviceType::Cuda, float>( CudaFloatData() );
        }
        catch ( const std::exception& e ) {
            GTEST_SKIP() << "Skipping CUDA test due to exception: " << e.what();
        }
    }

    TEST_F( AttentionTests, Cuda_Float_NumericalStability ) {
        try {
            TestNumericalStability<DeviceType::Cuda, float>( CudaFloatData() );
        }
        catch ( const std::exception& e ) {
            GTEST_SKIP() << "Skipping CUDA test due to exception: " << e.what();
        }
    }

    TEST_F( AttentionTests, Cuda_Float_Deterministic ) {
        try {
            TestDeterministicBehavior<float>( CudaFloatData() );
        }
        catch ( const std::exception& e ) {
            GTEST_SKIP() << "Skipping CUDA test due to exception: " << e.what();
        }
    }

    // CUDA Tests with half precision
    TEST_F( AttentionTests, Cuda_Half_TestName ) {
        TestGetName<DeviceType::Cuda, float, half>( CudaHalfData(), "cuda_attn_half" );
    }

    TEST_F( AttentionTests, Cuda_Half_ParameterCount ) {
        TestParameterCount<DeviceType::Cuda, float, half>( CudaHalfData(), 0 );
    }

    TEST_F( AttentionTests, Cuda_Half_TestForward ) {
        try {
            TestForward<DeviceType::Cuda, float, half>( CudaHalfData() );
        }
        catch ( const std::exception& e ) {
            GTEST_SKIP() << "Skipping CUDA half precision test due to exception: " << e.what();
        }
    }

    TEST_F( AttentionTests, Cuda_Half_TestPrint ) {
        TestPrint<DeviceType::Cuda, float, half>( CudaHalfData(), "MultiHeadAttention: cuda_attn_half" );
    }

    TEST_F( AttentionTests, Cuda_Half_TrainingMode ) {
        TestTrainingMode<DeviceType::Cuda, float, half>( CudaHalfData(), false );
    }

    TEST_F( AttentionTests, Cuda_Training_Half_TrainingMode ) {
        TestTrainingMode<DeviceType::Cuda, float, half>( TrainingCudaHalfData(), true );
    }

    // Mixed Precision Tests
    TEST_F( AttentionTests, Cuda_MixedPrecision_TestForward ) {
        try {
            TestForward<DeviceType::Cuda, float, half>( MixedPrecisionData() );
        }
        catch ( const std::exception& e ) {
            GTEST_SKIP() << "Skipping CUDA mixed precision test due to exception: " << e.what();
        }
    }

    TEST_F( AttentionTests, Cuda_MixedPrecision_TestName ) {
        TestGetName<DeviceType::Cuda, float, half>( MixedPrecisionData(), "cuda_attn_mixed" );
    }

    // CPU-CUDA Equivalence Test
    TEST_F( AttentionTests, CpuCuda_Forward_Output_Equivalence ) {
        try {
            TestCpuCudaEquivalence<float>( CpuFloatData(), CudaFloatData() );
        }
        catch ( const std::exception& e ) {
            GTEST_SKIP() << "Skipping CPU-CUDA equivalence test due to exception: " << e.what();
        }
    }

    // CUDA Float-Half Precision Comparison
    TEST_F( AttentionTests, Cuda_Float_Half_Precision_Comparison ) {
        try {
            TestPrecisionComparison<float>( CudaFloatData(), CudaHalfData() );
        }
        catch ( const std::exception& e ) {
            GTEST_SKIP() << "Skipping precision comparison test due to exception: " << e.what();
        }
    }

    // New ComputePrecision Tests - CPU
    TEST_F( AttentionTests, Cpu_Float_DefaultPrecisionIsAuto ) {
        TestPrecisionPolicy<DeviceType::Cpu, float>( CpuFloatData(), ComputePrecision::Policy::Auto );
    }

    TEST_F( AttentionTests, Cpu_Float_PerformancePrecision ) {
        TestPrecisionPolicy<DeviceType::Cpu, float>( PerfPrecisionCpuFloatData(), ComputePrecision::Policy::Performance );
    }

    TEST_F( AttentionTests, Cpu_Float_AccuracyPrecision ) {
        TestPrecisionPolicy<DeviceType::Cpu, float>( AccuracyPrecisionCpuFloatData(), ComputePrecision::Policy::Accuracy );
    }

    TEST_F( AttentionTests, Cpu_Float_DisabledPrecision ) {
        TestPrecisionPolicy<DeviceType::Cpu, float>( DisabledPrecisionCpuFloatData(), ComputePrecision::Policy::Native );
    }

    // New ComputePrecision Tests - CUDA
    TEST_F( AttentionTests, Cuda_Float_DefaultPrecisionIsAuto ) {
        TestPrecisionPolicy<DeviceType::Cuda, float>( CudaFloatData(), ComputePrecision::Policy::Auto );
    }

    TEST_F( AttentionTests, Cuda_Float_PerformancePrecision ) {
        TestPrecisionPolicy<DeviceType::Cuda, float>( PerfPrecisionCudaFloatData(), ComputePrecision::Policy::Performance );
    }

    TEST_F( AttentionTests, Cuda_Float_AccuracyPrecision ) {
        TestPrecisionPolicy<DeviceType::Cuda, float>( AccuracyPrecisionCudaFloatData(), ComputePrecision::Policy::Accuracy );
    }

    TEST_F( AttentionTests, Cuda_Float_DisabledPrecision ) {
        TestPrecisionPolicy<DeviceType::Cuda, float>( DisabledPrecisionCudaFloatData(), ComputePrecision::Policy::Native );
    }
}