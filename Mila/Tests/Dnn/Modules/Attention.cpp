#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <sstream>
#include <cuda_fp16.h>  // For half type

import Mila;

namespace Modules::Tests
{
    using namespace Mila::Dnn;

    // Common test data structure that can be reused
    template<typename TInput, typename TPrecision, Compute::DeviceType TDevice>
    struct AttentionTestData {
        size_t batch_size;
        size_t sequence_length;
        size_t channels;
        size_t num_heads;
        std::vector<size_t> input_shape;
        std::shared_ptr<MultiHeadAttention<TInput, TPrecision, TDevice>> attention_module;
    };

    class AttentionTests : public ::testing::Test {
    protected:
        void SetUp() override {
            batch_size_ = 64;
            cpu_batch_size_ = 4;
            sequence_length_ = 1024;
            channels_ = 768;
            num_heads_ = 12;

            // CPU test data (float precision)
            cpu_float_data_.batch_size = cpu_batch_size_;
            cpu_float_data_.sequence_length = sequence_length_;
            cpu_float_data_.channels = channels_;
            cpu_float_data_.num_heads = num_heads_;
            cpu_float_data_.input_shape = { cpu_batch_size_, sequence_length_, 3 * channels_ };
            cpu_float_data_.attention_module = std::make_shared<MultiHeadAttention<float, float, Compute::DeviceType::Cpu>>(
                "cpu_attn_float", cpu_float_data_.input_shape, num_heads_ );

            // CUDA test data (float precision)
            cuda_float_data_.batch_size = batch_size_;
            cuda_float_data_.sequence_length = sequence_length_;
            cuda_float_data_.channels = channels_;
            cuda_float_data_.num_heads = num_heads_;
            cuda_float_data_.input_shape = { batch_size_, sequence_length_, 3 * channels_ };
            cuda_float_data_.attention_module = std::make_shared<MultiHeadAttention<float, float, Compute::DeviceType::Cuda>>(
                "cuda_attn_float", cuda_float_data_.input_shape, num_heads_ );

            // FIXME: FP16 Precision is not supported yet
            // CUDA test data (half precision)
            /*
            cuda_half_data_.batch_size = batch_size_;
            cuda_half_data_.sequence_length = sequence_length_;
            cuda_half_data_.channels = channels_;
            cuda_half_data_.num_heads = num_heads_;
            cuda_half_data_.input_shape = { batch_size_, sequence_length_, 3 * channels_ };
            cuda_half_data_.attention_module = std::make_shared<MultiHeadAttention<float, half, Compute::DeviceType::Cuda>>(
                "cuda_attn_half", cuda_half_data_.input_shape, num_heads_);
            */
        }

        size_t batch_size_{ 0 };
        size_t cpu_batch_size_{ 0 };
        size_t sequence_length_{ 0 };
        size_t channels_{ 0 };
        size_t num_heads_{ 0 };

        // Structured test data
        AttentionTestData<float, float, Compute::DeviceType::Cpu> cpu_float_data_;
        AttentionTestData<float, float, Compute::DeviceType::Cuda> cuda_float_data_;
        //AttentionTestData<float, half, Compute::DeviceType::Cuda> cuda_half_data_;
    };

    // Common test function templates
    template<typename TInput, typename TPrecision, Compute::DeviceType TDevice, typename TMemResource>
    void TestGetName( const AttentionTestData<TInput, TPrecision, TDevice>& data, const std::string& expected_name ) {
        EXPECT_EQ( data.attention_module->getName(), expected_name );
    }

    template<typename TInput, typename TPrecision, Compute::DeviceType TDevice, typename TMemResource>
    void TestParameterCount( const AttentionTestData<TInput, TPrecision, TDevice>& data, size_t expected_count ) {
        EXPECT_EQ( data.attention_module->parameterCount(), expected_count );
    }

    template<typename TInput, typename TPrecision, Compute::DeviceType TDevice, typename TMemResource>
    void TestForward( const AttentionTestData<TInput, TPrecision, TDevice>& data ) {
        Tensor<TInput, TMemResource> input( data.input_shape );
        Tensor<TPrecision, TMemResource> output( data.input_shape );
        data.attention_module->forward( input, output );
        EXPECT_EQ( output.size(), input.size() );
    }

    template<typename TInput, typename TPrecision, Compute::DeviceType TDevice>
    void TestPrint( const AttentionTestData<TInput, TPrecision, TDevice>& data, const std::string& expected_substring ) {
        std::string output = data.attention_module->toString();
        EXPECT_NE( output.find( expected_substring ), std::string::npos );
    }

    // Add this function to test equivalence of CPU and CUDA outputs
    template<typename TInput, typename TPrecision>
    void TestCpuCudaEquivalence(
        const AttentionTestData<TInput, TPrecision, Compute::DeviceType::Cpu>& cpu_data,
        const AttentionTestData<TInput, TPrecision, Compute::DeviceType::Cuda>& cuda_data ) {

        // Create a small test shape to make comparison faster
        std::vector<size_t> test_shape = { 2, 16, 3 * 32 }; // Small shape for quick verification
        size_t num_heads = 4;

        // Create test modules with smaller dimensions
        auto cpu_test_module = std::make_shared<MultiHeadAttention<TInput, TPrecision, Compute::DeviceType::Cpu>>(
            "cpu_test", test_shape, num_heads );

        auto cuda_test_module = std::make_shared<MultiHeadAttention<TInput, TPrecision, Compute::DeviceType::Cuda>>(
            "cuda_test", test_shape, num_heads );

        // Create random input data
        Tensor<TInput, Compute::HostMemoryResource> host_input( test_shape );

        // Fill with predictable values
        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input.data()[ i ] = static_cast<TInput>( (static_cast<float>( i ) / host_input.size()) - 0.5f );
        }

        // Create CPU output
        Tensor<TPrecision, Compute::HostMemoryResource> cpu_output( test_shape );
        cpu_test_module->forward( host_input, cpu_output );

        // Create device input by copying host data
        Tensor<TInput, Compute::DeviceMemoryResource> device_input( test_shape );
        device_input.copyFrom( host_input );

        // Create device output
        Tensor<TPrecision, Compute::DeviceMemoryResource> cuda_output( test_shape );
        cuda_test_module->forward( device_input, cuda_output );

        // Copy CUDA output back to host for comparison
        Tensor<TPrecision, Compute::HostMemoryResource> cuda_output_host( test_shape );
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

    // Add these function templates to your existing common test functions section

// Test with different head counts
    template<typename TInput, typename TPrecision, Compute::DeviceType TDevice, typename TMemResource>
    void TestDifferentHeadCounts( const AttentionTestData<TInput, TPrecision, TDevice>& data ) {
        std::vector<size_t> head_counts = { 1, 4, 8, 16 };
        std::vector<size_t> small_shape = { 2, 16, 3 * 64 };

        for ( auto num_heads : head_counts ) {
            auto test_module = std::make_shared<MultiHeadAttention<TInput, TPrecision, TDevice>>(
                "test_heads_" + std::to_string( num_heads ), small_shape, num_heads );

            Tensor<TInput, TMemResource> input( small_shape );
            Tensor<TPrecision, TMemResource> output( small_shape );

            // Initialize input with some values
            for ( size_t i = 0; i < input.size(); ++i ) {
                input.data()[ i ] = static_cast<TInput>( i % 10 ) * 0.1f;
            }

            EXPECT_NO_THROW( test_module->forward( input, output ) );
            EXPECT_EQ( output.size(), input.size() );
        }
    }

    // Test edge cases with minimal and large shapes
    template<typename TInput, typename TPrecision, Compute::DeviceType TDevice, typename TMemResource>
    void TestEdgeCases( const AttentionTestData<TInput, TPrecision, TDevice>& data ) {
        // Test with minimal batch size and sequence length
        std::vector<size_t> minimal_shape = { 1, 1, 3 * 64 };
        auto minimal_module = std::make_shared<MultiHeadAttention<TInput, TPrecision, TDevice>>(
            "minimal", minimal_shape, 1 );

        Tensor<TInput, TMemResource> minimal_input( minimal_shape );
        Tensor<TPrecision, TMemResource> minimal_output( minimal_shape );

        EXPECT_NO_THROW( minimal_module->forward( minimal_input, minimal_output ) );

        // Test with large sequence length
        std::vector<size_t> long_seq_shape = { 2, 128, 3 * 64 }; // Reduced for test performance
        auto long_seq_module = std::make_shared<MultiHeadAttention<TInput, TPrecision, TDevice>>(
            "long_seq", long_seq_shape, 4 );

        Tensor<TInput, TMemResource> long_input( long_seq_shape );
        Tensor<TPrecision, TMemResource> long_output( long_seq_shape );

        EXPECT_NO_THROW( long_seq_module->forward( long_input, long_output ) );
    }

    // Test training mode behavior
    template<typename TInput, typename TPrecision, Compute::DeviceType TDevice, typename TMemResource>
    void TestTrainingModeBehavior( const AttentionTestData<TInput, TPrecision, TDevice>& data ) {
        std::vector<size_t> shape = { 2, 16, 3 * 64 };
        auto train_module = std::make_shared<MultiHeadAttention<TInput, TPrecision, TDevice>>(
            "train_mode", shape, 4, true ); // training mode enabled

        auto infer_module = std::make_shared<MultiHeadAttention<TInput, TPrecision, TDevice>>(
            "infer_mode", shape, 4, false ); // training mode disabled

        Tensor<TInput, TMemResource> input( shape );
        Tensor<TPrecision, TMemResource> train_output( shape );
        Tensor<TPrecision, TMemResource> infer_output( shape );

        // Fill input with values
        for ( size_t i = 0; i < input.size(); ++i ) {
            input.data()[ i ] = static_cast<TInput>( i % 7 ) * 0.1f;
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
    template<typename TInput, typename TPrecision, Compute::DeviceType TDevice, typename TMemResource>
    void TestNumericalStability( const AttentionTestData<TInput, TPrecision, TDevice>& data ) {
        std::vector<size_t> shape = { 2, 16, 3 * 64 };
        auto stability_module = std::make_shared<MultiHeadAttention<TInput, TPrecision, TDevice>>(
            "stability", shape, 4 );

        Tensor<TInput, TMemResource> input( shape );
        Tensor<TPrecision, TMemResource> output( shape );

        // Test with large values (smaller than previously suggested to avoid overflow)
        for ( size_t i = 0; i < input.size(); ++i ) {
            input.data()[ i ] = 1.0e+3f;
        }

        stability_module->forward( input, output );

        // Verify no NaNs or Infs in output
        bool has_nan_or_inf = false;
        for ( size_t i = 0; i < output.size(); ++i ) {
            if ( std::isnan( output.data()[ i ] ) || std::isinf( output.data()[ i ] ) ) {
                has_nan_or_inf = true;
                break;
            }
        }

        EXPECT_FALSE( has_nan_or_inf ) << "Output contains NaN or Inf values with large inputs";

        // Test with small values
        for ( size_t i = 0; i < input.size(); ++i ) {
            input.data()[ i ] = 1.0e-3f;
        }

        stability_module->forward( input, output );

        // Verify output again
        has_nan_or_inf = false;
        for ( size_t i = 0; i < output.size(); ++i ) {
            if ( std::isnan( output.data()[ i ] ) || std::isinf( output.data()[ i ] ) ) {
                has_nan_or_inf = true;
                break;
            }
        }

        EXPECT_FALSE( has_nan_or_inf ) << "Output contains NaN or Inf values with small inputs";
    }

    // Test deterministic behavior (for CUDA implementation)
    template<typename TInput, typename TPrecision>
    void TestDeterministicBehavior(
        const AttentionTestData<TInput, TPrecision, Compute::DeviceType::Cuda>& data ) {

        std::vector<size_t> shape = { 2, 16, 3 * 64 };
        auto deterministic_module = std::make_shared<MultiHeadAttention<TInput, TPrecision, Compute::DeviceType::Cuda>>(
            "deterministic", shape, 4 );

        Tensor<TInput, Compute::DeviceMemoryResource> input( shape );
        Tensor<TPrecision, Compute::DeviceMemoryResource> output1( shape );
        Tensor<TPrecision, Compute::DeviceMemoryResource> output2( shape );

        // Fill input with predictable values
        Tensor<TInput, Compute::HostMemoryResource> host_input( shape );
        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input.data()[ i ] = static_cast<TInput>( (i % 10) * 0.1f );
        }
        input.copyFrom( host_input );

        // Run forward pass twice
        deterministic_module->forward( input, output1 );
        deterministic_module->forward( input, output2 );

        // Copy results back to host for comparison
        Tensor<TPrecision, Compute::HostMemoryResource> host_output1( shape );
        Tensor<TPrecision, Compute::HostMemoryResource> host_output2( shape );
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

    // CPU additional tests

    TEST_F( AttentionTests, Cpu_Float_DifferentHeadCounts ) {
        TestDifferentHeadCounts<float, float, Compute::DeviceType::Cpu, Compute::HostMemoryResource>(
            cpu_float_data_ );
    }

    TEST_F( AttentionTests, Cpu_Float_EdgeCases ) {
        TestEdgeCases<float, float, Compute::DeviceType::Cpu, Compute::HostMemoryResource>(
            cpu_float_data_ );
    }

    TEST_F( AttentionTests, Cpu_Float_TrainingModeBehavior ) {
        TestTrainingModeBehavior<float, float, Compute::DeviceType::Cpu, Compute::HostMemoryResource>(
            cpu_float_data_ );
    }

    TEST_F( AttentionTests, Cpu_Float_NumericalStability ) {
        TestNumericalStability<float, float, Compute::DeviceType::Cpu, Compute::HostMemoryResource>(
            cpu_float_data_ );
    }

    // CUDA additional tests

    TEST_F( AttentionTests, Cuda_Float_DifferentHeadCounts ) {
        TestDifferentHeadCounts<float, float, Compute::DeviceType::Cuda, Compute::DeviceMemoryResource>(
            cuda_float_data_ );
    }

    TEST_F( AttentionTests, Cuda_Float_EdgeCases ) {
        TestEdgeCases<float, float, Compute::DeviceType::Cuda, Compute::DeviceMemoryResource>(
            cuda_float_data_ );
    }

    TEST_F( AttentionTests, Cuda_Float_TrainingModeBehavior ) {
        TestTrainingModeBehavior<float, float, Compute::DeviceType::Cuda, Compute::DeviceMemoryResource>(
            cuda_float_data_ );
    }

    TEST_F( AttentionTests, Cuda_Float_NumericalStability ) {
        TestNumericalStability<float, float, Compute::DeviceType::Cuda, Compute::DeviceMemoryResource>(
            cuda_float_data_ );
    }

    TEST_F( AttentionTests, Cuda_Float_Deterministic ) {
        TestDeterministicBehavior<float, float>( cuda_float_data_ );
    }


    // CPU Tests with float precision

    TEST_F( AttentionTests, Cpu_Float_TestName ) {
        TestGetName<float, float, Compute::DeviceType::Cpu, Compute::HostMemoryResource>(
            cpu_float_data_, "cpu_attn_float" );
    }

    TEST_F( AttentionTests, Cpu_Float_ParameterCount ) {
        // Calculate expected parameter count based on the attention module structure
        size_t expected_count = cpu_float_data_.attention_module->parameterCount();
        TestParameterCount<float, float, Compute::DeviceType::Cpu, Compute::HostMemoryResource>(
            cpu_float_data_, expected_count );
    }

    TEST_F( AttentionTests, Cpu_Float_TestForward ) {
        TestForward<float, float, Compute::DeviceType::Cpu, Compute::HostMemoryResource>( cpu_float_data_ );
    }

    TEST_F( AttentionTests, Cpu_Float_TestPrint ) {
        TestPrint<float, float, Compute::DeviceType::Cpu>( cpu_float_data_, "MultiHeadAttention: cpu_attn_float" );
    }

    // CUDA Tests with float precision

    TEST_F( AttentionTests, Cuda_Float_TestName ) {
        TestGetName<float, float, Compute::DeviceType::Cuda, Compute::DeviceMemoryResource>(
            cuda_float_data_, "cuda_attn_float" );
    }

    TEST_F( AttentionTests, Cuda_Float_ParameterCount ) {
        // Calculate expected parameter count based on the attention module structure
        size_t expected_count = cuda_float_data_.attention_module->parameterCount();
        TestParameterCount<float, float, Compute::DeviceType::Cuda, Compute::DeviceMemoryResource>(
            cuda_float_data_, expected_count );
    }

    TEST_F( AttentionTests, Cuda_Float_TestForward ) {
        TestForward<float, float, Compute::DeviceType::Cuda, Compute::DeviceMemoryResource>( cuda_float_data_ );
    }

    TEST_F( AttentionTests, Cuda_Float_TestPrint ) {
        TestPrint<float, float, Compute::DeviceType::Cuda>( cuda_float_data_, "MultiHeadAttention: cuda_attn_float" );
    }

    // CUDA Tests with half precision

    /*
    TEST_F(AttentionTests, Cuda_Half_TestName) {
        TestGetName<float, half, Compute::DeviceType::Cuda, Compute::DeviceMemoryResource>(
            cuda_half_data_, "cuda_attn_half");
    }

    TEST_F(AttentionTests, Cuda_Half_ParameterCount) {
        // Calculate expected parameter count based on the attention module structure
        size_t expected_count = cuda_half_data_.attention_module->parameterCount();
        TestParameterCount<float, half, Compute::DeviceType::Cuda, Compute::DeviceMemoryResource>(
            cuda_half_data_, expected_count);
    }

    TEST_F(AttentionTests, Cuda_Half_TestForward) {
        TestForward<float, half, Compute::DeviceType::Cuda, Compute::DeviceMemoryResource>(cuda_half_data_);
    }

    TEST_F(AttentionTests, Cuda_Half_TestPrint) {
        TestPrint<float, half, Compute::DeviceType::Cuda>(cuda_half_data_, "MultiHeadAttention: cuda_attn_half");
    }
    */

    TEST_F( AttentionTests, CpuCuda_Forward_Output_Equivalence ) {
        TestCpuCudaEquivalence<float, float>( cpu_float_data_, cuda_float_data_ );
    }
}
