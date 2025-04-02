#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <cuda_fp16.h>  // For half type

import Mila;

namespace Modules::Tests
{
    using namespace Mila::Dnn;

    // Common test data structure that can be reused
    template<typename TInput, typename TPrecision, Compute::DeviceType TDevice>
    struct LayerNormTestData {
        std::vector<size_t> input_shape;
        std::shared_ptr<LayerNorm<TInput, TPrecision, TDevice>> ln_module;
        bool has_bias;
        int64_t axis;
    };

    class LayerNormTests : public ::testing::Test {
    protected:
        void SetUp() override {
            batch_size_ = 128;
            cpu_batch_size_ = 4;
            sequence_length_ = 1024;
            channels_ = 768;
            has_bias_ = true;
            axis_ = -1; // Normalize across the last dimension (channels)

            // CPU test data (float precision)
            cpu_float_data_.input_shape = { cpu_batch_size_, sequence_length_, channels_ };
            cpu_float_data_.has_bias = has_bias_;
            cpu_float_data_.axis = axis_;
            cpu_float_data_.ln_module = std::make_shared<LayerNorm<float, float, Compute::DeviceType::Cpu>>(
                "cpu_ln_float", cpu_float_data_.input_shape, axis_, has_bias_ );

            // CPU test data (float precision) without bias
            cpu_no_bias_float_data_.input_shape = { cpu_batch_size_, sequence_length_, channels_ };
            cpu_no_bias_float_data_.has_bias = false;
            cpu_no_bias_float_data_.axis = axis_;
            cpu_no_bias_float_data_.ln_module = std::make_shared<LayerNorm<float, float, Compute::DeviceType::Cpu>>(
                "cpu_ln_no_bias_float", cpu_no_bias_float_data_.input_shape, axis_, false );

            // CUDA test data (float precision)
            cuda_float_data_.input_shape = { batch_size_, sequence_length_, channels_ };
            cuda_float_data_.has_bias = has_bias_;
            cuda_float_data_.axis = axis_;
            cuda_float_data_.ln_module = std::make_shared<LayerNorm<float, float, Compute::DeviceType::Cuda>>(
                "cuda_ln_float", cuda_float_data_.input_shape, axis_, has_bias_ );

            // CUDA test data (float precision) without bias
            cuda_no_bias_float_data_.input_shape = { batch_size_, sequence_length_, channels_ };
            cuda_no_bias_float_data_.has_bias = false;
            cuda_no_bias_float_data_.axis = axis_;
            cuda_no_bias_float_data_.ln_module = std::make_shared<LayerNorm<float, float, Compute::DeviceType::Cuda>>(
                "cuda_ln_no_bias_float", cuda_no_bias_float_data_.input_shape, axis_, false );

            // Training mode modules
            training_cpu_float_data_.input_shape = { cpu_batch_size_, sequence_length_, channels_ };
            training_cpu_float_data_.has_bias = has_bias_;
            training_cpu_float_data_.axis = axis_;
            training_cpu_float_data_.ln_module = std::make_shared<LayerNorm<float, float, Compute::DeviceType::Cpu>>(
                "cpu_ln_float_training", training_cpu_float_data_.input_shape, axis_, has_bias_, true );

            training_cuda_float_data_.input_shape = { batch_size_, sequence_length_, channels_ };
            training_cuda_float_data_.has_bias = has_bias_;
            training_cuda_float_data_.axis = axis_;
            training_cuda_float_data_.ln_module = std::make_shared<LayerNorm<float, float, Compute::DeviceType::Cuda>>(
                "cuda_ln_float_training", training_cuda_float_data_.input_shape, axis_, has_bias_, true );
        }

        size_t batch_size_{ 0 };
        size_t cpu_batch_size_{ 0 };
        size_t sequence_length_{ 0 };
        size_t channels_{ 0 };
        bool has_bias_{ true };
        int64_t axis_{ -1 };

        // Structured test data
        LayerNormTestData<float, float, Compute::DeviceType::Cpu> cpu_float_data_;
        LayerNormTestData<float, float, Compute::DeviceType::Cpu> cpu_no_bias_float_data_;
        LayerNormTestData<float, float, Compute::DeviceType::Cuda> cuda_float_data_;
        LayerNormTestData<float, float, Compute::DeviceType::Cuda> cuda_no_bias_float_data_;
        LayerNormTestData<float, float, Compute::DeviceType::Cpu> training_cpu_float_data_;
        LayerNormTestData<float, float, Compute::DeviceType::Cuda> training_cuda_float_data_;
    };

    // Common test function templates
    template<typename TInput, typename TPrecision, Compute::DeviceType TDevice, typename TMemResource>
    void TestGetName( const LayerNormTestData<TInput, TPrecision, TDevice>& data, const std::string& expected_name ) {
        EXPECT_EQ( data.ln_module->getName(), expected_name );
    }

    template<typename TInput, typename TPrecision, Compute::DeviceType TDevice, typename TMemResource>
    void TestParameterCount( const LayerNormTestData<TInput, TPrecision, TDevice>& data ) {
        // Determine expected parameter count based on normalization dimension and bias
        size_t channels = data.input_shape[ 2 ]; // Last dimension is channels
        size_t expected_count = channels; // Weight parameters

        if ( data.has_bias ) {
            expected_count += channels; // Bias parameters
        }

        EXPECT_EQ( data.ln_module->parameterCount(), expected_count );
    }

    template<typename TInput, typename TPrecision, Compute::DeviceType TDevice, typename TMemResource>
    void TestForward( const LayerNormTestData<TInput, TPrecision, TDevice>& data ) {
        Tensor<TInput, TMemResource> input( data.input_shape );
        Tensor<TPrecision, TMemResource> output( data.input_shape );

        // Fill input with predictable values
        for ( size_t i = 0; i < input.size(); ++i ) {
            input.data()[ i ] = static_cast<TInput>( (i % 10) * 0.1f );
        }

        data.ln_module->forward( input, output );
        EXPECT_EQ( output.size(), input.size() );
    }

    template<typename TInput, typename TPrecision, Compute::DeviceType TDevice, typename TMemResource>
    void TestSimpleNormalization() {
        // Create a small, predictable input
        std::vector<size_t> shape = { 1, 2, 3 };
        Tensor<TInput, TMemResource> input( shape );
        Tensor<TPrecision, TMemResource> output( shape );

        // Initialize with known values
        input.data()[ 0 ] = 1.0f;
        input.data()[ 1 ] = 2.0f;
        input.data()[ 2 ] = 3.0f;
        input.data()[ 3 ] = 4.0f;
        input.data()[ 4 ] = 5.0f;
        input.data()[ 5 ] = 6.0f;

        // Create LayerNorm module with all weights=1 and all biases=0
        auto ln = std::make_shared<LayerNorm<TInput, TPrecision, TDevice>>(
            "test_simple_ln", shape, -1, true );

        // Get weight and bias tensors and set them to known values
        auto weight = ln->getWeight();
        auto bias = ln->getBias();

        for ( size_t i = 0; i < weight->size(); ++i ) {
            weight->data()[ i ] = 1.0f;
        }

        for ( size_t i = 0; i < bias->size(); ++i ) {
            bias->data()[ i ] = 0.0f;
        }

        // Perform forward pass
        ln->forward( input, output );

        // Expected output for a normalized sequence with mean=0 and var=1
        const float tolerance = 1e-5f;

        // Row 1: [1,2,3] normalized should be approximately [-1.22474, 0, 1.22474]
        EXPECT_NEAR( output.data()[ 0 ], -1.22474f, tolerance );
        EXPECT_NEAR( output.data()[ 1 ], 0.0f, tolerance );
        EXPECT_NEAR( output.data()[ 2 ], 1.22474f, tolerance );

        // Row 2: [4,5,6] normalized should be approximately [-1.22474, 0, 1.22474]
        EXPECT_NEAR( output.data()[ 3 ], -1.22474f, tolerance );
        EXPECT_NEAR( output.data()[ 4 ], 0.0f, tolerance );
        EXPECT_NEAR( output.data()[ 5 ], 1.22474f, tolerance );
    }

    template<typename TInput, typename TPrecision, Compute::DeviceType TDevice>
    void TestPrint( const LayerNormTestData<TInput, TPrecision, TDevice>& data, const std::string& expected_substring ) {
        std::string output = data.ln_module->toString();
        EXPECT_NE( output.find( expected_substring ), std::string::npos );
    }

    // Function to test equivalence of CPU and CUDA outputs
    template<typename TInput, typename TPrecision>
    void TestCpuCudaEquivalence(
        const LayerNormTestData<TInput, TPrecision, Compute::DeviceType::Cpu>& cpu_data,
        const LayerNormTestData<TInput, TPrecision, Compute::DeviceType::Cuda>& cuda_data ) {

        // Create a small test shape to make comparison faster
        std::vector<size_t> test_shape = { 2, 4, 8 }; // Small shape for quick verification

        // Create test modules with smaller dimensions
        auto cpu_test_module = std::make_shared<LayerNorm<TInput, TPrecision, Compute::DeviceType::Cpu>>(
            "cpu_test", test_shape, -1, cpu_data.has_bias );

        auto cuda_test_module = std::make_shared<LayerNorm<TInput, TPrecision, Compute::DeviceType::Cuda>>(
            "cuda_test", test_shape, -1, cuda_data.has_bias );

        // Create random input data
        Tensor<TInput, Compute::HostMemoryResource> host_input( test_shape );

        // Fill with predictable values between -1 and 1
        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input.data()[ i ] = static_cast<TInput>( -1.0 + 2.0 * (static_cast<float>( i ) / host_input.size()) );
        }

        // Copy parameters from CPU module to CUDA module for fair comparison
        auto cpu_weight = cpu_test_module->getWeight();
        auto cuda_weight = cuda_test_module->getWeight();

        Tensor<TPrecision, Compute::DeviceMemoryResource> device_weight( cpu_weight->shape() );
        device_weight.copyFrom( *cpu_weight );
        cuda_weight->copyFrom( device_weight );

        if ( cpu_data.has_bias && cuda_data.has_bias ) {
            auto cpu_bias = cpu_test_module->getBias();
            auto cuda_bias = cuda_test_module->getBias();

            Tensor<TPrecision, Compute::DeviceMemoryResource> device_bias( cpu_bias->shape() );
            device_bias.copyFrom( *cpu_bias );
            cuda_bias->copyFrom( device_bias );
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
        const float epsilon = 1e-4f;
        bool all_equal = true;

        for ( size_t i = 0; i < cpu_output.size(); ++i ) {
            float diff = std::abs( static_cast<float>( cpu_output.data()[ i ] ) - static_cast<float>( cuda_output_host.data()[ i ] ) );
            if ( diff > epsilon ) {
                std::cout << "Difference at index " << i << ": CPU=" << cpu_output.data()[ i ]
                    << ", CUDA=" << cuda_output_host.data()[ i ] << ", diff=" << diff << std::endl;
                    all_equal = false;
                    // Only show first few differences to avoid flooding the output
                    if ( i > 10 ) break;
            }
        }

        EXPECT_TRUE( all_equal ) << "CPU and CUDA implementations produced different results";
    }

    // Test edge cases with minimal and large shapes
    template<typename TInput, typename TPrecision, Compute::DeviceType TDevice, typename TMemResource>
    void TestEdgeCases() {
        // Test with minimal shape
        std::vector<size_t> minimal_shape = { 1, 1, 4 };
        auto minimal_module = std::make_shared<LayerNorm<TInput, TPrecision, TDevice>>(
            "minimal", minimal_shape );

        Tensor<TInput, TMemResource> minimal_input( minimal_shape );
        Tensor<TPrecision, TMemResource> minimal_output( minimal_shape );

        // Fill with values
        for ( size_t i = 0; i < minimal_input.size(); ++i ) {
            minimal_input.data()[ i ] = static_cast<TInput>( i + 1 );
        }

        EXPECT_NO_THROW( minimal_module->forward( minimal_input, minimal_output ) );

        // Test with single value per channel (edge case for normalization)
        std::vector<size_t> single_shape = { 1, 1, 1 };
        auto single_module = std::make_shared<LayerNorm<TInput, TPrecision, TDevice>>(
            "single", single_shape );

        Tensor<TInput, TMemResource> single_input( single_shape );
        Tensor<TPrecision, TMemResource> single_output( single_shape );

        single_input.data()[ 0 ] = 42.0f;  // Any value should normalize to 0 since stddev is 0

        EXPECT_NO_THROW( single_module->forward( single_input, single_output ) );
        // For a single value, mean = value, so normalized should be 0 before weight/bias
        // After applying weight=1, bias=0, it should still be 0
        EXPECT_NEAR( single_output.data()[ 0 ], 0.0f, 1e-5f );
    }

    // Test training mode behavior
    template<typename TInput, typename TPrecision, Compute::DeviceType TDevice, typename TMemResource>
    void TestTrainingModeBehavior(
        const LayerNormTestData<TInput, TPrecision, TDevice>& training_data,
        const LayerNormTestData<TInput, TPrecision, TDevice>& inference_data ) {

        // Verify training status
        EXPECT_TRUE( training_data.ln_module->isTraining() );
        EXPECT_FALSE( inference_data.ln_module->isTraining() );

        // Test mode switching
        training_data.ln_module->setTraining( false );
        EXPECT_FALSE( training_data.ln_module->isTraining() );

        inference_data.ln_module->setTraining( true );
        EXPECT_TRUE( inference_data.ln_module->isTraining() );

        // Test that both can process data regardless of training mode
        std::vector<size_t> test_shape = { 2, 3, 4 };
        Tensor<TInput, TMemResource> input( test_shape );
        Tensor<TPrecision, TMemResource> train_output( test_shape );
        Tensor<TPrecision, TMemResource> infer_output( test_shape );

        // Fill with test values
        for ( size_t i = 0; i < input.size(); ++i ) {
            input.data()[ i ] = static_cast<TInput>( i * 0.1f );
        }

        // Both should be able to process data without errors
        EXPECT_NO_THROW( training_data.ln_module->forward( input, train_output ) );
        EXPECT_NO_THROW( inference_data.ln_module->forward( input, infer_output ) );

        // Reset for other tests
        training_data.ln_module->setTraining( true );
        inference_data.ln_module->setTraining( false );
    }

    // Test numerical stability with different input scales
    template<typename TInput, typename TPrecision, Compute::DeviceType TDevice, typename TMemResource>
    void TestNumericalStability() {
        std::vector<size_t> shape = { 2, 4, 8 };
        auto stability_module = std::make_shared<LayerNorm<TInput, TPrecision, TDevice>>(
            "stability", shape );

        Tensor<TInput, TMemResource> large_input( shape );
        Tensor<TInput, TMemResource> small_input( shape );
        Tensor<TPrecision, TMemResource> large_output( shape );
        Tensor<TPrecision, TMemResource> small_output( shape );

        // Test with large values
        for ( size_t i = 0; i < large_input.size(); ++i ) {
            large_input.data()[ i ] = 1.0e+4f;
        }

        stability_module->forward( large_input, large_output );

        // Verify no NaNs or Infs in output
        bool has_nan_or_inf = false;
        for ( size_t i = 0; i < large_output.size(); ++i ) {
            if ( std::isnan( large_output.data()[ i ] ) || std::isinf( large_output.data()[ i ] ) ) {
                has_nan_or_inf = true;
                break;
            }
        }

        EXPECT_FALSE( has_nan_or_inf ) << "Output contains NaN or Inf values with large inputs";

        // Test with small values
        for ( size_t i = 0; i < small_input.size(); ++i ) {
            small_input.data()[ i ] = 1.0e-6f;
        }

        stability_module->forward( small_input, small_output );

        // Verify output again
        has_nan_or_inf = false;
        for ( size_t i = 0; i < small_output.size(); ++i ) {
            if ( std::isnan( small_output.data()[ i ] ) || std::isinf( small_output.data()[ i ] ) ) {
                has_nan_or_inf = true;
                break;
            }
        }

        EXPECT_FALSE( has_nan_or_inf ) << "Output contains NaN or Inf values with small inputs";
    }

    // Test deterministic behavior (for CUDA implementation)
    template<typename TInput, typename TPrecision>
    void TestDeterministicBehavior() {
        std::vector<size_t> shape = { 2, 4, 8 };
        auto deterministic_module = std::make_shared<LayerNorm<TInput, TPrecision, Compute::DeviceType::Cuda>>(
            "deterministic", shape );

        Tensor<TInput, Compute::HostMemoryResource> host_input( shape );
        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input.data()[ i ] = static_cast<TInput>( i * 0.01f );
        }

        Tensor<TInput, Compute::DeviceMemoryResource> device_input( shape );
        device_input.copyFrom( host_input );

        Tensor<TPrecision, Compute::DeviceMemoryResource> output1( shape );
        Tensor<TPrecision, Compute::DeviceMemoryResource> output2( shape );

        // Run forward pass twice
        deterministic_module->forward( device_input, output1 );
        deterministic_module->forward( device_input, output2 );

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
                std::cout << "Mismatch at index " << i << ": "
                    << host_output1.data()[ i ] << " vs "
                    << host_output2.data()[ i ] << std::endl;
                break;
            }
        }

        EXPECT_TRUE( outputs_match ) << "Multiple runs with the same input produced different results";
    }

    // Mock test for save/load functionality
    template<typename TInput, typename TPrecision, Compute::DeviceType TDevice, typename TMemResource>
    void TestSaveLoadMockup( const LayerNormTestData<TInput, TPrecision, TDevice>& data ) {
        // This is a mock test as we don't have actual save/load implemented
        auto weight = data.ln_module->getWeight();
        EXPECT_NE( weight, nullptr );
        EXPECT_EQ( weight->size(), data.input_shape[ 2 ] );

        if ( data.has_bias ) {
            auto bias = data.ln_module->getBias();
            EXPECT_NE( bias, nullptr );
            EXPECT_EQ( bias->size(), data.input_shape[ 2 ] );
        }
    }

    // CPU Tests with float precision
    TEST_F( LayerNormTests, Cpu_Float_TestName ) {
        TestGetName<float, float, Compute::DeviceType::Cpu, Compute::HostMemoryResource>(
            cpu_float_data_, "cpu_ln_float" );
    }

    TEST_F( LayerNormTests, Cpu_Float_ParameterCount ) {
        TestParameterCount<float, float, Compute::DeviceType::Cpu, Compute::HostMemoryResource>(
            cpu_float_data_ );
    }

    TEST_F( LayerNormTests, Cpu_Float_TestForward ) {
        TestForward<float, float, Compute::DeviceType::Cpu, Compute::HostMemoryResource>(
            cpu_float_data_ );
    }

    TEST_F( LayerNormTests, Cpu_Float_TestSimpleNormalization ) {
        TestSimpleNormalization<float, float, Compute::DeviceType::Cpu, Compute::HostMemoryResource>();
    }

    TEST_F( LayerNormTests, Cpu_Float_TestPrint ) {
        TestPrint<float, float, Compute::DeviceType::Cpu>(
            cpu_float_data_, "LayerNorm: cpu_ln_float" );
    }

    // CUDA Tests with float precision
    TEST_F( LayerNormTests, Cuda_Float_TestName ) {
        TestGetName<float, float, Compute::DeviceType::Cuda, Compute::DeviceMemoryResource>(
            cuda_float_data_, "cuda_ln_float" );
    }

    TEST_F( LayerNormTests, Cuda_Float_ParameterCount ) {
        TestParameterCount<float, float, Compute::DeviceType::Cuda, Compute::DeviceMemoryResource>(
            cuda_float_data_ );
    }

    TEST_F( LayerNormTests, Cuda_Float_TestForward ) {
        TestForward<float, float, Compute::DeviceType::Cuda, Compute::DeviceMemoryResource>(
            cuda_float_data_ );
    }

    TEST_F( LayerNormTests, Cuda_Float_TestPrint ) {
        TestPrint<float, float, Compute::DeviceType::Cuda>(
            cuda_float_data_, "LayerNorm: cuda_ln_float" );
    }

    // Tests for no-bias configuration
    TEST_F( LayerNormTests, Cpu_Float_NoBias_TestName ) {
        TestGetName<float, float, Compute::DeviceType::Cpu, Compute::HostMemoryResource>(
            cpu_no_bias_float_data_, "cpu_ln_no_bias_float" );
    }

    TEST_F( LayerNormTests, Cpu_Float_NoBias_ParameterCount ) {
        TestParameterCount<float, float, Compute::DeviceType::Cpu, Compute::HostMemoryResource>(
            cpu_no_bias_float_data_ );
    }

    TEST_F( LayerNormTests, Cpu_Float_NoBias_TestForward ) {
        TestForward<float, float, Compute::DeviceType::Cpu, Compute::HostMemoryResource>(
            cpu_no_bias_float_data_ );
    }

    TEST_F( LayerNormTests, Cuda_Float_NoBias_TestName ) {
        TestGetName<float, float, Compute::DeviceType::Cuda, Compute::DeviceMemoryResource>(
            cuda_no_bias_float_data_, "cuda_ln_no_bias_float" );
    }

    TEST_F( LayerNormTests, Cuda_Float_NoBias_ParameterCount ) {
        TestParameterCount<float, float, Compute::DeviceType::Cuda, Compute::DeviceMemoryResource>(
            cuda_no_bias_float_data_ );
    }

    TEST_F( LayerNormTests, Cuda_Float_NoBias_TestForward ) {
        TestForward<float, float, Compute::DeviceType::Cuda, Compute::DeviceMemoryResource>(
            cuda_no_bias_float_data_ );
    }

    // Edge case tests
    TEST_F( LayerNormTests, Cpu_Float_EdgeCases ) {
        TestEdgeCases<float, float, Compute::DeviceType::Cpu, Compute::HostMemoryResource>();
    }

    TEST_F( LayerNormTests, Cuda_Float_EdgeCases ) {
        TestEdgeCases<float, float, Compute::DeviceType::Cuda, Compute::DeviceMemoryResource>();
    }

    // Training mode tests
    TEST_F( LayerNormTests, Cpu_Float_TrainingModeBehavior ) {
        TestTrainingModeBehavior<float, float, Compute::DeviceType::Cpu, Compute::HostMemoryResource>(
            training_cpu_float_data_, cpu_float_data_ );
    }

    TEST_F( LayerNormTests, Cuda_Float_TrainingModeBehavior ) {
        TestTrainingModeBehavior<float, float, Compute::DeviceType::Cuda, Compute::DeviceMemoryResource>(
            training_cuda_float_data_, cuda_float_data_ );
    }

    // Numerical stability tests
    TEST_F( LayerNormTests, Cpu_Float_NumericalStability ) {
        TestNumericalStability<float, float, Compute::DeviceType::Cpu, Compute::HostMemoryResource>();
    }

    TEST_F( LayerNormTests, Cuda_Float_NumericalStability ) {
        TestNumericalStability<float, float, Compute::DeviceType::Cuda, Compute::DeviceMemoryResource>();
    }

    // Deterministic behavior test (CUDA only)
    TEST_F( LayerNormTests, Cuda_Float_Deterministic ) {
        TestDeterministicBehavior<float, float>();
    }

    // Mock save/load tests
    TEST_F( LayerNormTests, Cpu_Float_SaveLoad ) {
        TestSaveLoadMockup<float, float, Compute::DeviceType::Cpu, Compute::HostMemoryResource>(
            cpu_float_data_ );
    }

    TEST_F( LayerNormTests, Cuda_Float_SaveLoad ) {
        TestSaveLoadMockup<float, float, Compute::DeviceType::Cuda, Compute::DeviceMemoryResource>(
            cuda_float_data_ );
    }

    // CPU-CUDA equivalence test
    TEST_F( LayerNormTests, CpuCuda_Forward_Output_Equivalence ) {
        TestCpuCudaEquivalence<float, float>( cpu_float_data_, cuda_float_data_ );
    }

    TEST_F( LayerNormTests, CpuCuda_NoBias_Forward_Output_Equivalence ) {
        TestCpuCudaEquivalence<float, float>( cpu_no_bias_float_data_, cuda_no_bias_float_data_ );
    }
}
