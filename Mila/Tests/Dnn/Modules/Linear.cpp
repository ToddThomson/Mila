#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <cuda_fp16.h>

import Mila;

namespace Modules::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    // Memory resource selector based on device type
    template<DeviceType TDevice, typename TPrecision>
    using MemoryResourceType = std::conditional_t<TDevice == Compute::DeviceType::Cuda,
        Compute::CudaMemoryResource,
        Compute::CpuMemoryResource>;

    // Test data structure for Linear tests
    template<DeviceType TDevice, typename TInput = float, typename TOutput = TInput>
    struct LinearTestData {
        std::vector<size_t> input_shape;
        std::vector<size_t> output_shape;
        std::shared_ptr<Linear<TDevice, TInput, TOutput>> linear_module;
        size_t input_features;
        size_t output_features;
        bool has_bias;
        bool is_training;
        ComputePrecision::Policy precision_policy;

        // Make the test data structure self-initializing
        static LinearTestData Create(
            const std::string& name,
            size_t batch_size,
            size_t sequence_length,
            size_t input_features,
            size_t output_features,
            bool has_bias = true,
            bool is_training = false,
            ComputePrecision::Policy precision = ComputePrecision::Policy::Auto )
        {
            LinearTestData data;
            data.input_shape = { batch_size, sequence_length, input_features };
            data.output_shape = { batch_size, sequence_length, output_features };
            data.input_features = input_features;
            data.output_features = output_features;
            data.has_bias = has_bias;
            data.is_training = is_training;
            data.precision_policy = precision;

            std::string device_str = TDevice == Compute::DeviceType::Cuda ? "CUDA:0" : "CPU";
            data.linear_module = std::make_shared<Linear<TDevice, TInput, TOutput>>(
                name, device_str, input_features, output_features, has_bias, is_training, precision );

            return data;
        }

        // Overload for creating with device context
        static LinearTestData CreateWithContext(
            const std::string& name,
            size_t batch_size,
            size_t sequence_length,
            size_t input_features,
            size_t output_features,
            std::shared_ptr<DeviceContext> context,
            bool has_bias = true,
            bool is_training = false,
            ComputePrecision::Policy precision = ComputePrecision::Policy::Auto )
        {
            LinearTestData data;
            data.input_shape = { batch_size, sequence_length, input_features };
            data.output_shape = { batch_size, sequence_length, output_features };
            data.input_features = input_features;
            data.output_features = output_features;
            data.has_bias = has_bias;
            data.is_training = is_training;
            data.precision_policy = precision;

            data.linear_module = std::make_shared<Linear<TDevice, TInput, TOutput>>(
                name, context, input_features, output_features, has_bias, is_training, precision );

            return data;
        }
    };

    class LinearTests : public ::testing::Test {
    protected:
        void SetUp() override {
            // Initialize test parameters only
            batch_size_ = 64;
            cpu_batch_size_ = 4;
            sequence_length_ = 128;
            input_features_ = 256;
            output_features_ = 512;
            has_bias_ = true;
            // Modules will be created on demand
        }

        // Factory methods to lazily create test data as needed
        LinearTestData<Compute::DeviceType::Cpu, float>& CpuFloatData() {
            if ( !cpu_float_data_.linear_module ) {
                cpu_float_data_ = LinearTestData<Compute::DeviceType::Cpu, float>::Create(
                    "cpu_linear_float", cpu_batch_size_, sequence_length_,
                    input_features_, output_features_, has_bias_ );
            }
            return cpu_float_data_;
        }

        LinearTestData<Compute::DeviceType::Cpu, float>& CpuNoBiasFloatData() {
            if ( !cpu_no_bias_float_data_.linear_module ) {
                cpu_no_bias_float_data_ = LinearTestData<Compute::DeviceType::Cpu, float>::Create(
                    "cpu_linear_no_bias_float", cpu_batch_size_, sequence_length_,
                    input_features_, output_features_, false );
            }
            return cpu_no_bias_float_data_;
        }

        LinearTestData<Compute::DeviceType::Cuda, float>& CudaFloatData() {
            if ( !cuda_float_data_.linear_module ) {
                cuda_float_data_ = LinearTestData<Compute::DeviceType::Cuda, float>::Create(
                    "cuda_linear_float", batch_size_, sequence_length_,
                    input_features_, output_features_, has_bias_ );
            }
            return cuda_float_data_;
        }

        LinearTestData<Compute::DeviceType::Cuda, float>& CudaNoBiasFloatData() {
            if ( !cuda_no_bias_float_data_.linear_module ) {
                cuda_no_bias_float_data_ = LinearTestData<Compute::DeviceType::Cuda, float>::Create(
                    "cuda_linear_no_bias_float", batch_size_, sequence_length_,
                    input_features_, output_features_, false );
            }
            return cuda_no_bias_float_data_;
        }

        LinearTestData<Compute::DeviceType::Cpu, float>& TrainingCpuFloatData() {
            if ( !training_cpu_float_data_.linear_module ) {
                training_cpu_float_data_ = LinearTestData<Compute::DeviceType::Cpu, float>::Create(
                    "cpu_linear_float_training", cpu_batch_size_, sequence_length_,
                    input_features_, output_features_, has_bias_, true );
            }
            return training_cpu_float_data_;
        }

        LinearTestData<Compute::DeviceType::Cuda, float>& TrainingCudaFloatData() {
            if ( !training_cuda_float_data_.linear_module ) {
                training_cuda_float_data_ = LinearTestData<Compute::DeviceType::Cuda, float>::Create(
                    "cuda_linear_float_training", batch_size_, sequence_length_,
                    input_features_, output_features_, has_bias_, true );
            }
            return training_cuda_float_data_;
        }

        LinearTestData<Compute::DeviceType::Cpu, float>& ContextCpuFloatData() {
            if ( !context_cpu_float_data_.linear_module ) {
                auto cpu_context = std::make_shared<DeviceContext>( "CPU" );
                context_cpu_float_data_ = LinearTestData<Compute::DeviceType::Cpu, float>::CreateWithContext(
                    "cpu_context_linear_float", cpu_batch_size_, sequence_length_,
                    input_features_, output_features_, cpu_context, has_bias_ );
            }
            return context_cpu_float_data_;
        }

        LinearTestData<Compute::DeviceType::Cuda, half>& CudaHalfData() {
            if ( !cuda_half_data_.linear_module ) {
                cuda_half_data_ = LinearTestData<Compute::DeviceType::Cuda, half>::Create(
                    "cuda_linear_half", batch_size_, sequence_length_,
                    input_features_, output_features_, has_bias_ );
            }
            return cuda_half_data_;
        }

        LinearTestData<Compute::DeviceType::Cuda, half>& CudaNoBiasHalfData() {
            if ( !cuda_no_bias_half_data_.linear_module ) {
                cuda_no_bias_half_data_ = LinearTestData<Compute::DeviceType::Cuda, half>::Create(
                    "cuda_linear_no_bias_half", batch_size_, sequence_length_,
                    input_features_, output_features_, false );
            }
            return cuda_no_bias_half_data_;
        }

        LinearTestData<Compute::DeviceType::Cuda, half>& TrainingCudaHalfData() {
            if ( !training_cuda_half_data_.linear_module ) {
                training_cuda_half_data_ = LinearTestData<Compute::DeviceType::Cuda, half>::Create(
                    "cuda_linear_half_training", batch_size_, sequence_length_,
                    input_features_, output_features_, has_bias_, true );
            }
            return training_cuda_half_data_;
        }

        // Test for mixed precision (input float, output half)
        LinearTestData<Compute::DeviceType::Cuda, float, half>& MixedPrecisionData() {
            if ( !mixed_precision_data_.linear_module ) {
                mixed_precision_data_ = LinearTestData<Compute::DeviceType::Cuda, float, half>::Create(
                    "cuda_linear_mixed", batch_size_, sequence_length_,
                    input_features_, output_features_, has_bias_ );
            }
            return mixed_precision_data_;
        }

        // Tests with specific precision policies
        LinearTestData<Compute::DeviceType::Cuda, float>& PerfPrecisionCudaFloatData() {
            if ( !perf_precision_cuda_float_data_.linear_module ) {
                perf_precision_cuda_float_data_ = LinearTestData<Compute::DeviceType::Cuda, float>::Create(
                    "cuda_linear_perf_precision", batch_size_, sequence_length_,
                    input_features_, output_features_, has_bias_, false,
                    ComputePrecision::Policy::Performance );
            }
            return perf_precision_cuda_float_data_;
        }

        LinearTestData<Compute::DeviceType::Cuda, float>& AccuracyPrecisionCudaFloatData() {
            if ( !accuracy_precision_cuda_float_data_.linear_module ) {
                accuracy_precision_cuda_float_data_ = LinearTestData<Compute::DeviceType::Cuda, float>::Create(
                    "cuda_linear_accuracy_precision", batch_size_, sequence_length_,
                    input_features_, output_features_, has_bias_, false,
                    ComputePrecision::Policy::Accuracy );
            }
            return accuracy_precision_cuda_float_data_;
        }

        LinearTestData<Compute::DeviceType::Cuda, float>& DisabledPrecisionCudaFloatData() {
            if ( !disabled_precision_cuda_float_data_.linear_module ) {
                disabled_precision_cuda_float_data_ = LinearTestData<Compute::DeviceType::Cuda, float>::Create(
                    "cuda_linear_disabled_precision", batch_size_, sequence_length_,
                    input_features_, output_features_, has_bias_, false,
                    ComputePrecision::Policy::Disabled );
            }
            return disabled_precision_cuda_float_data_;
        }

        // Test for invalid parameters
        void CreateInvalidLinear() {
            auto invalid_linear = std::make_shared<Linear<DeviceType::Cpu, float>>(
                "invalid_linear", "CPU", 0, 512, true, false );
        }

        // Test parameters
        size_t batch_size_{ 0 };
        size_t cpu_batch_size_{ 0 };
        size_t sequence_length_{ 0 };
        size_t input_features_{ 0 };
        size_t output_features_{ 0 };
        bool has_bias_{ true };

        // Test data objects - initialized on demand
        LinearTestData<Compute::DeviceType::Cpu, float> cpu_float_data_;
        LinearTestData<Compute::DeviceType::Cpu, float> context_cpu_float_data_;
        LinearTestData<Compute::DeviceType::Cpu, float> cpu_no_bias_float_data_;
        LinearTestData<Compute::DeviceType::Cpu, float> training_cpu_float_data_;

        LinearTestData<Compute::DeviceType::Cuda, float> cuda_float_data_;
        LinearTestData<Compute::DeviceType::Cuda, float> cuda_no_bias_float_data_;
        LinearTestData<Compute::DeviceType::Cuda, float> training_cuda_float_data_;

        LinearTestData<Compute::DeviceType::Cuda, half> cuda_half_data_;
        LinearTestData<Compute::DeviceType::Cuda, half> cuda_no_bias_half_data_;
        LinearTestData<Compute::DeviceType::Cuda, half> training_cuda_half_data_;

        // Mixed precision test data (float input to half output)
        LinearTestData<Compute::DeviceType::Cuda, float, half> mixed_precision_data_;

        // Precision policy test data
        LinearTestData<Compute::DeviceType::Cuda, float> perf_precision_cuda_float_data_;
        LinearTestData<Compute::DeviceType::Cuda, float> accuracy_precision_cuda_float_data_;
        LinearTestData<Compute::DeviceType::Cuda, float> disabled_precision_cuda_float_data_;
    };

    // Common test function templates
    template<DeviceType TDevice, typename TInput, typename TOutput = TInput>
    void TestGetName( const LinearTestData<TDevice, TInput, TOutput>& data, const std::string& expected_name ) {
        EXPECT_EQ( data.linear_module->getName(), expected_name );
    }

    template<DeviceType TDevice, typename TInput, typename TOutput = TInput>
    void TestParameterCount( const LinearTestData<TDevice, TInput, TOutput>& data ) {
        size_t expected_count = (data.output_features * data.input_features); // weights
        if ( data.has_bias ) {
            expected_count += data.output_features; // bias
        }
        EXPECT_EQ( data.linear_module->parameterCount(), expected_count );
    }

    template<DeviceType TDevice, typename TInput, typename TOutput = TInput>
    void TestForward( const LinearTestData<TDevice, TInput, TOutput>& data ) {
        using MR = MemoryResourceType<TDevice, TInput>;

        Tensor<TInput, MR> input( data.input_shape );
        Tensor<TOutput, MR> output( data.output_shape );

        // Fill with random values
        random<TInput, MR>( input, -1.0f, 1.0f );

        data.linear_module->forward( input, output );
        EXPECT_EQ( output.size(), data.output_shape[ 0 ] * data.output_shape[ 1 ] * data.output_shape[ 2 ] );
    }

    template<DeviceType TDevice, typename TInput, typename TOutput = TInput>
    void TestPrint( const LinearTestData<TDevice, TInput, TOutput>& data, const std::string& expected_substring ) {
        std::string output = data.linear_module->toString();
        EXPECT_NE( output.find( expected_substring ), std::string::npos );
        // Also verify the feature information is included
        std::string feature_info = "Input features: " + std::to_string( data.input_features );
        EXPECT_NE( output.find( feature_info ), std::string::npos );
    }

    template<DeviceType TDevice, typename TInput, typename TOutput = TInput>
    void TestPrecisionPolicy( const LinearTestData<TDevice, TInput, TOutput>& data, ComputePrecision::Policy expected_policy ) {
        // Check that the precision policy is correctly included in the string output
        std::string output = data.linear_module->toString();
        std::string policy_str;

        switch ( expected_policy ) {
            case ComputePrecision::Policy::Disabled:
                policy_str = "Disabled";
                break;
            case ComputePrecision::Policy::Performance:
                policy_str = "Performance";
                break;
            case ComputePrecision::Policy::Auto:
                policy_str = "Auto";
                break;
            case ComputePrecision::Policy::Accuracy:
                policy_str = "Accuracy";
                break;
        }

        EXPECT_NE( output.find( "Precision Policy: " + policy_str ), std::string::npos );
    }

    template<DeviceType TDevice, typename TInput, typename TOutput = TInput>
    void TestGetWeight( const LinearTestData<TDevice, TInput, TOutput>& data ) {
        auto weight = data.linear_module->getWeight();
        EXPECT_NE( weight, nullptr );
        EXPECT_EQ( weight->shape()[ 0 ], data.output_features );
        EXPECT_EQ( weight->shape()[ 1 ], data.input_features );
    }

    template<DeviceType TDevice, typename TInput, typename TOutput = TInput>
    void TestGetBias( const LinearTestData<TDevice, TInput, TOutput>& data ) {
        auto bias_opt = data.linear_module->getBias();

        if ( data.has_bias ) {
            EXPECT_TRUE( bias_opt.has_value() );
            auto bias = bias_opt.value();
            EXPECT_NE( bias, nullptr );
            EXPECT_EQ( bias->shape()[ 0 ], data.output_features );
        }
        else {
            EXPECT_FALSE( bias_opt.has_value() );
        }
    }

    template<DeviceType TDevice, typename TInput, typename TOutput = TInput>
    void TestHasBias( const LinearTestData<TDevice, TInput, TOutput>& data ) {
        EXPECT_EQ( data.linear_module->hasBias(), data.has_bias );
    }

    template<DeviceType TDevice, typename TInput, typename TOutput = TInput>
    void TestTrainingMode( const LinearTestData<TDevice, TInput, TOutput>& data, bool expected_mode ) {
        EXPECT_EQ( data.linear_module->isTraining(), expected_mode );
    }

    template<DeviceType TDevice, typename TInput, typename TOutput = TInput>
    void TestDeviceType( const LinearTestData<TDevice, TInput, TOutput>& data ) {
        auto device_context = data.linear_module->getDeviceContext();
        EXPECT_NE( device_context, nullptr );
        auto device = device_context->getDevice();
        EXPECT_NE( device, nullptr );
        EXPECT_EQ( device->getDeviceType(), TDevice );
    }

    // Function to test equivalence of CPU and CUDA outputs
    template<typename TInput, typename TOutput = TInput>
    void TestCpuCudaEquivalence(
        const LinearTestData<Compute::DeviceType::Cpu, TInput, TOutput>& cpu_data,
        const LinearTestData<Compute::DeviceType::Cuda, TInput, TOutput>& cuda_data ) {

        // Create a small test shape to make comparison faster
        std::vector<size_t> test_input_shape = { 2, 4, cpu_data.input_features };
        std::vector<size_t> test_output_shape = { 2, 4, cpu_data.output_features };

        // Create random input data
        Tensor<TInput, Compute::HostMemoryResource> host_input( test_input_shape );

        // Fill with predictable values
        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input.data()[ i ] = static_cast<TInput>( -1.0 + 2.0 * (static_cast<float>( i ) / host_input.size()) );
        }

        // Initialize the weights and biases with the same values for both CPU and CUDA modules
        // Copy parameters from CPU module to CUDA module for fair comparison
        auto cpu_params = cpu_data.linear_module->getParameterTensors();
        auto cuda_params = cuda_data.linear_module->getParameterTensors();

        // Copy weights
        Tensor<TOutput, Compute::CudaMemoryResource> cuda_weights( cpu_params[ "weight" ]->shape() );
        cuda_weights.copyFrom( *cpu_params[ "weight" ] );
        cuda_params[ "weight" ]->copyFrom( cuda_weights );

        // Copy bias if it exists
        if ( cpu_data.has_bias && cuda_data.has_bias ) {
            Tensor<TOutput, Compute::CudaMemoryResource> cuda_bias( cpu_params[ "bias" ]->shape() );
            cuda_bias.copyFrom( *cpu_params[ "bias" ] );
            cuda_params[ "bias" ]->copyFrom( cuda_bias );
        }

        // Create CPU output
        Tensor<TOutput, Compute::HostMemoryResource> cpu_output( test_output_shape );
        cpu_data.linear_module->forward( host_input, cpu_output );

        // Create device input by copying host data
        Tensor<TInput, Compute::CudaMemoryResource> device_input( test_input_shape );
        device_input.copyFrom( host_input );

        // Create device output
        Tensor<TOutput, Compute::CudaMemoryResource> cuda_output( test_output_shape );
        cuda_data.linear_module->forward( device_input, cuda_output );

        // Copy CUDA output back to host for comparison
        Tensor<TOutput, Compute::HostMemoryResource> cuda_output_host( test_output_shape );
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
            }
        }

        EXPECT_TRUE( all_equal ) << "CPU and CUDA implementations produced different results";
    }

    // Test with different dimensions (edge cases)
    template<DeviceType TDevice, typename TInput, typename TOutput = TInput>
    void TestEdgeCases() {
        using MR = MemoryResourceType<TDevice, TInput>;

        try {
            // Test with minimal sizes
            std::vector<size_t> minimal_input_shape = { 1, 1, 8 };
            std::vector<size_t> minimal_output_shape = { 1, 1, 16 };

            auto minimal_module = std::make_shared<Linear<TDevice, TInput, TOutput>>(
                "minimal_linear", TDevice == Compute::DeviceType::Cuda ? "CUDA:0" : "CPU", 8, 16 );

            Tensor<TInput, MR> minimal_input( minimal_input_shape );
            Tensor<TOutput, MR> minimal_output( minimal_output_shape );

            EXPECT_NO_THROW( minimal_module->forward( minimal_input, minimal_output ) );
            EXPECT_EQ( minimal_output.size(), 16 );

            // Test with larger dimensions
            std::vector<size_t> large_input_shape = { 2, 2, 1024 };
            std::vector<size_t> large_output_shape = { 2, 2, 512 };

            auto large_module = std::make_shared<Linear<TDevice, TInput, TOutput>>(
                "large_linear", TDevice == Compute::DeviceType::Cuda ? "CUDA:0" : "CPU", 1024, 512 );

            Tensor<TInput, MR> large_input( large_input_shape );
            Tensor<TOutput, MR> large_output( large_output_shape );

            EXPECT_NO_THROW( large_module->forward( large_input, large_output ) );
            EXPECT_EQ( large_output.size(), 2048 );
        }
        catch ( const std::exception& e ) {
            std::cerr << "Exception during edge case test: " << e.what() << std::endl;
            throw;
        }
    }

    // CPU Tests with float precision
    TEST_F( LinearTests, Cpu_Float_TestName ) {
        TestGetName<Compute::DeviceType::Cpu, float>( CpuFloatData(), "cpu_linear_float" );
    }

    TEST_F( LinearTests, Cpu_Float_ParameterCount ) {
        TestParameterCount<Compute::DeviceType::Cpu, float>( CpuFloatData() );
    }

    TEST_F( LinearTests, Cpu_Float_TestForward ) {
        TestForward<Compute::DeviceType::Cpu, float>( CpuFloatData() );
    }

    TEST_F( LinearTests, Cpu_Float_TestPrint ) {
        TestPrint<Compute::DeviceType::Cpu, float>( CpuFloatData(), "Linear: cpu_linear_float" );
    }

    TEST_F( LinearTests, Cpu_Float_GetWeight ) {
        TestGetWeight<Compute::DeviceType::Cpu, float>( CpuFloatData() );
    }

    TEST_F( LinearTests, Cpu_Float_GetBias ) {
        TestGetBias<Compute::DeviceType::Cpu, float>( CpuFloatData() );
    }

    TEST_F( LinearTests, Cpu_Float_HasBias ) {
        TestHasBias<Compute::DeviceType::Cpu, float>( CpuFloatData() );
    }

    TEST_F( LinearTests, Cpu_Float_TrainingMode ) {
        TestTrainingMode<Compute::DeviceType::Cpu, float>( CpuFloatData(), false );
    }

    TEST_F( LinearTests, Cpu_Float_DeviceType ) {
        TestDeviceType<Compute::DeviceType::Cpu, float>( CpuFloatData() );
    }

    // CPU No Bias Tests
    TEST_F( LinearTests, Cpu_NoBias_Float_ParameterCount ) {
        TestParameterCount<Compute::DeviceType::Cpu, float>( CpuNoBiasFloatData() );
    }

    TEST_F( LinearTests, Cpu_NoBias_Float_GetBias ) {
        TestGetBias<Compute::DeviceType::Cpu, float>( CpuNoBiasFloatData() );
    }

    TEST_F( LinearTests, Cpu_NoBias_Float_HasBias ) {
        TestHasBias<Compute::DeviceType::Cpu, float>( CpuNoBiasFloatData() );
    }

    TEST_F( LinearTests, Cpu_NoBias_Float_Forward ) {
        TestForward<Compute::DeviceType::Cpu, float>( CpuNoBiasFloatData() );
    }

    // CPU Training Mode Tests
    TEST_F( LinearTests, Cpu_Training_Float_TrainingMode ) {
        TestTrainingMode<Compute::DeviceType::Cpu, float>( TrainingCpuFloatData(), true );
    }

    // CUDA Tests with float precision
    TEST_F( LinearTests, Cuda_Float_TestName ) {
        TestGetName<Compute::DeviceType::Cuda, float>( CudaFloatData(), "cuda_linear_float" );
    }

    TEST_F( LinearTests, Cuda_Float_ParameterCount ) {
        TestParameterCount<Compute::DeviceType::Cuda, float>( CudaFloatData() );
    }

    TEST_F( LinearTests, Cuda_Float_TestForward ) {
        TestForward<Compute::DeviceType::Cuda, float>( CudaFloatData() );
    }

    TEST_F( LinearTests, Cuda_Float_TestPrint ) {
        TestPrint<Compute::DeviceType::Cuda, float>( CudaFloatData(), "Linear: cuda_linear_float" );
    }

    TEST_F( LinearTests, Cuda_Float_GetWeight ) {
        TestGetWeight<Compute::DeviceType::Cuda, float>( CudaFloatData() );
    }

    TEST_F( LinearTests, Cuda_Float_GetBias ) {
        TestGetBias<Compute::DeviceType::Cuda, float>( CudaFloatData() );
    }

    TEST_F( LinearTests, Cuda_Float_HasBias ) {
        TestHasBias<Compute::DeviceType::Cuda, float>( CudaFloatData() );
    }

    TEST_F( LinearTests, Cuda_Float_TrainingMode ) {
        TestTrainingMode<Compute::DeviceType::Cuda, float>( CudaFloatData(), false );
    }

    TEST_F( LinearTests, Cuda_Float_DeviceType ) {
        TestDeviceType<Compute::DeviceType::Cuda, float>( CudaFloatData() );
    }

    // CUDA No Bias Tests
    TEST_F( LinearTests, Cuda_NoBias_Float_ParameterCount ) {
        TestParameterCount<Compute::DeviceType::Cuda, float>( CudaNoBiasFloatData() );
    }

    TEST_F( LinearTests, Cuda_NoBias_Float_GetBias ) {
        TestGetBias<Compute::DeviceType::Cuda, float>( CudaNoBiasFloatData() );
    }

    TEST_F( LinearTests, Cuda_NoBias_Float_HasBias ) {
        TestHasBias<Compute::DeviceType::Cuda, float>( CudaNoBiasFloatData() );
    }

    TEST_F( LinearTests, Cuda_NoBias_Float_Forward ) {
        TestForward<Compute::DeviceType::Cuda, float>( CudaNoBiasFloatData() );
    }

    // CUDA Training Mode Tests
    TEST_F( LinearTests, Cuda_Training_Float_TrainingMode ) {
        TestTrainingMode<Compute::DeviceType::Cuda, float>( TrainingCudaFloatData(), true );
    }

    // CUDA Tests with half precision
    TEST_F( LinearTests, Cuda_Half_TestName ) {
        TestGetName<Compute::DeviceType::Cuda, half>( CudaHalfData(), "cuda_linear_half" );
    }

    TEST_F( LinearTests, Cuda_Half_ParameterCount ) {
        TestParameterCount<Compute::DeviceType::Cuda, half>( CudaHalfData() );
    }

    TEST_F( LinearTests, Cuda_Half_TestForward ) {
        TestForward<Compute::DeviceType::Cuda, half>( CudaHalfData() );
    }

    TEST_F( LinearTests, Cuda_Half_TestPrint ) {
        TestPrint<Compute::DeviceType::Cuda, half>( CudaHalfData(), "Linear: cuda_linear_half" );
    }

    TEST_F( LinearTests, Cuda_Half_GetBias ) {
        TestGetBias<Compute::DeviceType::Cuda, half>( CudaHalfData() );
    }

    TEST_F( LinearTests, Cuda_Half_HasBias ) {
        TestHasBias<Compute::DeviceType::Cuda, half>( CudaHalfData() );
    }

    TEST_F( LinearTests, Cuda_Half_TrainingMode ) {
        TestTrainingMode<Compute::DeviceType::Cuda, half>( CudaHalfData(), false );
    }

    // CUDA Half No Bias Tests
    TEST_F( LinearTests, Cuda_NoBias_Half_GetBias ) {
        TestGetBias<Compute::DeviceType::Cuda, half>( CudaNoBiasHalfData() );
    }

    TEST_F( LinearTests, Cuda_NoBias_Half_HasBias ) {
        TestHasBias<Compute::DeviceType::Cuda, half>( CudaNoBiasHalfData() );
    }

    TEST_F( LinearTests, Cuda_NoBias_Half_Forward ) {
        TestForward<Compute::DeviceType::Cuda, half>( CudaNoBiasHalfData() );
    }

    // Mixed Precision Tests
    TEST_F( LinearTests, Cuda_MixedPrecision_TestForward ) {
		// WIP: Mixed precision is not fully implemented yet
        //TestForward<Compute::DeviceType::Cuda, float, half>( MixedPrecisionData() );
    }

    TEST_F( LinearTests, Cuda_MixedPrecision_TestName ) {
		// WIP: Mixed precision is not fully implemented yet
        //TestGetName<Compute::DeviceType::Cuda, float, half>( MixedPrecisionData(), "cuda_linear_mixed" );
    }

    // Context Construction Tests
    TEST_F( LinearTests, Context_Cpu_Float_DeviceType ) {
        TestDeviceType<Compute::DeviceType::Cpu, float>( ContextCpuFloatData() );
    }

    TEST_F( LinearTests, Context_Cpu_Float_Forward ) {
        TestForward<Compute::DeviceType::Cpu, float>( ContextCpuFloatData() );
    }

    // Edge Case Tests
    TEST_F( LinearTests, Cpu_Float_EdgeCases ) {
        TestEdgeCases<Compute::DeviceType::Cpu, float>();
    }

    TEST_F( LinearTests, Cuda_Float_EdgeCases ) {
        TestEdgeCases<Compute::DeviceType::Cuda, float>();
    }

    // Invalid Parameter Tests
    TEST_F( LinearTests, InvalidParameters_ThrowsException ) {
        EXPECT_THROW( CreateInvalidLinear(), std::invalid_argument );
    }

    // Precision Policy Tests
    TEST_F( LinearTests, DefaultPrecisionPolicy_IsAuto ) {
        TestPrecisionPolicy( CudaFloatData(), ComputePrecision::Policy::Auto );
    }

    TEST_F( LinearTests, PerformancePrecisionPolicy ) {
        TestPrecisionPolicy( PerfPrecisionCudaFloatData(), ComputePrecision::Policy::Performance );
        TestForward( PerfPrecisionCudaFloatData() );
    }

    TEST_F( LinearTests, AccuracyPrecisionPolicy ) {
        TestPrecisionPolicy( AccuracyPrecisionCudaFloatData(), ComputePrecision::Policy::Accuracy );
        TestForward( AccuracyPrecisionCudaFloatData() );
    }

    TEST_F( LinearTests, DisabledPrecisionPolicy ) {
        TestPrecisionPolicy( DisabledPrecisionCudaFloatData(), ComputePrecision::Policy::Disabled );
        TestForward( DisabledPrecisionCudaFloatData() );
    }

    // CPU-CUDA Equivalence Test
    TEST_F( LinearTests, CpuCuda_Forward_Output_Equivalence ) {
        TestCpuCudaEquivalence<float>( CpuFloatData(), CudaFloatData() );
    }

    TEST_F( LinearTests, CpuCuda_NoBias_Forward_Output_Equivalence ) {
        TestCpuCudaEquivalence<float>( CpuNoBiasFloatData(), CudaNoBiasFloatData() );
    }
}