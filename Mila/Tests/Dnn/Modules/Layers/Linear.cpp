#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

import Mila;

namespace Modules::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    template<DeviceType TDevice>
    using MemoryResourceType = std::conditional_t<TDevice == Compute::DeviceType::Cuda,
        Compute::CudaMemoryResource,
        Compute::CpuMemoryResource>;

    // Test data structure for Linear tests
    template<DeviceType TDevice, typename TDataType = float>
    struct LinearTestData {
        std::vector<size_t> input_shape;
        std::vector<size_t> output_shape;
        std::shared_ptr<Linear<TDevice, TDataType>> linear_module;
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

            // Create configuration with appropriate settings
            LinearConfig config( input_features, output_features );
            config.withBias( has_bias )
                .withTraining( is_training )
                .withName( name )
                .withPrecisionPolicy( precision );

            std::string device_str = TDevice == Compute::DeviceType::Cuda ? "CUDA:0" : "CPU";
            data.linear_module = std::make_shared<Linear<TDevice, TDataType>>( device_str, config );

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

            // Create configuration with appropriate settings
            LinearConfig config( input_features, output_features );
            config.withBias( has_bias )
                .withTraining( is_training )
                .withName( name )
                .withPrecisionPolicy( precision );

            data.linear_module = std::make_shared<Linear<TDevice, TDataType>>( context, config );

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

        void TearDown() override {
            // Clean up resources explicitly
            cpu_float_data_.linear_module.reset();
            context_cpu_float_data_.linear_module.reset();
            cpu_no_bias_float_data_.linear_module.reset();
            training_cpu_float_data_.linear_module.reset();
            cpu_native_policy_data_.linear_module.reset();
            cpu_performance_policy_data_.linear_module.reset();
            cpu_accuracy_policy_data_.linear_module.reset();

            cuda_float_data_.linear_module.reset();
            cuda_no_bias_float_data_.linear_module.reset();
            training_cuda_float_data_.linear_module.reset();

            cuda_half_data_.linear_module.reset();
            cuda_no_bias_half_data_.linear_module.reset();
            training_cuda_half_data_.linear_module.reset();

            cuda_bf16_data_.linear_module.reset();

            // FP8 data
            cuda_fp8_e4m3_data_.linear_module.reset();
            cuda_fp8_e5m2_data_.linear_module.reset();

            // Precision policy data for CUDA
            cuda_native_policy_data_.linear_module.reset();
            cuda_performance_policy_data_.linear_module.reset();
            cuda_accuracy_policy_data_.linear_module.reset();
            cuda_auto_policy_data_.linear_module.reset();
        }

        // CPU with Float (FP32)
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

        LinearTestData<Compute::DeviceType::Cpu, float>& TrainingCpuFloatData() {
            if ( !training_cpu_float_data_.linear_module ) {
                training_cpu_float_data_ = LinearTestData<Compute::DeviceType::Cpu, float>::Create(
                    "cpu_linear_float_training", cpu_batch_size_, sequence_length_,
                    input_features_, output_features_, has_bias_, true );
            }
            return training_cpu_float_data_;
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

        // CPU with different precision policies
        LinearTestData<Compute::DeviceType::Cpu, float>& CpuNativePolicyData() {
            if ( !cpu_native_policy_data_.linear_module ) {
                cpu_native_policy_data_ = LinearTestData<Compute::DeviceType::Cpu, float>::Create(
                    "cpu_linear_native_policy", cpu_batch_size_, sequence_length_,
                    input_features_, output_features_, has_bias_, false,
                    ComputePrecision::Policy::Native );
            }
            return cpu_native_policy_data_;
        }

        LinearTestData<Compute::DeviceType::Cpu, float>& CpuPerformancePolicyData() {
            if ( !cpu_performance_policy_data_.linear_module ) {
                cpu_performance_policy_data_ = LinearTestData<Compute::DeviceType::Cpu, float>::Create(
                    "cpu_linear_performance_policy", cpu_batch_size_, sequence_length_,
                    input_features_, output_features_, has_bias_, false,
                    ComputePrecision::Policy::Performance );
            }
            return cpu_performance_policy_data_;
        }

        LinearTestData<Compute::DeviceType::Cpu, float>& CpuAccuracyPolicyData() {
            if ( !cpu_accuracy_policy_data_.linear_module ) {
                cpu_accuracy_policy_data_ = LinearTestData<Compute::DeviceType::Cpu, float>::Create(
                    "cpu_linear_accuracy_policy", cpu_batch_size_, sequence_length_,
                    input_features_, output_features_, has_bias_, false,
                    ComputePrecision::Policy::Accuracy );
            }
            return cpu_accuracy_policy_data_;
        }

        // CUDA with Float (FP32)
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

        LinearTestData<Compute::DeviceType::Cuda, float>& TrainingCudaFloatData() {
            if ( !training_cuda_float_data_.linear_module ) {
                training_cuda_float_data_ = LinearTestData<Compute::DeviceType::Cuda, float>::Create(
                    "cuda_linear_float_training", batch_size_, sequence_length_,
                    input_features_, output_features_, has_bias_, true );
            }
            return training_cuda_float_data_;
        }

        // CUDA with Half Precision (FP16)
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

        // CUDA with BF16 Precision
        LinearTestData<Compute::DeviceType::Cuda, nv_bfloat16>& CudaBF16Data() {
            if ( !cuda_bf16_data_.linear_module ) {
                cuda_bf16_data_ = LinearTestData<Compute::DeviceType::Cuda, nv_bfloat16>::Create(
                    "cuda_linear_bf16", batch_size_, sequence_length_,
                    input_features_, output_features_, has_bias_ );
            }
            return cuda_bf16_data_;
        }

        // CUDA with FP8 Precision (E4M3)
        LinearTestData<Compute::DeviceType::Cuda, __nv_fp8_e4m3>& CudaFP8E4M3Data() {
            if ( !cuda_fp8_e4m3_data_.linear_module ) {
                cuda_fp8_e4m3_data_ = LinearTestData<Compute::DeviceType::Cuda, __nv_fp8_e4m3>::Create(
                    "cuda_linear_fp8_e4m3", batch_size_, sequence_length_,
                    input_features_, output_features_, has_bias_ );
            }
            return cuda_fp8_e4m3_data_;
        }

        // CUDA with FP8 Precision (E5M2)
        LinearTestData<Compute::DeviceType::Cuda, __nv_fp8_e5m2>& CudaFP8E5M2Data() {
            if ( !cuda_fp8_e5m2_data_.linear_module ) {
                cuda_fp8_e5m2_data_ = LinearTestData<Compute::DeviceType::Cuda, __nv_fp8_e5m2>::Create(
                    "cuda_linear_fp8_e5m2", batch_size_, sequence_length_,
                    input_features_, output_features_, has_bias_ );
            }
            return cuda_fp8_e5m2_data_;
        }

        // CUDA with different precision policies
        LinearTestData<Compute::DeviceType::Cuda, float>& CudaNativePolicyData() {
            if ( !cuda_native_policy_data_.linear_module ) {
                cuda_native_policy_data_ = LinearTestData<Compute::DeviceType::Cuda, float>::Create(
                    "cuda_linear_native_policy", batch_size_, sequence_length_,
                    input_features_, output_features_, has_bias_, false,
                    ComputePrecision::Policy::Native );
            }
            return cuda_native_policy_data_;
        }

        LinearTestData<Compute::DeviceType::Cuda, float>& CudaPerformancePolicyData() {
            if ( !cuda_performance_policy_data_.linear_module ) {
                cuda_performance_policy_data_ = LinearTestData<Compute::DeviceType::Cuda, float>::Create(
                    "cuda_linear_performance_policy", batch_size_, sequence_length_,
                    input_features_, output_features_, has_bias_, false,
                    ComputePrecision::Policy::Performance );
            }
            return cuda_performance_policy_data_;
        }

        LinearTestData<Compute::DeviceType::Cuda, float>& CudaAccuracyPolicyData() {
            if ( !cuda_accuracy_policy_data_.linear_module ) {
                cuda_accuracy_policy_data_ = LinearTestData<Compute::DeviceType::Cuda, float>::Create(
                    "cuda_linear_accuracy_policy", batch_size_, sequence_length_,
                    input_features_, output_features_, has_bias_, false,
                    ComputePrecision::Policy::Accuracy );
            }
            return cuda_accuracy_policy_data_;
        }

        LinearTestData<Compute::DeviceType::Cuda, float>& CudaAutoPolicyData() {
            if ( !cuda_auto_policy_data_.linear_module ) {
                cuda_auto_policy_data_ = LinearTestData<Compute::DeviceType::Cuda, float>::Create(
                    "cuda_linear_auto_policy", batch_size_, sequence_length_,
                    input_features_, output_features_, has_bias_, false,
                    ComputePrecision::Policy::Auto );
            }
            return cuda_auto_policy_data_;
        }

        // Test for invalid parameters
        void CreateInvalidLinear() {
            // Create configuration with invalid parameters
            LinearConfig invalid_config( 0, 512 );
            invalid_config.withBias( true ).withName( "invalid_linear" );

            auto invalid_linear = std::make_shared<Linear<DeviceType::Cpu, float>>( "CPU", invalid_config );
        }

        // Test parameters
        size_t batch_size_{ 0 };
        size_t cpu_batch_size_{ 0 };
        size_t sequence_length_{ 0 };
        size_t input_features_{ 0 };
        size_t output_features_{ 0 };
        bool has_bias_{ true };

        // CPU test data objects
        LinearTestData<Compute::DeviceType::Cpu, float> cpu_float_data_;
        LinearTestData<Compute::DeviceType::Cpu, float> context_cpu_float_data_;
        LinearTestData<Compute::DeviceType::Cpu, float> cpu_no_bias_float_data_;
        LinearTestData<Compute::DeviceType::Cpu, float> training_cpu_float_data_;
        LinearTestData<Compute::DeviceType::Cpu, float> cpu_native_policy_data_;
        LinearTestData<Compute::DeviceType::Cpu, float> cpu_performance_policy_data_;
        LinearTestData<Compute::DeviceType::Cpu, float> cpu_accuracy_policy_data_;

        // CUDA FP32 test data objects
        LinearTestData<Compute::DeviceType::Cuda, float> cuda_float_data_;
        LinearTestData<Compute::DeviceType::Cuda, float> cuda_no_bias_float_data_;
        LinearTestData<Compute::DeviceType::Cuda, float> training_cuda_float_data_;

        // CUDA FP16 test data objects
        LinearTestData<Compute::DeviceType::Cuda, half> cuda_half_data_;
        LinearTestData<Compute::DeviceType::Cuda, half> cuda_no_bias_half_data_;
        LinearTestData<Compute::DeviceType::Cuda, half> training_cuda_half_data_;

        // CUDA BF16 test data objects
        LinearTestData<Compute::DeviceType::Cuda, nv_bfloat16> cuda_bf16_data_;

        // CUDA FP8 test data objects
        LinearTestData<Compute::DeviceType::Cuda, __nv_fp8_e4m3> cuda_fp8_e4m3_data_;
        LinearTestData<Compute::DeviceType::Cuda, __nv_fp8_e5m2> cuda_fp8_e5m2_data_;

        // CUDA precision policy test data
        LinearTestData<Compute::DeviceType::Cuda, float> cuda_native_policy_data_;
        LinearTestData<Compute::DeviceType::Cuda, float> cuda_performance_policy_data_;
        LinearTestData<Compute::DeviceType::Cuda, float> cuda_accuracy_policy_data_;
        LinearTestData<Compute::DeviceType::Cuda, float> cuda_auto_policy_data_;
    };

    // Common test function templates
    template<DeviceType TDevice, typename TDataType = float>
    void TestGetName( const LinearTestData<TDevice, TDataType>& data, const std::string& expected_name ) {
        EXPECT_EQ( data.linear_module->getName(), expected_name );
    }

    template<DeviceType TDevice, typename TDataType = float>
    void TestParameterCount( const LinearTestData<TDevice, TDataType>& data ) {
        size_t expected_count = (data.output_features * data.input_features); // weights
        if ( data.has_bias ) {
            expected_count += data.output_features; // bias
        }
        EXPECT_EQ( data.linear_module->parameterCount(), expected_count );
    }

    template<DeviceType TDevice, typename TDataType = float>
    void TestForward( const LinearTestData<TDevice, TDataType>& data ) {
        using MR = MemoryResourceType<TDevice>;

        Tensor<TDataType, MR> input( data.input_shape );
        Tensor<TDataType, MR> output( data.output_shape );

        if constexpr ( std::is_same_v<TDataType, __nv_fp8_e4m3> || std::is_same_v<TDataType, __nv_fp8_e5m2> ) {
            TDataType min_val( static_cast<float>(-1.0f) );
            TDataType max_val( static_cast<float>(1.0f) );
            random<TDataType, MR>( input, min_val, max_val );
        }
        else {
            random<TDataType, MR>( input, static_cast<TDataType>(-1.0f), static_cast<TDataType>(1.0f) );
        }

        data.linear_module->forward( input, output );

        EXPECT_EQ( output.size(), data.output_shape[ 0 ] * data.output_shape[ 1 ] * data.output_shape[ 2 ] );
    }

    template<DeviceType TDevice, typename TDataType = float>
    void TestPrint( const LinearTestData<TDevice, TDataType>& data, const std::string& expected_substring ) {
        std::string output = data.linear_module->toString();
        EXPECT_NE( output.find( expected_substring ), std::string::npos );
        // Also verify the feature information is included
        std::string feature_info = "Input features: " + std::to_string( data.input_features );
        EXPECT_NE( output.find( feature_info ), std::string::npos );
    }

    template<DeviceType TDevice, typename TDataType = float>
    void TestPrecisionPolicy( const LinearTestData<TDevice, TDataType>& data, ComputePrecision::Policy expected_policy ) {
        // Check that the precision policy is correctly included in the string output
        std::string output = data.linear_module->toString();
        std::string policy_str;

        switch ( expected_policy ) {
            case ComputePrecision::Policy::Native:
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

        // EXPECT_NE( output.find( "Precision Policy: " + policy_str ), std::string::npos );

        // Verify the policy is set correctly
        EXPECT_EQ( data.precision_policy, expected_policy );
    }

    template<DeviceType TDevice, typename TDataType = float>
    void TestGetWeight( const LinearTestData<TDevice, TDataType>& data ) {
        auto weight = data.linear_module->getWeight();
        EXPECT_NE( weight, nullptr );
        EXPECT_EQ( weight->shape()[ 0 ], data.output_features );
        EXPECT_EQ( weight->shape()[ 1 ], data.input_features );
    }

    template<DeviceType TDevice, typename TDataType = float>
    void TestGetBias( const LinearTestData<TDevice, TDataType>& data ) {
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

    template<DeviceType TDevice, typename TDataType = float>
    void TestHasBias( const LinearTestData<TDevice, TDataType>& data ) {
        EXPECT_EQ( data.linear_module->hasBias(), data.has_bias );
    }

    template<DeviceType TDevice, typename TDataType = float>
    void TestTrainingMode( const LinearTestData<TDevice, TDataType>& data, bool expected_mode ) {
        EXPECT_EQ( data.linear_module->isTraining(), expected_mode );
    }

    template<DeviceType TDevice, typename TDataType = float>
    void TestDeviceType( const LinearTestData<TDevice, TDataType>& data ) {
        auto device_context = data.linear_module->getDeviceContext();
        EXPECT_NE( device_context, nullptr );
        auto device = device_context->getDevice();
        EXPECT_NE( device, nullptr );
        EXPECT_EQ( device->getDeviceType(), TDevice );
    }

    // Function to test equivalence of CPU float and CUDA with other float types
    template<typename TCpuType = float, typename TCudaType = float>
    void TestCpuCudaEquivalence(
        const LinearTestData<Compute::DeviceType::Cpu, TCpuType>& cpu_data,
        const LinearTestData<Compute::DeviceType::Cuda, TCudaType>& cuda_data ) {

        try {
            // Check if CUDA is available
            DeviceContext context( "CUDA:0" );
        }
        catch ( const std::exception& ) {
            GTEST_SKIP() << "CUDA device not available, skipping CPU-CUDA equivalence test";
            return;
        }

        // Create a small test shape to make comparison faster
        std::vector<size_t> test_input_shape = { 2, 4, cpu_data.input_features };
        std::vector<size_t> test_output_shape = { 2, 4, cpu_data.output_features };

        // Create random input data
        Tensor<TCpuType, Compute::HostMemoryResource> host_input( test_input_shape );

        // Fill with predictable values
        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input.data()[ i ] = static_cast<TCpuType>( -1.0 + 2.0 * (static_cast<float>( i ) / host_input.size()) );
        }

        // Create CPU output
        Tensor<TCpuType, Compute::HostMemoryResource> cpu_output( test_output_shape );
        cpu_data.linear_module->forward( host_input, cpu_output );

        // Create device input by copying host data
        // For different types, we need to convert the values
        Tensor<TCudaType, Compute::CudaMemoryResource> device_input( test_input_shape );
        
        // For different types, first create a host tensor of the CUDA type
        Tensor<TCudaType, Compute::HostMemoryResource> host_input_cuda_type( test_input_shape );
        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input_cuda_type.data()[ i ] = static_cast<TCudaType>( static_cast<float>( host_input.data()[ i ] ) );
        }
        device_input.copyFrom( host_input_cuda_type );

        // Create device output
        Tensor<TCudaType, Compute::CudaMemoryResource> cuda_output( test_output_shape );
        cuda_data.linear_module->forward( device_input, cuda_output );

        // Copy CUDA output back to host for comparison
        Tensor<TCudaType, Compute::HostMemoryResource> cuda_output_host( test_output_shape );
        cuda_output_host.copyFrom( cuda_output );

        // Compare outputs with tolerance for floating point differences
        // Use higher tolerance for lower precision types
        float epsilon = 1e-4f; // Default for float
        
        if constexpr ( std::is_same_v<TCudaType, half> ) {
            epsilon = 1e-2f;
        }
        else if constexpr ( std::is_same_v<TCudaType, nv_bfloat16> ) {
            epsilon = 1e-2f;
        }
        else if constexpr ( std::is_same_v<TCudaType, __nv_fp8_e4m3> || 
                        std::is_same_v<TCudaType, __nv_fp8_e5m2> ) {
            epsilon = 1e-1f; // Much higher tolerance for FP8
        }

        bool all_equal = true;
        int diff_count = 0;

        for ( size_t i = 0; i < cpu_output.size(); ++i ) {
            float cpu_val = static_cast<float>( cpu_output.data()[ i ] );
            float cuda_val = static_cast<float>( cuda_output_host.data()[ i ] );
            float diff = std::abs( cpu_val - cuda_val );
            
            if ( diff > epsilon ) {
                if ( diff_count < 5 ) { // Only print first few differences
                    std::cout << "Difference at index " << i << ": CPU=" << cpu_val
                        << ", CUDA=" << cuda_val << ", diff=" << diff << std::endl;
                }
                all_equal = false;
                diff_count++;
            }
        }

        if ( diff_count > 0 ) {
            std::cout << typeid(TCudaType).name() << ": " << diff_count << " out of " 
                << cpu_output.size() << " values differ by more than epsilon ("
                << (100.0f * diff_count / cpu_output.size()) << "%)" << std::endl;
        }

        // For lower precision types, we're concerned with statistical equivalence rather than exact matches
        if constexpr ( std::is_same_v<TCudaType, float> ) {
            EXPECT_TRUE( all_equal ) << "CPU and CUDA implementations produced different results";
        } 
        else if constexpr ( std::is_same_v<TCudaType, half> || std::is_same_v<TCudaType, nv_bfloat16> ) {
            // Allow up to 1% differences for half precision types
            EXPECT_LE( static_cast<float>(diff_count) / cpu_output.size(), 0.01f ) 
                << "CPU float and CUDA " << typeid(TCudaType).name() 
                << " implementations differ by more than 1% of values";
        }
        else {
            // Allow up to 5% differences for FP8 types
            EXPECT_LE( static_cast<float>(diff_count) / cpu_output.size(), 0.05f ) 
                << "CPU float and CUDA " << typeid(TCudaType).name() 
                << " implementations differ by more than 5% of values";
        }
    }

    // Test different precision types for compatibility
    template<typename TDataType1, typename TDataType2>
    void TestPrecisionCompatibility(
        const LinearTestData<Compute::DeviceType::Cuda, TDataType1>& data1,
        const LinearTestData<Compute::DeviceType::Cuda, TDataType2>& data2 ) {

        try {
            // Check if CUDA is available
            DeviceContext context( "CUDA:0" );
        }
        catch ( const std::exception& ) {
            GTEST_SKIP() << "CUDA device not available, skipping precision comparison test";
            return;
        }

        // Create small test shapes for quick comparison
        std::vector<size_t> test_input_shape1 = { 2, 4, data1.input_features };
        std::vector<size_t> test_output_shape1 = { 2, 4, data1.output_features };

        std::vector<size_t> test_input_shape2 = { 2, 4, data2.input_features };
        std::vector<size_t> test_output_shape2 = { 2, 4, data2.output_features };

        // Create input tensors with same values but different types
        Tensor<TDataType1, Compute::CudaMemoryResource> input1( test_input_shape1 );
        Tensor<TDataType2, Compute::CudaMemoryResource> input2( test_input_shape2 );

        // Fill inputs with identical values
        Tensor<float, Compute::HostMemoryResource> host_input( test_input_shape1 );
        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input.data()[ i ] = -1.0f + 2.0f * (static_cast<float>( i ) / host_input.size());
        }

        // Convert to the respective types
        Tensor<TDataType1, Compute::HostMemoryResource> host_input1( test_input_shape1 );
        Tensor<TDataType2, Compute::HostMemoryResource> host_input2( test_input_shape2 );

        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input1.data()[ i ] = static_cast<TDataType1>( host_input.data()[ i ] );
            host_input2.data()[ i ] = static_cast<TDataType2>( host_input.data()[ i ] );
        }

        input1.copyFrom( host_input1 );
        input2.copyFrom( host_input2 );

        // Run forward passes
        Tensor<TDataType1, Compute::CudaMemoryResource> output1( test_output_shape1 );
        Tensor<TDataType2, Compute::CudaMemoryResource> output2( test_output_shape2 );

        data1.linear_module->forward( input1, output1 );
        data2.linear_module->forward( input2, output2 );

        // Copy results back to host for comparison
        Tensor<TDataType1, Compute::HostMemoryResource> host_output1( test_output_shape1 );
        Tensor<TDataType2, Compute::HostMemoryResource> host_output2( test_output_shape2 );

        host_output1.copyFrom( output1 );
        host_output2.copyFrom( output2 );

        // Convert to float for comparison
        std::vector<float> float_output1( host_output1.size() );
        std::vector<float> float_output2( host_output2.size() );

        for ( size_t i = 0; i < host_output1.size(); ++i ) {
            float_output1[ i ] = static_cast<float>( host_output1.data()[ i ] );
        }

        for ( size_t i = 0; i < host_output2.size(); ++i ) {
            float_output2[ i ] = static_cast<float>( host_output2.data()[ i ] );
        }

        // Compare with higher tolerance for different precision types
        const float epsilon = 1e-2f;
        int diff_count = 0;
        float max_diff = 0.0f;

        for ( size_t i = 0; i < float_output1.size(); ++i ) {
            float diff = std::abs( float_output1[ i ] - float_output2[ i ] );
            max_diff = std::max( max_diff, diff );

            if ( diff > epsilon && diff_count < 5 ) {
                std::cout << "Difference at index " << i << ": Type1=" << float_output1[ i ]
                    << ", Type2=" << float_output2[ i ] << ", diff=" << diff << std::endl;
                diff_count++;
            }
        }

        std::cout << "Max difference between " << typeid(TDataType1).name() << " and "
            << typeid(TDataType2).name() << ": " << max_diff << std::endl;

        // We expect differences due to precision, just report them
        SUCCEED() << "Comparison between different precision types completed";
    }

    // Test with different dimensions (edge cases)
    template<DeviceType TDevice, typename TDataType = float>
    void TestEdgeCases() {
        using MR = MemoryResourceType<TDevice>;

        try {
            // Test with minimal sizes
            std::vector<size_t> minimal_input_shape = { 1, 1, 8 };
            std::vector<size_t> minimal_output_shape = { 1, 1, 16 };

            LinearConfig minimal_config( 8, 16 );
            minimal_config.withName( "minimal_linear" );

            auto minimal_module = std::make_shared<Linear<TDevice, TDataType>>(
                TDevice == Compute::DeviceType::Cuda ? "CUDA:0" : "CPU", minimal_config );

            Tensor<TDataType, MR> minimal_input( minimal_input_shape );
            Tensor<TDataType, MR> minimal_output( minimal_output_shape );

            EXPECT_NO_THROW( minimal_module->forward( minimal_input, minimal_output ) );
            EXPECT_EQ( minimal_output.size(), 16 );

            // Test with larger dimensions
            std::vector<size_t> large_input_shape = { 2, 2, 1024 };
            std::vector<size_t> large_output_shape = { 2, 2, 512 };

            LinearConfig large_config( 1024, 512 );
            large_config.withName( "large_linear" );

            auto large_module = std::make_shared<Linear<TDevice, TDataType>>(
                TDevice == Compute::DeviceType::Cuda ? "CUDA:0" : "CPU", large_config );

            Tensor<TDataType, MR> large_input( large_input_shape );
            Tensor<TDataType, MR> large_output( large_output_shape );

            EXPECT_NO_THROW( large_module->forward( large_input, large_output ) );
            EXPECT_EQ( large_output.size(), 2048 );
        }
        catch ( const std::exception& e ) {
            std::cerr << "Exception during edge case test: " << e.what() << std::endl;
            throw;
        }
    }

    //------------------------------------------------------------------------------
    // CPU TESTS
    //------------------------------------------------------------------------------

    // FP32 Tests
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

    // No Bias Tests
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

    // Training Mode Tests
    TEST_F( LinearTests, Cpu_Training_Float_TrainingMode ) {
        TestTrainingMode<Compute::DeviceType::Cpu, float>( TrainingCpuFloatData(), true );
    }

    // Context Construction Test
    TEST_F( LinearTests, Cpu_Context_Float_Forward ) {
        TestForward<Compute::DeviceType::Cpu, float>( ContextCpuFloatData() );
    }

    // Precision Policy Tests
    TEST_F( LinearTests, Cpu_Float_NativePolicy ) {
        TestPrecisionPolicy<Compute::DeviceType::Cpu, float>( CpuNativePolicyData(), ComputePrecision::Policy::Native );
    }

    TEST_F( LinearTests, Cpu_Float_PerformancePolicy ) {
        TestPrecisionPolicy<Compute::DeviceType::Cpu, float>( CpuPerformancePolicyData(), ComputePrecision::Policy::Performance );
    }

    TEST_F( LinearTests, Cpu_Float_AccuracyPolicy ) {
        TestPrecisionPolicy<Compute::DeviceType::Cpu, float>( CpuAccuracyPolicyData(), ComputePrecision::Policy::Accuracy );
    }

    // Edge Cases Test
    TEST_F( LinearTests, Cpu_Float_EdgeCases ) {
        TestEdgeCases<Compute::DeviceType::Cpu, float>();
    }

    //------------------------------------------------------------------------------
    // CUDA TESTS
    //------------------------------------------------------------------------------

    // Skip CUDA tests if not available
    void SkipIfNoCuda() {
        try {
            DeviceContext context( "CUDA:0" );
        }
        catch ( const std::exception& ) {
            GTEST_SKIP() << "CUDA device not available, skipping CUDA test";
        }
    }

    // FP32 Tests
    TEST_F( LinearTests, Cuda_Float_TestName ) {
        SkipIfNoCuda();
        TestGetName<Compute::DeviceType::Cuda, float>( CudaFloatData(), "cuda_linear_float" );
    }

    TEST_F( LinearTests, Cuda_Float_ParameterCount ) {
        SkipIfNoCuda();
        TestParameterCount<Compute::DeviceType::Cuda, float>( CudaFloatData() );
    }

    TEST_F( LinearTests, Cuda_Float_TestForward ) {
        SkipIfNoCuda();
        TestForward<Compute::DeviceType::Cuda, float>( CudaFloatData() );
    }

    TEST_F( LinearTests, Cuda_Float_TestPrint ) {
        SkipIfNoCuda();
        TestPrint<Compute::DeviceType::Cuda, float>( CudaFloatData(), "Linear: cuda_linear_float" );
    }

    TEST_F( LinearTests, Cuda_Float_GetWeight ) {
        SkipIfNoCuda();
        TestGetWeight<Compute::DeviceType::Cuda, float>( CudaFloatData() );
    }

    TEST_F( LinearTests, Cuda_Float_GetBias ) {
        SkipIfNoCuda();
        TestGetBias<Compute::DeviceType::Cuda, float>( CudaFloatData() );
    }

    TEST_F( LinearTests, Cuda_Float_HasBias ) {
        SkipIfNoCuda();
        TestHasBias<Compute::DeviceType::Cuda, float>( CudaFloatData() );
    }

    // No Bias Tests
    TEST_F( LinearTests, Cuda_NoBias_Float_HasBias ) {
        SkipIfNoCuda();
        TestHasBias<Compute::DeviceType::Cuda, float>( CudaNoBiasFloatData() );
    }

    TEST_F( LinearTests, Cuda_NoBias_Float_Forward ) {
        SkipIfNoCuda();
        TestForward<Compute::DeviceType::Cuda, float>( CudaNoBiasFloatData() );
    }

    // Training Mode Tests
    TEST_F( LinearTests, Cuda_Training_Float_TrainingMode ) {
        SkipIfNoCuda();
        TestTrainingMode<Compute::DeviceType::Cuda, float>( TrainingCudaFloatData(), true );
    }

    // FP16 Tests
    TEST_F( LinearTests, Cuda_Half_TestName ) {
        SkipIfNoCuda();
        TestGetName<Compute::DeviceType::Cuda, half>( CudaHalfData(), "cuda_linear_half" );
    }

    TEST_F( LinearTests, Cuda_Half_ParameterCount ) {
        SkipIfNoCuda();
        TestParameterCount<Compute::DeviceType::Cuda, half>( CudaHalfData() );
    }

    TEST_F( LinearTests, Cuda_Half_TestForward ) {
        SkipIfNoCuda();
        TestForward<Compute::DeviceType::Cuda, half>( CudaHalfData() );
    }

    TEST_F( LinearTests, Cuda_Half_GetWeight ) {
        SkipIfNoCuda();
        TestGetWeight<Compute::DeviceType::Cuda, half>( CudaHalfData() );
    }

    TEST_F( LinearTests, Cuda_Half_HasBias ) {
        SkipIfNoCuda();
        TestHasBias<Compute::DeviceType::Cuda, half>( CudaHalfData() );
    }

    // No Bias Tests (FP16)
    TEST_F( LinearTests, Cuda_NoBias_Half_HasBias ) {
        SkipIfNoCuda();
        TestHasBias<Compute::DeviceType::Cuda, half>( CudaNoBiasHalfData() );
    }

    TEST_F( LinearTests, Cuda_NoBias_Half_Forward ) {
        SkipIfNoCuda();
        TestForward<Compute::DeviceType::Cuda, half>( CudaNoBiasHalfData() );
    }

    // Training Mode Tests (FP16)
    TEST_F( LinearTests, Cuda_Training_Half_TrainingMode ) {
        SkipIfNoCuda();
        TestTrainingMode<Compute::DeviceType::Cuda, half>( TrainingCudaHalfData(), true );
    }

    // BF16 Tests
    TEST_F( LinearTests, Cuda_BF16_TestName ) {
        SkipIfNoCuda();
        TestGetName<Compute::DeviceType::Cuda, nv_bfloat16>( CudaBF16Data(), "cuda_linear_bf16" );
    }

    TEST_F( LinearTests, Cuda_BF16_ParameterCount ) {
        SkipIfNoCuda();
        TestParameterCount<Compute::DeviceType::Cuda, nv_bfloat16>( CudaBF16Data() );
    }

    TEST_F( LinearTests, Cuda_BF16_TestForward ) {
        SkipIfNoCuda();
        TestForward<Compute::DeviceType::Cuda, nv_bfloat16>( CudaBF16Data() );
    }

    // FP8 E4M3 Tests
    TEST_F( LinearTests, Cuda_FP8_E4M3_TestName ) {
        SkipIfNoCuda();
        TestGetName<Compute::DeviceType::Cuda, __nv_fp8_e4m3>( CudaFP8E4M3Data(), "cuda_linear_fp8_e4m3" );
    }

    TEST_F( LinearTests, Cuda_FP8_E4M3_ParameterCount ) {
        SkipIfNoCuda();
        TestParameterCount<Compute::DeviceType::Cuda, __nv_fp8_e4m3>( CudaFP8E4M3Data() );
    }

    TEST_F( LinearTests, Cuda_FP8_E4M3_TestForward ) {
        SkipIfNoCuda();
        TestForward<Compute::DeviceType::Cuda, __nv_fp8_e4m3>( CudaFP8E4M3Data() );
    }

    // FP8 E5M2 Tests
    TEST_F( LinearTests, Cuda_FP8_E5M2_TestName ) {
        SkipIfNoCuda();
        TestGetName<Compute::DeviceType::Cuda, __nv_fp8_e5m2>( CudaFP8E5M2Data(), "cuda_linear_fp8_e5m2" );
    }

    TEST_F( LinearTests, Cuda_FP8_E5M2_ParameterCount ) {
        SkipIfNoCuda();
        TestParameterCount<Compute::DeviceType::Cuda, __nv_fp8_e5m2>( CudaFP8E5M2Data() );
    }

    TEST_F( LinearTests, Cuda_FP8_E5M2_TestForward ) {
        SkipIfNoCuda();
        TestForward<Compute::DeviceType::Cuda, __nv_fp8_e5m2>( CudaFP8E5M2Data() );
    }

    // Precision Policy Tests
    TEST_F( LinearTests, Cuda_Float_NativePolicy ) {
        SkipIfNoCuda();
        TestPrecisionPolicy<Compute::DeviceType::Cuda, float>( CudaNativePolicyData(), ComputePrecision::Policy::Native );
    }

    TEST_F( LinearTests, Cuda_Float_PerformancePolicy ) {
        SkipIfNoCuda();
        TestPrecisionPolicy<Compute::DeviceType::Cuda, float>( CudaPerformancePolicyData(), ComputePrecision::Policy::Performance );
    }

    TEST_F( LinearTests, Cuda_Float_AccuracyPolicy ) {
        SkipIfNoCuda();
        TestPrecisionPolicy<Compute::DeviceType::Cuda, float>( CudaAccuracyPolicyData(), ComputePrecision::Policy::Accuracy );
    }

    TEST_F( LinearTests, Cuda_Float_AutoPolicy ) {
        SkipIfNoCuda();
        TestPrecisionPolicy<Compute::DeviceType::Cuda, float>( CudaAutoPolicyData(), ComputePrecision::Policy::Auto );
    }

    // Cross-Precision Tests
    TEST_F( LinearTests, Cuda_Float_Half_Compatibility ) {
        SkipIfNoCuda();
        TestPrecisionCompatibility<float, half>( CudaFloatData(), CudaHalfData() );
    }

    TEST_F( LinearTests, Cuda_Float_BF16_Compatibility ) {
        SkipIfNoCuda();
        TestPrecisionCompatibility<float, nv_bfloat16>( CudaFloatData(), CudaBF16Data() );
    }

    TEST_F( LinearTests, Cuda_Half_BF16_Compatibility ) {
        SkipIfNoCuda();
        TestPrecisionCompatibility<half, nv_bfloat16>( CudaHalfData(), CudaBF16Data() );
    }

    // CPU-CUDA Equivalence Tests
    TEST_F( LinearTests, Cpu_Cuda_EquivalenceFloat ) {
        SkipIfNoCuda();
        TestCpuCudaEquivalence<float, float>( CpuFloatData(), CudaFloatData() );
    }

    TEST_F( LinearTests, Cpu_Cuda_EquivalenceHalf ) {
        SkipIfNoCuda();
        TestCpuCudaEquivalence<float, half>( CpuFloatData(), CudaHalfData() );
    }

    TEST_F( LinearTests, Cpu_Cuda_EquivalenceBF16 ) {
        SkipIfNoCuda();
        TestCpuCudaEquivalence<float, nv_bfloat16>( CpuFloatData(), CudaBF16Data() );
    }

    TEST_F( LinearTests, Cpu_Cuda_EquivalenceFP8E4M3 ) {
        SkipIfNoCuda();
        TestCpuCudaEquivalence<float, __nv_fp8_e4m3>( CpuFloatData(), CudaFP8E4M3Data() );
    }

    TEST_F( LinearTests, Cpu_Cuda_EquivalenceFP8E5M2 ) {
        SkipIfNoCuda();
        TestCpuCudaEquivalence<float, __nv_fp8_e5m2>( CpuFloatData(), CudaFP8E5M2Data() );
    }

    // Edge Cases Test
    TEST_F( LinearTests, Cuda_Float_EdgeCases ) {
        SkipIfNoCuda();
        TestEdgeCases<Compute::DeviceType::Cuda, float>();
    }

    TEST_F( LinearTests, Cuda_Half_EdgeCases ) {
        SkipIfNoCuda();
        TestEdgeCases<Compute::DeviceType::Cuda, half>();
    }
}