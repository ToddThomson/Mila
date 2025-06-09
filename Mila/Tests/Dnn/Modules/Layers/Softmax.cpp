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
    template<DeviceType TDevice>
    using MemoryResourceType = std::conditional_t<TDevice == DeviceType::Cuda,
        Compute::CudaMemoryResource,
        Compute::HostMemoryResource>;

    // Test data structure for Softmax tests
    template<DeviceType TDevice, typename TInput = float, typename TOutput = TInput>
    struct SoftmaxTestData {
        std::vector<size_t> shape;
        std::shared_ptr<Softmax<TDevice, TInput, TOutput>> softmax_module;
        int64_t axis;
        bool is_training;

        // Make the test data structure self-initializing
        static SoftmaxTestData Create(
            const std::string& name,
            size_t batch_size,
            size_t sequence_length,
            size_t vocab_size,
            int64_t axis = -1,
            bool is_training = false,
            ComputePrecision::Policy precision = ComputePrecision::Policy::Auto )
        {
            SoftmaxTestData data;
            data.shape = { batch_size, sequence_length, vocab_size };
            data.axis = axis;
            data.is_training = is_training;

            std::string device_str = TDevice == DeviceType::Cuda ? "CUDA:0" : "CPU";
            data.softmax_module = std::make_shared<Softmax<TDevice, TInput, TOutput>>(
                name, device_str, axis, is_training, precision );

            return data;
        }

        // Overload for creating with device context
        static SoftmaxTestData CreateWithContext(
            const std::string& name,
            size_t batch_size,
            size_t sequence_length,
            size_t vocab_size,
            std::shared_ptr<DeviceContext> context,
            int64_t axis = -1,
            bool is_training = false,
            ComputePrecision::Policy precision = ComputePrecision::Policy::Auto )
        {
            SoftmaxTestData data;
            data.shape = { batch_size, sequence_length, vocab_size };
            data.axis = axis;
            data.is_training = is_training;

            data.softmax_module = std::make_shared<Softmax<TDevice, TInput, TOutput>>(
                name, context, axis, is_training, precision );

            return data;
        }
    };

    class SoftmaxTests : public ::testing::Test {
    protected:
        void SetUp() override {
            // Initialize test parameters only
            batch_size_ = 64;
            cpu_batch_size_ = 4;
            sequence_length_ = 128;
            vocab_size_ = 1024;
            axis_ = -1;
            // Modules will be created on demand
        }

        // Factory methods to lazily create test data as needed
        SoftmaxTestData<DeviceType::Cpu, float>& CpuFloatData() {
            if ( !cpu_float_data_.softmax_module ) {
                cpu_float_data_ = SoftmaxTestData<DeviceType::Cpu, float>::Create(
                    "cpu_softmax_float", cpu_batch_size_, sequence_length_, vocab_size_, axis_ );
            }
            return cpu_float_data_;
        }

        SoftmaxTestData<DeviceType::Cuda, float>& CudaFloatData() {
            if ( !cuda_float_data_.softmax_module ) {
                cuda_float_data_ = SoftmaxTestData<DeviceType::Cuda, float>::Create(
                    "cuda_softmax_float", batch_size_, sequence_length_, vocab_size_, axis_ );
            }
            return cuda_float_data_;
        }

        SoftmaxTestData<DeviceType::Cpu, float>& TrainingCpuFloatData() {
            if ( !training_cpu_float_data_.softmax_module ) {
                training_cpu_float_data_ = SoftmaxTestData<DeviceType::Cpu, float>::Create(
                    "cpu_softmax_float_training", cpu_batch_size_, sequence_length_, vocab_size_, axis_, true );
            }
            return training_cpu_float_data_;
        }

        SoftmaxTestData<DeviceType::Cuda, float>& TrainingCudaFloatData() {
            if ( !training_cuda_float_data_.softmax_module ) {
                training_cuda_float_data_ = SoftmaxTestData<DeviceType::Cuda, float>::Create(
                    "cuda_softmax_float_training", batch_size_, sequence_length_, vocab_size_, axis_, true );
            }
            return training_cuda_float_data_;
        }

        SoftmaxTestData<DeviceType::Cpu, float>& ContextCpuFloatData() {
            if ( !context_cpu_float_data_.softmax_module ) {
                auto cpu_context = std::make_shared<DeviceContext>( "CPU" );
                context_cpu_float_data_ = SoftmaxTestData<DeviceType::Cpu, float>::CreateWithContext(
                    "cpu_context_softmax_float", cpu_batch_size_, sequence_length_, vocab_size_, cpu_context, axis_ );
            }
            return context_cpu_float_data_;
        }

        SoftmaxTestData<DeviceType::Cuda, half>& CudaHalfData() {
            if ( !cuda_half_data_.softmax_module ) {
                cuda_half_data_ = SoftmaxTestData<DeviceType::Cuda, half>::Create(
                    "cuda_softmax_half", batch_size_, sequence_length_, vocab_size_, axis_ );
            }
            return cuda_half_data_;
        }

        SoftmaxTestData<DeviceType::Cuda, half>& TrainingCudaHalfData() {
            if ( !training_cuda_half_data_.softmax_module ) {
                training_cuda_half_data_ = SoftmaxTestData<DeviceType::Cuda, half>::Create(
                    "cuda_softmax_half_training", batch_size_, sequence_length_, vocab_size_, axis_, true );
            }
            return training_cuda_half_data_;
        }

        // Test for mixed precision (input float, output half)
        SoftmaxTestData<DeviceType::Cuda, float, half>& MixedPrecisionData() {
            if ( !mixed_precision_data_.softmax_module ) {
                mixed_precision_data_ = SoftmaxTestData<DeviceType::Cuda, float, half>::Create(
                    "cuda_softmax_mixed", batch_size_, sequence_length_, vocab_size_, axis_ );
            }
            return mixed_precision_data_;
        }

        // Test with specific precision policies
        SoftmaxTestData<DeviceType::Cuda, float>& PerformancePolicyData() {
            if ( !performance_policy_data_.softmax_module ) {
                performance_policy_data_ = SoftmaxTestData<DeviceType::Cuda, float>::Create(
                    "cuda_softmax_performance", batch_size_, sequence_length_, vocab_size_, axis_, false,
                    ComputePrecision::Policy::Performance );
            }
            return performance_policy_data_;
        }

        SoftmaxTestData<DeviceType::Cuda, float>& AccuracyPolicyData() {
            if ( !accuracy_policy_data_.softmax_module ) {
                accuracy_policy_data_ = SoftmaxTestData<DeviceType::Cuda, float>::Create(
                    "cuda_softmax_accuracy", batch_size_, sequence_length_, vocab_size_, axis_, false,
                    ComputePrecision::Policy::Accuracy );
            }
            return accuracy_policy_data_;
        }

        SoftmaxTestData<DeviceType::Cuda, float>& DisabledPolicyData() {
            if ( !disabled_policy_data_.softmax_module ) {
                disabled_policy_data_ = SoftmaxTestData<DeviceType::Cuda, float>::Create(
                    "cuda_softmax_disabled", batch_size_, sequence_length_, vocab_size_, axis_, false,
                    ComputePrecision::Policy::Native );
            }
            return disabled_policy_data_;
        }

        // Test parameters
        size_t batch_size_{ 0 };
        size_t cpu_batch_size_{ 0 };
        size_t sequence_length_{ 0 };
        size_t vocab_size_{ 0 };
        int64_t axis_{ -1 };

        // Test data objects - initialized on demand
        SoftmaxTestData<DeviceType::Cpu, float> cpu_float_data_;
        SoftmaxTestData<DeviceType::Cpu, float> context_cpu_float_data_;
        SoftmaxTestData<DeviceType::Cpu, float> training_cpu_float_data_;

        SoftmaxTestData<DeviceType::Cuda, float> cuda_float_data_;
        SoftmaxTestData<DeviceType::Cuda, float> training_cuda_float_data_;

        SoftmaxTestData<DeviceType::Cuda, half> cuda_half_data_;
        SoftmaxTestData<DeviceType::Cuda, half> training_cuda_half_data_;

        // Mixed precision test data (float input to half output)
        SoftmaxTestData<DeviceType::Cuda, float, half> mixed_precision_data_;

        // Precision policy test data
        SoftmaxTestData<DeviceType::Cuda, float> performance_policy_data_;
        SoftmaxTestData<DeviceType::Cuda, float> accuracy_policy_data_;
        SoftmaxTestData<DeviceType::Cuda, float> disabled_policy_data_;
    };

    // Common test function templates
    template<DeviceType TDevice, typename TInput, typename TOutput = TInput>
    void TestGetName( const SoftmaxTestData<TDevice, TInput, TOutput>& data, const std::string& expected_name ) {
        EXPECT_EQ( data.softmax_module->getName(), expected_name );
    }

    template<DeviceType TDevice, typename TInput, typename TOutput = TInput>
    void TestParameterCount( const SoftmaxTestData<TDevice, TInput, TOutput>& data, size_t expected_count ) {
        EXPECT_EQ( data.softmax_module->parameterCount(), expected_count );
    }

    template<DeviceType TDevice, typename TInput, typename TOutput = TInput>
    void TestForward( const SoftmaxTestData<TDevice, TInput, TOutput>& data ) {
        using MR = MemoryResourceType<TDevice>;

        Tensor<TInput, MR> input( data.shape );
        Tensor<TOutput, MR> output( data.shape );

        // Fill with random values to test softmax normalization
        random<TInput, MR>( input, -5.0f, 5.0f );

        data.softmax_module->forward( input, output );
        EXPECT_EQ( output.size(), input.size() );

        // For each sample, check if softmax values sum to 1 along the specified axis
        if constexpr ( TDevice == DeviceType::Cpu ) {
            // For CPU tensors, we can directly verify the normalization property
            auto B = output.shape()[ 0 ];
            auto T = output.shape()[ 1 ];
            auto V = output.shape()[ 2 ];

            // The default axis is -1 (last dimension), which is the vocab dimension in our case
            for ( size_t i = 0; i < B; ++i ) {
                for ( size_t j = 0; j < T; ++j ) {
                    // Sum values across the vocabulary dimension
                    float sum = 0.0f;
                    for ( size_t v = 0; v < V; ++v ) {
                        sum += static_cast<float>( output[ i, j, v ] );
                    }
                    EXPECT_NEAR( sum, 1.0f, 1e-4f );
                }
            }
        }
    }

    template<DeviceType TDevice, typename TInput, typename TOutput = TInput>
    void TestPrint( const SoftmaxTestData<TDevice, TInput, TOutput>& data, const std::string& expected_substring ) {
        std::string output = data.softmax_module->toString();
        EXPECT_NE( output.find( expected_substring ), std::string::npos );
        // Also verify the dimension information is included
        std::string dimension_info = "Dimension: " + std::to_string( data.axis );
        EXPECT_NE( output.find( dimension_info ), std::string::npos );
        // Verify precision policy information is included
        EXPECT_NE( output.find( "Precision Policy:" ), std::string::npos );
    }

    template<DeviceType TDevice, typename TInput, typename TOutput = TInput>
    void TestPrecisionPolicy( const SoftmaxTestData<TDevice, TInput, TOutput>& data, ComputePrecision::Policy expected_policy ) {
        std::string output = data.softmax_module->toString();

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

        std::string expected_text = "Precision Policy: " + policy_string;
        EXPECT_NE( output.find( expected_text ), std::string::npos );
    }

    template<DeviceType TDevice, typename TInput, typename TOutput = TInput>
    void TestTrainingMode( const SoftmaxTestData<TDevice, TInput, TOutput>& data, bool expected_mode ) {
        EXPECT_EQ( data.softmax_module->isTraining(), expected_mode );
    }

    template<DeviceType TDevice, typename TInput, typename TOutput = TInput>
    void TestDeviceType( const SoftmaxTestData<TDevice, TInput, TOutput>& data ) {
        auto device_context = data.softmax_module->getDeviceContext();
        EXPECT_NE( device_context, nullptr );
        auto device = device_context->getDevice();
        EXPECT_NE( device, nullptr );
        EXPECT_EQ( device->getDeviceType(), TDevice );
    }

    // Function to test equivalence of CPU and CUDA outputs
    template<typename TInput, typename TOutput = TInput>
    void TestCpuCudaEquivalence(
        const SoftmaxTestData<DeviceType::Cpu, TInput, TOutput>& cpu_data,
        const SoftmaxTestData<DeviceType::Cuda, TInput, TOutput>& cuda_data ) {

        // Create a small test shape to make comparison faster
        std::vector<size_t> test_shape = { 2, 4, 8 }; // Small shape for quick verification

        // Create random input data
        Tensor<TInput, HostMemoryResource> host_input( test_shape );

        // Fill with predictable values between -2.0 and 2.0 
        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input.data()[ i ] = static_cast<TInput>( -2.0 + 4.0 * (static_cast<float>( i ) / host_input.size()) );
        }

        // Create CPU output
        Tensor<TOutput, HostMemoryResource> cpu_output( test_shape );
        cpu_data.softmax_module->forward( host_input, cpu_output );

        // Create device input by copying host data
        Tensor<TInput, CudaMemoryResource> device_input( test_shape );
        device_input.copyFrom( host_input );

        // Create device output
        Tensor<TOutput, CudaMemoryResource> cuda_output( test_shape );
        cuda_data.softmax_module->forward( device_input, cuda_output );

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
            }
        }

        EXPECT_TRUE( all_equal ) << "CPU and CUDA implementations produced different results";
    }

    // Test with different dimensions (edge cases)
    template<DeviceType TDevice, typename TInput, typename TOutput = TInput>
    void TestEdgeCases() {
        using MR = MemoryResourceType<TDevice>;

        try {
            // Test with minimal sizes
            std::vector<size_t> minimal_shape = { 1, 1, 8 };

            auto minimal_module = std::make_shared<Softmax<TDevice, TInput, TOutput>>(
                "minimal_softmax", TDevice == DeviceType::Cuda ? "CUDA:0" : "CPU" );

            Tensor<TInput, MR> minimal_input( minimal_shape );
            Tensor<TOutput, MR> minimal_output( minimal_shape );

            EXPECT_NO_THROW( minimal_module->forward( minimal_input, minimal_output ) );
            EXPECT_EQ( minimal_output.size(), 8 );

            // Test with larger dimensions
            std::vector<size_t> large_shape = { 2, 2, 1024 };

            auto large_module = std::make_shared<Softmax<TDevice, TInput, TOutput>>(
                "large_softmax", TDevice == DeviceType::Cuda ? "CUDA:0" : "CPU" );

            Tensor<TInput, MR> large_input( large_shape );
            Tensor<TOutput, MR> large_output( large_shape );

            EXPECT_NO_THROW( large_module->forward( large_input, large_output ) );
            EXPECT_EQ( large_output.size(), 4096 );
        }
        catch ( const std::exception& e ) {
            std::cerr << "Exception during edge case test: " << e.what() << std::endl;
            throw;
        }
    }

    // Test for different axis specifications
    template<DeviceType TDevice, typename TInput, typename TOutput = TInput>
    void TestDifferentAxes() {
        using MR = MemoryResourceType<TDevice>;

        // Test shape with 3 dimensions
        std::vector<size_t> test_shape = { 2, 3, 4 };

        // Test softmax on axis 0 (batch dimension)
        auto axis0_module = std::make_shared<Softmax<TDevice, TInput, TOutput>>(
            "axis0_softmax", TDevice == DeviceType::Cuda ? "CUDA:0" : "CPU", 0 );

        Tensor<TInput, MR> input( test_shape );
        random<TInput, MR>( input, -1.0f, 1.0f );

        Tensor<TOutput, MR> output0( test_shape );
        EXPECT_NO_THROW( axis0_module->forward( input, output0 ) );

        // Test softmax on axis 1 (sequence dimension)
        auto axis1_module = std::make_shared<Softmax<TDevice, TInput, TOutput>>(
            "axis1_softmax", TDevice == DeviceType::Cuda ? "CUDA:0" : "CPU", 1 );

        Tensor<TOutput, MR> output1( test_shape );
        EXPECT_NO_THROW( axis1_module->forward( input, output1 ) );

        // Test softmax on axis 2 (vocab/feature dimension) - this is the default
        auto axis2_module = std::make_shared<Softmax<TDevice, TInput, TOutput>>(
            "axis2_softmax", TDevice == DeviceType::Cuda ? "CUDA:0" : "CPU", 2 );

        Tensor<TOutput, MR> output2( test_shape );
        EXPECT_NO_THROW( axis2_module->forward( input, output2 ) );
    }

    // CPU Tests with float precision
    TEST_F( SoftmaxTests, Cpu_Float_TestName ) {
        TestGetName<DeviceType::Cpu, float>( CpuFloatData(), "cpu_softmax_float" );
    }

    TEST_F( SoftmaxTests, Cpu_Float_ParameterCount ) {
        TestParameterCount<DeviceType::Cpu, float>( CpuFloatData(), 0 );
    }

    TEST_F( SoftmaxTests, Cpu_Float_TestForward ) {
        TestForward<DeviceType::Cpu, float>( CpuFloatData() );
    }

    TEST_F( SoftmaxTests, Cpu_Float_TestPrint ) {
        TestPrint<DeviceType::Cpu, float>( CpuFloatData(), "Softmax: cpu_softmax_float" );
    }

    TEST_F( SoftmaxTests, Cpu_Float_TrainingMode ) {
        TestTrainingMode<DeviceType::Cpu, float>( CpuFloatData(), false );
    }

    TEST_F( SoftmaxTests, Cpu_Float_DeviceType ) {
        TestDeviceType<DeviceType::Cpu, float>( CpuFloatData() );
    }

    // CPU Training Mode Tests
    TEST_F( SoftmaxTests, Cpu_Training_Float_TrainingMode ) {
        TestTrainingMode<DeviceType::Cpu, float>( TrainingCpuFloatData(), true );
    }

    // CUDA Tests with float precision
    TEST_F( SoftmaxTests, Cuda_Float_TestName ) {
        TestGetName<DeviceType::Cuda, float>( CudaFloatData(), "cuda_softmax_float" );
    }

    TEST_F( SoftmaxTests, Cuda_Float_ParameterCount ) {
        TestParameterCount<DeviceType::Cuda, float>( CudaFloatData(), 0 );
    }

    TEST_F( SoftmaxTests, Cuda_Float_TestForward ) {
        TestForward<DeviceType::Cuda, float>( CudaFloatData() );
    }

    TEST_F( SoftmaxTests, Cuda_Float_TestPrint ) {
        TestPrint<DeviceType::Cuda, float>( CudaFloatData(), "Softmax: cuda_softmax_float" );
    }

    TEST_F( SoftmaxTests, Cuda_Float_TrainingMode ) {
        TestTrainingMode<DeviceType::Cuda, float>( CudaFloatData(), false );
    }

    TEST_F( SoftmaxTests, Cuda_Float_DeviceType ) {
        TestDeviceType<DeviceType::Cuda, float>( CudaFloatData() );
    }

    // CUDA Training Mode Tests
    TEST_F( SoftmaxTests, Cuda_Training_Float_TrainingMode ) {
        TestTrainingMode<DeviceType::Cuda, float>( TrainingCudaFloatData(), true );
    }

    // CUDA Tests with half precision
    TEST_F( SoftmaxTests, Cuda_Half_TestName ) {
        TestGetName<DeviceType::Cuda, half>( CudaHalfData(), "cuda_softmax_half" );
    }

    TEST_F( SoftmaxTests, Cuda_Half_ParameterCount ) {
        TestParameterCount<DeviceType::Cuda, half>( CudaHalfData(), 0 );
    }

    TEST_F( SoftmaxTests, Cuda_Half_TestForward ) {
        TestForward<DeviceType::Cuda, half>( CudaHalfData() );
    }

    TEST_F( SoftmaxTests, Cuda_Half_TestPrint ) {
        TestPrint<DeviceType::Cuda, half>( CudaHalfData(), "Softmax: cuda_softmax_half" );
    }

    TEST_F( SoftmaxTests, Cuda_Half_TrainingMode ) {
        TestTrainingMode<DeviceType::Cuda, half>( CudaHalfData(), false );
    }

    // Mixed Precision Tests (new in updated module)
    TEST_F( SoftmaxTests, Cuda_MixedPrecision_TestForward ) {
        // FIXME: TestForward<DeviceType::Cuda, float, half>( MixedPrecisionData() );
    }

    TEST_F( SoftmaxTests, Cuda_MixedPrecision_TestName ) {
        // FIXME: TestGetName<DeviceType::Cuda, float, half>( MixedPrecisionData(), "cuda_softmax_mixed" );
    }

    // Precision Policy Tests
    TEST_F( SoftmaxTests, Cuda_Float_PrecisionPolicy_Auto ) {
        TestPrecisionPolicy<DeviceType::Cuda, float>( CudaFloatData(), ComputePrecision::Policy::Auto );
    }

    TEST_F( SoftmaxTests, Cuda_Float_PrecisionPolicy_Performance ) {
        TestPrecisionPolicy<DeviceType::Cuda, float>( PerformancePolicyData(), ComputePrecision::Policy::Performance );
    }

    TEST_F( SoftmaxTests, Cuda_Float_PrecisionPolicy_Accuracy ) {
        TestPrecisionPolicy<DeviceType::Cuda, float>( AccuracyPolicyData(), ComputePrecision::Policy::Accuracy );
    }

    TEST_F( SoftmaxTests, Cuda_Float_PrecisionPolicy_Disabled ) {
        TestPrecisionPolicy<DeviceType::Cuda, float>( DisabledPolicyData(), ComputePrecision::Policy::Native );
    }

    // Context Construction Tests
    TEST_F( SoftmaxTests, Context_Cpu_Float_DeviceType ) {
        TestDeviceType<DeviceType::Cpu, float>( ContextCpuFloatData() );
    }

    TEST_F( SoftmaxTests, Context_Cpu_Float_Forward ) {
        TestForward<DeviceType::Cpu, float>( ContextCpuFloatData() );
    }

    // Edge Case Tests
    TEST_F( SoftmaxTests, Cpu_Float_EdgeCases ) {
        TestEdgeCases<DeviceType::Cpu, float>();
    }

    TEST_F( SoftmaxTests, Cuda_Float_EdgeCases ) {
        TestEdgeCases<DeviceType::Cuda, float>();
    }

    // Axis Tests
    TEST_F( SoftmaxTests, Cpu_Float_DifferentAxes ) {
        TestDifferentAxes<DeviceType::Cpu, float>();
    }

    TEST_F( SoftmaxTests, Cuda_Float_DifferentAxes ) {
        TestDifferentAxes<DeviceType::Cuda, float>();
    }

    // CPU-CUDA Equivalence Test
    TEST_F( SoftmaxTests, CpuCuda_Forward_Output_Equivalence ) {
        TestCpuCudaEquivalence<float>( CpuFloatData(), CudaFloatData() );
    }
}