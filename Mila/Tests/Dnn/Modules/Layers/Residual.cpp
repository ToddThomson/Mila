#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <cuda_fp16.h>  // For half type

import Mila;

namespace Modules::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    // Memory resource selector based on device type
    template<DeviceType TDeviceType>
    using MemoryResourceType = std::conditional_t<TDeviceType == DeviceType::Cuda,
        Compute::CudaDeviceMemoryResource,
        Compute::HostMemoryResource>;

    // Test data structure for Residual tests
    template<DeviceType TDeviceType, typename TInput = float, typename TOutput = TInput>
    struct ResidualTestData {
        std::vector<size_t> shape;
        std::shared_ptr<Residual<TDeviceType, TInput, TOutput>> residual_module;
        bool is_training;

        // Make the test data structure self-initializing
        static ResidualTestData Create(
            const std::string& name,
            size_t batch_size,
            size_t sequence_length,
            size_t channels,
            bool is_training = false )
        {
            ResidualTestData data;
            data.shape = { batch_size, sequence_length, channels };
            data.is_training = is_training;

            std::string device_str = TDeviceType == DeviceType::Cuda ? "CUDA:0" : "CPU";
            data.residual_module = std::make_shared<Residual<TDeviceType, TInput, TOutput>>(
                name, device_str, is_training );

            return data;
        }

        // Overload for creating with device context
        static ResidualTestData CreateWithContext(
            const std::string& name,
            size_t batch_size,
            size_t sequence_length,
            size_t channels,
            std::shared_ptr<DeviceContext> context,
            bool is_training = false )
        {
            ResidualTestData data;
            data.shape = { batch_size, sequence_length, channels };
            data.is_training = is_training;

            data.residual_module = std::make_shared<Residual<TDeviceType, TInput, TOutput>>(
                name, context, is_training );

            return data;
        }
    };

    class ResidualTests : public ::testing::Test {
    protected:
        void SetUp() override {
            // Initialize test parameters only
            batch_size_ = 128;
            cpu_batch_size_ = 4;
            sequence_length_ = 1024;
            channels_ = 768;
            // Modules will be created on demand
        }

        // Factory methods to lazily create test data as needed
        ResidualTestData<DeviceType::Cpu>& CpuFloatData() {
            if ( !cpu_float_data_.residual_module ) {
                cpu_float_data_ = ResidualTestData<DeviceType::Cpu>::Create(
                    "cpu_residual_float", cpu_batch_size_, sequence_length_, channels_ );
            }
            return cpu_float_data_;
        }

        ResidualTestData<DeviceType::Cuda>& CudaFloatData() {
            if ( !cuda_float_data_.residual_module ) {
                cuda_float_data_ = ResidualTestData<DeviceType::Cuda>::Create(
                    "cuda_residual_float", batch_size_, sequence_length_, channels_ );
            }
            return cuda_float_data_;
        }

        ResidualTestData<DeviceType::Cpu>& TrainingCpuFloatData() {
            if ( !training_cpu_float_data_.residual_module ) {
                training_cpu_float_data_ = ResidualTestData<DeviceType::Cpu>::Create(
                    "cpu_residual_float_training", cpu_batch_size_, sequence_length_, channels_, true );
            }
            return training_cpu_float_data_;
        }

        ResidualTestData<DeviceType::Cuda>& TrainingCudaFloatData() {
            if ( !training_cuda_float_data_.residual_module ) {
                training_cuda_float_data_ = ResidualTestData<DeviceType::Cuda>::Create(
                    "cuda_residual_float_training", batch_size_, sequence_length_, channels_, true );
            }
            return training_cuda_float_data_;
        }

        ResidualTestData<DeviceType::Cpu>& ContextCpuFloatData() {
            if ( !context_cpu_float_data_.residual_module ) {
                auto cpu_context = std::make_shared<DeviceContext>( "CPU" );
                context_cpu_float_data_ = ResidualTestData<DeviceType::Cpu>::CreateWithContext(
                    "cpu_context_residual_float", cpu_batch_size_, sequence_length_, channels_, cpu_context );
            }
            return context_cpu_float_data_;
        }

        ResidualTestData<DeviceType::Cuda, float, half>& CudaHalfData() {
            if ( !cuda_half_data_.residual_module ) {
                cuda_half_data_ = ResidualTestData<DeviceType::Cuda, float, half>::Create(
                    "cuda_residual_half", batch_size_, sequence_length_, channels_ );
            }
            return cuda_half_data_;
        }

        ResidualTestData<DeviceType::Cuda, float, half>& TrainingCudaHalfData() {
            if ( !training_cuda_half_data_.residual_module ) {
                training_cuda_half_data_ = ResidualTestData<DeviceType::Cuda, float, half>::Create(
                    "cuda_residual_half_training", batch_size_, sequence_length_, channels_, true );
            }
            return training_cuda_half_data_;
        }

        // Test for mixed precision (input float, output half)
        ResidualTestData<DeviceType::Cuda, float, half>& MixedPrecisionData() {
            if ( !mixed_precision_data_.residual_module ) {
                mixed_precision_data_ = ResidualTestData<DeviceType::Cuda, float, half>::Create(
                    "cuda_residual_mixed", batch_size_, sequence_length_, channels_ );
            }
            return mixed_precision_data_;
        }

        // Test parameters
        size_t batch_size_{ 0 };
        size_t cpu_batch_size_{ 0 };
        size_t sequence_length_{ 0 };
        size_t channels_{ 0 };

        // Test data objects - initialized on demand
        ResidualTestData<DeviceType::Cpu> cpu_float_data_;
        ResidualTestData<DeviceType::Cpu> context_cpu_float_data_;
        ResidualTestData<DeviceType::Cpu> training_cpu_float_data_;

        ResidualTestData<DeviceType::Cuda> cuda_float_data_;
        ResidualTestData<DeviceType::Cuda> training_cuda_float_data_;

        ResidualTestData<DeviceType::Cuda, float, half> cuda_half_data_;
        ResidualTestData<DeviceType::Cuda, float, half> training_cuda_half_data_;

        // Mixed precision test data (float input to half output)
        ResidualTestData<DeviceType::Cuda, float, half> mixed_precision_data_;
    };

    // Common test function templates
    template<DeviceType TDeviceType, typename TInput = float, typename TOutput = TInput>
    void TestGetName( const ResidualTestData<TDeviceType, TInput, TOutput>& data, const std::string& expected_name ) {
        EXPECT_EQ( data.residual_module->getName(), expected_name );
    }

    template<DeviceType TDeviceType, typename TInput = float, typename TOutput = TInput>
    void TestParameterCount( const ResidualTestData<TDeviceType, TInput, TOutput>& data, size_t expected_count ) {
        EXPECT_EQ( data.residual_module->parameterCount(), expected_count );
    }

    template<DeviceType TDeviceType, typename TInput = float, typename TOutput = TInput>
    void TestForward( const ResidualTestData<TDeviceType, TInput, TOutput>& data ) {
        using MR = MemoryResourceType<TDeviceType>;

        Tensor<TInput, MR> input_a( data.shape, 4.0f );
        Tensor<TInput, MR> input_b( data.shape, 2.0f );
        Tensor<TOutput, MR> output( data.shape );

        data.residual_module->forward( input_a, input_b, output );

        EXPECT_EQ( output.size(), input_a.size() );
    }

    template<DeviceType TDeviceType, typename TInput = float, typename TOutput = TInput>
    void TestPrint( const ResidualTestData<TDeviceType, TInput, TOutput>& data, const std::string& expected_substring ) {
        std::string output = data.residual_module->toString();
        EXPECT_NE( output.find( expected_substring ), std::string::npos );
    }

    template<DeviceType TDeviceType, typename TInput = float, typename TOutput = TInput>
    void TestTrainingMode( const ResidualTestData<TDeviceType, TInput, TOutput>& data, bool expected_mode ) {
        EXPECT_EQ( data.residual_module->isTraining(), expected_mode );
    }

    template<DeviceType TDeviceType, typename TInput = float, typename TOutput = TInput>
    void TestDeviceType( const ResidualTestData<TDeviceType, TInput, TOutput>& data ) {
        auto device_context = data.residual_module->getDeviceContext();
        EXPECT_NE( device_context, nullptr );
        auto device = device_context->getDevice();
        EXPECT_NE( device, nullptr );
        EXPECT_EQ( device->getDeviceType(), TDeviceType );
    }

    // Function to test equivalence of CPU and CUDA outputs
    template<typename TInput, typename TOutput = TInput>
    void TestCpuCudaEquivalence(
        const ResidualTestData<DeviceType::Cpu, TInput, TOutput>& cpu_data,
        const ResidualTestData<DeviceType::Cuda, TInput, TOutput>& cuda_data ) {

        // Create a small test shape to make comparison faster
        std::vector<size_t> test_shape = { 2, 4, 8 }; // Small shape for quick verification

        // Create random input data
        Tensor<TInput, HostMemoryResource> host_input_a( test_shape );
        Tensor<TInput, HostMemoryResource> host_input_b( test_shape );

        // Fill with predictable values between -2.0 and 2.0 to exercise the residual function
        for ( size_t i = 0; i < host_input_a.size(); ++i ) {
            host_input_a.data()[ i ] = static_cast<TInput>( -2.0 + 4.0 * (static_cast<float>( i ) / host_input_a.size()) );
            host_input_b.data()[ i ] = static_cast<TInput>( 2.0 - 4.0 * (static_cast<float>( i ) / host_input_b.size()) );
        }

        // Create CPU output
        Tensor<TOutput, HostMemoryResource> cpu_output( test_shape );
        cpu_data.residual_module->forward( host_input_a, host_input_b, cpu_output );

        // Create device input by copying host data
        Tensor<TInput, CudaDeviceMemoryResource> device_input_a( test_shape );
        Tensor<TInput, CudaDeviceMemoryResource> device_input_b( test_shape );
        device_input_a.copyFrom( host_input_a );
        device_input_b.copyFrom( host_input_b );

        // Create device output
        Tensor<TOutput, CudaDeviceMemoryResource> cuda_output( test_shape );
        cuda_data.residual_module->forward( device_input_a, device_input_b, cuda_output );

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
    template<DeviceType TDeviceType, typename TInput = float, typename TOutput = TInput>
    void TestEdgeCases() {
        using MR = MemoryResourceType<TDeviceType>;

        try {
            // Test with minimal sizes
            std::vector<size_t> minimal_shape = { 1, 1, 8 };

            auto minimal_module = std::make_shared<Residual<TDeviceType, TInput, TOutput>>(
                "minimal_residual", TDeviceType == DeviceType::Cuda ? "CUDA:0" : "CPU" );

            Tensor<TInput, MR> minimal_input_a( minimal_shape, 1.0f );
            Tensor<TInput, MR> minimal_input_b( minimal_shape, 2.0f );
            Tensor<TOutput, MR> minimal_output( minimal_shape );

            EXPECT_NO_THROW( minimal_module->forward( minimal_input_a, minimal_input_b, minimal_output ) );
            EXPECT_EQ( minimal_output.size(), 8 );

            // Test with larger dimensions
            std::vector<size_t> large_shape = { 2, 2, 1024 };

            auto large_module = std::make_shared<Residual<TDeviceType, TInput, TOutput>>(
                "large_residual", TDeviceType == DeviceType::Cuda ? "CUDA:0" : "CPU" );

            Tensor<TInput, MR> large_input_a( large_shape, 1.0f );
            Tensor<TInput, MR> large_input_b( large_shape, 2.0f );
            Tensor<TOutput, MR> large_output( large_shape );

            EXPECT_NO_THROW( large_module->forward( large_input_a, large_input_b, large_output ) );
            EXPECT_EQ( large_output.size(), 4096 );
        }
        catch ( const std::exception& e ) {
            std::cerr << "Exception during edge case test: " << e.what() << std::endl;
            throw;
        }
    }

    // CPU Tests with float precision
    TEST_F( ResidualTests, Cpu_Float_TestName ) {
        TestGetName<DeviceType::Cpu, float>( CpuFloatData(), "cpu_residual_float" );
    }

    TEST_F( ResidualTests, Cpu_Float_ParameterCount ) {
        TestParameterCount<DeviceType::Cpu, float>( CpuFloatData(), 0 );
    }

    TEST_F( ResidualTests, Cpu_Float_TestForward ) {
        TestForward<DeviceType::Cpu, float>( CpuFloatData() );
    }

    TEST_F( ResidualTests, Cpu_Float_TestPrint ) {
        TestPrint<DeviceType::Cpu, float>( CpuFloatData(), "Residual: cpu_residual_float" );
    }

    TEST_F( ResidualTests, Cpu_Float_TrainingMode ) {
        TestTrainingMode<DeviceType::Cpu, float>( CpuFloatData(), false );
    }

    TEST_F( ResidualTests, Cpu_Float_DeviceType ) {
        TestDeviceType<DeviceType::Cpu, float>( CpuFloatData() );
    }

    // CPU Training Mode Tests
    TEST_F( ResidualTests, Cpu_Training_Float_TrainingMode ) {
        TestTrainingMode<DeviceType::Cpu, float>( TrainingCpuFloatData(), true );
    }

    // CUDA Tests with float precision
    TEST_F( ResidualTests, Cuda_Float_TestName ) {
        TestGetName<DeviceType::Cuda, float>( CudaFloatData(), "cuda_residual_float" );
    }

    TEST_F( ResidualTests, Cuda_Float_ParameterCount ) {
        TestParameterCount<DeviceType::Cuda, float>( CudaFloatData(), 0 );
    }

    TEST_F( ResidualTests, Cuda_Float_TestForward ) {
        TestForward<DeviceType::Cuda, float>( CudaFloatData() );
    }

    TEST_F( ResidualTests, Cuda_Float_TestPrint ) {
        TestPrint<DeviceType::Cuda, float>( CudaFloatData(), "Residual: cuda_residual_float" );
    }

    TEST_F( ResidualTests, Cuda_Float_TrainingMode ) {
        TestTrainingMode<DeviceType::Cuda, float>( CudaFloatData(), false );
    }

    TEST_F( ResidualTests, Cuda_Float_DeviceType ) {
        TestDeviceType<DeviceType::Cuda, float>( CudaFloatData() );
    }

    // CUDA Training Mode Tests
    TEST_F( ResidualTests, Cuda_Training_Float_TrainingMode ) {
        TestTrainingMode<DeviceType::Cuda, float>( TrainingCudaFloatData(), true );
    }

    // CUDA Tests with half precision
    TEST_F( ResidualTests, Cuda_Half_TestName ) {
        TestGetName<DeviceType::Cuda, float, half>( CudaHalfData(), "cuda_residual_half" );
    }

    TEST_F( ResidualTests, Cuda_Half_ParameterCount ) {
        TestParameterCount<DeviceType::Cuda, float, half>( CudaHalfData(), 0 );
    }

    TEST_F( ResidualTests, Cuda_Half_TestForward ) {
        TestForward<DeviceType::Cuda, float, half>( CudaHalfData() );
    }

    TEST_F( ResidualTests, Cuda_Half_TestPrint ) {
        TestPrint<DeviceType::Cuda, float, half>( CudaHalfData(), "Residual: cuda_residual_half" );
    }

    TEST_F( ResidualTests, Cuda_Half_TrainingMode ) {
        TestTrainingMode<DeviceType::Cuda, float, half>( CudaHalfData(), false );
    }

    // Mixed Precision Tests (new in updated module)
    TEST_F( ResidualTests, Cuda_MixedPrecision_TestForward ) {
        TestForward<DeviceType::Cuda, float, half>( MixedPrecisionData() );
    }

    TEST_F( ResidualTests, Cuda_MixedPrecision_TestName ) {
        TestGetName<DeviceType::Cuda, float, half>( MixedPrecisionData(), "cuda_residual_mixed" );
    }

    // Context Construction Tests
    TEST_F( ResidualTests, Context_Cpu_Float_DeviceType ) {
        TestDeviceType<DeviceType::Cpu, float>( ContextCpuFloatData() );
    }

    TEST_F( ResidualTests, Context_Cpu_Float_Forward ) {
        TestForward<DeviceType::Cpu, float>( ContextCpuFloatData() );
    }

    // Edge Case Tests
    TEST_F( ResidualTests, Cpu_Float_EdgeCases ) {
        TestEdgeCases<DeviceType::Cpu, float>();
    }

    TEST_F( ResidualTests, Cuda_Float_EdgeCases ) {
        TestEdgeCases<DeviceType::Cuda, float>();
    }

    // CPU-CUDA Equivalence Test
    TEST_F( ResidualTests, CpuCuda_Forward_Output_Equivalence ) {
        TestCpuCudaEquivalence<float>( CpuFloatData(), CudaFloatData() );
    }
}