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
    using MemoryResourceType = std::conditional_t<TDevice == Compute::DeviceType::Cuda,
        Compute::CudaMemoryResource,
        Compute::HostMemoryResource>;

    // Test data structure for Gelu tests
    template<DeviceType TDevice, typename TInput = float, typename TOutput = TInput>
    struct GeluTestData {
        std::vector<size_t> shape;
        std::shared_ptr<Gelu<TDevice, TInput, TOutput>> gelu_module;
        bool is_training;

        // Make the test data structure self-initializing
        static GeluTestData Create(
            const std::string& name,
            size_t batch_size,
            size_t sequence_length,
            size_t channels,
            bool is_training = false )
        {
            GeluTestData data;
            data.shape = { batch_size, sequence_length, channels };
            data.is_training = is_training;

            // Create a GeluConfig and configure it
            GeluConfig config;
            config.withName( name )
                .withDeviceName( TDevice == Compute::DeviceType::Cuda ? "CUDA:0" : "CPU" )
                .withPrecision( ComputePrecision::Policy::Auto )
                .withTraining( is_training );

            data.gelu_module = std::make_shared<Gelu<TDevice, TInput, TOutput>>( config );

            return data;
        }

        // Overload for creating with device context
        static GeluTestData CreateWithContext(
            const std::string& name,
            size_t batch_size,
            size_t sequence_length,
            size_t channels,
            std::shared_ptr<DeviceContext> context,
            bool is_training = false )
        {
            GeluTestData data;
            data.shape = { batch_size, sequence_length, channels };
            data.is_training = is_training;

            // Create a GeluConfig and configure it with context
            GeluConfig config;
            config.withName( name )
                .withContext( context )
                .withPrecision( ComputePrecision::Policy::Auto )
                .withTraining( is_training );

            data.gelu_module = std::make_shared<Gelu<TDevice, TInput, TOutput>>( config );

            return data;
        }
    };

    class GeluTests : public ::testing::Test {
    protected:
        void SetUp() override {
            // Initialize test parameters only
            batch_size_ = 64;
            cpu_batch_size_ = 4;
            sequence_length_ = 1024;
            channels_ = 768;
            // Modules will be created on demand
        }

        void TearDown() override {
            // Clean up resources explicitly
            cpu_float_data_.gelu_module.reset();
            context_cpu_float_data_.gelu_module.reset();
            training_cpu_float_data_.gelu_module.reset();
            cuda_float_data_.gelu_module.reset();
            training_cuda_float_data_.gelu_module.reset();
            cuda_half_data_.gelu_module.reset();
            training_cuda_half_data_.gelu_module.reset();
        }

        // Factory methods to lazily create test data as needed
        GeluTestData<Compute::DeviceType::Cpu, float>& CpuFloatData() {
            if ( !cpu_float_data_.gelu_module ) {
                cpu_float_data_ = GeluTestData<Compute::DeviceType::Cpu, float>::Create(
                    "cpu_gelu_float", cpu_batch_size_, sequence_length_, channels_ );
            }
            return cpu_float_data_;
        }

        GeluTestData<Compute::DeviceType::Cuda, float>& CudaFloatData() {
            if ( !cuda_float_data_.gelu_module ) {
                cuda_float_data_ = GeluTestData<Compute::DeviceType::Cuda, float>::Create(
                    "cuda_gelu_float", batch_size_, sequence_length_, channels_ );
            }
            return cuda_float_data_;
        }

        GeluTestData<Compute::DeviceType::Cpu, float>& TrainingCpuFloatData() {
            if ( !training_cpu_float_data_.gelu_module ) {
                training_cpu_float_data_ = GeluTestData<Compute::DeviceType::Cpu, float>::Create(
                    "cpu_gelu_float_training", cpu_batch_size_, sequence_length_, channels_, true );
            }
            return training_cpu_float_data_;
        }

        GeluTestData<Compute::DeviceType::Cuda, float>& TrainingCudaFloatData() {
            if ( !training_cuda_float_data_.gelu_module ) {
                training_cuda_float_data_ = GeluTestData<Compute::DeviceType::Cuda, float>::Create(
                    "cuda_gelu_float_training", batch_size_, sequence_length_, channels_, true );
            }
            return training_cuda_float_data_;
        }

        GeluTestData<Compute::DeviceType::Cpu, float>& ContextCpuFloatData() {
            if ( !context_cpu_float_data_.gelu_module ) {
                auto cpu_context = std::make_shared<DeviceContext>( "CPU" );
                context_cpu_float_data_ = GeluTestData<Compute::DeviceType::Cpu, float>::CreateWithContext(
                    "cpu_context_gelu_float", cpu_batch_size_, sequence_length_, channels_, cpu_context );
            }
            return context_cpu_float_data_;
        }

        GeluTestData<Compute::DeviceType::Cuda, half>& CudaHalfData() {
            if ( !cuda_half_data_.gelu_module ) {
                cuda_half_data_ = GeluTestData<Compute::DeviceType::Cuda, half>::Create(
                    "cuda_gelu_half", batch_size_, sequence_length_, channels_ );
            }
            return cuda_half_data_;
        }

        GeluTestData<Compute::DeviceType::Cuda, half>& TrainingCudaHalfData() {
            if ( !training_cuda_half_data_.gelu_module ) {
                training_cuda_half_data_ = GeluTestData<Compute::DeviceType::Cuda, half>::Create(
                    "cuda_gelu_half_training", batch_size_, sequence_length_, channels_, true );
            }
            return training_cuda_half_data_;
        }

        // Test parameters
        size_t batch_size_{ 0 };
        size_t cpu_batch_size_{ 0 };
        size_t sequence_length_{ 0 };
        size_t channels_{ 0 };

        // Test data objects - initialized on demand
        GeluTestData<Compute::DeviceType::Cpu, float> cpu_float_data_;
        GeluTestData<Compute::DeviceType::Cpu, float> context_cpu_float_data_;
        GeluTestData<Compute::DeviceType::Cpu, float> training_cpu_float_data_;

        GeluTestData<Compute::DeviceType::Cuda, float> cuda_float_data_;
        GeluTestData<Compute::DeviceType::Cuda, float> training_cuda_float_data_;

        GeluTestData<Compute::DeviceType::Cuda, half> cuda_half_data_;
        GeluTestData<Compute::DeviceType::Cuda, half> training_cuda_half_data_;

        // Mixed precision test data (float input to half output)
        // REVIEW: TInput != TOutput in the Gelu module doesn't make sense? 
        // TInput and TOutput should be the same type but I need think through this.
        // 
        // GeluTestData<Compute::DeviceType::Cuda, float, half> mixed_precision_data_;
    };

    // Common test function templates
    template<DeviceType TDevice, typename TInput, typename TOutput = TInput>
    void TestGetName( const GeluTestData<TDevice, TInput, TOutput>& data, const std::string& expected_name ) {
        EXPECT_EQ( data.gelu_module->getName(), expected_name );
    }

    template<DeviceType TDevice, typename TInput, typename TOutput = TInput>
    void TestParameterCount( const GeluTestData<TDevice, TInput, TOutput>& data, size_t expected_count ) {
        EXPECT_EQ( data.gelu_module->parameterCount(), expected_count );
    }

    template<DeviceType TDevice, typename TInput, typename TOutput = TInput>
    void TestForward( const GeluTestData<TDevice, TInput, TOutput>& data ) {
        using MR = MemoryResourceType<TDevice>;

        Tensor<TInput, MR> input( data.shape );
        Tensor<TOutput, MR> output( data.shape );

        // Fill with random values between -5.0 and 5.0 to test GELU activation range
        random<TInput, MR>( input, -5.0f, 5.0f );

        data.gelu_module->forward( input, output );
        EXPECT_EQ( output.size(), input.size() );
    }

    template<DeviceType TDevice, typename TInput, typename TOutput = TInput>
    void TestPrint( const GeluTestData<TDevice, TInput, TOutput>& data, const std::string& expected_substring ) {
        std::string output = data.gelu_module->toString();
        EXPECT_NE( output.find( expected_substring ), std::string::npos );
    }

    template<DeviceType TDevice, typename TInput, typename TOutput = TInput>
    void TestTrainingMode( const GeluTestData<TDevice, TInput, TOutput>& data, bool expected_mode ) {
        EXPECT_EQ( data.gelu_module->isTraining(), expected_mode );
    }

    template<DeviceType TDevice, typename TInput, typename TOutput = TInput>
    void TestDeviceType( const GeluTestData<TDevice, TInput, TOutput>& data ) {
        auto device_context = data.gelu_module->getDeviceContext();
        EXPECT_NE( device_context, nullptr );
        auto device = device_context->getDevice();
        EXPECT_NE( device, nullptr );
        EXPECT_EQ( device->getDeviceType(), TDevice );
    }

    // Function to test equivalence of CPU and CUDA outputs
    template<typename TInput, typename TOutput = TInput>
    void TestCpuCudaEquivalence(
        const GeluTestData<Compute::DeviceType::Cpu, TInput, TOutput>& cpu_data,
        const GeluTestData<Compute::DeviceType::Cuda, TInput, TOutput>& cuda_data ) {

        // Create a small test shape to make comparison faster
        std::vector<size_t> test_shape = { 2, 4, 8 }; // Small shape for quick verification

        // Create random input data
        Tensor<TInput, Compute::HostMemoryResource> host_input( test_shape );

        // Fill with predictable values between -2.0 and 2.0 to exercise the GELU function
        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input.data()[ i ] = static_cast<TInput>( -2.0 + 4.0 * (static_cast<float>( i ) / host_input.size()) );
        }

        // Create CPU output
        Tensor<TOutput, Compute::HostMemoryResource> cpu_output( test_shape );
        cpu_data.gelu_module->forward( host_input, cpu_output );

        // Create device input by copying host data
        Tensor<TInput, Compute::CudaMemoryResource> device_input( test_shape );
        device_input.copyFrom( host_input );

        // Create device output
        Tensor<TOutput, Compute::CudaMemoryResource> cuda_output( test_shape );
        cuda_data.gelu_module->forward( device_input, cuda_output );

        // Copy CUDA output back to host for comparison
        Tensor<TOutput, Compute::HostMemoryResource> cuda_output_host( test_shape );
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

            // Create config for minimal module
            GeluConfig minimal_config;
            minimal_config.withName( "minimal_gelu" )
                .withDeviceName( TDevice == Compute::DeviceType::Cuda ? "CUDA:0" : "CPU" )
                .withPrecision( ComputePrecision::Policy::Auto );

            auto minimal_module = std::make_shared<Gelu<TDevice, TInput, TOutput>>( minimal_config );

            Tensor<TInput, MR> minimal_input( minimal_shape );
            Tensor<TOutput, MR> minimal_output( minimal_shape );

            EXPECT_NO_THROW( minimal_module->forward( minimal_input, minimal_output ) );
            EXPECT_EQ( minimal_output.size(), 8 );

            // Test with larger dimensions
            std::vector<size_t> large_shape = { 2, 2, 1024 };

            // Create config for large module
            GeluConfig large_config;
            large_config.withName( "large_gelu" )
                .withDeviceName( TDevice == Compute::DeviceType::Cuda ? "CUDA:0" : "CPU" )
                .withPrecision( ComputePrecision::Policy::Auto );

            auto large_module = std::make_shared<Gelu<TDevice, TInput, TOutput>>( large_config );

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

    // CPU Tests with float precision
    TEST_F( GeluTests, Cpu_Float_TestName ) {
        TestGetName<Compute::DeviceType::Cpu, float>( CpuFloatData(), "cpu_gelu_float" );
    }

    TEST_F( GeluTests, Cpu_Float_ParameterCount ) {
        TestParameterCount<Compute::DeviceType::Cpu, float>( CpuFloatData(), 0 );
    }

    TEST_F( GeluTests, Cpu_Float_TestForward ) {
        TestForward<Compute::DeviceType::Cpu, float>( CpuFloatData() );
    }

    TEST_F( GeluTests, Cpu_Float_TestPrint ) {
        TestPrint<Compute::DeviceType::Cpu, float>( CpuFloatData(), "Gelu: cpu_gelu_float" );
    }

    TEST_F( GeluTests, Cpu_Float_TrainingMode ) {
        TestTrainingMode<Compute::DeviceType::Cpu, float>( CpuFloatData(), false );
    }

    TEST_F( GeluTests, Cpu_Float_DeviceType ) {
        TestDeviceType<Compute::DeviceType::Cpu, float>( CpuFloatData() );
    }

    // CPU Training Mode Tests
    TEST_F( GeluTests, Cpu_Training_Float_TrainingMode ) {
        TestTrainingMode<Compute::DeviceType::Cpu, float>( TrainingCpuFloatData(), true );
    }

    // CUDA Tests with float precision
    TEST_F( GeluTests, Cuda_Float_TestName ) {
        TestGetName<Compute::DeviceType::Cuda, float>( CudaFloatData(), "cuda_gelu_float" );
    }

    TEST_F( GeluTests, Cuda_Float_ParameterCount ) {
        TestParameterCount<Compute::DeviceType::Cuda, float>( CudaFloatData(), 0 );
    }

    TEST_F( GeluTests, Cuda_Float_TestForward ) {
        TestForward<Compute::DeviceType::Cuda, float>( CudaFloatData() );
    }

    TEST_F( GeluTests, Cuda_Float_TestPrint ) {
        TestPrint<Compute::DeviceType::Cuda, float>( CudaFloatData(), "Gelu: cuda_gelu_float" );
    }

    TEST_F( GeluTests, Cuda_Float_TrainingMode ) {
        TestTrainingMode<Compute::DeviceType::Cuda, float>( CudaFloatData(), false );
    }

    TEST_F( GeluTests, Cuda_Float_DeviceType ) {
        TestDeviceType<Compute::DeviceType::Cuda, float>( CudaFloatData() );
    }

    // CUDA Training Mode Tests
    TEST_F( GeluTests, Cuda_Training_Float_TrainingMode ) {
        TestTrainingMode<Compute::DeviceType::Cuda, float>( TrainingCudaFloatData(), true );
    }

    // CUDA Tests with half precision
    TEST_F( GeluTests, Cuda_Half_TestName ) {
        TestGetName<Compute::DeviceType::Cuda, half>( CudaHalfData(), "cuda_gelu_half" );
    }

    TEST_F( GeluTests, Cuda_Half_ParameterCount ) {
        TestParameterCount<Compute::DeviceType::Cuda, half>( CudaHalfData(), 0 );
    }

    TEST_F( GeluTests, Cuda_Half_TestForward ) {
        TestForward<Compute::DeviceType::Cuda, half>( CudaHalfData() );
    }

    TEST_F( GeluTests, Cuda_Half_TestPrint ) {
        TestPrint<Compute::DeviceType::Cuda, half>( CudaHalfData(), "Gelu: cuda_gelu_half" );
    }

    TEST_F( GeluTests, Cuda_Half_TrainingMode ) {
        TestTrainingMode<Compute::DeviceType::Cuda, half>( CudaHalfData(), false );
    }

    // Mixed Precision Tests (new in updated module)
    TEST_F( GeluTests, Cuda_MixedPrecision_TestForward ) {
        // REVIEW: TestForward<Compute::DeviceType::Cuda, float, half>( MixedPrecisionData() );
    }

    TEST_F( GeluTests, Cuda_MixedPrecision_TestName ) {
        // REVIEW: TestGetName<Compute::DeviceType::Cuda, float, half>( MixedPrecisionData(), "cuda_gelu_mixed" );
    }

    // Context Construction Tests
    TEST_F( GeluTests, Context_Cpu_Float_DeviceType ) {
        TestDeviceType<Compute::DeviceType::Cpu, float>( ContextCpuFloatData() );
    }

    TEST_F( GeluTests, Context_Cpu_Float_Forward ) {
        TestForward<Compute::DeviceType::Cpu, float>( ContextCpuFloatData() );
    }

    // Edge Case Tests
    TEST_F( GeluTests, Cpu_Float_EdgeCases ) {
        TestEdgeCases<Compute::DeviceType::Cpu, float>();
    }

    TEST_F( GeluTests, Cuda_Float_EdgeCases ) {
        TestEdgeCases<Compute::DeviceType::Cuda, float>();
    }

    // CPU-CUDA Equivalence Test
    TEST_F( GeluTests, CpuCuda_Forward_Output_Equivalence ) {
        TestCpuCudaEquivalence<float>( CpuFloatData(), CudaFloatData() );
    }
}