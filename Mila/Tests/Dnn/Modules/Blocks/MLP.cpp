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
    template<DeviceType TDevice, typename TPrecision>
    using MemoryResourceType = std::conditional_t<TDevice == Compute::DeviceType::Cuda,
        Compute::CudaMemoryResource,
        Compute::HostMemoryResource>;

    // Test data structure for MLP tests
    template<DeviceType TDevice, typename TInput = float, typename TOutput = TInput, typename TPrecision = TOutput>
    struct MLPTestData {
        std::vector<size_t> input_shape;
        size_t output_channels;
        std::shared_ptr<MLP<TDevice, TInput, TOutput, TPrecision>> mlp_module;
        bool is_training;
        bool has_bias;

        // Make the test data structure self-initializing
        static MLPTestData Create(
            const std::string& name,
            size_t batch_size,
            size_t sequence_length,
            size_t input_channels,
            size_t output_channels,
            bool has_bias = true,
            bool is_training = false )
        {
            MLPTestData data;
            data.input_shape = { batch_size, sequence_length, input_channels };
            data.output_channels = output_channels;
            data.is_training = is_training;
            data.has_bias = has_bias;

            std::string device_str = TDevice == Compute::DeviceType::Cuda ? "CUDA:0" : "CPU";
            data.mlp_module = std::make_shared<MLP<TDevice, TInput, TOutput, TPrecision>>(
                name, device_str, data.input_shape, output_channels, has_bias, is_training );

            return data;
        }

        // Overload for creating with device context
        static MLPTestData CreateWithContext(
            const std::string& name,
            size_t batch_size,
            size_t sequence_length,
            size_t input_channels,
            size_t output_channels,
            std::shared_ptr<DeviceContext> context,
            bool has_bias = true,
            bool is_training = false )
        {
            MLPTestData data;
            data.input_shape = { batch_size, sequence_length, input_channels };
            data.output_channels = output_channels;
            data.is_training = is_training;
            data.has_bias = has_bias;

            data.mlp_module = std::make_shared<MLP<TDevice, TInput, TOutput, TPrecision>>(
                name, context, data.input_shape, output_channels, has_bias, is_training );

            return data;
        }
    };

    class MLPTests : public ::testing::Test {
    protected:
        void SetUp() override {
            // CUDA-specific parameters - larger sizes for parallel processing
            cuda_batch_size_ = 16;         // Reduced from 64
            cuda_sequence_length_ = 64;    // Reduced from 512

            // CPU-specific parameters - smaller sizes to prevent timeouts
            cpu_batch_size_ = 2;
            cpu_sequence_length_ = 16;

            // Common parameters for both
            input_channels_ = 768;
            output_channels_ = 3072; // Common MLP expansion factor: 4x

            // Set Google Test timeout to 60 seconds
            //::testing::GTEST_FLAG_SET( timeout, 60000 );  // 60 seconds in milliseconds
        }

        void TearDown() override {
            // Explicitly reset modules to release resources earlier
            cpu_float_data_.mlp_module.reset();
            context_cpu_float_data_.mlp_module.reset();
            training_cpu_float_data_.mlp_module.reset();
            no_bias_cpu_float_data_.mlp_module.reset();

            cuda_float_data_.mlp_module.reset();
            training_cuda_float_data_.mlp_module.reset();
            no_bias_cuda_float_data_.mlp_module.reset();

            cuda_half_data_.mlp_module.reset();
            training_cuda_half_data_.mlp_module.reset();

            mixed_precision_data_.mlp_module.reset();
        }

        // Factory methods to lazily create test data as needed
        MLPTestData<Compute::DeviceType::Cpu, float>& CpuFloatData() {
            if ( !cpu_float_data_.mlp_module ) {
                cpu_float_data_ = MLPTestData<Compute::DeviceType::Cpu, float>::Create(
                    "cpu_mlp_float", cpu_batch_size_, cpu_sequence_length_, input_channels_, output_channels_ );
            }
            return cpu_float_data_;
        }

        MLPTestData<Compute::DeviceType::Cuda, float>& CudaFloatData() {
            if ( !cuda_float_data_.mlp_module ) {
                cuda_float_data_ = MLPTestData<Compute::DeviceType::Cuda, float>::Create(
                    "cuda_mlp_float", cuda_batch_size_, cuda_sequence_length_, input_channels_, output_channels_ );
            }
            return cuda_float_data_;
        }

        MLPTestData<Compute::DeviceType::Cpu, float>& TrainingCpuFloatData() {
            if ( !training_cpu_float_data_.mlp_module ) {
                training_cpu_float_data_ = MLPTestData<Compute::DeviceType::Cpu, float>::Create(
                    "cpu_mlp_float_training", cpu_batch_size_, cpu_sequence_length_, input_channels_, output_channels_, true, true );
            }
            return training_cpu_float_data_;
        }

        MLPTestData<Compute::DeviceType::Cuda, float>& TrainingCudaFloatData() {
            if ( !training_cuda_float_data_.mlp_module ) {
                training_cuda_float_data_ = MLPTestData<Compute::DeviceType::Cuda, float>::Create(
                    "cuda_mlp_float_training", cuda_batch_size_, cuda_sequence_length_, input_channels_, output_channels_, true, true );
            }
            return training_cuda_float_data_;
        }

        MLPTestData<Compute::DeviceType::Cpu, float>& NoBiasCpuFloatData() {
            if ( !no_bias_cpu_float_data_.mlp_module ) {
                no_bias_cpu_float_data_ = MLPTestData<Compute::DeviceType::Cpu, float>::Create(
                    "cpu_mlp_float_nobias", cpu_batch_size_, cpu_sequence_length_, input_channels_, output_channels_, false );
            }
            return no_bias_cpu_float_data_;
        }

        MLPTestData<Compute::DeviceType::Cuda, float>& NoBiasCudaFloatData() {
            if ( !no_bias_cuda_float_data_.mlp_module ) {
                no_bias_cuda_float_data_ = MLPTestData<Compute::DeviceType::Cuda, float>::Create(
                    "cuda_mlp_float_nobias", cuda_batch_size_, cuda_sequence_length_, input_channels_, output_channels_, false );
            }
            return no_bias_cuda_float_data_;
        }

        MLPTestData<Compute::DeviceType::Cpu, float>& ContextCpuFloatData() {
            if ( !context_cpu_float_data_.mlp_module ) {
                auto cpu_context = std::make_shared<DeviceContext>( "CPU" );
                context_cpu_float_data_ = MLPTestData<Compute::DeviceType::Cpu, float>::CreateWithContext(
                    "cpu_context_mlp_float", cpu_batch_size_, cpu_sequence_length_, input_channels_, output_channels_, cpu_context );
            }
            return context_cpu_float_data_;
        }

        MLPTestData<Compute::DeviceType::Cuda, half>& CudaHalfData() {
            if ( !cuda_half_data_.mlp_module ) {
                cuda_half_data_ = MLPTestData<Compute::DeviceType::Cuda, half>::Create(
                    "cuda_mlp_half", cuda_batch_size_, cuda_sequence_length_, input_channels_, output_channels_ );
            }
            return cuda_half_data_;
        }

        MLPTestData<Compute::DeviceType::Cuda, half>& TrainingCudaHalfData() {
            if ( !training_cuda_half_data_.mlp_module ) {
                training_cuda_half_data_ = MLPTestData<Compute::DeviceType::Cuda, half>::Create(
                    "cuda_mlp_half_training", cuda_batch_size_, cuda_sequence_length_, input_channels_, output_channels_, true, true );
            }
            return training_cuda_half_data_;
        }

        // Test for mixed precision (float input, half output on CUDA)
        MLPTestData<Compute::DeviceType::Cuda, float, half>& MixedPrecisionData() {
            if ( !mixed_precision_data_.mlp_module ) {
                mixed_precision_data_ = MLPTestData<Compute::DeviceType::Cuda, float, half>::Create(
                    "cuda_mlp_mixed", cuda_batch_size_, cuda_sequence_length_, input_channels_, output_channels_ );
            }
            return mixed_precision_data_;
        }

        // Test parameters - device-specific sizes
        size_t cuda_batch_size_{ 0 };
        size_t cpu_batch_size_{ 0 };
        size_t cuda_sequence_length_{ 0 };
        size_t cpu_sequence_length_{ 0 };

        // Common parameters
        size_t input_channels_{ 0 };
        size_t output_channels_{ 0 };

        // Test data objects - initialized on demand
        MLPTestData<Compute::DeviceType::Cpu, float> cpu_float_data_;
        MLPTestData<Compute::DeviceType::Cpu, float> context_cpu_float_data_;
        MLPTestData<Compute::DeviceType::Cpu, float> training_cpu_float_data_;
        MLPTestData<Compute::DeviceType::Cpu, float> no_bias_cpu_float_data_;

        MLPTestData<Compute::DeviceType::Cuda, float> cuda_float_data_;
        MLPTestData<Compute::DeviceType::Cuda, float> training_cuda_float_data_;
        MLPTestData<Compute::DeviceType::Cuda, float> no_bias_cuda_float_data_;

        MLPTestData<Compute::DeviceType::Cuda, half> cuda_half_data_;
        MLPTestData<Compute::DeviceType::Cuda, half> training_cuda_half_data_;

        // Mixed precision test data
        MLPTestData<Compute::DeviceType::Cuda, float, half> mixed_precision_data_;
    };

    // Common test function templates
    template<DeviceType TDevice, typename TInput = float, typename TOutput = TInput, typename TPrecision = TOutput>
    void TestGetName( const MLPTestData<TDevice, TInput, TOutput, TPrecision>& data, const std::string& expected_name ) {
        EXPECT_EQ( data.mlp_module->getName(), expected_name );
    }

    template<DeviceType TDevice, typename TInput = float, typename TOutput = TInput, typename TPrecision = TOutput>
    void TestParameterCount( const MLPTestData<TDevice, TInput, TOutput, TPrecision>& data ) {
        // For MLP with bias: fc1 and fc_proj both have weights and biases
        // For each linear layer: parameters = input_dim * output_dim + (has_bias ? output_dim : 0)
        size_t input_channels = data.input_shape.back();
        size_t expected_fc1_params = input_channels * data.output_channels;
        size_t expected_fc_proj_params = data.output_channels * input_channels;

        if ( data.has_bias ) {
            expected_fc1_params += data.output_channels;
            expected_fc_proj_params += input_channels;
        }

        size_t expected_total_params = expected_fc1_params + expected_fc_proj_params;

        EXPECT_EQ( data.mlp_module->parameterCount(), expected_total_params );
    }

    template<DeviceType TDevice, typename TInput = float, typename TOutput = TInput, typename TPrecision = TOutput>
    void TestForward( const MLPTestData<TDevice, TInput, TOutput, TPrecision>& data ) {
        using MR = MemoryResourceType<TDevice, TPrecision>;

        Tensor<TInput, MR> input( data.input_shape );
        Tensor<TOutput, MR> output( data.input_shape ); // MLP preserves spatial dimensions but may change channel dim

        // Fill with random values
        random<TInput, MR>( input, -1.0f, 1.0f );

        data.mlp_module->forward( input, output );

        // Output should have same shape as input for MLP
        EXPECT_EQ( output.size(), input.size() );
        EXPECT_EQ( output.shape(), input.shape() );
    }

    template<DeviceType TDevice, typename TInput = float, typename TOutput = TInput, typename TPrecision = TOutput>
    void TestPrint( const MLPTestData<TDevice, TInput, TOutput, TPrecision>& data, const std::string& expected_substring ) {
        std::string output = data.mlp_module->toString();
        EXPECT_NE( output.find( expected_substring ), std::string::npos );
    }

    template<DeviceType TDevice, typename TInput = float, typename TOutput = TInput, typename TPrecision = TOutput>
    void TestTrainingMode( const MLPTestData<TDevice, TInput, TOutput, TPrecision>& data, bool expected_mode ) {
        EXPECT_EQ( data.mlp_module->isTraining(), expected_mode );
    }

    template<DeviceType TDevice, typename TInput = float, typename TOutput = TInput, typename TPrecision = TOutput>
    void TestDeviceType( const MLPTestData<TDevice, TInput, TOutput, TPrecision>& data ) {
        auto device_context = data.mlp_module->getDeviceContext();
        EXPECT_NE( device_context, nullptr );
        auto device = device_context->getDevice();
        EXPECT_NE( device, nullptr );
        EXPECT_EQ( device->getDeviceType(), TDevice );
    }

    // Test for submodule structure
    template<DeviceType TDevice, typename TInput = float, typename TOutput = TInput, typename TPrecision = TOutput>
    void TestSubModules( const MLPTestData<TDevice, TInput, TOutput, TPrecision>& data ) {
        auto modules = data.mlp_module->getNamedModules();

        // Verify we have all 3 expected submodules: fc_1, gelu, fc_proj
        EXPECT_EQ( modules.size(), 3 );
        EXPECT_NE( modules.find( "fc_1" ), modules.end() );
        EXPECT_NE( modules.find( "gelu" ), modules.end() );
        EXPECT_NE( modules.find( "fc_proj" ), modules.end() );
    }

    // Test save/load functionality
    template<DeviceType TDevice, typename TInput = float, typename TOutput = TInput, typename TPrecision = TOutput>
    void TestSaveLoad( const MLPTestData<TDevice, TInput, TOutput, TPrecision>& data ) {
        // This is a minimal test for the save/load methods
        // In a real test, you would need to set up a zip archive and verify serialization

        // Just verify methods exist and don't crash when called
        mz_zip_archive zip = {};
        EXPECT_NO_THROW( data.mlp_module->save( zip ) );
        EXPECT_NO_THROW( data.mlp_module->load( zip ) );
    }

    // Function to test equivalence of CPU and CUDA outputs
    template<typename TInput = float, typename TOutput = TInput, typename TPrecision = TOutput>
    void TestCpuCudaEquivalence(
        const MLPTestData<Compute::DeviceType::Cpu, TInput, TOutput, TPrecision>& cpu_data,
        const MLPTestData<Compute::DeviceType::Cuda, TInput, TOutput, TPrecision>& cuda_data ) {

        try {
            // Check if CUDA is available
            DeviceContext context( "CUDA:0" );
        }
        catch ( const std::exception& ) {
            GTEST_SKIP() << "CUDA device not available, skipping CPU-CUDA equivalence test";
            return;
        }

        // Create a very small test shape to make comparison faster
        std::vector<size_t> test_shape = { 1, 2, cpu_data.input_shape.back() }; // Minimal shape for quick verification

        // Use a smaller output channel count for faster testing
        size_t test_output_channels = 1024; // Reduced from 3072

        // Create new, smaller MLPs specifically for this test
        auto cpu_mlp = std::make_shared<MLP<Compute::DeviceType::Cpu, TInput, TOutput, TPrecision>>(
            "test_cpu_mlp", "CPU", test_shape, test_output_channels, cpu_data.has_bias );

        auto cuda_mlp = std::make_shared<MLP<Compute::DeviceType::Cuda, TInput, TOutput, TPrecision>>(
            "test_cuda_mlp", "CUDA:0", test_shape, test_output_channels, cuda_data.has_bias );

        // Create random input data
        Tensor<TInput, Compute::HostMemoryResource> host_input( test_shape );

        // Fill with predictable values between -2.0 and 2.0
        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input.data()[ i ] = static_cast<TInput>( -2.0 + 4.0 * (static_cast<float>( i ) / host_input.size()) );
        }

        Tensor<TOutput, Compute::HostMemoryResource> cpu_output( test_shape );

        cpu_mlp->forward( host_input, cpu_output );

        // Create device input by copying host data
        Tensor<TInput, Compute::CudaMemoryResource> device_input( test_shape );
        device_input.copyFrom( host_input );

        // Create device output
        Tensor<TOutput, Compute::CudaMemoryResource> cuda_output( test_shape );

        cuda_mlp->forward( device_input, cuda_output );

        // Ensure any CUDA errors are caught immediately
        cuda_mlp->getDeviceContext()->synchronize();

        // Copy CUDA output back to host for comparison
        Tensor<TOutput, Compute::HostMemoryResource> cuda_output_host( test_shape );
        cuda_output_host.copyFrom( cuda_output );

        // Compare outputs with tolerance for floating point differences
        const float epsilon = 1e-3f; // Slightly larger tolerance for MLP with multiple ops
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

    // Test with different dimensions (edge cases) - device-specific shapes
    template<DeviceType TDevice, typename TInput = float, typename TOutput = TInput, typename TPrecision = TOutput>
    void TestEdgeCases() {
        using MR = MemoryResourceType<TDevice, TPrecision>;

        try {
            // Test with minimal sizes
            std::vector<size_t> minimal_shape = { 1, 1, 8 };
            size_t minimal_output_channels = 16;

            auto minimal_module = std::make_shared<MLP<TDevice, TInput, TOutput, TPrecision>>(
                "minimal_mlp", TDevice == Compute::DeviceType::Cuda ? "CUDA:0" : "CPU",
                minimal_shape, minimal_output_channels );

            Tensor<TInput, MR> minimal_input( minimal_shape );
            Tensor<TOutput, MR> minimal_output( minimal_shape );

            EXPECT_NO_THROW( minimal_module->forward( minimal_input, minimal_output ) );
            EXPECT_EQ( minimal_output.size(), 8 );

            // Test with medium dimensions - adjusted by device type
            std::vector<size_t> medium_shape;
            if constexpr ( TDevice == Compute::DeviceType::Cuda ) {
                medium_shape = { 2, 2, 1024 };
            }
            else {
                medium_shape = { 1, 2, 512 }; // Smaller for CPU
            }

            size_t medium_output_channels = 2048;

            auto medium_module = std::make_shared<MLP<TDevice, TInput, TOutput, TPrecision>>(
                "medium_mlp", TDevice == Compute::DeviceType::Cuda ? "CUDA:0" : "CPU",
                medium_shape, medium_output_channels );

            Tensor<TInput, MR> medium_input( medium_shape );
            Tensor<TOutput, MR> medium_output( medium_shape );

            EXPECT_NO_THROW( medium_module->forward( medium_input, medium_output ) );
        }
        catch ( const std::exception& e ) {
            std::cerr << "Exception during edge case test: " << e.what() << std::endl;
            throw;
        }
    }

    // CPU Tests with float precision
    TEST_F( MLPTests, Cpu_Float_TestName ) {
        TestGetName<Compute::DeviceType::Cpu, float>( CpuFloatData(), "cpu_mlp_float" );
    }

    TEST_F( MLPTests, Cpu_Float_ParameterCount ) {
        TestParameterCount<Compute::DeviceType::Cpu, float>( CpuFloatData() );
    }

    TEST_F( MLPTests, Cpu_Float_TestForward ) {
        TestForward<Compute::DeviceType::Cpu, float>( CpuFloatData() );
    }

    TEST_F( MLPTests, Cpu_Float_TestPrint ) {
        TestPrint<Compute::DeviceType::Cpu, float>( CpuFloatData(), "MLP: cpu_mlp_float" );
    }

    TEST_F( MLPTests, Cpu_Float_TrainingMode ) {
        TestTrainingMode<Compute::DeviceType::Cpu, float>( CpuFloatData(), false );
    }

    TEST_F( MLPTests, Cpu_Float_DeviceType ) {
        TestDeviceType<Compute::DeviceType::Cpu, float>( CpuFloatData() );
    }

    TEST_F( MLPTests, Cpu_Float_SubModules ) {
        TestSubModules<Compute::DeviceType::Cpu, float>( CpuFloatData() );
    }

    TEST_F( MLPTests, Cpu_Float_SaveLoad ) {
        TestSaveLoad<Compute::DeviceType::Cpu, float>( CpuFloatData() );
    }

    // CPU NoBias Tests
    TEST_F( MLPTests, NoBias_Cpu_Float_ParameterCount ) {
        TestParameterCount<Compute::DeviceType::Cpu, float>( NoBiasCpuFloatData() );
    }

    TEST_F( MLPTests, NoBias_Cpu_Float_TestForward ) {
        TestForward<Compute::DeviceType::Cpu, float>( NoBiasCpuFloatData() );
    }

    // CPU Training Mode Tests
    TEST_F( MLPTests, Cpu_Training_Float_TrainingMode ) {
        TestTrainingMode<Compute::DeviceType::Cpu, float>( TrainingCpuFloatData(), true );
    }

    TEST_F( MLPTests, Cpu_Training_Float_TestForward ) {
        TestForward<Compute::DeviceType::Cpu, float>( TrainingCpuFloatData() );
    }

    // CUDA Tests with float precision
    TEST_F( MLPTests, Cuda_Float_TestName ) {
        TestGetName<Compute::DeviceType::Cuda, float>( CudaFloatData(), "cuda_mlp_float" );
    }

    TEST_F( MLPTests, Cuda_Float_ParameterCount ) {
        TestParameterCount<Compute::DeviceType::Cuda, float>( CudaFloatData() );
    }

    TEST_F( MLPTests, Cuda_Float_TestForward ) {
        TestForward<Compute::DeviceType::Cuda, float>( CudaFloatData() );
    }

    TEST_F( MLPTests, Cuda_Float_TestPrint ) {
        TestPrint<Compute::DeviceType::Cuda, float>( CudaFloatData(), "MLP: cuda_mlp_float" );
    }

    TEST_F( MLPTests, Cuda_Float_TrainingMode ) {
        TestTrainingMode<Compute::DeviceType::Cuda, float>( CudaFloatData(), false );
    }

    TEST_F( MLPTests, Cuda_Float_DeviceType ) {
        TestDeviceType<Compute::DeviceType::Cuda, float>( CudaFloatData() );
    }

    TEST_F( MLPTests, Cuda_Float_SubModules ) {
        TestSubModules<Compute::DeviceType::Cuda, float>( CudaFloatData() );
    }

    // CUDA NoBias Tests
    TEST_F( MLPTests, NoBias_Cuda_Float_ParameterCount ) {
        TestParameterCount<Compute::DeviceType::Cuda, float>( NoBiasCudaFloatData() );
    }

    TEST_F( MLPTests, NoBias_Cuda_Float_TestForward ) {
        TestForward<Compute::DeviceType::Cuda, float>( NoBiasCudaFloatData() );
    }

    // CUDA Training Mode Tests
    TEST_F( MLPTests, Cuda_Training_Float_TrainingMode ) {
        TestTrainingMode<Compute::DeviceType::Cuda, float>( TrainingCudaFloatData(), true );
    }

    TEST_F( MLPTests, Cuda_Training_Float_TestForward ) {
        TestForward<Compute::DeviceType::Cuda, float>( TrainingCudaFloatData() );
    }

    // CUDA Tests with half precision
    TEST_F( MLPTests, Cuda_Half_TestName ) {
        TestGetName<Compute::DeviceType::Cuda, half>( CudaHalfData(), "cuda_mlp_half" );
    }

    TEST_F( MLPTests, Cuda_Half_ParameterCount ) {
        TestParameterCount<Compute::DeviceType::Cuda, half>( CudaHalfData() );
    }

    TEST_F( MLPTests, Cuda_Half_TestForward ) {
        TestForward<Compute::DeviceType::Cuda, half>( CudaHalfData() );
    }

    TEST_F( MLPTests, Cuda_Half_TestPrint ) {
        TestPrint<Compute::DeviceType::Cuda, half>( CudaHalfData(), "MLP: cuda_mlp_half" );
    }

    TEST_F( MLPTests, Cuda_Half_TrainingMode ) {
        TestTrainingMode<Compute::DeviceType::Cuda, half>( CudaHalfData(), false );
    }

    // Mixed Precision Tests
    TEST_F( MLPTests, Cuda_MixedPrecision_TestForward ) {
        TestForward<Compute::DeviceType::Cuda, float, half>( MixedPrecisionData() );
    }

    TEST_F( MLPTests, Cuda_MixedPrecision_TestName ) {
        TestGetName<Compute::DeviceType::Cuda, float, half>( MixedPrecisionData(), "cuda_mlp_mixed" );
    }

    // Context Construction Tests
    TEST_F( MLPTests, Context_Cpu_Float_DeviceType ) {
        TestDeviceType<Compute::DeviceType::Cpu, float>( ContextCpuFloatData() );
    }

    TEST_F( MLPTests, Context_Cpu_Float_Forward ) {
        TestForward<Compute::DeviceType::Cpu, float>( ContextCpuFloatData() );
    }

    // Edge Case Tests
    TEST_F( MLPTests, Cpu_Float_EdgeCases ) {
        TestEdgeCases<Compute::DeviceType::Cpu, float>();
    }

    TEST_F( MLPTests, Cuda_Float_EdgeCases ) {
        TestEdgeCases<Compute::DeviceType::Cuda, float>();
    }

    // CPU-CUDA Equivalence Test
    TEST_F( MLPTests, CpuCuda_Forward_Output_Equivalence ) {
        TestCpuCudaEquivalence<float>( CpuFloatData(), CudaFloatData() );
    }
}