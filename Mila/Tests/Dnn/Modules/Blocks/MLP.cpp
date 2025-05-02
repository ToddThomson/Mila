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

    template<typename TPrecision, Compute::DeviceType TDevice>
    using MemoryResourceType = std::conditional_t<TDevice == Compute::DeviceType::Cuda,
        Compute::CudaMemoryResource,
        Compute::HostMemoryResource>;

    template<typename TPrecision, Compute::DeviceType TDevice>
    struct MLPTestData {
        std::vector<size_t> input_shape;
        size_t output_channels;
        std::shared_ptr<MLP<TPrecision, TDevice>> mlp_module;
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
            data.mlp_module = std::make_shared<MLP<TPrecision, TDevice>>(
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

            data.mlp_module = std::make_shared<MLP<TPrecision, TDevice>>(
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

        // Factory methods to lazily create test data as needed
        MLPTestData<float, Compute::DeviceType::Cpu>& CpuFloatData() {
            if ( !cpu_float_data_.mlp_module ) {
                cpu_float_data_ = MLPTestData<float, Compute::DeviceType::Cpu>::Create(
                    "cpu_mlp_float", cpu_batch_size_, cpu_sequence_length_, input_channels_, output_channels_ );
            }
            return cpu_float_data_;
        }

        MLPTestData<float, Compute::DeviceType::Cuda>& CudaFloatData() {
            if ( !cuda_float_data_.mlp_module ) {
                cuda_float_data_ = MLPTestData<float, Compute::DeviceType::Cuda>::Create(
                    "cuda_mlp_float", cuda_batch_size_, cuda_sequence_length_, input_channels_, output_channels_ );
            }
            return cuda_float_data_;
        }

        MLPTestData<float, Compute::DeviceType::Cpu>& TrainingCpuFloatData() {
            if ( !training_cpu_float_data_.mlp_module ) {
                training_cpu_float_data_ = MLPTestData<float, Compute::DeviceType::Cpu>::Create(
                    "cpu_mlp_float_training", cpu_batch_size_, cpu_sequence_length_, input_channels_, output_channels_, true, true );
            }
            return training_cpu_float_data_;
        }

        MLPTestData<float, Compute::DeviceType::Cuda>& TrainingCudaFloatData() {
            if ( !training_cuda_float_data_.mlp_module ) {
                training_cuda_float_data_ = MLPTestData<float, Compute::DeviceType::Cuda>::Create(
                    "cuda_mlp_float_training", cuda_batch_size_, cuda_sequence_length_, input_channels_, output_channels_, true, true );
            }
            return training_cuda_float_data_;
        }

        MLPTestData<float, Compute::DeviceType::Cpu>& NoBiasCpuFloatData() {
            if ( !no_bias_cpu_float_data_.mlp_module ) {
                no_bias_cpu_float_data_ = MLPTestData<float, Compute::DeviceType::Cpu>::Create(
                    "cpu_mlp_float_nobias", cpu_batch_size_, cpu_sequence_length_, input_channels_, output_channels_, false );
            }
            return no_bias_cpu_float_data_;
        }

        MLPTestData<float, Compute::DeviceType::Cuda>& NoBiasCudaFloatData() {
            if ( !no_bias_cuda_float_data_.mlp_module ) {
                no_bias_cuda_float_data_ = MLPTestData<float, Compute::DeviceType::Cuda>::Create(
                    "cuda_mlp_float_nobias", cuda_batch_size_, cuda_sequence_length_, input_channels_, output_channels_, false );
            }
            return no_bias_cuda_float_data_;
        }

        MLPTestData<float, Compute::DeviceType::Cpu>& ContextCpuFloatData() {
            if ( !context_cpu_float_data_.mlp_module ) {
                auto cpu_context = std::make_shared<DeviceContext>( "CPU" );
                context_cpu_float_data_ = MLPTestData<float, Compute::DeviceType::Cpu>::CreateWithContext(
                    "cpu_context_mlp_float", cpu_batch_size_, cpu_sequence_length_, input_channels_, output_channels_, cpu_context );
            }
            return context_cpu_float_data_;
        }

        MLPTestData<half, Compute::DeviceType::Cuda>& CudaHalfData() {
            if ( !cuda_half_data_.mlp_module ) {
                cuda_half_data_ = MLPTestData<half, Compute::DeviceType::Cuda>::Create(
                    "cuda_mlp_half", cuda_batch_size_, cuda_sequence_length_, input_channels_, output_channels_ );
            }
            return cuda_half_data_;
        }

        MLPTestData<half, Compute::DeviceType::Cuda>& TrainingCudaHalfData() {
            if ( !training_cuda_half_data_.mlp_module ) {
                training_cuda_half_data_ = MLPTestData<half, Compute::DeviceType::Cuda>::Create(
                    "cuda_mlp_half_training", cuda_batch_size_, cuda_sequence_length_, input_channels_, output_channels_, true, true );
            }
            return training_cuda_half_data_;
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
        MLPTestData<float, Compute::DeviceType::Cpu> cpu_float_data_;
        MLPTestData<float, Compute::DeviceType::Cpu> context_cpu_float_data_;
        MLPTestData<float, Compute::DeviceType::Cpu> training_cpu_float_data_;
        MLPTestData<float, Compute::DeviceType::Cpu> no_bias_cpu_float_data_;

        MLPTestData<float, Compute::DeviceType::Cuda> cuda_float_data_;
        MLPTestData<float, Compute::DeviceType::Cuda> training_cuda_float_data_;
        MLPTestData<float, Compute::DeviceType::Cuda> no_bias_cuda_float_data_;

        MLPTestData<half, Compute::DeviceType::Cuda> cuda_half_data_;
        MLPTestData<half, Compute::DeviceType::Cuda> training_cuda_half_data_;
    };

    // Common test function templates
    template<typename TPrecision, Compute::DeviceType TDevice>
    void TestGetName( const MLPTestData<TPrecision, TDevice>& data, const std::string& expected_name ) {
        EXPECT_EQ( data.mlp_module->getName(), expected_name );
    }

    template<typename TPrecision, Compute::DeviceType TDevice>
    void TestParameterCount( const MLPTestData<TPrecision, TDevice>& data ) {
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

    template<typename TPrecision, Compute::DeviceType TDevice>
    void TestForward( const MLPTestData<TPrecision, TDevice>& data ) {
        using MR = MemoryResourceType<TPrecision, TDevice>;

        Tensor<TPrecision, MR> input( data.input_shape );
        Tensor<TPrecision, MR> output( data.input_shape ); // MLP preserves spatial dimensions but may change channel dim

        // Initialize input with some values based on device type
        if constexpr ( TDevice == Compute::DeviceType::Cuda ) {
            // For CUDA device, create a host tensor first, initialize it, then copy to device
            Tensor<TPrecision, Compute::HostMemoryResource> host_input( data.input_shape );

            // Fill host input with values
            for ( size_t i = 0; i < host_input.size(); ++i ) {
                host_input.data()[ i ] = static_cast<TPrecision>( i % 10 * 0.1f );
            }

            // Copy initialized data to device tensor
            input.copyFrom( host_input );
        }
        else {
            // For CPU, initialize directly
            for ( size_t i = 0; i < input.size(); ++i ) {
                input.data()[ i ] = static_cast<TPrecision>( i % 10 * 0.1f );
            }
        }

        data.mlp_module->forward( input, output );

        // Output should have same shape as input for MLP
        EXPECT_EQ( output.size(), input.size() );
        EXPECT_EQ( output.shape(), input.shape() );
    }


    template<typename TPrecision, Compute::DeviceType TDevice>
    void TestPrint( const MLPTestData<TPrecision, TDevice>& data, const std::string& expected_substring ) {
        std::string output = data.mlp_module->toString();
        EXPECT_NE( output.find( expected_substring ), std::string::npos );
    }

    template<typename TPrecision, Compute::DeviceType TDevice>
    void TestTrainingMode( const MLPTestData<TPrecision, TDevice>& data, bool expected_mode ) {
        EXPECT_EQ( data.mlp_module->isTraining(), expected_mode );
    }

    template<typename TPrecision, Compute::DeviceType TDevice>
    void TestDeviceType( const MLPTestData<TPrecision, TDevice>& data ) {
        auto device_context = data.mlp_module->getDeviceContext();
        EXPECT_NE( device_context, nullptr );
        auto device = device_context->getDevice();
        EXPECT_NE( device, nullptr );
        EXPECT_EQ( device->getDeviceType(), TDevice );
    }

    // Test for submodule structure
    template<typename TPrecision, Compute::DeviceType TDevice>
    void TestSubModules( const MLPTestData<TPrecision, TDevice>& data ) {
        auto modules = data.mlp_module->getNamedModules();

        // Verify we have all 3 expected submodules: fc_1, gelu, fc_proj
        EXPECT_EQ( modules.size(), 3 );
        EXPECT_NE( modules.find( "fc_1" ), modules.end() );
        EXPECT_NE( modules.find( "gelu" ), modules.end() );
        EXPECT_NE( modules.find( "fc_proj" ), modules.end() );
    }

    // Test save/load functionality
    template<typename TPrecision, Compute::DeviceType TDevice>
    void TestSaveLoad( const MLPTestData<TPrecision, TDevice>& data ) {
        // This is a minimal test for the save/load methods
        // In a real test, you would need to set up a zip archive and verify serialization

        // Just verify methods exist and don't crash when called
        mz_zip_archive zip = {};
        EXPECT_NO_THROW( data.mlp_module->save( zip ) );
        EXPECT_NO_THROW( data.mlp_module->load( zip ) );
    }

    // Function to test equivalence of CPU and CUDA outputs
    template<typename TPrecision>
    void TestCpuCudaEquivalence(
        const MLPTestData<TPrecision, Compute::DeviceType::Cpu>& cpu_data,
        const MLPTestData<TPrecision, Compute::DeviceType::Cuda>& cuda_data ) {

        // Skip if no CUDA device is available
        if ( !cuda_data.mlp_module->getDeviceContext()->isCudaDevice() ) {
            GTEST_SKIP() << "CUDA device not available, skipping test";
        }

        // Create a very small test shape to make comparison faster
        std::vector<size_t> test_shape = { 1, 2, cpu_data.input_shape.back() }; // Minimal shape for quick verification

        // Use a smaller output channel count for faster testing
        size_t test_output_channels = 1024; // Reduced from 3072

        // Create new, smaller MLPs specifically for this test
        auto cpu_mlp = std::make_shared<MLP<TPrecision, Compute::DeviceType::Cpu>>(
            "test_cpu_mlp", "CPU", test_shape, test_output_channels, cpu_data.has_bias );

        auto cuda_mlp = std::make_shared<MLP<TPrecision, Compute::DeviceType::Cuda>>(
            "test_cuda_mlp", "CUDA:0", test_shape, test_output_channels, cuda_data.has_bias );

        // Create random input data
        Tensor<TPrecision, Compute::HostMemoryResource> host_input( test_shape );

        // Fill with predictable values between -2.0 and 2.0
        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input.data()[ i ] = static_cast<TPrecision>( -2.0 + 4.0 * (static_cast<float>( i ) / host_input.size()) );
        }

        Tensor<TPrecision, Compute::HostMemoryResource> cpu_output( test_shape );

        cpu_mlp->forward( host_input, cpu_output );

        // Create device input by copying host data
        Tensor<TPrecision, Compute::CudaMemoryResource> device_input( test_shape );
        device_input.copyFrom( host_input );

        // Create device output
        Tensor<TPrecision, Compute::CudaMemoryResource> cuda_output( test_shape );

        cuda_mlp->forward( device_input, cuda_output );

        // Ensure any CUDA errors are caught immediately
        cuda_mlp->getDeviceContext()->synchronize();

        // Copy CUDA output back to host for comparison
        Tensor<TPrecision, Compute::HostMemoryResource> cuda_output_host( test_shape );
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
    template<typename TPrecision, Compute::DeviceType TDevice>
    void TestEdgeCases() {
        using MR = MemoryResourceType<TPrecision, TDevice>;

        try {
            // Test with minimal sizes
            std::vector<size_t> minimal_shape = { 1, 1, 8 };
            size_t minimal_output_channels = 16;

            auto minimal_module = std::make_shared<MLP<TPrecision, TDevice>>(
                "minimal_mlp", TDevice == Compute::DeviceType::Cuda ? "CUDA:0" : "CPU",
                minimal_shape, minimal_output_channels );

            Tensor<TPrecision, MR> minimal_input( minimal_shape );
            Tensor<TPrecision, MR> minimal_output( minimal_shape );

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

            auto medium_module = std::make_shared<MLP<TPrecision, TDevice>>(
                "medium_mlp", TDevice == Compute::DeviceType::Cuda ? "CUDA:0" : "CPU",
                medium_shape, medium_output_channels );

            Tensor<TPrecision, MR> medium_input( medium_shape );
            Tensor<TPrecision, MR> medium_output( medium_shape );

            EXPECT_NO_THROW( medium_module->forward( medium_input, medium_output ) );
        }
        catch ( const std::exception& e ) {
            std::cerr << "Exception during edge case test: " << e.what() << std::endl;
            throw;
        }
    }

    // CPU Tests with float precision
    TEST_F( MLPTests, Cpu_Float_TestName ) {
        TestGetName<float, Compute::DeviceType::Cpu>( CpuFloatData(), "cpu_mlp_float" );
    }

    TEST_F( MLPTests, Cpu_Float_ParameterCount ) {
        TestParameterCount<float, Compute::DeviceType::Cpu>( CpuFloatData() );
    }

    TEST_F( MLPTests, Cpu_Float_TestForward ) {
        TestForward<float, Compute::DeviceType::Cpu>( CpuFloatData() );
    }

    TEST_F( MLPTests, Cpu_Float_TestPrint ) {
        TestPrint<float, Compute::DeviceType::Cpu>( CpuFloatData(), "MLP: cpu_mlp_float" );
    }

    TEST_F( MLPTests, Cpu_Float_TrainingMode ) {
        TestTrainingMode<float, Compute::DeviceType::Cpu>( CpuFloatData(), false );
    }

    TEST_F( MLPTests, Cpu_Float_DeviceType ) {
        TestDeviceType<float, Compute::DeviceType::Cpu>( CpuFloatData() );
    }

    TEST_F( MLPTests, Cpu_Float_SubModules ) {
        TestSubModules<float, Compute::DeviceType::Cpu>( CpuFloatData() );
    }

    TEST_F( MLPTests, Cpu_Float_SaveLoad ) {
        TestSaveLoad<float, Compute::DeviceType::Cpu>( CpuFloatData() );
    }

    // CPU NoBias Tests
    TEST_F( MLPTests, NoBias_Cpu_Float_ParameterCount ) {
        TestParameterCount<float, Compute::DeviceType::Cpu>( NoBiasCpuFloatData() );
    }

    TEST_F( MLPTests, NoBias_Cpu_Float_TestForward ) {
        TestForward<float, Compute::DeviceType::Cpu>( NoBiasCpuFloatData() );
    }

    // CPU Training Mode Tests
    TEST_F( MLPTests, Cpu_Training_Float_TrainingMode ) {
        TestTrainingMode<float, Compute::DeviceType::Cpu>( TrainingCpuFloatData(), true );
    }

    TEST_F( MLPTests, Cpu_Training_Float_TestForward ) {
        TestForward<float, Compute::DeviceType::Cpu>( TrainingCpuFloatData() );
    }

    // CUDA Tests with float precision
    TEST_F( MLPTests, Cuda_Float_TestName ) {
        TestGetName<float, Compute::DeviceType::Cuda>( CudaFloatData(), "cuda_mlp_float" );
    }

    TEST_F( MLPTests, Cuda_Float_ParameterCount ) {
        TestParameterCount<float, Compute::DeviceType::Cuda>( CudaFloatData() );
    }

    TEST_F( MLPTests, Cuda_Float_TestForward ) {
        TestForward<float, Compute::DeviceType::Cuda>( CudaFloatData() );
    }

    TEST_F( MLPTests, Cuda_Float_TestPrint ) {
        TestPrint<float, Compute::DeviceType::Cuda>( CudaFloatData(), "MLP: cuda_mlp_float" );
    }

    TEST_F( MLPTests, Cuda_Float_TrainingMode ) {
        TestTrainingMode<float, Compute::DeviceType::Cuda>( CudaFloatData(), false );
    }

    TEST_F( MLPTests, Cuda_Float_DeviceType ) {
        TestDeviceType<float, Compute::DeviceType::Cuda>( CudaFloatData() );
    }

    TEST_F( MLPTests, Cuda_Float_SubModules ) {
        TestSubModules<float, Compute::DeviceType::Cuda>( CudaFloatData() );
    }

    // CUDA NoBias Tests
    TEST_F( MLPTests, NoBias_Cuda_Float_ParameterCount ) {
        TestParameterCount<float, Compute::DeviceType::Cuda>( NoBiasCudaFloatData() );
    }

    TEST_F( MLPTests, NoBias_Cuda_Float_TestForward ) {
        TestForward<float, Compute::DeviceType::Cuda>( NoBiasCudaFloatData() );
    }

    // CUDA Training Mode Tests
    TEST_F( MLPTests, Cuda_Training_Float_TrainingMode ) {
        TestTrainingMode<float, Compute::DeviceType::Cuda>( TrainingCudaFloatData(), true );
    }

    TEST_F( MLPTests, Cuda_Training_Float_TestForward ) {
        TestForward<float, Compute::DeviceType::Cuda>( TrainingCudaFloatData() );
    }

    // CUDA Tests with half precision
    TEST_F( MLPTests, Cuda_Half_TestName ) {
        TestGetName<half, Compute::DeviceType::Cuda>( CudaHalfData(), "cuda_mlp_half" );
    }

    TEST_F( MLPTests, Cuda_Half_ParameterCount ) {
        TestParameterCount<half, Compute::DeviceType::Cuda>( CudaHalfData() );
    }

    TEST_F( MLPTests, Cuda_Half_TestForward ) {
        TestForward<half, Compute::DeviceType::Cuda>( CudaHalfData() );
    }

    TEST_F( MLPTests, Cuda_Half_TestPrint ) {
        TestPrint<half, Compute::DeviceType::Cuda>( CudaHalfData(), "MLP: cuda_mlp_half" );
    }

    TEST_F( MLPTests, Cuda_Half_TrainingMode ) {
        TestTrainingMode<half, Compute::DeviceType::Cuda>( CudaHalfData(), false );
    }

    // Context Construction Tests
    TEST_F( MLPTests, Context_Cpu_Float_DeviceType ) {
        TestDeviceType<float, Compute::DeviceType::Cpu>( ContextCpuFloatData() );
    }

    TEST_F( MLPTests, Context_Cpu_Float_Forward ) {
        TestForward<float, Compute::DeviceType::Cpu>( ContextCpuFloatData() );
    }

    // Edge Case Tests
    TEST_F( MLPTests, Cpu_Float_EdgeCases ) {
        TestEdgeCases<float, Compute::DeviceType::Cpu>();
    }

    TEST_F( MLPTests, Cuda_Float_EdgeCases ) {
        TestEdgeCases<float, Compute::DeviceType::Cuda>();
    }

    // CPU-CUDA Equivalence Test
    TEST_F( MLPTests, CpuCuda_Forward_Output_Equivalence ) {
        TestCpuCudaEquivalence<float>( CpuFloatData(), CudaFloatData() );
    }
}

