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
        Compute::CudaDeviceMemoryResource,
        Compute::CpuMemoryResource>;

    template<typename TPrecision, Compute::DeviceType TDevice>
    struct TransformerBlockTestData {
        std::vector<size_t> input_shape;
        size_t num_heads;
        std::shared_ptr<TransformerBlock<TPrecision, TDevice>> transformer_module;
        bool is_training;

        // Make the test data structure self-initializing
        static TransformerBlockTestData Create(
            const std::string& name,
            size_t batch_size,
            size_t sequence_length,
            size_t embedding_dim,
            size_t num_heads,
            bool is_training = false )
        {
            TransformerBlockTestData data;
            data.input_shape = { batch_size, sequence_length, embedding_dim };
            data.num_heads = num_heads;
            data.is_training = is_training;

            std::string device_name = TDevice == Compute::DeviceType::Cuda ? "CUDA:0" : "CPU";
            data.transformer_module = std::make_shared<TransformerBlock<TPrecision, TDevice>>(
                name, device_name, data.input_shape, num_heads, is_training );

            return data;
        }

        // Overload for creating with device context
        static TransformerBlockTestData CreateWithContext(
            const std::string& name,
            size_t batch_size,
            size_t sequence_length,
            size_t embedding_dim,
            size_t num_heads,
            std::shared_ptr<DeviceContext> context,
            bool is_training = false )
        {
            TransformerBlockTestData data;
            data.input_shape = { batch_size, sequence_length, embedding_dim };
            data.num_heads = num_heads;
            data.is_training = is_training;

            data.transformer_module = std::make_shared<TransformerBlock<TPrecision, TDevice>>(
                name, context, data.input_shape, num_heads, is_training );

            return data;
        }
    };

    class TransformerBlockTests : public ::testing::Test {
    protected:
        void SetUp() override {
            // CUDA-specific parameters - smaller sizes for parallel processing
            cuda_batch_size_ = 4;
            cuda_sequence_length_ = 32;

            // CPU-specific parameters - even smaller sizes to prevent timeouts
            cpu_batch_size_ = 1;
            cpu_sequence_length_ = 8;

            // Common parameters for both
            embedding_dim_ = 256;
            num_heads_ = 8; // Must divide evenly into embedding_dim
        }

        // Factory methods to lazily create test data as needed
        TransformerBlockTestData<float, Compute::DeviceType::Cpu>& CpuFloatData() {
            if ( !cpu_float_data_.transformer_module ) {
                cpu_float_data_ = TransformerBlockTestData<float, Compute::DeviceType::Cpu>::Create(
                    "cpu_transformer_float", cpu_batch_size_, cpu_sequence_length_, embedding_dim_, num_heads_ );
            }
            return cpu_float_data_;
        }

        TransformerBlockTestData<float, Compute::DeviceType::Cuda>& CudaFloatData() {
            if ( !cuda_float_data_.transformer_module ) {
                cuda_float_data_ = TransformerBlockTestData<float, Compute::DeviceType::Cuda>::Create(
                    "cuda_transformer_float", cuda_batch_size_, cuda_sequence_length_, embedding_dim_, num_heads_ );
            }
            return cuda_float_data_;
        }

        TransformerBlockTestData<float, Compute::DeviceType::Cpu>& TrainingCpuFloatData() {
            if ( !training_cpu_float_data_.transformer_module ) {
                training_cpu_float_data_ = TransformerBlockTestData<float, Compute::DeviceType::Cpu>::Create(
                    "cpu_transformer_float_training", cpu_batch_size_, cpu_sequence_length_, embedding_dim_, num_heads_, true );
            }
            return training_cpu_float_data_;
        }

        TransformerBlockTestData<float, Compute::DeviceType::Cuda>& TrainingCudaFloatData() {
            if ( !training_cuda_float_data_.transformer_module ) {
                training_cuda_float_data_ = TransformerBlockTestData<float, Compute::DeviceType::Cuda>::Create(
                    "cuda_transformer_float_training", cuda_batch_size_, cuda_sequence_length_, embedding_dim_, num_heads_, true );
            }
            return training_cuda_float_data_;
        }

        TransformerBlockTestData<float, Compute::DeviceType::Cpu>& ContextCpuFloatData() {
            if ( !context_cpu_float_data_.transformer_module ) {
                auto cpu_context = std::make_shared<DeviceContext>( "CPU" );
                context_cpu_float_data_ = TransformerBlockTestData<float, Compute::DeviceType::Cpu>::CreateWithContext(
                    "cpu_context_transformer_float", cpu_batch_size_, cpu_sequence_length_, embedding_dim_,
                    num_heads_, cpu_context );
            }
            return context_cpu_float_data_;
        }

        TransformerBlockTestData<half, Compute::DeviceType::Cuda>& CudaHalfData() {
            if ( !cuda_half_data_.transformer_module ) {
                cuda_half_data_ = TransformerBlockTestData<half, Compute::DeviceType::Cuda>::Create(
                    "cuda_transformer_half", cuda_batch_size_, cuda_sequence_length_, embedding_dim_, num_heads_ );
            }
            return cuda_half_data_;
        }

        TransformerBlockTestData<half, Compute::DeviceType::Cuda>& TrainingCudaHalfData() {
            if ( !training_cuda_half_data_.transformer_module ) {
                training_cuda_half_data_ = TransformerBlockTestData<half, Compute::DeviceType::Cuda>::Create(
                    "cuda_transformer_half_training", cuda_batch_size_, cuda_sequence_length_, embedding_dim_, num_heads_, true );
            }
            return training_cuda_half_data_;
        }

        // Test parameters - device-specific sizes
        size_t cuda_batch_size_{ 0 };
        size_t cpu_batch_size_{ 0 };
        size_t cuda_sequence_length_{ 0 };
        size_t cpu_sequence_length_{ 0 };

        // Common parameters
        size_t embedding_dim_{ 0 };
        size_t num_heads_{ 0 };

        // Test data objects - initialized on demand
        TransformerBlockTestData<float, Compute::DeviceType::Cpu> cpu_float_data_;
        TransformerBlockTestData<float, Compute::DeviceType::Cpu> context_cpu_float_data_;
        TransformerBlockTestData<float, Compute::DeviceType::Cpu> training_cpu_float_data_;

        TransformerBlockTestData<float, Compute::DeviceType::Cuda> cuda_float_data_;
        TransformerBlockTestData<float, Compute::DeviceType::Cuda> training_cuda_float_data_;

        TransformerBlockTestData<half, Compute::DeviceType::Cuda> cuda_half_data_;
        TransformerBlockTestData<half, Compute::DeviceType::Cuda> training_cuda_half_data_;
    };

    // Common test function templates
    template<typename TPrecision, Compute::DeviceType TDevice>
    void TestGetName( const TransformerBlockTestData<TPrecision, TDevice>& data, const std::string& expected_name ) {
        EXPECT_EQ( data.transformer_module->getName(), expected_name );
    }

    template<typename TPrecision, Compute::DeviceType TDevice>
    void TestParameterCount( const TransformerBlockTestData<TPrecision, TDevice>& data ) {
        // The TransformerBlock's parameter count is the sum of all its submodules' parameters
        size_t params_count = data.transformer_module->parameterCount();
        EXPECT_GT( params_count, 0 );
    }

    template<typename TPrecision, Compute::DeviceType TDevice>
    void TestForward( const TransformerBlockTestData<TPrecision, TDevice>& data ) {
        using MR = MemoryResourceType<TPrecision, TDevice>;

        Tensor<TPrecision, MR> input( data.input_shape );
        Tensor<TPrecision, MR> output( data.input_shape ); // Transformer preserves input shape

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

        data.transformer_module->forward( input, output );

        // Output should have same shape as input for TransformerBlock
        EXPECT_EQ( output.size(), input.size() );
        EXPECT_EQ( output.shape(), input.shape() );
    }

    template<typename TPrecision, Compute::DeviceType TDevice>
    void TestPrint( const TransformerBlockTestData<TPrecision, TDevice>& data, const std::string& expected_substring ) {
        std::string output = data.transformer_module->toString();
        EXPECT_NE( output.find( expected_substring ), std::string::npos );
    }

    template<typename TPrecision, Compute::DeviceType TDevice>
    void TestTrainingMode( const TransformerBlockTestData<TPrecision, TDevice>& data, bool expected_mode ) {
        EXPECT_EQ( data.transformer_module->isTraining(), expected_mode );
    }

    template<typename TPrecision, Compute::DeviceType TDevice>
    void TestDeviceType( const TransformerBlockTestData<TPrecision, TDevice>& data ) {
        auto device_context = data.transformer_module->getDeviceContext();
        EXPECT_NE( device_context, nullptr );
        auto device = device_context->getDevice();
        EXPECT_NE( device, nullptr );
        EXPECT_EQ( device->getDeviceType(), TDevice );
    }

    // Test for submodule structure
    template<typename TPrecision, Compute::DeviceType TDevice>
    void TestSubModules( const TransformerBlockTestData<TPrecision, TDevice>& data ) {
        auto modules = data.transformer_module->getNamedModules();

        // Verify we have all expected submodules
        EXPECT_EQ( modules.size(), 8 );
        EXPECT_NE( modules.find( "ln_1" ), modules.end() );
        EXPECT_NE( modules.find( "fc_qkv" ), modules.end() );
        EXPECT_NE( modules.find( "attn" ), modules.end() );
        EXPECT_NE( modules.find( "fc_attn_proj" ), modules.end() );
        EXPECT_NE( modules.find( "res_1" ), modules.end() );
        EXPECT_NE( modules.find( "ln_2" ), modules.end() );
        EXPECT_NE( modules.find( "mlp" ), modules.end() );
        EXPECT_NE( modules.find( "res_2" ), modules.end() );
    }

    // Test save/load functionality
    //template<typename TPrecision, Compute::DeviceType TDevice>
    //void TestSaveLoad( const TransformerBlockTestData<TPrecision, TDevice>& data ) {
    //    // Just verify methods exist and don't crash when called
    //    mz_zip_archive zip = {};
    //    EXPECT_NO_THROW( data.transformer_module->save( zip ) );
    //    EXPECT_NO_THROW( data.transformer_module->load( zip ) );
    //}

    // Function to test equivalence of CPU and CUDA outputs
    template<typename TPrecision>
    void TestCpuCudaEquivalence(
        const TransformerBlockTestData<TPrecision, Compute::DeviceType::Cpu>& cpu_data,
        const TransformerBlockTestData<TPrecision, Compute::DeviceType::Cuda>& cuda_data ) {

        // Skip if no CUDA device is available
        if ( !cuda_data.transformer_module->getDeviceContext()->isCudaDevice() ) {
            GTEST_SKIP() << "CUDA device not available, skipping test";
        }

        // Create a very small test shape to make comparison faster
        std::vector<size_t> test_shape = { 1, 2, 64 }; // Minimal shape for quick verification
        size_t test_num_heads = 2;

        // Create new, smaller TransformerBlocks specifically for this test
        auto cpu_transformer = std::make_shared<TransformerBlock<TPrecision, Compute::DeviceType::Cpu>>(
            "test_cpu_transformer", "CPU", test_shape, test_num_heads);

        auto cuda_transformer = std::make_shared<TransformerBlock<TPrecision, Compute::DeviceType::Cuda>>(
            "test_cuda_transformer", "CUDA:0", test_shape, test_num_heads);

        // Create random input data
        Tensor<TPrecision, Compute::HostMemoryResource> host_input( test_shape );

        // Fill with predictable values between -2.0 and 2.0
        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input.data()[ i ] = static_cast<TPrecision>( -2.0 + 4.0 * (static_cast<float>( i ) / host_input.size()) );
        }

        Tensor<TPrecision, Compute::HostMemoryResource> cpu_output( test_shape );

        cpu_transformer->forward( host_input, cpu_output );

        // Create device input by copying host data
        Tensor<TPrecision, Compute::CudaDeviceMemoryResource> device_input( test_shape );
        device_input.copyFrom( host_input );

        // Create device output
        Tensor<TPrecision, Compute::CudaDeviceMemoryResource> cuda_output( test_shape );

        cuda_transformer->forward( device_input, cuda_output );

        // Ensure any CUDA errors are caught immediately
        cuda_transformer->getDeviceContext()->synchronize();

        // Copy CUDA output back to host for comparison
        Tensor<TPrecision, Compute::HostMemoryResource> cuda_output_host( test_shape );
        cuda_output_host.copyFrom( cuda_output );

        // Compare outputs with tolerance for floating point differences
        const float epsilon = 1e-3f; // Slightly larger tolerance for transformer with multiple ops
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

        std::string device_name = TDevice == Compute::DeviceType::Cuda ? "CUDA:0" : "CPU";

        try {
            // Test with minimal sizes
            std::vector<size_t> minimal_shape = { 1, 1, 64 }; // Head size needs to be divisible by num_heads
            size_t minimal_num_heads = 2;

            auto minimal_module = std::make_shared<TransformerBlock<TPrecision, TDevice>>(
                "minimal_transformer", device_name, minimal_shape, minimal_num_heads );

            Tensor<TPrecision, MR> minimal_input( minimal_shape );
            Tensor<TPrecision, MR> minimal_output( minimal_shape );

            EXPECT_NO_THROW( minimal_module->forward( minimal_input, minimal_output ) );
            EXPECT_EQ( minimal_output.size(), minimal_input.size() );

            // Test with medium dimensions - adjusted by device type
            std::vector<size_t> medium_shape;
            if constexpr ( TDevice == Compute::DeviceType::Cuda ) {
                medium_shape = { 2, 2, 128 };
            }
            else {
                medium_shape = { 1, 2, 128 }; // Smaller for CPU
            }

            size_t medium_num_heads = 4;

            auto medium_module = std::make_shared<TransformerBlock<TPrecision, TDevice>>(
                "medium_transformer", device_name, medium_shape, medium_num_heads );

            Tensor<TPrecision, MR> medium_input( medium_shape );
            Tensor<TPrecision, MR> medium_output( medium_shape );

            EXPECT_NO_THROW( medium_module->forward( medium_input, medium_output ) );
        }
        catch ( const std::exception& e ) {
            std::cerr << "Exception during edge case test: " << e.what() << std::endl;
            throw;
        }
    }

    // Test exception handling for invalid input shapes
    template<typename TPrecision, Compute::DeviceType TDevice>
    void TestInvalidShape() {
        std::string device_name = TDevice == Compute::DeviceType::Cuda ? "CUDA:0" : "CPU";

        // Test with invalid shape (not rank 3)
        std::vector<size_t> invalid_shape = { 1, 64 }; // Only 2 dimensions
        size_t num_heads = 2;

        EXPECT_THROW( (
            std::make_shared<TransformerBlock<TPrecision, TDevice>>( "invalid_transformer", device_name, invalid_shape, num_heads ) ),
            std::invalid_argument
        );
    }

    // CPU Tests with float precision
    TEST_F( TransformerBlockTests, Cpu_Float_TestName ) {
        TestGetName<float, Compute::DeviceType::Cpu>( CpuFloatData(), "cpu_transformer_float" );
    }

    TEST_F( TransformerBlockTests, Cpu_Float_ParameterCount ) {
        TestParameterCount<float, Compute::DeviceType::Cpu>( CpuFloatData() );
    }

    TEST_F( TransformerBlockTests, Cpu_Float_TestForward ) {
        TestForward<float, Compute::DeviceType::Cpu>( CpuFloatData() );
    }

    TEST_F( TransformerBlockTests, Cpu_Float_TestPrint ) {
        TestPrint<float, Compute::DeviceType::Cpu>( CpuFloatData(), "Transformer: cpu_transformer_float" );
    }

    TEST_F( TransformerBlockTests, Cpu_Float_TrainingMode ) {
        TestTrainingMode<float, Compute::DeviceType::Cpu>( CpuFloatData(), false );
    }

    TEST_F( TransformerBlockTests, Cpu_Float_DeviceType ) {
        TestDeviceType<float, Compute::DeviceType::Cpu>( CpuFloatData() );
    }

    TEST_F( TransformerBlockTests, Cpu_Float_SubModules ) {
        TestSubModules<float, Compute::DeviceType::Cpu>( CpuFloatData() );
    }

    /*TEST_F( TransformerBlockTests, Cpu_Float_SaveLoad ) {
        TestSaveLoad<float, Compute::DeviceType::Cpu>( CpuFloatData() );
    }*/

    TEST_F( TransformerBlockTests, Cpu_Float_InvalidShape ) {
        TestInvalidShape<float, Compute::DeviceType::Cpu>();
    }

    // CPU Training Mode Tests
    TEST_F( TransformerBlockTests, Cpu_Training_Float_TrainingMode ) {
        TestTrainingMode<float, Compute::DeviceType::Cpu>( TrainingCpuFloatData(), true );
    }

    TEST_F( TransformerBlockTests, Cpu_Training_Float_TestForward ) {
        TestForward<float, Compute::DeviceType::Cpu>( TrainingCpuFloatData() );
    }

    // CUDA Tests with float precision
    TEST_F( TransformerBlockTests, Cuda_Float_TestName ) {
        TestGetName<float, Compute::DeviceType::Cuda>( CudaFloatData(), "cuda_transformer_float" );
    }

    TEST_F( TransformerBlockTests, Cuda_Float_ParameterCount ) {
        TestParameterCount<float, Compute::DeviceType::Cuda>( CudaFloatData() );
    }

    TEST_F( TransformerBlockTests, Cuda_Float_TestForward ) {
        TestForward<float, Compute::DeviceType::Cuda>( CudaFloatData() );
    }

    TEST_F( TransformerBlockTests, Cuda_Float_TestPrint ) {
        TestPrint<float, Compute::DeviceType::Cuda>( CudaFloatData(), "Transformer: cuda_transformer_float" );
    }

    TEST_F( TransformerBlockTests, Cuda_Float_TrainingMode ) {
        TestTrainingMode<float, Compute::DeviceType::Cuda>( CudaFloatData(), false );
    }

    TEST_F( TransformerBlockTests, Cuda_Float_DeviceType ) {
        TestDeviceType<float, Compute::DeviceType::Cuda>( CudaFloatData() );
    }

    TEST_F( TransformerBlockTests, Cuda_Float_SubModules ) {
        TestSubModules<float, Compute::DeviceType::Cuda>( CudaFloatData() );
    }

    TEST_F( TransformerBlockTests, Cuda_Float_InvalidShape ) {
        TestInvalidShape<float, Compute::DeviceType::Cuda>();
    }

    // CUDA Training Mode Tests
    TEST_F( TransformerBlockTests, Cuda_Training_Float_TrainingMode ) {
        TestTrainingMode<float, Compute::DeviceType::Cuda>( TrainingCudaFloatData(), true );
    }

    TEST_F( TransformerBlockTests, Cuda_Training_Float_TestForward ) {
        TestForward<float, Compute::DeviceType::Cuda>( TrainingCudaFloatData() );
    }

    // CUDA Tests with half precision
    TEST_F( TransformerBlockTests, Cuda_Half_TestName ) {
        TestGetName<half, Compute::DeviceType::Cuda>( CudaHalfData(), "cuda_transformer_half" );
    }

    TEST_F( TransformerBlockTests, Cuda_Half_ParameterCount ) {
        TestParameterCount<half, Compute::DeviceType::Cuda>( CudaHalfData() );
    }

    TEST_F( TransformerBlockTests, Cuda_Half_TestForward ) {
        TestForward<half, Compute::DeviceType::Cuda>( CudaHalfData() );
    }

    TEST_F( TransformerBlockTests, Cuda_Half_TestPrint ) {
        TestPrint<half, Compute::DeviceType::Cuda>( CudaHalfData(), "Transformer: cuda_transformer_half" );
    }

    TEST_F( TransformerBlockTests, Cuda_Half_TrainingMode ) {
        TestTrainingMode<half, Compute::DeviceType::Cuda>( CudaHalfData(), false );
    }

    // Context Construction Tests
    TEST_F( TransformerBlockTests, Context_Cpu_Float_DeviceType ) {
        TestDeviceType<float, Compute::DeviceType::Cpu>( ContextCpuFloatData() );
    }

    TEST_F( TransformerBlockTests, Context_Cpu_Float_Forward ) {
        TestForward<float, Compute::DeviceType::Cpu>( ContextCpuFloatData() );
    }

    // Edge Case Tests
    TEST_F( TransformerBlockTests, Cpu_Float_EdgeCases ) {
        TestEdgeCases<float, Compute::DeviceType::Cpu>();
    }

    TEST_F( TransformerBlockTests, Cuda_Float_EdgeCases ) {
        TestEdgeCases<float, Compute::DeviceType::Cuda>();
    }

    // CPU-CUDA Equivalence Test
    TEST_F( TransformerBlockTests, CpuCuda_Forward_Output_Equivalence ) {
        TestCpuCudaEquivalence<float>( CpuFloatData(), CudaFloatData() );
    }
}
