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
    struct GeluTestData {
        std::vector<size_t> shape;
        std::shared_ptr<Gelu<TPrecision, TPrecision, TPrecision, TDevice>> gelu_module;
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

            std::string device_str = TDevice == Compute::DeviceType::Cuda ? "CUDA:0" : "CPU";
            data.gelu_module = std::make_shared<Gelu<TPrecision, TDevice>>(
                name, device_str, is_training );

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

            data.gelu_module = std::make_shared<Gelu<TPrecision, TDevice>>(
                name, context, is_training );

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

        // Factory methods to lazily create test data as needed
        GeluTestData<float, Compute::DeviceType::Cpu>& CpuFloatData() {
            if ( !cpu_float_data_.gelu_module ) {
                cpu_float_data_ = GeluTestData<float, Compute::DeviceType::Cpu>::Create(
                    "cpu_gelu_float", cpu_batch_size_, sequence_length_, channels_ );
            }
            return cpu_float_data_;
        }

        GeluTestData<float, Compute::DeviceType::Cuda>& CudaFloatData() {
            if ( !cuda_float_data_.gelu_module ) {
                cuda_float_data_ = GeluTestData<float, Compute::DeviceType::Cuda>::Create(
                    "cuda_gelu_float", batch_size_, sequence_length_, channels_ );
            }
            return cuda_float_data_;
        }

        GeluTestData<float, Compute::DeviceType::Cpu>& TrainingCpuFloatData() {
            if ( !training_cpu_float_data_.gelu_module ) {
                training_cpu_float_data_ = GeluTestData<float, Compute::DeviceType::Cpu>::Create(
                    "cpu_gelu_float_training", cpu_batch_size_, sequence_length_, channels_, true );
            }
            return training_cpu_float_data_;
        }

        GeluTestData<float, Compute::DeviceType::Cuda>& TrainingCudaFloatData() {
            if ( !training_cuda_float_data_.gelu_module ) {
                training_cuda_float_data_ = GeluTestData<float, Compute::DeviceType::Cuda>::Create(
                    "cuda_gelu_float_training", batch_size_, sequence_length_, channels_, true );
            }
            return training_cuda_float_data_;
        }

        GeluTestData<float, Compute::DeviceType::Cpu>& ContextCpuFloatData() {
            if ( !context_cpu_float_data_.gelu_module ) {
                auto cpu_context = std::make_shared<DeviceContext>( "CPU" );
                context_cpu_float_data_ = GeluTestData<float, Compute::DeviceType::Cpu>::CreateWithContext(
                    "cpu_context_gelu_float", cpu_batch_size_, sequence_length_, channels_, cpu_context );
            }
            return context_cpu_float_data_;
        }

        GeluTestData<half, Compute::DeviceType::Cuda>& CudaHalfData() {
            if ( !cuda_half_data_.gelu_module ) {
                cuda_half_data_ = GeluTestData<half, Compute::DeviceType::Cuda>::Create(
                    "cuda_gelu_half", batch_size_, sequence_length_, channels_ );
            }
            return cuda_half_data_;
        }

        GeluTestData<half, Compute::DeviceType::Cuda>& TrainingCudaHalfData() {
            if ( !training_cuda_half_data_.gelu_module ) {
                training_cuda_half_data_ = GeluTestData<half, Compute::DeviceType::Cuda>::Create(
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
        GeluTestData<float, Compute::DeviceType::Cpu> cpu_float_data_;
        GeluTestData<float, Compute::DeviceType::Cpu> context_cpu_float_data_;
        GeluTestData<float, Compute::DeviceType::Cpu> training_cpu_float_data_;

        GeluTestData<float, Compute::DeviceType::Cuda> cuda_float_data_;
        GeluTestData<float, Compute::DeviceType::Cuda> training_cuda_float_data_;

        GeluTestData<half, Compute::DeviceType::Cuda> cuda_half_data_;
        GeluTestData<half, Compute::DeviceType::Cuda> training_cuda_half_data_;
    };

    // Common test function templates
    template<typename TPrecision, Compute::DeviceType TDevice>
    void TestGetName( const GeluTestData<TPrecision, TDevice>& data, const std::string& expected_name ) {
        EXPECT_EQ( data.gelu_module->getName(), expected_name );
    }

    template<typename TPrecision, Compute::DeviceType TDevice>
    void TestParameterCount( const GeluTestData<TPrecision, TDevice>& data, size_t expected_count ) {
        EXPECT_EQ( data.gelu_module->parameterCount(), expected_count );
    }

    template<typename TPrecision, Compute::DeviceType TDevice>
    void TestForward( const GeluTestData<TPrecision, TDevice>& data ) {
        using MR = MemoryResourceType<TPrecision, TDevice>;

        Tensor<TPrecision, MR> input( data.shape );
        Tensor<TPrecision, MR> output( data.shape );
        data.gelu_module->forward( input, output );
        EXPECT_EQ( output.size(), input.size() );
    }

    template<typename TPrecision, Compute::DeviceType TDevice>
    void TestPrint( const GeluTestData<TPrecision, TDevice>& data, const std::string& expected_substring ) {
        std::string output = data.gelu_module->toString();
        EXPECT_NE( output.find( expected_substring ), std::string::npos );
    }

    template<typename TPrecision, Compute::DeviceType TDevice>
    void TestTrainingMode( const GeluTestData<TPrecision, TDevice>& data, bool expected_mode ) {
        EXPECT_EQ( data.gelu_module->isTraining(), expected_mode );
    }

    template<typename TPrecision, Compute::DeviceType TDevice>
    void TestDeviceType( const GeluTestData<TPrecision, TDevice>& data ) {
        auto device_context = data.gelu_module->getDeviceContext();
        EXPECT_NE( device_context, nullptr );
        auto device = device_context->getDevice();
        EXPECT_NE( device, nullptr );
        EXPECT_EQ( device->getDeviceType(), TDevice );
    }

    // Function to test equivalence of CPU and CUDA outputs
    template<typename TPrecision>
    void TestCpuCudaEquivalence(
        const GeluTestData<TPrecision, Compute::DeviceType::Cpu>& cpu_data,
        const GeluTestData<TPrecision, Compute::DeviceType::Cuda>& cuda_data ) {

        // Create a small test shape to make comparison faster
        std::vector<size_t> test_shape = { 2, 4, 8 }; // Small shape for quick verification

        // Create random input data
        Tensor<TPrecision, Compute::HostMemoryResource> host_input( test_shape );

        // Fill with predictable values between -2.0 and 2.0 to exercise the GELU function
        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input.data()[ i ] = static_cast<TPrecision>( -2.0 + 4.0 * (static_cast<float>( i ) / host_input.size()) );
        }

        // Create CPU output
        Tensor<TPrecision, Compute::HostMemoryResource> cpu_output( test_shape );
        cpu_data.gelu_module->forward( host_input, cpu_output );

        // Create device input by copying host data
        Tensor<TPrecision, Compute::CudaMemoryResource> device_input( test_shape );
        device_input.copyFrom( host_input );

        // Create device output
        Tensor<TPrecision, Compute::CudaMemoryResource> cuda_output( test_shape );
        cuda_data.gelu_module->forward( device_input, cuda_output );

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
            }
        }

        EXPECT_TRUE( all_equal ) << "CPU and CUDA implementations produced different results";
    }

    // Test with different dimensions (edge cases)
    template<typename TPrecision, Compute::DeviceType TDevice>
    void TestEdgeCases() {
        using MR = MemoryResourceType<TPrecision, TDevice>;

        try {
            // Test with minimal sizes
            std::vector<size_t> minimal_shape = { 1, 1, 8 };

            auto minimal_module = std::make_shared<Gelu<TPrecision, TDevice>>(
                "minimal_gelu", TDevice == Compute::DeviceType::Cuda ? "CUDA:0" : "CPU" );

            Tensor<TPrecision, MR> minimal_input( minimal_shape );
            Tensor<TPrecision, MR> minimal_output( minimal_shape );

            EXPECT_NO_THROW( minimal_module->forward( minimal_input, minimal_output ) );
            EXPECT_EQ( minimal_output.size(), 8 );

            // Test with larger dimensions
            std::vector<size_t> large_shape = { 2, 2, 1024 };

            auto large_module = std::make_shared<Gelu<TPrecision, TDevice>>(
                "large_gelu", TDevice == Compute::DeviceType::Cuda ? "CUDA:0" : "CPU" );

            Tensor<TPrecision, MR> large_input( large_shape );
            Tensor<TPrecision, MR> large_output( large_shape );

            EXPECT_NO_THROW( large_module->forward( large_input, large_output ) );
            EXPECT_EQ( large_output.size(), 4096 );
        }
        catch ( const std::exception& e ) {
            std::cerr << "Exception during edge case test: " << e.what() << std::endl;
            throw;
        }
    }

    // Test for device change events
    //template<typename TPrecision>
    //void TestDeviceChange( const GeluTestData<TPrecision, Compute::DeviceType::Cpu>& data ) {
    //    // Create a new device context
    //    auto new_context = std::make_shared<DeviceContext>( "CPU" );

    //    // Change device
    //    EXPECT_NO_THROW( data.gelu_module->setDeviceContext( new_context ) );

    //    // Verify operations still work after device change
    //    std::vector<size_t> test_shape = { 2, 4, 8 };
    //    Tensor<TPrecision, Compute::HostMemoryResource> input( test_shape );
    //    Tensor<TPrecision, Compute::HostMemoryResource> output( test_shape );
    //    EXPECT_NO_THROW( data.gelu_module->forward( input, output ) );
    //}

    // CPU Tests with float precision
    TEST_F( GeluTests, Cpu_Float_TestName ) {
        TestGetName<float, Compute::DeviceType::Cpu>( CpuFloatData(), "cpu_gelu_float" );
    }

    TEST_F( GeluTests, Cpu_Float_ParameterCount ) {
        TestParameterCount<float, Compute::DeviceType::Cpu>( CpuFloatData(), 0 );
    }

    TEST_F( GeluTests, Cpu_Float_TestForward ) {
        TestForward<float, Compute::DeviceType::Cpu>( CpuFloatData() );
    }

    TEST_F( GeluTests, Cpu_Float_TestPrint ) {
        TestPrint<float, Compute::DeviceType::Cpu>( CpuFloatData(), "Gelu: cpu_gelu_float" );
    }

    TEST_F( GeluTests, Cpu_Float_TrainingMode ) {
        TestTrainingMode<float, Compute::DeviceType::Cpu>( CpuFloatData(), false );
    }

    TEST_F( GeluTests, Cpu_Float_DeviceType ) {
        TestDeviceType<float, Compute::DeviceType::Cpu>( CpuFloatData() );
    }

    // CPU Training Mode Tests
    TEST_F( GeluTests, Cpu_Training_Float_TrainingMode ) {
        TestTrainingMode<float, Compute::DeviceType::Cpu>( TrainingCpuFloatData(), true );
    }

    // CUDA Tests with float precision
    TEST_F( GeluTests, Cuda_Float_TestName ) {
        TestGetName<float, Compute::DeviceType::Cuda>( CudaFloatData(), "cuda_gelu_float" );
    }

    TEST_F( GeluTests, Cuda_Float_ParameterCount ) {
        TestParameterCount<float, Compute::DeviceType::Cuda>( CudaFloatData(), 0 );
    }

    TEST_F( GeluTests, Cuda_Float_TestForward ) {
        TestForward<float, Compute::DeviceType::Cuda>( CudaFloatData() );
    }

    TEST_F( GeluTests, Cuda_Float_TestPrint ) {
        TestPrint<float, Compute::DeviceType::Cuda>( CudaFloatData(), "Gelu: cuda_gelu_float" );
    }

    TEST_F( GeluTests, Cuda_Float_TrainingMode ) {
        TestTrainingMode<float, Compute::DeviceType::Cuda>( CudaFloatData(), false );
    }

    TEST_F( GeluTests, Cuda_Float_DeviceType ) {
        TestDeviceType<float, Compute::DeviceType::Cuda>( CudaFloatData() );
    }

    // CUDA Training Mode Tests
    TEST_F( GeluTests, Cuda_Training_Float_TrainingMode ) {
        TestTrainingMode<float, Compute::DeviceType::Cuda>( TrainingCudaFloatData(), true );
    }

    // CUDA Tests with half precision
    TEST_F( GeluTests, Cuda_Half_TestName ) {
        TestGetName<half, Compute::DeviceType::Cuda>( CudaHalfData(), "cuda_gelu_half" );
    }

    TEST_F( GeluTests, Cuda_Half_ParameterCount ) {
        TestParameterCount<half, Compute::DeviceType::Cuda>( CudaHalfData(), 0 );
    }

    TEST_F( GeluTests, Cuda_Half_TestForward ) {
        TestForward<half, Compute::DeviceType::Cuda>( CudaHalfData() );
    }

    TEST_F( GeluTests, Cuda_Half_TestPrint ) {
        TestPrint<half, Compute::DeviceType::Cuda>( CudaHalfData(), "Gelu: cuda_gelu_half" );
    }

    TEST_F( GeluTests, Cuda_Half_TrainingMode ) {
        TestTrainingMode<half, Compute::DeviceType::Cuda>( CudaHalfData(), false );
    }

    // Context Construction Tests
    TEST_F( GeluTests, Context_Cpu_Float_DeviceType ) {
        TestDeviceType<float, Compute::DeviceType::Cpu>( ContextCpuFloatData() );
    }

    TEST_F( GeluTests, Context_Cpu_Float_Forward ) {
        TestForward<float, Compute::DeviceType::Cpu>( ContextCpuFloatData() );
    }

    // Device Change Tests
    /*TEST_F( GeluTests, Cpu_Float_DeviceChange ) {
        TestDeviceChange<float>( CpuFloatData() );
    }*/

    // Edge Case Tests
    TEST_F( GeluTests, Cpu_Float_EdgeCases ) {
        TestEdgeCases<float, Compute::DeviceType::Cpu>();
    }

    TEST_F( GeluTests, Cuda_Float_EdgeCases ) {
        TestEdgeCases<float, Compute::DeviceType::Cuda>();
    }

    // CPU-CUDA Equivalence Test
    TEST_F( GeluTests, CpuCuda_Forward_Output_Equivalence ) {
        TestCpuCudaEquivalence<float>( CpuFloatData(), CudaFloatData() );
    }
}
