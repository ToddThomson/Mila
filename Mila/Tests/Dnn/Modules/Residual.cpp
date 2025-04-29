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
        Compute::CpuMemoryResource>;

    template<typename TPrecision, Compute::DeviceType TDevice>
    struct ResidualTestData {
        std::vector<size_t> shape;
        std::shared_ptr<Residual<TPrecision, TDevice>> residual_module;
        bool is_training;

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

            std::string device_str = TDevice == Compute::DeviceType::Cuda ? "CUDA:0" : "CPU";
            
            data.residual_module = std::make_shared<Residual<TPrecision, TDevice>>(
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

            data.residual_module = std::make_shared<Residual<TPrecision, TDevice>>(
                name, context, is_training );

            return data;
        }
    };

    class ResidualTests : public ::testing::Test {
    protected:
        void SetUp() override {
            batch_size_ = 128;
            cpu_batch_size_ = 4;
            sequence_length_ = 1024;
            channels_ = 768;
        }

        // Factory methods to lazily create test data as needed
        ResidualTestData<float, Compute::DeviceType::Cpu>& CpuFloatData() {
            if ( !cpu_float_data_.residual_module ) {
                cpu_float_data_ = ResidualTestData<float, Compute::DeviceType::Cpu>::Create(
                    "cpu_residual_float", cpu_batch_size_, sequence_length_, channels_ );
            }
            return cpu_float_data_;
        }

        ResidualTestData<float, Compute::DeviceType::Cuda>& CudaFloatData() {
            if ( !cuda_float_data_.residual_module ) {
                cuda_float_data_ = ResidualTestData<float, Compute::DeviceType::Cuda>::Create(
                    "cuda_residual_float", batch_size_, sequence_length_, channels_ );
            }
            return cuda_float_data_;
        }

        ResidualTestData<float, Compute::DeviceType::Cpu>& TrainingCpuFloatData() {
            if ( !training_cpu_float_data_.residual_module ) {
                training_cpu_float_data_ = ResidualTestData<float, Compute::DeviceType::Cpu>::Create(
                    "cpu_residual_float_training", cpu_batch_size_, sequence_length_, channels_, true );
            }
            return training_cpu_float_data_;
        }

        ResidualTestData<float, Compute::DeviceType::Cuda>& TrainingCudaFloatData() {
            if ( !training_cuda_float_data_.residual_module ) {
                training_cuda_float_data_ = ResidualTestData<float, Compute::DeviceType::Cuda>::Create(
                    "cuda_residual_float_training", batch_size_, sequence_length_, channels_, true );
            }
            return training_cuda_float_data_;
        }

        ResidualTestData<float, Compute::DeviceType::Cpu>& ContextCpuFloatData() {
            if ( !context_cpu_float_data_.residual_module ) {
                auto cpu_context = std::make_shared<DeviceContext>( "CPU" );
                context_cpu_float_data_ = ResidualTestData<float, Compute::DeviceType::Cpu>::CreateWithContext(
                    "cpu_context_residual_float", cpu_batch_size_, sequence_length_, channels_, cpu_context );
            }
            return context_cpu_float_data_;
        }

        ResidualTestData<half, Compute::DeviceType::Cuda>& CudaHalfData() {
            if ( !cuda_half_data_.residual_module ) {
                cuda_half_data_ = ResidualTestData<half, Compute::DeviceType::Cuda>::Create(
                    "cuda_residual_half", batch_size_, sequence_length_, channels_ );
            }
            return cuda_half_data_;
        }

        ResidualTestData<half, Compute::DeviceType::Cuda>& TrainingCudaHalfData() {
            if ( !training_cuda_half_data_.residual_module ) {
                training_cuda_half_data_ = ResidualTestData<half, Compute::DeviceType::Cuda>::Create(
                    "cuda_residual_half_training", batch_size_, sequence_length_, channels_, true );
            }
            return training_cuda_half_data_;
        }

        // Test parameters
        size_t batch_size_{ 0 };
        size_t cpu_batch_size_{ 0 };
        size_t sequence_length_{ 0 };
        size_t channels_{ 0 };

        // Test data objects - initialized on demand
        ResidualTestData<float, Compute::DeviceType::Cpu> cpu_float_data_;
        ResidualTestData<float, Compute::DeviceType::Cpu> context_cpu_float_data_;
        ResidualTestData<float, Compute::DeviceType::Cpu> training_cpu_float_data_;

        ResidualTestData<float, Compute::DeviceType::Cuda> cuda_float_data_;
        ResidualTestData<float, Compute::DeviceType::Cuda> training_cuda_float_data_;

        ResidualTestData<half, Compute::DeviceType::Cuda> cuda_half_data_;
        ResidualTestData<half, Compute::DeviceType::Cuda> training_cuda_half_data_;
    };

    // Common test function templates
    template<typename TPrecision, Compute::DeviceType TDevice>
    void TestGetName( const ResidualTestData<TPrecision, TDevice>& data, const std::string& expected_name ) {
        EXPECT_EQ( data.residual_module->getName(), expected_name );
    }

    template<typename TPrecision, Compute::DeviceType TDevice>
    void TestParameterCount( const ResidualTestData<TPrecision, TDevice>& data, size_t expected_count ) {
        EXPECT_EQ( data.residual_module->parameterCount(), expected_count );
    }

    template<typename TPrecision, Compute::DeviceType TDevice>
    void TestForward( const ResidualTestData<TPrecision, TDevice>& data ) {
        using MR = std::conditional_t<TDevice == Compute::DeviceType::Cuda,
            Compute::CudaMemoryResource,
            Compute::CpuMemoryResource>;

        Tensor<TPrecision, MR> input_a( data.shape, 4.0f );
        Tensor<TPrecision, MR> input_b( data.shape, 2.0f );
        Tensor<TPrecision, MR> output( data.shape );

        data.residual_module->forward( input_a, input_b, output );
        
        EXPECT_EQ( output.size(), input_a.size() );
    }

    template<typename TPrecision, Compute::DeviceType TDevice>
    void TestPrint( const ResidualTestData<TPrecision, TDevice>& data, const std::string& expected_substring ) {
        std::string output = data.residual_module->toString();
        EXPECT_NE( output.find( expected_substring ), std::string::npos );
    }

    template<typename TPrecision, Compute::DeviceType TDevice>
    void TestTrainingMode( const ResidualTestData<TPrecision, TDevice>& data, bool expected_mode ) {
        EXPECT_EQ( data.residual_module->isTraining(), expected_mode );
    }

    template<typename TPrecision, Compute::DeviceType TDevice>
    void TestDeviceType( const ResidualTestData<TPrecision, TDevice>& data ) {
        auto device_context = data.residual_module->getDeviceContext();
        EXPECT_NE( device_context, nullptr );
        auto device = device_context->getDevice();
        EXPECT_NE( device, nullptr );
        EXPECT_EQ( device->getDeviceType(), TDevice );
    }

    // Function to test equivalence of CPU and CUDA outputs
    template<typename TPrecision>
    void TestCpuCudaEquivalence(
        const ResidualTestData<TPrecision, Compute::DeviceType::Cpu>& cpu_data,
        const ResidualTestData<TPrecision, Compute::DeviceType::Cuda>& cuda_data ) {

        // Create a small test shape to make comparison faster
        std::vector<size_t> test_shape = { 2, 4, 8 }; // Small shape for quick verification

        // Create random input data
        Tensor<TPrecision, Compute::CpuMemoryResource> host_input_a( test_shape );
        Tensor<TPrecision, Compute::CpuMemoryResource> host_input_b( test_shape );

        // Fill with predictable values between -2.0 and 2.0 to exercise the residual function
        for ( size_t i = 0; i < host_input_a.size(); ++i ) {
            host_input_a.data()[ i ] = static_cast<TPrecision>( -2.0 + 4.0 * (static_cast<float>( i ) / host_input_a.size()) );
            host_input_b.data()[ i ] = static_cast<TPrecision>( 2.0 - 4.0 * (static_cast<float>( i ) / host_input_b.size()) );
        }

        // Create CPU output
        Tensor<TPrecision, Compute::CpuMemoryResource> cpu_output( test_shape );
        cpu_data.residual_module->forward( host_input_a, host_input_b, cpu_output );

        // Create device input by copying host data
        Tensor<TPrecision, Compute::CudaMemoryResource> device_input_a( test_shape );
        Tensor<TPrecision, Compute::CudaMemoryResource> device_input_b( test_shape );
        device_input_a.copyFrom( host_input_a );
        device_input_b.copyFrom( host_input_b );

        // Create device output
        Tensor<TPrecision, Compute::CudaMemoryResource> cuda_output( test_shape );
        cuda_data.residual_module->forward( device_input_a, device_input_b, cuda_output );

        // Copy CUDA output back to host for comparison
        Tensor<TPrecision, Compute::CpuMemoryResource> cuda_output_host( test_shape );
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

            auto minimal_module = std::make_shared<Residual<TPrecision, TDevice>>(
                "minimal_residual", TDevice == Compute::DeviceType::Cuda ? "CUDA:0" : "CPU" );

            Tensor<TPrecision, MR> minimal_input_a( minimal_shape, 1.0f );
            Tensor<TPrecision, MR> minimal_input_b( minimal_shape, 2.0f );
            Tensor<TPrecision, MR> minimal_output( minimal_shape );

            EXPECT_NO_THROW( minimal_module->forward( minimal_input_a, minimal_input_b, minimal_output ) );
            EXPECT_EQ( minimal_output.size(), 8 );

            // Test with larger dimensions
            std::vector<size_t> large_shape = { 2, 2, 1024 };

            auto large_module = std::make_shared<Residual<TPrecision, TDevice>>(
                "large_residual", TDevice == Compute::DeviceType::Cuda ? "CUDA:0" : "CPU" );

            Tensor<TPrecision, MR> large_input_a( large_shape, 1.0f );
            Tensor<TPrecision, MR> large_input_b( large_shape, 2.0f );
            Tensor<TPrecision, MR> large_output( large_shape );

            EXPECT_NO_THROW( large_module->forward( large_input_a, large_input_b, large_output ) );
            EXPECT_EQ( large_output.size(), 4096 );
        }
        catch ( const std::exception& e ) {
            std::cerr << "Exception during edge case test: " << e.what() << std::endl;
            throw;
        }
    }

    // Test for device change events
    //template<typename TPrecision>
    //void TestDeviceChange( const ResidualTestData<TPrecision, Compute::DeviceType::Cpu>& data ) {
    //    // Create a new device context
    //    auto new_context = std::make_shared<DeviceContext>( "CPU" );

    //    // Change device
    //    EXPECT_NO_THROW( data.residual_module->setDeviceContext( new_context ) );

    //    // Verify operations still work after device change
    //    std::vector<size_t> test_shape = { 2, 4, 8 };
    //    Tensor<TPrecision, Compute::CpuMemoryResource> input_a( test_shape, 1.0f );
    //    Tensor<TPrecision, Compute::CpuMemoryResource> input_b( test_shape, 2.0f );
    //    Tensor<TPrecision, Compute::CpuMemoryResource> output( test_shape );
    //    EXPECT_NO_THROW( data.residual_module->forward( input_a, input_b, output ) );
    //}

    // CPU Tests with float precision
    TEST_F( ResidualTests, Cpu_Float_TestName ) {
        TestGetName<float, Compute::DeviceType::Cpu>( CpuFloatData(), "cpu_residual_float" );
    }

    TEST_F( ResidualTests, Cpu_Float_ParameterCount ) {
        TestParameterCount<float, Compute::DeviceType::Cpu>( CpuFloatData(), 0 );
    }

    TEST_F( ResidualTests, Cpu_Float_TestForward ) {
        TestForward<float, Compute::DeviceType::Cpu>( CpuFloatData() );
    }

    TEST_F( ResidualTests, Cpu_Float_TestPrint ) {
        TestPrint<float, Compute::DeviceType::Cpu>( CpuFloatData(), "Residual: cpu_residual_float" );
    }

    TEST_F( ResidualTests, Cpu_Float_TrainingMode ) {
        TestTrainingMode<float, Compute::DeviceType::Cpu>( CpuFloatData(), false );
    }

    TEST_F( ResidualTests, Cpu_Float_DeviceType ) {
        TestDeviceType<float, Compute::DeviceType::Cpu>( CpuFloatData() );
    }

    // CPU Training Mode Tests
    TEST_F( ResidualTests, Cpu_Training_Float_TrainingMode ) {
        TestTrainingMode<float, Compute::DeviceType::Cpu>( TrainingCpuFloatData(), true );
    }

    // CUDA Tests with float precision
    TEST_F( ResidualTests, Cuda_Float_TestName ) {
        TestGetName<float, Compute::DeviceType::Cuda>( CudaFloatData(), "cuda_residual_float" );
    }

    TEST_F( ResidualTests, Cuda_Float_ParameterCount ) {
        TestParameterCount<float, Compute::DeviceType::Cuda>( CudaFloatData(), 0 );
    }

    TEST_F( ResidualTests, Cuda_Float_TestForward ) {
        TestForward<float, Compute::DeviceType::Cuda>( CudaFloatData() );
    }

    TEST_F( ResidualTests, Cuda_Float_TestPrint ) {
        TestPrint<float, Compute::DeviceType::Cuda>( CudaFloatData(), "Residual: cuda_residual_float" );
    }

    TEST_F( ResidualTests, Cuda_Float_TrainingMode ) {
        TestTrainingMode<float, Compute::DeviceType::Cuda>( CudaFloatData(), false );
    }

    TEST_F( ResidualTests, Cuda_Float_DeviceType ) {
        TestDeviceType<float, Compute::DeviceType::Cuda>( CudaFloatData() );
    }

    // CUDA Training Mode Tests
    TEST_F( ResidualTests, Cuda_Training_Float_TrainingMode ) {
        TestTrainingMode<float, Compute::DeviceType::Cuda>( TrainingCudaFloatData(), true );
    }

    // CUDA Tests with half precision
    TEST_F( ResidualTests, Cuda_Half_TestName ) {
        TestGetName<half, Compute::DeviceType::Cuda>( CudaHalfData(), "cuda_residual_half" );
    }

    TEST_F( ResidualTests, Cuda_Half_ParameterCount ) {
        TestParameterCount<half, Compute::DeviceType::Cuda>( CudaHalfData(), 0 );
    }

    TEST_F( ResidualTests, Cuda_Half_TestForward ) {
        TestForward<half, Compute::DeviceType::Cuda>( CudaHalfData() );
    }

    TEST_F( ResidualTests, Cuda_Half_TestPrint ) {
        TestPrint<half, Compute::DeviceType::Cuda>( CudaHalfData(), "Residual: cuda_residual_half" );
    }

    TEST_F( ResidualTests, Cuda_Half_TrainingMode ) {
        TestTrainingMode<half, Compute::DeviceType::Cuda>( CudaHalfData(), false );
    }

    // Context Construction Tests
    TEST_F( ResidualTests, Context_Cpu_Float_DeviceType ) {
        TestDeviceType<float, Compute::DeviceType::Cpu>( ContextCpuFloatData() );
    }

    TEST_F( ResidualTests, Context_Cpu_Float_Forward ) {
        TestForward<float, Compute::DeviceType::Cpu>( ContextCpuFloatData() );
    }

    // Device Change Tests
    /*TEST_F( ResidualTests, Cpu_Float_DeviceChange ) {
        TestDeviceChange<float>( CpuFloatData() );
    }*/

    // Edge Case Tests
    TEST_F( ResidualTests, Cpu_Float_EdgeCases ) {
        TestEdgeCases<float, Compute::DeviceType::Cpu>();
    }

    TEST_F( ResidualTests, Cuda_Float_EdgeCases ) {
        TestEdgeCases<float, Compute::DeviceType::Cuda>();
    }

    // CPU-CUDA Equivalence Test
    TEST_F( ResidualTests, CpuCuda_Forward_Output_Equivalence ) {
        TestCpuCudaEquivalence<float>( CpuFloatData(), CudaFloatData() );
    }
}
