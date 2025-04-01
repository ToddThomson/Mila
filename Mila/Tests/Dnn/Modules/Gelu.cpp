#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <cuda_fp16.h>  // For half type

import Mila;

namespace Modules::Tests
{
    using namespace Mila::Dnn;

    // Common test data structure that can be reused
    template<typename TInput, typename TPrecision, Compute::DeviceType TDevice>
    struct GeluTestData {
        std::vector<size_t> shape;
        std::shared_ptr<Gelu<TInput, TPrecision, TDevice>> gelu_module;
    };

    class GeluTests : public ::testing::Test {
    protected:
        void SetUp() override {
            batch_size_ = 64;
            cpu_batch_size_ = 4;
            sequence_length_ = 1024;
            channels_ = 768;

            // CPU test data (float precision)
            cpu_float_data_.shape = { cpu_batch_size_, sequence_length_, 4 * channels_ };
            cpu_float_data_.gelu_module = std::make_shared<Gelu<float, float, Compute::DeviceType::Cpu>>( "cpu_gelu_float" );

            // CUDA test data (float precision)
            cuda_float_data_.shape = { batch_size_, sequence_length_, 4 * channels_ };
            cuda_float_data_.gelu_module = std::make_shared<Gelu<float, float, Compute::DeviceType::Cuda>>( "cuda_gelu_float" );

			// FIXME: FP16 Precision is not supported yet
            // CUDA test data (half precision)
            //cuda_half_data_.shape = { batch_size_, sequence_length_, 4 * channels_ };
            //cuda_half_data_.gelu_module = std::make_shared<Gelu<float, half, Compute::DeviceType::Cuda>>( "cuda_gelu_half" );
        }

        size_t batch_size_{ 0 };
        size_t cpu_batch_size_{ 0 };
        size_t sequence_length_{ 0 };
        size_t channels_{ 0 };

        // Structured test data
        GeluTestData<float, float, Compute::DeviceType::Cpu> cpu_float_data_;
        GeluTestData<float, float, Compute::DeviceType::Cuda> cuda_float_data_;
        //GeluTestData<float, half, Compute::DeviceType::Cuda> cuda_half_data_;
    };

    // Common test function templates
    template<typename TInput, typename TPrecision, Compute::DeviceType TDevice, typename TMemResource>
    void TestGetName( const GeluTestData<TInput, TPrecision, TDevice>& data, const std::string& expected_name ) {
        EXPECT_EQ( data.gelu_module->getName(), expected_name );
    }

    template<typename TInput, typename TPrecision, Compute::DeviceType TDevice, typename TMemResource>
    void TestParameterCount( const GeluTestData<TInput, TPrecision, TDevice>& data, size_t expected_count ) {
        EXPECT_EQ( data.gelu_module->parameterCount(), expected_count );
    }

    template<typename TInput, typename TPrecision, Compute::DeviceType TDevice, typename TMemResource>
    void TestForward( const GeluTestData<TInput, TPrecision, TDevice>& data ) {
        Tensor<TInput, TMemResource> input( data.shape );
        Tensor<TPrecision, TMemResource> output( data.shape );
        data.gelu_module->forward( input, output );
        EXPECT_EQ( output.size(), input.size() );
    }

    template<typename TInput, typename TPrecision, Compute::DeviceType TDevice>
    void TestPrint( const GeluTestData<TInput, TPrecision, TDevice>& data, const std::string& expected_substring ) {
        std::string output = data.gelu_module->toString();
        EXPECT_NE( output.find( expected_substring ), std::string::npos );
    }

    // Add this function to test equivalence of CPU and CUDA outputs
    template<typename TInput, typename TPrecision>
    void TestCpuCudaEquivalence(
        const GeluTestData<TInput, TPrecision, Compute::DeviceType::Cpu>& cpu_data,
        const GeluTestData<TInput, TPrecision, Compute::DeviceType::Cuda>& cuda_data ) {

        // Create a small test shape to make comparison faster
        std::vector<size_t> test_shape = { 2, 4, 8 }; // Small shape for quick verification

        // Create random input data
        Tensor<TInput, Compute::HostMemoryResource> host_input( test_shape );

        // Fill with predictable values between -2.0 and 2.0 to exercise the GELU function
        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input.data()[ i ] = static_cast<TInput>( -2.0 + 4.0 * (static_cast<float>( i ) / host_input.size()) );
        }

        // Create CPU output
        Tensor<TPrecision, Compute::HostMemoryResource> cpu_output( test_shape );
        cpu_data.gelu_module->forward( host_input, cpu_output );

        // Create device input by copying host data
        Tensor<TInput, Compute::DeviceMemoryResource> device_input( test_shape );
        device_input.copyFrom( host_input );

        // Create device output
        Tensor<TPrecision, Compute::DeviceMemoryResource> cuda_output( test_shape );
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


    // CPU Tests with float precision

    TEST_F( GeluTests, Cpu_Float_TestName ) {
        TestGetName<float, float, Compute::DeviceType::Cpu, Compute::HostMemoryResource>(
            cpu_float_data_, "cpu_gelu_float" );
    }

    TEST_F( GeluTests, Cpu_Float_ParameterCount ) {
        TestParameterCount<float, float, Compute::DeviceType::Cpu, Compute::HostMemoryResource>(
            cpu_float_data_, 0 );
    }

    TEST_F( GeluTests, Cpu_Float_TestForward ) {
        TestForward<float, float, Compute::DeviceType::Cpu, Compute::HostMemoryResource>( cpu_float_data_ );
    }

    TEST_F( GeluTests, Cpu_Float_TestPrint ) {
        TestPrint<float, float, Compute::DeviceType::Cpu>( cpu_float_data_, "Gelu: cpu_gelu_float" );
    }

    // CUDA Tests with float precision

    TEST_F( GeluTests, Cuda_Float_TestName ) {
        TestGetName<float, float, Compute::DeviceType::Cuda, Compute::DeviceMemoryResource>(
            cuda_float_data_, "cuda_gelu_float" );
    }

    TEST_F( GeluTests, Cuda_Float_ParameterCount ) {
        TestParameterCount<float, float, Compute::DeviceType::Cuda, Compute::DeviceMemoryResource>(
            cuda_float_data_, 0 );
    }

    TEST_F( GeluTests, Cuda_Float_TestForward ) {
        TestForward<float, float, Compute::DeviceType::Cuda, Compute::DeviceMemoryResource>( cuda_float_data_ );
    }

    TEST_F( GeluTests, Cuda_Float_TestPrint ) {
        TestPrint<float, float, Compute::DeviceType::Cuda>( cuda_float_data_, "Gelu: cuda_gelu_float" );
    }

    // CUDA Tests with half precision

    /*TEST_F( GeluTests, Cuda_Half_TestName ) {
        TestGetName<float, half, Compute::DeviceType::Cuda, Compute::DeviceMemoryResource>(
            cuda_half_data_, "cuda_gelu_half" );
    }

    TEST_F( GeluTests, Cuda_Half_ParameterCount ) {
        TestParameterCount<float, half, Compute::DeviceType::Cuda, Compute::DeviceMemoryResource>(
            cuda_half_data_, 0 );
    }

    TEST_F( GeluTests, Cuda_Half_TestForward ) {
        TestForward<float, half, Compute::DeviceType::Cuda, Compute::DeviceMemoryResource>( cuda_half_data_ );
    }

    TEST_F( GeluTests, Cuda_Half_TestPrint ) {
        TestPrint<float, half, Compute::DeviceType::Cuda>( cuda_half_data_, "Gelu: cuda_gelu_half" );
    }*/

    TEST_F( GeluTests, CpuCuda_Forward_Output_Equivalence ) {
        TestCpuCudaEquivalence<float, float>( cpu_float_data_, cuda_float_data_ );
    }
}
