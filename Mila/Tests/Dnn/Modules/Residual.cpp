#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>

import Mila;

namespace Modules::Tests
{
    using namespace Mila::Dnn;

    // Common test data structure that can be reused
    template<typename TInput, typename TPrecision, Compute::DeviceType TDevice>
    struct ResidualTestData {
        std::vector<size_t> shape;
        std::shared_ptr<Residual<TInput, TPrecision, TDevice>> residual_module;
    };

    class ResidualTests : public ::testing::Test {
    protected:
        void SetUp() override {
            batch_size_ = 128;
            cpu_batch_size_ = 4;
            sequence_length_ = 1024;
            channels_ = 768;

            // CPU test data (float precision)
            cpu_float_data_.shape = { cpu_batch_size_, sequence_length_, channels_ };
            cpu_float_data_.residual_module = std::make_shared<Residual<float, float, Compute::DeviceType::Cpu>>( "cpu_residual_float" );

            // CUDA test data (float precision)
            cuda_float_data_.shape = { batch_size_, sequence_length_, channels_ };
            cuda_float_data_.residual_module = std::make_shared<Residual<float, float, Compute::DeviceType::Cuda>>( "cuda_residual_float" );

            // FIXME: FP16 Precision is not supported yet
            // CUDA test data (half precision)
            //cuda_half_data_.shape = { batch_size_, sequence_length_, channels_ };
            //cuda_half_data_.residual_module = std::make_shared<Residual<float, half, Compute::DeviceType::Cuda>>( "cuda_residual_half" );
        }

        size_t batch_size_{ 0 };
        size_t cpu_batch_size_{ 0 };
        size_t sequence_length_{ 0 };
        size_t channels_{ 0 };

        // Structured test data
        ResidualTestData<float, float, Compute::DeviceType::Cpu> cpu_float_data_;
        ResidualTestData<float, float, Compute::DeviceType::Cuda> cuda_float_data_;
        //ResidualTestData<float, half, Compute::DeviceType::Cuda> cuda_half_data_;
    };

    // Common test function templates
    template<typename TInput, typename TPrecision, Compute::DeviceType TDevice, typename TMemResource>
    void TestGetName( const ResidualTestData<TInput, TPrecision, TDevice>& data, const std::string& expected_name ) {
        EXPECT_EQ( data.residual_module->getName(), expected_name );
    }

    template<typename TInput, typename TPrecision, Compute::DeviceType TDevice, typename TMemResource>
    void TestParameterCount( const ResidualTestData<TInput, TPrecision, TDevice>& data, size_t expected_count ) {
        EXPECT_EQ( data.residual_module->parameterCount(), expected_count );
    }

    template<typename TInput, typename TPrecision, Compute::DeviceType TDevice, typename TMemResource>
    void TestForward( const ResidualTestData<TInput, TPrecision, TDevice>& data ) {
        Tensor<TInput, TMemResource> input_a( data.shape, 4.0f );
        Tensor<TInput, TMemResource> input_b( data.shape, 2.0f );
        Tensor<TPrecision, TMemResource> output( data.shape );
        data.residual_module->forward( input_a, input_b, output );
        EXPECT_EQ( output.size(), input_a.size() );
    }

    template<typename TInput, typename TPrecision, Compute::DeviceType TDevice>
    void TestPrint( const ResidualTestData<TInput, TPrecision, TDevice>& data, const std::string& expected_substring ) {
        std::string output = data.residual_module->toString();
        EXPECT_NE( output.find( expected_substring ), std::string::npos );
    }

    // Add this function to test equivalence of CPU and CUDA outputs
    template<typename TInput, typename TPrecision>
    void TestCpuCudaEquivalence(
        const ResidualTestData<TInput, TPrecision, Compute::DeviceType::Cpu>& cpu_data,
        const ResidualTestData<TInput, TPrecision, Compute::DeviceType::Cuda>& cuda_data ) {

        // Create a small test shape to make comparison faster
        std::vector<size_t> test_shape = { 2, 4, 8 }; // Small shape for quick verification

        // Create random input data
        Tensor<TInput, Compute::HostMemoryResource> host_input_a( test_shape );
        Tensor<TInput, Compute::HostMemoryResource> host_input_b( test_shape );

        // Fill with predictable values between -2.0 and 2.0 to exercise the residual function
        for ( size_t i = 0; i < host_input_a.size(); ++i ) {
            host_input_a.data()[ i ] = static_cast<TInput>( -2.0 + 4.0 * (static_cast<float>( i ) / host_input_a.size()) );
            host_input_b.data()[ i ] = static_cast<TInput>( 2.0 - 4.0 * (static_cast<float>( i ) / host_input_b.size()) );
        }

        // Create CPU output
        Tensor<TPrecision, Compute::HostMemoryResource> cpu_output( test_shape );
        cpu_data.residual_module->forward( host_input_a, host_input_b, cpu_output );

        // Create device input by copying host data
        Tensor<TInput, Compute::DeviceMemoryResource> device_input_a( test_shape );
        Tensor<TInput, Compute::DeviceMemoryResource> device_input_b( test_shape );
        device_input_a.copyFrom( host_input_a );
        device_input_b.copyFrom( host_input_b );

        // Create device output
        Tensor<TPrecision, Compute::DeviceMemoryResource> cuda_output( test_shape );
        cuda_data.residual_module->forward( device_input_a, device_input_b, cuda_output );

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

    TEST_F( ResidualTests, Cpu_Float_TestName ) {
        TestGetName<float, float, Compute::DeviceType::Cpu, Compute::HostMemoryResource>(
            cpu_float_data_, "cpu_residual_float" );
    }

    TEST_F( ResidualTests, Cpu_Float_ParameterCount ) {
        TestParameterCount<float, float, Compute::DeviceType::Cpu, Compute::HostMemoryResource>(
            cpu_float_data_, 0 );
    }

    TEST_F( ResidualTests, Cpu_Float_TestForward ) {
        TestForward<float, float, Compute::DeviceType::Cpu, Compute::HostMemoryResource>( cpu_float_data_ );
    }

    TEST_F( ResidualTests, Cpu_Float_TestPrint ) {
        TestPrint<float, float, Compute::DeviceType::Cpu>( cpu_float_data_, "Residual: cpu_residual_float" );
    }

    // CUDA Tests with float precision

    TEST_F( ResidualTests, Cuda_Float_TestName ) {
        TestGetName<float, float, Compute::DeviceType::Cuda, Compute::DeviceMemoryResource>(
            cuda_float_data_, "cuda_residual_float" );
    }

    TEST_F( ResidualTests, Cuda_Float_ParameterCount ) {
        TestParameterCount<float, float, Compute::DeviceType::Cuda, Compute::DeviceMemoryResource>(
            cuda_float_data_, 0 );
    }

    TEST_F( ResidualTests, Cuda_Float_TestForward ) {
        TestForward<float, float, Compute::DeviceType::Cuda, Compute::DeviceMemoryResource>( cuda_float_data_ );
    }

    TEST_F( ResidualTests, Cuda_Float_TestPrint ) {
        TestPrint<float, float, Compute::DeviceType::Cuda>( cuda_float_data_, "Residual: cuda_residual_float" );
    }

    // CUDA Tests with half precision

    /*TEST_F( ResidualTests, Cuda_Half_TestName ) {
        TestGetName<float, half, Compute::DeviceType::Cuda, Compute::DeviceMemoryResource>(
            cuda_half_data_, "cuda_residual_half" );
    }

    TEST_F( ResidualTests, Cuda_Half_ParameterCount ) {
        TestParameterCount<float, half, Compute::DeviceType::Cuda, Compute::DeviceMemoryResource>(
            cuda_half_data_, 0 );
    }

    TEST_F( ResidualTests, Cuda_Half_TestForward ) {
        TestForward<float, half, Compute::DeviceType::Cuda, Compute::DeviceMemoryResource>( cuda_half_data_ );
    }

    TEST_F( ResidualTests, Cuda_Half_TestPrint ) {
        TestPrint<float, half, Compute::DeviceType::Cuda>( cuda_half_data_, "Residual: cuda_residual_half" );
    }*/

    TEST_F( ResidualTests, CpuCuda_Forward_Output_Equivalence ) {
        TestCpuCudaEquivalence<float, float>( cpu_float_data_, cuda_float_data_ );
    }
}
