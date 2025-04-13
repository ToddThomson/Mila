#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>

import Mila;

namespace Modules::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    // Common test data structure that can be reused
    template<typename TInput, typename TPrecision>
    struct ResidualTestData {
        std::vector<size_t> shape;
        std::shared_ptr<Residual<TInput, TPrecision>> residual_module;
        std::shared_ptr<DeviceContext> device_context;
    };

    class ResidualTests : public ::testing::Test {
    protected:
        void SetUp() override {
            batch_size_ = 128;
            cpu_batch_size_ = 4;
            sequence_length_ = 1024;
            channels_ = 768;

            // Create device contexts
            cpu_device_context_ = std::make_shared<DeviceContext>( "CPU" );
            cuda_device_context_ = std::make_shared<DeviceContext>( "CUDA:0" );

            // CPU test data (float precision)
            cpu_float_data_.shape = { cpu_batch_size_, sequence_length_, channels_ };
            cpu_float_data_.device_context = cpu_device_context_;
            cpu_float_data_.residual_module = std::make_shared<Residual<float, float>>(
                "cpu_residual_float", cpu_device_context_ );

            // CUDA test data (float precision)
            cuda_float_data_.shape = { batch_size_, sequence_length_, channels_ };
            cuda_float_data_.device_context = cuda_device_context_;
            cuda_float_data_.residual_module = std::make_shared<Residual<float, float>>(
                "cuda_residual_float", cuda_device_context_ );

            // FIXME: FP16 Precision is not supported yet
            // CUDA test data (half precision)
            //cuda_half_data_.shape = { batch_size_, sequence_length_, channels_ };
            //cuda_half_data_.device_context = cuda_device_context_;
            //cuda_half_data_.residual_module = std::make_shared<Residual<float, half>>(
            //    "cuda_residual_half", cuda_device_context_);
        }

        size_t batch_size_{ 0 };
        size_t cpu_batch_size_{ 0 };
        size_t sequence_length_{ 0 };
        size_t channels_{ 0 };

        std::shared_ptr<DeviceContext> cpu_device_context_;
        std::shared_ptr<DeviceContext> cuda_device_context_;

        // Structured test data
        ResidualTestData<float, float> cpu_float_data_;
        ResidualTestData<float, float> cuda_float_data_;
        //ResidualTestData<float, half> cuda_half_data_;
    };

    // Common test function templates
    template<typename TInput, typename TPrecision, typename TMemResource>
    void TestGetName( const ResidualTestData<TInput, TPrecision>& data, const std::string& expected_name ) {
        EXPECT_EQ( data.residual_module->getName(), expected_name );
    }

    template<typename TInput, typename TPrecision, typename TMemResource>
    void TestParameterCount( const ResidualTestData<TInput, TPrecision>& data, size_t expected_count ) {
        EXPECT_EQ( data.residual_module->parameterCount(), expected_count );
    }

    template<typename TInput, typename TPrecision, typename TMemResource>
    void TestForward( const ResidualTestData<TInput, TPrecision>& data ) {
        Tensor<TInput, TMemResource> input_a( data.shape, 4.0f );
        Tensor<TInput, TMemResource> input_b( data.shape, 2.0f );
        Tensor<TPrecision, TMemResource> output( data.shape );
        data.residual_module->forward( input_a, input_b, output );
        EXPECT_EQ( output.size(), input_a.size() );
    }

    template<typename TInput, typename TPrecision>
    void TestPrint( const ResidualTestData<TInput, TPrecision>& data, const std::string& expected_substring ) {
        std::string output = data.residual_module->toString();
        EXPECT_NE( output.find( expected_substring ), std::string::npos );
    }

    // Add this function to test equivalence of CPU and CUDA outputs
    template<typename TInput, typename TPrecision>
    void TestCpuCudaEquivalence(
        const ResidualTestData<TInput, TPrecision>& cpu_data,
        const ResidualTestData<TInput, TPrecision>& cuda_data ) {

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
        Tensor<TPrecision, HostMemoryResource> cpu_output( test_shape );
        cpu_data.residual_module->forward( host_input_a, host_input_b, cpu_output );

        // Create device input by copying host data
        Tensor<TInput, DeviceMemoryResource> device_input_a( test_shape );
        Tensor<TInput, DeviceMemoryResource> device_input_b( test_shape );
        device_input_a.copyFrom( host_input_a );
        device_input_b.copyFrom( host_input_b );

        // Create device output
        Tensor<TPrecision, DeviceMemoryResource> cuda_output( test_shape );
        cuda_data.residual_module->forward( device_input_a, device_input_b, cuda_output );

        // Copy CUDA output back to host for comparison
        Tensor<TPrecision, HostMemoryResource> cuda_output_host( test_shape );
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
        TestGetName<float, float, HostMemoryResource>(
            cpu_float_data_, "cpu_residual_float" );
    }

    TEST_F( ResidualTests, Cpu_Float_ParameterCount ) {
        TestParameterCount<float, float, HostMemoryResource>(
            cpu_float_data_, 0 );
    }

    TEST_F( ResidualTests, Cpu_Float_TestForward ) {
        TestForward<float, float, HostMemoryResource>( cpu_float_data_ );
    }

    TEST_F( ResidualTests, Cpu_Float_TestPrint ) {
        TestPrint<float, float>( cpu_float_data_, "Residual: cpu_residual_float" );
    }

    // CUDA Tests with float precision

    TEST_F( ResidualTests, Cuda_Float_TestName ) {
        TestGetName<float, float, DeviceMemoryResource>(
            cuda_float_data_, "cuda_residual_float" );
    }

    TEST_F( ResidualTests, Cuda_Float_ParameterCount ) {
        TestParameterCount<float, float, DeviceMemoryResource>(
            cuda_float_data_, 0 );
    }

    TEST_F( ResidualTests, Cuda_Float_TestForward ) {
        TestForward<float, float, DeviceMemoryResource>( cuda_float_data_ );
    }

    TEST_F( ResidualTests, Cuda_Float_TestPrint ) {
        TestPrint<float, float>( cuda_float_data_, "Residual: cuda_residual_float" );
    }

    // CUDA Tests with half precision

    /*TEST_F(ResidualTests, Cuda_Half_TestName) {
        TestGetName<float, half, DeviceMemoryResource>(
            cuda_half_data_, "cuda_residual_half");
    }

    TEST_F(ResidualTests, Cuda_Half_ParameterCount) {
        TestParameterCount<float, half, DeviceMemoryResource>(
            cuda_half_data_, 0);
    }

    TEST_F(ResidualTests, Cuda_Half_TestForward) {
        TestForward<float, half, DeviceMemoryResource>(cuda_half_data_);
    }

    TEST_F(ResidualTests, Cuda_Half_TestPrint) {
        TestPrint<float, half>(cuda_half_data_, "Residual: cuda_residual_half");
    }*/

    TEST_F( ResidualTests, CpuCuda_Forward_Output_Equivalence ) {
        TestCpuCudaEquivalence<float, float>( cpu_float_data_, cuda_float_data_ );
    }
}