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
    struct SoftmaxTestData {
        std::vector<size_t> shape;
        std::shared_ptr<Softmax<TInput, TPrecision, TDevice>> softmax_module;
    };

    class SoftmaxTests : public ::testing::Test {
    protected:
        void SetUp() override {
            batch_size_ = 64;
            cpu_batch_size_ = 4;
            sequence_length_ = 128;
            vocab_size_ = 1024;
            axis_ = -1;

            // CPU test data (float precision)
            cpu_float_data_.shape = { cpu_batch_size_, sequence_length_, vocab_size_ };
            cpu_float_data_.softmax_module = std::make_shared<Softmax<float, float, Compute::DeviceType::Cpu>>( "cpu_softmax_float", axis_ );

            // CUDA test data (float precision)
            cuda_float_data_.shape = { batch_size_, sequence_length_, vocab_size_ };
            cuda_float_data_.softmax_module = std::make_shared<Softmax<float, float, Compute::DeviceType::Cuda>>( "cuda_softmax_float", axis_ );

            // FIXME: FP16 Precision is not supported yet
            // CUDA test data (half precision)
            //cuda_half_data_.shape = { batch_size_, sequence_length_, vocab_size_ };
            //cuda_half_data_.softmax_module = std::make_shared<Softmax<float, half, Compute::DeviceType::Cuda>>( "cuda_softmax_half", axis_ );
        }

        size_t batch_size_{ 0 };
        size_t cpu_batch_size_{ 0 };
        size_t sequence_length_{ 0 };
        size_t vocab_size_{ 0 };
        int64_t axis_{ -1 };

        // Structured test data
        SoftmaxTestData<float, float, Compute::DeviceType::Cpu> cpu_float_data_;
        SoftmaxTestData<float, float, Compute::DeviceType::Cuda> cuda_float_data_;
        //SoftmaxTestData<float, half, Compute::DeviceType::Cuda> cuda_half_data_;
    };

    // Common test function templates
    template<typename TInput, typename TPrecision, Compute::DeviceType TDevice, typename TMemResource>
    void TestGetName( const SoftmaxTestData<TInput, TPrecision, TDevice>& data, const std::string& expected_name ) {
        EXPECT_EQ( data.softmax_module->getName(), expected_name );
    }

    template<typename TInput, typename TPrecision, Compute::DeviceType TDevice, typename TMemResource>
    void TestParameterCount( const SoftmaxTestData<TInput, TPrecision, TDevice>& data, size_t expected_count ) {
        EXPECT_EQ( data.softmax_module->parameterCount(), expected_count );
    }

    template<typename TInput, typename TPrecision, Compute::DeviceType TDevice, typename TMemResource>
    void TestForward( const SoftmaxTestData<TInput, TPrecision, TDevice>& data ) {
        Tensor<TInput, TMemResource> input( data.shape );
        Tensor<TPrecision, TMemResource> output( data.shape );
        random<TInput, TMemResource>( input, -5.0f, 5.0f );
        data.softmax_module->forward( input, output );
        EXPECT_EQ( output.size(), input.size() );

        auto B = output.shape()[ 0 ];
        auto T = output.shape()[ 1 ];
        auto V = output.shape()[ 2 ];

        for ( size_t i = 0; i < B; ++i ) {
            for ( size_t j = 0; j < T; ++j ) {
                // For each (b,t) position, sum the values across the vocabulary dimension
                // Check if the sum is a value close to 1
                auto sum = 0.0f;
                for ( size_t v = 0; v < V; ++v ) {
                    sum += output[ i, j, v ];
                }
                EXPECT_NEAR( sum, 1.0f, 1e-4 );
            }
        }
    }

    template<typename TInput, typename TPrecision, Compute::DeviceType TDevice>
    void TestPrint( const SoftmaxTestData<TInput, TPrecision, TDevice>& data, const std::string& expected_substring ) {
        std::string output = data.softmax_module->toString();
        EXPECT_NE( output.find( expected_substring ), std::string::npos );
    }

    // Add this function to test equivalence of CPU and CUDA outputs
    template<typename TInput, typename TPrecision>
    void TestCpuCudaEquivalence(
        const SoftmaxTestData<TInput, TPrecision, Compute::DeviceType::Cpu>& cpu_data,
        const SoftmaxTestData<TInput, TPrecision, Compute::DeviceType::Cuda>& cuda_data ) {

        // Create a small test shape to make comparison faster
        std::vector<size_t> test_shape = { 2, 4, 8 }; // Small shape for quick verification

        // Create random input data
        Tensor<TInput, Compute::HostMemoryResource> host_input( test_shape );

        // Fill with predictable values between -2.0 and 2.0 to exercise the softmax function
        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input.data()[ i ] = static_cast<TInput>( -2.0 + 4.0 * (static_cast<float>( i ) / host_input.size()) );
        }

        // Create CPU output
        Tensor<TPrecision, Compute::HostMemoryResource> cpu_output( test_shape );
        cpu_data.softmax_module->forward( host_input, cpu_output );

        // Create device input by copying host data
        Tensor<TInput, Compute::DeviceMemoryResource> device_input( test_shape );
        device_input.copyFrom( host_input );

        // Create device output
        Tensor<TPrecision, Compute::DeviceMemoryResource> cuda_output( test_shape );
        cuda_data.softmax_module->forward( device_input, cuda_output );

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

    TEST_F( SoftmaxTests, Cpu_Float_TestName ) {
        TestGetName<float, float, Compute::DeviceType::Cpu, Compute::HostMemoryResource>(
            cpu_float_data_, "cpu_softmax_float" );
    }

    TEST_F( SoftmaxTests, Cpu_Float_ParameterCount ) {
        TestParameterCount<float, float, Compute::DeviceType::Cpu, Compute::HostMemoryResource>(
            cpu_float_data_, 0 );
    }

    TEST_F( SoftmaxTests, Cpu_Float_TestForward ) {
        TestForward<float, float, Compute::DeviceType::Cpu, Compute::HostMemoryResource>( cpu_float_data_ );
    }

    TEST_F( SoftmaxTests, Cpu_Float_TestPrint ) {
        TestPrint<float, float, Compute::DeviceType::Cpu>( cpu_float_data_, "Softmax: cpu_softmax_float" );
    }

    // CUDA Tests with float precision

    TEST_F( SoftmaxTests, Cuda_Float_TestName ) {
        TestGetName<float, float, Compute::DeviceType::Cuda, Compute::DeviceMemoryResource>(
            cuda_float_data_, "cuda_softmax_float" );
    }

    TEST_F( SoftmaxTests, Cuda_Float_ParameterCount ) {
        TestParameterCount<float, float, Compute::DeviceType::Cuda, Compute::DeviceMemoryResource>(
            cuda_float_data_, 0 );
    }

    TEST_F( SoftmaxTests, Cuda_Float_TestForward ) {
        TestForward<float, float, Compute::DeviceType::Cuda, Compute::DeviceMemoryResource>( cuda_float_data_ );
    }

    TEST_F( SoftmaxTests, Cuda_Float_TestPrint ) {
        TestPrint<float, float, Compute::DeviceType::Cuda>( cuda_float_data_, "Softmax: cuda_softmax_float" );
    }

    // CUDA Tests with half precision

    /*TEST_F( SoftmaxTests, Cuda_Half_TestName ) {
        TestGetName<float, half, Compute::DeviceType::Cuda, Compute::DeviceMemoryResource>(
            cuda_half_data_, "cuda_softmax_half" );
    }

    TEST_F( SoftmaxTests, Cuda_Half_ParameterCount ) {
        TestParameterCount<float, half, Compute::DeviceType::Cuda, Compute::DeviceMemoryResource>(
            cuda_half_data_, 0 );
    }

    TEST_F( SoftmaxTests, Cuda_Half_TestForward ) {
        TestForward<float, half, Compute::DeviceType::Cuda, Compute::DeviceMemoryResource>( cuda_half_data_ );
    }

    TEST_F( SoftmaxTests, Cuda_Half_TestPrint ) {
        TestPrint<float, half, Compute::DeviceType::Cuda>( cuda_half_data_, "Softmax: cuda_softmax_half" );
    }*/

    TEST_F( SoftmaxTests, CpuCuda_Forward_Output_Equivalence ) {
        TestCpuCudaEquivalence<float, float>( cpu_float_data_, cuda_float_data_ );
    }
}
