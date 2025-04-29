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
    struct SoftmaxTestData {
        std::vector<size_t> shape;
        std::shared_ptr<Softmax<TPrecision, TDevice>> softmax_module;
        int64_t axis;
        bool is_training;

        // Make the test data structure self-initializing
        static SoftmaxTestData Create(
            const std::string& name,
            size_t batch_size,
            size_t sequence_length,
            size_t vocab_size,
            int64_t axis = -1,
            bool is_training = false )
        {
            SoftmaxTestData data;
            data.shape = { batch_size, sequence_length, vocab_size };
            data.axis = axis;
            data.is_training = is_training;

            std::string device_str = TDevice == Compute::DeviceType::Cuda ? "CUDA:0" : "CPU";
            data.softmax_module = std::make_shared<Softmax<TPrecision, TDevice>>(
                name, device_str, axis, is_training );

            return data;
        }

        // Overload for creating with device context
        static SoftmaxTestData CreateWithContext(
            const std::string& name,
            size_t batch_size,
            size_t sequence_length,
            size_t vocab_size,
            std::shared_ptr<DeviceContext> context,
            int64_t axis = -1,
            bool is_training = false )
        {
            SoftmaxTestData data;
            data.shape = { batch_size, sequence_length, vocab_size };
            data.axis = axis;
            data.is_training = is_training;

            data.softmax_module = std::make_shared<Softmax<TPrecision, TDevice>>(
                name, context, axis, is_training );

            return data;
        }
    };

    class SoftmaxTests : public ::testing::Test {
    protected:
        void SetUp() override {
            // Initialize test parameters only
            batch_size_ = 64;
            cpu_batch_size_ = 4;
            sequence_length_ = 128;
            vocab_size_ = 1024;
            axis_ = -1;
            // Modules will be created on demand
        }

        // Factory methods to lazily create test data as needed
        SoftmaxTestData<float, Compute::DeviceType::Cpu>& CpuFloatData() {
            if ( !cpu_float_data_.softmax_module ) {
                cpu_float_data_ = SoftmaxTestData<float, Compute::DeviceType::Cpu>::Create(
                    "cpu_softmax_float", cpu_batch_size_, sequence_length_, vocab_size_, axis_ );
            }
            return cpu_float_data_;
        }

        SoftmaxTestData<float, Compute::DeviceType::Cuda>& CudaFloatData() {
            if ( !cuda_float_data_.softmax_module ) {
                cuda_float_data_ = SoftmaxTestData<float, Compute::DeviceType::Cuda>::Create(
                    "cuda_softmax_float", batch_size_, sequence_length_, vocab_size_, axis_ );
            }
            return cuda_float_data_;
        }

        SoftmaxTestData<float, Compute::DeviceType::Cpu>& TrainingCpuFloatData() {
            if ( !training_cpu_float_data_.softmax_module ) {
                training_cpu_float_data_ = SoftmaxTestData<float, Compute::DeviceType::Cpu>::Create(
                    "cpu_softmax_float_training", cpu_batch_size_, sequence_length_, vocab_size_, axis_, true );
            }
            return training_cpu_float_data_;
        }

        SoftmaxTestData<float, Compute::DeviceType::Cuda>& TrainingCudaFloatData() {
            if ( !training_cuda_float_data_.softmax_module ) {
                training_cuda_float_data_ = SoftmaxTestData<float, Compute::DeviceType::Cuda>::Create(
                    "cuda_softmax_float_training", batch_size_, sequence_length_, vocab_size_, axis_, true );
            }
            return training_cuda_float_data_;
        }

        SoftmaxTestData<float, Compute::DeviceType::Cpu>& ContextCpuFloatData() {
            if ( !context_cpu_float_data_.softmax_module ) {
                auto cpu_context = std::make_shared<DeviceContext>( "CPU" );
                context_cpu_float_data_ = SoftmaxTestData<float, Compute::DeviceType::Cpu>::CreateWithContext(
                    "cpu_context_softmax_float", cpu_batch_size_, sequence_length_, vocab_size_, cpu_context, axis_ );
            }
            return context_cpu_float_data_;
        }

        SoftmaxTestData<half, Compute::DeviceType::Cuda>& CudaHalfData() {
            if ( !cuda_half_data_.softmax_module ) {
                cuda_half_data_ = SoftmaxTestData<half, Compute::DeviceType::Cuda>::Create(
                    "cuda_softmax_half", batch_size_, sequence_length_, vocab_size_, axis_ );
            }
            return cuda_half_data_;
        }

        SoftmaxTestData<half, Compute::DeviceType::Cuda>& TrainingCudaHalfData() {
            if ( !training_cuda_half_data_.softmax_module ) {
                training_cuda_half_data_ = SoftmaxTestData<half, Compute::DeviceType::Cuda>::Create(
                    "cuda_softmax_half_training", batch_size_, sequence_length_, vocab_size_, axis_, true );
            }
            return training_cuda_half_data_;
        }

        // Test parameters
        size_t batch_size_{ 0 };
        size_t cpu_batch_size_{ 0 };
        size_t sequence_length_{ 0 };
        size_t vocab_size_{ 0 };
        int64_t axis_{ -1 };

        // Test data objects - initialized on demand
        SoftmaxTestData<float, Compute::DeviceType::Cpu> cpu_float_data_;
        SoftmaxTestData<float, Compute::DeviceType::Cpu> context_cpu_float_data_;
        SoftmaxTestData<float, Compute::DeviceType::Cpu> training_cpu_float_data_;

        SoftmaxTestData<float, Compute::DeviceType::Cuda> cuda_float_data_;
        SoftmaxTestData<float, Compute::DeviceType::Cuda> training_cuda_float_data_;

        SoftmaxTestData<half, Compute::DeviceType::Cuda> cuda_half_data_;
        SoftmaxTestData<half, Compute::DeviceType::Cuda> training_cuda_half_data_;
    };

    // Common test function templates
    template<typename TPrecision, Compute::DeviceType TDevice>
    void TestGetName( const SoftmaxTestData<TPrecision, TDevice>& data, const std::string& expected_name ) {
        EXPECT_EQ( data.softmax_module->getName(), expected_name );
    }

    template<typename TPrecision, Compute::DeviceType TDevice>
    void TestParameterCount( const SoftmaxTestData<TPrecision, TDevice>& data, size_t expected_count ) {
        EXPECT_EQ( data.softmax_module->parameterCount(), expected_count );
    }

    template<typename TPrecision, Compute::DeviceType TDevice>
    void TestForward( const SoftmaxTestData<TPrecision, TDevice>& data ) {
        using MR = MemoryResourceType<TPrecision, TDevice>;

        Tensor<TPrecision, MR> input( data.shape );
        Tensor<TPrecision, MR> output( data.shape );

        // Fill with random values to test softmax normalization
        random<TPrecision, MR>( input, -5.0f, 5.0f );

        data.softmax_module->forward( input, output );
        EXPECT_EQ( output.size(), input.size() );

        // For each sample, check if softmax values sum to 1 along the specified axis
        if constexpr ( TDevice == Compute::DeviceType::Cpu ) {
            // For CPU tensors, we can directly verify the normalization property
            auto B = output.shape()[ 0 ];
            auto T = output.shape()[ 1 ];
            auto V = output.shape()[ 2 ];

            // The default axis is -1 (last dimension), which is the vocab dimension in our case
            for ( size_t i = 0; i < B; ++i ) {
                for ( size_t j = 0; j < T; ++j ) {
                    // Sum values across the vocabulary dimension
                    float sum = 0.0f;
                    for ( size_t v = 0; v < V; ++v ) {
                        sum += static_cast<float>( output[ i, j, v ] );
                    }
                    EXPECT_NEAR( sum, 1.0f, 1e-4f );
                }
            }
        }
    }

    template<typename TPrecision, Compute::DeviceType TDevice>
    void TestPrint( const SoftmaxTestData<TPrecision, TDevice>& data, const std::string& expected_substring ) {
        std::string output = data.softmax_module->toString();
        EXPECT_NE( output.find( expected_substring ), std::string::npos );
    }

    template<typename TPrecision, Compute::DeviceType TDevice>
    void TestTrainingMode( const SoftmaxTestData<TPrecision, TDevice>& data, bool expected_mode ) {
        EXPECT_EQ( data.softmax_module->isTraining(), expected_mode );
    }

    template<typename TPrecision, Compute::DeviceType TDevice>
    void TestDeviceType( const SoftmaxTestData<TPrecision, TDevice>& data ) {
        auto device_context = data.softmax_module->getDeviceContext();
        EXPECT_NE( device_context, nullptr );
        auto device = device_context->getDevice();
        EXPECT_NE( device, nullptr );
        EXPECT_EQ( device->getDeviceType(), TDevice );
    }

    // Function to test equivalence of CPU and CUDA outputs
    template<typename TPrecision>
    void TestCpuCudaEquivalence(
        const SoftmaxTestData<TPrecision, Compute::DeviceType::Cpu>& cpu_data,
        const SoftmaxTestData<TPrecision, Compute::DeviceType::Cuda>& cuda_data ) {

        // Create a small test shape to make comparison faster
        std::vector<size_t> test_shape = { 2, 4, 8 }; // Small shape for quick verification

        // Create random input data
        Tensor<TPrecision, Compute::HostMemoryResource> host_input( test_shape );

        // Fill with predictable values between -2.0 and 2.0 
        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input.data()[ i ] = static_cast<TPrecision>( -2.0 + 4.0 * (static_cast<float>( i ) / host_input.size()) );
        }

        // Create CPU output
        Tensor<TPrecision, Compute::HostMemoryResource> cpu_output( test_shape );
        cpu_data.softmax_module->forward( host_input, cpu_output );

        // Create device input by copying host data
        Tensor<TPrecision, Compute::CudaMemoryResource> device_input( test_shape );
        device_input.copyFrom( host_input );

        // Create device output
        Tensor<TPrecision, Compute::CudaMemoryResource> cuda_output( test_shape );
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

    // Test with different dimensions (edge cases)
    template<typename TPrecision, Compute::DeviceType TDevice>
    void TestEdgeCases() {
        using MR = MemoryResourceType<TPrecision, TDevice>;

        try {
            // Test with minimal sizes
            std::vector<size_t> minimal_shape = { 1, 1, 8 };

            auto minimal_module = std::make_shared<Softmax<TPrecision, TDevice>>(
                "minimal_softmax", TDevice == Compute::DeviceType::Cuda ? "CUDA:0" : "CPU" );

            Tensor<TPrecision, MR> minimal_input( minimal_shape );
            Tensor<TPrecision, MR> minimal_output( minimal_shape );

            EXPECT_NO_THROW( minimal_module->forward( minimal_input, minimal_output ) );
            EXPECT_EQ( minimal_output.size(), 8 );

            // Test with larger dimensions
            std::vector<size_t> large_shape = { 2, 2, 1024 };

            auto large_module = std::make_shared<Softmax<TPrecision, TDevice>>(
                "large_softmax", TDevice == Compute::DeviceType::Cuda ? "CUDA:0" : "CPU" );

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

    // Test for different axis specifications
    template<typename TPrecision, Compute::DeviceType TDevice>
    void TestDifferentAxes() {
        using MR = MemoryResourceType<TPrecision, TDevice>;

        // Test shape with 3 dimensions
        std::vector<size_t> test_shape = { 2, 3, 4 };

        // Test softmax on axis 0 (batch dimension)
        auto axis0_module = std::make_shared<Softmax<TPrecision, TDevice>>(
            "axis0_softmax", TDevice == Compute::DeviceType::Cuda ? "CUDA:0" : "CPU", 0 );

        Tensor<TPrecision, MR> input( test_shape );
        random<TPrecision, MR>( input, -1.0f, 1.0f );

        Tensor<TPrecision, MR> output0( test_shape );
        EXPECT_NO_THROW( axis0_module->forward( input, output0 ) );

        // Test softmax on axis 1 (sequence dimension)
        auto axis1_module = std::make_shared<Softmax<TPrecision, TDevice>>(
            "axis1_softmax", TDevice == Compute::DeviceType::Cuda ? "CUDA:0" : "CPU", 1 );

        Tensor<TPrecision, MR> output1( test_shape );
        EXPECT_NO_THROW( axis1_module->forward( input, output1 ) );

        // Test softmax on axis 2 (vocab/feature dimension) - this is the default
        auto axis2_module = std::make_shared<Softmax<TPrecision, TDevice>>(
            "axis2_softmax", TDevice == Compute::DeviceType::Cuda ? "CUDA:0" : "CPU", 2 );

        Tensor<TPrecision, MR> output2( test_shape );
        EXPECT_NO_THROW( axis2_module->forward( input, output2 ) );
    }

    // CPU Tests with float precision
    TEST_F( SoftmaxTests, Cpu_Float_TestName ) {
        TestGetName<float, Compute::DeviceType::Cpu>( CpuFloatData(), "cpu_softmax_float" );
    }

    TEST_F( SoftmaxTests, Cpu_Float_ParameterCount ) {
        TestParameterCount<float, Compute::DeviceType::Cpu>( CpuFloatData(), 0 );
    }

    TEST_F( SoftmaxTests, Cpu_Float_TestForward ) {
        TestForward<float, Compute::DeviceType::Cpu>( CpuFloatData() );
    }

    TEST_F( SoftmaxTests, Cpu_Float_TestPrint ) {
        TestPrint<float, Compute::DeviceType::Cpu>( CpuFloatData(), "Softmax: cpu_softmax_float" );
    }

    TEST_F( SoftmaxTests, Cpu_Float_TrainingMode ) {
        TestTrainingMode<float, Compute::DeviceType::Cpu>( CpuFloatData(), false );
    }

    TEST_F( SoftmaxTests, Cpu_Float_DeviceType ) {
        TestDeviceType<float, Compute::DeviceType::Cpu>( CpuFloatData() );
    }

    // CPU Training Mode Tests
    TEST_F( SoftmaxTests, Cpu_Training_Float_TrainingMode ) {
        TestTrainingMode<float, Compute::DeviceType::Cpu>( TrainingCpuFloatData(), true );
    }

    // CUDA Tests with float precision
    TEST_F( SoftmaxTests, Cuda_Float_TestName ) {
        TestGetName<float, Compute::DeviceType::Cuda>( CudaFloatData(), "cuda_softmax_float" );
    }

    TEST_F( SoftmaxTests, Cuda_Float_ParameterCount ) {
        TestParameterCount<float, Compute::DeviceType::Cuda>( CudaFloatData(), 0 );
    }

    TEST_F( SoftmaxTests, Cuda_Float_TestForward ) {
        TestForward<float, Compute::DeviceType::Cuda>( CudaFloatData() );
    }

    TEST_F( SoftmaxTests, Cuda_Float_TestPrint ) {
        TestPrint<float, Compute::DeviceType::Cuda>( CudaFloatData(), "Softmax: cuda_softmax_float" );
    }

    TEST_F( SoftmaxTests, Cuda_Float_TrainingMode ) {
        TestTrainingMode<float, Compute::DeviceType::Cuda>( CudaFloatData(), false );
    }

    TEST_F( SoftmaxTests, Cuda_Float_DeviceType ) {
        TestDeviceType<float, Compute::DeviceType::Cuda>( CudaFloatData() );
    }

    // CUDA Training Mode Tests
    TEST_F( SoftmaxTests, Cuda_Training_Float_TrainingMode ) {
        TestTrainingMode<float, Compute::DeviceType::Cuda>( TrainingCudaFloatData(), true );
    }

    // CUDA Tests with half precision
    TEST_F( SoftmaxTests, Cuda_Half_TestName ) {
        TestGetName<half, Compute::DeviceType::Cuda>( CudaHalfData(), "cuda_softmax_half" );
    }

    TEST_F( SoftmaxTests, Cuda_Half_ParameterCount ) {
        TestParameterCount<half, Compute::DeviceType::Cuda>( CudaHalfData(), 0 );
    }

    TEST_F( SoftmaxTests, Cuda_Half_TestForward ) {
        TestForward<half, Compute::DeviceType::Cuda>( CudaHalfData() );
    }

    TEST_F( SoftmaxTests, Cuda_Half_TestPrint ) {
        TestPrint<half, Compute::DeviceType::Cuda>( CudaHalfData(), "Softmax: cuda_softmax_half" );
    }

    TEST_F( SoftmaxTests, Cuda_Half_TrainingMode ) {
        TestTrainingMode<half, Compute::DeviceType::Cuda>( CudaHalfData(), false );
    }

    // Context Construction Tests
    TEST_F( SoftmaxTests, Context_Cpu_Float_DeviceType ) {
        TestDeviceType<float, Compute::DeviceType::Cpu>( ContextCpuFloatData() );
    }

    TEST_F( SoftmaxTests, Context_Cpu_Float_Forward ) {
        TestForward<float, Compute::DeviceType::Cpu>( ContextCpuFloatData() );
    }

    // Edge Case Tests
    TEST_F( SoftmaxTests, Cpu_Float_EdgeCases ) {
        TestEdgeCases<float, Compute::DeviceType::Cpu>();
    }

    TEST_F( SoftmaxTests, Cuda_Float_EdgeCases ) {
        TestEdgeCases<float, Compute::DeviceType::Cuda>();
    }

    // Axis Tests
    TEST_F( SoftmaxTests, Cpu_Float_DifferentAxes ) {
        TestDifferentAxes<float, Compute::DeviceType::Cpu>();
    }

    TEST_F( SoftmaxTests, Cuda_Float_DifferentAxes ) {
        TestDifferentAxes<float, Compute::DeviceType::Cuda>();
    }

    // CPU-CUDA Equivalence Test
    TEST_F( SoftmaxTests, CpuCuda_Forward_Output_Equivalence ) {
        TestCpuCudaEquivalence<float>( CpuFloatData(), CudaFloatData() );
    }
}
