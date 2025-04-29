#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <memory>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

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
    struct FullyConnectedTestData {
        std::vector<size_t> input_shape;
        std::vector<size_t> output_shape;
        std::shared_ptr<FullyConnected<TPrecision, TDevice>> fc_module;
        size_t input_channels;
        size_t output_channels;
        bool has_bias;
        bool is_training;

        // Make the test data structure self-initializing
        static FullyConnectedTestData Create(
            const std::string& name,
            size_t batch_size,
            size_t sequence_length,
            size_t input_channels,
            size_t output_channels,
            bool has_bias = true,
            bool is_training = false )
        {
            FullyConnectedTestData data;
            data.input_shape = { batch_size, sequence_length, input_channels };
            data.output_shape = { batch_size, sequence_length, output_channels };
            data.input_channels = input_channels;
            data.output_channels = output_channels;
            data.has_bias = has_bias;
            data.is_training = is_training;

            std::string device_str = TDevice == Compute::DeviceType::Cuda ? "CUDA:0" : "CPU";
            data.fc_module = std::make_shared<FullyConnected<TPrecision, TDevice>>(
                name, device_str, input_channels, output_channels, has_bias, is_training );

            return data;
        }

        // Overload for creating with device context
        static FullyConnectedTestData CreateWithContext(
            const std::string& name,
            size_t batch_size,
            size_t sequence_length,
            size_t input_channels,
            size_t output_channels,
            std::shared_ptr<DeviceContext> context,
            bool has_bias = true )
        {
            FullyConnectedTestData data;
            data.input_shape = { batch_size, sequence_length, input_channels };
            data.output_shape = { batch_size, sequence_length, output_channels };
            data.input_channels = input_channels;
            data.output_channels = output_channels;
            data.has_bias = has_bias;
            data.is_training = false;

            data.fc_module = std::make_shared<FullyConnected<TPrecision, TDevice>>(
                name, context, input_channels, output_channels,  has_bias );

            return data;
        }
    };

    class FullyConnectedTests : public ::testing::Test {
    protected:
        void SetUp() override {
            // Initialize test parameters only
            batch_size_ = 64;
            cpu_batch_size_ = 4;
            sequence_length_ = 128;
            input_channels_ = 256;
            output_features_ = 4;
            output_channels_ = output_features_ * input_channels_;
            has_bias_ = true;
            // Modules will be created on demand
        }

        // Factory methods to lazily create test data as needed
        FullyConnectedTestData<float, Compute::DeviceType::Cpu>& CpuFloatData() {
            if ( !cpu_float_data_.fc_module ) {
                cpu_float_data_ = FullyConnectedTestData<float, Compute::DeviceType::Cpu>::Create(
                    "fc_cpu_float", cpu_batch_size_, sequence_length_,
                    input_channels_, output_channels_, has_bias_ );
            }
            return cpu_float_data_;
        }

        FullyConnectedTestData<float, Compute::DeviceType::Cpu>& CpuNoBiasFloatData() {
            if ( !cpu_no_bias_float_data_.fc_module ) {
                cpu_no_bias_float_data_ = FullyConnectedTestData<float, Compute::DeviceType::Cpu>::Create(
                    "fc_cpu_no_bias_float", cpu_batch_size_, sequence_length_,
                    input_channels_, output_channels_, false );
            }
            return cpu_no_bias_float_data_;
        }

        FullyConnectedTestData<float, Compute::DeviceType::Cuda>& CudaFloatData() {
            if ( !cuda_float_data_.fc_module ) {
                cuda_float_data_ = FullyConnectedTestData<float, Compute::DeviceType::Cuda>::Create(
                    "fc_cuda_float", batch_size_, sequence_length_,
                    input_channels_, output_channels_, has_bias_ );
            }
            return cuda_float_data_;
        }

        FullyConnectedTestData<float, Compute::DeviceType::Cuda>& CudaNoBiasFloatData() {
            if ( !cuda_no_bias_float_data_.fc_module ) {
                cuda_no_bias_float_data_ = FullyConnectedTestData<float, Compute::DeviceType::Cuda>::Create(
                    "fc_cuda_no_bias_float", batch_size_, sequence_length_,
                    input_channels_, output_channels_, false );
            }
            return cuda_no_bias_float_data_;
        }

        FullyConnectedTestData<float, Compute::DeviceType::Cpu>& TrainingCpuFloatData() {
            if ( !training_cpu_float_data_.fc_module ) {
                training_cpu_float_data_ = FullyConnectedTestData<float, Compute::DeviceType::Cpu>::Create(
                    "fc_cpu_float_training", cpu_batch_size_, sequence_length_,
                    input_channels_, output_channels_, has_bias_, true );
            }
            return training_cpu_float_data_;
        }

        FullyConnectedTestData<float, Compute::DeviceType::Cuda>& TrainingCudaFloatData() {
            if ( !training_cuda_float_data_.fc_module ) {
                training_cuda_float_data_ = FullyConnectedTestData<float, Compute::DeviceType::Cuda>::Create(
                    "fc_cuda_float_training", batch_size_, sequence_length_,
                    input_channels_, output_channels_, has_bias_, true );
            }
            return training_cuda_float_data_;
        }

        FullyConnectedTestData<float, Compute::DeviceType::Cpu>& ContextCpuFloatData() {
            if ( !context_cpu_float_data_.fc_module ) {
                auto cpu_context = std::make_shared<DeviceContext>( "CPU" );
                context_cpu_float_data_ = FullyConnectedTestData<float, Compute::DeviceType::Cpu>::CreateWithContext(
                    "fc_cpu_context_float", cpu_batch_size_, sequence_length_,
                    input_channels_, output_channels_, cpu_context, has_bias_ );
            }
            return context_cpu_float_data_;
        }

        FullyConnectedTestData<half, Compute::DeviceType::Cuda>& CudaHalfData() {
            if ( !cuda_half_data_.fc_module ) {
                cuda_half_data_ = FullyConnectedTestData<half, Compute::DeviceType::Cuda>::Create(
                    "fc_cuda_half", batch_size_, sequence_length_,
                    input_channels_, output_channels_, has_bias_ );
            }
            return cuda_half_data_;
        }

        FullyConnectedTestData<half, Compute::DeviceType::Cuda>& CudaNoBiasHalfData() {
            if ( !cuda_no_bias_half_data_.fc_module ) {
                cuda_no_bias_half_data_ = FullyConnectedTestData<half, Compute::DeviceType::Cuda>::Create(
                    "fc_cuda_no_bias_half", batch_size_, sequence_length_,
                    input_channels_, output_channels_, false );
            }
            return cuda_no_bias_half_data_;
        }

        FullyConnectedTestData<half, Compute::DeviceType::Cuda>& TrainingCudaHalfData() {
            if ( !training_cuda_half_data_.fc_module ) {
                training_cuda_half_data_ = FullyConnectedTestData<half, Compute::DeviceType::Cuda>::Create(
                    "fc_cuda_half_training", batch_size_, sequence_length_,
                    input_channels_, output_channels_, has_bias_, true );
            }
            return training_cuda_half_data_;
        }

        // Test parameters
        size_t batch_size_{ 0 };
        size_t cpu_batch_size_{ 0 };
        size_t sequence_length_{ 0 };
        size_t input_channels_{ 0 };
        size_t output_channels_{ 0 };
        size_t output_features_{ 0 };
        bool has_bias_{ true };

        // Test data objects - initialized on demand
        FullyConnectedTestData<float, Compute::DeviceType::Cpu> cpu_float_data_;
        FullyConnectedTestData<float, Compute::DeviceType::Cpu> context_cpu_float_data_;
        FullyConnectedTestData<float, Compute::DeviceType::Cpu> cpu_no_bias_float_data_;
        FullyConnectedTestData<float, Compute::DeviceType::Cpu> training_cpu_float_data_;
       
        FullyConnectedTestData<float, Compute::DeviceType::Cuda> cuda_float_data_;
        FullyConnectedTestData<float, Compute::DeviceType::Cuda> cuda_no_bias_float_data_;
        FullyConnectedTestData<float, Compute::DeviceType::Cuda> training_cuda_float_data_;

        FullyConnectedTestData<half, Compute::DeviceType::Cuda> cuda_half_data_;
        FullyConnectedTestData<half, Compute::DeviceType::Cuda> cuda_no_bias_half_data_;
        FullyConnectedTestData<half, Compute::DeviceType::Cuda> training_cuda_half_data_;
    };

    // Test implementations - grouped by functionality

    // Common test function templates
    template<typename TPrecision, Compute::DeviceType TDevice>
    void TestGetName( const FullyConnectedTestData<TPrecision, TDevice>& data, const std::string& expected_name ) {
        EXPECT_EQ( data.fc_module->getName(), expected_name );
    }

    template<typename TPrecision, Compute::DeviceType TDevice>
    void TestParameterCount( const FullyConnectedTestData<TPrecision, TDevice>& data ) {
        auto num_parameters = (data.output_channels * data.input_channels); // weights
        if ( data.has_bias ) {
            num_parameters += data.output_channels; // bias
        }
        EXPECT_EQ( data.fc_module->parameterCount(), num_parameters );
    }

    template<typename TPrecision, Compute::DeviceType TDevice>
    void TestInitializeParameterTensors( const FullyConnectedTestData<TPrecision, TDevice>& data ) {
        auto parameters = data.fc_module->getParameterTensors();
        size_t expected_size = data.has_bias ? 2 : 1; // weights and bias or just weights
        EXPECT_EQ( parameters.size(), expected_size );

        // Validate weight tensor exists
        EXPECT_NE( parameters.find( "weight" ), parameters.end() );

        // Validate bias tensor exists if has_bias is true
        if ( data.has_bias ) {
            EXPECT_NE( parameters.find( "bias" ), parameters.end() );
        }
        else {
            EXPECT_EQ( parameters.find( "bias" ), parameters.end() );
        }
    }

    template<typename TPrecision, Compute::DeviceType TDevice>
    void TestForward( const FullyConnectedTestData<TPrecision, TDevice>& data ) {
        using MR = MemoryResourceType<TPrecision, TDevice>;

        Tensor<TPrecision, MR> input( data.input_shape );
        Tensor<TPrecision, MR> output( data.output_shape );
        data.fc_module->forward( input, output );
        EXPECT_EQ( output.size(), data.output_shape[ 0 ] * data.output_shape[ 1 ] * data.output_shape[ 2 ] );
    }

    template<typename TPrecision, Compute::DeviceType TDevice>
    void TestPrint( const FullyConnectedTestData<TPrecision, TDevice>& data, const std::string& expected_substring ) {
        std::string output = data.fc_module->toString();
        EXPECT_NE( output.find( expected_substring ), std::string::npos );
    }

    template<typename TPrecision, Compute::DeviceType TDevice>
    void TestGetWeight( const FullyConnectedTestData<TPrecision, TDevice>& data ) {
        auto weight = data.fc_module->getWeight();
        EXPECT_NE( weight, nullptr );
        EXPECT_EQ( weight->shape()[ 0 ], data.output_channels );
        EXPECT_EQ( weight->shape()[ 1 ], data.input_channels );
    }

    template<typename TPrecision, Compute::DeviceType TDevice>
    void TestGetBias( const FullyConnectedTestData<TPrecision, TDevice>& data ) {
        auto bias_opt = data.fc_module->getBias();

        if ( data.has_bias ) {
            EXPECT_TRUE( bias_opt.has_value() );
            auto bias = bias_opt.value();
            EXPECT_NE( bias, nullptr );
            EXPECT_EQ( bias->shape()[ 0 ], data.output_channels );
        }
        else {
            EXPECT_FALSE( bias_opt.has_value() );
        }
    }

    template<typename TPrecision, Compute::DeviceType TDevice>
    void TestHasBias( const FullyConnectedTestData<TPrecision, TDevice>& data ) {
        EXPECT_EQ( data.fc_module->hasBias(), data.has_bias );
    }

    template<typename TPrecision, Compute::DeviceType TDevice>
    void TestTrainingMode( const FullyConnectedTestData<TPrecision, TDevice>& data, bool expected_mode ) {
        EXPECT_EQ( data.fc_module->isTraining(), expected_mode );
    }

    template<typename TPrecision, Compute::DeviceType TDevice>
    void TestDeviceType( const FullyConnectedTestData<TPrecision, TDevice>& data ) {
        auto device_context = data.fc_module->getDeviceContext();
        EXPECT_NE( device_context, nullptr );
        auto device = device_context->getDevice();
        EXPECT_NE( device, nullptr );
        EXPECT_EQ( device->getDeviceType(), TDevice );
    }

    // Function to test equivalence of CPU and CUDA outputs
    template<typename TPrecision>
    void TestCpuCudaEquivalence(
        const FullyConnectedTestData<TPrecision, Compute::DeviceType::Cpu>& cpu_data,
        const FullyConnectedTestData<TPrecision, Compute::DeviceType::Cuda>& cuda_data ) {

        // Create a small test shape to make comparison faster
        std::vector<size_t> test_input_shape = { 2, 4, cpu_data.input_channels };
        std::vector<size_t> test_output_shape = { 2, 4, cpu_data.output_channels };

        // Create random input data
        Tensor<TPrecision, Compute::HostMemoryResource> host_input( test_input_shape );

        // Fill with predictable values
        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input.data()[ i ] = static_cast<TPrecision>( -1.0 + 2.0 * (static_cast<float>( i ) / host_input.size()) );
        }

        // Initialize the weights and biases with the same values for both CPU and CUDA modules
        // Copy parameters from CPU module to CUDA module for fair comparison
        auto cpu_params = cpu_data.fc_module->getParameterTensors();
        auto cuda_params = cuda_data.fc_module->getParameterTensors();

        // Copy weights
        Tensor<TPrecision, Compute::CudaMemoryResource> cuda_weights( cpu_params[ "weight" ]->shape() );
        cuda_weights.copyFrom( *cpu_params[ "weight" ] );
        cuda_params[ "weight" ]->copyFrom( cuda_weights );

        // Copy bias if it exists
        if ( cpu_data.has_bias && cuda_data.has_bias ) {
            Tensor<TPrecision, Compute::CudaMemoryResource> cuda_bias( cpu_params[ "bias" ]->shape() );
            cuda_bias.copyFrom( *cpu_params[ "bias" ] );
            cuda_params[ "bias" ]->copyFrom( cuda_bias );
        }

        // Create CPU output
        Tensor<TPrecision, Compute::HostMemoryResource> cpu_output( test_output_shape );
        cpu_data.fc_module->forward( host_input, cpu_output );

        // Create device input by copying host data
        Tensor<TPrecision, Compute::CudaMemoryResource> device_input( test_input_shape );
        device_input.copyFrom( host_input );

        // Create device output
        Tensor<TPrecision, Compute::CudaMemoryResource> cuda_output( test_output_shape );
        cuda_data.fc_module->forward( device_input, cuda_output );

        // Copy CUDA output back to host for comparison
        Tensor<TPrecision, Compute::HostMemoryResource> cuda_output_host( test_output_shape );
        cuda_output_host.copyFrom( cuda_output );

        // Compare outputs with tolerance for floating point differences
        const float epsilon = 1e-4f;
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
            std::vector<size_t> minimal_input_shape = { 1, 1, 8 };
            std::vector<size_t> minimal_output_shape = { 1, 1, 16 };

            auto minimal_module = std::make_shared<FullyConnected<TPrecision, TDevice>>(
                "minimal_fc", TDevice == Compute::DeviceType::Cuda ? "CUDA:0" : "CPU", 8, 16 );

            Tensor<TPrecision, MR> minimal_input( minimal_input_shape );
            Tensor<TPrecision, MR> minimal_output( minimal_output_shape );

            EXPECT_NO_THROW( minimal_module->forward( minimal_input, minimal_output ) );
            EXPECT_EQ( minimal_output.size(), 16 );

            // Test with larger dimensions
            std::vector<size_t> large_input_shape = { 2, 2, 1024 };
            std::vector<size_t> large_output_shape = { 2, 2, 512 };

            auto large_module = std::make_shared<FullyConnected<TPrecision, TDevice>>(
                "large_fc", TDevice == Compute::DeviceType::Cuda ? "CUDA:0" : "CPU", 1024, 512 );

            Tensor<TPrecision, MR> large_input( large_input_shape );
            Tensor<TPrecision, MR> large_output( large_output_shape );

            EXPECT_NO_THROW( large_module->forward( large_input, large_output ) );
            EXPECT_EQ( large_output.size(), 2048 );
        }
        catch ( const std::exception& e ) {
            std::cerr << "Exception during edge case test: " << e.what() << std::endl;
            throw;
        }
    }

    // Test for device change events
    //template<typename TPrecision>
    //void TestDeviceChange( const FullyConnectedTestData<TPrecision, Compute::DeviceType::Cpu>& data ) {
    //    // Create a new device context
    //    auto new_context = std::make_shared<DeviceContext>( "CPU" );

    //    // Store original parameter shapes and sizes
    //    auto original_params = data.fc_module->getParameterTensors();
    //    auto weight_shape = original_params[ "weight" ]->shape();

    //    // Change device
    //    EXPECT_NO_THROW( data.fc_module->setDeviceContext( new_context ) );

    //    // Verify parameters were recreated correctly
    //    auto new_params = data.fc_module->getParameterTensors();
    //    EXPECT_EQ( new_params[ "weight" ]->shape(), weight_shape );

    //    // Verify operations still work after device change
    //    std::vector<size_t> test_input_shape = { 2, 4, data.input_channels };
    //    std::vector<size_t> test_output_shape = { 2, 4, data.output_channels };
    //    Tensor<TPrecision, Compute::HostMemoryResource> input( test_input_shape );
    //    Tensor<TPrecision, Compute::HostMemoryResource> output( test_output_shape );
    //    EXPECT_NO_THROW( data.fc_module->forward( input, output ) );
    //}

    // CPU Tests with float precision
    TEST_F( FullyConnectedTests, Cpu_Float_TestName ) {
        TestGetName<float, Compute::DeviceType::Cpu>( CpuFloatData(), "fc_cpu_float" );
    }

    TEST_F( FullyConnectedTests, Cpu_Float_ParameterCount ) {
        TestParameterCount<float, Compute::DeviceType::Cpu>( CpuFloatData() );
    }

    TEST_F( FullyConnectedTests, Cpu_Float_InitParameters ) {
        TestInitializeParameterTensors<float, Compute::DeviceType::Cpu>( CpuFloatData() );
    }

    TEST_F( FullyConnectedTests, Cpu_Float_Forward ) {
        TestForward<float, Compute::DeviceType::Cpu>( CpuFloatData() );
    }

    TEST_F( FullyConnectedTests, Cpu_Float_Print ) {
        TestPrint<float, Compute::DeviceType::Cpu>( CpuFloatData(), "FullyConnected: fc_cpu_float" );
    }

    TEST_F( FullyConnectedTests, Cpu_Float_GetWeight ) {
        TestGetWeight<float, Compute::DeviceType::Cpu>( CpuFloatData() );
    }

    TEST_F( FullyConnectedTests, Cpu_Float_GetBias ) {
        TestGetBias<float, Compute::DeviceType::Cpu>( CpuFloatData() );
    }

    TEST_F( FullyConnectedTests, Cpu_Float_HasBias ) {
        TestHasBias<float, Compute::DeviceType::Cpu>( CpuFloatData() );
    }

    TEST_F( FullyConnectedTests, Cpu_Float_TrainingMode ) {
        TestTrainingMode<float, Compute::DeviceType::Cpu>( CpuFloatData(), false );
    }

    TEST_F( FullyConnectedTests, Cpu_Float_DeviceType ) {
        TestDeviceType<float, Compute::DeviceType::Cpu>( CpuFloatData() );
    }

    // CPU No Bias Tests
    TEST_F( FullyConnectedTests, Cpu_NoBias_Float_ParameterCount ) {
        TestParameterCount<float, Compute::DeviceType::Cpu>( CpuNoBiasFloatData() );
    }

    TEST_F( FullyConnectedTests, Cpu_NoBias_Float_InitParameters ) {
        TestInitializeParameterTensors<float, Compute::DeviceType::Cpu>( CpuNoBiasFloatData() );
    }

    TEST_F( FullyConnectedTests, Cpu_NoBias_Float_GetBias ) {
        TestGetBias<float, Compute::DeviceType::Cpu>( CpuNoBiasFloatData() );
    }

    TEST_F( FullyConnectedTests, Cpu_NoBias_Float_HasBias ) {
        TestHasBias<float, Compute::DeviceType::Cpu>( CpuNoBiasFloatData() );
    }

    TEST_F( FullyConnectedTests, Cpu_NoBias_Float_Forward ) {
        TestForward<float, Compute::DeviceType::Cpu>( CpuNoBiasFloatData() );
    }

    // CPU Training Mode Tests
    TEST_F( FullyConnectedTests, Cpu_Training_Float_TrainingMode ) {
        TestTrainingMode<float, Compute::DeviceType::Cpu>( TrainingCpuFloatData(), true );
    }

    // CUDA Tests with float precision
    TEST_F( FullyConnectedTests, Cuda_Float_TestName ) {
        TestGetName<float, Compute::DeviceType::Cuda>( CudaFloatData(), "fc_cuda_float" );
    }

    TEST_F( FullyConnectedTests, Cuda_Float_ParameterCount ) {
        TestParameterCount<float, Compute::DeviceType::Cuda>( CudaFloatData() );
    }

    TEST_F( FullyConnectedTests, Cuda_Float_InitParameters ) {
        TestInitializeParameterTensors<float, Compute::DeviceType::Cuda>( CudaFloatData() );
    }

    TEST_F( FullyConnectedTests, Cuda_Float_Forward ) {
        TestForward<float, Compute::DeviceType::Cuda>( CudaFloatData() );
    }

    TEST_F( FullyConnectedTests, Cuda_Float_Print ) {
        TestPrint<float, Compute::DeviceType::Cuda>( CudaFloatData(), "FullyConnected: fc_cuda_float" );
    }

    TEST_F( FullyConnectedTests, Cuda_Float_GetWeight ) {
        TestGetWeight<float, Compute::DeviceType::Cuda>( CudaFloatData() );
    }

    TEST_F( FullyConnectedTests, Cuda_Float_GetBias ) {
        TestGetBias<float, Compute::DeviceType::Cuda>( CudaFloatData() );
    }

    TEST_F( FullyConnectedTests, Cuda_Float_HasBias ) {
        TestHasBias<float, Compute::DeviceType::Cuda>( CudaFloatData() );
    }

    TEST_F( FullyConnectedTests, Cuda_Float_TrainingMode ) {
        TestTrainingMode<float, Compute::DeviceType::Cuda>( CudaFloatData(), false );
    }

    TEST_F( FullyConnectedTests, Cuda_Float_DeviceType ) {
        TestDeviceType<float, Compute::DeviceType::Cuda>( CudaFloatData() );
    }

    // CUDA No Bias Tests
    TEST_F( FullyConnectedTests, Cuda_NoBias_Float_ParameterCount ) {
        TestParameterCount<float, Compute::DeviceType::Cuda>( CudaNoBiasFloatData() );
    }

    TEST_F( FullyConnectedTests, Cuda_NoBias_Float_InitParameters ) {
        TestInitializeParameterTensors<float, Compute::DeviceType::Cuda>( CudaNoBiasFloatData() );
    }

    TEST_F( FullyConnectedTests, Cuda_NoBias_Float_GetBias ) {
        TestGetBias<float, Compute::DeviceType::Cuda>( CudaNoBiasFloatData() );
    }

    TEST_F( FullyConnectedTests, Cuda_NoBias_Float_HasBias ) {
        TestHasBias<float, Compute::DeviceType::Cuda>( CudaNoBiasFloatData() );
    }

    TEST_F( FullyConnectedTests, Cuda_NoBias_Float_Forward ) {
        TestForward<float, Compute::DeviceType::Cuda>( CudaNoBiasFloatData() );
    }

    // CUDA Training Mode Tests
    TEST_F( FullyConnectedTests, Cuda_Training_Float_TrainingMode ) {
        TestTrainingMode<float, Compute::DeviceType::Cuda>( TrainingCudaFloatData(), true );
    }

    // Context Construction Tests
    TEST_F( FullyConnectedTests, Context_Cpu_Float_DeviceType ) {
        TestDeviceType<float, Compute::DeviceType::Cpu>( ContextCpuFloatData() );
    }

    TEST_F( FullyConnectedTests, Context_Cpu_Float_Forward ) {
        TestForward<float, Compute::DeviceType::Cpu>( ContextCpuFloatData() );
    }

    // Device Change Tests
    /*TEST_F( FullyConnectedTests, Cpu_Float_DeviceChange ) {
        TestDeviceChange<float>( CpuFloatData() );
    }*/

    // Edge Case Tests
    TEST_F( FullyConnectedTests, Cpu_Float_EdgeCases ) {
        TestEdgeCases<float, Compute::DeviceType::Cpu>();
    }

    TEST_F( FullyConnectedTests, Cuda_Float_EdgeCases ) {
        TestEdgeCases<float, Compute::DeviceType::Cuda>();
    }

    // CPU-CUDA Equivalence Test
    TEST_F( FullyConnectedTests, Cpu_Cuda_Equivalence_Float ) {
        TestCpuCudaEquivalence<float>( CpuFloatData(), CudaFloatData() );
    }

    TEST_F( FullyConnectedTests, Cpu_Cuda_Equivalence_NoBias_Float ) {
        TestCpuCudaEquivalence<float>( CpuNoBiasFloatData(), CudaNoBiasFloatData() );
    }
}
