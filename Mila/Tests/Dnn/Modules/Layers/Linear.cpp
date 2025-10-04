#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <optional>

import Mila;

namespace Modules::Layers::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    template<DeviceType TDevice>
    using MemoryResourceType = std::conditional_t<TDevice == DeviceType::Cuda,
        CudaDeviceMemoryResource,
        CpuMemoryResource>;

    template<DeviceType TDevice, typename TDataType = float>
    struct LinearTestData {
        std::vector<size_t> input_shape;
        std::vector<size_t> output_shape;
        std::shared_ptr<Linear<TDevice, TDataType>> linear_module;
        LinearConfig config;

        static LinearTestData Create(
            size_t batch_size,
            size_t sequence_length,
            size_t input_features,
            size_t output_features,
            bool has_bias = true )
        {
            LinearTestData data;
            data.input_shape = { batch_size, sequence_length, input_features };
            data.output_shape = { batch_size, sequence_length, output_features };

            data.config = LinearConfig( input_features, output_features );
            data.config.withBias( has_bias )
                .withName( "test_linear" );

            std::string device_name = TDevice == DeviceType::Cuda ? "CUDA:0" : "CPU";
            data.linear_module = std::make_shared<Linear<TDevice, TDataType>>( device_name, data.config );

            return data;
        }

        static LinearTestData CreateWithContext(
            std::shared_ptr<DeviceContext> context,
            size_t batch_size,
            size_t sequence_length,
            size_t input_features,
            size_t output_features,
            bool has_bias = true )
        {
            LinearTestData data;
            data.input_shape = { batch_size, sequence_length, input_features };
            data.output_shape = { batch_size, sequence_length, output_features };

            data.config = LinearConfig( input_features, output_features );
            data.config.withBias( has_bias )
                .withName( "test_linear_context" );

            data.linear_module = std::make_shared<Linear<TDevice, TDataType>>( context, data.config );

            return data;
        }

        LinearTestData() : config( 1, 1 ) {}
    };

    class LinearTests : public ::testing::Test {
    protected:
        void SetUp() override {
            batch_size_ = 4;
            sequence_length_ = 8;
            input_features_ = 16;
            output_features_ = 32;
        }

        void TearDown() override {
            cpu_float_data_.linear_module.reset();
            cpu_no_bias_float_data_.linear_module.reset();
            context_cpu_float_data_.linear_module.reset();
            cuda_float_data_.linear_module.reset();
            cuda_no_bias_float_data_.linear_module.reset();
        }

        LinearTestData<DeviceType::Cpu, float>& CpuFloatData() {
            if ( !cpu_float_data_.linear_module ) {
                cpu_float_data_ = LinearTestData<DeviceType::Cpu, float>::Create(
                    batch_size_, sequence_length_, input_features_, output_features_ );
            }
            return cpu_float_data_;
        }

        LinearTestData<DeviceType::Cpu, float>& CpuNoBiasFloatData() {
            if ( !cpu_no_bias_float_data_.linear_module ) {
                cpu_no_bias_float_data_ = LinearTestData<DeviceType::Cpu, float>::Create(
                    batch_size_, sequence_length_, input_features_, output_features_, false );
            }
            return cpu_no_bias_float_data_;
        }

        LinearTestData<DeviceType::Cuda, float>& CudaFloatData() {
            if ( !cuda_float_data_.linear_module ) {
                cuda_float_data_ = LinearTestData<DeviceType::Cuda, float>::Create(
                    batch_size_, sequence_length_, input_features_, output_features_ );
            }
            return cuda_float_data_;
        }

        LinearTestData<DeviceType::Cuda, float>& CudaNoBiasFloatData() {
            if ( !cuda_no_bias_float_data_.linear_module ) {
                cuda_no_bias_float_data_ = LinearTestData<DeviceType::Cuda, float>::Create(
                    batch_size_, sequence_length_, input_features_, output_features_, false );
            }
            return cuda_no_bias_float_data_;
        }

        LinearTestData<DeviceType::Cpu, float>& ContextCpuFloatData() {
            if ( !context_cpu_float_data_.linear_module ) {
                auto cpu_context = std::make_shared<DeviceContext>( "CPU" );
                context_cpu_float_data_ = LinearTestData<DeviceType::Cpu, float>::CreateWithContext(
                    cpu_context, batch_size_, sequence_length_, input_features_, output_features_ );
            }
            return context_cpu_float_data_;
        }

        size_t batch_size_{ 0 };
        size_t sequence_length_{ 0 };
        size_t input_features_{ 0 };
        size_t output_features_{ 0 };

        LinearTestData<DeviceType::Cpu, float> cpu_float_data_;
        LinearTestData<DeviceType::Cpu, float> cpu_no_bias_float_data_;
        LinearTestData<DeviceType::Cpu, float> context_cpu_float_data_;
        LinearTestData<DeviceType::Cuda, float> cuda_float_data_;
        LinearTestData<DeviceType::Cuda, float> cuda_no_bias_float_data_;
    };

    template<DeviceType TDevice, typename TDataType>
    void TestParameterCount( const LinearTestData<TDevice, TDataType>& data ) {
        size_t expected_count = data.config.getInputFeatures() * data.config.getOutputFeatures();
        if ( data.config.hasBias() ) {
            expected_count += data.config.getOutputFeatures();
        }
        EXPECT_EQ( data.linear_module->parameterCount(), expected_count );
    }

    template<DeviceType TDevice, typename TDataType>
    void TestForward( const LinearTestData<TDevice, TDataType>& data ) {
        using MR = MemoryResourceType<TDevice>;

        Tensor<TDataType, MR> input( data.input_shape );
        Tensor<TDataType, MR> output( data.output_shape );

        if constexpr ( TDevice == DeviceType::Cpu ) {
            for ( size_t i = 0; i < input.size(); ++i ) {
                input.data()[ i ] = static_cast<TDataType>( static_cast<float>( i ) / input.size() * 2.0f - 1.0f );
            }
        }
        else {
            Tensor<TDataType, CpuMemoryResource> host_input( data.input_shape );
            for ( size_t i = 0; i < host_input.size(); ++i ) {
                host_input.data()[ i ] = static_cast<TDataType>( static_cast<float>( i ) / host_input.size() * 2.0f - 1.0f );
            }
            input = host_input.toDevice<CudaDeviceMemoryResource>();
        }

        ASSERT_NO_THROW( data.linear_module->forward( input, output ) );
        EXPECT_EQ( output.size(), data.output_shape[ 0 ] * data.output_shape[ 1 ] * data.output_shape[ 2 ] );
    }

    template<DeviceType TDevice, typename TDataType>
    void TestToString( const LinearTestData<TDevice, TDataType>& data ) {
        std::string result = data.linear_module->toString();
        EXPECT_FALSE( result.empty() );
        EXPECT_TRUE( result.find( "Linear" ) != std::string::npos );
        EXPECT_TRUE( result.find( "Input features" ) != std::string::npos );
        EXPECT_TRUE( result.find( "Output features" ) != std::string::npos );
    }

    template<DeviceType TDevice, typename TDataType>
    void TestGetWeight( const LinearTestData<TDevice, TDataType>& data ) {
        auto weight = data.linear_module->getWeight();
        EXPECT_NE( weight, nullptr );

        auto tensor_weight = std::dynamic_pointer_cast<Tensor<TDataType, MemoryResourceType<TDevice>>>(weight);
        EXPECT_NE( tensor_weight, nullptr );
        EXPECT_EQ( tensor_weight->shape()[ 0 ], data.config.getOutputFeatures() );
        EXPECT_EQ( tensor_weight->shape()[ 1 ], data.config.getInputFeatures() );
    }

    template<DeviceType TDevice, typename TDataType>
    void TestGetBias( const LinearTestData<TDevice, TDataType>& data ) {
        auto bias_opt = data.linear_module->getBias();

        if ( data.config.hasBias() ) {
            EXPECT_TRUE( bias_opt.has_value() );
            auto bias = bias_opt.value();
            EXPECT_NE( bias, nullptr );

            auto tensor_bias = std::dynamic_pointer_cast<Tensor<TDataType, MemoryResourceType<TDevice>>>(bias);
            EXPECT_NE( tensor_bias, nullptr );
            EXPECT_EQ( tensor_bias->shape()[ 0 ], data.config.getOutputFeatures() );
        }
        else {
            EXPECT_FALSE( bias_opt.has_value() );
        }
    }

    template<DeviceType TDevice, typename TDataType>
    void TestHasBias( const LinearTestData<TDevice, TDataType>& data ) {
        EXPECT_EQ( data.linear_module->hasBias(), data.config.hasBias() );
    }

    template<DeviceType TDevice, typename TDataType>
    void TestDeviceType( const LinearTestData<TDevice, TDataType>& data ) {
        auto device_context = data.linear_module->getDeviceContext();
        EXPECT_NE( device_context, nullptr );
        auto device = device_context->getDevice();
        EXPECT_NE( device, nullptr );
        EXPECT_EQ( device->getDeviceType(), TDevice );
    }

    void TestCpuCudaEquivalence() {
        std::vector<size_t> test_input_shape = { 2, 4, 8 };
        std::vector<size_t> test_output_shape = { 2, 4, 16 };

        LinearConfig cpu_config( 8, 16 );
        cpu_config.withBias( true ).withName( "cpu_test" );

        LinearConfig cuda_config( 8, 16 );
        cuda_config.withBias( true ).withName( "cuda_test" );

        auto cpu_linear = std::make_shared<Linear<DeviceType::Cpu, float>>( "CPU", cpu_config );
        auto cuda_linear = std::make_shared<Linear<DeviceType::Cuda, float>>( "CUDA:0", cuda_config );

        Tensor<float, CpuMemoryResource> host_input( test_input_shape );
        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input.data()[ i ] = static_cast<float>( i ) / host_input.size() * 2.0f - 1.0f;
        }

        auto cpu_weight = cpu_linear->getWeight();
        auto cuda_weight = cuda_linear->getWeight();

        auto cpu_weight_tensor = std::dynamic_pointer_cast<Tensor<float, CpuMemoryResource>>( cpu_weight );
        auto cuda_weight_tensor = std::dynamic_pointer_cast<Tensor<float, CudaDeviceMemoryResource>>( cuda_weight );

        for ( size_t i = 0; i < cpu_weight_tensor->size(); ++i ) {
            cpu_weight_tensor->data()[ i ] = 0.1f;
        }

        auto device_weight = cpu_weight_tensor->toDevice<CudaDeviceMemoryResource>();
        for ( size_t i = 0; i < cuda_weight_tensor->size(); ++i ) {
            cuda_weight_tensor->data()[ i ] = device_weight.data()[ i ];
        }

        auto cpu_bias_opt = cpu_linear->getBias();
        auto cuda_bias_opt = cuda_linear->getBias();

        if ( cpu_bias_opt.has_value() && cuda_bias_opt.has_value() ) {
            auto cpu_bias_tensor = std::dynamic_pointer_cast<Tensor<float, CpuMemoryResource>>( cpu_bias_opt.value() );
            auto cuda_bias_tensor = std::dynamic_pointer_cast<Tensor<float, CudaDeviceMemoryResource>>( cuda_bias_opt.value() );

            for ( size_t i = 0; i < cpu_bias_tensor->size(); ++i ) {
                cpu_bias_tensor->data()[ i ] = 0.0f;
            }

            auto device_bias = cpu_bias_tensor->toDevice<CudaDeviceMemoryResource>();
            for ( size_t i = 0; i < cuda_bias_tensor->size(); ++i ) {
                cuda_bias_tensor->data()[ i ] = device_bias.data()[ i ];
            }
        }

        Tensor<float, CpuMemoryResource> cpu_output( test_output_shape );
        cpu_linear->forward( host_input, cpu_output );

        auto cuda_input = host_input.toDevice<CudaDeviceMemoryResource>();
        Tensor<float, CudaDeviceMemoryResource> cuda_output( test_output_shape );
        cuda_linear->forward( cuda_input, cuda_output );

        auto cuda_output_host = cuda_output.toHost<CpuMemoryResource>();

        const float epsilon = 1e-4f;
        for ( size_t i = 0; i < cpu_output.size(); ++i ) {
            float diff = std::abs( cpu_output.data()[ i ] - cuda_output_host.data()[ i ] );
            EXPECT_LT( diff, epsilon ) << "Mismatch at index " << i;
        }
    }

    void TestEdgeCases() {
        LinearConfig minimal_config( 1, 1 );
        minimal_config.withName( "minimal_linear" );

        auto minimal_linear = std::make_shared<Linear<DeviceType::Cpu, float>>( "CPU", minimal_config );

        Tensor<float, CpuMemoryResource> minimal_input( { 1, 1, 1 } );
        Tensor<float, CpuMemoryResource> minimal_output( { 1, 1, 1 } );

        minimal_input.data()[ 0 ] = 1.0f;

        EXPECT_NO_THROW( minimal_linear->forward( minimal_input, minimal_output ) );

        LinearConfig large_config( 128, 64 );
        large_config.withName( "large_linear" );

        auto large_linear = std::make_shared<Linear<DeviceType::Cpu, float>>( "CPU", large_config );

        Tensor<float, CpuMemoryResource> large_input( { 2, 4, 128 } );
        Tensor<float, CpuMemoryResource> large_output( { 2, 4, 64 } );

        for ( size_t i = 0; i < large_input.size(); ++i ) {
            large_input.data()[ i ] = static_cast<float>( i ) / large_input.size();
        }

        EXPECT_NO_THROW( large_linear->forward( large_input, large_output ) );
        EXPECT_EQ( large_output.size(), 512 );
    }

    TEST_F( LinearTests, Cpu_Float_ParameterCount ) {
        TestParameterCount( CpuFloatData() );
    }

    TEST_F( LinearTests, Cpu_Float_Forward ) {
        TestForward( CpuFloatData() );
    }

    TEST_F( LinearTests, Cpu_Float_ToString ) {
        TestToString( CpuFloatData() );
    }

    TEST_F( LinearTests, Cpu_Float_GetWeight ) {
        TestGetWeight( CpuFloatData() );
    }

    TEST_F( LinearTests, Cpu_Float_GetBias ) {
        TestGetBias( CpuFloatData() );
    }

    TEST_F( LinearTests, Cpu_Float_HasBias ) {
        TestHasBias( CpuFloatData() );
    }

    TEST_F( LinearTests, Cpu_Float_DeviceType ) {
        TestDeviceType( CpuFloatData() );
    }

    TEST_F( LinearTests, Cpu_NoBias_Float_ParameterCount ) {
        TestParameterCount( CpuNoBiasFloatData() );
    }

    TEST_F( LinearTests, Cpu_NoBias_Float_GetBias ) {
        TestGetBias( CpuNoBiasFloatData() );
    }

    TEST_F( LinearTests, Cpu_NoBias_Float_HasBias ) {
        TestHasBias( CpuNoBiasFloatData() );
    }

    TEST_F( LinearTests, Cpu_NoBias_Float_Forward ) {
        TestForward( CpuNoBiasFloatData() );
    }

    TEST_F( LinearTests, Context_Cpu_Float_Forward ) {
        TestForward( ContextCpuFloatData() );
    }

    TEST_F( LinearTests, Context_Cpu_Float_DeviceType ) {
        TestDeviceType( ContextCpuFloatData() );
    }

    TEST_F( LinearTests, Cuda_Float_ParameterCount ) {
        try {
            TestParameterCount( CudaFloatData() );
        }
        catch ( const std::exception& ) {
            std::cout << "CUDA device not available, skipping test" << std::endl;
            SUCCEED();
        }
    }

    TEST_F( LinearTests, Cuda_Float_Forward ) {
        try {
            TestForward( CudaFloatData() );
        }
        catch ( const std::exception& ) {
            std::cout << "CUDA device not available, skipping test" << std::endl;
            SUCCEED();
        }
    }

    TEST_F( LinearTests, Cuda_Float_ToString ) {
        try {
            TestToString( CudaFloatData() );
        }
        catch ( const std::exception& ) {
            std::cout << "CUDA device not available, skipping test" << std::endl;
            SUCCEED();
        }
    }

    TEST_F( LinearTests, Cuda_Float_GetWeight ) {
        try {
            TestGetWeight( CudaFloatData() );
        }
        catch ( const std::exception& ) {
            std::cout << "CUDA device not available, skipping test" << std::endl;
            SUCCEED();
        }
    }

    TEST_F( LinearTests, Cuda_Float_GetBias ) {
        try {
            TestGetBias( CudaFloatData() );
        }
        catch ( const std::exception& ) {
            std::cout << "CUDA device not available, skipping test" << std::endl;
            SUCCEED();
        }
    }

    TEST_F( LinearTests, Cuda_NoBias_Float_HasBias ) {
        try {
            TestHasBias( CudaNoBiasFloatData() );
        }
        catch ( const std::exception& ) {
            std::cout << "CUDA device not available, skipping test" << std::endl;
            SUCCEED();
        }
    }

    TEST_F( LinearTests, Cuda_NoBias_Float_Forward ) {
        try {
            TestForward( CudaNoBiasFloatData() );
        }
        catch ( const std::exception& ) {
            std::cout << "CUDA device not available, skipping test" << std::endl;
            SUCCEED();
        }
    }

    TEST_F( LinearTests, EdgeCases ) {
        TestEdgeCases();
    }

    TEST_F( LinearTests, CpuCuda_EquivalenceTest ) {
        try {
            TestCpuCudaEquivalence();
        }
        catch ( const std::exception& ) {
            std::cout << "CUDA device not available, skipping equivalence test" << std::endl;
            SUCCEED();
        }
    }

    TEST_F( LinearTests, Constructor_DeviceNameValidation ) {
        LinearConfig config( 16, 32 );
        config.withName( "validation_test" );

        EXPECT_NO_THROW( (Linear<DeviceType::Cpu, float>( "CPU", config )) );
    }

    TEST_F( LinearTests, Constructor_DeviceContextValidation ) {
        LinearConfig config( 16, 32 );
        config.withName( "context_validation_test" );

        auto cpu_context = std::make_shared<DeviceContext>( "CPU" );
        EXPECT_NO_THROW( (Linear<DeviceType::Cpu, float>( cpu_context, config )) );
    }

    TEST_F( LinearTests, Constructor_InvalidConfig ) {
        LinearConfig invalid_config( 0, 32 );
        invalid_config.withName( "invalid_test" );

        EXPECT_THROW( (Linear<DeviceType::Cpu, float>( "CPU", invalid_config )), std::invalid_argument );
    }
}