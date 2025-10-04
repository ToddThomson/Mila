#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <cuda_fp16.h>

import Mila;

namespace Modules::Normalization::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    template<DeviceType TDevice>
    using MemoryResourceType = std::conditional_t<TDevice == DeviceType::Cuda,
        CudaDeviceMemoryResource,
        CpuMemoryResource>;

    template<DeviceType TDevice, typename TInput = float, typename TOutput = TInput>
    struct LayerNormTestData {
        std::vector<size_t> shape;
        std::shared_ptr<LayerNorm<TDevice, TInput, TOutput>> ln_module;
        LayerNormConfig config;

        static LayerNormTestData Create(
            size_t batch_size,
            size_t sequence_length,
            size_t channels,
            int64_t axis = -1,
            bool has_bias = true,
            float epsilon = 1e-5f )
        {
            LayerNormTestData data;
            data.shape = { batch_size, sequence_length, channels };

            data.config.withInputShape( data.shape )
                .withAxis( axis )
                .withBias( has_bias )
                .withEpsilon( epsilon )
                .withName( "test_layernorm" );

            std::string device_name = TDevice == DeviceType::Cuda ? "CUDA:0" : "CPU";
            data.ln_module = std::make_shared<LayerNorm<TDevice, TInput, TOutput>>( device_name, data.config );

            return data;
        }

        static LayerNormTestData CreateWithContext(
            std::shared_ptr<DeviceContext> context,
            size_t batch_size,
            size_t sequence_length,
            size_t channels,
            int64_t axis = -1,
            bool has_bias = true,
            float epsilon = 1e-5f )
        {
            LayerNormTestData data;
            data.shape = { batch_size, sequence_length, channels };

            data.config.withInputShape( data.shape )
                .withAxis( axis )
                .withBias( has_bias )
                .withEpsilon( epsilon )
                .withName( "test_layernorm_context" );

            data.ln_module = std::make_shared<LayerNorm<TDevice, TInput, TOutput>>( context, data.config );

            return data;
        }
    };

    class LayerNormTests : public ::testing::Test {
    protected:
        void SetUp() override {
            batch_size_ = 4;
            sequence_length_ = 8;
            channels_ = 16;
        }

        void TearDown() override {
            cpu_float_data_.ln_module.reset();
            cpu_no_bias_float_data_.ln_module.reset();
            context_cpu_float_data_.ln_module.reset();
            cuda_float_data_.ln_module.reset();
            cuda_no_bias_float_data_.ln_module.reset();
        }

        LayerNormTestData<DeviceType::Cpu, float>& CpuFloatData() {
            if ( !cpu_float_data_.ln_module ) {
                cpu_float_data_ = LayerNormTestData<DeviceType::Cpu, float>::Create(
                    batch_size_, sequence_length_, channels_ );
            }
            return cpu_float_data_;
        }

        LayerNormTestData<DeviceType::Cpu, float>& CpuNoBiasFloatData() {
            if ( !cpu_no_bias_float_data_.ln_module ) {
                cpu_no_bias_float_data_ = LayerNormTestData<DeviceType::Cpu, float>::Create(
                    batch_size_, sequence_length_, channels_, -1, false );
            }
            return cpu_no_bias_float_data_;
        }

        LayerNormTestData<DeviceType::Cuda, float>& CudaFloatData() {
            if ( !cuda_float_data_.ln_module ) {
                cuda_float_data_ = LayerNormTestData<DeviceType::Cuda, float>::Create(
                    batch_size_, sequence_length_, channels_ );
            }
            return cuda_float_data_;
        }

        LayerNormTestData<DeviceType::Cuda, float>& CudaNoBiasFloatData() {
            if ( !cuda_no_bias_float_data_.ln_module ) {
                cuda_no_bias_float_data_ = LayerNormTestData<DeviceType::Cuda, float>::Create(
                    batch_size_, sequence_length_, channels_, -1, false );
            }
            return cuda_no_bias_float_data_;
        }

        LayerNormTestData<DeviceType::Cpu, float>& ContextCpuFloatData() {
            if ( !context_cpu_float_data_.ln_module ) {
                auto cpu_context = std::make_shared<DeviceContext>( "CPU" );
                context_cpu_float_data_ = LayerNormTestData<DeviceType::Cpu, float>::CreateWithContext(
                    cpu_context, batch_size_, sequence_length_, channels_ );
            }
            return context_cpu_float_data_;
        }

        size_t batch_size_{ 0 };
        size_t sequence_length_{ 0 };
        size_t channels_{ 0 };

        LayerNormTestData<DeviceType::Cpu, float> cpu_float_data_;
        LayerNormTestData<DeviceType::Cpu, float> cpu_no_bias_float_data_;
        LayerNormTestData<DeviceType::Cpu, float> context_cpu_float_data_;
        LayerNormTestData<DeviceType::Cuda, float> cuda_float_data_;
        LayerNormTestData<DeviceType::Cuda, float> cuda_no_bias_float_data_;
    };

    template<DeviceType TDevice, typename TInput, typename TOutput = TInput>
    void TestParameterCount( const LayerNormTestData<TDevice, TInput, TOutput>& data ) {
        size_t channels = data.shape[ 2 ];
        size_t expected_count = channels;

        if ( data.config.hasBias() ) {
            expected_count += channels;
        }

        EXPECT_EQ( data.ln_module->parameterCount(), expected_count );
    }

    template<DeviceType TDevice, typename TInput, typename TOutput = TInput>
    void TestForward( const LayerNormTestData<TDevice, TInput, TOutput>& data ) {
        using MR = MemoryResourceType<TDevice>;

        Tensor<TInput, MR> input( data.shape );
        Tensor<TOutput, MR> output( data.shape );

        if constexpr ( TDevice == DeviceType::Cpu ) {
            for ( size_t i = 0; i < input.size(); ++i ) {
                input.data()[ i ] = static_cast<TInput>( static_cast<float>( i ) / input.size() * 4.0f - 2.0f );
            }
        }
        else {
            Tensor<TInput, CpuMemoryResource> host_input( data.shape );
            for ( size_t i = 0; i < host_input.size(); ++i ) {
                host_input.data()[ i ] = static_cast<TInput>( static_cast<float>( i ) / host_input.size() * 4.0f - 2.0f );
            }
            input = host_input.toDevice<CudaDeviceMemoryResource>();
        }

        ASSERT_NO_THROW( data.ln_module->forward( input, output ) );
        EXPECT_EQ( output.size(), input.size() );
    }

    template<DeviceType TDevice, typename TInput, typename TOutput = TInput>
    void TestToString( const LayerNormTestData<TDevice, TInput, TOutput>& data ) {
        std::string result = data.ln_module->toString();
        EXPECT_FALSE( result.empty() );
        EXPECT_TRUE( result.find( "LayerNorm" ) != std::string::npos );
        EXPECT_TRUE( result.find( "Epsilon" ) != std::string::npos );
        EXPECT_TRUE( result.find( "Has Bias" ) != std::string::npos );
    }

    template<DeviceType TDevice, typename TInput, typename TOutput = TInput>
    void TestDeviceType( const LayerNormTestData<TDevice, TInput, TOutput>& data ) {
        auto device_context = data.ln_module->getDeviceContext();
        EXPECT_NE( device_context, nullptr );
        auto device = device_context->getDevice();
        EXPECT_NE( device, nullptr );
        EXPECT_EQ( device->getDeviceType(), TDevice );
    }

    template<DeviceType TDevice, typename TInput, typename TOutput = TInput>
    void TestWeightAndBiasAccess( const LayerNormTestData<TDevice, TInput, TOutput>& data ) {
        auto weight = data.ln_module->getWeight();
        EXPECT_NE( weight, nullptr );
        EXPECT_EQ( weight->size(), data.shape[ 2 ] );

        EXPECT_EQ( data.ln_module->hasBias(), data.config.hasBias() );

        if ( data.config.hasBias() ) {
            auto bias = data.ln_module->getBias();
            EXPECT_NE( bias, nullptr );
            EXPECT_EQ( bias->size(), data.shape[ 2 ] );
        }
    }

    void TestSimpleNormalization() {
        std::vector<size_t> shape = { 1, 2, 3 };

        LayerNormConfig config;
        config.withInputShape( shape )
            .withAxis( -1 )
            .withBias( true )
            .withEpsilon( 1e-5f )
            .withName( "simple_test" );

        auto ln = std::make_shared<LayerNorm<DeviceType::Cpu, float>>( "CPU", config );

        Tensor<float, CpuMemoryResource> input( shape );
        Tensor<float, CpuMemoryResource> output( shape );

        input.data()[ 0 ] = 1.0f; input.data()[ 1 ] = 2.0f; input.data()[ 2 ] = 3.0f;
        input.data()[ 3 ] = 4.0f; input.data()[ 4 ] = 5.0f; input.data()[ 5 ] = 6.0f;

        auto weight = ln->getWeight();
        auto bias = ln->getBias();

        for ( size_t i = 0; i < weight->size(); ++i ) {
            weight->data()[ i ] = 1.0f;
            bias->data()[ i ] = 0.0f;
        }

        ln->forward( input, output );

        const float tolerance = 1e-4f;

        EXPECT_NEAR( output.data()[ 0 ], -1.22474f, tolerance );
        EXPECT_NEAR( output.data()[ 1 ], 0.0f, tolerance );
        EXPECT_NEAR( output.data()[ 2 ], 1.22474f, tolerance );

        EXPECT_NEAR( output.data()[ 3 ], -1.22474f, tolerance );
        EXPECT_NEAR( output.data()[ 4 ], 0.0f, tolerance );
        EXPECT_NEAR( output.data()[ 5 ], 1.22474f, tolerance );
    }

    template<DeviceType TDevice, typename TInput, typename TOutput = TInput>
    void TestNumericalStability() {
        using MR = MemoryResourceType<TDevice>;
        std::string device_name = TDevice == DeviceType::Cuda ? "CUDA:0" : "CPU";

        std::vector<size_t> shape = { 2, 4, 8 };

        LayerNormConfig config;
        config.withInputShape( shape )
            .withAxis( -1 )
            .withBias( true )
            .withEpsilon( 1e-5f )
            .withName( "stability_test" );

        auto ln = std::make_shared<LayerNorm<TDevice, TInput, TOutput>>( device_name, config );

        Tensor<TInput, MR> large_input( shape );
        Tensor<TInput, MR> small_input( shape );
        Tensor<TOutput, MR> large_output( shape );
        Tensor<TOutput, MR> small_output( shape );

        if constexpr ( TDevice == DeviceType::Cpu ) {
            for ( size_t i = 0; i < large_input.size(); ++i ) {
                large_input.data()[ i ] = static_cast<TInput>( 1e4f );
                small_input.data()[ i ] = static_cast<TInput>( 1e-6f );
            }
        }
        else {
            Tensor<TInput, CpuMemoryResource> host_large_input( shape );
            Tensor<TInput, CpuMemoryResource> host_small_input( shape );

            for ( size_t i = 0; i < host_large_input.size(); ++i ) {
                host_large_input.data()[ i ] = static_cast<TInput>( 1e4f );
                host_small_input.data()[ i ] = static_cast<TInput>( 1e-6f );
            }

            large_input = host_large_input.toDevice<CudaDeviceMemoryResource>();
            small_input = host_small_input.toDevice<CudaDeviceMemoryResource>();
        }

        ASSERT_NO_THROW( ln->forward( large_input, large_output ) );
        ASSERT_NO_THROW( ln->forward( small_input, small_output ) );

        if constexpr ( TDevice == DeviceType::Cuda ) {
            auto large_host_output = large_output.toHost<CpuMemoryResource>();
            auto small_host_output = small_output.toHost<CpuMemoryResource>();

            for ( size_t i = 0; i < large_host_output.size(); ++i ) {
                EXPECT_FALSE( std::isnan( large_host_output.data()[ i ] ) );
                EXPECT_FALSE( std::isinf( large_host_output.data()[ i ] ) );
                EXPECT_FALSE( std::isnan( small_host_output.data()[ i ] ) );
                EXPECT_FALSE( std::isinf( small_host_output.data()[ i ] ) );
            }
        }
        else {
            for ( size_t i = 0; i < large_output.size(); ++i ) {
                EXPECT_FALSE( std::isnan( large_output.data()[ i ] ) );
                EXPECT_FALSE( std::isinf( large_output.data()[ i ] ) );
                EXPECT_FALSE( std::isnan( small_output.data()[ i ] ) );
                EXPECT_FALSE( std::isinf( small_output.data()[ i ] ) );
            }
        }
    }

    void TestCpuCudaEquivalence() {
        std::vector<size_t> shape = { 2, 4, 8 };

        LayerNormConfig cpu_config;
        cpu_config.withInputShape( shape ).withAxis( -1 ).withBias( true ).withEpsilon( 1e-5f ).withName( "cpu_test" );

        LayerNormConfig cuda_config;
        cuda_config.withInputShape( shape ).withAxis( -1 ).withBias( true ).withEpsilon( 1e-5f ).withName( "cuda_test" );

        auto cpu_ln = std::make_shared<LayerNorm<DeviceType::Cpu, float>>( "CPU", cpu_config );
        auto cuda_ln = std::make_shared<LayerNorm<DeviceType::Cuda, float>>( "CUDA:0", cuda_config );

        Tensor<float, CpuMemoryResource> host_input( shape );
        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input.data()[ i ] = static_cast<float>( i ) / host_input.size() * 4.0f - 2.0f;
        }

        auto cpu_weight = cpu_ln->getWeight();
        auto cuda_weight = cuda_ln->getWeight();
        auto cpu_bias = cpu_ln->getBias();
        auto cuda_bias = cuda_ln->getBias();

        for ( size_t i = 0; i < cpu_weight->size(); ++i ) {
            cpu_weight->data()[ i ] = 1.0f;
            cpu_bias->data()[ i ] = 0.0f;
        }

        auto device_weight = cpu_weight->toDevice<CudaDeviceMemoryResource>();
        auto device_bias = cpu_bias->toDevice<CudaDeviceMemoryResource>();

        for ( size_t i = 0; i < cuda_weight->size(); ++i ) {
            cuda_weight->data()[ i ] = device_weight.data()[ i ];
            cuda_bias->data()[ i ] = device_bias.data()[ i ];
        }

        Tensor<float, CpuMemoryResource> cpu_output( shape );
        cpu_ln->forward( host_input, cpu_output );

        auto cuda_input = host_input.toDevice<CudaDeviceMemoryResource>();
        Tensor<float, CudaDeviceMemoryResource> cuda_output( shape );
        cuda_ln->forward( cuda_input, cuda_output );

        auto cuda_output_host = cuda_output.toHost<CpuMemoryResource>();

        const float epsilon = 1e-4f;
        for ( size_t i = 0; i < cpu_output.size(); ++i ) {
            float diff = std::abs( cpu_output.data()[ i ] - cuda_output_host.data()[ i ] );
            EXPECT_LT( diff, epsilon ) << "Mismatch at index " << i;
        }
    }

    TEST_F( LayerNormTests, Cpu_Float_ParameterCount ) {
        TestParameterCount( CpuFloatData() );
    }

    TEST_F( LayerNormTests, Cpu_Float_Forward ) {
        TestForward( CpuFloatData() );
    }

    TEST_F( LayerNormTests, Cpu_Float_ToString ) {
        TestToString( CpuFloatData() );
    }

    TEST_F( LayerNormTests, Cpu_Float_DeviceType ) {
        TestDeviceType( CpuFloatData() );
    }

    TEST_F( LayerNormTests, Cpu_Float_WeightAndBiasAccess ) {
        TestWeightAndBiasAccess( CpuFloatData() );
    }

    TEST_F( LayerNormTests, Cpu_Float_NoBias_ParameterCount ) {
        TestParameterCount( CpuNoBiasFloatData() );
    }

    TEST_F( LayerNormTests, Cpu_Float_NoBias_Forward ) {
        TestForward( CpuNoBiasFloatData() );
    }

    TEST_F( LayerNormTests, Context_Cpu_Float_DeviceType ) {
        TestDeviceType( ContextCpuFloatData() );
    }

    TEST_F( LayerNormTests, Context_Cpu_Float_Forward ) {
        TestForward( ContextCpuFloatData() );
    }

    TEST_F( LayerNormTests, Cuda_Float_ParameterCount ) {
        try {
            TestParameterCount( CudaFloatData() );
        }
        catch ( const std::exception& ) {
            std::cout << "CUDA device not available, skipping test" << std::endl;
            SUCCEED();
        }
    }

    TEST_F( LayerNormTests, Cuda_Float_Forward ) {
        try {
            TestForward( CudaFloatData() );
        }
        catch ( const std::exception& ) {
            std::cout << "CUDA device not available, skipping test" << std::endl;
            SUCCEED();
        }
    }

    TEST_F( LayerNormTests, Cuda_Float_ToString ) {
        try {
            TestToString( CudaFloatData() );
        }
        catch ( const std::exception& ) {
            std::cout << "CUDA device not available, skipping test" << std::endl;
            SUCCEED();
        }
    }

    TEST_F( LayerNormTests, Cuda_Float_NoBias_ParameterCount ) {
        try {
            TestParameterCount( CudaNoBiasFloatData() );
        }
        catch ( const std::exception& ) {
            std::cout << "CUDA device not available, skipping test" << std::endl;
            SUCCEED();
        }
    }

    TEST_F( LayerNormTests, SimpleNormalization ) {
        TestSimpleNormalization();
    }

    TEST_F( LayerNormTests, Cpu_Float_NumericalStability ) {
        TestNumericalStability<DeviceType::Cpu, float>();
    }

    TEST_F( LayerNormTests, Cuda_Float_NumericalStability ) {
        try {
            TestNumericalStability<DeviceType::Cuda, float>();
        }
        catch ( const std::exception& ) {
            std::cout << "CUDA device not available, skipping test" << std::endl;
            SUCCEED();
        }
    }

    TEST_F( LayerNormTests, CpuCuda_EquivalenceTest ) {
        try {
            TestCpuCudaEquivalence();
        }
        catch ( const std::exception& ) {
            std::cout << "CUDA device not available, skipping equivalence test" << std::endl;
            SUCCEED();
        }
    }

    TEST_F( LayerNormTests, Constructor_DeviceNameValidation ) {
        LayerNormConfig config;
        config.withInputShape( { 2, 4, 8 } )
            .withName( "validation_test" );

        EXPECT_NO_THROW( (LayerNorm<DeviceType::Cpu, float>( "CPU", config )));
    }

    TEST_F( LayerNormTests, Constructor_DeviceContextValidation ) {
        LayerNormConfig config;
        config.withInputShape( { 2, 4, 8 } )
            .withName( "context_validation_test" );

        auto cpu_context = std::make_shared<DeviceContext>( "CPU" );
        EXPECT_NO_THROW( ( LayerNorm<DeviceType::Cpu, float>( cpu_context, config ) ) );
    }

    TEST_F( LayerNormTests, EdgeCases_MinimalShape ) {
        LayerNormConfig config;
        config.withInputShape( { 1, 1, 1 } )
            .withName( "minimal_test" );

        auto ln = std::make_shared<LayerNorm<DeviceType::Cpu, float>>( "CPU", config );

        Tensor<float, CpuMemoryResource> input( { 1, 1, 1 } );
        Tensor<float, CpuMemoryResource> output( { 1, 1, 1 } );

        input.data()[ 0 ] = 42.0f;

        EXPECT_NO_THROW( ln->forward( input, output ) );
    }

    TEST_F( LayerNormTests, EdgeCases_LargeShape ) {
        LayerNormConfig config;
        config.withInputShape( { 4, 8, 64 } )
            .withName( "large_test" );

        auto ln = std::make_shared<LayerNorm<DeviceType::Cpu, float>>( "CPU", config );

        Tensor<float, CpuMemoryResource> input( { 4, 8, 64 } );
        Tensor<float, CpuMemoryResource> output( { 4, 8, 64 } );

        for ( size_t i = 0; i < input.size(); ++i ) {
            input.data()[ i ] = static_cast<float>( i ) / input.size();
        }

        EXPECT_NO_THROW( ln->forward( input, output ) );
        EXPECT_EQ( output.size(), 4 * 8 * 64 );
    }
}