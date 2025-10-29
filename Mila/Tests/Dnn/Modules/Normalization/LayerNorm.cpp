#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>
#include <exception>
#include <type_traits>

import Mila;

namespace Modules::Normalization::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    template<DeviceType TDevice>
    using MemoryResourceType = std::conditional_t<TDevice == DeviceType::Cuda,
        CudaDeviceMemoryResource,
        CpuMemoryResource>;

    template<DeviceType TDevice, TensorDataType TPrecision>
    struct LayerNormTestData
    {
        LayerNormConfig config;
        std::shared_ptr<LayerNorm<TDevice, TPrecision>> module;
        std::shared_ptr<ExecutionContext<TDevice>> exec_context;
        shape_t input_shape;
        dim_t channels{ 0 };

        LayerNormTestData() = default;

        static LayerNormTestData Create(
            const std::string& /*name*/,
            const shape_t& input_shape,
            bool has_bias = true,
            float epsilon = 1e-5f,
            dim_t axis = -1 )
        {
            LayerNormTestData d;
            d.input_shape = input_shape;

            // LayerNormConfig API no longer supports setting input shape/name/training via fluent setters.
            // Only set the supported parameters here (bias, epsilon, axis).
            d.config = LayerNormConfig()
                .withBias( has_bias )
                .withEpsilon( epsilon )
                .withAxis( axis );

            if (input_shape.size() >= 3)
            {
                d.channels = static_cast<dim_t>( input_shape[2] );
            }

            d.exec_context = std::make_shared<ExecutionContext<TDevice>>(0);

            d.module = std::make_shared<LayerNorm<TDevice, TPrecision>>( d.exec_context, d.config );

            return d;
        }

        static LayerNormTestData CreateWithContext(
            const std::string& /*name*/,
            const shape_t& input_shape,
            std::shared_ptr<ExecutionContext<TDevice>> ctx,
            bool has_bias = true,
            float epsilon = 1e-5f,
            dim_t axis = -1 )
        {
            LayerNormTestData d;
            d.input_shape = input_shape;

            // See note in Create(): configure only supported options.
            d.config = LayerNormConfig()
                .withBias( has_bias )
                .withEpsilon( epsilon )
                .withAxis( axis );

            if (input_shape.size() >= 3)
            {
                d.channels = static_cast<dim_t>( input_shape[2] );
            }

            d.exec_context = ctx;
            d.module = std::make_shared<LayerNorm<TDevice, TPrecision>>( ctx, d.config );

            return d;
        }
    };

    class LayerNormTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            cpu_shape_ = { 2, 4, 16 };
            cuda_shape_ = { 8, 16, 64 };
        }

        shape_t cpu_shape_;
        shape_t cuda_shape_;
    };

    template<DeviceType TDevice, TensorDataType TPrecision>
    void TestGetName( const LayerNormTestData<TDevice, TPrecision>& d, const std::string& expected = "" )
    {
        ASSERT_NE( d.module, nullptr );
        auto name = d.module->getName();
        // Ensure module provides a non-empty name; if caller provided an expected name, check equality.
        EXPECT_FALSE( name.empty() );
        if (!expected.empty())
        {
            EXPECT_EQ( name, expected );
        }
    }

    template<DeviceType TDevice, TensorDataType TPrecision>
    void TestParameterCount( const LayerNormTestData<TDevice, TPrecision>& d )
    {
        ASSERT_NE( d.module, nullptr );

        // normalize comparison to size_t because parameterCount() returns size_t
        size_t expected = static_cast<size_t>( d.channels );

        if ( d.config.hasBias() )
        {
            expected += static_cast<size_t>( d.channels );
        }

        EXPECT_EQ( d.module->parameterCount(), expected );
    }

    template<DeviceType TDevice, TensorDataType TPrecision>
    void TestForward( const LayerNormTestData<TDevice, TPrecision>& d )
    {
        ASSERT_NE( d.module, nullptr );
        ASSERT_NE( d.exec_context, nullptr );

        using MR = MemoryResourceType<TDevice>;
        using TensorType = Tensor<TPrecision, MR>;

        auto device = d.exec_context->getDevice();
        ASSERT_NE( device, nullptr );

        TensorType input( device, d.input_shape );
        TensorType output( device, d.input_shape );

        if constexpr (TDevice == DeviceType::Cpu)
        {
            // Fill host tensor deterministically
            for (size_t i = 0; i < input.size(); ++i)
            {
                input.data()[i] = static_cast<typename TensorHostTypeMap<TPrecision>::host_type>( (i % 13) * 0.1f );
            }
        }

        // Forward may rely on backend; if backend missing or device unavailable tests should skip.
        EXPECT_NO_THROW( d.module->forward( input, output ) );
        EXPECT_EQ( output.shape(), input.shape() );
        EXPECT_EQ( output.size(), input.size() );
    }

    template<DeviceType TDevice, TensorDataType TPrecision>
    void TestToString( const LayerNormTestData<TDevice, TPrecision>& d, const std::string& substr )
    {
        ASSERT_NE( d.module, nullptr );
        auto s = d.module->toString();
        EXPECT_NE( s.find( substr ), std::string::npos );
    }

    template<DeviceType TDevice, TensorDataType TPrecision>
    void TestDeviceType( const LayerNormTestData<TDevice, TPrecision>& d )
    {
        ASSERT_NE( d.exec_context, nullptr );
        auto device = d.exec_context->getDevice();
        ASSERT_NE( device, nullptr );
        EXPECT_EQ( device->getDeviceType(), TDevice );
    }

    template<DeviceType TDevice, TensorDataType TPrecision>
    void TestSaveLoad( const LayerNormTestData<TDevice, TPrecision>& d )
    {
        ASSERT_NE( d.module, nullptr );
        ModelArchive arc;
        EXPECT_NO_THROW( d.module->save( arc ) );
        EXPECT_NO_THROW( d.module->load( arc ) );
    }

    // --- Test cases ---

    TEST_F( LayerNormTests, Cpu_FP32_Basic )
    {
        auto data = LayerNormTestData<DeviceType::Cpu, TensorDataType::FP32>::Create( "ln_cpu_fp32", { 2,4,16 } );
        
        TestGetName( data );
        TestParameterCount( data );
        TestForward( data );
        TestToString( data, "LayerNorm" );
        TestSaveLoad( data );
        TestDeviceType( data );
    }

    TEST_F( LayerNormTests, Cpu_NoBias )
    {
        auto data = LayerNormTestData<DeviceType::Cpu, TensorDataType::FP32>::Create( "ln_cpu_nobias", { 2,4,16 }, false );
        TestParameterCount( data );
    }

    TEST_F( LayerNormTests, Cpu_EdgeCases )
    {
        // small channel count
        auto data = LayerNormTestData<DeviceType::Cpu, TensorDataType::FP32>::Create( "ln_cpu_small", { 1,1,1 } );
        TestForward( data );
    }

    TEST_F( LayerNormTests, Cuda_FP32_Smoke )
    {
        try
        {
            auto data = LayerNormTestData<DeviceType::Cuda, TensorDataType::FP32>::Create( "ln_cuda_fp32", { 8,16,64 } );
            TestGetName( data );
            TestParameterCount( data );
            TestForward( data );
            TestToString( data, "LayerNorm" );
            TestSaveLoad( data );
            TestDeviceType( data );
        }
        catch (const std::exception&)
        {
            GTEST_SKIP() << "CUDA device or backend not available, skipping CUDA LayerNorm tests";
        }
    }

    TEST_F( LayerNormTests, Cuda_FP16_Smoke )
    {
        try
        {
            auto data = LayerNormTestData<DeviceType::Cuda, TensorDataType::FP16>::Create( "ln_cuda_fp16", { 8,16,64 } );
            TestForward( data );
        }
        catch (const std::exception&)
        {
            GTEST_SKIP() << "CUDA device or backend not available, skipping CUDA FP16 test";
        }
    }

    TEST_F( LayerNormTests, Constructor_InvalidConfig )
    {
        LayerNormConfig cfg;
        cfg.withEpsilon( 0.0f ); // invalid epsilon should trigger validation failure
        auto exec = std::make_shared<ExecutionContext<DeviceType::Cpu>>();

        EXPECT_THROW( (LayerNorm<DeviceType::Cpu, TensorDataType::FP32>( exec, cfg )), std::invalid_argument );
    }

    TEST_F( LayerNormTests, Constructor_ExecutionContextValidation )
    {
        LayerNormConfig cfg;
        cfg.withEpsilon( 1e-5f ).withAxis( -1 );

        auto exec = std::make_shared<ExecutionContext<DeviceType::Cpu>>();

        EXPECT_NO_THROW( (LayerNorm<DeviceType::Cpu, TensorDataType::FP32>( exec, cfg )) );
    }
}