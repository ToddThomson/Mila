#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <stdexcept>
#include <cmath>
#include <cstdint>

import Mila;

namespace Modules::Connections::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    template<TensorDataType TPrecision>
    using CudaTensor = Tensor<TPrecision, CudaDeviceMemoryResource>;

    using HostTensor = Tensor<TensorDataType::FP32, CpuMemoryResource>;

    class CudaResidualTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // Use DeviceRegistry to determine whether CUDA support was registered.
            // Tests must not call CUDA runtime APIs directly.
            cuda_available_ = DeviceRegistry::instance().hasDeviceType( "CUDA" );

            small_shape_ = { 2, 3, 4 }; // small tensor for correctness checks
        }

        bool cuda_available_{ false };
        shape_t small_shape_;
    };

    TEST_F( CudaResidualTests, Constructor_NullContext_Throws )
    {
        if (!cuda_available_) GTEST_SKIP() << "CUDA not available";

        ResidualConfig cfg;
        cfg.withName( "res_null_ctx" );

        std::shared_ptr<ExecutionContext<DeviceType::Cuda>> null_ctx;
        EXPECT_THROW(
            (std::make_shared<Residual<DeviceType::Cuda, TensorDataType::FP32>>( null_ctx, cfg )),
            std::invalid_argument );
    }

    TEST_F( CudaResidualTests, ParameterCount_DefaultsToZero )
    {
        if (!cuda_available_) GTEST_SKIP() << "CUDA not available";

        ResidualConfig cfg;
        cfg.withName( "res_paramcount" );

        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cuda>>( 0 );
        auto module = std::make_shared<Residual<DeviceType::Cuda, TensorDataType::FP32>>( ctx, cfg );

        EXPECT_EQ( module->parameterCount(), 0u );
    }

    TEST_F( CudaResidualTests, GetName_ToString )
    {
        if (!cuda_available_) GTEST_SKIP() << "CUDA not available";

        ResidualConfig cfg;
        cfg.withName( "res_info" );

        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cuda>>( 0 );
        auto module = std::make_shared<Residual<DeviceType::Cuda, TensorDataType::FP32>>( ctx, cfg );

        EXPECT_EQ( module->getName(), "res_info" );

        std::string s = module->toString();
        EXPECT_NE( s.find( "Residual" ), std::string::npos );
        EXPECT_NE( s.find( "Parameter count" ), std::string::npos );
    }

    TEST_F( CudaResidualTests, Forward_ElementwiseAdd )
    {
        if (!cuda_available_) GTEST_SKIP() << "CUDA not available";

        ResidualConfig cfg;
        cfg.withName( "res_forward" ).withScalingFactor( 1.0f );

        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cuda>>( 0 );
        auto module = std::make_shared<Residual<DeviceType::Cuda, TensorDataType::FP32>>( ctx, cfg );

        // Host inputs (CPU)
        HostTensor host_A( "CPU", small_shape_ );
        HostTensor host_B( "CPU", small_shape_ );

        // Fill deterministic values
        for (size_t i = 0; i < host_A.size(); ++i)
        {
            host_A.data()[i] = static_cast<float>( i ) * 0.25f;
            host_B.data()[i] = static_cast<float>( i ) * 0.75f;
        }

        // Device tensors
        CudaTensor<TensorDataType::FP32> device_A( ctx->getDevice(), small_shape_ );
        CudaTensor<TensorDataType::FP32> device_B( ctx->getDevice(), small_shape_ );
        CudaTensor<TensorDataType::FP32> device_Y( ctx->getDevice(), small_shape_ );

        // Copy host -> device
        copy( host_A, device_A );
        copy( host_B, device_B );

        // Forward via Residual module (delegates to registered CUDA backend)
        EXPECT_NO_THROW( module->forward( device_A, device_B, device_Y ) );

        // Copy result back to host for verification
        HostTensor host_Y = toHost<TensorDataType::FP32>( device_Y );

        const float eps = 1e-4f;
        for (size_t i = 0; i < host_Y.size(); ++i)
        {
            float expected = host_A.data()[i] + host_B.data()[i];
            float got = host_Y.data()[i];
            EXPECT_NEAR( got, expected, eps ) << "Mismatch at index " << i;
        }
    }

    TEST_F( CudaResidualTests, EdgeCase_MinimalShape )
    {
        if (!cuda_available_) GTEST_SKIP() << "CUDA not available";

        ResidualConfig cfg;
        cfg.withName( "res_minimal" );

        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cuda>>( 0 );
        auto module = std::make_shared<Residual<DeviceType::Cuda, TensorDataType::FP32>>( ctx, cfg );

        shape_t minimal = { 1 };
        HostTensor host_A( "CPU", minimal );
        HostTensor host_B( "CPU", minimal );
        host_A.data()[0] = 1.0f;
        host_B.data()[0] = 2.0f;

        CudaTensor<TensorDataType::FP32> device_A( ctx->getDevice(), minimal );
        CudaTensor<TensorDataType::FP32> device_B( ctx->getDevice(), minimal );
        CudaTensor<TensorDataType::FP32> device_Y( ctx->getDevice(), minimal );

        copy( host_A, device_A );
        copy( host_B, device_B );

        EXPECT_NO_THROW( module->forward( device_A, device_B, device_Y ) );

        HostTensor host_Y = toHost<TensorDataType::FP32>( device_Y );
        EXPECT_FLOAT_EQ( host_Y.data()[0], 3.0f );
    }
}