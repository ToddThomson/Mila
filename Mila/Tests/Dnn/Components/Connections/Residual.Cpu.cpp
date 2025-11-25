#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstdint>

import Mila;

namespace Modules::Connections::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    template<TensorDataType TPrecision>
    using CpuTensor = Tensor<TPrecision, CpuMemoryResource>;

    class ResidualCpuTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            small_shape_ = { 2, 3, 4 }; // small 3D tensor for elementwise checks
        }

        shape_t small_shape_;
    };

    TEST_F( ResidualCpuTests, Constructor_NullContext_Throws )
    {
        ResidualConfig cfg;
        cfg.withName( "res_null_ctx" );

        std::shared_ptr<ExecutionContext<DeviceType::Cpu>> null_ctx;
        EXPECT_THROW(
            (std::make_shared<Residual<DeviceType::Cpu, TensorDataType::FP32>>( null_ctx, cfg )),
            std::invalid_argument );
    }

    TEST_F( ResidualCpuTests, ParameterCount_DefaultsToZero )
    {
        ResidualConfig cfg;
        cfg.withName( "res_paramcount" );

        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
        auto module = std::make_shared<Residual<DeviceType::Cpu, TensorDataType::FP32>>( ctx, cfg );

        EXPECT_EQ( module->parameterCount(), 0u );
    }

    TEST_F( ResidualCpuTests, GetName_ToString )
    {
        ResidualConfig cfg;
        cfg.withName( "res_info" );

        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
        auto module = std::make_shared<Residual<DeviceType::Cpu, TensorDataType::FP32>>( ctx, cfg );

        EXPECT_EQ( module->getName(), "res_info" );

        std::string s = module->toString();
        EXPECT_NE( s.find( "Residual" ), std::string::npos );
        EXPECT_NE( s.find( "Parameter count" ), std::string::npos );
    }

    TEST_F( ResidualCpuTests, Forward_ElementwiseAdd )
    {
        ResidualConfig cfg;
        cfg.withName( "res_forward" ).withScalingFactor( 1.0f );

        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
        auto module = std::make_shared<Residual<DeviceType::Cpu, TensorDataType::FP32>>( ctx, cfg );

        // Create CPU tensors on the execution context's device
        CpuTensor<TensorDataType::FP32> A( ctx->getDevice(), small_shape_ );
        CpuTensor<TensorDataType::FP32> B( ctx->getDevice(), small_shape_ );
        CpuTensor<TensorDataType::FP32> Y( ctx->getDevice(), small_shape_ );

        // Populate inputs deterministically
        float* a_ptr = static_cast<float*>(A.rawData());
        float* b_ptr = static_cast<float*>(B.rawData());

        for (size_t i = 0; i < A.size(); ++i)
        {
            a_ptr[i] = static_cast<float>( i ) * 0.125f;
            b_ptr[i] = static_cast<float>( i ) * 0.375f;
        }

        // Execute forward through the Residual module (delegates to backend op)
        EXPECT_NO_THROW( module->forward( A, B, Y ) );

        float* y_ptr = static_cast<float*>( Y.rawData() );
        for (size_t i = 0; i < Y.size(); ++i)
        {
            EXPECT_FLOAT_EQ( y_ptr[i], a_ptr[i] + b_ptr[i] );
        }
    }

    TEST_F( ResidualCpuTests, EdgeCase_MinimalShape )
    {
        ResidualConfig cfg;
        cfg.withName( "res_minimal" );

        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
        auto module = std::make_shared<Residual<DeviceType::Cpu, TensorDataType::FP32>>( ctx, cfg );

        shape_t minimal = { 1 };
        CpuTensor<TensorDataType::FP32> A( ctx->getDevice(), minimal );
        CpuTensor<TensorDataType::FP32> B( ctx->getDevice(), minimal );
        CpuTensor<TensorDataType::FP32> Y( ctx->getDevice(), minimal );

        // simple values
        *static_cast<float*>(A.rawData()) = 1.0f;
        *static_cast<float*>(B.rawData()) = 2.5f;

        EXPECT_NO_THROW( module->forward( A, B, Y ) );

        float got = *static_cast<float*>(Y.rawData());
        EXPECT_FLOAT_EQ( got, 3.5f );
    }
}