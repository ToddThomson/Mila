#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstdint>

import Mila;

namespace Components::Connections::Tests
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

    TEST_F( ResidualCpuTests, Constructor_WithValidDeviceId_CreatesComponent )
    {
        ResidualConfig cfg;
        auto component = std::shared_ptr<Residual<DeviceType::Cpu, TensorDataType::FP32>>( nullptr );

        ASSERT_NO_THROW(
            (component = std::make_shared<Residual<DeviceType::Cpu, TensorDataType::FP32>>(
                "ctor_device_cpu",
                cfg,
                Device::Cpu()
            ))
        );

        ASSERT_NE( component, nullptr );
        EXPECT_EQ( component->getDeviceType(), DeviceType::Cpu );

        auto device = component->getDeviceId();
        EXPECT_EQ( device.type, DeviceType::Cpu );
    }

    TEST_F( ResidualCpuTests, Constructor_WithoutDeviceId_CreatesComponent )
    {
        ResidualConfig cfg;
        auto component = std::shared_ptr<Residual<DeviceType::Cpu, TensorDataType::FP32>>( nullptr );

        ASSERT_NO_THROW(
            (component = std::make_shared<Residual<DeviceType::Cpu, TensorDataType::FP32>>(
                "ctor_shared_cpu",
                cfg ))
        );

        ASSERT_NE( component, nullptr );
    }

    TEST_F( ResidualCpuTests, IsBuilt_BeforeBuild_ReturnsFalse )
    {
        ResidualConfig cfg;
        auto component = std::make_shared<Residual<DeviceType::Cpu, TensorDataType::FP32>>(
            "res_not_built",
            cfg,
            Device::Cpu() );

        EXPECT_FALSE( component->isBuilt() );
    }

    TEST_F( ResidualCpuTests, Build_WithSmallShape_SetsBuiltState )
    {
        ResidualConfig cfg;
        auto component = std::make_shared<Residual<DeviceType::Cpu, TensorDataType::FP32>>(
            "res_build",
            cfg,
            Device::Cpu() );

        EXPECT_NO_THROW( component->build( small_shape_ ) );
        EXPECT_TRUE( component->isBuilt() );
    }

    TEST_F( ResidualCpuTests, IsBuilt_AfterBuild_ReturnsTrue )
    {
        ResidualConfig cfg;
        auto component = std::make_shared<Residual<DeviceType::Cpu, TensorDataType::FP32>>(
            "res_built",
            cfg,
            Device::Cpu() );

        component->build( small_shape_ );

        EXPECT_TRUE( component->isBuilt() );
    }

    TEST_F( ResidualCpuTests, ParameterCount_DefaultsToZero )
    {
        ResidualConfig cfg;
        auto component = std::make_shared<Residual<DeviceType::Cpu, TensorDataType::FP32>>(
            "res_paramcount",
            cfg,
            Device::Cpu() );

        EXPECT_EQ( component->parameterCount(), 0u );
    }

    TEST_F( ResidualCpuTests, ToString_ContainsComponentInfo )
    {
        ResidualConfig cfg;
        auto component = std::make_shared<Residual<DeviceType::Cpu, TensorDataType::FP32>>(
            "res_info",
            cfg,
            Device::Cpu() );

        std::string s = component->toString();

        EXPECT_NE( s.find( "Residual" ), std::string::npos );
        EXPECT_NE( s.find( "Device:" ), std::string::npos );
    }

    TEST_F( ResidualCpuTests, Forward_BeforeBuild_ThrowsRuntimeError )
    {
        ResidualConfig cfg;
        auto component = std::make_shared<Residual<DeviceType::Cpu, TensorDataType::FP32>>(
            "res_forward_no_build",
            cfg,
            Device::Cpu() );

        CpuTensor<TensorDataType::FP32> A( Device::Cpu(), small_shape_ );
        CpuTensor<TensorDataType::FP32> B( Device::Cpu(), small_shape_ );

        EXPECT_THROW( component->forward( A, B ), std::runtime_error );
    }

    TEST_F( ResidualCpuTests, Forward_ElementwiseAdd )
    {
        ResidualConfig cfg;
        cfg.withScalingFactor( 1.0f );

        auto component = std::make_shared<Residual<DeviceType::Cpu, TensorDataType::FP32>>(
            "res_forward",
            cfg,
            Device::Cpu() );

        // Build before running forward so backends are initialized
        component->build( small_shape_ );

        // Create CPU tensors on the device
        CpuTensor<TensorDataType::FP32> A( Device::Cpu(), small_shape_ );
        CpuTensor<TensorDataType::FP32> B( Device::Cpu(), small_shape_ );

        // Populate inputs deterministically
        float* a_ptr = A.data();
        float* b_ptr = B.data();

        for ( size_t i = 0; i < A.size(); ++i )
        {
            a_ptr[ i ] = static_cast<float>( i ) * 0.125f;
            b_ptr[ i ] = static_cast<float>( i ) * 0.375f;
        }

        // Execute forward through the Residual component (new API returns component-owned Tensor&)
        auto& out_tensor = component->forward( A, B );

        float* y_ptr = out_tensor.data();
        for ( size_t i = 0; i < out_tensor.size(); ++i )
        {
            EXPECT_FLOAT_EQ( y_ptr[ i ], a_ptr[ i ] + b_ptr[ i ] );
        }
    }

    TEST_F( ResidualCpuTests, Backward_ReturnsInputGradients )
    {
        ResidualConfig cfg;
        cfg.withScalingFactor( 1.0f );

        auto component = std::make_shared<Residual<DeviceType::Cpu, TensorDataType::FP32>>(
            "res_backward",
            cfg,
            Device::Cpu() );

        // Build to allocate component-owned buffers
        component->build( small_shape_ );

        // Enable training mode so backward is allowed
        component->setTraining( true );
        EXPECT_TRUE( component->isTraining() );

        // Prepare inputs
        CpuTensor<TensorDataType::FP32> A( Device::Cpu(), small_shape_ );
        CpuTensor<TensorDataType::FP32> B( Device::Cpu(), small_shape_ );

        for ( size_t i = 0; i < A.size(); ++i )
        {
            A.data()[ i ] = static_cast<float>( i ) * 0.1f;
            B.data()[ i ] = static_cast<float>( i ) * 0.2f;
        }

        // Prepare output gradient (same shape)
        CpuTensor<TensorDataType::FP32> output_grad( Device::Cpu(), small_shape_ );

        for ( size_t i = 0; i < output_grad.size(); ++i )
        {
            output_grad.data()[ i ] = static_cast<float>( (i % 97) ) * 0.01f;
        }

        // Call backward: returns pair of references to component-owned input gradients
        auto grads = component->backward( A, B, output_grad );

        auto& a_grad = grads.first;
        auto& b_grad = grads.second;

        // Shapes and sizes should match inputs
        EXPECT_EQ( a_grad.shape(), small_shape_ );
        EXPECT_EQ( b_grad.shape(), small_shape_ );

        EXPECT_EQ( a_grad.size(), A.size() );
        EXPECT_EQ( b_grad.size(), B.size() );

        // For elementwise addition y = a + b, gradients w.r.t inputs should equal output_grad
        float* og_ptr = output_grad.data();
        float* a_grad_ptr = a_grad.data();
        float* b_grad_ptr = b_grad.data();

        for ( size_t i = 0; i < output_grad.size(); ++i )
        {
            EXPECT_FLOAT_EQ( a_grad_ptr[ i ], og_ptr[ i ] );
            EXPECT_FLOAT_EQ( b_grad_ptr[ i ], og_ptr[ i ] );
        }
    }

    TEST_F( ResidualCpuTests, EdgeCase_MinimalShape )
    {
        ResidualConfig cfg;
        auto component = std::make_shared<Residual<DeviceType::Cpu, TensorDataType::FP32>>(
            "res_minimal",
            cfg,
            Device::Cpu() );

        component->build( small_shape_ );

        shape_t minimal = { 1 };
        CpuTensor<TensorDataType::FP32> A( Device::Cpu(), minimal );
        CpuTensor<TensorDataType::FP32> B( Device::Cpu(), minimal );

        *A.data() = 1.0f;
        *B.data() = 2.5f;

        auto& out_tensor = component->forward( A, B );

        float got = out_tensor.data()[ 0 ];
        EXPECT_FLOAT_EQ( got, 3.5f );
    }
}