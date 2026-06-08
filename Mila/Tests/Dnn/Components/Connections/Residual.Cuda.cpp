#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>
#include <cstdint>

import Mila;

namespace Components_Connections_Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    template<TensorDataType TPrecision>
    using CudaTensor = Tensor<TPrecision, CudaDeviceMemoryResource>;

    template<TensorDataType TPrecision>
    using CpuTensor = Tensor<TPrecision, CpuMemoryResource>;

    // ====================================================================
    // Test Shape Definitions
    // ====================================================================

    enum class TestShapeSize
    {
        Small,
        Medium,
        Large,
        Minimal
    };

    struct TestShape
    {
        TestShapeSize size;
        shape_t dimensions;
        std::string name;

        static TestShape Small()
        {
            return { TestShapeSize::Small, { 2, 3, 4 }, "Small" };
        }

        static TestShape Medium()
        {
            return { TestShapeSize::Medium, { 4, 64, 256 }, "Medium" };
        }

        static TestShape Large()
        {
            return { TestShapeSize::Large, { 8, 128, 512 }, "Large" };
        }

        static TestShape Minimal()
        {
            return { TestShapeSize::Minimal, { 1 }, "Minimal" };
        }

        static std::vector<TestShape> StandardShapes()
        {
            return { Small(), Medium(), Large() };
        }
    };

    // ====================================================================
    // Precision Traits
    // ====================================================================

    template<TensorDataType TPrecision>
    struct PrecisionTraits
    {
        static constexpr TensorDataType value = TPrecision;
        static constexpr const char* name = "Unknown";
        static constexpr float tolerance = 1e-4f;
    };

    template<>
    struct PrecisionTraits<TensorDataType::FP32>
    {
        static constexpr TensorDataType value = TensorDataType::FP32;
        static constexpr const char* name = "FP32";
        static constexpr float tolerance = 1e-4f;
    };

    template<>
    struct PrecisionTraits<TensorDataType::FP16>
    {
        static constexpr TensorDataType value = TensorDataType::FP16;
        static constexpr const char* name = "FP16";
        static constexpr float tolerance = 1e-2f;
    };

    // ====================================================================
    // Fixture
    // ====================================================================

    template<TensorDataType TPrecision>
    struct ResidualTestFixture
    {
        TestShape test_shape;
        ResidualConfig config;
        std::shared_ptr<Residual<DeviceType::Cuda, TPrecision>> component;
        float scaling;
        bool is_training;

        static ResidualTestFixture Create( TestShape shape, float scaling = 1.0f, bool is_training = false )
        {
            ResidualTestFixture fixture;
            fixture.test_shape = shape;
            fixture.scaling = scaling;
            fixture.is_training = is_training;

            fixture.config = ResidualConfig();
            fixture.config.withScalingFactor( scaling );

            std::string name = std::string( "residual_cuda_" ) + shape.name + "_" + PrecisionTraits<TPrecision>::name;

            fixture.component = std::make_shared<Residual<DeviceType::Cuda, TPrecision>>(
                name,
                fixture.config,
                Device::Cuda( 0 )
            );

            // Note: do not call setTraining() here. Tests must call build() before setTraining()
            // because setTraining() is propagated to the backend operation which is initialized
            // during build/after the execution context is set.

            return fixture;
        }

        const shape_t& shape() const
        {
            return test_shape.dimensions;
        }
    };

    // ====================================================================
    // Typed Tests
    // ====================================================================

    template<typename T>
    class ResidualCudaTests : public testing::Test
    {
    protected:
        void SetUp() override
        {
            int device_count = getDeviceCount( DeviceType::Cuda );
            cuda_available_ = (device_count > 0);
        }

        bool cuda_available_{ false };
    };

    template<TensorDataType TPrecision>
    struct PrecisionType {
        static constexpr TensorDataType value = TPrecision;
    };

    using PrecisionTypes = ::testing::Types<
        PrecisionType<TensorDataType::FP32>,
        PrecisionType<TensorDataType::FP16>
    >;

    TYPED_TEST_SUITE( ResidualCudaTests, PrecisionTypes );

    // ====================================================================
    // Construction / Device tests
    // ====================================================================

    TYPED_TEST( ResidualCudaTests, Constructor_WithValidDeviceId_CreatesComponent )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        ResidualConfig cfg;
        cfg.withScalingFactor( 1.0f );

        std::shared_ptr<Residual<DeviceType::Cuda, TPrecision>> component{ nullptr };

        ASSERT_NO_THROW(
            (component = std::make_shared<Residual<DeviceType::Cuda, TPrecision>>(
                "ctor_device_cuda",
                cfg,
                Device::Cuda( 0 )
            ))
        );

        ASSERT_NE( component, nullptr );
        EXPECT_EQ( component->getDeviceType(), DeviceType::Cuda );

        auto device = component->getDeviceId();

        EXPECT_EQ( device.type, DeviceType::Cuda );
    }

    TYPED_TEST( ResidualCudaTests, Constructor_WithoutDeviceId_CreatesComponent )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        ResidualConfig cfg;
        cfg.withScalingFactor( 1.0f );

        std::shared_ptr<Residual<DeviceType::Cuda, TPrecision>> component;

        ASSERT_NO_THROW(
            (component = std::make_shared<Residual<DeviceType::Cuda, TPrecision>>(
                "ctor_shared_cuda",
                cfg
            ))
        );

        ASSERT_NE( component, nullptr );
    }

    TYPED_TEST( ResidualCudaTests, Constructor_WithInvalidDeviceType_ThrowsInvalidArgument )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        ResidualConfig cfg;
        cfg.withScalingFactor( 1.0f );

        EXPECT_THROW(
            ((void)std::make_shared<Residual<DeviceType::Cuda, TPrecision>>(
                "invalid_ctor",
                cfg,
                Device::Cpu()
            )),
            std::invalid_argument
        );
    }

    // ====================================================================
    // Introspection / lifecycle tests
    // ====================================================================

    TYPED_TEST( ResidualCudaTests, IsTraining_DefaultsToFalse )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = ResidualTestFixture<TPrecision>::Create( TestShape::Small(), 1.0f, false );

        EXPECT_FALSE( fixture.component->isTraining() );
    }

    TYPED_TEST( ResidualCudaTests, SetTraining_TogglesState )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = ResidualTestFixture<TPrecision>::Create( TestShape::Small() );

        // Build before toggling training to satisfy backend operation lifecycle.
        EXPECT_NO_THROW( fixture.component->build( fixture.shape() ) );

        EXPECT_FALSE( fixture.component->isTraining() );

        fixture.component->setTraining( true );
        EXPECT_TRUE( fixture.component->isTraining() );

        fixture.component->setTraining( false );
        EXPECT_FALSE( fixture.component->isTraining() );
    }

    TYPED_TEST( ResidualCudaTests, ParameterCount_AfterConstruction_ReturnsZero )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = ResidualTestFixture<TPrecision>::Create( TestShape::Small() );

        EXPECT_EQ( fixture.component->parameterCount(), 0 );
    }

    TYPED_TEST( ResidualCudaTests, ToString_ContainsComponentInfo )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = ResidualTestFixture<TPrecision>::Create( TestShape::Small() );
        std::string output = fixture.component->toString();

        EXPECT_NE( output.find( "Residual" ), std::string::npos );
        EXPECT_NE( output.find( "Device:" ), std::string::npos );
    }

    TYPED_TEST( ResidualCudaTests, Synchronize_AfterConstruction_Succeeds )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = ResidualTestFixture<TPrecision>::Create( TestShape::Small() );

        EXPECT_NO_THROW( fixture.component->synchronize() );
    }

    // ====================================================================
    // Forward / Build tests
    // ====================================================================

    TYPED_TEST( ResidualCudaTests, Forward_BeforeBuild_ThrowsRuntimeError )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = ResidualTestFixture<TPrecision>::Create( TestShape::Small() );

        CudaTensor<TPrecision> A( Device::Cuda( 0 ), fixture.shape() );
        CudaTensor<TPrecision> B( Device::Cuda( 0 ), fixture.shape() );

        EXPECT_THROW( fixture.component->forward( A, B ), std::runtime_error );
    }

    TYPED_TEST( ResidualCudaTests, Build_WithVariousShapes_SetsBuiltState )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto shapes = TestShape::StandardShapes();

        for ( const auto& test_shape : shapes )
        {
            auto fixture = ResidualTestFixture<TPrecision>::Create( test_shape );

            EXPECT_FALSE( fixture.component->isBuilt() )
                << "Component should not be built before build() for shape: " << test_shape.name;

            EXPECT_NO_THROW( fixture.component->build( fixture.shape() ) )
                << "Build failed for shape: " << test_shape.name;

            EXPECT_TRUE( fixture.component->isBuilt() )
                << "Component should be built after build() for shape: " << test_shape.name;
        }
    }

    TYPED_TEST( ResidualCudaTests, Forward_WithVariousShapes_ProducesValidOutput )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto shapes = TestShape::StandardShapes();

        for ( const auto& test_shape : shapes )
        {
            auto fixture = ResidualTestFixture<TPrecision>::Create( test_shape, 1.0f );
            fixture.component->build( fixture.shape() );

            CpuTensor<TensorDataType::FP32> host_A( Device::Cpu(), fixture.shape() );
            CpuTensor<TensorDataType::FP32> host_B( Device::Cpu(), fixture.shape() );

            random( host_A, -2.0f, 2.0f );
            random( host_B, -2.0f, 2.0f );

            CudaTensor<TPrecision> device_A( Device::Cuda( 0 ), fixture.shape() );
            CudaTensor<TPrecision> device_B( Device::Cuda( 0 ), fixture.shape() );

            copy( host_A, device_A );
            copy( host_B, device_B );

            CudaTensor<TPrecision>* out_tensor = nullptr;
            ASSERT_NO_THROW( out_tensor = &fixture.component->forward( device_A, device_B ) )
                << "Forward failed for shape: " << test_shape.name;

            ASSERT_NE( out_tensor, nullptr );

            EXPECT_EQ( out_tensor->shape(), fixture.shape() )
                << "Output shape mismatch for shape: " << test_shape.name;

            auto host_out = toHost<TensorDataType::FP32>( *out_tensor );

            EXPECT_EQ( host_out.size(), out_tensor->size() )
                << "Output size mismatch for shape: " << test_shape.name;
        }
    }

    // ====================================================================
    // CPU <-> CUDA parity tests
    // ====================================================================

    TYPED_TEST( ResidualCudaTests, Forward_ElementwiseAdd_MatchesCpu )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Small();
        ResidualConfig cfg;
        cfg.withScalingFactor( 1.0f );

        auto cpu_component = std::make_shared<Residual<DeviceType::Cpu, TensorDataType::FP32>>(
            "cpu_ref",
            cfg,
            Device::Cpu()
        );

        auto cuda_fixture = ResidualTestFixture<TPrecision>::Create( test_shape, 1.0f );

        cpu_component->build( test_shape.dimensions );
        cuda_fixture.component->build( cuda_fixture.shape() );

        CpuTensor<TensorDataType::FP32> host_A( Device::Cpu(), test_shape.dimensions );
        CpuTensor<TensorDataType::FP32> host_B( Device::Cpu(), test_shape.dimensions );

        random( host_A, -1.0f, 1.0f );
        random( host_B, -1.0f, 1.0f );

        // CPU forward (new API)
        Tensor<TensorDataType::FP32, CpuMemoryResource>* cpu_out_tensor = nullptr;
        ASSERT_NO_THROW( cpu_out_tensor = &cpu_component->forward( host_A, host_B ) );
        ASSERT_NE( cpu_out_tensor, nullptr );

        CudaTensor<TPrecision> device_A( Device::Cuda( 0 ), test_shape.dimensions );
        CudaTensor<TPrecision> device_B( Device::Cuda( 0 ), test_shape.dimensions );

        copy( host_A, device_A );
        copy( host_B, device_B );

        Tensor<TPrecision, CudaDeviceMemoryResource>* cuda_out_tensor = nullptr;
        ASSERT_NO_THROW( cuda_out_tensor = &cuda_fixture.component->forward( device_A, device_B ) );
        ASSERT_NE( cuda_out_tensor, nullptr );

        cuda_fixture.component->synchronize();

        CpuTensor<TensorDataType::FP32> cuda_Y_host = toHost<TensorDataType::FP32>( *cuda_out_tensor );

        const float eps = PrecisionTraits<TPrecision>::tolerance;
        for ( size_t i = 0; i < cpu_out_tensor->size(); ++i )
        {
            float expected = cpu_out_tensor->data()[ i ];
            float got = cuda_Y_host.data()[ i ];
            EXPECT_NEAR( got, expected, eps ) << "Mismatch at index " << i;
        }
    }

    TYPED_TEST( ResidualCudaTests, EdgeCase_MinimalShape )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape minimal_shape = TestShape::Minimal();
        auto fixture = ResidualTestFixture<TPrecision>::Create( minimal_shape );

        fixture.component->build( fixture.shape() );

        CpuTensor<TensorDataType::FP32> host_A( Device::Cpu(), minimal_shape.dimensions );
        CpuTensor<TensorDataType::FP32> host_B( Device::Cpu(), minimal_shape.dimensions );

        host_A.data()[ 0 ] = 1.0f;
        host_B.data()[ 0 ] = 2.0f;

        CudaTensor<TPrecision> device_A( Device::Cuda( 0 ), minimal_shape.dimensions );
        CudaTensor<TPrecision> device_B( Device::Cuda( 0 ), minimal_shape.dimensions );

        copy( host_A, device_A );
        copy( host_B, device_B );

        Tensor<TPrecision, CudaDeviceMemoryResource>* out_tensor = nullptr;
        ASSERT_NO_THROW( out_tensor = &fixture.component->forward( device_A, device_B ) );
        ASSERT_NE( out_tensor, nullptr );

        fixture.component->synchronize();

        CpuTensor<TensorDataType::FP32> host_Y = toHost<TensorDataType::FP32>( *out_tensor );
        EXPECT_FLOAT_EQ( host_Y.data()[ 0 ], 3.0f );
    }
}