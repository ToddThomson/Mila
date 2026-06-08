#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>
#include <cstdint>

import Mila;

namespace Components_Normalization_Tests
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
            return { TestShapeSize::Medium, { 64, 128, 1024 }, "Medium" };
        }

        static TestShape Large()
        {
            return { TestShapeSize::Large, { 128, 256, 2048 }, "Large" };
        }

        static TestShape Minimal()
        {
            return { TestShapeSize::Minimal, { 1, 1, 8 }, "Minimal" };
        }

        static std::vector<TestShape> AllShapes()
        {
            return { Small(), Medium(), Large(), Minimal() };
        }

        static std::vector<TestShape> StandardShapes()
        {
            return { Small(), Medium(), Large() };
        }
    };

    // ====================================================================
    // Precision Type Wrapper
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

    template<>
    struct PrecisionTraits<TensorDataType::FP8_E4M3>
    {
        static constexpr TensorDataType value = TensorDataType::FP8_E4M3;
        static constexpr const char* name = "FP8";
        static constexpr float tolerance = 5e-2f;
    };

    // ====================================================================
    // Test Fixture Structure
    // ====================================================================

    template<TensorDataType TPrecision>
    struct SoftmaxTestFixture
    {
        TestShape test_shape;
        SoftmaxConfig config;
        std::shared_ptr<Softmax<DeviceType::Cuda, TPrecision>> component;
        int64_t axis;
        bool is_training;

        static SoftmaxTestFixture Create(
            TestShape shape,
            int64_t axis = -1,
            bool is_training = false )
        {
            SoftmaxTestFixture fixture;
            fixture.test_shape = shape;
            fixture.axis = axis;
            fixture.is_training = is_training;

            fixture.config = SoftmaxConfig();
            fixture.config.withAxis( axis );

            std::string name = "softmax_cuda_" + shape.name + "_" + PrecisionTraits<TPrecision>::name;

            fixture.component = std::make_shared<Softmax<DeviceType::Cuda, TPrecision>>(
                name,
                fixture.config,
                Device::Cuda( 0 )
            );

            // NOTE: do not call setTraining() here. The backend operation is created
            // during build(), so tests must call build() before enabling training.

            return fixture;
        }

        const shape_t& shape() const
        {
            return test_shape.dimensions;
        }
    };

    // ====================================================================
    // Typed Tests (Precision-Based)
    // ====================================================================

    template<typename T>
    class SoftmaxCudaTests : public testing::Test
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
    struct PrecisionType
    {
        static constexpr TensorDataType value = TPrecision;
    };

    using PrecisionTypes = ::testing::Types<
        PrecisionType<TensorDataType::FP32>,
        PrecisionType<TensorDataType::FP16>
    >;

    TYPED_TEST_SUITE( SoftmaxCudaTests, PrecisionTypes );

    TYPED_TEST( SoftmaxCudaTests, Constructor_WithValidDeviceId_CreatesComponent )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        SoftmaxConfig config;
        config.withAxis( -1 );

        std::shared_ptr<Softmax<DeviceType::Cuda, TPrecision>> component{ nullptr };

        ASSERT_NO_THROW(
            (component = std::make_shared<Softmax<DeviceType::Cuda, TPrecision>>(
                "ctor_device_cuda",
                config,
                Device::Cuda( 0 )
            ))
        );

        ASSERT_NE( component, nullptr );
        EXPECT_EQ( component->getDeviceType(), DeviceType::Cuda );
    }

    TYPED_TEST( SoftmaxCudaTests, Constructor_WithoutDeviceId_CreatesComponent )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        SoftmaxConfig config;
        config.withAxis( -1 );

        std::shared_ptr<Softmax<DeviceType::Cuda, TPrecision>> component;

        ASSERT_NO_THROW(
            (component = std::make_shared<Softmax<DeviceType::Cuda, TPrecision>>(
                "ctor_shared_cuda",
                config
            ))
        );

        ASSERT_NE( component, nullptr );
    }

    TYPED_TEST( SoftmaxCudaTests, Constructor_WithInvalidDeviceType_ThrowsInvalidArgument )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        SoftmaxConfig config;
        config.withAxis( -1 );

        EXPECT_THROW(
            ( (void)std::make_shared<Softmax<DeviceType::Cuda, TPrecision>>(
                "invalid_ctor",
                config,
                Device::Cpu()
            ) ),
            std::invalid_argument
        );
    }

    TYPED_TEST( SoftmaxCudaTests, GetDeviceType_AfterConstruction_ReturnsCuda )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Small();
        auto fixture = SoftmaxTestFixture<TPrecision>::Create( test_shape );

        EXPECT_EQ( fixture.component->getDeviceType(), DeviceType::Cuda );

        auto device = fixture.component->getDeviceId();

        EXPECT_EQ( device.type, DeviceType::Cuda );
    }

    TYPED_TEST( SoftmaxCudaTests, IsTraining_InferenceFixture_ReturnsFalse )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Small();
        auto fixture = SoftmaxTestFixture<TPrecision>::Create( test_shape, -1, false );

        EXPECT_FALSE( fixture.component->isTraining() );
    }

    TYPED_TEST( SoftmaxCudaTests, IsTraining_TrainingFixture_ReturnsTrue )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Small();
        auto fixture = SoftmaxTestFixture<TPrecision>::Create( test_shape, -1, true );

        // Build then enable training so the backend operation receives the flag
        fixture.component->build( fixture.shape() );
        fixture.component->setTraining( true );

        EXPECT_TRUE( fixture.component->isTraining() );
    }

    TYPED_TEST( SoftmaxCudaTests, SetTraining_TogglingMode_UpdatesState )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Small();
        auto fixture = SoftmaxTestFixture<TPrecision>::Create( test_shape );

        // Build before toggling training to satisfy backend lifecycle.
        fixture.component->build( fixture.shape() );

        EXPECT_FALSE( fixture.component->isTraining() );

        fixture.component->setTraining( true );
        EXPECT_TRUE( fixture.component->isTraining() );

        fixture.component->setTraining( false );
        EXPECT_FALSE( fixture.component->isTraining() );
    }

    TYPED_TEST( SoftmaxCudaTests, ParameterCount_AfterConstruction_ReturnsZero )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Small();
        auto fixture = SoftmaxTestFixture<TPrecision>::Create( test_shape );

        EXPECT_EQ( fixture.component->parameterCount(), 0 );
    }

    TYPED_TEST( SoftmaxCudaTests, ToString_AfterConstruction_ContainsComponentInfo )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Small();
        auto fixture = SoftmaxTestFixture<TPrecision>::Create( test_shape );
        std::string output = fixture.component->toString();

        EXPECT_NE( output.find( "Softmax" ), std::string::npos );
        EXPECT_NE( output.find( "Device:" ), std::string::npos );
        EXPECT_NE( output.find( "Axis:" ), std::string::npos );
    }

    TYPED_TEST( SoftmaxCudaTests, Synchronize_AfterConstruction_Succeeds )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Small();
        auto fixture = SoftmaxTestFixture<TPrecision>::Create( test_shape );

        EXPECT_NO_THROW( fixture.component->synchronize() );
    }

    TYPED_TEST( SoftmaxCudaTests, Forward_BeforeBuild_ThrowsRuntimeError )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = SoftmaxTestFixture<TPrecision>::Create( TestShape::Small() );

        CudaTensor<TPrecision> input( Device::Cuda( 0 ), fixture.shape() );
        CudaTensor<TPrecision> output( Device::Cuda( 0 ), fixture.shape() );

        EXPECT_THROW(
            fixture.component->forward( input, output ),
            std::runtime_error
        );
    }

    TYPED_TEST( SoftmaxCudaTests, Build_WithVariousShapes_SetsBuiltState )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto shapes = TestShape::StandardShapes();

        for ( const auto& test_shape : shapes )
        {
            auto fixture = SoftmaxTestFixture<TPrecision>::Create( test_shape );

            EXPECT_FALSE( fixture.component->isBuilt() )
                << "Component should not be built before build() for shape: " << test_shape.name;

            EXPECT_NO_THROW( fixture.component->build( fixture.shape() ) )
                << "Build failed for shape: " << test_shape.name;

            EXPECT_TRUE( fixture.component->isBuilt() )
                << "Component should be built after build() for shape: " << test_shape.name;
        }
    }

    TYPED_TEST( SoftmaxCudaTests, Forward_WithVariousShapes_ProducesValidOutput )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto shapes = TestShape::StandardShapes();

        for ( const auto& test_shape : shapes )
        {
            auto fixture = SoftmaxTestFixture<TPrecision>::Create( test_shape );
            fixture.component->build( fixture.shape() );

            CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), fixture.shape() );
            random( host_input, -5.0f, 5.0f );

            CudaTensor<TPrecision> device_input( Device::Cuda( 0 ), fixture.shape() );
            CudaTensor<TPrecision> device_output( Device::Cuda( 0 ), fixture.shape() );

            copy( host_input, device_input );

            EXPECT_NO_THROW( fixture.component->forward( device_input, device_output ) )
                << "Forward failed for shape: " << test_shape.name;

            EXPECT_EQ( device_output.size(), device_input.size() )
                << "Output size mismatch for shape: " << test_shape.name;

            EXPECT_EQ( device_output.shape(), device_input.shape() )
                << "Output shape mismatch for shape: " << test_shape.name;
        }
    }

    TYPED_TEST( SoftmaxCudaTests, Forward_WithVariousShapes_ProducesNormalizedOutput )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto shapes = TestShape::StandardShapes();

        for ( const auto& test_shape : shapes )
        {
            auto fixture = SoftmaxTestFixture<TPrecision>::Create( test_shape );
            fixture.component->build( fixture.shape() );

            CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), fixture.shape() );
            random( host_input, -5.0f, 5.0f );

            CudaTensor<TPrecision> device_input( Device::Cuda( 0 ), fixture.shape() );
            CudaTensor<TPrecision> device_output( Device::Cuda( 0 ), fixture.shape() );

            copy( host_input, device_input );

            fixture.component->forward( device_input, device_output );
            fixture.component->synchronize();

            CpuTensor<TensorDataType::FP32> host_output = toHost<TensorDataType::FP32>( device_output );

            const auto& shape = host_output.shape();
            const int64_t ndim = static_cast<int64_t>(shape.size());

            int64_t normalized_axis = fixture.axis;

            if ( normalized_axis < 0 )
            {
                normalized_axis = ndim + normalized_axis;
            }

            int64_t outer_size = 1;

            for ( int64_t i = 0; i < normalized_axis; ++i )
            {
                outer_size *= shape[ i ];
            }

            int64_t dim_size = shape[ normalized_axis ];

            int64_t inner_size = 1;

            for ( int64_t i = normalized_axis + 1; i < ndim; ++i )
            {
                inner_size *= shape[ i ];
            }

            auto output_ptr = host_output.data();

            auto debug_output = host_output.toString( true );

            for ( int64_t outer = 0; outer < outer_size; ++outer )
            {
                for ( int64_t inner = 0; inner < inner_size; ++inner )
                {
                    float sum = 0.0f;

                    for ( int64_t i = 0; i < dim_size; ++i )
                    {
                        size_t idx = (outer * dim_size * inner_size) + (i * inner_size) + inner;
                        sum += static_cast<float>( output_ptr[ idx ] );
                    }

                    EXPECT_NEAR( sum, 1.0f, PrecisionTraits<TPrecision>::tolerance )
                        << "Softmax sum not normalized for shape: " << test_shape.name
                        << " at outer=" << outer << ", inner=" << inner;
                }
            }
        }
    }

    TYPED_TEST( SoftmaxCudaTests, Forward_WithDifferentAxes_ProducesValidOutput )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Small();
        std::vector<int64_t> axes = { 0, 1, 2 };

        for ( int64_t axis : axes )
        {
            auto fixture = SoftmaxTestFixture<TPrecision>::Create( test_shape, axis );

            fixture.component->build( fixture.shape() );

            CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), fixture.shape() );
            random( host_input, -5.0f, 5.0f );

            CudaTensor<TPrecision> device_input( Device::Cuda( 0 ), fixture.shape() );
            CudaTensor<TPrecision> device_output( Device::Cuda( 0 ), fixture.shape() );

            copy( host_input, device_input );

            EXPECT_NO_THROW( fixture.component->forward( device_input, device_output ) )
                << "Forward failed for axis: " << axis;

            EXPECT_EQ( device_output.size(), device_input.size() )
                << "Output size mismatch for axis: " << axis;
        }
    }

    TYPED_TEST( SoftmaxCudaTests, Forward_ComparedToCpu_ProducesEquivalentOutput )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = { TestShapeSize::Small, { 2, 4, 8 }, "Equivalence" };

        SoftmaxConfig config;
        config.withAxis( -1 );

        auto cpu_component = std::make_shared<Softmax<DeviceType::Cpu, TensorDataType::FP32>>(
            "cpu_equiv",
            config,
            Device::Cpu()
        );

        auto cuda_fixture = SoftmaxTestFixture<TPrecision>::Create( test_shape );

        cpu_component->build( test_shape.dimensions );
        cuda_fixture.component->build( cuda_fixture.shape() );

        CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), test_shape.dimensions );
        random( host_input, -2.0f, 2.0f );

        CpuTensor<TensorDataType::FP32> cpu_output( Device::Cpu(), test_shape.dimensions );
        cpu_component->forward( host_input, cpu_output );

        CudaTensor<TPrecision> device_input( Device::Cuda( 0 ), test_shape.dimensions );
        CudaTensor<TPrecision> device_output( Device::Cuda( 0 ), test_shape.dimensions );
        copy( host_input, device_input );
        cuda_fixture.component->forward( device_input, device_output );
        cuda_fixture.component->synchronize();

        CpuTensor<TensorDataType::FP32> cuda_output_host = toHost<TensorDataType::FP32>( device_output );

        const float epsilon = PrecisionTraits<TPrecision>::tolerance;
        bool all_close = true;
        size_t first_mismatch_idx = 0;
        float max_diff = 0.0f;

        for ( size_t i = 0; i < cpu_output.size(); ++i )
        {
            float cpu_val = cpu_output.data()[ i ];
            float cuda_val = cuda_output_host.data()[ i ];
            float diff = std::abs( cpu_val - cuda_val );

            if ( diff > max_diff )
            {
                max_diff = diff;
            }

            if ( diff > epsilon )
            {
                all_close = false;
                first_mismatch_idx = i;
                break;
            }
        }

        EXPECT_TRUE( all_close )
            << "CPU and CUDA implementations produced different results\n"
            << "First mismatch at index " << first_mismatch_idx << "\n"
            << "CPU value: " << cpu_output.data()[ first_mismatch_idx ] << "\n"
            << "CUDA value: " << cuda_output_host.data()[ first_mismatch_idx ] << "\n"
            << "Max difference: " << max_diff;
    }
}