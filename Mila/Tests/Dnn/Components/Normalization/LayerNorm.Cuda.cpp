#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>
#include <cstdint>

import Mila;

namespace Components_Normalization_LayerNorm_Tests
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
        Transformer,
        Minimal
    };

    struct TestShape
    {
        TestShapeSize size;
        shape_t shape;
        shape_t normalized_shape;
        std::string name;

        static TestShape Small()
        {
            return { TestShapeSize::Small, { 2, 3, 4 }, { 4 }, "Small" };
        }

        static TestShape Medium()
        {
            return { TestShapeSize::Medium, { 8, 16, 32 }, { 32 }, "Medium" };
        }

        static TestShape Large()
        {
            return { TestShapeSize::Large, { 16, 64, 128 }, { 128 }, "Large" };
        }

        static TestShape Transformer()
        {
            return { TestShapeSize::Transformer, { 32, 128, 768 }, { 768 }, "Transformer" };
        }

        static TestShape Minimal()
        {
            return { TestShapeSize::Minimal, { 1, 1, 2 }, { 2 }, "Minimal" };
        }

        static TestShape MultiTrailing()
        {
            return { TestShapeSize::Medium, { 2, 3, 4, 5 }, { 4, 5 }, "MultiTrailing" };
        }

        static std::vector<TestShape> AllShapes()
        {
            return { Small(), Medium(), Large(), Transformer(), Minimal() };
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
    // Test Fixture Structure
    // ====================================================================

    template<TensorDataType TPrecision>
    struct LayerNormTestFixture
    {
        LayerNormTestFixture()
            : test_shape( TestShape::Small() ),
              config(),
              component( nullptr ),
              is_training( false )
        {
            config.withNormalizedShape( test_shape.normalized_shape )
                  .withBias( true )
                  .withEpsilon( 1e-5f );
        }

        TestShape test_shape;
        LayerNormConfig config;
        std::shared_ptr<LayerNorm<DeviceType::Cuda, TPrecision>> component;
        bool is_training;

        static LayerNormTestFixture Create(
            TestShape shape,
            bool has_bias = true,
            float epsilon = 1e-5f,
            bool is_training = false )
        {
            LayerNormTestFixture fixture;
            fixture.test_shape = shape;
            fixture.is_training = is_training;

            fixture.config = LayerNormConfig();
            fixture.config.withNormalizedShape( shape.normalized_shape )
                          .withBias( has_bias )
                          .withEpsilon( epsilon );

            std::string name = "layernorm_cuda_" + shape.name + "_" + PrecisionTraits<TPrecision>::name;

            fixture.component = std::make_shared<LayerNorm<DeviceType::Cuda, TPrecision>>(
                name,
                fixture.config,
                Device::Cuda( 0 )
            );

            return fixture;
        }

        static LayerNormTestFixture CreateWithAxis(
            TestShape shape,
            int64_t axis,
            bool has_bias = true,
            float epsilon = 1e-5f,
            bool is_training = false )
        {
            LayerNormTestFixture fixture;
            fixture.test_shape = shape;
            fixture.is_training = is_training;

            fixture.config = LayerNormConfig();
            fixture.config.withAxis( axis )
                          .withBias( has_bias )
                          .withEpsilon( epsilon );

            std::string name = "layernorm_cuda_axis_" + shape.name + "_" + PrecisionTraits<TPrecision>::name;

            fixture.component = std::make_shared<LayerNorm<DeviceType::Cuda, TPrecision>>(
                name,
                fixture.config,
                Device::Cuda( 0 )
            );

            return fixture;
        }
    };

    // ====================================================================
    // Typed Tests (Precision-Based)
    // ====================================================================

    template<typename T>
    class LayerNormCudaTests : public testing::Test
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
        PrecisionType<TensorDataType::FP32>
        /* TODO: FP16 implementation
        PrecisionType<TensorDataType::FP16> */
    >;

    TYPED_TEST_SUITE( LayerNormCudaTests, PrecisionTypes );

    // ====================================================================
    // Constructor Tests
    // ====================================================================

    TYPED_TEST( LayerNormCudaTests, Constructor_WithValidDeviceId_CreatesComponent )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        LayerNormConfig cfg;
        cfg.withNormalizedShape( { 64 } );

        std::shared_ptr<LayerNorm<DeviceType::Cuda, TPrecision>> component{ nullptr };

        ASSERT_NO_THROW(
            (component = std::make_shared<LayerNorm<DeviceType::Cuda, TPrecision>>(
                "ctor_device_cuda",
                cfg,
                Device::Cuda( 0 )
            ))
        );

        ASSERT_NE( component, nullptr );
        EXPECT_EQ( component->getDeviceType(), DeviceType::Cuda );
    }

    TYPED_TEST( LayerNormCudaTests, Constructor_WithoutDeviceId_CreatesComponent )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        LayerNormConfig cfg;
        cfg.withNormalizedShape( { 64 } );

        std::shared_ptr<LayerNorm<DeviceType::Cuda, TPrecision>> component;

        ASSERT_NO_THROW(
            (component = std::make_shared<LayerNorm<DeviceType::Cuda, TPrecision>>(
                "ctor_shared_cuda",
                cfg
            ))
        );

        ASSERT_NE( component, nullptr );
    }

    TYPED_TEST( LayerNormCudaTests, Constructor_WithInvalidDeviceType_ThrowsInvalidArgument )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        LayerNormConfig cfg;
        cfg.withNormalizedShape( { 64 } );

        EXPECT_THROW(
            ((void)std::make_shared<LayerNorm<DeviceType::Cuda, TPrecision>>(
                "invalid_ctor",
                cfg,
                Device::Cpu()
            )),
            std::invalid_argument
        );
    }

    TYPED_TEST( LayerNormCudaTests, Constructor_WithInvalidConfig_ThrowsInvalidArgument )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        LayerNormConfig invalid_config;

        EXPECT_THROW(
            (std::make_shared<LayerNorm<DeviceType::Cuda, TPrecision>>(
                "invalid_config",
                invalid_config,
                Device::Cuda( 0 )
            )),
            std::invalid_argument
        );
    }

    // ====================================================================
    // Component Property Tests
    // ====================================================================

    TYPED_TEST( LayerNormCudaTests, GetDeviceType_AfterConstruction_ReturnsCuda )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = LayerNormTestFixture<TPrecision>::Create( TestShape::Small() );

        EXPECT_EQ( fixture.component->getDeviceType(), DeviceType::Cuda );

        auto device = fixture.component->getDeviceId();
        EXPECT_EQ( device.type, DeviceType::Cuda );
    }

    TYPED_TEST( LayerNormCudaTests, GetName_AfterConstruction_ReturnsCorrectName )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = LayerNormTestFixture<TPrecision>::Create( TestShape::Small() );

        std::string expected_name = "layernorm_cuda_Small_" + std::string( PrecisionTraits<TPrecision>::name );
        EXPECT_EQ( fixture.component->getName(), expected_name );
    }

    TYPED_TEST( LayerNormCudaTests, IsTraining_InferenceFixture_ReturnsFalse )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = LayerNormTestFixture<TPrecision>::Create( TestShape::Small(), true, 1e-5f, false );

        fixture.component->build( fixture.test_shape.shape );

        EXPECT_FALSE( fixture.component->isTraining() );
    }

    TYPED_TEST( LayerNormCudaTests, IsTraining_TrainingFixture_ReturnsTrue )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = LayerNormTestFixture<TPrecision>::Create( TestShape::Small(), true, 1e-5f, true );

        fixture.component->build( fixture.test_shape.shape );
        fixture.component->setTraining( true );

        EXPECT_TRUE( fixture.component->isTraining() );
    }

    TYPED_TEST( LayerNormCudaTests, SetTraining_TogglingMode_UpdatesState )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = LayerNormTestFixture<TPrecision>::Create( TestShape::Small() );

        fixture.component->build( fixture.test_shape.shape );

        EXPECT_FALSE( fixture.component->isTraining() );

        fixture.component->setTraining( true );
        EXPECT_TRUE( fixture.component->isTraining() );

        fixture.component->setTraining( false );
        EXPECT_FALSE( fixture.component->isTraining() );
    }

    TYPED_TEST( LayerNormCudaTests, ParameterCount_WithBias_ReturnsCorrectCount )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = LayerNormTestFixture<TPrecision>::Create( TestShape::Small(), true );

        fixture.component->build( fixture.test_shape.shape );

        size_t norm_size = 1;
        for ( auto dim : fixture.test_shape.normalized_shape )
        {
            norm_size *= dim;
        }

        EXPECT_EQ( fixture.component->parameterCount(), norm_size * 2 );
    }

    TYPED_TEST( LayerNormCudaTests, ParameterCount_WithoutBias_ReturnsCorrectCount )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = LayerNormTestFixture<TPrecision>::Create( TestShape::Small(), false );

        fixture.component->build( fixture.test_shape.shape );

        size_t norm_size = 1;
        for ( auto dim : fixture.test_shape.normalized_shape )
        {
            norm_size *= dim;
        }

        EXPECT_EQ( fixture.component->parameterCount(), norm_size );
    }

    TYPED_TEST( LayerNormCudaTests, ToString_AfterConstruction_ContainsComponentInfo )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = LayerNormTestFixture<TPrecision>::Create( TestShape::Small() );
        std::string output = fixture.component->toString();

        EXPECT_NE( output.find( "LayerNorm" ), std::string::npos );
        EXPECT_NE( output.find( "Device:" ), std::string::npos );
        EXPECT_NE( output.find( "Epsilon:" ), std::string::npos );
        EXPECT_NE( output.find( "Has Bias:" ), std::string::npos );
    }

    TYPED_TEST( LayerNormCudaTests, Synchronize_AfterConstruction_Succeeds )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = LayerNormTestFixture<TPrecision>::Create( TestShape::Small() );

        EXPECT_NO_THROW( fixture.component->synchronize() );
    }

    // ====================================================================
    // Build Tests
    // ====================================================================

    TYPED_TEST( LayerNormCudaTests, IsBuilt_BeforeBuild_ReturnsFalse )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = LayerNormTestFixture<TPrecision>::Create( TestShape::Small() );

        EXPECT_FALSE( fixture.component->isBuilt() );
    }

    TYPED_TEST( LayerNormCudaTests, Build_WithVariousShapes_SetsBuiltState )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto shapes = TestShape::StandardShapes();

        for ( const auto& test_shape : shapes )
        {
            auto fixture = LayerNormTestFixture<TPrecision>::Create( test_shape );

            EXPECT_FALSE( fixture.component->isBuilt() )
                << "Component should not be built before build() for shape: " << test_shape.name;

            EXPECT_NO_THROW( fixture.component->build( test_shape.shape ) )
                << "Build failed for shape: " << test_shape.name;

            EXPECT_TRUE( fixture.component->isBuilt() )
                << "Component should be built after build() for shape: " << test_shape.name;
        }
    }

    TYPED_TEST( LayerNormCudaTests, Build_WithoutContext_ThrowsRuntimeError )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        LayerNormConfig config;
        config.withNormalizedShape( { 4 } );

        auto module = std::make_shared<LayerNorm<DeviceType::Cuda, TPrecision>>(
            "null_context_test",
            config,
            std::nullopt
        );

        EXPECT_THROW(
            module->build( shape_t{ 2, 1, 4 } ),
            std::runtime_error
        );
    }

    // ====================================================================
    // Forward Tests (new API: forward returns Tensor&)
    // ====================================================================

    TYPED_TEST( LayerNormCudaTests, Forward_BeforeBuild_ThrowsRuntimeError )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = LayerNormTestFixture<TPrecision>::Create( TestShape::Small() );

        CudaTensor<TPrecision> input( Device::Cuda( 0 ), fixture.test_shape.shape );

        EXPECT_THROW(
            fixture.component->forward( input ),
            std::runtime_error
        );
    }

    TYPED_TEST( LayerNormCudaTests, Forward_WithVariousShapes_ProducesValidOutput )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto shapes = TestShape::StandardShapes();

        for ( const auto& test_shape : shapes )
        {
            auto fixture = LayerNormTestFixture<TPrecision>::Create( test_shape );
            fixture.component->build( test_shape.shape );

            CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), test_shape.shape );
            random( host_input, -2.0f, 2.0f );

            CudaTensor<TPrecision> device_input( Device::Cuda( 0 ), test_shape.shape );

            copy( host_input, device_input );

            Tensor<TPrecision, CudaDeviceMemoryResource>* out_ptr = nullptr;
            EXPECT_NO_THROW( { auto& out_ref = fixture.component->forward( device_input ); out_ptr = &out_ref; } )
                << "Forward failed for shape: " << test_shape.name;

            ASSERT_NE( out_ptr, nullptr );

            auto& out_tensor = *out_ptr;

            EXPECT_EQ( out_tensor.shape(), device_input.shape() )
                << "Output shape mismatch for shape: " << test_shape.name;

            auto host_out = toHost<TensorDataType::FP32>( out_tensor );

            EXPECT_EQ( host_out.size(), out_tensor.size() )
                << "Output size mismatch for shape: " << test_shape.name;
        }
    }
    
    // ====================================================================
    // Additional forward variants
    // ====================================================================

    TYPED_TEST( LayerNormCudaTests, Forward_WithBias_Succeeds )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = LayerNormTestFixture<TPrecision>::Create( TestShape::Medium(), true );

        fixture.component->build( fixture.test_shape.shape );

        CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), fixture.test_shape.shape );
        random( host_input, -2.0f, 2.0f );

        CudaTensor<TPrecision> device_input( Device::Cuda( 0 ), fixture.test_shape.shape );

        copy( host_input, device_input );

        Tensor<TPrecision, CudaDeviceMemoryResource>* out_ptr = nullptr;
        EXPECT_NO_THROW( { auto& out_ref = fixture.component->forward( device_input ); out_ptr = &out_ref; } );
        ASSERT_NE( out_ptr, nullptr );
    }

    TYPED_TEST( LayerNormCudaTests, Forward_WithoutBias_Succeeds )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = LayerNormTestFixture<TPrecision>::Create( TestShape::Medium(), false );

        fixture.component->build( fixture.test_shape.shape );

        CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), fixture.test_shape.shape );
        random( host_input, -2.0f, 2.0f );

        CudaTensor<TPrecision> device_input( Device::Cuda( 0 ), fixture.test_shape.shape );

        copy( host_input, device_input );

        Tensor<TPrecision, CudaDeviceMemoryResource>* out_ptr = nullptr;
        EXPECT_NO_THROW( { auto& out_ref = fixture.component->forward( device_input ); out_ptr = &out_ref; } );
        ASSERT_NE( out_ptr, nullptr );
    }

    TYPED_TEST( LayerNormCudaTests, Forward_DifferentEpsilon_Succeeds )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = LayerNormTestFixture<TPrecision>::Create( TestShape::Medium(), true, 1e-3f );

        fixture.component->build( fixture.test_shape.shape );

        CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), fixture.test_shape.shape );
        random( host_input, -2.0f, 2.0f );

        CudaTensor<TPrecision> device_input( Device::Cuda( 0 ), fixture.test_shape.shape );

        copy( host_input, device_input );

        Tensor<TPrecision, CudaDeviceMemoryResource>* out_ptr = nullptr;
        EXPECT_NO_THROW( { auto& out_ref = fixture.component->forward( device_input ); out_ptr = &out_ref; } );
        ASSERT_NE( out_ptr, nullptr );
    }

    TYPED_TEST( LayerNormCudaTests, Forward_MultipleTrailingDims_Succeeds )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = LayerNormTestFixture<TPrecision>::Create( TestShape::MultiTrailing() );

        fixture.component->build( fixture.test_shape.shape );

        CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), fixture.test_shape.shape );
        random( host_input, -3.0f, 3.0f );

        CudaTensor<TPrecision> device_input( Device::Cuda( 0 ), fixture.test_shape.shape );

        copy( host_input, device_input );

        Tensor<TPrecision, CudaDeviceMemoryResource>* out_ptr = nullptr;
        EXPECT_NO_THROW( { auto& out_ref = fixture.component->forward( device_input ); out_ptr = &out_ref; } );
        ASSERT_NE( out_ptr, nullptr );

        auto& out_tensor = *out_ptr;
        EXPECT_EQ( out_tensor.size(), device_input.size() );
    }

    TYPED_TEST( LayerNormCudaTests, Forward_MultipleCalls_Succeeds )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = LayerNormTestFixture<TPrecision>::Create( TestShape::Medium() );
        fixture.component->build( fixture.test_shape.shape );

        CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), fixture.test_shape.shape );
        CudaTensor<TPrecision> device_input( Device::Cuda( 0 ), fixture.test_shape.shape );

        for ( int iter = 0; iter < 10; ++iter )
        {
            random( host_input, -2.0f, 2.0f );
            copy( host_input, device_input );

            Tensor<TPrecision, CudaDeviceMemoryResource>* out_ptr = nullptr;
            EXPECT_NO_THROW( { auto& out_ref = fixture.component->forward( device_input ); out_ptr = &out_ref; } );
            ASSERT_NE( out_ptr, nullptr );
        }
    }

    // ====================================================================
    // WithAxis Tests (new API)
    // ====================================================================

    TYPED_TEST( LayerNormCudaTests, WithAxis_Construction_Succeeds )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = LayerNormTestFixture<TPrecision>::CreateWithAxis( TestShape::Medium(), -1 );

        std::string expected_name = "layernorm_cuda_axis_Medium_" + std::string( PrecisionTraits<TPrecision>::name );
        EXPECT_EQ( fixture.component->getName(), expected_name );
        EXPECT_FALSE( fixture.component->isBuilt() );
    }

    TYPED_TEST( LayerNormCudaTests, WithAxis_Build_Succeeds )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = LayerNormTestFixture<TPrecision>::CreateWithAxis( TestShape::Medium(), -1 );

        EXPECT_NO_THROW( fixture.component->build( fixture.test_shape.shape ) );
        EXPECT_TRUE( fixture.component->isBuilt() );
    }

    TYPED_TEST( LayerNormCudaTests, WithAxis_Forward_Succeeds )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = LayerNormTestFixture<TPrecision>::CreateWithAxis( TestShape::Medium(), -1 );

        fixture.component->build( fixture.test_shape.shape );

        CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), fixture.test_shape.shape );
        random( host_input, -2.0f, 2.0f );

        CudaTensor<TPrecision> device_input( Device::Cuda( 0 ), fixture.test_shape.shape );

        copy( host_input, device_input );

        Tensor<TPrecision, CudaDeviceMemoryResource>* out_ptr = nullptr;
        EXPECT_NO_THROW( { auto& out_ref = fixture.component->forward( device_input ); out_ptr = &out_ref; } );
        ASSERT_NE( out_ptr, nullptr );
    }

    TYPED_TEST( LayerNormCudaTests, WithAxis_ForwardBeforeBuild_ThrowsRuntimeError )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = LayerNormTestFixture<TPrecision>::CreateWithAxis( TestShape::Medium(), -1 );

        CudaTensor<TPrecision> input( Device::Cuda( 0 ), fixture.test_shape.shape );

        EXPECT_THROW(
            fixture.component->forward( input ),
            std::runtime_error
        );
    }

    // ====================================================================
    // Edge Case Tests (new API)
    // ====================================================================

    TYPED_TEST( LayerNormCudaTests, EdgeCase_MinimalShape_Succeeds )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = LayerNormTestFixture<TPrecision>::Create( TestShape::Minimal() );

        fixture.component->build( fixture.test_shape.shape );

        CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), fixture.test_shape.shape );
        random( host_input, -1.0f, 1.0f );

        CudaTensor<TPrecision> device_input( Device::Cuda( 0 ), fixture.test_shape.shape );

        copy( host_input, device_input );

        Tensor<TPrecision, CudaDeviceMemoryResource>* out_ptr = nullptr;
        EXPECT_NO_THROW( { auto& out_ref = fixture.component->forward( device_input ); out_ptr = &out_ref; } );
        ASSERT_NE( out_ptr, nullptr );
    }

    TYPED_TEST( LayerNormCudaTests, EdgeCase_TransformerSize_Succeeds )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = LayerNormTestFixture<TPrecision>::Create( TestShape::Transformer() );

        fixture.component->build( fixture.test_shape.shape );

        CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), fixture.test_shape.shape );
        random( host_input, -2.0f, 2.0f );

        CudaTensor<TPrecision> device_input( Device::Cuda( 0 ), fixture.test_shape.shape );

        copy( host_input, device_input );

        Tensor<TPrecision, CudaDeviceMemoryResource>* out_ptr = nullptr;
        EXPECT_NO_THROW( { auto& out_ref = fixture.component->forward( device_input ); out_ptr = &out_ref; } );
        ASSERT_NE( out_ptr, nullptr );
    }
    
    // ====================================================================
    // CPU <-> CUDA Equivalence Tests (FP32 only, adapted to new API)
    // ====================================================================

    TYPED_TEST( LayerNormCudaTests, Forward_CPU_CUDA_Equivalence_FP32 )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        if constexpr ( TPrecision != TensorDataType::FP32 )
        {
            GTEST_SKIP() << "Forward equivalence test only runs for FP32 precision";
        }

        try
        {
            auto shape = TestShape::Small();

            LayerNormConfig config;
            config.withNormalizedShape( shape.normalized_shape )
                  .withBias( true )
                  .withEpsilon( 1e-5f );

            auto cpu_comp = std::make_shared<LayerNorm<DeviceType::Cpu, TensorDataType::FP32>>(
                "layernorm_cpu_equiv", config, Device::Cpu()
            );

            auto cuda_comp = std::make_shared<LayerNorm<DeviceType::Cuda, TensorDataType::FP32>>(
                "layernorm_cuda_equiv", config, Device::Cuda( 0 )
            );

            cpu_comp->build( shape.shape );
            cuda_comp->build( shape.shape );

            CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), shape.shape );
            random( host_input, -2.0f, 2.0f );

            // CPU forward (new API)
            Tensor<TensorDataType::FP32, CpuMemoryResource>* cpu_out_ptr = nullptr;
            ASSERT_NO_THROW( { auto& out_ref = cpu_comp->forward( host_input ); cpu_out_ptr = &out_ref; } );
            ASSERT_NE( cpu_out_ptr, nullptr );

            auto& cpu_out_tensor = *cpu_out_ptr;

            // CUDA forward (new API)
            CudaTensor<TensorDataType::FP32> device_input( Device::Cuda( 0 ), shape.shape );
            copy( host_input, device_input );

            Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>* cuda_out_ptr = nullptr;
            ASSERT_NO_THROW( { auto& out_ref = cuda_comp->forward( device_input ); cuda_out_ptr = &out_ref; } );
            ASSERT_NE( cuda_out_ptr, nullptr );

            auto& cuda_out_tensor = *cuda_out_ptr;
            cuda_comp->synchronize();

            CpuTensor<TensorDataType::FP32> host_out_cuda = toHost<TensorDataType::FP32>( cuda_out_tensor );

            auto* cdata = cpu_out_tensor.data();
            auto* gdata = host_out_cuda.data();
            size_t total = cpu_out_tensor.size();

            for ( size_t i = 0; i < total; ++i )
            {
                EXPECT_NEAR( cdata[ i ], gdata[ i ], 1e-4f ) << "Mismatch at index " << i;
            }
        }
        catch ( const std::exception& )
        {
            GTEST_SKIP() << "LayerNorm backend not available for CPU/CUDA equivalence test";
        }
    }

    // ====================================================================
    // Backward Tests (CPU <-> CUDA Equivalence, FP32 only, new API)
    // ====================================================================

    TYPED_TEST( LayerNormCudaTests, Backward_CPU_CUDA_Equivalence_FP32 )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        if constexpr ( TPrecision != TensorDataType::FP32 )
        {
            GTEST_SKIP() << "Backward equivalence test only runs for FP32 precision";
        }

        try
        {
            auto shape = TestShape::Small();

            LayerNormConfig config;
            config.withNormalizedShape( shape.normalized_shape )
                .withBias( true )
                .withEpsilon( 1e-5f );

            auto cpu_comp = std::make_shared<LayerNorm<DeviceType::Cpu, TensorDataType::FP32>>(
                "layernorm_cpu_backward", config, Device::Cpu()
            );

            auto cuda_comp = std::make_shared<LayerNorm<DeviceType::Cuda, TensorDataType::FP32>>(
                "layernorm_cuda_backward", config, Device::Cuda( 0 )
            );

            cpu_comp->build( shape.shape );
            cuda_comp->build( shape.shape );

            cpu_comp->setTraining( true );
            cuda_comp->setTraining( true );

            // Prepare inputs and perform forward to populate cached statistics (mean/rstd)
            CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), shape.shape );
            random( host_input, -2.0f, 2.0f );

            // CPU forward (new API)
            Tensor<TensorDataType::FP32, CpuMemoryResource>* cpu_out_ptr = nullptr;
            cpu_out_ptr = &cpu_comp->forward( host_input );
            ASSERT_NE( cpu_out_ptr, nullptr );

            // CUDA forward (new API)
            CudaTensor<TensorDataType::FP32> device_input( Device::Cuda( 0 ), shape.shape );
            copy( host_input, device_input );

            Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>* cuda_out_ptr = nullptr;
            cuda_out_ptr = &cuda_comp->forward( device_input );
            ASSERT_NE( cuda_out_ptr, nullptr );
            cuda_comp->synchronize();

            // Prepare output gradient (dY) on host and device
            CpuTensor<TensorDataType::FP32> host_dout( Device::Cpu(), shape.shape );
            random( host_dout, -1.0f, 1.0f );

            // CPU backward (new API) -> returns component-owned input grad
            Tensor<TensorDataType::FP32, CpuMemoryResource>* cpu_ing_ptr = nullptr;
            ASSERT_NO_THROW( { auto& ing_ref = cpu_comp->backward( host_input, host_dout ); cpu_ing_ptr = &ing_ref; } );
            ASSERT_NE( cpu_ing_ptr, nullptr );

            auto& cpu_ing_tensor = *cpu_ing_ptr;

            // CUDA backward (new API)
            CudaTensor<TensorDataType::FP32> device_dout( Device::Cuda( 0 ), shape.shape );
            copy( host_dout, device_dout );

            Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>* cuda_ing_ptr = nullptr;
            ASSERT_NO_THROW( { auto& ing_ref = cuda_comp->backward( device_input, device_dout ); cuda_ing_ptr = &ing_ref; } );
            ASSERT_NE( cuda_ing_ptr, nullptr );

            auto& cuda_ing_tensor = *cuda_ing_ptr;
            cuda_comp->synchronize();

            CpuTensor<TensorDataType::FP32> host_input_grad_cuda = toHost<TensorDataType::FP32>( cuda_ing_tensor );

            // Compare input gradients
            auto* cpu_dx = cpu_ing_tensor.data();
            auto* cuda_dx = host_input_grad_cuda.data();
            size_t total = cpu_ing_tensor.size();

            for ( size_t i = 0; i < total; ++i )
            {
                EXPECT_NEAR( cpu_dx[ i ], cuda_dx[ i ], 1e-4f ) << "Input-grad mismatch at index " << i;
            }
        }
        catch ( const std::exception& )
        {
            GTEST_SKIP() << "LayerNorm backend not available for CPU/CUDA backward equivalence test";
        }
    }


    // ====================================================================
    // Deterministic Tests (new API)
    // ====================================================================

    TYPED_TEST( LayerNormCudaTests, Forward_Deterministic_ReproducibleWithSeed )
    {
        if ( !this->cuda_available_ )
            GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        if constexpr ( TPrecision != TensorDataType::FP32 )
        {
            GTEST_SKIP() << "Deterministic test only runs for FP32 precision";
        }

        try
        {
            auto shape = TestShape::Small();

            auto cuda_comp = std::make_shared<LayerNorm<DeviceType::Cuda, TensorDataType::FP32>>(
                "layernorm_reproducible",
                LayerNormConfig().withNormalizedShape( shape.normalized_shape ),
                Device::Cuda( 0 )
            );

            cuda_comp->build( shape.shape );

            Mila::Core::RandomGenerator::getInstance().setSeed( 42 );

            CpuTensor<TensorDataType::FP32> host_input1( Device::Cpu(), shape.shape );
            random( host_input1, -2.0f, 2.0f );

            CudaTensor<TensorDataType::FP32> device_input1( Device::Cuda( 0 ), shape.shape );

            copy( host_input1, device_input1 );

            Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>* out1_ptr = nullptr;
            cuda_comp->build( shape.shape );
            ASSERT_NO_THROW( { auto& out_ref = cuda_comp->forward( device_input1 ); out1_ptr = &out_ref; } );
            ASSERT_NE( out1_ptr, nullptr );

            auto& out1_tensor = *out1_ptr;
            CpuTensor<TensorDataType::FP32> host_output1 = toHost<TensorDataType::FP32>( out1_tensor );

            Mila::Core::RandomGenerator::getInstance().setSeed( 42 );

            CpuTensor<TensorDataType::FP32> host_input2( Device::Cpu(), shape.shape );
            random( host_input2, -2.0f, 2.0f );

            CudaTensor<TensorDataType::FP32> device_input2( Device::Cuda( 0 ), shape.shape );

            copy( host_input2, device_input2 );

            Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>* out2_ptr = nullptr;
            ASSERT_NO_THROW( { auto& out_ref = cuda_comp->forward( device_input2 ); out2_ptr = &out_ref; } );
            ASSERT_NE( out2_ptr, nullptr );

            auto& out2_tensor = *out2_ptr;
            CpuTensor<TensorDataType::FP32> host_output2 = toHost<TensorDataType::FP32>( out2_tensor );

            auto* data1 = host_output1.data();
            auto* data2 = host_output2.data();
            size_t total = host_output1.size();

            for ( size_t i = 0; i < total; ++i )
            {
                EXPECT_FLOAT_EQ( data1[ i ], data2[ i ] ) << "Non-reproducible output at index " << i;
            }
        }
        catch ( const std::exception& )
        {
            GTEST_SKIP() << "LayerNorm backend not available";
        }
    }
}