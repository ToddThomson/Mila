#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>
#include <cstdint>

import Mila;

namespace Components_Layers_Attention_Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    template<TensorDataType TPrecision>
    using CudaTensor = Tensor<TPrecision, CudaDeviceMemoryResource>;

    template<TensorDataType TPrecision>
    using CpuTensor = Tensor<TPrecision, CpuMemoryResource>;

    // ====================================================================
    // Test Shape Definitions (input is concatenated Q||K||V -> last dim == 3 * embedding)
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
        int64_t batch;
        int64_t seq;
        int64_t embedding_dim;
        int64_t num_heads;
        std::string name;

        shape_t inputShape() const
        {
            return { batch, seq, static_cast<int64_t>(3 * embedding_dim) };
        }

        shape_t outputShape() const
        {
            return { batch, seq, embedding_dim };
        }

        static TestShape Small()
        {
            return { TestShapeSize::Small, 2, 4, 64, 8, "Small" };
        }

        static TestShape Medium()
        {
            return { TestShapeSize::Medium, 8, 16, 128, 8, "Medium" };
        }

        static TestShape Large()
        {
            return { TestShapeSize::Large, 16, 32, 256, 8, "Large" };
        }

        static TestShape Minimal()
        {
            return { TestShapeSize::Minimal, 1, 1, 1, 1, "Minimal" };
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
    // Precision Traits used for tolerances / naming
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
    struct AttentionTestFixture
    {
        // Provide an explicit default constructor because `AttentionConfig`
        // does not have a default constructor. Initialize `test_shape` first
        // (declared order) and then use its fields to construct `config`.
        AttentionTestFixture()
            : test_shape( TestShape::Small() ),
            config( test_shape.embedding_dim, test_shape.num_heads ),
            component( nullptr ),
            is_training( false )
        {}

        TestShape test_shape;
        AttentionConfig config;
        std::shared_ptr<Attention<DeviceType::Cuda, TPrecision>> component;
        bool is_training{ false };

        static AttentionTestFixture Create( TestShape shape, bool is_training = false )
        {
            AttentionTestFixture fixture;
            fixture.test_shape = shape;
            fixture.is_training = is_training;

            fixture.config = AttentionConfig( shape.embedding_dim, shape.num_heads );

            std::string name = "attention_cuda_" + shape.name + "_" + PrecisionTraits<TPrecision>::name;

            fixture.component = std::make_shared<Attention<DeviceType::Cuda, TPrecision>>(
                name,
                fixture.config,
                Device::Cuda( 0 )
            );

            fixture.component->setTraining( is_training );

            return fixture;
        }

        const shape_t& input_shape() const
        {
            static thread_local shape_t s;
            s = test_shape.inputShape();
            return s;
        }

        const shape_t& output_shape() const
        {
            static thread_local shape_t s;
            s = test_shape.outputShape();
            return s;
        }
    };

    // ====================================================================
    // Typed Tests (Precision-Based)
    // ====================================================================

    template<typename T>
    class AttentionCudaTests : public testing::Test
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

    TYPED_TEST_SUITE( AttentionCudaTests, PrecisionTypes );

    TYPED_TEST( AttentionCudaTests, Constructor_WithValidDeviceId_CreatesComponent )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        AttentionConfig cfg( 64, 8 );

        std::shared_ptr<Attention<DeviceType::Cuda, TPrecision>> component{ nullptr };

        ASSERT_NO_THROW(
            (component = std::make_shared<Attention<DeviceType::Cuda, TPrecision>>(
                "ctor_device_cuda",
                cfg,
                Device::Cuda( 0 )
            ))
        );

        ASSERT_NE( component, nullptr );
        EXPECT_EQ( component->getDeviceType(), DeviceType::Cuda );
    }

    TYPED_TEST( AttentionCudaTests, Constructor_WithoutDeviceId_CreatesComponent )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        AttentionConfig cfg( 64, 8 );

        std::shared_ptr<Attention<DeviceType::Cuda, TPrecision>> component;

        ASSERT_NO_THROW(
            (component = std::make_shared<Attention<DeviceType::Cuda, TPrecision>>(
                "ctor_shared_cuda",
                cfg
            ))
        );

        ASSERT_NE( component, nullptr );
    }

    TYPED_TEST( AttentionCudaTests, Constructor_WithInvalidDeviceType_ThrowsInvalidArgument )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        AttentionConfig cfg( 64, 8 );

        EXPECT_THROW(
            ((void)std::make_shared<Attention<DeviceType::Cuda, TPrecision>>(
                "invalid_ctor",
                cfg,
                Device::Cpu()
            )),
            std::invalid_argument
        );
    }

    TYPED_TEST( AttentionCudaTests, GetDeviceType_AfterConstruction_ReturnsCuda )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = AttentionTestFixture<TPrecision>::Create( TestShape::Small() );

        EXPECT_EQ( fixture.component->getDeviceType(), DeviceType::Cuda );

        auto device = fixture.component->getDeviceId();
        EXPECT_EQ( device.type, DeviceType::Cuda );
    }

    TYPED_TEST( AttentionCudaTests, IsTraining_InferenceFixture_ReturnsFalse )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = AttentionTestFixture<TPrecision>::Create( TestShape::Small(), false );

        EXPECT_FALSE( fixture.component->isTraining() );
    }

    TYPED_TEST( AttentionCudaTests, IsTraining_TrainingFixture_ReturnsTrue )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = AttentionTestFixture<TPrecision>::Create( TestShape::Small(), true );

        EXPECT_TRUE( fixture.component->isTraining() );
    }

    TYPED_TEST( AttentionCudaTests, SetTraining_TogglingMode_UpdatesState )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = AttentionTestFixture<TPrecision>::Create( TestShape::Small(), false );

        EXPECT_FALSE( fixture.component->isTraining() );

        fixture.component->setTraining( true );
        EXPECT_TRUE( fixture.component->isTraining() );

        fixture.component->setTraining( false );
        EXPECT_FALSE( fixture.component->isTraining() );
    }

    TYPED_TEST( AttentionCudaTests, ParameterCount_AfterConstruction_ReturnsZero )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = AttentionTestFixture<TPrecision>::Create( TestShape::Small() );

        EXPECT_EQ( fixture.component->parameterCount(), 0 );
    }

    TYPED_TEST( AttentionCudaTests, ToString_AfterConstruction_ContainsComponentInfo )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = AttentionTestFixture<TPrecision>::Create( TestShape::Small() );
        std::string output = fixture.component->toString();

        EXPECT_NE( output.find( "Attention" ), std::string::npos );
        EXPECT_NE( output.find( "Embedding dimension" ), std::string::npos );
        EXPECT_NE( output.find( "Number of heads" ), std::string::npos );
    }

    TYPED_TEST( AttentionCudaTests, Synchronize_AfterConstruction_Succeeds )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = AttentionTestFixture<TPrecision>::Create( TestShape::Small() );

        EXPECT_NO_THROW( fixture.component->synchronize() );
    }

    TYPED_TEST( AttentionCudaTests, Forward_BeforeBuild_ThrowsRuntimeError )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = AttentionTestFixture<TPrecision>::Create( TestShape::Small() );

        CudaTensor<TPrecision> input( Device::Cuda( 0 ), fixture.input_shape() );
        CudaTensor<TPrecision> output( Device::Cuda( 0 ), fixture.output_shape() );

        EXPECT_THROW(
            fixture.component->forward( input, output ),
            std::runtime_error
        );
    }

    TYPED_TEST( AttentionCudaTests, Build_WithVariousShapes_SetsBuiltState )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto shapes = TestShape::StandardShapes();

        for ( const auto& test_shape : shapes )
        {
            auto fixture = AttentionTestFixture<TPrecision>::Create( test_shape );

            EXPECT_FALSE( fixture.component->isBuilt() )
                << "Component should not be built before build() for shape: " << test_shape.name;

            EXPECT_NO_THROW( fixture.component->build( fixture.input_shape() ) )
                << "Build failed for shape: " << test_shape.name;

            EXPECT_TRUE( fixture.component->isBuilt() )
                << "Component should be built after build() for shape: " << test_shape.name;
        }
    }

    TYPED_TEST( AttentionCudaTests, Forward_WithVariousShapes_ProducesValidOutput )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto shapes = TestShape::StandardShapes();

        for ( const auto& test_shape : shapes )
        {
            auto fixture = AttentionTestFixture<TPrecision>::Create( test_shape );
            fixture.component->build( fixture.input_shape() );

            CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), fixture.input_shape() );
            random( host_input, -2.0f, 2.0f );

            CudaTensor<TPrecision> device_input( Device::Cuda( 0 ), fixture.input_shape() );
            CudaTensor<TPrecision> device_output( Device::Cuda( 0 ), fixture.output_shape() );

            copy( host_input, device_input );

            EXPECT_NO_THROW( fixture.component->forward( device_input, device_output ) )
                << "Forward failed for shape: " << test_shape.name;

            EXPECT_EQ( device_output.size(), fixture.output_shape()[ 0 ] * fixture.output_shape()[ 1 ] * fixture.output_shape()[ 2 ] )
                << "Output size mismatch for shape: " << test_shape.name;

            EXPECT_EQ( device_output.shape(), fixture.output_shape() )
                << "Output shape mismatch for shape: " << test_shape.name;
        }
    }
}