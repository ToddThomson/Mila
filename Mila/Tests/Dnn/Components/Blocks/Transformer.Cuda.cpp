#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <cmath>
#include <type_traits>
#include <stdexcept>
#include <exception>
#include <cstdint>

import Mila;

namespace CompositeComponents_Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    template<TensorDataType TPrecision>
    using CudaTensor = Tensor<TPrecision, CudaDeviceMemoryResource>;

    template<TensorDataType TPrecision>
    using CpuTensor = Tensor<TPrecision, CpuMemoryResource>;

    // ====================================================================
    // Test Shape Definitions
    // ====================================================================
    namespace
    {
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
                return { TestShapeSize::Small, { 2, 16, 128 }, "Small" };
            }

            static TestShape Medium()
            {
                return { TestShapeSize::Medium, { 8, 64, 256 }, "Medium" };
            }

            static TestShape Large()
            {
                return { TestShapeSize::Large, { 16, 128, 512 }, "Large" };
            }

            static TestShape Minimal()
            {
                return { TestShapeSize::Minimal, { 1, 1, 64 }, "Minimal" };
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
    }

    // ====================================================================
    // Test Network for Shared Context Testing
    // ====================================================================

    template<TensorDataType TPrecision>
    class TransformerTestNetwork : public Network<DeviceType::Cuda, TPrecision>
    {
    private:
        std::unique_ptr<IExecutionContext> owned_context_;
        std::shared_ptr<Transformer<DeviceType::Cuda, TPrecision>> transformer_;
        TransformerConfig config_;

    public:
        explicit TransformerTestNetwork(
            const std::string& name,
            const TransformerConfig& config,
            DeviceId device_id )
            : Network<DeviceType::Cuda, TPrecision>( name ),
            owned_context_( createExecutionContext( device_id ) ),
            config_( config )
        {
            createGraph();
            this->setExecutionContext( owned_context_.get() );
        }

        std::shared_ptr<Transformer<DeviceType::Cuda, TPrecision>> getTransformer() const
        {
            return transformer_;
        }

        const TransformerConfig& getConfig() const
        {
            return config_;
        }

    protected:
        void save_( ModelArchive& archive, SerializationMode mode ) const override
        {
            // Minimal implementation for testing
        }

    private:
        void createGraph()
        {
            transformer_ = std::make_shared<Transformer<DeviceType::Cuda, TPrecision>>(
                this->getName() + ".transformer",
                config_
            );

            this->addComponent( transformer_ );
        }
    };

    // ====================================================================
    // Test Fixture Structure
    // ====================================================================

    template<TensorDataType TPrecision>
    struct TransformerTestFixture
    {
        TestShape test_shape;
        TransformerConfig config;
        std::shared_ptr<Transformer<DeviceType::Cuda, TPrecision>> component;
        std::unique_ptr<TransformerTestNetwork<TPrecision>> network;
        dim_t embedding_dim;
        dim_t num_heads;
        bool is_training;
        bool use_shared_context;

        TransformerTestFixture()
            : config( 1, 1 ), embedding_dim( 0 ), num_heads( 0 )
        {}

        static TransformerTestFixture CreateStandalone(
            TestShape shape,
            dim_t embedding_dim,
            dim_t num_heads,
            dim_t hidden_dim = 0,
            bool use_bias = true,
            ActivationType activation = ActivationType::Gelu,
            bool is_training = false )
        {
            TransformerTestFixture fixture;
            fixture.test_shape = shape;
            fixture.embedding_dim = embedding_dim;
            fixture.num_heads = num_heads;
            fixture.is_training = is_training;
            fixture.use_shared_context = false;

            fixture.config = TransformerConfig( embedding_dim, num_heads );

            if ( hidden_dim > 0 )
            {
                fixture.config.withHiddenDimension( hidden_dim );
            }

            fixture.config.withBias( use_bias )
                .withActivation( activation );

            std::string name = "transformer_cuda_" + shape.name + "_" + PrecisionTraits<TPrecision>::name;

            fixture.component = std::make_shared<Transformer<DeviceType::Cuda, TPrecision>>(
                name,
                fixture.config,
                Device::Cuda( 0 )
            );

            fixture.component->setTraining( is_training );

            return fixture;
        }

        static TransformerTestFixture CreateWithSharedContext(
            TestShape shape,
            dim_t embedding_dim,
            dim_t num_heads,
            dim_t hidden_dim = 0,
            bool use_bias = true,
            ActivationType activation = ActivationType::Gelu,
            bool is_training = false )
        {
            TransformerTestFixture fixture;
            fixture.test_shape = shape;
            fixture.embedding_dim = embedding_dim;
            fixture.num_heads = num_heads;
            fixture.is_training = is_training;
            fixture.use_shared_context = true;

            fixture.config = TransformerConfig( embedding_dim, num_heads );

            if ( hidden_dim > 0 )
            {
                fixture.config.withHiddenDimension( hidden_dim );
            }

            fixture.config.withBias( use_bias )
                .withActivation( activation );

            std::string name = "transformer_network_" + shape.name + "_" + PrecisionTraits<TPrecision>::name;

            fixture.network = std::make_unique<TransformerTestNetwork<TPrecision>>(
                name,
                fixture.config,
                Device::Cuda( 0 )
            );

            fixture.component = fixture.network->getTransformer();
            fixture.network->setTraining( is_training );

            return fixture;
        }

        const shape_t& shape() const
        {
            return test_shape.dimensions;
        }

        static dim_t getEmbeddingDimFromShape( const TestShape& shape )
        {
            return static_cast<dim_t>(shape.dimensions.back());
        }

        static dim_t getCompatibleHeadCount( dim_t embedding_dim )
        {
            // Find a compatible head count that divides evenly
            for ( dim_t heads : { 8, 4, 2, 1 } )
            {
                if ( embedding_dim % heads == 0 )
                {
                    return heads;
                }
            }
            return 1;
        }
    };

    // ====================================================================
    // Typed Tests (Precision-Based)
    // ====================================================================

    template<typename T>
    class TransformerCudaTests : public testing::Test
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

    TYPED_TEST_SUITE( TransformerCudaTests, PrecisionTypes );

    // ====================================================================
    // Constructor Tests
    // ====================================================================

    TYPED_TEST( TransformerCudaTests, Constructor_WithValidDeviceId_CreatesComponent )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = TransformerTestFixture<TPrecision>::getCompatibleHeadCount( embedding_dim );

        auto fixture = TransformerTestFixture<TPrecision>::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads
        );

        ASSERT_NE( fixture.component, nullptr );
        EXPECT_EQ( fixture.component->getDeviceType(), DeviceType::Cuda );
        EXPECT_EQ( fixture.component->getDeviceId().type, DeviceType::Cuda );
    }

    TYPED_TEST( TransformerCudaTests, Constructor_WithoutDeviceId_CreatesComponent )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = TransformerTestFixture<TPrecision>::getCompatibleHeadCount( embedding_dim );

        auto fixture = TransformerTestFixture<TPrecision>::CreateWithSharedContext(
            test_shape,
            embedding_dim,
            num_heads
        );

        ASSERT_NE( fixture.component, nullptr );
        ASSERT_NE( fixture.network, nullptr );
    }

    TYPED_TEST( TransformerCudaTests, Constructor_WithInvalidConfiguration_ThrowsInvalidArgument )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        // Test invalid embedding dimension
        EXPECT_THROW(
            TransformerConfig( 0, 8 ),
            std::invalid_argument
        );

        // Test invalid num_heads
        EXPECT_THROW(
            TransformerConfig( 128, 0 ),
            std::invalid_argument
        );

        // Test non-divisible heads
        EXPECT_THROW(
            TransformerConfig( 127, 8 ),
            std::invalid_argument
        );

        // Test with valid config but wrong device
        TransformerConfig valid_config( 128, 8 );
        EXPECT_THROW(
            (Transformer<DeviceType::Cuda, TPrecision>( "invalid_device", valid_config, Device::Cpu() )),
            std::invalid_argument
        );
    }

    TYPED_TEST( TransformerCudaTests, Constructor_WithInvalidDeviceType_ThrowsInvalidArgument )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TransformerConfig config( 128, 8 );

        EXPECT_THROW(
            (Transformer<DeviceType::Cuda, TPrecision>( "invalid_device", config, Device::Cpu() )),
            std::invalid_argument
        );
    }

    // ====================================================================
    // Basic Property Tests
    // ====================================================================

    TYPED_TEST( TransformerCudaTests, GetDeviceType_AfterConstruction_ReturnsCuda )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = TransformerTestFixture<TPrecision>::getCompatibleHeadCount( embedding_dim );

        auto fixture = TransformerTestFixture<TPrecision>::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads
        );

        EXPECT_EQ( fixture.component->getDeviceType(), DeviceType::Cuda );

        auto device = fixture.component->getDeviceId();
        EXPECT_EQ( device.type, DeviceType::Cuda );
    }

    TYPED_TEST( TransformerCudaTests, GetName_AfterConstruction_ReturnsCorrectName )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = TransformerTestFixture<TPrecision>::getCompatibleHeadCount( embedding_dim );

        auto fixture = TransformerTestFixture<TPrecision>::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads
        );

        std::string expected_name = "transformer_cuda_" + test_shape.name + "_" + PrecisionTraits<TPrecision>::name;
        EXPECT_EQ( fixture.component->getName(), expected_name );
    }

    TYPED_TEST( TransformerCudaTests, IsTraining_InferenceFixture_ReturnsFalse )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = TransformerTestFixture<TPrecision>::getCompatibleHeadCount( embedding_dim );

        auto fixture = TransformerTestFixture<TPrecision>::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads,
            0,
            true,
            ActivationType::Gelu,
            false
        );

        EXPECT_FALSE( fixture.component->isTraining() );
    }

    TYPED_TEST( TransformerCudaTests, IsTraining_TrainingFixture_ReturnsTrue )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = TransformerTestFixture<TPrecision>::getCompatibleHeadCount( embedding_dim );

        auto fixture = TransformerTestFixture<TPrecision>::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads,
            0,
            true,
            ActivationType::Gelu,
            true
        );

        EXPECT_TRUE( fixture.component->isTraining() );
    }

    TYPED_TEST( TransformerCudaTests, SetTraining_TogglingMode_UpdatesState )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = TransformerTestFixture<TPrecision>::getCompatibleHeadCount( embedding_dim );

        auto fixture = TransformerTestFixture<TPrecision>::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads
        );

        EXPECT_FALSE( fixture.component->isTraining() );

        fixture.component->setTraining( true );
        EXPECT_TRUE( fixture.component->isTraining() );

        fixture.component->setTraining( false );
        EXPECT_FALSE( fixture.component->isTraining() );
    }

    TYPED_TEST( TransformerCudaTests, ParameterCount_AfterBuild_ReturnsNonZero )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = TransformerTestFixture<TPrecision>::getCompatibleHeadCount( embedding_dim );

        auto fixture = TransformerTestFixture<TPrecision>::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads
        );

        fixture.component->build( fixture.shape() );

        EXPECT_GT( fixture.component->parameterCount(), 0u );
    }

    TYPED_TEST( TransformerCudaTests, ToString_AfterConstruction_ContainsComponentInfo )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = TransformerTestFixture<TPrecision>::getCompatibleHeadCount( embedding_dim );

        auto fixture = TransformerTestFixture<TPrecision>::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads
        );

        std::string output = fixture.component->toString();

        EXPECT_NE( output.find( "Transformer" ), std::string::npos );
        EXPECT_NE( output.find( fixture.component->getName() ), std::string::npos );
        EXPECT_NE( output.find( "Number of heads" ), std::string::npos );
    }

    TYPED_TEST( TransformerCudaTests, Synchronize_AfterConstruction_Succeeds )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = TransformerTestFixture<TPrecision>::getCompatibleHeadCount( embedding_dim );

        auto fixture = TransformerTestFixture<TPrecision>::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads
        );

        EXPECT_NO_THROW( fixture.component->synchronize() );
    }

    // ====================================================================
    // Build State Tests
    // ====================================================================

    TYPED_TEST( TransformerCudaTests, IsBuilt_BeforeBuild_ReturnsFalse )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = TransformerTestFixture<TPrecision>::getCompatibleHeadCount( embedding_dim );

        auto fixture = TransformerTestFixture<TPrecision>::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads
        );

        EXPECT_FALSE( fixture.component->isBuilt() );
    }

    TYPED_TEST( TransformerCudaTests, Build_WithVariousShapes_SetsBuiltState )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto shapes = TestShape::StandardShapes();

        for ( const auto& test_shape : shapes )
        {
            dim_t embedding_dim = test_shape.dimensions.back();
            dim_t num_heads = TransformerTestFixture<TPrecision>::getCompatibleHeadCount( embedding_dim );

            auto fixture = TransformerTestFixture<TPrecision>::CreateStandalone(
                test_shape,
                embedding_dim,
                num_heads
            );

            EXPECT_FALSE( fixture.component->isBuilt() )
                << "Component should not be built before build() for shape: " << test_shape.name;

            EXPECT_NO_THROW( fixture.component->build( fixture.shape() ) )
                << "Build failed for shape: " << test_shape.name;

            EXPECT_TRUE( fixture.component->isBuilt() )
                << "Component should be built after build() for shape: " << test_shape.name;
        }
    }

    TYPED_TEST( TransformerCudaTests, Build_WithInvalidShape_ThrowsInvalidArgument )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        dim_t embedding_dim = 128;
        dim_t num_heads = 8;

        auto fixture = TransformerTestFixture<TPrecision>::CreateStandalone(
            TestShape::Small(),
            embedding_dim,
            num_heads
        );

        // Test wrong number of dimensions
        shape_t wrong_dims_2d = { 16, 128 };
        shape_t wrong_dims_4d = { 1, 16, 128, 1 };

        EXPECT_THROW( fixture.component->build( wrong_dims_2d ), std::invalid_argument );
        EXPECT_THROW( fixture.component->build( wrong_dims_4d ), std::invalid_argument );

        // Test mismatched embedding dimension
        shape_t wrong_embed = { 2, 16, 256 }; // Should be 128
        EXPECT_THROW( fixture.component->build( wrong_embed ), std::invalid_argument );
    }

    TYPED_TEST( TransformerCudaTests, Forward_BeforeBuild_ThrowsRuntimeError )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = TransformerTestFixture<TPrecision>::getCompatibleHeadCount( embedding_dim );

        auto fixture = TransformerTestFixture<TPrecision>::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads
        );

        CudaTensor<TPrecision> input( Device::Cuda( 0 ), fixture.shape() );
        CudaTensor<TPrecision> output( Device::Cuda( 0 ), fixture.shape() );

        EXPECT_THROW(
            fixture.component->forward( input, output ),
            std::runtime_error
        );
    }

    // ====================================================================
    // Forward Pass Tests
    // ====================================================================

    TYPED_TEST( TransformerCudaTests, Forward_WithVariousShapes_ProducesValidOutput )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto shapes = TestShape::StandardShapes();

        for ( const auto& test_shape : shapes )
        {
            dim_t embedding_dim = test_shape.dimensions.back();
            dim_t num_heads = TransformerTestFixture<TPrecision>::getCompatibleHeadCount( embedding_dim );

            auto fixture = TransformerTestFixture<TPrecision>::CreateStandalone(
                test_shape,
                embedding_dim,
                num_heads
            );

            fixture.component->build( fixture.shape() );

            CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), fixture.shape() );
            random( host_input, -0.5f, 0.5f );

            CudaTensor<TPrecision> device_input( Device::Cuda( 0 ), fixture.shape() );
            CudaTensor<TPrecision> device_output( Device::Cuda( 0 ), fixture.shape() );

            copy( host_input, device_input );

            EXPECT_NO_THROW( fixture.component->forward( device_input, device_output ) )
                << "Forward failed for shape: " << test_shape.name;

            EXPECT_EQ( device_output.size(), device_input.size() )
                << "Output size mismatch for shape: " << test_shape.name;

            EXPECT_EQ( device_output.shape(), device_input.shape() )
                << "Output shape mismatch for shape: " << test_shape.name;

            CpuTensor<TensorDataType::FP32> host_output = toHost<TensorDataType::FP32>( device_output );
            EXPECT_EQ( host_output.size(), device_input.size() )
                << "Host output size mismatch for shape: " << test_shape.name;
        }
    }

    TYPED_TEST( TransformerCudaTests, Forward_MultipleInvocations_Succeeds )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = TransformerTestFixture<TPrecision>::getCompatibleHeadCount( embedding_dim );

        auto fixture = TransformerTestFixture<TPrecision>::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads
        );

        fixture.component->build( fixture.shape() );

        CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), fixture.shape() );
        CudaTensor<TPrecision> device_input( Device::Cuda( 0 ), fixture.shape() );
        CudaTensor<TPrecision> device_output( Device::Cuda( 0 ), fixture.shape() );

        for ( int iter = 0; iter < 3; ++iter )
        {
            random( host_input, -0.5f, 0.5f );
            copy( host_input, device_input );

            EXPECT_NO_THROW( fixture.component->forward( device_input, device_output ) )
                << "Forward failed on iteration " << iter;
        }
    }

    // ====================================================================
    // Configuration Variant Tests
    // ====================================================================

    TYPED_TEST( TransformerCudaTests, NoBias_Forward_ProducesValidOutput )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = TransformerTestFixture<TPrecision>::getCompatibleHeadCount( embedding_dim );

        auto fixture = TransformerTestFixture<TPrecision>::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads,
            0,     // default hidden
            false  // no bias
        );

        fixture.component->build( fixture.shape() );

        CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), fixture.shape() );
        random( host_input, -0.5f, 0.5f );

        CudaTensor<TPrecision> device_input( Device::Cuda( 0 ), fixture.shape() );
        CudaTensor<TPrecision> device_output( Device::Cuda( 0 ), fixture.shape() );

        copy( host_input, device_input );

        EXPECT_NO_THROW( fixture.component->forward( device_input, device_output ) );
        EXPECT_EQ( device_output.size(), device_input.size() );
    }

    TYPED_TEST( TransformerCudaTests, CustomHiddenDimension_Forward_ProducesValidOutput )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = TransformerTestFixture<TPrecision>::getCompatibleHeadCount( embedding_dim );
        dim_t custom_hidden = embedding_dim * 2; // 2x instead of default 4x

        auto fixture = TransformerTestFixture<TPrecision>::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads,
            custom_hidden
        );

        fixture.component->build( fixture.shape() );

        CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), fixture.shape() );
        random( host_input, -0.5f, 0.5f );

        CudaTensor<TPrecision> device_input( Device::Cuda( 0 ), fixture.shape() );
        CudaTensor<TPrecision> device_output( Device::Cuda( 0 ), fixture.shape() );

        copy( host_input, device_input );

        EXPECT_NO_THROW( fixture.component->forward( device_input, device_output ) );
        EXPECT_EQ( device_output.shape(), device_input.shape() );
    }

    TYPED_TEST( TransformerCudaTests, VariousActivationTypes_Forward_AllSucceed )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = TransformerTestFixture<TPrecision>::getCompatibleHeadCount( embedding_dim );

        std::vector<ActivationType> activations = {
            ActivationType::Gelu
            // TODO: Uncomment when other activations are supported
            //ActivationType::Relu,
            //ActivationType::Tanh
        };

        for ( auto activation : activations )
        {
            auto fixture = TransformerTestFixture<TPrecision>::CreateStandalone(
                test_shape,
                embedding_dim,
                num_heads,
                0,
                true,
                activation
            );

            fixture.component->build( fixture.shape() );

            CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), fixture.shape() );
            random( host_input, -0.5f, 0.5f );

            CudaTensor<TPrecision> device_input( Device::Cuda( 0 ), fixture.shape() );
            CudaTensor<TPrecision> device_output( Device::Cuda( 0 ), fixture.shape() );

            copy( host_input, device_input );

            EXPECT_NO_THROW( fixture.component->forward( device_input, device_output ) )
                << "Forward failed for activation: " << static_cast<int>(activation);
        }
    }

    // ====================================================================
    // Shared Context Tests
    // ====================================================================

    TYPED_TEST( TransformerCudaTests, SharedContext_Construction_Succeeds )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = TransformerTestFixture<TPrecision>::getCompatibleHeadCount( embedding_dim );

        auto fixture = TransformerTestFixture<TPrecision>::CreateWithSharedContext(
            test_shape,
            embedding_dim,
            num_heads
        );

        ASSERT_NE( fixture.component, nullptr );
        ASSERT_NE( fixture.network, nullptr );

        std::string expected_name = "transformer_network_" + test_shape.name + "_" + PrecisionTraits<TPrecision>::name + ".transformer";
        EXPECT_EQ( fixture.component->getName(), expected_name );
    }

    TYPED_TEST( TransformerCudaTests, SharedContext_Forward_ProducesValidOutput )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = TransformerTestFixture<TPrecision>::getCompatibleHeadCount( embedding_dim );

        auto fixture = TransformerTestFixture<TPrecision>::CreateWithSharedContext(
            test_shape,
            embedding_dim,
            num_heads
        );

        fixture.component->build( fixture.shape() );

        CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), fixture.shape() );
        random( host_input, -0.5f, 0.5f );

        CudaTensor<TPrecision> device_input( Device::Cuda( 0 ), fixture.shape() );
        CudaTensor<TPrecision> device_output( Device::Cuda( 0 ), fixture.shape() );

        copy( host_input, device_input );

        EXPECT_NO_THROW( fixture.component->forward( device_input, device_output ) );
        EXPECT_EQ( device_output.size(), device_input.size() );
        EXPECT_EQ( device_output.shape(), device_input.shape() );
    }

    // ====================================================================
    // Child Component Tests
    // ====================================================================

    TYPED_TEST( TransformerCudaTests, GetNamedComponents_ReturnsExpectedChildren )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = TransformerTestFixture<TPrecision>::getCompatibleHeadCount( embedding_dim );

        auto fixture = TransformerTestFixture<TPrecision>::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads
        );

        fixture.component->build( fixture.shape() );

        auto modules = fixture.component->getNamedComponents();
        const std::string base = fixture.component->getName();

        // Check for expected child components
        EXPECT_GE( modules.size(), 6u ); // at least attn, ln1, ln2, qkv_proj, res1, res2, mlp

        EXPECT_NE( modules.find( base + ".attn" ), modules.end() );
        EXPECT_NE( modules.find( base + ".ln1" ), modules.end() );
        EXPECT_NE( modules.find( base + ".ln2" ), modules.end() );
        EXPECT_NE( modules.find( base + ".qkv_proj" ), modules.end() );
        EXPECT_NE( modules.find( base + ".res1" ), modules.end() );
        EXPECT_NE( modules.find( base + ".res2" ), modules.end() );
        EXPECT_NE( modules.find( base + ".mlp" ), modules.end() );
    }

    // ====================================================================
    // Configuration Access Tests
    // ====================================================================

    TYPED_TEST( TransformerCudaTests, GetConfig_ReturnsCorrectConfiguration )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = TransformerTestFixture<TPrecision>::getCompatibleHeadCount( embedding_dim );
        dim_t custom_hidden = embedding_dim * 3;

        auto fixture = TransformerTestFixture<TPrecision>::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads,
            custom_hidden,
            false,
            ActivationType::Gelu
        );

        const auto& config = fixture.component->getConfig();

        EXPECT_EQ( config.getEmbeddingDim(), embedding_dim );
        EXPECT_EQ( config.getNumHeads(), num_heads );
        EXPECT_EQ( config.getHiddenDimension(), custom_hidden );
        EXPECT_FALSE( config.useBias() );
        EXPECT_EQ( config.getActivationType(), ActivationType::Gelu );
    }

    // ====================================================================
    // Edge Case Tests
    // ====================================================================

    TYPED_TEST( TransformerCudaTests, EdgeCase_MinimalShape_Forward )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Minimal();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = TransformerTestFixture<TPrecision>::getCompatibleHeadCount( embedding_dim );

        auto fixture = TransformerTestFixture<TPrecision>::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads
        );

        fixture.component->build( fixture.shape() );

        CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), fixture.shape() );
        random( host_input, -0.5f, 0.5f );

        CudaTensor<TPrecision> device_input( Device::Cuda( 0 ), fixture.shape() );
        CudaTensor<TPrecision> device_output( Device::Cuda( 0 ), fixture.shape() );

        copy( host_input, device_input );

        EXPECT_NO_THROW( fixture.component->forward( device_input, device_output ) );
    }

    TYPED_TEST( TransformerCudaTests, EdgeCase_SingleHead_Forward )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = { TestShapeSize::Small, { 1, 4, 64 }, "SingleHead" };
        dim_t embedding_dim = 64;
        dim_t num_heads = 1;

        auto fixture = TransformerTestFixture<TPrecision>::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads
        );

        fixture.component->build( fixture.shape() );

        CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), fixture.shape() );
        random( host_input, -0.5f, 0.5f );

        CudaTensor<TPrecision> device_input( Device::Cuda( 0 ), fixture.shape() );
        CudaTensor<TPrecision> device_output( Device::Cuda( 0 ), fixture.shape() );

        copy( host_input, device_input );

        EXPECT_NO_THROW( fixture.component->forward( device_input, device_output ) );
    }

    // ====================================================================
    // CPU/CUDA Equivalence Test
    // ====================================================================

    TYPED_TEST( TransformerCudaTests, Forward_ComparedToCpu_ProducesEquivalentOutput )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = { TestShapeSize::Small, { 1, 2, 64 }, "Equivalence" };
        dim_t embedding_dim = 64;
        dim_t num_heads = 2;

        TransformerConfig config( embedding_dim, num_heads );

        auto cpu_transformer = std::make_shared<Transformer<DeviceType::Cpu, TensorDataType::FP32>>(
            "cpu_equiv",
            config,
            Device::Cpu()
        );

        auto cuda_fixture = TransformerTestFixture<TPrecision>::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads
        );

        cpu_transformer->build( test_shape.dimensions );
        cuda_fixture.component->build( cuda_fixture.shape() );

        CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), test_shape.dimensions );
        random( host_input, -0.1f, 0.1f ); // Smaller range for stability

        CpuTensor<TensorDataType::FP32> cpu_output( Device::Cpu(), test_shape.dimensions );
        cpu_transformer->forward( host_input, cpu_output );

        CudaTensor<TPrecision> device_input( Device::Cuda( 0 ), test_shape.dimensions );
        CudaTensor<TPrecision> device_output( Device::Cuda( 0 ), test_shape.dimensions );
        copy( host_input, device_input );
        cuda_fixture.component->forward( device_input, device_output );

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
            << "Max difference: " << max_diff << "\n"
            << "Tolerance: " << epsilon;
    }
}