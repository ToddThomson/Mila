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
    class GptBlockTestNetwork : public Network<DeviceType::Cuda, TPrecision>
    {
    private:
        std::unique_ptr<IExecutionContext> owned_context_;
        std::shared_ptr<GptBlock<DeviceType::Cuda, TPrecision>> block_;
        GptBlockConfig config_;

    public:
        explicit GptBlockTestNetwork(
            const std::string& name,
            const GptBlockConfig& config,
            DeviceId device_id )
            : Network<DeviceType::Cuda, TPrecision>( name ),
            owned_context_( createExecutionContext( device_id ) ),
            config_( config )
        {
            createGraph();
            this->setExecutionContext( owned_context_.get() );
        }

        std::shared_ptr<GptBlock<DeviceType::Cuda, TPrecision>> getBlock() const
        {
            return block_;
        }

        const GptBlockConfig& getConfig() const
        {
            return config_;
        }

        const ComponentType getType() const override
        {
            return ComponentType::MockComponent;
        }

    protected:
        void save_( ModelArchive& archive, SerializationMode mode ) const override
        {
            // Minimal implementation for testing
        }

    private:
        void createGraph()
        {
            block_ = std::make_shared<GptBlock<DeviceType::Cuda, TPrecision>>(
                this->getName() + ".gptblock",
                config_
            );

            this->addComponent( block_ );
        }
    };

    // ====================================================================
    // Test Fixture Structure
    // ====================================================================

    template<TensorDataType TPrecision>
    struct GptBlockTestFixture
    {
        TestShape test_shape;
        GptBlockConfig config;
        std::shared_ptr<GptBlock<DeviceType::Cuda, TPrecision>> component;
        std::unique_ptr<GptBlockTestNetwork<TPrecision>> network;
        dim_t embedding_dim;
        dim_t num_heads;
        bool is_training;
        bool use_shared_context;

        GptBlockTestFixture()
            : config( 1, 1 ), embedding_dim( 0 ), num_heads( 0 )
        {}

        static GptBlockTestFixture CreateStandalone(
            TestShape shape,
            dim_t embedding_dim,
            dim_t num_heads,
            dim_t hidden_dim = 0,
            bool use_bias = true,
            ActivationType activation = ActivationType::Gelu,
            bool is_training = false )
        {
            GptBlockTestFixture fixture;
            fixture.test_shape = shape;
            fixture.embedding_dim = embedding_dim;
            fixture.num_heads = num_heads;
            fixture.is_training = is_training;
            fixture.use_shared_context = false;

            fixture.config = GptBlockConfig( embedding_dim, num_heads );

            if ( hidden_dim > 0 )
            {
                fixture.config.withHiddenSize( hidden_dim );
            }

            fixture.config.withBias( use_bias )
                .withActivation( activation );

            std::string name = "gptblock_cuda_" + shape.name + "_" + PrecisionTraits<TPrecision>::name;

            fixture.component = std::make_shared<GptBlock<DeviceType::Cuda, TPrecision>>(
                name,
                fixture.config,
                Device::Cuda( 0 )
            );

            // Build before enabling training to satisfy Component lifecycle contract.
            if ( fixture.is_training )
            {
                fixture.component->build( fixture.shape() );
                fixture.component->setTraining( true );
            }

            return fixture;
        }

        static GptBlockTestFixture CreateWithSharedContext(
            TestShape shape,
            dim_t embedding_dim,
            dim_t num_heads,
            dim_t hidden_dim = 0,
            bool use_bias = true,
            ActivationType activation = ActivationType::Gelu,
            bool is_training = false )
        {
            GptBlockTestFixture fixture;
            fixture.test_shape = shape;
            fixture.embedding_dim = embedding_dim;
            fixture.num_heads = num_heads;
            fixture.is_training = is_training;
            fixture.use_shared_context = true;

            fixture.config = GptBlockConfig( embedding_dim, num_heads );

            if ( hidden_dim > 0 )
            {
                fixture.config.withHiddenSize( hidden_dim );
            }

            fixture.config.withBias( use_bias )
                .withActivation( activation );

            std::string name = "gptblock_network_" + shape.name + "_" + PrecisionTraits<TPrecision>::name;

            fixture.network = std::make_unique<GptBlockTestNetwork<TPrecision>>(
                name,
                fixture.config,
                Device::Cuda( 0 )
            );

            fixture.component = fixture.network->getBlock();

            // Build network before enabling training to satisfy Component lifecycle contract.
            if ( fixture.is_training )
            {
                fixture.network->build( fixture.shape() );
                fixture.network->setTraining( true );
            }

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
    class GptBlockCudaTests : public testing::Test
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
        // TODO: Uncomment when FP16 has been implemented
        // PrecisionType<TensorDataType::FP16>
    >;

    TYPED_TEST_SUITE( GptBlockCudaTests, PrecisionTypes );

    // ====================================================================
    // Constructor Tests
    // ====================================================================

    TYPED_TEST( GptBlockCudaTests, Constructor_WithValidDeviceId_CreatesComponent )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = GptBlockTestFixture<TPrecision>::getCompatibleHeadCount( embedding_dim );

        auto fixture = GptBlockTestFixture<TPrecision>::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads
        );

        ASSERT_NE( fixture.component, nullptr );
        EXPECT_EQ( fixture.component->getDeviceType(), DeviceType::Cuda );
        EXPECT_EQ( fixture.component->getDeviceId().type, DeviceType::Cuda );
    }

    TYPED_TEST( GptBlockCudaTests, Constructor_WithoutDeviceId_CreatesComponent )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = GptBlockTestFixture<TPrecision>::getCompatibleHeadCount( embedding_dim );

        auto fixture = GptBlockTestFixture<TPrecision>::CreateWithSharedContext(
            test_shape,
            embedding_dim,
            num_heads
        );

        ASSERT_NE( fixture.component, nullptr );
        ASSERT_NE( fixture.network, nullptr );
    }

    TYPED_TEST( GptBlockCudaTests, Constructor_WithInvalidConfiguration_ThrowsInvalidArgument )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        // Test invalid embedding dimension
        EXPECT_THROW(
            GptBlockConfig( 0, 8 ),
            std::invalid_argument
        );

        // Test invalid num_heads
        EXPECT_THROW(
            GptBlockConfig( 128, 0 ),
            std::invalid_argument
        );

        // Test non-divisible heads
        EXPECT_THROW(
            GptBlockConfig( 127, 8 ),
            std::invalid_argument
        );

        // Test with valid config but wrong device
        GptBlockConfig valid_config( 128, 8 );
        EXPECT_THROW(
            (GptBlock<DeviceType::Cuda, TPrecision>( "invalid_device", valid_config, Device::Cpu() )),
            std::invalid_argument
        );
    }

    TYPED_TEST( GptBlockCudaTests, Constructor_WithInvalidDeviceType_ThrowsInvalidArgument )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        GptBlockConfig config( 128, 8 );

        EXPECT_THROW(
            (GptBlock<DeviceType::Cuda, TPrecision>( "invalid_device", config, Device::Cpu() )),
            std::invalid_argument
        );
    }

    // ====================================================================
    // Basic Property Tests
    // ====================================================================

    TYPED_TEST( GptBlockCudaTests, GetDeviceType_AfterConstruction_ReturnsCuda )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = GptBlockTestFixture<TPrecision>::getCompatibleHeadCount( embedding_dim );

        auto fixture = GptBlockTestFixture<TPrecision>::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads
        );

        EXPECT_EQ( fixture.component->getDeviceType(), DeviceType::Cuda );

        auto device = fixture.component->getDeviceId();
        EXPECT_EQ( device.type, DeviceType::Cuda );
    }

    TYPED_TEST( GptBlockCudaTests, GetName_AfterConstruction_ReturnsCorrectName )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = GptBlockTestFixture<TPrecision>::getCompatibleHeadCount( embedding_dim );

        auto fixture = GptBlockTestFixture<TPrecision>::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads
        );

        std::string expected_name = "gptblock_cuda_" + test_shape.name + "_" + PrecisionTraits<TPrecision>::name;
        EXPECT_EQ( fixture.component->getName(), expected_name );
    }

    TYPED_TEST( GptBlockCudaTests, IsTraining_InferenceFixture_ReturnsFalse )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = GptBlockTestFixture<TPrecision>::getCompatibleHeadCount( embedding_dim );

        auto fixture = GptBlockTestFixture<TPrecision>::CreateStandalone(
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

    TYPED_TEST( GptBlockCudaTests, IsTraining_TrainingFixture_ReturnsTrue )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = GptBlockTestFixture<TPrecision>::getCompatibleHeadCount( embedding_dim );

        auto fixture = GptBlockTestFixture<TPrecision>::CreateStandalone(
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

    TYPED_TEST( GptBlockCudaTests, SetTraining_TogglingMode_UpdatesState )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = GptBlockTestFixture<TPrecision>::getCompatibleHeadCount( embedding_dim );

        auto fixture = GptBlockTestFixture<TPrecision>::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads
        );

        EXPECT_FALSE( fixture.component->isTraining() );

        // Build before enabling training to satisfy Component lifecycle contract.
        fixture.component->build( fixture.shape() );

        fixture.component->setTraining( true );
        EXPECT_TRUE( fixture.component->isTraining() );

        fixture.component->setTraining( false );
        EXPECT_FALSE( fixture.component->isTraining() );
    }

    TYPED_TEST( GptBlockCudaTests, ParameterCount_AfterBuild_ReturnsNonZero )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = GptBlockTestFixture<TPrecision>::getCompatibleHeadCount( embedding_dim );

        auto fixture = GptBlockTestFixture<TPrecision>::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads
        );

        fixture.component->build( fixture.shape() );

        EXPECT_GT( fixture.component->parameterCount(), 0u );
    }

    TYPED_TEST( GptBlockCudaTests, ToString_AfterConstruction_ContainsComponentInfo )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = GptBlockTestFixture<TPrecision>::getCompatibleHeadCount( embedding_dim );

        auto fixture = GptBlockTestFixture<TPrecision>::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads
        );

        std::string output = fixture.component->toString();

        EXPECT_NE( output.find( "GptBlock" ), std::string::npos );
        EXPECT_NE( output.find( fixture.component->getName() ), std::string::npos );
        EXPECT_NE( output.find( "Number of heads" ), std::string::npos );
    }

    TYPED_TEST( GptBlockCudaTests, Synchronize_AfterConstruction_Succeeds )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = GptBlockTestFixture<TPrecision>::getCompatibleHeadCount( embedding_dim );

        auto fixture = GptBlockTestFixture<TPrecision>::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads
        );

        EXPECT_NO_THROW( fixture.component->synchronize() );
    }

    // ====================================================================
    // Build State Tests
    // ====================================================================

    TYPED_TEST( GptBlockCudaTests, IsBuilt_BeforeBuild_ReturnsFalse )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = GptBlockTestFixture<TPrecision>::getCompatibleHeadCount( embedding_dim );

        auto fixture = GptBlockTestFixture<TPrecision>::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads
        );

        EXPECT_FALSE( fixture.component->isBuilt() );
    }

    TYPED_TEST( GptBlockCudaTests, Build_WithVariousShapes_SetsBuiltState )
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
            dim_t num_heads = GptBlockTestFixture<TPrecision>::getCompatibleHeadCount( embedding_dim );

            auto fixture = GptBlockTestFixture<TPrecision>::CreateStandalone(
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

    TYPED_TEST( GptBlockCudaTests, Build_WithInvalidShape_ThrowsInvalidArgument )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        dim_t embedding_dim = 128;
        dim_t num_heads = 8;

        auto fixture = GptBlockTestFixture<TPrecision>::CreateStandalone(
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

    TYPED_TEST( GptBlockCudaTests, Forward_BeforeBuild_ThrowsRuntimeError )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = GptBlockTestFixture<TPrecision>::getCompatibleHeadCount( embedding_dim );

        auto fixture = GptBlockTestFixture<TPrecision>::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads
        );

        CudaTensor<TPrecision> input( Device::Cuda( 0 ), fixture.shape() );

        EXPECT_THROW(
            fixture.component->forward( input ),
            std::runtime_error
        );
    }

    // ====================================================================
    // Forward Pass Tests
    // ====================================================================

    TYPED_TEST( GptBlockCudaTests, Forward_WithVariousShapes_ProducesValidOutput )
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
            dim_t num_heads = GptBlockTestFixture<TPrecision>::getCompatibleHeadCount( embedding_dim );

            auto fixture = GptBlockTestFixture<TPrecision>::CreateStandalone(
                test_shape,
                embedding_dim,
                num_heads
            );

            fixture.component->build( fixture.shape() );

            CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), fixture.shape() );
            random( host_input, -0.5f, 0.5f );

            CudaTensor<TPrecision> device_input( Device::Cuda( 0 ), fixture.shape() );
            copy( host_input, device_input );

            EXPECT_NO_THROW(
                {
                    auto& out = fixture.component->forward( device_input );
                    (void)out;
                }
            ) << "Forward failed for shape: " << test_shape.name;

            auto& out = fixture.component->forward( device_input );

            EXPECT_EQ( out.shape(), device_input.shape() )
                << "Output shape mismatch for shape: " << test_shape.name;

            auto host_output = toHost<TensorDataType::FP32>( out );

            EXPECT_EQ( host_output.size(), out.size() )
                << "Host output size mismatch for shape: " << test_shape.name;
        }
    }

    TYPED_TEST( GptBlockCudaTests, Forward_MultipleInvocations_Succeeds )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = GptBlockTestFixture<TPrecision>::getCompatibleHeadCount( embedding_dim );

        auto fixture = GptBlockTestFixture<TPrecision>::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads
        );

        fixture.component->build( fixture.shape() );

        CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), fixture.shape() );
        CudaTensor<TPrecision> device_input( Device::Cuda( 0 ), fixture.shape() );

        for ( int iter = 0; iter < 3; ++iter )
        {
            random( host_input, -0.5f, 0.5f );
            copy( host_input, device_input );

            EXPECT_NO_THROW(
                {
                    auto& out = fixture.component->forward( device_input );
                    (void)out;
                }
            ) << "Forward failed on iteration " << iter;
        }
    }

    // ====================================================================
    // Configuration Variant Tests
    // ====================================================================

    TYPED_TEST( GptBlockCudaTests, NoBias_Forward_ProducesValidOutput )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = GptBlockTestFixture<TPrecision>::getCompatibleHeadCount( embedding_dim );

        auto fixture = GptBlockTestFixture<TPrecision>::CreateStandalone(
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
        copy( host_input, device_input );

        EXPECT_NO_THROW(
            {
                auto& out = fixture.component->forward( device_input );
                (void)out;
            }
        );

        auto& out = fixture.component->forward( device_input );
        EXPECT_EQ( out.size(), device_input.size() );
    }

    TYPED_TEST( GptBlockCudaTests, CustomHiddenDimension_Forward_ProducesValidOutput )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = GptBlockTestFixture<TPrecision>::getCompatibleHeadCount( embedding_dim );
        dim_t custom_hidden = embedding_dim * 2; // 2x instead of default 4x

        auto fixture = GptBlockTestFixture<TPrecision>::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads,
            custom_hidden
        );

        fixture.component->build( fixture.shape() );

        CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), fixture.shape() );
        random( host_input, -0.5f, 0.5f );

        CudaTensor<TPrecision> device_input( Device::Cuda( 0 ), fixture.shape() );
        copy( host_input, device_input );

        EXPECT_NO_THROW(
            {
                auto& out = fixture.component->forward( device_input );
                (void)out;
            }
        );

        auto& out = fixture.component->forward( device_input );
        EXPECT_EQ( out.shape(), device_input.shape() );
    }

    TYPED_TEST( GptBlockCudaTests, VariousActivationTypes_Forward_AllSucceed )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = GptBlockTestFixture<TPrecision>::getCompatibleHeadCount( embedding_dim );

        std::vector<ActivationType> activations = {
            ActivationType::Gelu
            // TODO: Uncomment when other activations are supported
            //ActivationType::Relu,
            //ActivationType::Tanh
        };

        for ( auto activation : activations )
        {
            auto fixture = GptBlockTestFixture<TPrecision>::CreateStandalone(
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
            copy( host_input, device_input );

            EXPECT_NO_THROW(
                {
                    auto& out = fixture.component->forward( device_input );
                    (void)out;
                }
            ) << "Forward failed for activation: " << static_cast<int>(activation);
        }
    }

    // ====================================================================
    // Shared Context Tests
    // ====================================================================

    TYPED_TEST( GptBlockCudaTests, SharedContext_Construction_Succeeds )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = GptBlockTestFixture<TPrecision>::getCompatibleHeadCount( embedding_dim );

        auto fixture = GptBlockTestFixture<TPrecision>::CreateWithSharedContext(
            test_shape,
            embedding_dim,
            num_heads
        );

        ASSERT_NE( fixture.component, nullptr );
        ASSERT_NE( fixture.network, nullptr );

        std::string expected_name = "gptblock_network_" + test_shape.name + "_" + PrecisionTraits<TPrecision>::name + ".gptblock";
        EXPECT_EQ( fixture.component->getName(), expected_name );
    }

    TYPED_TEST( GptBlockCudaTests, SharedContext_Forward_ProducesValidOutput )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = GptBlockTestFixture<TPrecision>::getCompatibleHeadCount( embedding_dim );

        auto fixture = GptBlockTestFixture<TPrecision>::CreateWithSharedContext(
            test_shape,
            embedding_dim,
            num_heads
        );

        fixture.component->build( fixture.shape() );

        CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), fixture.shape() );
        random( host_input, -0.5f, 0.5f );

        CudaTensor<TPrecision> device_input( Device::Cuda( 0 ), fixture.shape() );
        copy( host_input, device_input );

        EXPECT_NO_THROW(
            {
                auto& out = fixture.component->forward( device_input );
                (void)out;
            }
        );

        auto& out = fixture.component->forward( device_input );
        EXPECT_EQ( out.shape(), device_input.shape() );
    }

    // ====================================================================
    // Child Component Tests
    // ====================================================================

    TYPED_TEST( GptBlockCudaTests, GetNamedComponents_ReturnsExpectedChildren )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = GptBlockTestFixture<TPrecision>::getCompatibleHeadCount( embedding_dim );

        auto fixture = GptBlockTestFixture<TPrecision>::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads
        );

        fixture.component->build( fixture.shape() );

        auto modules = fixture.component->getComponents();
        const std::string base = fixture.component->getName();

        // Check for expected child components
        EXPECT_GE( modules.size(), 6u ); // at least attn, ln1, ln2, qkv_proj, res1, res2, mlp

    }

    // ====================================================================
    // Edge Case Tests
    // ====================================================================

    TYPED_TEST( GptBlockCudaTests, EdgeCase_MinimalShape_Forward )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Minimal();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = GptBlockTestFixture<TPrecision>::getCompatibleHeadCount( embedding_dim );

        auto fixture = GptBlockTestFixture<TPrecision>::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads
        );

        fixture.component->build( fixture.shape() );

        CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), fixture.shape() );
        random( host_input, -0.5f, 0.5f );

        CudaTensor<TPrecision> device_input( Device::Cuda( 0 ), fixture.shape() );
        copy( host_input, device_input );

        EXPECT_NO_THROW( { auto& out = fixture.component->forward( device_input ); (void)out; } );
    }

    TYPED_TEST( GptBlockCudaTests, EdgeCase_TwoHead_Forward )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = { TestShapeSize::Small, { 1, 4, 64 }, "TwoHead" };
        dim_t embedding_dim = 64;
        dim_t num_heads = 2;

        auto fixture = GptBlockTestFixture<TPrecision>::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads
        );

        fixture.component->build( fixture.shape() );

        CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), fixture.shape() );
        random( host_input, -0.5f, 0.5f );

        CudaTensor<TPrecision> device_input( Device::Cuda( 0 ), fixture.shape() );
        copy( host_input, device_input );

        EXPECT_NO_THROW( { auto& out = fixture.component->forward( device_input ); (void)out; } );
    }

    // ====================================================================
    // CPU/CUDA Equivalence Test
    // ====================================================================

    TYPED_TEST( GptBlockCudaTests, Forward_ComparedToCpu_ProducesEquivalentOutput )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = { TestShapeSize::Small, { 2, 2, 64 }, "Equivalence" };
        dim_t embedding_dim = 64;
        dim_t num_heads = 2;

        // TJT: HS = embedding_dim / num_heads must be integer

        GptBlockConfig config( embedding_dim, num_heads );

        auto cpu_block = std::make_shared<GptBlock<DeviceType::Cpu, TensorDataType::FP32>>(
            "cpu_equiv",
            config,
            Device::Cpu()
        );

        auto cuda_fixture = GptBlockTestFixture<TPrecision>::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads
        );

        cpu_block->build( test_shape.dimensions );
        cuda_fixture.component->build( cuda_fixture.shape() );

        CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), test_shape.dimensions );
        random( host_input, -0.1f, 0.1f ); // Smaller range for stability

        // CPU forward (new API)
        ASSERT_NO_THROW( { auto& cpu_out = cpu_block->forward( host_input ); (void)cpu_out; } );
        auto& cpu_out = cpu_block->forward( host_input );

        // CUDA forward (new API)
        CudaTensor<TPrecision> device_input( Device::Cuda( 0 ), test_shape.dimensions );
        copy( host_input, device_input );

        ASSERT_NO_THROW( { auto& cuda_out = cuda_fixture.component->forward( device_input ); (void)cuda_out; } );
        auto& cuda_out = cuda_fixture.component->forward( device_input );
        cuda_fixture.component->synchronize();

        CpuTensor<TensorDataType::FP32> cuda_output_host = toHost<TensorDataType::FP32>( cuda_out );

        const float epsilon = PrecisionTraits<TPrecision>::tolerance;
        bool all_close = true;
        size_t first_mismatch_idx = 0;
        float max_diff = 0.0f;

        for ( size_t i = 0; i < cpu_out.size(); ++i )
        {
            float cpu_val = cpu_out.data()[ i ];
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
            << "CPU value: " << cpu_out.data()[ first_mismatch_idx ] << "\n"
            << "CUDA value: " << cuda_output_host.data()[ first_mismatch_idx ] << "\n"
            << "Max difference: " << max_diff << "\n"
            << "Tolerance: " << epsilon;
    }

    // ====================================================================
    // Backward / CPU-CUDA Equivalence Tests
    // ====================================================================

    TYPED_TEST( GptBlockCudaTests, Backward_CPU_CUDA_Equivalence )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        // Small deterministic shape used for backward equivalence checks
        TestShape test_shape = { TestShapeSize::Small, { 2, 2, 8 }, "BackwardEquiv" };
        dim_t embedding_dim = 8;
        dim_t num_heads = 2;

        GptBlockConfig config( embedding_dim, num_heads );

        // CPU block (FP32)
        auto cpu_block = std::make_shared<GptBlock<DeviceType::Cpu, TensorDataType::FP32>>(
            "cpu_backward_equiv",
            config,
            Device::Cpu()
        );

        // CUDA block constructed the same way as the CPU block (FP32 on CUDA)
        auto cuda_block = std::make_shared<GptBlock<DeviceType::Cuda, TensorDataType::FP32>>(
            "cuda_backward_equiv",
            config,
            Device::Cuda( 0 )
        );

        // Build and enable training for both components
        cpu_block->build( test_shape.dimensions );
        cpu_block->setTraining( true );

        cuda_block->build( test_shape.dimensions );
        cuda_block->setTraining( true );

        // Create deterministic input with known seed
        Mila::Core::RandomGenerator::getInstance().setSeed( 12345 );

        // Prepare inputs and grads
        CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), test_shape.dimensions );
        random( host_input, -0.1f, 0.1f );

        auto& cpu_out = cpu_block->forward( host_input );

        CudaTensor<TensorDataType::FP32> device_input( Device::Cuda( 0 ), test_shape.dimensions );
        copy( host_input, device_input );

        auto& cuda_out = cuda_block->forward( device_input );
        cuda_block->synchronize();

        CpuTensor<TensorDataType::FP32> host_output_grad( Device::Cpu(), test_shape.dimensions );

        float* data = host_output_grad.data();
        const float grad_scale = 0.5f;
        for ( size_t i = 0; i < host_output_grad.size(); ++i )
        {
            data[ i ] = host_input.data()[ i ] * grad_scale;
        }

        cpu_block->zeroGradients();

        auto& cpu_in_grad = cpu_block->backward( host_input, host_output_grad );

        CudaTensor<TensorDataType::FP32> device_output_grad( Device::Cuda( 0 ), test_shape.dimensions );
        copy( host_output_grad, device_output_grad );

        cuda_block->zeroGradients();

        auto& cuda_in_grad = cuda_block->backward( device_input, device_output_grad );
        cuda_block->synchronize();

        CpuTensor<TensorDataType::FP32> cuda_input_grad_host = toHost<TensorDataType::FP32>( cuda_in_grad );

        std::cout << "CPU backward input gradient: " << std::endl;
        std::cout << cpu_in_grad.toString( true ) << std::endl;
        std::cout << "CUDA backward input gradient: " << std::endl;
        std::cout << cuda_input_grad_host.toString( true ) << std::endl;

        // Compare elementwise
        const float epsilon = PrecisionTraits<TPrecision>::tolerance;
        bool all_close = true;
        size_t first_mismatch = 0;
        float max_diff = 0.0f;

        for ( size_t i = 0; i < cpu_in_grad.size(); ++i )
        {
            float cpu_val = cpu_in_grad.data()[ i ];
            float cuda_val = cuda_input_grad_host.data()[ i ];
            float diff = std::abs( cpu_val - cuda_val );

            if ( diff > max_diff ) max_diff = diff;

            if ( diff > epsilon )
            {
                all_close = false;
                first_mismatch = i;
                break;
            }
        }

        EXPECT_TRUE( all_close )
            << "CPU and CUDA backward input gradients differ\n"
            << "First mismatch index: " << first_mismatch << "\n"
            << "CPU value: " << cpu_in_grad.data()[ first_mismatch ] << "\n"
            << "CUDA value: " << cuda_input_grad_host.data()[ first_mismatch ] << "\n"
            << "Max difference: " << max_diff << "\n"
            << "Tolerance: " << epsilon;
    }

    TYPED_TEST( GptBlockCudaTests, Backward_Sanity_SmallDeterministic )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        // TJT: This test fails intermittently

        //    The difference between cpu_in_grad.data()[ i ] and cuda_input_grad_host.data()[ i ] is 0.27118046581745148, which exceeds epsilon, where
        //    cpu_in_grad.data()[ i ] evaluates to 0.32491850852966309,
        //    cuda_input_grad_host.data()[ i ] evaluates to 0.053738042712211609, and
        //    epsilon evaluates to 9.9999997473787516e-05.
        //    Mismatch at element 0
        //    #2 - The difference between cpu_in_grad.data()[ i ] and cuda_input_grad_host.data()[ i ] is 0.023988515138626099, which exceeds epsilon, where
        //    cpu_in_grad.data()[ i ] evaluates to 0.066527165472507477,
        //    cuda_input_grad_host.data()[ i ] evaluates to 0.042538650333881378, and
        //    epsilon evaluates to 9.9999997473787516e-05.
        //    Mismatch at element 1

        constexpr TensorDataType TPrecision = TypeParam::value;

        // Minimal deterministic case (batch=1, seq=1) to ensure shapes and indexing are correct
        TestShape test_shape = { TestShapeSize::Minimal, { 1, 1, 64 }, "BackwardMinimal" };
        dim_t embedding_dim = 64;
        dim_t num_heads = 1;

        GptBlockConfig config( embedding_dim, num_heads );

        auto cpu_block = std::make_shared<GptBlock<DeviceType::Cpu, TensorDataType::FP32>>(
            "cpu_backward_minimal",
            config,
            Device::Cpu()
        );

        auto cuda_block = std::make_shared<GptBlock<DeviceType::Cuda, TensorDataType::FP32>>(
            "cuda_backward_minimal",
            config,
            Device::Cuda( 0 )
        );

        cpu_block->build( test_shape.dimensions );
        cpu_block->setTraining( true );

        cuda_block->build( test_shape.dimensions );
        cuda_block->setTraining( true );

        // Create deterministic input with known seed
        Mila::Core::RandomGenerator::getInstance().setSeed( 12345 );

        CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), test_shape.dimensions );
        random( host_input, -0.1f, 0.1f );

        ASSERT_NO_THROW( { auto& cpu_out = cpu_block->forward( host_input ); (void)cpu_out; } );
        auto& cpu_out = cpu_block->forward( host_input );

        CudaTensor<TensorDataType::FP32> device_input( Device::Cuda( 0 ), test_shape.dimensions );
        copy( host_input, device_input );

        ASSERT_NO_THROW( { auto& cuda_out = cuda_block->forward( device_input ); (void)cuda_out; } );
        auto& cuda_out = cuda_block->forward( device_input );
        cuda_block->synchronize();

        // deterministic output grad
        CpuTensor<TensorDataType::FP32> host_output_grad( Device::Cpu(), test_shape.dimensions );
        float* data = host_output_grad.data();
        const float grad_scale = 0.5f;
        for ( size_t i = 0; i < host_output_grad.size(); ++i )
        {
            data[ i ] = host_input.data()[ i ] * grad_scale;
        }

        cpu_block->zeroGradients();
        auto& cpu_in_grad = cpu_block->backward( host_input, host_output_grad );

        CudaTensor<TensorDataType::FP32> device_output_grad( Device::Cuda( 0 ), test_shape.dimensions );
        copy( host_output_grad, device_output_grad );

        cuda_block->zeroGradients();
        auto& cuda_in_grad = cuda_block->backward( device_input, device_output_grad );
        cuda_block->synchronize();

        CpuTensor<TensorDataType::FP32> cuda_input_grad_host = toHost<TensorDataType::FP32>( cuda_in_grad );

        // Sanity checks: sizes and exact-ish equality
        EXPECT_EQ( cpu_in_grad.size(), cuda_input_grad_host.size() );

        const float epsilon = PrecisionTraits<TPrecision>::tolerance;
        for ( size_t i = 0; i < cpu_in_grad.size(); ++i )
        {
            EXPECT_NEAR( cpu_in_grad.data()[ i ], cuda_input_grad_host.data()[ i ], epsilon )
                << "Mismatch at element " << i;
        }
    }
}