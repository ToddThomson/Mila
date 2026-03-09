#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>
#include <exception>
#include <cstdint>

import Mila;

namespace CompositeComponents_Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    using CpuTensor = Tensor<TensorDataType::FP32, CpuMemoryResource>;

    namespace
    {
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
                return { Small(), Medium() /* Large() TJT: This takes a bit too long. Perhaps re-enable after optimization work TBD */ };
            }
        };
    }

    // ====================================================================
    // Test Network for Shared Context Testing
    // ====================================================================

    class GptBlockTestNetwork : public Network<DeviceType::Cpu, TensorDataType::FP32>
    {
    private:
        std::unique_ptr<IExecutionContext> owned_context_;
        std::shared_ptr<GptBlock<DeviceType::Cpu, TensorDataType::FP32>> block_;
        GptBlockConfig config_;

    public:
        explicit GptBlockTestNetwork(
            const std::string& name,
            const GptBlockConfig& config,
            DeviceId device_id )
            : Network<DeviceType::Cpu, TensorDataType::FP32>( name ),
            owned_context_( createExecutionContext( device_id ) ),
            config_( config )
        {
            createGraph();
            this->setExecutionContext( owned_context_.get() );
        }

        std::shared_ptr<GptBlock<DeviceType::Cpu, TensorDataType::FP32>> getBlock() const
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

        MemoryStats getMemoryStats() const override
        {
            MemoryStats stats;

            for ( const auto& child : this->getComponents() )
            {
                stats += child->getMemoryStats();
            }

            return stats;
        }

    protected:
        void save_( ModelArchive& archive, SerializationMode mode ) const override
        {
            // Minimal implementation for testing
        }

    private:
        void createGraph()
        {
            block_ = std::make_shared<GptBlock<DeviceType::Cpu, TensorDataType::FP32>>(
                this->getName() + ".gptblock",
                config_
            );

            this->addComponent( block_ );
        }
    };

    // ====================================================================
    // Test Fixture Structure
    // ====================================================================

    struct GptBlockTestFixture
    {
        TestShape test_shape;
        GptBlockConfig config;
        std::shared_ptr<GptBlock<DeviceType::Cpu, TensorDataType::FP32>> component;
        std::unique_ptr<GptBlockTestNetwork> network;
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

            std::string name = "gptblock_cpu_" + shape.name;

            fixture.component = std::make_shared<GptBlock<DeviceType::Cpu, TensorDataType::FP32>>(
                name,
                fixture.config,
                Device::Cpu()
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

            std::string name = "gptblock_network_" + shape.name;

            fixture.network = std::make_unique<GptBlockTestNetwork>(
                name,
                fixture.config,
                Device::Cpu()
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
    // Test Class
    // ====================================================================

    class GptBlockCpuTests : public testing::Test
    {
    protected:
        void SetUp() override
        {
            // CPU is always available
        }
    };

    // ====================================================================
    // Constructor Tests
    // ====================================================================

    TEST_F( GptBlockCpuTests, Constructor_WithValidDeviceId_CreatesComponent )
    {
        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = GptBlockTestFixture::getCompatibleHeadCount( embedding_dim );

        auto fixture = GptBlockTestFixture::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads
        );

        ASSERT_NE( fixture.component, nullptr );
        EXPECT_EQ( fixture.component->getDeviceType(), DeviceType::Cpu );
        EXPECT_EQ( fixture.component->getDeviceId().type, DeviceType::Cpu );
    }

    TEST_F( GptBlockCpuTests, Constructor_WithoutDeviceId_CreatesComponent )
    {
        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = GptBlockTestFixture::getCompatibleHeadCount( embedding_dim );

        auto fixture = GptBlockTestFixture::CreateWithSharedContext(
            test_shape,
            embedding_dim,
            num_heads
        );

        ASSERT_NE( fixture.component, nullptr );
        ASSERT_NE( fixture.network, nullptr );
    }

    TEST_F( GptBlockCpuTests, Constructor_WithInvalidConfiguration_ThrowsInvalidArgument )
    {
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

        // Test with valid config but wrong device type
        GptBlockConfig valid_config( 128, 8 );

        // CPU component constructed with CUDA device should throw
        EXPECT_THROW(
            (GptBlock<DeviceType::Cpu, TensorDataType::FP32>( "invalid_device", valid_config, Device::Cuda( 0 ) )),
            std::invalid_argument
        );
    }

    TEST_F( GptBlockCpuTests, Constructor_WithInvalidDeviceType_ThrowsInvalidArgument )
    {
        GptBlockConfig config( 128, 8 );

        EXPECT_THROW(
            (GptBlock<DeviceType::Cpu, TensorDataType::FP32>( "invalid_device", config, Device::Cuda( 0 ) )),
            std::invalid_argument
        );
    }

    // ====================================================================
    // Basic Property Tests
    // ====================================================================

    TEST_F( GptBlockCpuTests, GetDeviceType_AfterConstruction_ReturnsCpu )
    {
        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = GptBlockTestFixture::getCompatibleHeadCount( embedding_dim );

        auto fixture = GptBlockTestFixture::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads
        );

        EXPECT_EQ( fixture.component->getDeviceType(), DeviceType::Cpu );

        auto device = fixture.component->getDeviceId();
        EXPECT_EQ( device.type, DeviceType::Cpu );
    }

    TEST_F( GptBlockCpuTests, GetName_AfterConstruction_ReturnsCorrectName )
    {
        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = GptBlockTestFixture::getCompatibleHeadCount( embedding_dim );

        auto fixture = GptBlockTestFixture::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads
        );

        std::string expected_name = "gptblock_cpu_" + test_shape.name;
        EXPECT_EQ( fixture.component->getName(), expected_name );
    }

    TEST_F( GptBlockCpuTests, IsTraining_InferenceFixture_ReturnsFalse )
    {
        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = GptBlockTestFixture::getCompatibleHeadCount( embedding_dim );

        auto fixture = GptBlockTestFixture::CreateStandalone(
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

    TEST_F( GptBlockCpuTests, IsTraining_TrainingFixture_ReturnsTrue )
    {
        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = GptBlockTestFixture::getCompatibleHeadCount( embedding_dim );

        auto fixture = GptBlockTestFixture::CreateStandalone(
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

    TEST_F( GptBlockCpuTests, SetTraining_TogglingMode_UpdatesState )
    {
        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = GptBlockTestFixture::getCompatibleHeadCount( embedding_dim );

        auto fixture = GptBlockTestFixture::CreateStandalone(
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

    TEST_F( GptBlockCpuTests, ParameterCount_AfterBuild_ReturnsNonZero )
    {
        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = GptBlockTestFixture::getCompatibleHeadCount( embedding_dim );

        auto fixture = GptBlockTestFixture::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads
        );

        fixture.component->build( fixture.shape() );

        EXPECT_GT( fixture.component->parameterCount(), 0u );
    }

    TEST_F( GptBlockCpuTests, ToString_AfterConstruction_ContainsComponentInfo )
    {
        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = GptBlockTestFixture::getCompatibleHeadCount( embedding_dim );

        auto fixture = GptBlockTestFixture::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads
        );

        std::string output = fixture.component->toString();

        EXPECT_NE( output.find( "GptBlock" ), std::string::npos );
        EXPECT_NE( output.find( fixture.component->getName() ), std::string::npos );
        EXPECT_NE( output.find( "Number of heads" ), std::string::npos );
    }

    TEST_F( GptBlockCpuTests, Synchronize_AfterConstruction_Succeeds )
    {
        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = GptBlockTestFixture::getCompatibleHeadCount( embedding_dim );

        auto fixture = GptBlockTestFixture::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads
        );

        EXPECT_NO_THROW( fixture.component->synchronize() );
    }

    // ====================================================================
    // Build State Tests
    // ====================================================================

    TEST_F( GptBlockCpuTests, IsBuilt_BeforeBuild_ReturnsFalse )
    {
        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = GptBlockTestFixture::getCompatibleHeadCount( embedding_dim );

        auto fixture = GptBlockTestFixture::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads
        );

        EXPECT_FALSE( fixture.component->isBuilt() );
    }

    TEST_F( GptBlockCpuTests, Build_WithVariousShapes_SetsBuiltState )
    {
        auto shapes = TestShape::StandardShapes();

        for ( const auto& test_shape : shapes )
        {
            dim_t embedding_dim = test_shape.dimensions.back();
            dim_t num_heads = GptBlockTestFixture::getCompatibleHeadCount( embedding_dim );

            auto fixture = GptBlockTestFixture::CreateStandalone(
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

    TEST_F( GptBlockCpuTests, Build_WithInvalidShape_ThrowsInvalidArgument )
    {
        dim_t embedding_dim = 128;
        dim_t num_heads = 8;

        auto fixture = GptBlockTestFixture::CreateStandalone(
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

    TEST_F( GptBlockCpuTests, Forward_BeforeBuild_ThrowsRuntimeError )
    {
        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = GptBlockTestFixture::getCompatibleHeadCount( embedding_dim );

        auto fixture = GptBlockTestFixture::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads
        );

        CpuTensor input( Device::Cpu(), fixture.shape() );

        EXPECT_THROW(
            fixture.component->forward( input ),
            std::runtime_error
        );
    }

    // ====================================================================
    // Forward Pass Tests
    // ====================================================================

    TEST_F( GptBlockCpuTests, Forward_WithVariousShapes_ProducesValidOutput )
    {
        auto shapes = TestShape::StandardShapes();

        for ( const auto& test_shape : shapes )
        {
            dim_t embedding_dim = test_shape.dimensions.back();
            dim_t num_heads = GptBlockTestFixture::getCompatibleHeadCount( embedding_dim );

            auto fixture = GptBlockTestFixture::CreateStandalone(
                test_shape,
                embedding_dim,
                num_heads
            );

            fixture.component->build( fixture.shape() );

            CpuTensor input( Device::Cpu(), fixture.shape() );

            random( input, -0.5f, 0.5f );

            // Call forward and get component-owned tensor reference
            auto& out = fixture.component->forward( input );

            EXPECT_EQ( out.size(), input.size() )
                << "Output size mismatch for shape: " << test_shape.name;

            EXPECT_EQ( out.shape(), input.shape() )
                << "Output shape mismatch for shape: " << test_shape.name;
        }
    }

    TEST_F( GptBlockCpuTests, Forward_MultipleInvocations_Succeeds )
    {
        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = GptBlockTestFixture::getCompatibleHeadCount( embedding_dim );

        auto fixture = GptBlockTestFixture::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads
        );

        fixture.component->build( fixture.shape() );

        CpuTensor input( Device::Cpu(), fixture.shape() );

        for ( int iter = 0; iter < 3; ++iter )
        {
            random( input, -0.5f, 0.5f );

            auto& out = fixture.component->forward( input );

            EXPECT_EQ( out.size(), input.size() ) << "Forward failed on iteration " << iter;
        }
    }

    TEST_F( GptBlockCpuTests, Forward_TrainingMode_ProducesValidOutput )
    {
        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = GptBlockTestFixture::getCompatibleHeadCount( embedding_dim );

        auto fixture = GptBlockTestFixture::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads,
            0,
            true,
            ActivationType::Gelu,
            true // training mode
        );

        CpuTensor input( Device::Cpu(), fixture.shape() );

        random( input, -0.5f, 0.5f );

        auto& out = fixture.component->forward( input );

        EXPECT_EQ( out.size(), input.size() );
    }

    // ====================================================================
    // Configuration Variant Tests
    // ====================================================================

    TEST_F( GptBlockCpuTests, NoBias_Forward_ProducesValidOutput )
    {
        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = GptBlockTestFixture::getCompatibleHeadCount( embedding_dim );

        auto fixture = GptBlockTestFixture::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads,
            0,     // default hidden
            false  // no bias
        );

        fixture.component->build( fixture.shape() );

        CpuTensor input( Device::Cpu(), fixture.shape() );

        random( input, -0.5f, 0.5f );

        auto& out = fixture.component->forward( input );
        EXPECT_EQ( out.size(), input.size() );
    }

    TEST_F( GptBlockCpuTests, CustomHiddenDimension_Forward_ProducesValidOutput )
    {
        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = GptBlockTestFixture::getCompatibleHeadCount( embedding_dim );
        dim_t custom_hidden = embedding_dim * 2; // 2x instead of default 4x

        auto fixture = GptBlockTestFixture::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads,
            custom_hidden
        );

        fixture.component->build( fixture.shape() );

        CpuTensor input( Device::Cpu(), fixture.shape() );

        random( input, -0.5f, 0.5f );

        auto& out = fixture.component->forward( input );
        EXPECT_EQ( out.shape(), input.shape() );
    }

    TEST_F( GptBlockCpuTests, VariousActivationTypes_Forward_AllSucceed )
    {
        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = GptBlockTestFixture::getCompatibleHeadCount( embedding_dim );

        std::vector<ActivationType> activations = {
            ActivationType::Gelu
            //ActivationType::Relu,
            //ActivationType::Tanh
        };

        for ( auto activation : activations )
        {
            auto fixture = GptBlockTestFixture::CreateStandalone(
                test_shape,
                embedding_dim,
                num_heads,
                0,
                true,
                activation
            );

            fixture.component->build( fixture.shape() );

            CpuTensor input( Device::Cpu(), fixture.shape() );

            random( input, -0.5f, 0.5f );

            auto& out = fixture.component->forward( input );

            EXPECT_EQ( out.size(), input.size() ) << "Forward failed for activation: " << static_cast<int>(activation);
        }
    }

    // ====================================================================
    // Shared Context Tests
    // ====================================================================

    TEST_F( GptBlockCpuTests, SharedContext_Construction_Succeeds )
    {
        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = GptBlockTestFixture::getCompatibleHeadCount( embedding_dim );

        auto fixture = GptBlockTestFixture::CreateWithSharedContext(
            test_shape,
            embedding_dim,
            num_heads
        );

        ASSERT_NE( fixture.component, nullptr );
        ASSERT_NE( fixture.network, nullptr );

        std::string expected_name = "gptblock_network_" + test_shape.name + ".gptblock";
        EXPECT_EQ( fixture.component->getName(), expected_name );
    }

    TEST_F( GptBlockCpuTests, SharedContext_Forward_ProducesValidOutput )
    {
        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = GptBlockTestFixture::getCompatibleHeadCount( embedding_dim );

        auto fixture = GptBlockTestFixture::CreateWithSharedContext(
            test_shape,
            embedding_dim,
            num_heads
        );

        fixture.component->build( fixture.shape() );

        CpuTensor input( Device::Cpu(), fixture.shape() );

        random( input, -0.5f, 0.5f );

        auto& out = fixture.component->forward( input );

        EXPECT_EQ( out.shape(), input.shape() );
    }

    // ====================================================================
    // Child Component Tests
    // ====================================================================

    TEST_F( GptBlockCpuTests, GetComponents_ReturnsExpectedChildren )
    {
        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = GptBlockTestFixture::getCompatibleHeadCount( embedding_dim );

        auto fixture = GptBlockTestFixture::CreateStandalone(
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

    TEST_F( GptBlockCpuTests, EdgeCase_MinimalShape_Forward )
    {
        TestShape test_shape = TestShape::Minimal();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = GptBlockTestFixture::getCompatibleHeadCount( embedding_dim );

        auto fixture = GptBlockTestFixture::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads
        );

        fixture.component->build( fixture.shape() );

        CpuTensor input( Device::Cpu(), fixture.shape() );

        random( input, -0.5f, 0.5f );

        auto& out = fixture.component->forward( input );
        (void)out;
    }

    TEST_F( GptBlockCpuTests, EdgeCase_TwoHead_Forward )
    {
        TestShape test_shape = { TestShapeSize::Small, { 1, 4, 64 }, "TwoHead" };
        dim_t embedding_dim = 64;
        dim_t num_heads = 2;

        auto fixture = GptBlockTestFixture::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads
        );

        fixture.component->build( fixture.shape() );

        CpuTensor input( Device::Cpu(), fixture.shape() );

        random( input, -0.5f, 0.5f );

        auto& out = fixture.component->forward( input );
        (void)out;
    }

    TEST_F( GptBlockCpuTests, EdgeCase_LargeHeadCount_Forward )
    {
        TestShape test_shape = { TestShapeSize::Medium, { 1, 8, 256 }, "ManyHeads" };
        dim_t embedding_dim = 256;
        dim_t num_heads = 16; // Many heads

        auto fixture = GptBlockTestFixture::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads
        );

        fixture.component->build( fixture.shape() );

        CpuTensor input( Device::Cpu(), fixture.shape() );

        random( input, -0.5f, 0.5f );

        auto& out = fixture.component->forward( input );
        (void)out;
    }

    // ====================================================================
    // Performance and Consistency Tests
    // ====================================================================

    TEST_F( GptBlockCpuTests, Forward_DeterministicOutput_WithSameInput )
    {
        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = GptBlockTestFixture::getCompatibleHeadCount( embedding_dim );

        auto fixture = GptBlockTestFixture::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads
        );

        fixture.component->build( fixture.shape() );

        CpuTensor input( Device::Cpu(), fixture.shape() );

        // Fill with specific values for deterministic test
        for ( size_t i = 0; i < input.size(); ++i )
        {
            input.data()[ i ] = 0.1f * (i % 10) - 0.5f;
        }

        // Run forward twice with same input and capture host copies of outputs
        auto& out1 = fixture.component->forward( input );
        CpuTensor out1_host = toHost<TensorDataType::FP32>( out1 );

        auto& out2 = fixture.component->forward( input );
        CpuTensor out2_host = toHost<TensorDataType::FP32>( out2 );

        // Outputs should be identical
        const float epsilon = 1e-6f;
        bool outputs_match = true;

        for ( size_t i = 0; i < out1_host.size(); ++i )
        {
            float diff = std::abs( out1_host.data()[ i ] - out2_host.data()[ i ] );
            if ( diff > epsilon )
            {
                outputs_match = false;
                break;
            }
        }

        EXPECT_TRUE( outputs_match )
            << "GptBlock should produce deterministic output for same input";
    }

    // ====================================================================
    // Numerical Stability Tests
    // ====================================================================

    TEST_F( GptBlockCpuTests, Forward_WithLargeInputValues_HandlesGracefully )
    {
        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = GptBlockTestFixture::getCompatibleHeadCount( embedding_dim );

        auto fixture = GptBlockTestFixture::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads
        );

        fixture.component->build( fixture.shape() );

        CpuTensor input( Device::Cpu(), fixture.shape() );

        // Test with larger input values
        random( input, -5.0f, 5.0f );

        auto& out = fixture.component->forward( input );

        // Check output is not NaN or infinite
        bool output_valid = true;
        for ( size_t i = 0; i < out.size(); ++i )
        {
            float val = out.data()[ i ];
            if ( std::isnan( val ) || std::isinf( val ) )
            {
                output_valid = false;
                break;
            }
        }

        EXPECT_TRUE( output_valid )
            << "GptBlock output should not contain NaN or infinite values";
    }

    TEST_F( GptBlockCpuTests, Forward_WithZeroInput_ProducesValidOutput )
    {
        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = GptBlockTestFixture::getCompatibleHeadCount( embedding_dim );

        auto fixture = GptBlockTestFixture::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads
        );

        fixture.component->build( fixture.shape() );

        CpuTensor input( Device::Cpu(), fixture.shape() );

        // Zero input
        zeros( input );

        auto& out = fixture.component->forward( input );

        EXPECT_EQ( out.size(), input.size() );
    }

    TEST_F( GptBlockCpuTests, Forward_WithSmallInputValues_HandlesGracefully )
    {
        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = GptBlockTestFixture::getCompatibleHeadCount( embedding_dim );

        auto fixture = GptBlockTestFixture::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads
        );

        fixture.component->build( fixture.shape() );

        CpuTensor input( Device::Cpu(), fixture.shape() );

        // Test with very small input values
        random( input, -1e-6f, 1e-6f );

        auto& out = fixture.component->forward( input );

        EXPECT_EQ( out.size(), input.size() );
    }

    // ====================================================================
    // Component Lifecycle Tests
    // ====================================================================

    TEST_F( GptBlockCpuTests, BuildMultipleTimes_ThrowsException )
    {
        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = GptBlockTestFixture::getCompatibleHeadCount( embedding_dim );

        auto fixture = GptBlockTestFixture::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads
        );

        // First build should succeed
        EXPECT_NO_THROW( fixture.component->build( fixture.shape() ) );
        EXPECT_TRUE( fixture.component->isBuilt() );

        // Second build should throw
        EXPECT_THROW( fixture.component->build( fixture.shape() ), std::runtime_error );
    }

    TEST_F( GptBlockCpuTests, ParameterCount_BeforeBuild_ThrowsRuntimeError )
    {
        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = GptBlockTestFixture::getCompatibleHeadCount( embedding_dim );

        auto fixture = GptBlockTestFixture::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads
        );

        EXPECT_THROW( fixture.component->parameterCount(), std::runtime_error );
    }
}