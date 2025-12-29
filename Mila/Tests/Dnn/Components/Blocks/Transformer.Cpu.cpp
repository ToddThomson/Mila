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
                return { Small(), Medium() /* Large() TJT: This takes a bit too long. Perhaps with reenable after optimization work TBD */ };
            }
        };
    }

    // ====================================================================
    // Test Network for Shared Context Testing
    // ====================================================================

    class TransformerTestNetwork : public Network<DeviceType::Cpu, TensorDataType::FP32>
    {
    private:
        std::unique_ptr<IExecutionContext> owned_context_;
        std::shared_ptr<Transformer<DeviceType::Cpu, TensorDataType::FP32>> transformer_;
        TransformerConfig config_;

    public:
        explicit TransformerTestNetwork(
            const std::string& name,
            const TransformerConfig& config,
            DeviceId device_id )
            : Network<DeviceType::Cpu, TensorDataType::FP32>( name ),
            owned_context_( createExecutionContext( device_id ) ),
            config_( config )
        {
            createGraph();
            this->setExecutionContext( owned_context_.get() );
        }

        std::shared_ptr<Transformer<DeviceType::Cpu, TensorDataType::FP32>> getTransformer() const
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
            transformer_ = std::make_shared<Transformer<DeviceType::Cpu, TensorDataType::FP32>>(
                this->getName() + ".transformer",
                config_
            );

            this->addComponent( transformer_ );
        }
    };

    // ====================================================================
    // Test Fixture Structure
    // ====================================================================

    struct TransformerTestFixture
    {
        TestShape test_shape;
        TransformerConfig config;
        std::shared_ptr<Transformer<DeviceType::Cpu, TensorDataType::FP32>> component;
        std::unique_ptr<TransformerTestNetwork> network;
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

            std::string name = "transformer_cpu_" + shape.name;

            fixture.component = std::make_shared<Transformer<DeviceType::Cpu, TensorDataType::FP32>>(
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

            std::string name = "transformer_network_" + shape.name;

            fixture.network = std::make_unique<TransformerTestNetwork>(
                name,
                fixture.config,
                Device::Cpu()
            );

            fixture.component = fixture.network->getTransformer();

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

    class TransformerCpuTests : public testing::Test
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

    TEST_F( TransformerCpuTests, Constructor_WithValidDeviceId_CreatesComponent )
    {
        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = TransformerTestFixture::getCompatibleHeadCount( embedding_dim );

        auto fixture = TransformerTestFixture::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads
        );

        ASSERT_NE( fixture.component, nullptr );
        EXPECT_EQ( fixture.component->getDeviceType(), DeviceType::Cpu );
        EXPECT_EQ( fixture.component->getDeviceId().type, DeviceType::Cpu );
    }

    TEST_F( TransformerCpuTests, Constructor_WithoutDeviceId_CreatesComponent )
    {
        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = TransformerTestFixture::getCompatibleHeadCount( embedding_dim );

        auto fixture = TransformerTestFixture::CreateWithSharedContext(
            test_shape,
            embedding_dim,
            num_heads
        );

        ASSERT_NE( fixture.component, nullptr );
        ASSERT_NE( fixture.network, nullptr );
    }

    TEST_F( TransformerCpuTests, Constructor_WithInvalidConfiguration_ThrowsInvalidArgument )
    {
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

        // Test with valid config but wrong device type
        TransformerConfig valid_config( 128, 8 );

        // CPU component constructed with CUDA device should throw
        EXPECT_THROW(
            (Transformer<DeviceType::Cpu, TensorDataType::FP32>( "invalid_device", valid_config, Device::Cuda( 0 ) )),
            std::invalid_argument
        );
    }

    TEST_F( TransformerCpuTests, Constructor_WithInvalidDeviceType_ThrowsInvalidArgument )
    {
        TransformerConfig config( 128, 8 );

        EXPECT_THROW(
            (Transformer<DeviceType::Cpu, TensorDataType::FP32>( "invalid_device", config, Device::Cuda( 0 ) )),
            std::invalid_argument
        );
    }

    // ====================================================================
    // Basic Property Tests
    // ====================================================================

    TEST_F( TransformerCpuTests, GetDeviceType_AfterConstruction_ReturnsCpu )
    {
        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = TransformerTestFixture::getCompatibleHeadCount( embedding_dim );

        auto fixture = TransformerTestFixture::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads
        );

        EXPECT_EQ( fixture.component->getDeviceType(), DeviceType::Cpu );

        auto device = fixture.component->getDeviceId();
        EXPECT_EQ( device.type, DeviceType::Cpu );
    }

    TEST_F( TransformerCpuTests, GetName_AfterConstruction_ReturnsCorrectName )
    {
        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = TransformerTestFixture::getCompatibleHeadCount( embedding_dim );

        auto fixture = TransformerTestFixture::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads
        );

        std::string expected_name = "transformer_cpu_" + test_shape.name;
        EXPECT_EQ( fixture.component->getName(), expected_name );
    }

    TEST_F( TransformerCpuTests, IsTraining_InferenceFixture_ReturnsFalse )
    {
        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = TransformerTestFixture::getCompatibleHeadCount( embedding_dim );

        auto fixture = TransformerTestFixture::CreateStandalone(
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

    TEST_F( TransformerCpuTests, IsTraining_TrainingFixture_ReturnsTrue )
    {
        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = TransformerTestFixture::getCompatibleHeadCount( embedding_dim );

        auto fixture = TransformerTestFixture::CreateStandalone(
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

    TEST_F( TransformerCpuTests, SetTraining_TogglingMode_UpdatesState )
    {
        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = TransformerTestFixture::getCompatibleHeadCount( embedding_dim );

        auto fixture = TransformerTestFixture::CreateStandalone(
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

    TEST_F( TransformerCpuTests, ParameterCount_AfterBuild_ReturnsNonZero )
    {
        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = TransformerTestFixture::getCompatibleHeadCount( embedding_dim );

        auto fixture = TransformerTestFixture::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads
        );

        fixture.component->build( fixture.shape() );

        EXPECT_GT( fixture.component->parameterCount(), 0u );
    }

    TEST_F( TransformerCpuTests, ToString_AfterConstruction_ContainsComponentInfo )
    {
        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = TransformerTestFixture::getCompatibleHeadCount( embedding_dim );

        auto fixture = TransformerTestFixture::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads
        );

        std::string output = fixture.component->toString();

        EXPECT_NE( output.find( "Transformer" ), std::string::npos );
        EXPECT_NE( output.find( fixture.component->getName() ), std::string::npos );
        EXPECT_NE( output.find( "Number of heads" ), std::string::npos );
    }

    TEST_F( TransformerCpuTests, Synchronize_AfterConstruction_Succeeds )
    {
        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = TransformerTestFixture::getCompatibleHeadCount( embedding_dim );

        auto fixture = TransformerTestFixture::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads
        );

        EXPECT_NO_THROW( fixture.component->synchronize() );
    }

    // ====================================================================
    // Build State Tests
    // ====================================================================

    TEST_F( TransformerCpuTests, IsBuilt_BeforeBuild_ReturnsFalse )
    {
        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = TransformerTestFixture::getCompatibleHeadCount( embedding_dim );

        auto fixture = TransformerTestFixture::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads
        );

        EXPECT_FALSE( fixture.component->isBuilt() );
    }

    TEST_F( TransformerCpuTests, Build_WithVariousShapes_SetsBuiltState )
    {
        auto shapes = TestShape::StandardShapes();

        for ( const auto& test_shape : shapes )
        {
            dim_t embedding_dim = test_shape.dimensions.back();
            dim_t num_heads = TransformerTestFixture::getCompatibleHeadCount( embedding_dim );

            auto fixture = TransformerTestFixture::CreateStandalone(
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

    TEST_F( TransformerCpuTests, Build_WithInvalidShape_ThrowsInvalidArgument )
    {
        dim_t embedding_dim = 128;
        dim_t num_heads = 8;

        auto fixture = TransformerTestFixture::CreateStandalone(
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

    TEST_F( TransformerCpuTests, Forward_BeforeBuild_ThrowsRuntimeError )
    {
        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = TransformerTestFixture::getCompatibleHeadCount( embedding_dim );

        auto fixture = TransformerTestFixture::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads
        );

        CpuTensor input( Device::Cpu(), fixture.shape() );
        CpuTensor output( Device::Cpu(), fixture.shape() );

        EXPECT_THROW(
            fixture.component->forward( input, output ),
            std::runtime_error
        );
    }

    // ====================================================================
    // Forward Pass Tests
    // ====================================================================

    TEST_F( TransformerCpuTests, Forward_WithVariousShapes_ProducesValidOutput )
    {
        auto shapes = TestShape::StandardShapes();

        for ( const auto& test_shape : shapes )
        {
            dim_t embedding_dim = test_shape.dimensions.back();
            dim_t num_heads = TransformerTestFixture::getCompatibleHeadCount( embedding_dim );

            auto fixture = TransformerTestFixture::CreateStandalone(
                test_shape,
                embedding_dim,
                num_heads
            );

            fixture.component->build( fixture.shape() );

            CpuTensor input( Device::Cpu(), fixture.shape() );
            CpuTensor output( Device::Cpu(), fixture.shape() );

            random( input, -0.5f, 0.5f );

            EXPECT_NO_THROW( fixture.component->forward( input, output ) )
                << "Forward failed for shape: " << test_shape.name;

            EXPECT_EQ( output.size(), input.size() )
                << "Output size mismatch for shape: " << test_shape.name;

            EXPECT_EQ( output.shape(), input.shape() )
                << "Output shape mismatch for shape: " << test_shape.name;
        }
    }

    TEST_F( TransformerCpuTests, Forward_MultipleInvocations_Succeeds )
    {
        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = TransformerTestFixture::getCompatibleHeadCount( embedding_dim );

        auto fixture = TransformerTestFixture::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads
        );

        fixture.component->build( fixture.shape() );

        CpuTensor input( Device::Cpu(), fixture.shape() );
        CpuTensor output( Device::Cpu(), fixture.shape() );

        for ( int iter = 0; iter < 3; ++iter )
        {
            random( input, -0.5f, 0.5f );

            EXPECT_NO_THROW( fixture.component->forward( input, output ) )
                << "Forward failed on iteration " << iter;
        }
    }

    TEST_F( TransformerCpuTests, Forward_TrainingMode_ProducesValidOutput )
    {
        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = TransformerTestFixture::getCompatibleHeadCount( embedding_dim );

        auto fixture = TransformerTestFixture::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads,
            0,
            true,
            ActivationType::Gelu,
            true // training mode
        );

        //fixture.component->build( fixture.shape() );

        CpuTensor input( Device::Cpu(), fixture.shape() );
        CpuTensor output( Device::Cpu(), fixture.shape() );

        random( input, -0.5f, 0.5f );

        EXPECT_NO_THROW( fixture.component->forward( input, output ) );

        EXPECT_EQ( output.size(), input.size() );
    }

    // ====================================================================
    // Configuration Variant Tests
    // ====================================================================

    TEST_F( TransformerCpuTests, NoBias_Forward_ProducesValidOutput )
    {
        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = TransformerTestFixture::getCompatibleHeadCount( embedding_dim );

        auto fixture = TransformerTestFixture::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads,
            0,     // default hidden
            false  // no bias
        );

        fixture.component->build( fixture.shape() );

        CpuTensor input( Device::Cpu(), fixture.shape() );
        CpuTensor output( Device::Cpu(), fixture.shape() );

        random( input, -0.5f, 0.5f );

        EXPECT_NO_THROW( fixture.component->forward( input, output ) );
        EXPECT_EQ( output.size(), input.size() );
    }

    TEST_F( TransformerCpuTests, CustomHiddenDimension_Forward_ProducesValidOutput )
    {
        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = TransformerTestFixture::getCompatibleHeadCount( embedding_dim );
        dim_t custom_hidden = embedding_dim * 2; // 2x instead of default 4x

        auto fixture = TransformerTestFixture::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads,
            custom_hidden
        );

        fixture.component->build( fixture.shape() );

        CpuTensor input( Device::Cpu(), fixture.shape() );
        CpuTensor output( Device::Cpu(), fixture.shape() );

        random( input, -0.5f, 0.5f );

        EXPECT_NO_THROW( fixture.component->forward( input, output ) );
        EXPECT_EQ( output.shape(), input.shape() );
    }

    TEST_F( TransformerCpuTests, VariousActivationTypes_Forward_AllSucceed )
    {
        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = TransformerTestFixture::getCompatibleHeadCount( embedding_dim );

        std::vector<ActivationType> activations = {
            ActivationType::Gelu
            //ActivationType::Relu,
            //ActivationType::Tanh
        };

        for ( auto activation : activations )
        {
            auto fixture = TransformerTestFixture::CreateStandalone(
                test_shape,
                embedding_dim,
                num_heads,
                0,
                true,
                activation
            );

            fixture.component->build( fixture.shape() );

            CpuTensor input( Device::Cpu(), fixture.shape() );
            CpuTensor output( Device::Cpu(), fixture.shape() );

            random( input, -0.5f, 0.5f );

            EXPECT_NO_THROW( fixture.component->forward( input, output ) )
                << "Forward failed for activation: " << static_cast<int>(activation);
        }
    }

    // ====================================================================
    // Shared Context Tests
    // ====================================================================

    TEST_F( TransformerCpuTests, SharedContext_Construction_Succeeds )
    {
        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = TransformerTestFixture::getCompatibleHeadCount( embedding_dim );

        auto fixture = TransformerTestFixture::CreateWithSharedContext(
            test_shape,
            embedding_dim,
            num_heads
        );

        ASSERT_NE( fixture.component, nullptr );
        ASSERT_NE( fixture.network, nullptr );

        std::string expected_name = "transformer_network_" + test_shape.name + ".transformer";
        EXPECT_EQ( fixture.component->getName(), expected_name );
    }

    TEST_F( TransformerCpuTests, SharedContext_Forward_ProducesValidOutput )
    {
        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = TransformerTestFixture::getCompatibleHeadCount( embedding_dim );

        auto fixture = TransformerTestFixture::CreateWithSharedContext(
            test_shape,
            embedding_dim,
            num_heads
        );

        fixture.component->build( fixture.shape() );

        CpuTensor input( Device::Cpu(), fixture.shape() );
        CpuTensor output( Device::Cpu(), fixture.shape() );

        random( input, -0.5f, 0.5f );

        EXPECT_NO_THROW( fixture.component->forward( input, output ) );
        EXPECT_EQ( output.size(), input.size() );
        EXPECT_EQ( output.shape(), input.shape() );
    }

    // ====================================================================
    // Child Component Tests
    // ====================================================================

    TEST_F( TransformerCpuTests, GetNamedComponents_ReturnsExpectedChildren )
    {
        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = TransformerTestFixture::getCompatibleHeadCount( embedding_dim );

        auto fixture = TransformerTestFixture::CreateStandalone(
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
        EXPECT_NE( modules.find( base + ".lnorm_1" ), modules.end() );
        EXPECT_NE( modules.find( base + ".lnorm_2" ), modules.end() );
        EXPECT_NE( modules.find( base + ".fc_qkv_proj" ), modules.end() );
        EXPECT_NE( modules.find( base + ".res_1" ), modules.end() );
        EXPECT_NE( modules.find( base + ".res_2" ), modules.end() );
        EXPECT_NE( modules.find( base + ".mlp" ), modules.end() );
    }

    // ====================================================================
    // Edge Case Tests
    // ====================================================================

    TEST_F( TransformerCpuTests, EdgeCase_MinimalShape_Forward )
    {
        TestShape test_shape = TestShape::Minimal();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = TransformerTestFixture::getCompatibleHeadCount( embedding_dim );

        auto fixture = TransformerTestFixture::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads
        );

        fixture.component->build( fixture.shape() );

        CpuTensor input( Device::Cpu(), fixture.shape() );
        CpuTensor output( Device::Cpu(), fixture.shape() );

        random( input, -0.5f, 0.5f );

        EXPECT_NO_THROW( fixture.component->forward( input, output ) );
    }

    TEST_F( TransformerCpuTests, EdgeCase_SingleHead_Forward )
    {
        TestShape test_shape = { TestShapeSize::Small, { 1, 4, 64 }, "SingleHead" };
        dim_t embedding_dim = 64;
        dim_t num_heads = 1;

        auto fixture = TransformerTestFixture::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads
        );

        fixture.component->build( fixture.shape() );

        CpuTensor input( Device::Cpu(), fixture.shape() );
        CpuTensor output( Device::Cpu(), fixture.shape() );

        random( input, -0.5f, 0.5f );

        EXPECT_NO_THROW( fixture.component->forward( input, output ) );
    }

    TEST_F( TransformerCpuTests, EdgeCase_LargeHeadCount_Forward )
    {
        TestShape test_shape = { TestShapeSize::Medium, { 1, 8, 256 }, "ManyHeads" };
        dim_t embedding_dim = 256;
        dim_t num_heads = 16; // Many heads

        auto fixture = TransformerTestFixture::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads
        );

        fixture.component->build( fixture.shape() );

        CpuTensor input( Device::Cpu(), fixture.shape() );
        CpuTensor output( Device::Cpu(), fixture.shape() );

        random( input, -0.5f, 0.5f );

        EXPECT_NO_THROW( fixture.component->forward( input, output ) );
    }

    // ====================================================================
    // Performance and Consistency Tests
    // ====================================================================

    TEST_F( TransformerCpuTests, Forward_DeterministicOutput_WithSameInput )
    {
        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = TransformerTestFixture::getCompatibleHeadCount( embedding_dim );

        auto fixture = TransformerTestFixture::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads
        );

        fixture.component->build( fixture.shape() );

        CpuTensor input( Device::Cpu(), fixture.shape() );
        CpuTensor output1( Device::Cpu(), fixture.shape() );
        CpuTensor output2( Device::Cpu(), fixture.shape() );

        // Fill with specific values for deterministic test
        for ( size_t i = 0; i < input.size(); ++i )
        {
            input.data()[ i ] = 0.1f * (i % 10) - 0.5f;
        }

        // Run forward twice with same input
        fixture.component->forward( input, output1 );
        fixture.component->forward( input, output2 );

        // Outputs should be identical
        const float epsilon = 1e-6f;
        bool outputs_match = true;

        for ( size_t i = 0; i < output1.size(); ++i )
        {
            float diff = std::abs( output1.data()[ i ] - output2.data()[ i ] );
            if ( diff > epsilon )
            {
                outputs_match = false;
                break;
            }
        }

        EXPECT_TRUE( outputs_match )
            << "Transformer should produce deterministic output for same input";
    }

    // ====================================================================
    // Numerical Stability Tests
    // ====================================================================

    TEST_F( TransformerCpuTests, Forward_WithLargeInputValues_HandlesGracefully )
    {
        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = TransformerTestFixture::getCompatibleHeadCount( embedding_dim );

        auto fixture = TransformerTestFixture::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads
        );

        fixture.component->build( fixture.shape() );

        CpuTensor input( Device::Cpu(), fixture.shape() );
        CpuTensor output( Device::Cpu(), fixture.shape() );

        // Test with larger input values
        random( input, -5.0f, 5.0f );

        EXPECT_NO_THROW( fixture.component->forward( input, output ) );

        // Check output is not NaN or infinite
        bool output_valid = true;
        for ( size_t i = 0; i < output.size(); ++i )
        {
            float val = output.data()[ i ];
            if ( std::isnan( val ) || std::isinf( val ) )
            {
                output_valid = false;
                break;
            }
        }

        EXPECT_TRUE( output_valid )
            << "Transformer output should not contain NaN or infinite values";
    }

    TEST_F( TransformerCpuTests, Forward_WithZeroInput_ProducesValidOutput )
    {
        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = TransformerTestFixture::getCompatibleHeadCount( embedding_dim );

        auto fixture = TransformerTestFixture::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads
        );

        fixture.component->build( fixture.shape() );

        CpuTensor input( Device::Cpu(), fixture.shape() );
        CpuTensor output( Device::Cpu(), fixture.shape() );

        // Zero input
        zeros( input );

        EXPECT_NO_THROW( fixture.component->forward( input, output ) );
        EXPECT_EQ( output.size(), input.size() );
    }

    TEST_F( TransformerCpuTests, Forward_WithSmallInputValues_HandlesGracefully )
    {
        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = TransformerTestFixture::getCompatibleHeadCount( embedding_dim );

        auto fixture = TransformerTestFixture::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads
        );

        fixture.component->build( fixture.shape() );

        CpuTensor input( Device::Cpu(), fixture.shape() );
        CpuTensor output( Device::Cpu(), fixture.shape() );

        // Test with very small input values
        random( input, -1e-6f, 1e-6f );

        EXPECT_NO_THROW( fixture.component->forward( input, output ) );
        EXPECT_EQ( output.size(), input.size() );
    }

    // ====================================================================
    // Component Lifecycle Tests
    // ====================================================================

    TEST_F( TransformerCpuTests, BuildMultipleTimes_ThrowsException )
    {
        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = TransformerTestFixture::getCompatibleHeadCount( embedding_dim );

        auto fixture = TransformerTestFixture::CreateStandalone(
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

    TEST_F( TransformerCpuTests, ParameterCount_BeforeBuild_ThrowsRuntimeError )
    {
        TestShape test_shape = TestShape::Small();
        dim_t embedding_dim = test_shape.dimensions.back();
        dim_t num_heads = TransformerTestFixture::getCompatibleHeadCount( embedding_dim );

        auto fixture = TransformerTestFixture::CreateStandalone(
            test_shape,
            embedding_dim,
            num_heads
        );

        EXPECT_THROW( fixture.component->parameterCount(), std::runtime_error );
    }
}