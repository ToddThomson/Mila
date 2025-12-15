#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <vector>
#include <stdexcept>
#include <filesystem>
#include <chrono>
#include <format>
#include <system_error>
#include <optional>

import Mila;

namespace Dnn::Networks::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    /**
     * @brief Minimal test component for testing Network.
     *
     * Updated to support both standalone and shared context modes following
     * the new Component architecture pattern.
     */
    class TestComponent : public Component<DeviceType::Cpu, TensorDataType::FP32>
    {
    public:
        using ComponentBase = Component<DeviceType::Cpu, TensorDataType::FP32>;

        /**
         * @brief Constructor for shared mode (context provided by parent).
         *
         * @param param_count Mock parameter count for testing aggregation
         * @param device_id Optional device for standalone mode
         */
        explicit TestComponent( size_t param_count = 0, std::optional<DeviceId> device_id = std::nullopt )
            : ComponentBase( "test_component" ), param_count_( param_count )
        {
            if ( device_id.has_value() )
            {
                if ( device_id->type != DeviceType::Cpu )
                {
                    throw std::invalid_argument( "TestComponent: device type mismatch" );
                }

                owned_exec_context_ = createExecutionContext( device_id.value() );
                setExecutionContext( owned_exec_context_.get() );
            }
        }

        void synchronize() override
        {}

        size_t parameterCount() const override
        {
            return param_count_;
        }

        std::vector<ITensor*> getParameters() const override
        {
            return {};
        }

        std::vector<ITensor*> getGradients() const override
        {
            return {};
        }

        void save_( ModelArchive& /*archive*/, SerializationMode /*mode*/ ) const override
        {}

        std::string toString() const override
        {
            return std::string( "TestComponent:" ) + this->getName();
        }

        DeviceId getDeviceId() const override
        {
            return this->getExecutionContext()->getDeviceId();
        }

    protected:

        void onBuilding( const shape_t& /*input_shape*/ ) override
        {}

    private:

        size_t param_count_;
        std::unique_ptr<IExecutionContext> owned_exec_context_{ nullptr };
    };

    /**
     * @brief Concrete Network subclass for testing.
     *
     * Implements createGraph() to define network architecture and onBuilding()
     * to propagate build to all children. Uses the component-owns-name pattern.
     */
    template<DeviceType TDeviceType, TensorDataType TPrecision = TensorDataType::FP32>
    class TestableNetwork : public Network<TDeviceType, TPrecision>
    {
    public:
        using Base = Network<TDeviceType, TPrecision>;

        explicit TestableNetwork( DeviceId device_id, const std::string& name )
            : Base( device_id, name )
        {}

        explicit TestableNetwork( IExecutionContext* exec_context, const std::string& name )
            : Base( exec_context, name )
        {}

        /**
         * @brief Helper to add a test component with explicit name and parameter count.
         *
         * Creates a TestComponent in shared mode, sets its name, and registers it.
         * This mimics the pattern used in real networks like MnistClassifier.
         */
        void addTestComponent( const std::string& name, size_t param_count = 0 )
        {
            auto component = std::make_shared<TestComponent>( param_count, std::nullopt );
            component->setName( name );
            this->addComponent( component );
        }

    protected:

        void onBuilding( const shape_t& input_shape ) override
        {
            for ( const auto& [name, component] : this->getNamedComponents() )
            {
                if ( !component->isBuilt() )
                {
                    component->build( input_shape );
                }
            }
        }
    };

    // ====================================================================
    // Test Fixture
    // ====================================================================

    class NetworkTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {}

        static std::filesystem::path makeTempArchivePath()
        {
            auto tmp = std::filesystem::temp_directory_path();
            auto ts = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            return tmp / std::format( "mila_test_network_{}.mila", ts );
        }
    };

    // ====================================================================
    // Construction Tests
    // ====================================================================

    TEST_F( NetworkTests, ConstructWithDeviceId_CreatesOwnedContext )
    {
        TestableNetwork<DeviceType::Cpu> net( Device::Cpu(), "test_network" );

        EXPECT_EQ( net.getName(), "test_network" );
        EXPECT_EQ( net.getDeviceId().type, DeviceType::Cpu );
    }

    TEST_F( NetworkTests, ConstructWithEmptyName_Throws )
    {
        EXPECT_THROW(
            TestableNetwork<DeviceType::Cpu>( Device::Cpu(), "" ),
            std::invalid_argument
        );
    }

    TEST_F( NetworkTests, ConstructWithSharedContext_SharesContext )
    {
        auto exec_context = createExecutionContext( Device::Cpu() );
        TestableNetwork<DeviceType::Cpu> net( exec_context.get(), "shared_ctx_net" );

        EXPECT_EQ( net.getName(), "shared_ctx_net" );
        EXPECT_EQ( net.getDeviceId().type, DeviceType::Cpu );
    }

    TEST_F( NetworkTests, ConstructWithNullContext_Throws )
    {
        IExecutionContext* null_ctx = nullptr;

        EXPECT_THROW(
            TestableNetwork<DeviceType::Cpu>( null_ctx, "null_ctx_net" ),
            std::invalid_argument
        );
    }

    TEST_F( NetworkTests, ConstructWithDeviceTypeMismatch_Throws )
    {
        EXPECT_THROW(
            TestableNetwork<DeviceType::Cpu>( Device::Cuda( 0 ), "mismatch_net" ),
            std::invalid_argument
        );
    }

    TEST_F( NetworkTests, ConstructWithSharedContext_DeviceTypeMismatch_Throws )
    {
        auto cuda_context = createExecutionContext( Device::Cuda( 0 ) );

        EXPECT_THROW(
            TestableNetwork<DeviceType::Cpu>( cuda_context.get(), "mismatch_net" ),
            std::invalid_argument
        );
    }

    // ====================================================================
    // Component Registration Tests (New API)
    // ====================================================================

    TEST_F( NetworkTests, AddComponent_ComponentOwnsName )
    {
        TestableNetwork<DeviceType::Cpu> net( Device::Cpu(), "name_test" );

        auto comp = std::make_shared<TestComponent>( 5, std::nullopt );
        comp->setName( "named_comp" );

        net.addComponent( comp );

        EXPECT_TRUE( net.hasComponent( "named_comp" ) );
        EXPECT_EQ( net.getComponent( "named_comp" )->getName(), "named_comp" );
    }

    TEST_F( NetworkTests, AddComponent_Chainable )
    {
        TestableNetwork<DeviceType::Cpu> net( Device::Cpu(), "chain_test" );

        net.addTestComponent( "comp1", 5 );
        net.addTestComponent( "comp2", 10 );

        EXPECT_TRUE( net.hasComponent( "comp1" ) );
        EXPECT_TRUE( net.hasComponent( "comp2" ) );
        EXPECT_EQ( net.childCount(), 2u );
    }

    TEST_F( NetworkTests, AddComponent_MultipleChained )
    {
        TestableNetwork<DeviceType::Cpu> net( Device::Cpu(), "multi_chain" );

        net.addTestComponent( "comp1", 1 );
        net.addTestComponent( "comp2", 2 );
        net.addTestComponent( "comp3", 3 );
        net.addTestComponent( "comp4", 4 );

        EXPECT_EQ( net.childCount(), 4u );
        EXPECT_TRUE( net.hasComponent( "comp1" ) );
        EXPECT_TRUE( net.hasComponent( "comp4" ) );
    }

    TEST_F( NetworkTests, AddComponent_NullComponent_Throws )
    {
        TestableNetwork<DeviceType::Cpu> net( Device::Cpu(), "null_test" );

        EXPECT_THROW(
            net.addComponent( nullptr ),
            std::invalid_argument
        );
    }

    TEST_F( NetworkTests, AddComponent_DuplicateName_Throws )
    {
        TestableNetwork<DeviceType::Cpu> net( Device::Cpu(), "dup_test" );

        net.addTestComponent( "duplicate", 1 );

        auto duplicate = std::make_shared<TestComponent>( 2, std::nullopt );
        duplicate->setName( "duplicate" );

        EXPECT_THROW(
            net.addComponent( duplicate ),
            std::invalid_argument
        );
    }

    TEST_F( NetworkTests, AddComponent_AfterBuild_Throws )
    {
        TestableNetwork<DeviceType::Cpu> net( Device::Cpu(), "post_build_test" );

        net.addTestComponent( "pre_build", 0 );

        net.build( { 1 } );

        EXPECT_THROW(
            net.addTestComponent( "post_build", 0 ),
            std::runtime_error
        );
    }

    TEST_F( NetworkTests, AddComponent_SharesExecutionContext )
    {
        TestableNetwork<DeviceType::Cpu> net( Device::Cpu(), "ctx_share_test" );

        net.addTestComponent( "ctx1", 0 );
        net.addTestComponent( "ctx2", 0 );

        auto comp1 = net.getComponent( "ctx1" );
        auto comp2 = net.getComponent( "ctx2" );

        EXPECT_EQ( comp1->getDeviceId().type, net.getDeviceId().type );
        EXPECT_EQ( comp2->getDeviceId().type, net.getDeviceId().type );
        EXPECT_EQ( comp1->getDeviceId().type, comp2->getDeviceId().type );
    }

    TEST_F( NetworkTests, AddComponent_ContextPropagatedImmediately )
    {
        TestableNetwork<DeviceType::Cpu> net( Device::Cpu(), "ctx_prop_test" );

        auto comp = std::make_shared<TestComponent>( 0, std::nullopt );
        comp->setName( "late_add" );

        //EXPECT_FALSE( comp->hasExecutionContext() );

        net.addComponent( comp );

        //EXPECT_TRUE( comp->hasExecutionContext() );
        EXPECT_EQ( comp->getDeviceId().type, DeviceType::Cpu );
    }

    // ====================================================================
    // Component Management Tests
    // ====================================================================

    TEST_F( NetworkTests, GetComponent_ExistingName_ReturnsComponent )
    {
        TestableNetwork<DeviceType::Cpu> net( Device::Cpu(), "get_test" );

        net.addTestComponent( "findme", 3 );

        auto retrieved = net.getComponent( "findme" );

        EXPECT_NE( retrieved, nullptr );
        EXPECT_EQ( retrieved->parameterCount(), 3u );
    }

    TEST_F( NetworkTests, GetComponent_NonExistentName_Throws )
    {
        TestableNetwork<DeviceType::Cpu> net( Device::Cpu(), "get_fail_test" );

        EXPECT_THROW(
            net.getComponent( "nonexistent" ),
            std::out_of_range
        );
    }

    TEST_F( NetworkTests, HasComponent_ReturnsCorrectStatus )
    {
        TestableNetwork<DeviceType::Cpu> net( Device::Cpu(), "has_test" );

        net.addTestComponent( "exists", 0 );

        EXPECT_TRUE( net.hasComponent( "exists" ) );
        EXPECT_FALSE( net.hasComponent( "does_not_exist" ) );
    }

    TEST_F( NetworkTests, RemoveComponent_ExistingComponent_ReturnsTrue )
    {
        TestableNetwork<DeviceType::Cpu> net( Device::Cpu(), "remove_test" );

        net.addTestComponent( "removeme", 0 );

        EXPECT_TRUE( net.hasComponent( "removeme" ) );
        EXPECT_TRUE( net.removeComponent( "removeme" ) );
        EXPECT_FALSE( net.hasComponent( "removeme" ) );
        EXPECT_EQ( net.childCount(), 0u );
    }

    TEST_F( NetworkTests, RemoveComponent_NonExistent_ReturnsFalse )
    {
        TestableNetwork<DeviceType::Cpu> net( Device::Cpu(), "remove_fail_test" );

        EXPECT_FALSE( net.removeComponent( "never_existed" ) );
    }

    TEST_F( NetworkTests, RemoveComponent_AfterBuild_Throws )
    {
        TestableNetwork<DeviceType::Cpu> net( Device::Cpu(), "remove_post_build" );

        net.addTestComponent( "locked", 0 );

        net.build( { 1 } );

        EXPECT_THROW(
            net.removeComponent( "locked" ),
            std::runtime_error
        );
    }

    TEST_F( NetworkTests, ClearComponents_RemovesAllChildren )
    {
        TestableNetwork<DeviceType::Cpu> net( Device::Cpu(), "clear_test" );

        net.addTestComponent( "comp1", 0 );
        net.addTestComponent( "comp2", 0 );
        net.addTestComponent( "comp3", 0 );

        EXPECT_EQ( net.childCount(), 3u );

        net.clearComponents();

        EXPECT_EQ( net.childCount(), 0u );
        EXPECT_FALSE( net.hasComponent( "comp1" ) );
        EXPECT_FALSE( net.hasComponent( "comp2" ) );
        EXPECT_FALSE( net.hasComponent( "comp3" ) );
    }

    TEST_F( NetworkTests, ClearComponents_AfterBuild_Throws )
    {
        TestableNetwork<DeviceType::Cpu> net( Device::Cpu(), "clear_post_build" );

        net.addTestComponent( "locked", 0 );

        net.build( { 1 } );

        EXPECT_THROW(
            net.clearComponents(),
            std::runtime_error
        );
    }

    // ====================================================================
    // Build and Parameter Tests
    // ====================================================================

    TEST_F( NetworkTests, Build_PropagatestoChildren )
    {
        TestableNetwork<DeviceType::Cpu> net( Device::Cpu(), "build_test" );

        net.addTestComponent( "build1", 5 );
        net.addTestComponent( "build2", 7 );

        auto comp1 = net.getComponent( "build1" );
        auto comp2 = net.getComponent( "build2" );

        EXPECT_FALSE( net.isBuilt() );
        EXPECT_FALSE( comp1->isBuilt() );
        EXPECT_FALSE( comp2->isBuilt() );

        net.build( { 2, 3 } );

        EXPECT_TRUE( net.isBuilt() );
        EXPECT_TRUE( comp1->isBuilt() );
        EXPECT_TRUE( comp2->isBuilt() );
    }

    TEST_F( NetworkTests, ParameterCount_AggregatesChildren )
    {
        TestableNetwork<DeviceType::Cpu> net( Device::Cpu(), "param_test" );

        net.addTestComponent( "p1", 10 );
        net.addTestComponent( "p2", 20 );
        net.addTestComponent( "p3", 15 );

        net.build( { 1 } );

        EXPECT_EQ( net.parameterCount(), 45u );
    }

    TEST_F( NetworkTests, ParameterCount_BeforeBuild_Throws )
    {
        TestableNetwork<DeviceType::Cpu> net( Device::Cpu(), "param_pre_build" );

        net.addTestComponent( "unbuild", 5 );

        EXPECT_THROW(
            net.parameterCount(),
            std::runtime_error
        );
    }

    TEST_F( NetworkTests, GetParameters_BeforeBuild_Throws )
    {
        TestableNetwork<DeviceType::Cpu> net( Device::Cpu(), "get_params_pre_build" );

        net.addTestComponent( "unbuild", 0 );

        EXPECT_THROW(
            net.getParameters(),
            std::runtime_error
        );
    }

    TEST_F( NetworkTests, GetGradients_NotTraining_Throws )
    {
        TestableNetwork<DeviceType::Cpu> net( Device::Cpu(), "grads_no_train" );

        net.addTestComponent( "eval", 0 );

        net.build( { 1 } );

        EXPECT_FALSE( net.isTraining() );
        EXPECT_THROW(
            net.getGradients(),
            std::runtime_error
        );
    }

    TEST_F( NetworkTests, GetGradients_TrainingMode_Succeeds )
    {
        TestableNetwork<DeviceType::Cpu> net( Device::Cpu(), "grads_train" );

        net.addTestComponent( "train", 0 );

        net.build( { 1 } );

        net.setTraining( true );

        EXPECT_NO_THROW( net.getGradients() );
    }

    // ====================================================================
    // Training Mode Tests
    // ====================================================================

    TEST_F( NetworkTests, SetTraining_PropagatestoChildren )
    {
        TestableNetwork<DeviceType::Cpu> net( Device::Cpu(), "training_prop" );

        net.addTestComponent( "t1", 0 );
        net.addTestComponent( "t2", 0 );

        auto comp1 = net.getComponent( "t1" );
        auto comp2 = net.getComponent( "t2" );

        EXPECT_FALSE( net.isTraining() );
        EXPECT_FALSE( comp1->isTraining() );
        EXPECT_FALSE( comp2->isTraining() );

        net.setTraining( true );

        EXPECT_TRUE( net.isTraining() );
        EXPECT_TRUE( comp1->isTraining() );
        EXPECT_TRUE( comp2->isTraining() );

        net.setTraining( false );

        EXPECT_FALSE( net.isTraining() );
        EXPECT_FALSE( comp1->isTraining() );
        EXPECT_FALSE( comp2->isTraining() );
    }

    // ====================================================================
    // Synchronization Tests
    // ====================================================================

    TEST_F( NetworkTests, Synchronize_DoesNotThrow )
    {
        TestableNetwork<DeviceType::Cpu> net( Device::Cpu(), "sync_test" );

        net.addTestComponent( "s1", 0 );
        net.addTestComponent( "s2", 0 );

        EXPECT_NO_THROW( net.synchronize() );
    }

    // ====================================================================
    // ToString Tests
    // ====================================================================

    TEST_F( NetworkTests, ToString_ContainsNameAndChildren )
    {
        TestableNetwork<DeviceType::Cpu> net( Device::Cpu(), "info_test" );

        net.addTestComponent( "child_a", 0 );
        net.addTestComponent( "child_b", 0 );

        std::string info = net.toString();

        EXPECT_NE( info.find( "info_test" ), std::string::npos );
        EXPECT_NE( info.find( "child_a" ), std::string::npos );
        EXPECT_NE( info.find( "child_b" ), std::string::npos );
    }

    // ====================================================================
    // Serialization Tests
    // ====================================================================

    TEST_F( NetworkTests, Save_WritesMetadataAndArchitecture )
    {
        auto path = makeTempArchivePath();
        std::error_code ec;
        std::filesystem::remove( path, ec );

        TestableNetwork<DeviceType::Cpu> net( Device::Cpu(), "save_test" );

        net.addTestComponent( "mod1", 10 );
        net.addTestComponent( "mod2", 20 );

        net.build( { 1 } );

        {
            auto writer = std::make_unique<ZipSerializer>();
            ModelArchive archive( path.string(), std::move( writer ), OpenMode::Write );

            EXPECT_NO_THROW( net.save( archive, SerializationMode::Architecture ) );

            archive.close();
        }

        {
            auto reader = std::make_unique<ZipSerializer>();
            ModelArchive archive( path.string(), std::move( reader ), OpenMode::Read );

            EXPECT_TRUE( archive.hasFile( "network/meta.json" ) );
            auto meta = archive.readJson( "network/meta.json" );

            EXPECT_EQ( meta.at( "name" ).get<std::string>(), "save_test" );
            EXPECT_EQ( meta.at( "num_components" ).get<size_t>(), 2u );
            EXPECT_EQ( meta.at( "format_version" ).get<int>(), 1 );
            EXPECT_TRUE( meta.contains( "export_time" ) );
            EXPECT_EQ( meta.at( "mode" ).get<std::string>(),
                serializationModeToString( SerializationMode::Architecture ) );

            EXPECT_TRUE( archive.hasFile( "network/architecture.json" ) );
            auto arch = archive.readJson( "network/architecture.json" );

            ASSERT_TRUE( arch.is_array() );
            EXPECT_EQ( arch.size(), 2u );

            EXPECT_EQ( arch[ 0 ].at( "name" ).get<std::string>(), "mod1" );
            EXPECT_EQ( arch[ 0 ].at( "path" ).get<std::string>(), "components/mod1" );
            EXPECT_EQ( arch[ 0 ].at( "index" ).get<int>(), 0 );

            EXPECT_EQ( arch[ 1 ].at( "name" ).get<std::string>(), "mod2" );
            EXPECT_EQ( arch[ 1 ].at( "path" ).get<std::string>(), "components/mod2" );
            EXPECT_EQ( arch[ 1 ].at( "index" ).get<int>(), 1 );

            archive.close();
        }

        std::filesystem::remove( path, ec );
    }

    TEST_F( NetworkTests, Save_BeforeBuild_Succeeds )
    {
        auto path = makeTempArchivePath();
        std::error_code ec;
        std::filesystem::remove( path, ec );

        TestableNetwork<DeviceType::Cpu> net( Device::Cpu(), "unbuilt_save" );

        net.addTestComponent( "unbuild", 0 );

        auto writer = std::make_unique<ZipSerializer>();
        ModelArchive archive( path.string(), std::move( writer ), OpenMode::Write );

        EXPECT_NO_THROW(
            net.save( archive, SerializationMode::Architecture )
        );

        archive.close();
        std::filesystem::remove( path, ec );
    }

    TEST_F( NetworkTests, Load_ThrowsUntilFactoryImplemented )
    {
        auto path = makeTempArchivePath();
        std::error_code ec;
        std::filesystem::remove( path, ec );

        TestableNetwork<DeviceType::Cpu> net( Device::Cpu(), "load_test" );

        net.addTestComponent( "loadme", 5 );

        net.build( { 1 } );

        {
            auto writer = std::make_unique<ZipSerializer>();
            ModelArchive archive( path.string(), std::move( writer ), OpenMode::Write );
            net.save( archive, SerializationMode::Architecture );
            archive.close();
        }

        {
            auto reader = std::make_unique<ZipSerializer>();
            ModelArchive archive( path.string(), std::move( reader ), OpenMode::Read );

            auto exec_ctx = createExecutionContext( Device::Cpu() );

            EXPECT_THROW(
                TestableNetwork<DeviceType::Cpu>::load( archive, exec_ctx.get() ),
                std::runtime_error
            );

            archive.close();
        }

        std::filesystem::remove( path, ec );
    }

    // ====================================================================
    // Empty Network Tests
    // ====================================================================

    TEST_F( NetworkTests, EmptyNetwork_HasNoChildren )
    {
        TestableNetwork<DeviceType::Cpu> net( Device::Cpu(), "empty" );

        EXPECT_EQ( net.childCount(), 0u );
        EXPECT_FALSE( net.hasChildren() );
        EXPECT_TRUE( net.getComponents().empty() );
        EXPECT_TRUE( net.getNamedComponents().empty() );
    }

    // ====================================================================
    // Edge Case Tests
    // ====================================================================

    TEST_F( NetworkTests, GetComponents_ReturnsInsertionOrder )
    {
        TestableNetwork<DeviceType::Cpu> net( Device::Cpu(), "order_test" );

        net.addTestComponent( "first", 1 );
        net.addTestComponent( "second", 2 );
        net.addTestComponent( "third", 3 );

        const auto& components = net.getComponents();

        ASSERT_EQ( components.size(), 3u );
        EXPECT_EQ( components[ 0 ]->parameterCount(), 1u );
        EXPECT_EQ( components[ 1 ]->parameterCount(), 2u );
        EXPECT_EQ( components[ 2 ]->parameterCount(), 3u );
    }

    TEST_F( NetworkTests, GetNamedComponents_ContainsAllChildren )
    {
        TestableNetwork<DeviceType::Cpu> net( Device::Cpu(), "named_test" );

        net.addTestComponent( "child1", 0 );
        net.addTestComponent( "child2", 0 );
        net.addTestComponent( "child3", 0 );

        const auto& named = net.getNamedComponents();

        EXPECT_EQ( named.size(), 3u );
        EXPECT_TRUE( named.find( "child1" ) != named.end() );
        EXPECT_TRUE( named.find( "child2" ) != named.end() );
        EXPECT_TRUE( named.find( "child3" ) != named.end() );
    }
}