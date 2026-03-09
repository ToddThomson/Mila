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

namespace Dnn::Core::Networks::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    /**
     * @brief Minimal test component for testing Network.
     *
     * Component owns its name at construction time and supports both
     * standalone mode (owns ExecutionContext) and shared mode (borrows context).
     */
    class TestComponent : public Component<DeviceType::Cpu, TensorDataType::FP32>
    {
    public:
        using ComponentBase = Component<DeviceType::Cpu, TensorDataType::FP32>;

        /**
         * @brief Constructor for test component.
         *
         * @param name Component name (required by Component base class)
         * @param param_count Mock parameter count for testing aggregation
         * @param device_id Optional device for standalone mode (testing only)
         */
        explicit TestComponent( const std::string& name,
            size_t param_count = 0,
            std::optional<DeviceId> device_id = std::nullopt )
            : ComponentBase( name ), param_count_( param_count )
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

        void save_( ModelArchive& archive, SerializationMode /*mode*/ ) const override
        {
            SerializationMetadata meta;
            meta.set( "component_type", "TestComponent" )
                .set( "param_count", static_cast<int64_t>(param_count_) );

            archive.writeMetadata( "meta.json", meta );
        }

        MemoryStats getMemoryStats() const override
        {
            return {};
        }

        std::string toString() const override
        {
            return std::string( "TestComponent:" ) + this->getName();
        }

        const ComponentType getType() const override
        {
            return ComponentType::MockComponent;
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
     * Follows the new Network architecture:
     * - Owns ExecutionContext (created in constructor)
     * - Builds component graph via createGraph()
     * - Propagates context to children via setExecutionContext()
     * - Implements save_() hook for type-specific metadata
     */
    template<DeviceType TDeviceType, TensorDataType TPrecision = TensorDataType::FP32>
    class TestableNetwork : public Network<TDeviceType, TPrecision>
    {
    public:
        using Base = Network<TDeviceType, TPrecision>;

        /**
         * @brief Construct testable network with owned ExecutionContext.
         *
         * Follows concrete network pattern:
         * 1. Create ExecutionContext
         * 2. Build component graph (context-independent)
         * 3. Propagate context to self and children
         */
        explicit TestableNetwork( const std::string& name, DeviceId device_id )
            : Base( name ),
            owned_context_( createExecutionContext( device_id ) )
        {
            if ( device_id.type != TDeviceType )
            {
                throw std::invalid_argument(
                    std::format( "TestableNetwork: device type mismatch: expected {}, got {}",
                        deviceTypeToString( TDeviceType ),
                        deviceTypeToString( device_id.type ) ) );
            }
        }

        /**
         * @brief Finalize network construction and propagate context.
         *
         * Call this after adding all components to propagate ExecutionContext
         * to all children. This mimics the pattern used by concrete networks
         * like MnistClassifier.
         */
        void finalizeConstruction()
        {
            if ( !owned_context_ )
            {
                throw std::runtime_error( "TestableNetwork: owned_context_ is null" );
            }

            this->setExecutionContext( owned_context_.get() );
        }

        /**
         * @brief Helper to add a test component with explicit name and parameter count.
         *
         * Creates a TestComponent in shared mode (no ExecutionContext) and registers it.
         * Context will be propagated when finalizeConstruction() is called.
         */
        void addTestComponent( const std::string& name, size_t param_count = 0 )
        {
            auto component = std::make_shared<TestComponent>( name, param_count, std::nullopt );
            this->addComponent( component );
        }

        /**
         * @brief Static factory method for deserialization (testing only).
         *
         * Not fully implemented - used to test that concrete classes can provide
         * their own Load() methods.
         */
        static std::unique_ptr<TestableNetwork> Load( ModelArchive& /*archive*/,
            DeviceId /*device_id*/ )
        {
            throw std::runtime_error(
                "TestableNetwork::Load: deserialization not implemented for test class" );
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
        /**
         * @brief Hook for saving type-specific metadata (required by Network base).
         *
         * Implements the serialization contract by writing type identifier using
         * SerializationMetadata abstraction.
         */
        void save_( ModelArchive& archive, SerializationMode /*mode*/ ) const override
        {
            SerializationMetadata meta;
            meta.set( "type", "TestableNetwork" )
                .set( "test_metadata", "This is a test network" );

            archive.writeMetadata( "network/testable_meta.json", meta );
        }

        /**
         * @brief Build hook - propagates build to all children.
         */
        void onBuilding( const shape_t& input_shape ) override
        {
            for ( const auto& component : this->getComponents() )
            {
                if ( !component->isBuilt() )
                {
                    component->build( input_shape );
                }
            }
        }

    private:
        std::unique_ptr<IExecutionContext> owned_context_{ nullptr };
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
        TestableNetwork<DeviceType::Cpu> net( "test_network", Device::Cpu() );
        net.finalizeConstruction();

        EXPECT_EQ( net.getName(), "test_network" );
        EXPECT_EQ( net.getDeviceId().type, DeviceType::Cpu );
    }

    TEST_F( NetworkTests, ConstructWithEmptyName_Throws )
    {
        EXPECT_THROW(
            TestableNetwork<DeviceType::Cpu>( "", Device::Cpu() ),
            std::invalid_argument
        );
    }

    TEST_F( NetworkTests, ConstructWithDeviceTypeMismatch_Throws )
    {
        EXPECT_THROW(
            TestableNetwork<DeviceType::Cpu>( "mismatch_net", Device::Cuda( 0 ) ),
            std::invalid_argument
        );
    }

    // ====================================================================
    // Component Registration Tests
    // ====================================================================

    TEST_F( NetworkTests, AddComponent_ComponentOwnsName )
    {
        TestableNetwork<DeviceType::Cpu> net( "name_test", Device::Cpu() );

        auto comp = std::make_shared<TestComponent>( "named_comp", 5, std::nullopt );
        net.addComponent( comp );
        net.finalizeConstruction();

        EXPECT_TRUE( net.hasComponent( "named_comp" ) );
        EXPECT_EQ( net.getComponent( "named_comp" )->getName(), "named_comp" );
    }

    TEST_F( NetworkTests, AddComponent_Chainable )
    {
        TestableNetwork<DeviceType::Cpu> net( "chain_test", Device::Cpu() );

        net.addTestComponent( "comp1", 5 );
        net.addTestComponent( "comp2", 10 );
        net.finalizeConstruction();

        EXPECT_TRUE( net.hasComponent( "comp1" ) );
        EXPECT_TRUE( net.hasComponent( "comp2" ) );
        EXPECT_EQ( net.childCount(), 2u );
    }

    TEST_F( NetworkTests, AddComponent_MultipleChained )
    {
        TestableNetwork<DeviceType::Cpu> net( "multi_chain", Device::Cpu() );

        net.addTestComponent( "comp1", 1 );
        net.addTestComponent( "comp2", 2 );
        net.addTestComponent( "comp3", 3 );
        net.addTestComponent( "comp4", 4 );
        net.finalizeConstruction();

        EXPECT_EQ( net.childCount(), 4u );
        EXPECT_TRUE( net.hasComponent( "comp1" ) );
        EXPECT_TRUE( net.hasComponent( "comp4" ) );
    }

    TEST_F( NetworkTests, AddComponent_NullComponent_Throws )
    {
        TestableNetwork<DeviceType::Cpu> net( "null_test", Device::Cpu() );

        EXPECT_THROW(
            net.addComponent( nullptr ),
            std::invalid_argument
        );
    }

    TEST_F( NetworkTests, AddComponent_DuplicateName_Throws )
    {
        TestableNetwork<DeviceType::Cpu> net( "dup_test", Device::Cpu() );

        net.addTestComponent( "duplicate", 1 );

        auto duplicate = std::make_shared<TestComponent>( "duplicate", 2, std::nullopt );

        EXPECT_THROW(
            net.addComponent( duplicate ),
            std::invalid_argument
        );
    }

    TEST_F( NetworkTests, AddComponent_AfterBuild_Throws )
    {
        TestableNetwork<DeviceType::Cpu> net( "post_build_test", Device::Cpu() );

        net.addTestComponent( "pre_build", 0 );
        net.finalizeConstruction();

        net.build( { 1 } );

        EXPECT_THROW(
            net.addTestComponent( "post_build", 0 ),
            std::runtime_error
        );
    }

    TEST_F( NetworkTests, AddComponent_SharesExecutionContext )
    {
        TestableNetwork<DeviceType::Cpu> net( "ctx_share_test", Device::Cpu() );

        net.addTestComponent( "ctx1", 0 );
        net.addTestComponent( "ctx2", 0 );
        net.finalizeConstruction();

        auto comp1 = net.getComponent( "ctx1" );
        auto comp2 = net.getComponent( "ctx2" );

        EXPECT_EQ( comp1->getDeviceId().type, net.getDeviceId().type );
        EXPECT_EQ( comp2->getDeviceId().type, net.getDeviceId().type );
        EXPECT_EQ( comp1->getDeviceId().type, comp2->getDeviceId().type );
    }

    TEST_F( NetworkTests, AddComponent_ContextPropagatedOnFinalize )
    {
        TestableNetwork<DeviceType::Cpu> net( "ctx_prop_test", Device::Cpu() );

        auto comp = std::make_shared<TestComponent>( "late_add", 0, std::nullopt );
        net.addComponent( comp );

        net.finalizeConstruction();

        EXPECT_EQ( comp->getDeviceId().type, DeviceType::Cpu );
    }

    // ====================================================================
    // Component Management Tests
    // ====================================================================

    TEST_F( NetworkTests, GetComponent_ExistingName_ReturnsComponent )
    {
        TestableNetwork<DeviceType::Cpu> net( "get_test", Device::Cpu() );

        net.addTestComponent( "findme", 3 );
        net.finalizeConstruction();

        auto retrieved = net.getComponent( "findme" );

        EXPECT_NE( retrieved, nullptr );
        EXPECT_EQ( retrieved->parameterCount(), 3u );
    }

    TEST_F( NetworkTests, GetComponent_NonExistentName_Throws )
    {
        TestableNetwork<DeviceType::Cpu> net( "get_fail_test", Device::Cpu() );
        net.finalizeConstruction();

        EXPECT_THROW(
            net.getComponent( "nonexistent" ),
            std::out_of_range
        );
    }

    TEST_F( NetworkTests, HasComponent_ReturnsCorrectStatus )
    {
        TestableNetwork<DeviceType::Cpu> net( "has_test", Device::Cpu() );

        net.addTestComponent( "exists", 0 );
        net.finalizeConstruction();

        EXPECT_TRUE( net.hasComponent( "exists" ) );
        EXPECT_FALSE( net.hasComponent( "does_not_exist" ) );
    }

    TEST_F( NetworkTests, RemoveComponent_ExistingComponent_ReturnsTrue )
    {
        TestableNetwork<DeviceType::Cpu> net( "remove_test", Device::Cpu() );

        net.addTestComponent( "removeme", 0 );

        EXPECT_TRUE( net.hasComponent( "removeme" ) );
        EXPECT_TRUE( net.removeComponent( "removeme" ) );
        EXPECT_FALSE( net.hasComponent( "removeme" ) );
        EXPECT_EQ( net.childCount(), 0u );
    }

    TEST_F( NetworkTests, RemoveComponent_NonExistent_ReturnsFalse )
    {
        TestableNetwork<DeviceType::Cpu> net( "remove_fail_test", Device::Cpu() );
        net.finalizeConstruction();

        EXPECT_FALSE( net.removeComponent( "never_existed" ) );
    }

    TEST_F( NetworkTests, RemoveComponent_AfterBuild_Throws )
    {
        TestableNetwork<DeviceType::Cpu> net( "remove_post_build", Device::Cpu() );

        net.addTestComponent( "locked", 0 );
        net.finalizeConstruction();

        net.build( { 1 } );

        EXPECT_THROW(
            net.removeComponent( "locked" ),
            std::runtime_error
        );
    }

    TEST_F( NetworkTests, ClearComponents_RemovesAllChildren )
    {
        TestableNetwork<DeviceType::Cpu> net( "clear_test", Device::Cpu() );

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
        TestableNetwork<DeviceType::Cpu> net( "clear_post_build", Device::Cpu() );

        net.addTestComponent( "locked", 0 );
        net.finalizeConstruction();

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
        TestableNetwork<DeviceType::Cpu> net( "build_test", Device::Cpu() );

        net.addTestComponent( "build1", 5 );
        net.addTestComponent( "build2", 7 );
        net.finalizeConstruction();

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
        TestableNetwork<DeviceType::Cpu> net( "param_test", Device::Cpu() );

        net.addTestComponent( "p1", 10 );
        net.addTestComponent( "p2", 20 );
        net.addTestComponent( "p3", 15 );
        net.finalizeConstruction();

        net.build( { 1 } );

        EXPECT_EQ( net.parameterCount(), 45u );
    }

    TEST_F( NetworkTests, ParameterCount_BeforeBuild_Throws )
    {
        TestableNetwork<DeviceType::Cpu> net( "param_pre_build", Device::Cpu() );

        net.addTestComponent( "unbuild", 5 );
        net.finalizeConstruction();

        EXPECT_THROW(
            net.parameterCount(),
            std::runtime_error
        );
    }

    

    // ====================================================================
    // Training Mode Tests
    // ====================================================================

    TEST_F( NetworkTests, SetTraining_PropagatestoChildren )
    {
        TestableNetwork<DeviceType::Cpu> net( "training_prop", Device::Cpu() );

        net.addTestComponent( "t1", 0 );
        net.addTestComponent( "t2", 0 );
        net.finalizeConstruction();

        net.build( { 1 } );

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
        TestableNetwork<DeviceType::Cpu> net( "sync_test", Device::Cpu() );

        net.addTestComponent( "s1", 0 );
        net.addTestComponent( "s2", 0 );
        net.finalizeConstruction();

        EXPECT_NO_THROW( net.synchronize() );
    }

    // ====================================================================
    // ToString Tests
    // ====================================================================

    TEST_F( NetworkTests, ToString_ContainsNameAndChildren )
    {
        TestableNetwork<DeviceType::Cpu> net( "info_test", Device::Cpu() );

        net.addTestComponent( "child_a", 0 );
        net.addTestComponent( "child_b", 0 );
        net.finalizeConstruction();

        std::string info = net.toString();

        EXPECT_NE( info.find( "info_test" ), std::string::npos );
        EXPECT_NE( info.find( "child_a" ), std::string::npos );
        EXPECT_NE( info.find( "child_b" ), std::string::npos );
    }

    // ====================================================================
    // Serialization Tests (Refactored to use SerializationMetadata)
    // ====================================================================

    TEST_F( NetworkTests, Save_WritesMetadataAndArchitecture )
    {
        auto path = makeTempArchivePath();
        std::error_code ec;
        std::filesystem::remove( path, ec );

        TestableNetwork<DeviceType::Cpu> net( "save_test", Device::Cpu() );

        net.addTestComponent( "mod1", 10 );
        net.addTestComponent( "mod2", 20 );
        net.finalizeConstruction();

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

            // Verify base network metadata using SerializationMetadata
            EXPECT_TRUE( archive.hasFile( "network/meta.json" ) );
            auto meta = archive.readMetadata( "network/meta.json" );

            EXPECT_EQ( meta.getString( "name" ), "save_test" );
            EXPECT_EQ( meta.getInt( "num_components" ), 2 );
            EXPECT_EQ( meta.getInt( "format_version" ), 1 );
            EXPECT_TRUE( meta.has( "export_time" ) );
            EXPECT_EQ( meta.getString( "mode" ),
                serializationModeToString( SerializationMode::Architecture ) );

            // Verify concrete class metadata
            EXPECT_TRUE( archive.hasFile( "network/testable_meta.json" ) );
            auto testable_meta = archive.readMetadata( "network/testable_meta.json" );
            EXPECT_EQ( testable_meta.getString( "type" ), "TestableNetwork" );
            EXPECT_EQ( testable_meta.getString( "test_metadata" ), "This is a test network" );

            // Verify architecture metadata
            EXPECT_TRUE( archive.hasFile( "network/architecture.json" ) );
            auto arch_meta = archive.readMetadata( "network/architecture.json" );
            EXPECT_EQ( arch_meta.getInt( "num_components" ), 2 );

            // Verify individual component descriptors
            EXPECT_TRUE( archive.hasFile( "network/component_mod1.json" ) );
            auto comp1_desc = archive.readMetadata( "network/component_mod1.json" );
            EXPECT_EQ( comp1_desc.getString( "name" ), "mod1" );
            EXPECT_EQ( comp1_desc.getString( "path" ), "components/mod1" );
            EXPECT_EQ( comp1_desc.getInt( "index" ), 0 );

            EXPECT_TRUE( archive.hasFile( "network/component_mod2.json" ) );
            auto comp2_desc = archive.readMetadata( "network/component_mod2.json" );
            EXPECT_EQ( comp2_desc.getString( "name" ), "mod2" );
            EXPECT_EQ( comp2_desc.getString( "path" ), "components/mod2" );
            EXPECT_EQ( comp2_desc.getInt( "index" ), 1 );

            // Verify component metadata was written
            EXPECT_TRUE( archive.hasFile( "components/mod1/meta.json" ) );
            auto mod1_meta = archive.readMetadata( "components/mod1/meta.json" );
            EXPECT_EQ( mod1_meta.getString( "component_type" ), "TestComponent" );
            EXPECT_EQ( mod1_meta.getInt( "param_count" ), 10 );

            EXPECT_TRUE( archive.hasFile( "components/mod2/meta.json" ) );
            auto mod2_meta = archive.readMetadata( "components/mod2/meta.json" );
            EXPECT_EQ( mod2_meta.getString( "component_type" ), "TestComponent" );
            EXPECT_EQ( mod2_meta.getInt( "param_count" ), 20 );

            archive.close();
        }

        std::filesystem::remove( path, ec );
    }

    TEST_F( NetworkTests, Save_BeforeBuild_Succeeds )
    {
        auto path = makeTempArchivePath();
        std::error_code ec;
        std::filesystem::remove( path, ec );

        TestableNetwork<DeviceType::Cpu> net( "unbuilt_save", Device::Cpu() );

        net.addTestComponent( "unbuild", 0 );
        net.finalizeConstruction();

        auto writer = std::make_unique<ZipSerializer>();
        ModelArchive archive( path.string(), std::move( writer ), OpenMode::Write );

        EXPECT_NO_THROW(
            net.save( archive, SerializationMode::Architecture )
        );

        archive.close();
        std::filesystem::remove( path, ec );
    }

    // ====================================================================
    // Empty Network Tests
    // ====================================================================

    TEST_F( NetworkTests, EmptyNetwork_HasNoChildren )
    {
        TestableNetwork<DeviceType::Cpu> net( "empty", Device::Cpu() );
        net.finalizeConstruction();

        EXPECT_EQ( net.childCount(), 0u );
        EXPECT_FALSE( net.hasChildren() );
        EXPECT_TRUE( net.getComponents().empty() );
    }

    // ====================================================================
    // Edge Case Tests
    // ====================================================================

    TEST_F( NetworkTests, GetComponents_ReturnsInsertionOrder )
    {
        TestableNetwork<DeviceType::Cpu> net( "order_test", Device::Cpu() );

        net.addTestComponent( "first", 1 );
        net.addTestComponent( "second", 2 );
        net.addTestComponent( "third", 3 );
        net.finalizeConstruction();

        const auto& components = net.getComponents();

        ASSERT_EQ( components.size(), 3u );
        EXPECT_EQ( components[ 0 ]->parameterCount(), 1u );
        EXPECT_EQ( components[ 1 ]->parameterCount(), 2u );
        EXPECT_EQ( components[ 2 ]->parameterCount(), 3u );
    }

    TEST_F( NetworkTests, GetNamedComponents_ContainsAllChildren )
    {
        TestableNetwork<DeviceType::Cpu> net( "named_test", Device::Cpu() );

        net.addTestComponent( "child1", 0 );
        net.addTestComponent( "child2", 0 );
        net.addTestComponent( "child3", 0 );
        net.finalizeConstruction();

        const auto& named = net.getComponents();

        EXPECT_EQ( named.size(), 3u );
    }
}