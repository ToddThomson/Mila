#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <vector>
#include <stdexcept>
#include <optional>

import Mila;

namespace Dnn::Core::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    /**
     * @brief Minimal concrete child component for testing.
     *
     * Updated to support both standalone and shared context modes following
     * the new Component architecture pattern. Component now requires a name
     * at construction time (component-owns-name).
     */
    class TestChildComponent : public Component<DeviceType::Cpu, TensorDataType::FP32>
    {
    public:
        using ComponentBase = Component<DeviceType::Cpu, TensorDataType::FP32>;

        /**
         * @brief Constructor for shared mode (context provided by parent).
         *
         * @param name Component name (required by new Component ctor)
         * @param param_count Mock parameter count for testing aggregation
         * @param device_id Optional device for standalone mode
         */
        explicit TestChildComponent( 
            const std::string& name,
            size_t param_count = 0,
            std::optional<DeviceId> device_id = std::nullopt )
            : ComponentBase( name ), param_count_( param_count )
        {
            if ( device_id.has_value() )
            {
                if ( device_id->type != DeviceType::Cpu )
                {
                    throw std::invalid_argument( "TestChildComponent: device type mismatch" );
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

            return std::string( "TestChild:" ) + this->getName();
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
     * @brief Testable composite that implements onBuilding to propagate build to children.
     *
     * Updated to use the new addComponent(component) API with component-owns-name pattern.
     * Provides public wrapper for setExecutionContext() to enable test fixtures to set context.
     */
    class TestableComposite : public CompositeComponent<DeviceType::Cpu, TensorDataType::FP32>
    {
    public:
        using Base = CompositeComponent<DeviceType::Cpu, TensorDataType::FP32>;

        explicit TestableComposite( const std::string& name = "testable" )
            : Base( name )
        {}

        /**
         * @brief Public wrapper to set execution context (for testing only).
         *
         * Exposes the protected setExecutionContext() method for test fixtures.
         * In production code, context is set by parent or via standalone mode.
         *
         * @param context Non-owning pointer to shared execution context
         *
         * @throws std::invalid_argument if context is null or device type mismatches
         */
        void setTestExecutionContext( IExecutionContext* context )
        {
            this->setExecutionContext( context );
        }

        bool hasExecutionContext() const noexcept
        {
            return this->hasExecutionContext();
        }

        /**
         * @brief Helper to add a child component with explicit name and parameter count.
         *
         * Creates a TestChildComponent in shared mode (name-owned-by-component) and registers it.
         * This mimics the pattern used in real composites like MLP.
         */
        void addTestChild( const std::string& name, size_t param_count = 0 )
        {
            auto component = std::make_shared<TestChildComponent>( name, param_count, std::nullopt );
            this->addComponent( component );
        }

        const ComponentType getType() const override
        {
            return ComponentType::MockComponent;
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

    class CompositeComponentTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            exec_context_ = createExecutionContext( Device::Cpu() );
            comp_ = std::make_unique<TestableComposite>();

            comp_->setTestExecutionContext( exec_context_.get() );
        }

        void TearDown() override
        {
            comp_.reset();
            exec_context_.reset();
        }

        std::unique_ptr<IExecutionContext> exec_context_{ nullptr };
        std::unique_ptr<TestableComposite> comp_;
    };

    // ====================================================================
    // Chainable addComponent() Tests
    // ====================================================================

    TEST_F( CompositeComponentTests, AddComponent_Chainable )
    {
        comp_->addTestChild( "child_a", 3 );
        comp_->addTestChild( "child_b", 5 );

        EXPECT_TRUE( comp_->hasComponent( "child_a" ) );
        EXPECT_TRUE( comp_->hasComponent( "child_b" ) );
        EXPECT_EQ( comp_->childCount(), 2u );
    }

    TEST_F( CompositeComponentTests, AddComponent_MultipleChained )
    {
        comp_->addTestChild( "comp1", 1 );
        comp_->addTestChild( "comp2", 2 );
        comp_->addTestChild( "comp3", 3 );
        comp_->addTestChild( "comp4", 4 );

        EXPECT_EQ( comp_->childCount(), 4u );
        EXPECT_TRUE( comp_->hasComponent( "comp1" ) );
        EXPECT_TRUE( comp_->hasComponent( "comp2" ) );
        EXPECT_TRUE( comp_->hasComponent( "comp3" ) );
        EXPECT_TRUE( comp_->hasComponent( "comp4" ) );
    }

    TEST_F( CompositeComponentTests, AddComponent_ComponentOwnsName )
    {
        auto child = std::make_shared<TestChildComponent>( "named_child", 5, std::nullopt );

        comp_->addComponent( child );

        EXPECT_TRUE( comp_->hasComponent( "named_child" ) );
        EXPECT_EQ( comp_->getComponent( "named_child" )->getName(), "named_child" );
    }

    // ====================================================================
    // Component Management Tests
    // ====================================================================

    TEST_F( CompositeComponentTests, AddGetHasRemoveComponentsBasic )
    {
        comp_->addTestChild( "child_a", 3 );
        comp_->addTestChild( "child_b", 5 );

        EXPECT_TRUE( comp_->hasComponent( "child_a" ) );
        EXPECT_TRUE( comp_->hasComponent( "child_b" ) );
        EXPECT_EQ( comp_->childCount(), 2u );

        auto got = comp_->getComponent( "child_a" );
        EXPECT_NE( got, nullptr );
        EXPECT_EQ( got->parameterCount(), 3u );

        EXPECT_FALSE( comp_->removeComponent( "no_such" ) );

        EXPECT_TRUE( comp_->removeComponent( "child_a" ) );
        EXPECT_FALSE( comp_->hasComponent( "child_a" ) );
        EXPECT_EQ( comp_->childCount(), 1u );

        EXPECT_NO_THROW( comp_->clearComponents() );
        EXPECT_EQ( comp_->childCount(), 0u );
    }

    TEST_F( CompositeComponentTests, DuplicateNameRejected )
    {
        comp_->addTestChild( "dup", 1 );

        auto duplicate = std::make_shared<TestChildComponent>( "dup", 2, std::nullopt );

        EXPECT_THROW(
            comp_->addComponent( duplicate ),
            std::invalid_argument
        );
    }

    TEST_F( CompositeComponentTests, AddComponent_NullComponent_Throws )
    {
        EXPECT_THROW(
            comp_->addComponent( nullptr ),
            std::invalid_argument
        );
    }

    TEST_F( CompositeComponentTests, GetComponent_ExistingName )
    {
        comp_->addTestChild( "findme", 3 );

        auto retrieved = comp_->getComponent( "findme" );

        EXPECT_NE( retrieved, nullptr );
        EXPECT_EQ( retrieved->parameterCount(), 3u );
    }

    TEST_F( CompositeComponentTests, GetComponent_NonExistentName_Throws )
    {
        EXPECT_THROW(
            comp_->getComponent( "nonexistent" ),
            std::out_of_range
        );
    }

    TEST_F( CompositeComponentTests, ClearComponents_RemovesAll )
    {
        comp_->addTestChild( "comp1", 0 );
        comp_->addTestChild( "comp2", 0 );
        comp_->addTestChild( "comp3", 0 );

        EXPECT_EQ( comp_->childCount(), 3u );

        comp_->clearComponents();

        EXPECT_EQ( comp_->childCount(), 0u );
        EXPECT_FALSE( comp_->hasComponent( "comp1" ) );
        EXPECT_FALSE( comp_->hasComponent( "comp2" ) );
        EXPECT_FALSE( comp_->hasComponent( "comp3" ) );
    }

    // ====================================================================
    // Build Lifecycle Tests
    // ====================================================================

    TEST_F( CompositeComponentTests, Build_PropagatestoChildren )
    {
        comp_->addTestChild( "build1", 5 );
        comp_->addTestChild( "build2", 7 );

        auto child1 = comp_->getComponent( "build1" );
        auto child2 = comp_->getComponent( "build2" );

        EXPECT_FALSE( comp_->isBuilt() );
        EXPECT_FALSE( child1->isBuilt() );
        EXPECT_FALSE( child2->isBuilt() );

        comp_->build( { 2, 3 } );

        EXPECT_TRUE( comp_->isBuilt() );
        EXPECT_TRUE( child1->isBuilt() );
        EXPECT_TRUE( child2->isBuilt() );
    }

    TEST_F( CompositeComponentTests, BuildAndParameterAggregation )
    {
        comp_->addTestChild( "a", 10 );
        comp_->addTestChild( "b", 20 );

        EXPECT_THROW( comp_->parameterCount(), std::runtime_error );
        //EXPECT_THROW( comp_->getParameters(), std::runtime_error );

        comp_->build( { 1 } );

        EXPECT_EQ( comp_->parameterCount(), 30u );

        //EXPECT_NO_THROW( comp_->getParameters() );
    }

    TEST_F( CompositeComponentTests, ParameterCount_BeforeBuild_Throws )
    {
        comp_->addTestChild( "unbuild", 5 );

        EXPECT_THROW(
            comp_->parameterCount(),
            std::runtime_error
        );
    }

    /*TEST_F( CompositeComponentTests, GetParameters_BeforeBuild_Throws )
    {
        comp_->addTestChild( "unbuild", 0 );

        EXPECT_THROW(
            comp_->getParameters(),
            std::runtime_error
        );
    }*/

    // ====================================================================
    // Training Mode Tests
    // ====================================================================

    TEST_F( CompositeComponentTests, TrainingModePropagation )
    {
        comp_->addTestChild( "train_a", 0 );
        comp_->addTestChild( "train_b", 0 );

        auto child_a = comp_->getComponent( "train_a" );
        auto child_b = comp_->getComponent( "train_b" );

        EXPECT_FALSE( comp_->isTraining() );
        EXPECT_FALSE( child_a->isTraining() );
        EXPECT_FALSE( child_b->isTraining() );

        // Build before enabling training to satisfy Component lifecycle contract.
        comp_->build( { 1 } );

        comp_->setTraining( true );

        EXPECT_TRUE( comp_->isTraining() );
        EXPECT_TRUE( child_a->isTraining() );
        EXPECT_TRUE( child_b->isTraining() );

        comp_->setTraining( false );

        EXPECT_FALSE( comp_->isTraining() );
        EXPECT_FALSE( child_a->isTraining() );
        EXPECT_FALSE( child_b->isTraining() );
    }

    // ====================================================================
    // Mutation After Build Tests
    // ====================================================================

    TEST_F( CompositeComponentTests, MutationAfterBuildIsRejected )
    {
        comp_->addTestChild( "a", 1 );

        comp_->build( { 1 } );

        EXPECT_THROW(
            comp_->addTestChild( "new", 1 ),
            std::runtime_error
        );
        EXPECT_THROW( comp_->removeComponent( "a" ), std::runtime_error );
        EXPECT_THROW( comp_->clearComponents(), std::runtime_error );
    }

    TEST_F( CompositeComponentTests, ClearComponents_AfterBuild_Throws )
    {
        comp_->addTestChild( "locked", 0 );

        comp_->build( { 1 } );

        EXPECT_THROW(
            comp_->clearComponents(),
            std::runtime_error
        );
    }

    TEST_F( CompositeComponentTests, RemoveComponent_AfterBuild_Throws )
    {
        comp_->addTestChild( "locked", 0 );

        comp_->build( { 1 } );

        EXPECT_THROW(
            comp_->removeComponent( "locked" ),
            std::runtime_error
        );
    }

    // ====================================================================
    // ExecutionContext Sharing Tests
    // ====================================================================

    TEST_F( CompositeComponentTests, ExecutionContextSharedAcrossChildren )
    {
        comp_->addTestChild( "ctx_a", 0 );
        comp_->addTestChild( "ctx_b", 0 );

        auto child_a = comp_->getComponent( "ctx_a" );
        auto child_b = comp_->getComponent( "ctx_b" );

        EXPECT_EQ( child_a->getDeviceId().type, DeviceType::Cpu );
        EXPECT_EQ( child_b->getDeviceId().type, DeviceType::Cpu );
        EXPECT_EQ( child_a->getDeviceId().type, comp_->getDeviceId().type );
        EXPECT_EQ( child_b->getDeviceId().type, comp_->getDeviceId().type );
    }

    TEST_F( CompositeComponentTests, ExecutionContextPropagatedOnAdd )
    {
        auto composite = std::make_unique<TestableComposite>();
        auto child = std::make_shared<TestChildComponent>( "late_add", 0, std::nullopt );

        composite->addComponent( child );

        composite->setTestExecutionContext( exec_context_.get() );

        EXPECT_EQ( child->getDeviceId().type, DeviceType::Cpu );
    }

    TEST_F( CompositeComponentTests, ExecutionContextPropagatedViaHook )
    {
        auto composite = std::make_unique<TestableComposite>();

        composite->addTestChild( "child1", 0 );
        composite->addTestChild( "child2", 0 );

        auto child1 = composite->getComponent( "child1" );
        auto child2 = composite->getComponent( "child2" );

        //EXPECT_FALSE( child1->hasExecutionContext() );
        //EXPECT_FALSE( child2->hasExecutionContext() );

        composite->setTestExecutionContext( exec_context_.get() );

        //EXPECT_TRUE( child1->hasExecutionContext() );
        //EXPECT_TRUE( child2->hasExecutionContext() );
        EXPECT_EQ( child1->getDeviceId().type, DeviceType::Cpu );
        EXPECT_EQ( child2->getDeviceId().type, DeviceType::Cpu );
    }

    // ====================================================================
    // Synchronization Tests
    // ====================================================================

    TEST_F( CompositeComponentTests, SynchronizePropagates )
    {
        comp_->addTestChild( "sync_a", 0 );
        comp_->addTestChild( "sync_b", 0 );

        EXPECT_NO_THROW( comp_->synchronize() );
    }

    // ====================================================================
    // ToString Tests
    // ====================================================================

    TEST_F( CompositeComponentTests, ToStringContainsChildren )
    {
        comp_->addTestChild( "aa", 0 );
        comp_->addTestChild( "bb", 0 );

        std::string s = comp_->toString();
        EXPECT_NE( s.find( "aa" ), std::string::npos );
        EXPECT_NE( s.find( "bb" ), std::string::npos );
    }

    // ====================================================================
    // Edge Case Tests
    // ====================================================================

    TEST_F( CompositeComponentTests, EmptyCompositeHasNoChildren )
    {
        EXPECT_EQ( comp_->childCount(), 0u );
        EXPECT_FALSE( comp_->hasChildren() );
        EXPECT_TRUE( comp_->getComponents().empty() );
        EXPECT_TRUE( comp_->getNamedComponents().empty() );
    }

    TEST_F( CompositeComponentTests, GetComponents_ReturnsInsertionOrder )
    {
        comp_->addTestChild( "first", 1 );
        comp_->addTestChild( "second", 2 );
        comp_->addTestChild( "third", 3 );

        const auto& components = comp_->getComponents();

        ASSERT_EQ( components.size(), 3u );
        EXPECT_EQ( components[ 0 ]->parameterCount(), 1u );
        EXPECT_EQ( components[ 1 ]->parameterCount(), 2u );
        EXPECT_EQ( components[ 2 ]->parameterCount(), 3u );
    }

    TEST_F( CompositeComponentTests, GetNamedComponents_ContainsAllChildren )
    {
        comp_->addTestChild( "child1", 0 );
        comp_->addTestChild( "child2", 0 );
        comp_->addTestChild( "child3", 0 );

        const auto& named = comp_->getNamedComponents();

        EXPECT_EQ( named.size(), 3u );
        EXPECT_TRUE( named.find( "child1" ) != named.end() );
        EXPECT_TRUE( named.find( "child2" ) != named.end() );
        EXPECT_TRUE( named.find( "child3" ) != named.end() );
    }

    // ====================================================================
    // Constructor Validation Tests
    // ====================================================================

    TEST_F( CompositeComponentTests, Constructor_DeviceTypeMismatch_ThrowsInvalidArgument )
    {
        auto cuda_context = createExecutionContext( Device::Cuda( 0 ) );
        auto composite = std::make_unique<TestableComposite>();

        EXPECT_THROW(
            composite->setTestExecutionContext( cuda_context.get() ),
            std::invalid_argument
        );
    }
}