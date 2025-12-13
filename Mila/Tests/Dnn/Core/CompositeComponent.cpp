#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <vector>
#include <stdexcept>

import Mila;

namespace Dnn::Core::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    /**
     * @brief Minimal concrete child component used by tests.
     *
     * Updated to use the single IExecutionContext* constructor pattern
     * enforced by the Component base class. All TestChildComponent instances
     * share an ExecutionContext owned by the parent CompositeComponent.
     */
    class TestChildComponent : public Component<DeviceType::Cpu, TensorDataType::FP32>
    {
    public:
        using ComponentBase = Component<DeviceType::Cpu, TensorDataType::FP32>;

        /**
         * @brief Constructor for factory creation (shares parent's ExecutionContext).
         *
         * @param exec_context Non-owning pointer to shared execution context
         * @param param_count Mock parameter count for testing aggregation
         *
         * @throws std::invalid_argument if exec_context is null or device type mismatches
         */
        explicit TestChildComponent( IExecutionContext* exec_context, size_t param_count = 0 )
            : ComponentBase()
            , param_count_( param_count )
        {
            // setExecutionContext is protected in Component; derived class may call it.
            setExecutionContext( exec_context );
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

        std::string getName() const override
        {
            return name_;
        }

        void setName( const std::string& name )
        {
            name_ = name;
        }

        std::string toString() const override
        {
            return std::string( "TestChild:" ) + name_;
        }

        DeviceId getDeviceId() const override
        {
            return this->getExecutionContext()->getDeviceId();
        }

    protected:

        void onBuilding( const shape_t& /*input_shape*/ ) override
        {}

    private:
        std::string name_;
        size_t param_count_;
    };

    /**
     * @brief Testable composite that implements onBuilding to propagate build to children.
     *
     * Uses the simplified CompositeComponent which only accepts IExecutionContext* via
     * setExecutionContext called by derived class constructor for tests.
     */
    class TestableComposite : public CompositeComponent<DeviceType::Cpu, TensorDataType::FP32>
    {
    public:
        using Base = CompositeComponent<DeviceType::Cpu, TensorDataType::FP32>;

        explicit TestableComposite( IExecutionContext* exec_context )
            : Base()
        {
            // Set the execution context for this composite (protected helper)
            setExecutionContext( exec_context );
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
            comp_ = std::make_unique<TestableComposite>( exec_context_.get() );
        }

        void TearDown() override
        {
            comp_.reset();
            exec_context_.reset();
        }

        std::unique_ptr<IExecutionContext> exec_context_;
        std::unique_ptr<TestableComposite> comp_;
    };

    // ====================================================================
    // Chainable Factory Method Tests
    // ====================================================================

    TEST_F( CompositeComponentTests, AddComponent_Chainable )
    {
        comp_->addComponent<TestChildComponent>( "child_a", 3 )
            .addComponent<TestChildComponent>( "child_b", 5 );

        EXPECT_TRUE( comp_->hasComponent( "child_a" ) );
        EXPECT_TRUE( comp_->hasComponent( "child_b" ) );
        EXPECT_EQ( comp_->childCount(), 2u );
    }

    TEST_F( CompositeComponentTests, AddComponent_MultipleChained )
    {
        comp_->addComponent<TestChildComponent>( "comp1", 1 )
            .addComponent<TestChildComponent>( "comp2", 2 )
            .addComponent<TestChildComponent>( "comp3", 3 )
            .addComponent<TestChildComponent>( "comp4", 4 );

        EXPECT_EQ( comp_->childCount(), 4u );
        EXPECT_TRUE( comp_->hasComponent( "comp1" ) );
        EXPECT_TRUE( comp_->hasComponent( "comp2" ) );
        EXPECT_TRUE( comp_->hasComponent( "comp3" ) );
        EXPECT_TRUE( comp_->hasComponent( "comp4" ) );
    }

    // ====================================================================
    // Component Management Tests
    // ====================================================================

    TEST_F( CompositeComponentTests, AddGetHasRemoveComponentsBasic )
    {
        comp_->addComponent<TestChildComponent>( "child_a", 3 )
            .addComponent<TestChildComponent>( "child_b", 5 );

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
        comp_->addComponent<TestChildComponent>( "dup", 1 );

        EXPECT_THROW(
            comp_->addComponent<TestChildComponent>( "dup", 2 ),
            std::invalid_argument
        );
    }

    TEST_F( CompositeComponentTests, GetComponent_ExistingName )
    {
        comp_->addComponent<TestChildComponent>( "findme", 3 );

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
        comp_->addComponent<TestChildComponent>( "comp1", 0 )
            .addComponent<TestChildComponent>( "comp2", 0 )
            .addComponent<TestChildComponent>( "comp3", 0 );

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
        comp_->addComponent<TestChildComponent>( "build1", 5 )
            .addComponent<TestChildComponent>( "build2", 7 );

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
        comp_->addComponent<TestChildComponent>( "a", 10 )
            .addComponent<TestChildComponent>( "b", 20 );

        EXPECT_THROW( comp_->parameterCount(), std::runtime_error );
        EXPECT_THROW( comp_->getParameters(), std::runtime_error );

        comp_->build( { 1 } );

        EXPECT_EQ( comp_->parameterCount(), 30u );

        EXPECT_NO_THROW( comp_->getParameters() );
    }

    TEST_F( CompositeComponentTests, ParameterCount_BeforeBuild_Throws )
    {
        comp_->addComponent<TestChildComponent>( "unbuild", 5 );

        EXPECT_THROW(
            comp_->parameterCount(),
            std::runtime_error
        );
    }

    TEST_F( CompositeComponentTests, GetParameters_BeforeBuild_Throws )
    {
        comp_->addComponent<TestChildComponent>( "unbuild", 0 );

        EXPECT_THROW(
            comp_->getParameters(),
            std::runtime_error
        );
    }

    // ====================================================================
    // Training Mode Tests
    // ====================================================================

    TEST_F( CompositeComponentTests, GetGradientsPreconditions )
    {
        comp_->addComponent<TestChildComponent>( "a", 0 );

        EXPECT_THROW( comp_->getGradients(), std::runtime_error );

        comp_->build( { 1 } );

        EXPECT_FALSE( comp_->isTraining() );
        EXPECT_THROW( comp_->getGradients(), std::runtime_error );

        comp_->setTraining( true );
        EXPECT_NO_THROW( comp_->getGradients() );
    }

    TEST_F( CompositeComponentTests, TrainingModePropagation )
    {
        comp_->addComponent<TestChildComponent>( "train_a", 0 )
            .addComponent<TestChildComponent>( "train_b", 0 );

        auto child_a = comp_->getComponent( "train_a" );
        auto child_b = comp_->getComponent( "train_b" );

        EXPECT_FALSE( comp_->isTraining() );
        EXPECT_FALSE( child_a->isTraining() );
        EXPECT_FALSE( child_b->isTraining() );

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
        comp_->addComponent<TestChildComponent>( "a", 1 );

        comp_->build( { 1 } );

        EXPECT_THROW(
            comp_->addComponent<TestChildComponent>( "new", 1 ),
            std::runtime_error
        );
        EXPECT_THROW( comp_->removeComponent( "a" ), std::runtime_error );
        EXPECT_THROW( comp_->clearComponents(), std::runtime_error );
    }

    TEST_F( CompositeComponentTests, ClearComponents_AfterBuild_Throws )
    {
        comp_->addComponent<TestChildComponent>( "locked", 0 );

        comp_->build( { 1 } );

        EXPECT_THROW(
            comp_->clearComponents(),
            std::runtime_error
        );
    }

    TEST_F( CompositeComponentTests, RemoveComponent_AfterBuild_Throws )
    {
        comp_->addComponent<TestChildComponent>( "locked", 0 );

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
        comp_->addComponent<TestChildComponent>( "ctx_a", 0 )
            .addComponent<TestChildComponent>( "ctx_b", 0 );

        auto child_a = comp_->getComponent( "ctx_a" );
        auto child_b = comp_->getComponent( "ctx_b" );

        EXPECT_EQ( child_a->getDeviceId().type, DeviceType::Cpu );
        EXPECT_EQ( child_b->getDeviceId().type, DeviceType::Cpu );
        EXPECT_EQ( child_a->getDeviceId().type, comp_->getDeviceId().type );
        EXPECT_EQ( child_b->getDeviceId().type, comp_->getDeviceId().type );
    }

    // ====================================================================
    // Synchronization Tests
    // ====================================================================

    TEST_F( CompositeComponentTests, SynchronizePropagates )
    {
        comp_->addComponent<TestChildComponent>( "sync_a", 0 )
            .addComponent<TestChildComponent>( "sync_b", 0 );

        EXPECT_NO_THROW( comp_->synchronize() );
    }

    // ====================================================================
    // ToString Tests
    // ====================================================================

    TEST_F( CompositeComponentTests, ToStringContainsChildren )
    {
        comp_->addComponent<TestChildComponent>( "aa", 0 )
            .addComponent<TestChildComponent>( "bb", 0 );

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
        comp_->addComponent<TestChildComponent>( "first", 1 )
            .addComponent<TestChildComponent>( "second", 2 )
            .addComponent<TestChildComponent>( "third", 3 );

        const auto& components = comp_->getComponents();

        ASSERT_EQ( components.size(), 3u );
        EXPECT_EQ( components[ 0 ]->parameterCount(), 1u );
        EXPECT_EQ( components[ 1 ]->parameterCount(), 2u );
        EXPECT_EQ( components[ 2 ]->parameterCount(), 3u );
    }

    TEST_F( CompositeComponentTests, GetNamedComponents_ContainsAllChildren )
    {
        comp_->addComponent<TestChildComponent>( "child1", 0 )
            .addComponent<TestChildComponent>( "child2", 0 )
            .addComponent<TestChildComponent>( "child3", 0 );

        const auto& named = comp_->getNamedComponents();

        EXPECT_EQ( named.size(), 3u );
        EXPECT_TRUE( named.find( "child1" ) != named.end() );
        EXPECT_TRUE( named.find( "child2" ) != named.end() );
        EXPECT_TRUE( named.find( "child3" ) != named.end() );
    }

    // ====================================================================
    // Constructor Validation Tests
    // ====================================================================

    /*TEST_F( CompositeComponentTests, Constructor_NullExecutionContext_ThrowsInvalidArgument )
    {
        IExecutionContext* null_context = nullptr;

        EXPECT_THROW(
            TestableComposite( null_context ),
            std::invalid_argument
        );
    }*/

    TEST_F( CompositeComponentTests, Constructor_DeviceTypeMismatch_ThrowsInvalidArgument )
    {
        auto cuda_context = createExecutionContext( Device::Cuda( 0 ) );

        EXPECT_THROW(
            TestableComposite( cuda_context.get() ),
            std::invalid_argument
        );
    }
}