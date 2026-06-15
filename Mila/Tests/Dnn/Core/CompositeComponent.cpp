/**
 * @file CompositeComponent.cpp
 * @brief Base-contract tests for CompositeComponent<TDeviceType, TPrecision>.
 *
 * Companion to Component.cpp: where that file proves the leaf base contract,
 * this proves the CONTAINER contract that every composite (Network, MLP,
 * GptBlock, and the MNIST MnistClassifier) inherits exactly once -- child
 * management, execution-context propagation to children, parameter/gradient
 * aggregation, build propagation, and training-mode propagation. Concrete
 * composites (e.g. the MLP in Modeling/Network.Cpu.cpp) inherit this guarantee
 * and assert only their own forward/backward delta.
 *
 * CompositeComponent is abstract, so the mock IS the surface: TestComposite
 * implements the minimal pure-virtual remainder (getType / getMemoryStats /
 * onBuilding) and exposes the protected setExecutionContext / getComponentAs
 * forwarders so the container paths can be driven without a Network. MockChild
 * is a leaf carrying a configurable number of parameter/gradient tensors so the
 * aggregation paths return non-empty.
 *
 * CPU device, so this rides the MILA_ENABLE_CUDA=OFF CI gate.
 */

#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <vector>
#include <stdexcept>

import Mila;

namespace Mila::Tests::Dnn::Core
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    namespace
    {
        // ================================================================
        // Mock leaf child: carries num_params parameter/gradient tensors so
        // the composite's aggregation paths have something to collect.
        // ================================================================
        class MockChild : public Component<DeviceType::Cpu, TensorDataType::FP32>
        {
        public:
            using Base = Component<DeviceType::Cpu, TensorDataType::FP32>;
            using TensorType = Tensor<TensorDataType::FP32, CpuMemoryResource>;

            explicit MockChild( const std::string& name, size_t num_params = 0 )
                : Base( name )
            {
                for ( size_t i = 0; i < num_params; ++i )
                {
                    parameters_.push_back( std::make_shared<TensorType>( Device::Cpu(), shape_t{ 1 } ) );
                    gradients_.push_back( std::make_shared<TensorType>( Device::Cpu(), shape_t{ 1 } ) );
                }
            }

            void exposeSetExecutionContext( IExecutionContext* context )
            {
                this->setExecutionContext( context );
            }

            void synchronize() override
            {}

            size_t parameterCount() const override
            {
                return parameters_.size();
            }

            std::vector<ITensor*> getParameters() const override
            {
                std::vector<ITensor*> out;
                for ( const auto& p : parameters_ )
                {
                    out.push_back( p.get() );
                }

                return out;
            }

            std::vector<ITensor*> getGradients() const override
            {
                std::vector<ITensor*> out;
                for ( const auto& g : gradients_ )
                {
                    out.push_back( g.get() );
                }

                return out;
            }

            void save_( ModelArchive&, SerializationMode ) const override
            {}

            MemoryStats getMemoryStats() const override
            {
                return {};
            }

            const ComponentType getType() const override
            {
                return ComponentType::MockComponent;
            }

            DeviceId getDeviceId() const override
            {
                return this->getExecutionContext()->getDeviceId();
            }

            std::string toString() const override
            {
                return std::string( "MockChild:" ) + this->getName();
            }

        protected:
            void onBuilding( const BuildContext& ) override
            {}

        private:
            std::vector<std::shared_ptr<TensorType>> parameters_;
            std::vector<std::shared_ptr<TensorType>> gradients_;
        };

        // ================================================================
        // Concrete composite under test. Builds children with the composite's
        // own BuildContext (mock children ignore the shape).
        // ================================================================
        template<DeviceType TDeviceType, TensorDataType TPrecision = TensorDataType::FP32>
        class TestComposite : public CompositeComponent<TDeviceType, TPrecision>
        {
        public:
            using Base = CompositeComponent<TDeviceType, TPrecision>;

            explicit TestComposite( const std::string& name )
                : Base( name )
            {}

            void exposeSetExecutionContext( IExecutionContext* context )
            {
                this->setExecutionContext( context );
            }

            template<typename TComponent>
            std::shared_ptr<TComponent> exposeGetComponentAs( const std::string& name ) const
            {
                return this->template getComponentAs<TComponent>( name );
            }

            int onBuilding_calls = 0;

            MemoryStats getMemoryStats() const override
            {
                return {};
            }

            const ComponentType getType() const override
            {
                return ComponentType::MockComponent;
            }

        protected:
            void onBuilding( const BuildContext& context ) override
            {
                ++onBuilding_calls;

                for ( const auto& child : this->getComponents() )
                {
                    if ( !child->isBuilt() )
                    {
                        child->build( context );
                    }
                }
            }
        };

        using Composite = TestComposite<DeviceType::Cpu>;
    }

    class CompositeComponentTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            exec_context_ = createExecutionContext( Device::Cpu() );
        }

        // A composite already wired to the CPU context, ready for addComponent/build.
        std::unique_ptr<Composite> contextual( const std::string& name = "composite" )
        {
            auto composite = std::make_unique<Composite>( name );
            composite->exposeSetExecutionContext( exec_context_.get() );

            return composite;
        }

        static BuildContext build( RuntimeMode mode )
        {
            return BuildContext( shape_t{ 2, 3 }, mode );
        }

        std::unique_ptr<IExecutionContext> exec_context_;
    };

    // ====================================================================
    // Child registration
    // ====================================================================

    TEST_F( CompositeComponentTests, AddComponent_RegistersAndIsChainable )
    {
        auto composite = contextual();

        composite->addComponent( std::make_shared<MockChild>( "a" ) )
            .addComponent( std::make_shared<MockChild>( "b" ) );

        EXPECT_EQ( composite->childCount(), 2u );
        EXPECT_TRUE( composite->hasComponent( "a" ) );
        EXPECT_TRUE( composite->hasComponent( "b" ) );
        EXPECT_TRUE( composite->hasChildren() );
    }

    TEST_F( CompositeComponentTests, AddComponent_ThrowsOnNull )
    {
        auto composite = contextual();

        EXPECT_THROW( composite->addComponent( nullptr ), std::invalid_argument );
    }

    TEST_F( CompositeComponentTests, AddComponent_ThrowsOnDuplicateName )
    {
        auto composite = contextual();
        composite->addComponent( std::make_shared<MockChild>( "dup" ) );

        EXPECT_THROW( composite->addComponent( std::make_shared<MockChild>( "dup" ) ), std::invalid_argument );
    }

    TEST_F( CompositeComponentTests, AddComponent_ThrowsWhenChildHasOwnContext )
    {
        auto composite = contextual();

        auto child = std::make_shared<MockChild>( "standalone" );
        child->exposeSetExecutionContext( exec_context_.get() );

        EXPECT_THROW( composite->addComponent( child ), std::invalid_argument );
    }

    TEST_F( CompositeComponentTests, AddComponent_ThrowsAfterBuild )
    {
        auto composite = contextual();
        composite->addComponent( std::make_shared<MockChild>( "a" ) );
        composite->build( build( RuntimeMode::Inference ) );

        EXPECT_THROW( composite->addComponent( std::make_shared<MockChild>( "late" ) ), std::runtime_error );
    }

    // ====================================================================
    // Context propagation to children
    // ====================================================================

    TEST_F( CompositeComponentTests, AddComponent_PropagatesContextWhenCompositeAlreadyHasOne )
    {
        auto composite = contextual();

        auto child = std::make_shared<MockChild>( "c" );
        composite->addComponent( child );

        // Composite already had context, so the child receives it on add.
        EXPECT_EQ( child->getDeviceId().type, DeviceType::Cpu );
    }

    TEST_F( CompositeComponentTests, SetExecutionContext_PropagatesToPreexistingChildren )
    {
        // Add children BEFORE the context exists, then attach: onExecutionContextSet
        // must propagate to all pre-registered children.
        auto composite = std::make_unique<Composite>( "deferred" );

        auto child = std::make_shared<MockChild>( "c" );
        composite->addComponent( child );

        composite->exposeSetExecutionContext( exec_context_.get() );

        EXPECT_EQ( child->getDeviceId().type, DeviceType::Cpu );
    }

    // ====================================================================
    // Lookup, removal, ordering
    // ====================================================================

    TEST_F( CompositeComponentTests, GetComponent_ReturnsChild )
    {
        auto composite = contextual();
        auto child = std::make_shared<MockChild>( "find_me", 3 );
        composite->addComponent( child );

        EXPECT_EQ( composite->getComponent( "find_me" ), child );
    }

    TEST_F( CompositeComponentTests, GetComponent_ThrowsOnUnknownName )
    {
        auto composite = contextual();

        EXPECT_THROW( composite->getComponent( "missing" ), std::out_of_range );
    }

    TEST_F( CompositeComponentTests, TryFindComponent_ReturnsChildOrNull )
    {
        auto composite = contextual();
        composite->addComponent( std::make_shared<MockChild>( "child1" ) );

        EXPECT_NE( composite->tryFindComponent( "child1" ), nullptr );
        EXPECT_EQ( composite->tryFindComponent( "nope" ), nullptr );
    }

    TEST_F( CompositeComponentTests, GetComponents_PreservesInsertionOrder )
    {
        auto composite = contextual();
        composite->addComponent( std::make_shared<MockChild>( "first", 1 ) );
        composite->addComponent( std::make_shared<MockChild>( "second", 2 ) );
        composite->addComponent( std::make_shared<MockChild>( "third", 3 ) );

        const auto& children = composite->getComponents();

        ASSERT_EQ( children.size(), 3u );
        EXPECT_EQ( children[ 0 ]->getName(), "first" );
        EXPECT_EQ( children[ 1 ]->getName(), "second" );
        EXPECT_EQ( children[ 2 ]->getName(), "third" );
    }

    TEST_F( CompositeComponentTests, RemoveComponent_RemovesExistingReturnsTrue )
    {
        auto composite = contextual();
        composite->addComponent( std::make_shared<MockChild>( "victim" ) );

        EXPECT_TRUE( composite->removeComponent( "victim" ) );
        EXPECT_FALSE( composite->hasComponent( "victim" ) );
        EXPECT_EQ( composite->childCount(), 0u );
    }

    TEST_F( CompositeComponentTests, RemoveComponent_ReturnsFalseWhenMissing )
    {
        auto composite = contextual();

        EXPECT_FALSE( composite->removeComponent( "never" ) );
    }

    TEST_F( CompositeComponentTests, RemoveComponent_ThrowsAfterBuild )
    {
        auto composite = contextual();
        composite->addComponent( std::make_shared<MockChild>( "locked" ) );
        composite->build( build( RuntimeMode::Inference ) );

        EXPECT_THROW( composite->removeComponent( "locked" ), std::runtime_error );
    }

    TEST_F( CompositeComponentTests, ClearComponents_RemovesAll )
    {
        auto composite = contextual();
        composite->addComponent( std::make_shared<MockChild>( "a" ) );
        composite->addComponent( std::make_shared<MockChild>( "b" ) );

        composite->clearComponents();

        EXPECT_EQ( composite->childCount(), 0u );
        EXPECT_FALSE( composite->hasChildren() );
    }

    TEST_F( CompositeComponentTests, ClearComponents_ThrowsAfterBuild )
    {
        auto composite = contextual();
        composite->addComponent( std::make_shared<MockChild>( "a" ) );
        composite->build( build( RuntimeMode::Inference ) );

        EXPECT_THROW( composite->clearComponents(), std::runtime_error );
    }

    // ====================================================================
    // getComponentAs (typed retrieval used by concrete composites)
    // ====================================================================

    TEST_F( CompositeComponentTests, GetComponentAs_ReturnsTypedChild )
    {
        auto composite = contextual();
        composite->addComponent( std::make_shared<MockChild>( "typed", 2 ) );

        auto child = composite->exposeGetComponentAs<MockChild>( "typed" );

        ASSERT_NE( child, nullptr );
        EXPECT_EQ( child->parameterCount(), 2u );
    }

    TEST_F( CompositeComponentTests, GetComponentAs_ThrowsOnTypeMismatch )
    {
        auto composite = contextual();
        composite->addComponent( std::make_shared<MockChild>( "leaf" ) );

        // A MockChild is not a Composite; the dynamic cast must fail.
        EXPECT_THROW( composite->exposeGetComponentAs<Composite>( "leaf" ), std::runtime_error );
    }

    // ====================================================================
    // Build propagation
    // ====================================================================

    TEST_F( CompositeComponentTests, Build_InvokesOnBuildingAndPropagatesToChildren )
    {
        auto composite = contextual();
        auto a = std::make_shared<MockChild>( "a" );
        auto b = std::make_shared<MockChild>( "b" );
        composite->addComponent( a );
        composite->addComponent( b );

        EXPECT_FALSE( composite->isBuilt() );

        composite->build( build( RuntimeMode::Inference ) );

        EXPECT_TRUE( composite->isBuilt() );
        EXPECT_EQ( composite->onBuilding_calls, 1 );
        EXPECT_TRUE( a->isBuilt() );
        EXPECT_TRUE( b->isBuilt() );
    }

    // ====================================================================
    // Parameter / gradient aggregation
    // ====================================================================

    TEST_F( CompositeComponentTests, ParameterCount_ThrowsBeforeBuild )
    {
        auto composite = contextual();
        composite->addComponent( std::make_shared<MockChild>( "a", 5 ) );

        EXPECT_THROW( composite->parameterCount(), std::runtime_error );
    }

    TEST_F( CompositeComponentTests, ParameterCount_AggregatesChildren )
    {
        auto composite = contextual();
        composite->addComponent( std::make_shared<MockChild>( "a", 2 ) );
        composite->addComponent( std::make_shared<MockChild>( "b", 3 ) );
        composite->build( build( RuntimeMode::Inference ) );

        EXPECT_EQ( composite->parameterCount(), 5u );
    }

    TEST_F( CompositeComponentTests, GetParameters_ThrowsBeforeBuild )
    {
        auto composite = contextual();
        composite->addComponent( std::make_shared<MockChild>( "a", 2 ) );

        EXPECT_THROW( composite->getParameters(), std::runtime_error );
    }

    TEST_F( CompositeComponentTests, GetParameters_AggregatesAcrossChildren )
    {
        auto composite = contextual();
        composite->addComponent( std::make_shared<MockChild>( "a", 2 ) );
        composite->addComponent( std::make_shared<MockChild>( "b", 3 ) );
        composite->build( build( RuntimeMode::Training ) );

        EXPECT_EQ( composite->getParameters().size(), 5u );
    }

    TEST_F( CompositeComponentTests, GetGradients_ThrowsBeforeBuild )
    {
        auto composite = contextual();
        composite->addComponent( std::make_shared<MockChild>( "a", 2 ) );

        EXPECT_THROW( composite->getGradients(), std::runtime_error );
    }

    TEST_F( CompositeComponentTests, GetGradients_ThrowsWhenBuiltForInference )
    {
        auto composite = contextual();
        composite->addComponent( std::make_shared<MockChild>( "a", 2 ) );
        composite->build( build( RuntimeMode::Inference ) );

        EXPECT_THROW( composite->getGradients(), std::runtime_error );
    }

    TEST_F( CompositeComponentTests, GetGradients_AggregatesWhenBuiltForTraining )
    {
        auto composite = contextual();
        composite->addComponent( std::make_shared<MockChild>( "a", 2 ) );
        composite->addComponent( std::make_shared<MockChild>( "b", 3 ) );
        composite->build( build( RuntimeMode::Training ) );

        EXPECT_EQ( composite->getGradients().size(), 5u );
    }

    // ====================================================================
    // Training-mode propagation
    // ====================================================================

    TEST_F( CompositeComponentTests, SetTrainingMode_PropagatesToChildren )
    {
        auto composite = contextual();
        auto a = std::make_shared<MockChild>( "a" );
        auto b = std::make_shared<MockChild>( "b" );
        composite->addComponent( a );
        composite->addComponent( b );
        composite->build( build( RuntimeMode::Training ) );

        composite->setTrainingMode( TrainingMode::Eval );

        EXPECT_EQ( composite->getTrainingMode(), TrainingMode::Eval );
        EXPECT_EQ( a->getTrainingMode(), TrainingMode::Eval );
        EXPECT_EQ( b->getTrainingMode(), TrainingMode::Eval );
    }

    // ====================================================================
    // Diagnostics
    // ====================================================================

    TEST_F( CompositeComponentTests, ToString_NamesChildren )
    {
        auto composite = contextual( "root" );
        composite->addComponent( std::make_shared<MockChild>( "alpha" ) );

        const std::string text = composite->toString();

        EXPECT_NE( text.find( "root" ), std::string::npos );
        EXPECT_NE( text.find( "alpha" ), std::string::npos );
    }
}
