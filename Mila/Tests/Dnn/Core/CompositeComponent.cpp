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
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <format>
#include <memory>
#include <string>
#include <system_error>
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
        // Mock leaf child: carries num_params parameter tensors (always) plus
        // matching gradient tensors (only when built for training) so the
        // composite's aggregation paths have something to collect.
        // ================================================================
        class MockChild : public Component<DeviceType::Cpu, TensorDataType::FP32>
        {
        public:
            using Base = Component<DeviceType::Cpu, TensorDataType::FP32>;
            using TensorType = Tensor<TensorDataType::FP32, CpuMemoryResource>;

            explicit MockChild( const std::string& name, size_t num_params = 0 )
                : Base( name ), num_params_( num_params )
            {
                for ( size_t i = 0; i < num_params; ++i )
                {
                    parameters_.push_back( std::make_shared<TensorType>( Device::Cpu(), shape_t{ 1 } ) );
                }
            }

            void exposeSetExecutionContext( IExecutionContext* context )
            {
                this->setExecutionContext( context );
            }

            void synchronize() override
            {}

            dim_t parameterCount() const override
            {
                return static_cast<dim_t>( parameters_.size() );
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

            // Names its parameters "p0", "p1", ... so the composite's save traversal has a
            // real leaf to drive: the guard passes and save_ writes actual tensor blobs,
            // which is what makes a path-collision observable.
            std::vector<std::string> getParameterNames() const override
            {
                std::vector<std::string> names;

                for ( size_t i = 0; i < parameters_.size(); ++i )
                {
                    names.push_back( "p" + std::to_string( i ) );
                }

                return names;
            }

            void save_( ModelArchive& archive, SerializationMode ) const override
            {
                SerializationMetadata meta;
                meta.set( "type", "MockChild" )
                    .set( "name", this->getName() );

                archive.writeMetadata( "meta.json", meta );

                const auto names = getParameterNames();

                for ( size_t i = 0; i < names.size(); ++i )
                {
                    this->saveParameterToArchive( archive, names[ i ], *parameters_[ i ] );
                }
            }

            // The load counterpart of save_: resolves the name back to its slot through
            // the same getParameterNames() vector, so the round trip is keyed on one list.
            void loadParameter( const std::string& name, const ITensorBlob& blob ) override
            {
                const auto names = getParameterNames();
                const auto position = std::find( names.begin(), names.end(), name );

                if ( position == names.end() )
                {
                    Base::loadParameter( name, blob );
                    return;
                }

                const size_t index = static_cast<size_t>( std::distance( names.begin(), position ) );

                this->loadParameterFromBlob( name, blob, *parameters_[ index ], parameters_[ index ]->shape() );
            }

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
            void onBuilding( const BuildContext& context ) override
            {
                // Mirror the real leaves (e.g. Linear): gradient buffers are allocated
                // only when built for training, so getGradients() is empty after an
                // inference build.
                if ( context.isTrainingMode() )
                {
                    for ( size_t i = 0; i < num_params_; ++i )
                    {
                        gradients_.push_back( std::make_shared<TensorType>( Device::Cpu(), shape_t{ 1 } ) );
                    }
                }
            }

        private:
            size_t num_params_;
            std::vector<std::shared_ptr<TensorType>> parameters_;
            std::vector<std::shared_ptr<TensorType>> gradients_;
        };

        // ================================================================
        // A leaf that owns a parameter but names none, and whose save_ writes
        // nothing -- the exact shape that produced archives missing most of a
        // model's weights while reporting success. The save traversal must reject
        // it rather than walk past it.
        // ================================================================
        class UnnamedParameterChild : public MockChild
        {
        public:
            explicit UnnamedParameterChild( const std::string& name )
                : MockChild( name, 1 )
            {}

            std::vector<std::string> getParameterNames() const override
            {
                return {};
            }

            void save_( ModelArchive&, SerializationMode ) const override
            {}
        };

        std::filesystem::path makeTempArchivePath( const std::string& tag )
        {
            const auto stamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();

            return std::filesystem::temp_directory_path()
                / std::format( "mila_test_composite_{}_{}.mila", tag, stamp );
        }

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

            // save_ is public on Component but protected on CompositeComponent, so a test
            // cannot reach it through a Composite object without a forwarder.
            void exposeSave( ModelArchive& archive, SerializationMode mode ) const
            {
                this->save_( archive, mode );
            }

            void exposeLoad( ModelArchive& archive, SerializationMode mode )
            {
                this->load_( archive, mode );
            }

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
        EXPECT_EQ( child->parameterCount(), 2 );
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

        EXPECT_EQ( composite->parameterCount(), 5 );
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

    TEST_F( CompositeComponentTests, GetGradients_EmptyWhenBuiltForInference )
    {
        auto composite = contextual();
        composite->addComponent( std::make_shared<MockChild>( "a", 2 ) );
        composite->build( build( RuntimeMode::Inference ) );

        EXPECT_TRUE( composite->getGradients().empty() );
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

    // ====================================================================
    // Serialization: archive layout
    // ====================================================================
    //
    // The container contract these pin is the one every transformer depends on:
    // a child's state must land under a scope of its own. Without that, the leaf
    // path names ("meta.json", "tensors/<name>/data.bin") are identical for every
    // sibling and each write overwrites the last -- so a 48-block model would
    // serialize to a single block's worth of tensors and report success.

    TEST_F( CompositeComponentTests, Save_NestsChildScopesSoSiblingsDoNotCollide )
    {
        const auto path = makeTempArchivePath( "scopes" );
        std::error_code ec;
        std::filesystem::remove( path, ec );

        auto composite = contextual( "root" );
        auto inner = std::make_shared<Composite>( "inner" );
        inner->addComponent( std::make_shared<MockChild>( "leaf", 1 ) );

        composite->addComponent( std::make_shared<MockChild>( "a", 1 ) );
        composite->addComponent( std::make_shared<MockChild>( "b", 1 ) );
        composite->addComponent( inner );
        composite->build( build( RuntimeMode::Inference ) );

        {
            ModelArchive archive( path.string(), std::make_unique<ZipSerializer>(), OpenMode::Write );
            composite->exposeSave( archive, SerializationMode::Checkpoint );
        }

        ModelArchive reader( path.string(), std::make_unique<ZipSerializer>(), OpenMode::Read );

        // Siblings at the same level.
        EXPECT_TRUE( reader.hasFile( "a/tensors/p0/data.bin" ) );
        EXPECT_TRUE( reader.hasFile( "b/tensors/p0/data.bin" ) );

        // And a grandchild, so nesting is proven to compose rather than flatten.
        EXPECT_TRUE( reader.hasFile( "inner/leaf/tensors/p0/data.bin" ) );

        // Three distinct parameter blobs, not one overwritten three times.
        const auto files = reader.listFiles();
        const auto blob_count = std::count_if( files.begin(), files.end(),
            []( const std::string& name )
            {
                return name.ends_with( "/data.bin" );
            } );

        EXPECT_EQ( blob_count, 3 );

        std::filesystem::remove( path, ec );
    }

    TEST_F( CompositeComponentTests, Save_WritesCompositeMetadataUnderItsOwnScope )
    {
        const auto path = makeTempArchivePath( "meta" );
        std::error_code ec;
        std::filesystem::remove( path, ec );

        auto composite = contextual( "root" );
        composite->addComponent( std::make_shared<MockChild>( "a", 1 ) );
        composite->build( build( RuntimeMode::Inference ) );

        {
            ModelArchive archive( path.string(), std::make_unique<ZipSerializer>(), OpenMode::Write );
            composite->exposeSave( archive, SerializationMode::Checkpoint );
        }

        ModelArchive reader( path.string(), std::make_unique<ZipSerializer>(), OpenMode::Read );

        EXPECT_TRUE( reader.hasFile( "meta.json" ) );

        // addMetadata() writes the unscoped archive-global path "metadata/<key>", so
        // every composite in a model would overwrite the same keys. Nothing may land
        // there.
        EXPECT_FALSE( reader.hasFile( "metadata/type" ) );
        EXPECT_FALSE( reader.hasFile( "metadata/child_names" ) );

        std::filesystem::remove( path, ec );
    }

    TEST_F( CompositeComponentTests, Save_RecordsChildNamesInRegistrationOrder )
    {
        const auto path = makeTempArchivePath( "order" );
        std::error_code ec;
        std::filesystem::remove( path, ec );

        // Names deliberately not in alphabetical order: an unordered or sorted
        // container would not reproduce this sequence, and a non-reproducible archive
        // is a non-comparable one.
        auto composite = contextual( "root" );
        composite->addComponent( std::make_shared<MockChild>( "zulu", 1 ) );
        composite->addComponent( std::make_shared<MockChild>( "alpha", 1 ) );
        composite->addComponent( std::make_shared<MockChild>( "mike", 1 ) );
        composite->build( build( RuntimeMode::Inference ) );

        {
            ModelArchive archive( path.string(), std::make_unique<ZipSerializer>(), OpenMode::Write );
            composite->exposeSave( archive, SerializationMode::Checkpoint );
        }

        ModelArchive reader( path.string(), std::make_unique<ZipSerializer>(), OpenMode::Read );
        const SerializationMetadata meta = reader.readMetadata( "meta.json" );

        const std::vector<std::string> expected{ "zulu", "alpha", "mike" };

        EXPECT_EQ( meta.getStringVector( "child_names" ), expected );
        EXPECT_EQ( meta.getInt( "child_count" ), 3 );

        std::filesystem::remove( path, ec );
    }

    TEST_F( CompositeComponentTests, Save_ThrowsWhenAChildOwnsUnnamedParameters )
    {
        const auto path = makeTempArchivePath( "guard" );
        std::error_code ec;
        std::filesystem::remove( path, ec );

        auto composite = contextual( "root" );
        composite->addComponent( std::make_shared<MockChild>( "named", 1 ) );
        composite->addComponent( std::make_shared<UnnamedParameterChild>( "orphan" ) );
        composite->build( build( RuntimeMode::Inference ) );

        ModelArchive archive( path.string(), std::make_unique<ZipSerializer>(), OpenMode::Write );

        EXPECT_THROW( composite->exposeSave( archive, SerializationMode::Checkpoint ), std::runtime_error );

        std::filesystem::remove( path, ec );
    }

    // ====================================================================
    // Serialization: round trip
    // ====================================================================
    //
    // The oracle the save side never had. Everything above proves bytes land at the
    // right paths; only this proves they come back -- and it is the first thing that
    // exercises getParameterNames() as the SHARED vocabulary rather than as two
    // independent lists that happen to agree today.

    TEST_F( CompositeComponentTests, SaveThenLoad_RestoresEveryParameterExactly )
    {
        const auto path = makeTempArchivePath( "roundtrip" );
        std::error_code ec;
        std::filesystem::remove( path, ec );

        auto source = contextual( "root" );
        auto inner = std::make_shared<Composite>( "inner" );
        auto leaf = std::make_shared<MockChild>( "leaf", 2 );
        inner->addComponent( leaf );

        auto a = std::make_shared<MockChild>( "a", 1 );
        source->addComponent( a );
        source->addComponent( inner );
        source->build( build( RuntimeMode::Inference ) );

        // Distinct values per tensor so a cross-wired restore is visible, not masked by
        // every parameter holding the same number.
        float next_value = 1.0f;

        for ( auto* parameter : source->getParameters() )
        {
            fill( *static_cast<MockChild::TensorType*>( parameter ), next_value );
            next_value += 1.0f;
        }

        {
            ModelArchive archive( path.string(), std::make_unique<ZipSerializer>(), OpenMode::Write );
            source->exposeSave( archive, SerializationMode::Checkpoint );
        }

        // A separately constructed graph of the same shape, deliberately initialised to a
        // value none of the saved tensors hold, so "restored" cannot be confused with
        // "already correct".
        auto target = contextual( "root" );
        auto target_inner = std::make_shared<Composite>( "inner" );
        target_inner->addComponent( std::make_shared<MockChild>( "leaf", 2 ) );
        target->addComponent( std::make_shared<MockChild>( "a", 1 ) );
        target->addComponent( target_inner );
        target->build( build( RuntimeMode::Inference ) );

        for ( auto* parameter : target->getParameters() )
        {
            fill( *static_cast<MockChild::TensorType*>( parameter ), -1.0f );
        }

        {
            ModelArchive archive( path.string(), std::make_unique<ZipSerializer>(), OpenMode::Read );
            target->exposeLoad( archive, SerializationMode::Checkpoint );
        }

        const auto source_parameters = source->getParameters();
        const auto target_parameters = target->getParameters();

        ASSERT_EQ( source_parameters.size(), target_parameters.size() );
        ASSERT_EQ( source_parameters.size(), 3u );

        for ( size_t i = 0; i < source_parameters.size(); ++i )
        {
            const auto* expected = static_cast<const MockChild::TensorType*>( source_parameters[ i ] );
            const auto* actual = static_cast<const MockChild::TensorType*>( target_parameters[ i ] );

            ASSERT_EQ( expected->size(), actual->size() ) << "parameter " << i;

            const float* expected_data = static_cast<const float*>( expected->rawData() );
            const float* actual_data = static_cast<const float*>( actual->rawData() );

            for ( dim_t element = 0; element < expected->size(); ++element )
            {
                EXPECT_EQ( expected_data[ element ], actual_data[ element ] )
                    << "parameter " << i << " element " << element;
            }
        }

        std::filesystem::remove( path, ec );
    }

    TEST_F( CompositeComponentTests, Load_ThrowsWhenTheArchiveIsMissingAParameter )
    {
        const auto path = makeTempArchivePath( "missing" );
        std::error_code ec;
        std::filesystem::remove( path, ec );

        // Saved with one parameter per child.
        auto source = contextual( "root" );
        source->addComponent( std::make_shared<MockChild>( "a", 1 ) );
        source->build( build( RuntimeMode::Inference ) );

        {
            ModelArchive archive( path.string(), std::make_unique<ZipSerializer>(), OpenMode::Write );
            source->exposeSave( archive, SerializationMode::Checkpoint );
        }

        // Restored into a child expecting two. Skipping the absent one would leave it at
        // its initialised value and report success -- the failure mode the load path
        // exists to make loud.
        auto target = contextual( "root" );
        target->addComponent( std::make_shared<MockChild>( "a", 2 ) );
        target->build( build( RuntimeMode::Inference ) );

        ModelArchive archive( path.string(), std::make_unique<ZipSerializer>(), OpenMode::Read );

        EXPECT_THROW( target->exposeLoad( archive, SerializationMode::Checkpoint ), std::runtime_error );

        std::filesystem::remove( path, ec );
    }

    TEST_F( CompositeComponentTests, Save_SucceedsWhenEveryChildNamesItsParameters )
    {
        const auto path = makeTempArchivePath( "guard-pass" );
        std::error_code ec;
        std::filesystem::remove( path, ec );

        auto composite = contextual( "root" );
        composite->addComponent( std::make_shared<MockChild>( "a", 2 ) );
        composite->build( build( RuntimeMode::Inference ) );

        ModelArchive archive( path.string(), std::make_unique<ZipSerializer>(), OpenMode::Write );

        EXPECT_NO_THROW( composite->exposeSave( archive, SerializationMode::Checkpoint ) );

        std::filesystem::remove( path, ec );
    }
}
