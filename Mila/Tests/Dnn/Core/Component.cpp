/**
 * @file Component.cpp
 * @brief Base-contract tests for the Component<TDeviceType, TPrecision> template.
 *
 * Reference instance of the Mila base-contract test archetype (see
 * Specifications/Testing.md). Exercises the inherited lifecycle, the
 * execution-context state machine, the runtime/training-mode rules, name
 * validation, and the introspection surface ONCE, through a spy-capable harness
 * subclass. Concrete component tests (e.g. Gelu.Cpu.cpp) inherit this guarantee
 * and assert only their own delta.
 *
 * Device-agnostic base logic is tested on DeviceType::Cpu so it runs under the
 * MILA_ENABLE_CUDA=OFF CI gate. Tests that need a CUDA instantiation (e.g. the
 * device-type mismatch throw) live in the Component.Cuda.cpp companion.
 */

#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>

import Mila;

namespace Mila::Tests::Dnn::Core
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    namespace
    {
        // ================================================================
        // Spy-capable harness
        // ================================================================
        // A minimal Component subclass that implements the pure-virtual surface
        // trivially, records lifecycle-hook invocations, and surfaces the
        // protected members through expose* forwarders so negative paths can be
        // driven without a parent Network/CompositeComponent. Internal linkage
        // (anonymous namespace) keeps it from colliding with other test TUs.
        template<DeviceType TDeviceType, TensorDataType TPrecision = TensorDataType::FP32>
        class HarnessComponent : public Component<TDeviceType, TPrecision>
        {
        public:
            using Base = Component<TDeviceType, TPrecision>;

            explicit HarnessComponent( const std::string& name = "harness" )
                : Base( name )
            {}

            // -- Hook spies ---------------------------------------------
            int onBuilding_calls = 0;
            int onExecutionContextSet_calls = 0;
            int onTrainingModeChanging_calls = 0;
            TrainingMode last_training_mode{ TrainingMode::Normal };
            bool throw_on_execution_context_set = false;

            // -- Protected-surface forwarders (test only) ---------------
            void exposeSetExecutionContext( IExecutionContext* context )
            {
                this->setExecutionContext( context );
            }

            IExecutionContext* exposeGetExecutionContext() const
            {
                return this->getExecutionContext();
            }

            bool exposeHasExecutionContext() const noexcept
            {
                return this->hasExecutionContext();
            }

            // -- Pure-virtual implementations (trivial) -----------------
            void synchronize() override
            {
                this->getExecutionContext()->synchronize();
            }

            dim_t parameterCount() const override
            {
                return 0;
            }

            std::vector<ITensor*> getParameters() const override
            {
                return {};
            }

            std::vector<ITensor*> getGradients() const override
            {
                return {};
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
                std::ostringstream oss;
                oss << "HarnessComponent[" << this->getName()
                    << ", mode=" << static_cast<int>( this->getTrainingMode() ) << "]";

                return oss.str();
            }

        protected:
            void onExecutionContextSet() override
            {
                ++onExecutionContextSet_calls;

                if ( throw_on_execution_context_set )
                {
                    throw std::runtime_error( "harness: forced onExecutionContextSet failure" );
                }
            }

            void onBuilding( const BuildContext& ) override
            {
                ++onBuilding_calls;
            }

            void onTrainingModeChanging( TrainingMode mode ) override
            {
                ++onTrainingModeChanging_calls;
                last_training_mode = mode;
            }
        };

        using Harness = HarnessComponent<DeviceType::Cpu>;

        BuildContext cpuBuild( RuntimeMode mode, shape_t shape = shape_t{ 2, 3 } )
        {
            return BuildContext( std::move( shape ), mode );
        }

        // J. Static traits — compile-time contracts.
        static_assert( Harness::getDeviceType() == DeviceType::Cpu );
        static_assert( Harness::getPrecision() == TensorDataType::FP32 );
    }

    class ComponentTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            exec_context_ = createExecutionContext( Device::Cpu() );
        }

        // A harness already wired to the CPU context, ready to build().
        std::unique_ptr<Harness> contextualHarness( const std::string& name = "harness" )
        {
            auto component = std::make_unique<Harness>( name );
            component->exposeSetExecutionContext( exec_context_.get() );

            return component;
        }

        std::unique_ptr<IExecutionContext> exec_context_;
    };

    // ====================================================================
    // A. Construction & Name Validation
    // ====================================================================

    TEST_F( ComponentTests, Construct_AcceptsValidName )
    {
        Harness component( "valid_name-1.2" );

        EXPECT_EQ( component.getName(), "valid_name-1.2" );
    }

    TEST_F( ComponentTests, Construct_RejectsEmptyName )
    {
        EXPECT_THROW( Harness( "" ), std::invalid_argument );
    }

    TEST_F( ComponentTests, Construct_RejectsLeadingDigit )
    {
        EXPECT_THROW( Harness( "1abc" ), std::invalid_argument );
    }

    TEST_F( ComponentTests, Construct_RejectsLeadingNonAlpha )
    {
        EXPECT_THROW( Harness( "_abc" ), std::invalid_argument );
    }

    TEST_F( ComponentTests, Construct_RejectsIllegalCharacter )
    {
        EXPECT_THROW( Harness( "bad name" ), std::invalid_argument );
    }

    TEST_F( ComponentTests, Construct_RejectsOverlongName )
    {
        EXPECT_THROW( Harness( std::string( 129, 'a' ) ), std::invalid_argument );
    }

    TEST_F( ComponentTests, Construct_AcceptsMaxLengthName )
    {
        EXPECT_NO_THROW( Harness( std::string( 128, 'a' ) ) );
    }

    TEST_F( ComponentTests, Construct_AllowsDotUnderscoreHyphen )
    {
        EXPECT_NO_THROW( Harness( "a.b_c-d" ) );
    }

    // ====================================================================
    // B. Build Lifecycle
    // ====================================================================

    TEST_F( ComponentTests, Build_ThrowsWhenNoExecutionContext )
    {
        Harness component( "no_context" );

        EXPECT_THROW( component.build( cpuBuild( RuntimeMode::Training ) ), std::runtime_error );
    }

    TEST_F( ComponentTests, IsBuilt_FalseBeforeBuild )
    {
        auto component = contextualHarness();

        EXPECT_FALSE( component->isBuilt() );
    }

    TEST_F( ComponentTests, Build_SetsIsBuiltAndInvokesOnBuildingOnce )
    {
        auto component = contextualHarness();

        component->build( cpuBuild( RuntimeMode::Training ) );

        EXPECT_TRUE( component->isBuilt() );
        EXPECT_EQ( component->onBuilding_calls, 1 );
    }

    TEST_F( ComponentTests, Build_ThrowsWhenAlreadyBuilt )
    {
        auto component = contextualHarness();
        component->build( cpuBuild( RuntimeMode::Training ) );

        EXPECT_THROW( component->build( cpuBuild( RuntimeMode::Training ) ), std::runtime_error );
    }

    // ====================================================================
    // C. Execution Context state machine
    // ====================================================================

    TEST_F( ComponentTests, SetExecutionContext_ThrowsOnNull )
    {
        Harness component( "null_context" );

        EXPECT_THROW( component.exposeSetExecutionContext( nullptr ), std::invalid_argument );
    }

    TEST_F( ComponentTests, SetExecutionContext_InvokesHookOnce )
    {
        auto component = contextualHarness();

        EXPECT_EQ( component->onExecutionContextSet_calls, 1 );
        EXPECT_TRUE( component->exposeHasExecutionContext() );
    }

    TEST_F( ComponentTests, SetExecutionContext_ThrowsWhenAlreadySet )
    {
        auto component = contextualHarness();

        EXPECT_THROW( component->exposeSetExecutionContext( exec_context_.get() ), std::runtime_error );
    }

    TEST_F( ComponentTests, SetExecutionContext_RestoresNullOnHookThrow )
    {
        Harness component( "hook_throws" );
        component.throw_on_execution_context_set = true;

        EXPECT_THROW( component.exposeSetExecutionContext( exec_context_.get() ), std::runtime_error );
        EXPECT_FALSE( component.exposeHasExecutionContext() );
    }

    TEST_F( ComponentTests, GetExecutionContext_ThrowsWhenUnset )
    {
        Harness component( "unset" );

        EXPECT_THROW( component.exposeGetExecutionContext(), std::runtime_error );
    }

    TEST_F( ComponentTests, HasExecutionContext_TracksState )
    {
        Harness component( "tracks" );

        EXPECT_FALSE( component.exposeHasExecutionContext() );

        component.exposeSetExecutionContext( exec_context_.get() );

        EXPECT_TRUE( component.exposeHasExecutionContext() );
    }

    // ====================================================================
    // D. Runtime + Training Mode
    // ====================================================================

    TEST_F( ComponentTests, RuntimeMode_ReflectsTrainingBuild )
    {
        auto component = contextualHarness();
        component->build( cpuBuild( RuntimeMode::Training ) );

        EXPECT_EQ( component->getRuntimeMode(), RuntimeMode::Training );
        EXPECT_TRUE( component->isTrainingMode() );
        EXPECT_FALSE( component->isInferenceMode() );
    }

    TEST_F( ComponentTests, RuntimeMode_ReflectsInferenceBuild )
    {
        auto component = contextualHarness();
        component->build( cpuBuild( RuntimeMode::Inference ) );

        EXPECT_EQ( component->getRuntimeMode(), RuntimeMode::Inference );
        EXPECT_TRUE( component->isInferenceMode() );
        EXPECT_FALSE( component->isTrainingMode() );
    }

    TEST_F( ComponentTests, TrainingMode_DefaultsToNormalOnTrainingBuild )
    {
        auto component = contextualHarness();
        component->build( cpuBuild( RuntimeMode::Training ) );

        EXPECT_EQ( component->getTrainingMode(), TrainingMode::Normal );
    }

    TEST_F( ComponentTests, TrainingMode_ForcedEvalOnInferenceBuild )
    {
        auto component = contextualHarness();
        component->build( cpuBuild( RuntimeMode::Inference ) );

        EXPECT_EQ( component->getTrainingMode(), TrainingMode::Eval );
    }

    TEST_F( ComponentTests, SetTrainingMode_ThrowsBeforeBuild )
    {
        auto component = contextualHarness();

        EXPECT_THROW( component->setTrainingMode( TrainingMode::Eval ), std::runtime_error );
    }

    TEST_F( ComponentTests, SetTrainingMode_ThrowsOnInferenceBuilt )
    {
        auto component = contextualHarness();
        component->build( cpuBuild( RuntimeMode::Inference ) );

        EXPECT_THROW( component->setTrainingMode( TrainingMode::Eval ), std::runtime_error );
    }

    TEST_F( ComponentTests, SetTrainingMode_NormalToEvalInvokesHookOnce )
    {
        auto component = contextualHarness();
        component->build( cpuBuild( RuntimeMode::Training ) );

        component->setTrainingMode( TrainingMode::Eval );

        EXPECT_EQ( component->onTrainingModeChanging_calls, 1 );
        EXPECT_EQ( component->last_training_mode, TrainingMode::Eval );
        EXPECT_EQ( component->getTrainingMode(), TrainingMode::Eval );
    }

    TEST_F( ComponentTests, SetTrainingMode_SameModeIsNoOp )
    {
        auto component = contextualHarness();
        component->build( cpuBuild( RuntimeMode::Training ) );

        // Default is Normal; setting Normal again must not invoke the hook.
        component->setTrainingMode( TrainingMode::Normal );

        EXPECT_EQ( component->onTrainingModeChanging_calls, 0 );
    }

    TEST_F( ComponentTests, SetTrainingMode_EvalBackToNormal )
    {
        auto component = contextualHarness();
        component->build( cpuBuild( RuntimeMode::Training ) );

        component->setTrainingMode( TrainingMode::Eval );
        component->setTrainingMode( TrainingMode::Normal );

        EXPECT_EQ( component->onTrainingModeChanging_calls, 2 );
        EXPECT_EQ( component->getTrainingMode(), TrainingMode::Normal );
    }

    // ====================================================================
    // G. Parameters & Gradients (base defaults)
    // ====================================================================

    TEST_F( ComponentTests, Parameters_DefaultsAreEmpty )
    {
        auto component = contextualHarness();

        EXPECT_EQ( component->parameterCount(), 0 );
        EXPECT_TRUE( component->getParameters().empty() );
        EXPECT_TRUE( component->getGradients().empty() );
        EXPECT_TRUE( component->getParameterNames().empty() );
    }

    TEST_F( ComponentTests, ZeroGradients_DefaultIsNoOp )
    {
        auto component = contextualHarness();

        EXPECT_NO_THROW( component->zeroGradients() );
    }

    // NOTE: loadParameter() default-throw belongs here but requires an
    // ITensorBlob fixture; it is covered in the serialization-focused pass.

    // ====================================================================
    // I. Diagnostics
    // ====================================================================

    TEST_F( ComponentTests, ToString_NonEmpty )
    {
        auto component = contextualHarness( "diag" );

        EXPECT_FALSE( component->toString().empty() );
    }

    TEST_F( ComponentTests, StreamOperator_UsesToString )
    {
        auto component = contextualHarness( "diag" );

        std::ostringstream oss;
        oss << *component;

        EXPECT_EQ( oss.str(), component->toString() );
    }

    // ====================================================================
    // J. Static traits & type identity
    // ====================================================================

    TEST_F( ComponentTests, StaticTraits_DeviceTypeAndPrecision )
    {
        auto component = contextualHarness();

        EXPECT_EQ( component->getDeviceType(), DeviceType::Cpu );
        EXPECT_EQ( component->getPrecision(), TensorDataType::FP32 );
    }

    TEST_F( ComponentTests, GetType_ReturnsComponentType )
    {
        auto component = contextualHarness();

        EXPECT_EQ( component->getType(), ComponentType::MockComponent );
    }

    TEST_F( ComponentTests, GetDeviceId_ReturnsCpu )
    {
        auto component = contextualHarness();

        EXPECT_EQ( component->getDeviceId().type, DeviceType::Cpu );
    }
}
