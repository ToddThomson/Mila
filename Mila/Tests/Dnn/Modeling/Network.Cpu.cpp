#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <vector>
#include <stdexcept>
#include <utility>

import Mila;

namespace Dnn::NetworkTests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    // A small concrete Network subclass exposing the protected constructor for tests.
    template<DeviceType TDeviceType>
    class NetworkUnderTest : public Network<TDeviceType>
    {
    public:
        using Base = Network<TDeviceType>;
        explicit NetworkUnderTest( std::shared_ptr<ExecutionContext<TDeviceType>> ctx, const std::string& name )
            : Base( ctx, name )
        {
        }

        // Expose buildImpl publicly so we can call Composite behavior if needed (delegates to base).
        //using Base::buildImpl;
    };

    // Minimal test module implementing required Module<T> interface used by CompositeModule.
    class SimpleTestModule : public Module<DeviceType::Cpu>
    {
    public:
        SimpleTestModule( std::shared_ptr<ExecutionContext<DeviceType::Cpu>> ctx,
            std::string name,
            size_t param_count = 0,
            bool built = false )
            : ctx_( std::move( ctx ) ), name_( std::move( name ) ), param_count_( param_count ), is_built_( built )
        {
            if (!ctx_) throw std::invalid_argument( "context required" );
        }

        // Computational interface (not exercised in many of these tests)
        void forward( const ITensor& /*input*/, ITensor& /*output*/ )
        {
        }
        void backward( const ITensor& /*input*/, const ITensor& /*output_grad*/, ITensor& /*input_grad*/ )
        {
        }

        void synchronize() override
        {
            ctx_->synchronize();
        }

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
        {
        }

        std::string toString() const override
        {
            return std::string( "SimpleTestModule: " ) + name_;
        }

        // Build lifecycle
        void build( const shape_t& /*input_shape*/ ) override
        {
            is_built_ = true;
        }

        bool isBuilt() const override
        {
            return is_built_;
        }

        // Introspection
        std::string getName() const override
        {
            return name_;
        }

        std::shared_ptr<ComputeDevice> getDevice() const override
        {
            return ctx_->getDevice();
        }

        // Allow tests to mutate built flag
        void setBuilt( bool v )
        {
            is_built_ = v;
        }

    private:
        std::shared_ptr<ExecutionContext<DeviceType::Cpu>> ctx_;
        std::string name_;
        size_t param_count_;
        bool is_built_;
    };

    class NetworkCpuTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            exec_ctx_ = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
        }

        std::shared_ptr<ExecutionContext<DeviceType::Cpu>> exec_ctx_;
    };

    TEST_F( NetworkCpuTests, ConstructAndBasicAccessors )
    {
        NetworkUnderTest<DeviceType::Cpu> net( exec_ctx_, "test_net" );
        EXPECT_EQ( net.getName(), "test_net" );

        auto device = net.getDevice();
        ASSERT_NE( device, nullptr );
        EXPECT_EQ( device->getDeviceType(), DeviceType::Cpu );

        // toString contains name
        std::string info = net.toString();
        EXPECT_NE( info.find( "test_net" ), std::string::npos );

        // synchronize should be a no-op (not throw)
        EXPECT_NO_THROW( net.synchronize() );
    }

    TEST_F( NetworkCpuTests, AddGetHasRemoveReplaceModules )
    {
        NetworkUnderTest<DeviceType::Cpu> net( exec_ctx_, "container" );

        auto m1 = std::make_shared<SimpleTestModule>( exec_ctx_, "child1", 5 );
        auto m2 = std::make_shared<SimpleTestModule>( exec_ctx_, "child2", 7 );

        // add with explicit name
        net.addModule( "first", m1 );
        EXPECT_TRUE( net.hasModule( "first" ) );
        EXPECT_EQ( net.getNamedModules().size(), 1u );

        // add unnamed (auto-generated) module
        net.addModule( m2 );
        EXPECT_GE( net.getModules().size(), 2u );

        // getModule returns correct pointer
        auto g = net.getModule( "first" );
        EXPECT_EQ( g, m1 );

        // replaceModule returns false for missing, true when present
        auto m3 = std::make_shared<SimpleTestModule>( exec_ctx_, "replacement", 3 );
        EXPECT_FALSE( net.replaceModule( "missing", m3 ) );
        EXPECT_TRUE( net.replaceModule( "first", m3 ) );
        EXPECT_EQ( net.getModule( "first" ), m3 );

        // removeModule on non-existing false, on existing true
        EXPECT_FALSE( net.removeModule( "does_not_exist" ) );
        EXPECT_TRUE( net.removeModule( "first" ) );
        EXPECT_FALSE( net.hasModule( "first" ) );
    }

    TEST_F( NetworkCpuTests, AddInvalidModuleCases )
    {
        NetworkUnderTest<DeviceType::Cpu> net( exec_ctx_, "invalids" );

        // empty name invalid
        auto m = std::make_shared<SimpleTestModule>( exec_ctx_, "x" );
        EXPECT_THROW( net.addModule( "", m ), std::invalid_argument );

        // null module invalid
        EXPECT_THROW( net.addModule( "n", nullptr ), std::invalid_argument );

        // duplicate name invalid
        net.addModule( "dup", m );
        EXPECT_THROW( net.addModule( "dup", m ), std::invalid_argument );
    }

    TEST_F( NetworkCpuTests, BuildAndIsBuiltAndParameterCount )
    {
        NetworkUnderTest<DeviceType::Cpu> net( exec_ctx_, "builder" );

        auto a = std::make_shared<SimpleTestModule>( exec_ctx_, "a", 10, false );
        auto b = std::make_shared<SimpleTestModule>( exec_ctx_, "b", 20, false );

        net.addModule( "a", a );
        net.addModule( "b", b );

        // before build, isBuilt returns false
        EXPECT_FALSE( net.isBuilt() );

        // build should propagate to children
        shape_t shape = { 2, 3 };
        net.build( shape );

        EXPECT_TRUE( net.isBuilt() );
        EXPECT_TRUE( a->isBuilt() );
        EXPECT_TRUE( b->isBuilt() );

        // parameterCount aggregates children
        EXPECT_EQ( net.parameterCount(), 30u );
    }

    TEST_F( NetworkCpuTests, ParameterAccessPreconditions )
    {
        NetworkUnderTest<DeviceType::Cpu> net( exec_ctx_, "preconds" );
        auto a = std::make_shared<SimpleTestModule>( exec_ctx_, "a", 1, false );
        net.addModule( "a", a );

        // parameterCount throws if not built
        EXPECT_THROW( net.parameterCount(), std::runtime_error );

        // getParameters throws if not built
        EXPECT_THROW( net.getParameters(), std::runtime_error );

        // build then getParameters works (returns empty vector because SimpleTestModule returns none)
        net.build( { 1 } );
        EXPECT_NO_THROW( net.getParameters() );
    }

    TEST_F( NetworkCpuTests, GetGradientsRequiresTrainingAndBuilt )
    {
        NetworkUnderTest<DeviceType::Cpu> net( exec_ctx_, "grads" );
        auto a = std::make_shared<SimpleTestModule>( exec_ctx_, "a", 0, false );
        net.addModule( "a", a );

        // not built -> getGradients throws
        EXPECT_THROW( net.getGradients(), std::runtime_error );

        net.build( { 1 } );

        // not training -> getGradients throws
        EXPECT_FALSE( net.isTraining() );
        EXPECT_THROW( net.getGradients(), std::runtime_error );

        // enable training and getGradients should succeed (returns empty vector here)
        net.setTraining( true );
        EXPECT_NO_THROW( net.getGradients() );
    }
}