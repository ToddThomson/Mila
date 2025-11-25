#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <vector>
#include <stdexcept>
#include <utility>
#include <filesystem>
#include <chrono>
#include <format>
#include <system_error>

import Mila;

namespace Dnn::NetworkTests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    // A small concrete Network subclass exposing the protected constructor for tests.
    template<DeviceType TDeviceType, TensorDataType TPrecision = TensorDataType::FP32>
    class NetworkUnderTest : public Network<TDeviceType, TPrecision>
    {
    public:
        using Base = Network<TDeviceType, TPrecision>;
        explicit NetworkUnderTest( std::shared_ptr<ExecutionContext<TDeviceType>> ctx, const std::string& name )
            : Base( ctx, name )
        {
        }

        // Expose buildImpl publicly so we can call Composite behavior if needed (delegates to base).
        //using Base::buildImpl;
    };

    // Minimal test component implementing required Component<TDeviceType, TPrecision> interface used by CompositeComponent.
    class SimpleTestComponent : public Component<DeviceType::Cpu, TensorDataType::FP32>
    {
    public:
        SimpleTestComponent( std::shared_ptr<ExecutionContext<DeviceType::Cpu>> ctx,
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
            // Intentionally empty for tests - network-level metadata is the main concern here.
        }

        std::string toString() const override
        {
            return std::string( "SimpleTestComponent: " ) + name_;
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

    // Reuse helper used elsewhere in tests to create unique temporary archive paths.
    static std::filesystem::path makeTempZipPath()
    {
        auto tmp = std::filesystem::temp_directory_path();
        auto ts = std::chrono::high_resolution_clock::now().time_since_epoch().count();

        return tmp / std::format( "mila_test_network_{}.mila", ts );
    }

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

        auto m1 = std::make_shared<SimpleTestComponent>( exec_ctx_, "child1", 5 );
        auto m2 = std::make_shared<SimpleTestComponent>( exec_ctx_, "child2", 7 );

        // add with explicit name
        net.addComponent( "first", m1 );
        EXPECT_TRUE( net.hasComponent( "first" ) );
        EXPECT_EQ( net.getNamedComponents().size(), 1u );

        // add unnamed (auto-generated) module
        net.addComponent( m2 );
        EXPECT_GE( net.getComponents().size(), 2u );

        // getModule returns correct pointer
        auto g = net.getComponent( "first" );
        EXPECT_EQ( g, m1 );

        // replaceModule returns false for missing, true when present
        auto m3 = std::make_shared<SimpleTestComponent>( exec_ctx_, "replacement", 3 );
        EXPECT_FALSE( net.replaceComponent( "missing", m3 ) );
        EXPECT_TRUE( net.replaceComponent( "first", m3 ) );
        EXPECT_EQ( net.getComponent( "first" ), m3 );

        // removeModule on non-existing false, on existing true
        EXPECT_FALSE( net.removeComponent( "does_not_exist" ) );
        EXPECT_TRUE( net.removeComponent( "first" ) );
        EXPECT_FALSE( net.hasComponent( "first" ) );
    }

    TEST_F( NetworkCpuTests, AddInvalidModuleCases )
    {
        NetworkUnderTest<DeviceType::Cpu> net( exec_ctx_, "invalids" );

        // empty name invalid
        auto m = std::make_shared<SimpleTestComponent>( exec_ctx_, "x" );
        EXPECT_THROW( net.addComponent( "", m ), std::invalid_argument );

        // null module invalid
        EXPECT_THROW( net.addComponent( "n", nullptr ), std::invalid_argument );

        // duplicate name invalid
        net.addComponent( "dup", m );
        EXPECT_THROW( net.addComponent( "dup", m ), std::invalid_argument );
    }

    TEST_F( NetworkCpuTests, BuildAndIsBuiltAndParameterCount )
    {
        NetworkUnderTest<DeviceType::Cpu> net( exec_ctx_, "builder" );

        auto a = std::make_shared<SimpleTestComponent>( exec_ctx_, "a", 10, false );
        auto b = std::make_shared<SimpleTestComponent>( exec_ctx_, "b", 20, false );

        net.addComponent( "a", a );
        net.addComponent( "b", b );

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
        auto a = std::make_shared<SimpleTestComponent>( exec_ctx_, "a", 1, false );
        net.addComponent( "a", a );

        // parameterCount throws if not built
        EXPECT_THROW( net.parameterCount(), std::runtime_error );

        // getParameters throws if not built
        EXPECT_THROW( net.getParameters(), std::runtime_error );

        // build then getParameters works (returns empty vector because SimpleTestComponent returns none)
        net.build( { 1 } );
        EXPECT_NO_THROW( net.getParameters() );
    }

    TEST_F( NetworkCpuTests, GetGradientsRequiresTrainingAndBuilt )
    {
        NetworkUnderTest<DeviceType::Cpu> net( exec_ctx_, "grads" );
        auto a = std::make_shared<SimpleTestComponent>( exec_ctx_, "a", 0, false );
        net.addComponent( "a", a );

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

    // New tests: serialization API for Network.save / ModelArchive integration
    TEST_F( NetworkCpuTests, SaveNetworkWritesMetaAndArchitecture )
    {
        auto path = makeTempZipPath();

        // Ensure no prior file
        std::error_code ec;
        std::filesystem::remove( path, ec );

        NetworkUnderTest<DeviceType::Cpu> net( exec_ctx_, "serial_net" );

        auto m1 = std::make_shared<SimpleTestComponent>( exec_ctx_, "mod1", 1 );
        auto m2 = std::make_shared<SimpleTestComponent>( exec_ctx_, "mod2", 2 );

        net.addComponent( "mod1", m1 );
        net.addComponent( "mod2", m2 );

        // Create writer archive using ZipSerializer
        {
            auto writer = std::make_unique<ZipSerializer>();
            ModelArchive archive( path.string(), std::move( writer ), OpenMode::Write );

            // Save network (Architecture mode)
            EXPECT_NO_THROW( net.save( archive, SerializationMode::Architecture ) );

            archive.close();
        }

        // Re-open for read and validate presence/content of metadata files
        {
            auto reader = std::make_unique<ZipSerializer>();
            ModelArchive rar( path.string(), std::move( reader ), OpenMode::Read );

            // network meta.json must exist
            EXPECT_TRUE( rar.hasFile( "network/meta.json" ) );
            auto meta = rar.readJson( "network/meta.json" );

            EXPECT_EQ( meta.at( "name" ).get<std::string>(), "serial_net" );
            EXPECT_EQ( meta.at( "num_modules" ).get<size_t>(), 2u );

            // new meta fields
            EXPECT_TRUE( meta.contains( "format_version" ) );
            EXPECT_EQ( meta.at( "format_version" ).get<int>(), 1 );
            EXPECT_TRUE( meta.contains( "export_time" ) );

            // mode is now a string
            EXPECT_TRUE( meta.at( "mode" ).is_string() );
            EXPECT_EQ( meta.at( "mode" ).get<std::string>(), serializationModeToString( SerializationMode::Architecture ) );

            // architecture.json should exist and list module descriptors
            EXPECT_TRUE( rar.hasFile( "network/architecture.json" ) );
            auto arch = rar.readJson( "network/architecture.json" );
            ASSERT_TRUE( arch.is_array() );
            EXPECT_EQ( arch.size(), 2u );

            // entries are objects with name/path/index (deterministic sorted order)
            EXPECT_TRUE( arch[0].is_object() );
            EXPECT_EQ( arch[0].at( "name" ).get<std::string>(), "mod1" );
            EXPECT_EQ( arch[0].at( "path" ).get<std::string>(), "modules/mod1" );
            EXPECT_EQ( arch[0].at( "index" ).get<int>(), 0 );

            EXPECT_TRUE( arch[1].is_object() );
            EXPECT_EQ( arch[1].at( "name" ).get<std::string>(), "mod2" );
            EXPECT_EQ( arch[1].at( "path" ).get<std::string>(), "modules/mod2" );
            EXPECT_EQ( arch[1].at( "index" ).get<int>(), 1 );

            rar.close();
        }

        // Clean up
        std::filesystem::remove( path, ec );
        EXPECT_FALSE( std::filesystem::exists( path ) );
    }
}