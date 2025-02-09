#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include <memory>

import Mila;

namespace Dnn::Models::Tests
{
    using namespace Mila::Dnn;

    class CpuDummyModule : public Module<float, Compute::CpuMemoryResource> {
    public:
        CpuDummyModule( const std::string& name ) : name_( name ) {}

        std::shared_ptr<HostTensor<float>> forward( const std::shared_ptr<HostTensor<float>> input ) override {
            return input;
        }

        size_t parameters() const override { return 0; }
        void print() const override {}
        std::string name() const override { return name_; }

    private:
        bool is_training_{ false };
        std::string name_;
    };

    template<typename T>
    class CpuTestModel : public Model<T, Compute::CpuMemoryResource> {
    public:
        std::string name() const override {
            return "TestModel";
        }
        void print() const override {}
        size_t parameters() const override { return 0; }
    };

    class ModelTests : public ::testing::Test {
    protected:
        void SetUp() override {
            // Setup code if needed
        }

        void TearDown() override {
            // Cleanup code if needed
        }
    };

    TEST( ModelTests, CpuModel_AddModule ) {
        auto model = CpuTestModel<float>();
        auto module = std::make_shared<CpuDummyModule>( "module1" );
        size_t index = model.add( module );
        EXPECT_EQ( index, 0 );
        EXPECT_EQ( model.size(), 1 );
    }

    TEST( ModelTests, CpuModel_AddDuplicateNamedModule ) {
        auto model = CpuTestModel<float>();
        auto module = std::make_shared < CpuDummyModule>( "module1" );
        auto module2 = std::make_shared < CpuDummyModule>( "module1" );
        model.add( module );
        EXPECT_THROW( model.add( module2 ), std::invalid_argument );
    }

    TEST( ModelTests, CpuModel_ForwardPass ) {
        auto model = CpuTestModel<float>();
        auto module = std::make_shared<CpuDummyModule>( "module1" );
        model.add( module );
        model.build();
        auto input = std::make_shared<HostTensor<float>>();
        auto output = model.forward( input );
        EXPECT_EQ( input, output );
    }

    TEST( ModelTests, CpuModel_ForwardPassWithoutBuild ) {
        auto model = CpuTestModel<float>();
        auto module = std::make_shared<CpuDummyModule>( "module1" );
        model.add( module );
        auto input = std::make_shared<HostTensor<float>>();
        EXPECT_THROW( model.forward( input ), std::runtime_error );
    }

    TEST( ModelTests, CpuModel_BuildModel ) {
        auto model = CpuTestModel<float>();
        auto module = std::make_shared<CpuDummyModule>( "module1" );
        model.add( module );
        model.build();
        EXPECT_THROW( model.build(), std::runtime_error );
    }

    TEST( ModelTests, AccessModuleByIndex ) {
        auto model = CpuTestModel<float>();
        auto module = std::make_shared<CpuDummyModule>( "module1" );
        model.add( module );
        EXPECT_EQ( model[ 0 ], module );
        EXPECT_THROW( model[ 1 ], std::out_of_range );
    }

    TEST( ModelTests, AccessModuleByName ) {
        auto model = CpuTestModel<float>();
        auto module = std::make_shared<CpuDummyModule>( "module1" );
        model.add( module );
        EXPECT_EQ( model[ "module1" ], module );
        EXPECT_THROW( model[ "module2" ], std::out_of_range );
    }

    TEST( ModelTests, ParametersCount ) {
        auto model = CpuTestModel<float>();
        auto module = std::make_shared<CpuDummyModule>( "module1" );
        model.add( module );
        EXPECT_EQ( model.parameters(), 0 );
    }

    /* FIXME: This test is not working
    TEST( ModelTests, PrintModel ) {
        auto model = TestModel<float>();
        auto module = std::make_shared<DummyModule>( "module1" );
        model.add( module );
        testing::internal::CaptureStdout();
        model.print();
        std::string output = testing::internal::GetCapturedStdout();
        EXPECT_NE( output.find( "Modules:" ), std::string::npos );
        EXPECT_NE( output.find( "Total parameters: 0" ), std::string::npos );
    }*/

    TEST( ModelTests, Constructor_ShouldInitializeCorrectly ) {
        auto model = CpuTestModel<float>();
        EXPECT_EQ( model.name(), "TestModel" );
    }

    /*TEST( ModelTests, AddModule_ShouldAddModuleCorrectly ) {
        auto model = CpuTestModel<float>();
        auto input_shape = std::vector<size_t>{ 2, 3, 4 };
        auto module = std::make_shared<Modules::LayerNorm<float, Compute::CpuMemoryResource>>( "ln1", input_shape );

        model.add( module );

        EXPECT_EQ( model.size(), 1 );
    }*/

    //TEST( ModelTests, Forward_ShouldReturnCorrectOutput ) {
    //    auto model = CpuTestModel<float>();
    //    auto input_shape = std::vector<size_t>{ 2, 3, 4 };
    //    auto ln1 = std::make_shared<Modules::LayerNorm<float, Compute::CpuMemoryResource>>( "ln1", input_shape );
    //    model.add( ln1 );

    //    // Create a random input tensor with shape (B=2, T=3, C=4)
    //    HostTensor<float> X( input_shape );
    //    random<float, Compute::CpuMemoryResource>( X, -1.0f, 1.0f );

    //    model.build();

    //    auto Y = model.forward( std::make_shared<HostTensor<float>>( X ) );

    //    EXPECT_EQ( Y->size(), X.size() );
    //}
}