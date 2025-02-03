#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include <memory>

import Mila;

namespace Dnn::Models::Tests
{
    using namespace Mila::Dnn;

    class DummyModule : public Module<float> {
    public:
        DummyModule( const std::string& name ) : name_( name ) {}
        std::shared_ptr<Tensor<float>> forward( const std::shared_ptr<Tensor<float>>& input ) override {
            return input; // Dummy forward pass
        }
        void setTrainingMode( bool training ) {}
        size_t parameters() const override { return 0; }
        void print() const override {}
        std::string name() const override { return name_; }
    private:
		bool is_training_{ false };
        std::string name_;
    };

	template<typename T>
    class TestModel : public Model<T> {
    public:
        std::string name() const override {
            return "TestModel";
        }
        void print() const override {}
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

    TEST( ModelTests, AddModule ) {
        auto model = TestModel<float>();
        auto module = std::make_shared<DummyModule>( "module1" );
        size_t index = model.add( module );
        EXPECT_EQ( index, 0 );
        EXPECT_EQ( model.size(), 1 );
    }

    TEST( ModelTests, AddDuplicateModule ) {
        auto model = TestModel<float>();
        auto module = std::make_shared<DummyModule>( "module1" );
        model.add( module );
        EXPECT_THROW( model.add( module ), std::invalid_argument );
    }

    TEST( ModelTests, ForwardPass ) {
        auto model = TestModel<float>();
        auto module = std::make_shared<DummyModule>( "module1" );
        model.add( module );
        model.build();
        auto input = std::make_shared<Tensor<float>>();
        auto output = model.forward( input );
        EXPECT_EQ( input, output );
    }

    TEST( ModelTests, ForwardPassWithoutBuild ) {
        auto model = TestModel<float>();
        auto module = std::make_shared<DummyModule>( "module1" );
        model.add( module );
        auto input = std::make_shared<Tensor<float>>();
        EXPECT_THROW( model.forward( input ), std::runtime_error );
    }

    TEST( ModelTests, BuildModel ) {
        auto model = TestModel<float>();
        auto module = std::make_shared<DummyModule>( "module1" );
        model.add( module );
        model.build();
        EXPECT_THROW( model.build(), std::runtime_error );
    }

    TEST( ModelTests, AccessModuleByIndex ) {
        auto model = TestModel<float>();
        auto module = std::make_shared<DummyModule>( "module1" );
        model.add( module );
        EXPECT_EQ( model[ 0 ], module );
        EXPECT_THROW( model[ 1 ], std::out_of_range );
    }

    TEST( ModelTests, AccessModuleByName ) {
        auto model = TestModel<float>();
        auto module = std::make_shared<DummyModule>( "module1" );
        model.add( module );
        EXPECT_EQ( model[ "module1" ], module );
        EXPECT_THROW( model[ "module2" ], std::out_of_range );
    }

    TEST( ModelTests, ParametersCount ) {
        auto model = TestModel<float>();
        auto module = std::make_shared<DummyModule>( "module1" );
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

    TEST(ModelTests, Constructor_ShouldInitializeCorrectly) {
        TestModel<float> mock = TestModel<float>();
        EXPECT_EQ(mock.name(), "TestModel");
    }

    TEST(ModelTests, AddModule_ShouldAddModuleCorrectly) {
        auto model = TestModel<float>();
        auto layer = std::make_shared<Modules::LayerNorm<float>>( "ln1", 2, 3, 4 );

        model.add( layer );

        EXPECT_EQ(model.size(), 1);
    }

    TEST(ModelTests, Forward_ShouldReturnCorrectOutput) {
        TestModel<float> model = TestModel<float>();
        auto layer = std::make_shared<Modules::LayerNorm<float>>("ln1", 2, 3, 4);
        model.add( layer );
		
        // Create a random input tensor with shape (B=2, T=3, C=4)
        Tensor<float> X({2, 3, 4}); 
        random(X, -1.0f, 1.0f);

        model.build();

        auto Y = model.forward( std::make_shared<Tensor<float>>( X ) );

        EXPECT_EQ( Y->size(), X.size());
    }
}