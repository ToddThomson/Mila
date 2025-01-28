#include <gtest/gtest.h>
#include <thrust/host_vector.h>
#include <iostream>
#include <string>
#include <memory>

import Mila;

namespace Dnn::Models::Tests
{
    using namespace Mila::Dnn;

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

    TEST_F(ModelTests, Constructor_ShouldInitializeCorrectly) {
        TestModel<float> mock = TestModel<float>();
        EXPECT_EQ(mock.name(), "TestModel");
    }

    TEST_F(ModelTests, AddLayer_ShouldAddLayerCorrectly) {
        // Arrange
        auto model = TestModel<float>();
        auto layer = std::make_shared<Modules::LayerNorm<float>>( "ln1", 2, 3, 4 );
        
        // Act
        model.add( layer );

        // Assert
        EXPECT_EQ(model.size(), 1);
    }

    TEST_F(ModelTests, Forward_ShouldReturnCorrectOutput) {
        // Arrange
        TestModel<float> model = TestModel<float>();
        auto layer = std::make_shared<Modules::LayerNorm<float>>("ln1", 2, 3, 4);
        model.add( layer );
		
        // Create a random input tensor with shape (B=2, T=3, C=4)
        Tensor<float> X({2, 3, 4}); 
        random(X, -1.0f, 1.0f);

        // Act
        auto Y = model.forward( std::make_shared<Tensor<float>>( X ) );

        // Assert
        EXPECT_EQ( Y->size(), X.size());
    }
}