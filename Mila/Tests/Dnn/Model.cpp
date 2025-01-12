#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include <memory>

import Mila;

import Compute.Cpu.Ops.layernorm;

namespace Dnn::Tensors::Tests
{
    using namespace Mila::Dnn;

	template<typename T>
    class MockModel : public Model<T> {
    public:
        //MockModel() = default;
           
        std::string name() override {
            return "Mock";
        }
        
        void print() override {}
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
        MockModel<float> mock = MockModel<float>();
        EXPECT_EQ(mock.name(), "Mock");
    }

    TEST_F(ModelTests, AddLayer_ShouldAddLayerCorrectly) {
        // Arrange
        auto mock_model = MockModel<float>();
        //auto layer = Mila::Dnn::Compute::Cpu::Ops::LayerNormOp<float>("ln1", 2, 3, 4);
        
        // Act
        mock_model.add( Mila::Dnn::Compute::Cpu::Ops::LayerNormOp<float>( "ln1", 2, 3, 4 ) );

        // Assert
        EXPECT_EQ(mock_model.size(), 1);
    }

    TEST_F(ModelTests, Forward_ShouldReturnCorrectOutput) {
        // Arrange
        MockModel<float> mock_model = MockModel<float>();
        //auto layer = Mila::Dnn::Compute::Cpu::Ops::LayerNormOp<float>("ln1", 2, 3, 4);
        mock_model.add( Mila::Dnn::Compute::Cpu::Ops::LayerNormOp<float>( "ln1", 2, 3, 4 ) );
        Tensor<float> input({2 * 3 * 4});
        random(input, -1.0f, 1.0f);

        // Act
        Tensor<float> output = mock_model.forward(input);

        // Assert
        EXPECT_EQ(output.size(), input.size());
    }

    
}