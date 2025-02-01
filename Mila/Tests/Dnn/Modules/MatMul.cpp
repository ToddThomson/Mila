#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>

import Mila;

namespace Dnn::Modules::Tests
{
    using namespace Mila::Dnn::Modules;

    class MatMulTests : public ::testing::Test {
    protected:
        void SetUp() override {
			batch_size_ = 64;
			sequence_length_ = 1024;
			channels_ = 768;
			output_channels_ = 3 * channels_;

            matmul = std::make_shared<MatMul<float>>( "MatMulTest", batch_size_, sequence_length_, channels_, output_channels_, true );
        }

        std::shared_ptr<MatMul<float>> matmul;
        size_t batch_size_{ 0 };
        size_t sequence_length_{ 0 };
        size_t channels_{ 0 };
        size_t output_channels_{ 0 };
    };

    TEST_F( MatMulTests, TestName ) {
        EXPECT_EQ( matmul->name(), "MatMulTest" );
    }

    TEST_F( MatMulTests, TestParameters ) {
        EXPECT_EQ( matmul->parameters(), 2 * output_channels_ );
    }

    TEST_F( MatMulTests, TestWeightInitialization ) {
        auto& weight = matmul->Weight();
        EXPECT_EQ( weight.size(), output_channels_ * channels_ );
    }

    TEST_F( MatMulTests, TestBiasInitialization ) {
        auto& bias = matmul->Bias();
        EXPECT_EQ( bias.size(), output_channels_ );
    }

    TEST_F( MatMulTests, TestForward ) {
        Mila::Dnn::Tensor<float> input( { batch_size_, sequence_length_, channels_ } );
        auto output = matmul->forward( std::make_shared<Mila::Dnn::Tensor<float>>( input ) );
        EXPECT_EQ( output->size(), batch_size_ * sequence_length_ * output_channels_ );
    }

    TEST_F( MatMulTests, TestPrint ) {
        testing::internal::CaptureStdout();
        matmul->print();
        std::string output = testing::internal::GetCapturedStdout();
        EXPECT_NE( output.find( "Module: MatMulTest" ), std::string::npos );
        EXPECT_NE( output.find( "Parameters: " ), std::string::npos );
    }
}