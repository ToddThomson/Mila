#include <gtest/gtest.h>

import Mila;

namespace Dnn::Tensors::Tests {

    using namespace Mila::Dnn;
    
    class TensorTest : public testing::Test {

    protected:
        TensorTest() {
        }
    };

    TEST( TensorTest, DefaultConstructor ) {
        Tensor<float> tensor;
		auto device = Compute::DeviceContext::instance().getDevice();
        EXPECT_TRUE( tensor.empty() );
        EXPECT_EQ( tensor.size(), 0 );
        EXPECT_EQ( tensor.rank(), 0 );
		//EXPECT_EQ( tensor.device()->getName(), device->getName() );
    }

    TEST( TensorTest, ConstructorWithShape ) {
        auto device = Compute::DeviceContext::instance().getDevice();
		std::vector<size_t> shape = { 2, 3 };
        Tensor<float> tensor( shape );
        EXPECT_FALSE( tensor.empty() );
        EXPECT_EQ( tensor.size(), 6 );
        EXPECT_EQ( tensor.rank(), 2 );
        EXPECT_EQ( tensor.shape(), shape );
        //EXPECT_EQ( tensor.device()->getName(), device->getName() );
    }

    TEST( TensorTest, Reshape ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float> tensor( shape );
        std::vector<size_t> new_shape = { 3, 2 };
        tensor.reshape( new_shape );
        EXPECT_EQ( tensor.shape(), new_shape );
        EXPECT_EQ( tensor.size(), 6 );
    }

    TEST( TensorTest, VectorSpan ) {
        std::vector<size_t> shape = { 6 };
        Tensor<float> tensor( {6});
        auto span = tensor.vectorSpan();
        EXPECT_EQ( span.extent( 0 ), 6 );
    }

    TEST( TensorTest, MatrixSpan ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float> tensor( shape );
        auto span = tensor.matrixSpan( shape );
        EXPECT_EQ( span.extent( 0 ), 2 );
        EXPECT_EQ( span.extent( 1 ), 3 );
    }

    TEST( TensorTest, OperatorIndex1D ) {
        std::vector<size_t> shape = { 6 };
        Tensor<float> tensor( shape );
        tensor[ 0 ] = 1.0f;
        EXPECT_EQ( tensor[ 0 ], 1.0f );
    }

    TEST( TensorTest, OperatorIndex2D ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float> tensor( shape );
        tensor.fill( 3.14f );
        tensor[ 0, 1 ] = 1.0f;
		auto val = tensor[ 0, 1 ];
        EXPECT_EQ( val, 1.0f );
    }

    TEST( TensorTest, Fill ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float> tensor( shape );
        tensor.fill( 3.1415f );
        for ( size_t i = 0; i < tensor.shape()[0]; ++i ) {
            for ( size_t j = 0; j < tensor.shape()[ 1 ]; ++j ) {
                auto val = tensor[ i, j ];
                EXPECT_EQ( val, 3.1415f );
            }
        }
    }

    TEST( TensorTest, Print ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float> tensor( shape );
        tensor.fill( 1.0f );
        testing::internal::CaptureStdout();
        tensor.print();
        std::string output = testing::internal::GetCapturedStdout();
        EXPECT_NE( output.find( "Tensor of shape: 2 3" ), std::string::npos );
        EXPECT_NE( output.find( "1" ), std::string::npos );
    }

    TEST( TensorTest, ViewReturnsCorrectMatrixMdspan ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float> tensor( shape );
        tensor.fill( 3.14f );

        auto matrix_view = tensor.matrixSpan( { 3,2 } );

        ASSERT_EQ( matrix_view.extent( 0 ), shape[ 1] );
        ASSERT_EQ( matrix_view.extent( 1 ), shape[ 0 ] );

        // Fill the tensor with some values and check if the view reflects them
        tensor.fill( 1.0f );

        for ( size_t i = 0; i < shape[ 0 ]; ++i ) {
            for ( size_t j = 0; j < shape[ 1 ]; ++j ) {
//                ASSERT_EQ( matrix_view[ i, j ], 1.0f );
            }
        }
    }
    
    TEST( TensorTest, Strides ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float> tensor( shape );
        std::vector<size_t> expected_strides = { 3, 1 };
        EXPECT_EQ( tensor.strides(), expected_strides );
    }
}