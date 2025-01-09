// Description: Unit tests for the Tensor class.
#include <gtest/gtest.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#ifdef USE_OMP
#include <omp.h>
#endif

import Mila;

namespace Dnn::Tensors::Tests {

    using namespace Mila::Dnn;
    
    class TensorTest : public testing::Test {

    protected:
        TensorTest() {
        }
    };

    TEST( TensorTest, Constructor ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<int> tensor( shape );
        EXPECT_EQ( tensor.shape(), shape );
        EXPECT_EQ( tensor.size(), 6 );
    }

    TEST( TensorTest, ViewReturnsCorrectMatrixMdspan ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float> tensor( shape );
        tensor.fill( 3.14f );

        auto matrix_view = tensor.as_matrix( { 3,2 } );

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

    TEST( TensorTest, Fill ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float> tensor( shape );
        tensor.fill( 3.14f );
        for (size_t i = 0; i < tensor.size(); ++i) {
            EXPECT_EQ( tensor.data()[i], 3.14f );
        }
    }

    /*TEST( TensorTest, Access ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor tensor( shape );
        tensor.fill( 1.0f );
        std::vector<size_t> indices = { 1, 2 };
        EXPECT_EQ( tensor( indices ), 1.0f );
    }*/

    TEST( TensorTest, Strides ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<double> tensor( shape );
        std::vector<size_t> expected_strides = { 3, 1 };
        EXPECT_EQ( tensor.strides(), expected_strides );
    }

    TEST( TensorTest, Print ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float> tensor( shape );
        tensor.fill( 2.0f );
        testing::internal::CaptureStdout();
        tensor.print();
        std::string output = testing::internal::GetCapturedStdout();
        std::string expected_output = "Tensor of shape: 2 3 \nTensorType::kFP32\nData:\n[ 2 2 2 ]\n[ 2 2 2 ]\n";
        EXPECT_EQ( output, expected_output );
    }

    TEST( TensorTest, ParallelFill ) {
        Tensor<float> tensor( { 1000,1000 } );
        tensor.fill( 42.0f );
        for ( size_t i = 0; i < tensor.size(); i++ ) {
            EXPECT_EQ( tensor.data()[ i ], 42 );
        }
    }
}