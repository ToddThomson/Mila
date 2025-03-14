#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <sstream>

import Mila;

namespace Tensors::Tests {

    using namespace Mila::Dnn;
    
    class TensorTest : public testing::Test {

    protected:
        TensorTest() {
        }
    };

    TEST( TensorTest, Cpu_DefaultConstructor ) {
        Tensor<float, Compute::CpuMemoryResource> tensor;
		auto device = Compute::DeviceContext::instance().getDevice();
        EXPECT_TRUE( tensor.empty() );
        EXPECT_EQ( tensor.size(), 0 );
        EXPECT_EQ( tensor.rank(), 0 );
    }

    TEST( TensorTest, ConstructorWithShape ) {
        auto device = Compute::DeviceContext::instance().getDevice();
		std::vector<size_t> shape = { 2, 3 };
        Tensor<float,Compute::CudaMemoryResource> tensor( shape );
        EXPECT_FALSE( tensor.empty() );
        EXPECT_EQ( tensor.size(), 6 );
        EXPECT_EQ( tensor.rank(), 2 );
        EXPECT_EQ( tensor.shape(), shape );
    }

    TEST( TensorTest, Cpu_ConstructorWithEmptyShape ) {
        std::vector<size_t> shape = {};
        Tensor<float> tensor( shape );
        EXPECT_TRUE( tensor.empty() );
        EXPECT_EQ( tensor.size(), 0 );
        EXPECT_EQ( tensor.rank(), 0 );
		EXPECT_EQ( tensor.strides().size(), 0 );
        EXPECT_EQ( tensor.shape(), shape );
    }

    TEST( TensorTest, ConstructorWithValidShapeAndValue ) {
        // Test case 1: 1D tensor with shape {5} and value 1
        std::vector<size_t> shape1D = { 5 };
        int value1D = 1;
        Tensor<int> tensor1D( shape1D, value1D );
        EXPECT_EQ( tensor1D.size(), 5 );
        EXPECT_EQ( tensor1D.shape(), shape1D );
        for ( size_t i = 0; i < tensor1D.size(); ++i ) {
            EXPECT_EQ( tensor1D[ i ], value1D );
        }

        // Test case 2: 2D tensor with shape {3, 3} and value 2.5
        std::vector<size_t> shape2D = { 3, 3 };
        float value2D = 2.5f;
        Tensor<float> tensor2D( shape2D, value2D );
        EXPECT_EQ( tensor2D.size(), 9 );
        EXPECT_EQ( tensor2D.shape(), shape2D );
        /*for ( size_t i = 0; i < shape2D[ 0 ]; ++i ) {
            for ( size_t j = 0; j < shape2D[ 1 ]; ++j ) {
                EXPECT_EQ( tensor2D[ i, j ], value2D );
            }
        }*/

        // Test case 3: 3D tensor with shape {2, 2, 2} and value -1
        std::vector<size_t> shape3D = { 2, 2, 2 };
        int value3D = -1;
        Tensor<int> tensor3D( shape3D, value3D );
        EXPECT_EQ( tensor3D.size(), 8 );
        EXPECT_EQ( tensor3D.shape(), shape3D );
        /*for ( size_t i = 0; i < shape3D[ 0 ]; ++i ) {
            for ( size_t j = 0; j < shape3D[ 1 ]; ++j ) {
                for ( size_t k = 0; k < shape3D[ 2 ]; ++k ) {
                    EXPECT_EQ( tensor3D[ i, j, k ], value3D );
                }
            }
        }*/

        // Test case 4: 4D tensor with shape {2, 2, 2, 2} and value 0.5
        std::vector<size_t> shape4D = { 2, 2, 2, 2 };
        double value4D = 0.5f;
        Tensor<float> tensor4D( shape4D, value4D );
        EXPECT_EQ( tensor4D.size(), 16 );
        EXPECT_EQ( tensor4D.shape(), shape4D );
        /*for ( size_t i = 0; i < shape4D[ 0 ]; ++i ) {
            for ( size_t j = 0; j < shape4D[ 1 ]; ++j ) {
                for ( size_t k = 0; k < shape4D[ 2 ]; ++k ) {
                    for ( size_t l = 0; l < shape4D[ 3 ]; ++l ) {
                        EXPECT_EQ( tensor4D[ i, j, k, l ], value4D );
                    }
                }
            }
        }*/
    }

    TEST( TensorTest, Reshape ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float, Compute::CpuMemoryResource> tensor( shape );
        std::vector<size_t> new_shape = { 3, 2 };
        tensor.reshape( new_shape );
        EXPECT_EQ( tensor.shape(), new_shape );
        EXPECT_EQ( tensor.size(), 6 );
    }

    TEST( TensorTest, VectorSpan ) {
        std::vector<size_t> shape = { 6 };
        Tensor<float,Compute::CpuMemoryResource> tensor( {6});
        auto span = tensor.vectorSpan();
        EXPECT_EQ( span.extent( 0 ), 6 );
    }

    TEST( TensorTest, MatrixSpan ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float, Compute::CpuMemoryResource> tensor( shape );
        auto span = tensor.matrixSpan( shape );
        EXPECT_EQ( span.extent( 0 ), 2 );
        EXPECT_EQ( span.extent( 1 ), 3 );
    }

    TEST( TensorTest, OperatorIndex1D ) {
        std::vector<size_t> shape = { 6 };
        Tensor<float,Compute::CpuMemoryResource> tensor( shape );
        tensor[ 0 ] = 1.0f;
        EXPECT_EQ( tensor[ 0 ], 1.0f );
    }

    TEST( TensorTest, OperatorIndex2D ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float, Compute::CpuMemoryResource> tensor( shape );
        tensor.fill( 3.14f );
        tensor[ 0, 1 ] = 1.0f;
		auto val = tensor[ 0, 1 ];
        EXPECT_EQ( val, 1.0f );
    }

    TEST( TensorTest, Fill ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float, Compute::CpuMemoryResource> tensor( shape );
        tensor.fill( 3.1415f );
        for ( size_t i = 0; i < tensor.shape()[0]; ++i ) {
            for ( size_t j = 0; j < tensor.shape()[ 1 ]; ++j ) {
                auto val = tensor[ i, j ];
                EXPECT_EQ( val, 3.1415f );
            }
        }
    }

    TEST( TensorTest, toString ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float, Compute::CpuMemoryResource> tensor( shape );
        tensor.fill( 1.0f );
        std::string output = tensor.toString();
        EXPECT_NE( output.find( "Tensor: " ), std::string::npos );
        EXPECT_NE( output.find( "Shape: (2,3)" ), std::string::npos );
     }

    TEST( TensorTest, ViewReturnsCorrectMatrixMdspan ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float, Compute::CpuMemoryResource> tensor( shape );
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
        Tensor<float, Compute::CpuMemoryResource> tensor( shape );
        std::vector<size_t> expected_strides = { 3, 1 };
        EXPECT_EQ( tensor.strides(), expected_strides );
    }
}