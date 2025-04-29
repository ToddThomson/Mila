#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <sstream>

import Mila;

namespace Core::Tests{

    using namespace Mila::Dnn;
    
    class TensorTest : public testing::Test {

    protected:
        TensorTest() {
        }
    };

    TEST( TensorTest, Cpu_DefaultConstructor ) {
        Tensor<float, Compute::HostMemoryResource> tensor;
        EXPECT_TRUE( tensor.empty() );
        EXPECT_EQ( tensor.size(), 0 );
        EXPECT_EQ( tensor.rank(), 0 );
    }

    TEST( TensorTest, Cuda_DefaultConstructor ) {
        Tensor<float, Compute::CudaMemoryResource> tensor;
        EXPECT_TRUE( tensor.empty() );
        EXPECT_EQ( tensor.size(), 0 );
        EXPECT_EQ( tensor.rank(), 0 );
    }

    TEST( TensorTest, ConstructorWithShape ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float, Compute::CudaMemoryResource> tensor( shape );
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
        Tensor<int, Compute::HostMemoryResource> tensor1D( shape1D, value1D );
        EXPECT_EQ( tensor1D.size(), 5 );
        EXPECT_EQ( tensor1D.shape(), shape1D );
        for ( size_t i = 0; i < tensor1D.size(); ++i ) {
            EXPECT_EQ( tensor1D[ i ], value1D );
        }

        // Test case 2: 2D tensor with shape {3, 3} and value 2.5
        std::vector<size_t> shape2D = { 3, 3 };
        float value2D = 2.5f;
        Tensor<float, Compute::HostMemoryResource> tensor2D( shape2D, value2D );
        EXPECT_EQ( tensor2D.size(), 9 );
        EXPECT_EQ( tensor2D.shape(), shape2D );
        for ( size_t i = 0; i < shape2D[ 0 ]; ++i ) {
            for ( size_t j = 0; j < shape2D[ 1 ]; ++j ) {
                EXPECT_EQ( ( tensor2D[ i, j ] ), value2D );
            }
        }

        // Test case 3: 3D tensor with shape {2, 2, 2} and value -1
        std::vector<size_t> shape3D = { 2, 2, 2 };
        int value3D = -1;
        Tensor<int, Compute::HostMemoryResource> tensor3D( shape3D, value3D );
        EXPECT_EQ( tensor3D.size(), 8 );
        EXPECT_EQ( tensor3D.shape(), shape3D );
        for ( size_t i = 0; i < shape3D[ 0 ]; ++i ) {
            for ( size_t j = 0; j < shape3D[ 1 ]; ++j ) {
                for ( size_t k = 0; k < shape3D[ 2 ]; ++k ) {
                    EXPECT_EQ( ( tensor3D[ i, j, k ] ), value3D );
                }
            }
        }
    }

    // Memory Resource Tests
    //----------------------------------------------------------------------------------------
    TEST( TensorTest, CpuMemoryResourceProperties ) {
        Tensor<float, Compute::HostMemoryResource> tensor( { 2, 3 } );
        // Use reflection to test properties
        EXPECT_TRUE( (Tensor<float, Compute::HostMemoryResource>::is_host_accessible()) );
        EXPECT_FALSE( (Tensor<float, Compute::HostMemoryResource>::is_device_accessible()) );
    }

    TEST( TensorTest, CudaMemoryResourceProperties ) {
        // Use reflection to test properties
        EXPECT_FALSE( (Tensor<float, Compute::CudaMemoryResource>::is_host_accessible()) );
        EXPECT_TRUE( (Tensor<float, Compute::CudaMemoryResource>::is_device_accessible()) );
    }

    TEST( TensorTest, PinnedMemoryResourceProperties ) {
        // Use reflection to test properties
        EXPECT_TRUE( (Tensor<float, Compute::CudaPinnedMemoryResource>::is_host_accessible()) );
        EXPECT_TRUE( (Tensor<float, Compute::CudaPinnedMemoryResource>::is_device_accessible()) );
    }

    TEST( TensorTest, ManagedMemoryResourceProperties ) {
        // Use reflection to test properties
        EXPECT_TRUE( (Tensor<float, Compute::CudaManagedMemoryResource>::is_host_accessible()) );
        EXPECT_TRUE( (Tensor<float, Compute::CudaManagedMemoryResource>::is_device_accessible()) );
    }

    // Conversion between Memory Types
    //----------------------------------------------------------------------------------------
    TEST( TensorTest, ConvertCpuToCuda ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float, Compute::HostMemoryResource> cpu_tensor( shape, 3.14f );

        // Convert to CUDA
        auto cuda_tensor = cpu_tensor.to<Compute::CudaMemoryResource>();

        EXPECT_EQ( cuda_tensor.shape(), shape );
        EXPECT_EQ( cuda_tensor.size(), 6 );

        // Convert back to CPU for validation
        auto cpu_tensor2 = cuda_tensor.to<Compute::HostMemoryResource>();
        for ( size_t i = 0; i < shape[ 0 ]; ++i ) {
            for ( size_t j = 0; j < shape[ 1 ]; ++j ) {
                EXPECT_FLOAT_EQ( ( cpu_tensor2[ i, j ] ), 3.14f );
            }
        }
    }

    TEST( TensorTest, ConvertCudaToCpu ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float, Compute::CudaMemoryResource> cuda_tensor( shape, 2.71f );

        // Convert to CPU
        auto cpu_tensor = cuda_tensor.to<Compute::HostMemoryResource>();

        EXPECT_EQ( cpu_tensor.shape(), shape );
        EXPECT_EQ( cpu_tensor.size(), 6 );

        // Validate values
        for ( size_t i = 0; i < shape[ 0 ]; ++i ) {
            for ( size_t j = 0; j < shape[ 1 ]; ++j ) {
                EXPECT_FLOAT_EQ( ( cpu_tensor[ i, j ] ), 2.71f );
            }
        }
    }

    TEST( TensorTest, ToHostAccessible ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float, Compute::CudaMemoryResource> cuda_tensor( shape, 1.23f );

        // Convert to host-accessible (should be CPU by default)
        auto host_tensor = cuda_tensor.toHostAccessible();

        EXPECT_EQ( host_tensor.shape(), shape );
        EXPECT_EQ( host_tensor.size(), 6 );

        // Validate we can access the values
        for ( size_t i = 0; i < shape[ 0 ]; ++i ) {
            for ( size_t j = 0; j < shape[ 1 ]; ++j ) {
                EXPECT_FLOAT_EQ( ( host_tensor[ i, j ] ), 1.23f );
            }
        }
    }

    // Shape & Data Manipulation
    //----------------------------------------------------------------------------------------
    TEST( TensorTest, Reshape ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float, Compute::HostMemoryResource> tensor( shape );
        std::vector<size_t> new_shape = { 3, 2 };
        tensor.reshape( new_shape );
        EXPECT_EQ( tensor.shape(), new_shape );
        EXPECT_EQ( tensor.size(), 6 );
    }

    TEST( TensorTest, ReshapeEmptyTensor ) {
        Tensor<float, Compute::HostMemoryResource> tensor;
        std::vector<size_t> new_shape = { 2, 3 };
        tensor.reshape( new_shape );
        EXPECT_EQ( tensor.shape(), new_shape );
        EXPECT_EQ( tensor.size(), 6 );
        EXPECT_FALSE( tensor.empty() );
    }

    TEST( TensorTest, ReshapeInvalidSize ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float, Compute::HostMemoryResource> tensor( shape );
        std::vector<size_t> new_shape = { 4, 4 }; // Different total size
        EXPECT_THROW( tensor.reshape( new_shape ), std::runtime_error );
    }

    TEST( TensorTest, Fill ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float, Compute::HostMemoryResource> tensor( shape );
        tensor.fill( 3.1415f );
        for ( size_t i = 0; i < tensor.shape()[ 0 ]; ++i ) {
            for ( size_t j = 0; j < tensor.shape()[ 1 ]; ++j ) {
                auto val = tensor[ i, j ];
                EXPECT_EQ( val, 3.1415f );
            }
        }
    }

    TEST( TensorTest, FillWithCudaTensor ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float, Compute::CudaMemoryResource> tensor( shape );
        tensor.fill( 3.1415f );

        // Convert to CPU to check values
        auto cpu_tensor = tensor.to<Compute::HostMemoryResource>();
        for ( size_t i = 0; i < cpu_tensor.shape()[ 0 ]; ++i ) {
            for ( size_t j = 0; j < cpu_tensor.shape()[ 1 ]; ++j ) {
                auto val = cpu_tensor[ i, j ];
                EXPECT_FLOAT_EQ( val, 3.1415f );
            }
        }
    }

    // Index Access Tests
    //----------------------------------------------------------------------------------------
    TEST( TensorTest, OperatorIndex1D ) {
        std::vector<size_t> shape = { 6 };
        Tensor<float, Compute::HostMemoryResource> tensor( shape );
        tensor[ 0 ] = 1.0f;
        EXPECT_EQ( tensor[ 0 ], 1.0f );
    }

    TEST( TensorTest, OperatorIndex2D ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float, Compute::HostMemoryResource> tensor( shape );
        tensor.fill( 3.14f );
        tensor[ 0, 1 ] = 1.0f;
        auto val = tensor[ 0, 1 ];
        EXPECT_EQ( val, 1.0f );
    }

    TEST( TensorTest, OperatorIndexOutOfBounds ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float, Compute::HostMemoryResource> tensor( shape );
        EXPECT_THROW( ( tensor[ 2, 0 ] ), std::out_of_range );
        EXPECT_THROW( ( tensor[ 0, 3 ] ), std::out_of_range );
    }

    TEST( TensorTest, AtMethod ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float, Compute::HostMemoryResource> tensor( shape, 5.0f );
        auto val = tensor.at( { 0, 1 } );
        EXPECT_FLOAT_EQ( val, 5.0f );

        EXPECT_THROW( (tensor.at( { 2, 0 } ) ), std::out_of_range);
        EXPECT_THROW( (tensor.at( { 0, 3 } ) ), std::out_of_range);
    }

    TEST( TensorTest, SetMethod ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float, Compute::HostMemoryResource> tensor( shape );
        tensor.set( { 0, 1 }, 7.5f );
        EXPECT_FLOAT_EQ( ( tensor[ 0, 1 ] ), 7.5f );

        EXPECT_THROW( (tensor.set( { 2, 0 }, 7.5f ) ), std::out_of_range);
        EXPECT_THROW( (tensor.set( { 0, 3 }, 7.5f ) ), std::out_of_range);
    }

    TEST( TensorTest, AtAndSetWithCudaTensor ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float, Compute::CudaMemoryResource> tensor( shape, 3.0f );

        // Get value using at (should work by creating a CPU copy)
        auto val = tensor.at( { 0, 1 } );
        EXPECT_FLOAT_EQ( val, 3.0f );

        // Set value (should work by creating a CPU copy, modifying, and copying back)
        tensor.set( { 0, 1 }, 9.0f );

        // Verify the change worked
        val = tensor.at( { 0, 1 } );
        EXPECT_FLOAT_EQ( val, 9.0f );
    }

    // Memory Slices/Views
    //----------------------------------------------------------------------------------------
    /*TEST( TensorTest, VectorSpan ) {
        std::vector<size_t> shape = { 6 };
        Tensor<float, Compute::HostMemoryResource> tensor( shape );
        auto span = tensor.vectorSpan();
        EXPECT_EQ( span.extent( 0 ), 6 );
    }*/

    /*TEST( TensorTest, CudaTensorVectorSpanThrows ) {
        std::vector<size_t> shape = { 6 };
        Tensor<float, Compute::CudaMemoryResource> tensor( shape );
        EXPECT_THROW( tensor.vectorSpan(), std::runtime_error );
    }*/

    /*TEST( TensorTest, MatrixSpan ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float, Compute::HostMemoryResource> tensor( shape );
        auto span = tensor.matrixSpan( shape );
        EXPECT_EQ( span.extent( 0 ), 2 );
        EXPECT_EQ( span.extent( 1 ), 3 );
    }*/

    //TEST( TensorTest, MatrixSpanWithInvalidShape ) {
    //    std::vector<size_t> shape = { 2, 3 };
    //    Tensor<float, Compute::HostMemoryResource> tensor( shape );
    //    EXPECT_THROW( tensor.matrixSpan( { 7 } ), std::runtime_error ); // Wrong dimensions
    //    EXPECT_THROW( tensor.matrixSpan( { 3, 3 } ), std::runtime_error ); // Exceeds size
    //}

    /*TEST( TensorTest, CudaTensorMatrixSpanThrows ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float, Compute::CudaMemoryResource> tensor( shape );
        EXPECT_THROW( tensor.matrixSpan( shape ), std::runtime_error );
    }*/

    // Data Copy Tests
    //----------------------------------------------------------------------------------------
    TEST( TensorTest, CopyFrom ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float, Compute::HostMemoryResource> src( shape, 1.0f );
        Tensor<float, Compute::HostMemoryResource> dst( shape );

        dst.copyFrom( src );

        for ( size_t i = 0; i < shape[ 0 ]; ++i ) {
            for ( size_t j = 0; j < shape[ 1 ]; ++j ) {
                EXPECT_FLOAT_EQ( ( dst[ i, j ] ), 1.0f );
            }
        }
    }

    TEST( TensorTest, CopyFromCudaToCpu ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float, Compute::CudaMemoryResource> src( shape, 2.0f );
        Tensor<float, Compute::HostMemoryResource> dst( shape );

        dst.copyFrom( src );

        for ( size_t i = 0; i < shape[ 0 ]; ++i ) {
            for ( size_t j = 0; j < shape[ 1 ]; ++j ) {
                EXPECT_FLOAT_EQ( ( dst[ i, j ] ), 2.0f );
            }
        }
    }

    TEST( TensorTest, CopyFromCpuToCuda ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float, Compute::HostMemoryResource> src( shape, 3.0f );
        Tensor<float, Compute::CudaMemoryResource> dst( shape );

        dst.copyFrom( src );

        // Convert back to CPU to verify
        auto cpu_dst = dst.to<Compute::HostMemoryResource>();
        for ( size_t i = 0; i < shape[ 0 ]; ++i ) {
            for ( size_t j = 0; j < shape[ 1 ]; ++j ) {
                EXPECT_FLOAT_EQ( ( cpu_dst[ i, j ] ), 3.0f );
            }
        }
    }

    TEST( TensorTest, CopyFromDifferentShapes ) {
        Tensor<float, Compute::HostMemoryResource> src( { 3, 2 }, 1.0f );
        Tensor<float, Compute::HostMemoryResource> dst( { 2, 3 } );

        EXPECT_THROW( dst.copyFrom( src ), std::runtime_error );
    }

    // Metadata & Introspection
    //----------------------------------------------------------------------------------------
    TEST( TensorTest, Strides ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float, Compute::HostMemoryResource> tensor( shape );
        std::vector<size_t> expected_strides = { 3, 1 };
        EXPECT_EQ( tensor.strides(), expected_strides );
    }

    TEST( TensorTest, Strides3D ) {
        std::vector<size_t> shape = { 2, 3, 4 };
        Tensor<float, Compute::HostMemoryResource> tensor( shape );
        std::vector<size_t> expected_strides = { 12, 4, 1 };
        EXPECT_EQ( tensor.strides(), expected_strides );
    }

    TEST( TensorTest, GetName ) {
        Tensor<float, Compute::HostMemoryResource> tensor;
        EXPECT_TRUE( tensor.getName().empty() );

        // Set and get name
        tensor.setName( "test_tensor" );
        EXPECT_EQ( tensor.getName(), "test_tensor" );
    }

    TEST( TensorTest, SetEmptyNameThrows ) {
        Tensor<float, Compute::HostMemoryResource> tensor;
        EXPECT_THROW( tensor.setName( "" ), std::invalid_argument );
    }

    TEST( TensorTest, GetUID ) {
        Tensor<float, Compute::HostMemoryResource> tensor1;
        Tensor<float, Compute::HostMemoryResource> tensor2;

        // Each tensor should have a unique ID
        EXPECT_NE( tensor1.get_uid(), tensor2.get_uid() );
    }

    TEST( TensorTest, ToString ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float, Compute::HostMemoryResource> tensor( shape );
        tensor.fill( 1.0f );
        tensor.setName( "test_tensor" );

        std::string output = tensor.toString();
        EXPECT_NE( output.find( "Tensor:" ), std::string::npos );
        EXPECT_NE( output.find( "::test_tensor" ), std::string::npos );
        EXPECT_NE( output.find( "Shape: (2,3)" ), std::string::npos );
        EXPECT_NE( output.find( "Size: 6" ), std::string::npos );

        // Test with showBuffer=true
        std::string output_with_buffer = tensor.toString( true );
        EXPECT_GT( output_with_buffer.length(), output.length() );
    }

    TEST( TensorTest, ToStringWithCudaTensor ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float, Compute::CudaMemoryResource> tensor( shape, 1.0f );

        std::string output = tensor.toString();
        EXPECT_NE( output.find( "Tensor:" ), std::string::npos );
        EXPECT_NE( output.find( "Shape: (2,3)" ), std::string::npos );

        // Test with showBuffer=true (should work by creating a CPU copy)
        std::string output_with_buffer = tensor.toString( true );
        EXPECT_GT( output_with_buffer.length(), output.length() );
    }

    // Copy & Move Construction
    //----------------------------------------------------------------------------------------
    TEST( TensorTest, CopyConstructor_Shallow ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float, Compute::HostMemoryResource> original( shape, 3.14f );
        Tensor<float, Compute::HostMemoryResource> copy( original );

        EXPECT_EQ( copy.shape(), original.shape() );
        EXPECT_EQ( copy.size(), original.size() );

        for ( size_t i = 0; i < shape[ 0 ]; ++i ) {
            for ( size_t j = 0; j < shape[ 1 ]; ++j ) {
                EXPECT_FLOAT_EQ( ( copy[ i, j ] ), ( original[ i, j ] ));
            }
        }

		// The original and copy should share the same data
        copy[ 0, 0 ] = 0.0f;

        EXPECT_FLOAT_EQ( ( copy[ 0, 0 ] ), 0.0f );
        EXPECT_FLOAT_EQ( ( original[ 0, 0 ] ), 0.0f );
    }

    TEST( TensorTest, MoveConstructor ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float, Compute::HostMemoryResource> original( shape, 3.14f );
        std::string original_uid = original.get_uid();

        Tensor<float, Compute::HostMemoryResource> moved( std::move( original ) );

        EXPECT_EQ( moved.shape(), shape );
        EXPECT_EQ( moved.size(), 6 );
        EXPECT_EQ( moved.get_uid(), original_uid );

        // The original should be in a moved-from state
        EXPECT_TRUE( original.empty() );
        EXPECT_EQ( original.size(), 0 );
    }

    TEST( TensorTest, CopyAssignment ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float, Compute::HostMemoryResource> original( shape, 3.14f );
        Tensor<float, Compute::HostMemoryResource> copy;

        copy = original;

        EXPECT_EQ( copy.shape(), original.shape() );
        EXPECT_EQ( copy.size(), original.size() );

        for ( size_t i = 0; i < shape[ 0 ]; ++i ) {
            for ( size_t j = 0; j < shape[ 1 ]; ++j ) {
                EXPECT_FLOAT_EQ( ( copy[ i, j ] ), ( original[ i, j ] ) );
            }
        }
    }

    TEST( TensorTest, MoveAssignment ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float, Compute::HostMemoryResource> original( shape, 3.14f );
        std::string original_uid = original.get_uid();

        Tensor<float, Compute::HostMemoryResource> moved;
        moved = std::move( original );

        EXPECT_EQ( moved.shape(), shape );
        EXPECT_EQ( moved.size(), 6 );
        EXPECT_EQ( moved.get_uid(), original_uid );

        // The original should be in a moved-from state
        EXPECT_TRUE( original.empty() );
        EXPECT_EQ( original.size(), 0 );
    }

    // External Data Tests
    //----------------------------------------------------------------------------------------
    TEST( TensorTest, ConstructWithExternalData ) {
        std::vector<size_t> shape = { 2, 3 };
        auto data_ptr = std::make_shared<float[]>(6);
        for ( int i = 0; i < 6; i++ ) data_ptr[ i ] = static_cast<float>( i );

        std::shared_ptr<float> data_ptr_float( data_ptr, data_ptr.get() );
        Tensor<float, Compute::HostMemoryResource> tensor( shape, data_ptr_float );
        //Tensor<float, Compute::HostMemoryResource> tensor( shape, data_ptr );

        EXPECT_EQ( tensor.shape(), shape );
        EXPECT_EQ( tensor.size(), 6 );

        // Verify data was properly shared
        for ( size_t i = 0; i < shape[ 0 ]; ++i ) {
            for ( size_t j = 0; j < shape[ 1 ]; ++j ) {
                size_t index = i * shape[ 1 ] + j;
                EXPECT_FLOAT_EQ( ( tensor[ i, j ] ), static_cast<float>( index ) );
            }
        }

        // Modifying external data should affect the tensor
        data_ptr[ 0 ] = 99.0f;
        EXPECT_FLOAT_EQ( ( tensor[ 0, 0 ] ), 99.0f );
    }

    TEST( TensorTest, ConstructWithNullExternalData ) {
        std::vector<size_t> shape = { 2, 3 };
        std::shared_ptr<float> null_ptr;

        EXPECT_THROW( ( Tensor<float, Compute::HostMemoryResource>( shape, null_ptr ) ), std::invalid_argument );
    }
    
}