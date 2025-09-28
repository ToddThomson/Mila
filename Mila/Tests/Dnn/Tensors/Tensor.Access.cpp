#include <gtest/gtest.h>
#include <vector>

import Mila;

namespace Tensors::Tests
{
    using namespace Mila::Dnn;

    class TensorElementAccessTest : public testing::Test {
    protected:
        TensorElementAccessTest() {}
    };

    // ====================================================================
    // Basic Operator[] Tests
    // ====================================================================

    TEST( TensorElementAccessTest, OperatorIndex1D ) {
        std::vector<size_t> shape = { 6 };
        Tensor<float, Compute::HostMemoryResource> tensor( shape );
        tensor[ 0 ] = 1.0f;
        EXPECT_EQ( tensor[ 0 ], 1.0f );
    }

    TEST( TensorElementAccessTest, OperatorIndex2D ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float, Compute::HostMemoryResource> tensor( shape );
        tensor.fill( 3.14f );
        tensor[ 0, 1 ] = 1.0f;
        auto val = tensor[ 0, 1 ];
        EXPECT_EQ( val, 1.0f );
    }

    TEST( TensorElementAccessTest, OperatorIndex3D ) {
        std::vector<size_t> shape = { 2, 3, 4 };
        Tensor<float, Compute::HostMemoryResource> tensor( shape );
        tensor[ 1, 2, 3 ] = 5.5f;
        EXPECT_FLOAT_EQ( ( tensor[ 1, 2, 3 ] ), 5.5f );
    }

    TEST( TensorElementAccessTest, OperatorIndex4D ) {
        std::vector<size_t> shape = { 2, 2, 2, 2 };
        Tensor<float, Compute::HostMemoryResource> tensor( shape );
        tensor[ 1, 0, 1, 0 ] = 7.7f;
        EXPECT_FLOAT_EQ( ( tensor[ 1, 0, 1, 0 ] ), 7.7f );
    }

    TEST( TensorElementAccessTest, OperatorIndexOutOfBounds ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float, Compute::HostMemoryResource> tensor( shape );
        EXPECT_THROW( (tensor[ 2, 0 ]), std::out_of_range );
        EXPECT_THROW( (tensor[ 0, 3 ]), std::out_of_range );
        EXPECT_THROW( (tensor[ 2, 3 ]), std::out_of_range );
    }

    TEST( TensorElementAccessTest, OperatorIndexWrongDimensions ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float, Compute::HostMemoryResource> tensor( shape );

        EXPECT_THROW( tensor[ 0 ], std::runtime_error );
        EXPECT_THROW( (tensor[ 0, 1, 2 ]), std::runtime_error );
    }

    // ====================================================================
    // Vector-based Operator[] Tests  
    // ====================================================================

    TEST( TensorElementAccessTest, VectorOperatorIndex ) {
        std::vector<size_t> shape = { 2, 3, 4 };
        Tensor<float, Compute::HostMemoryResource> tensor( shape );

        std::vector<size_t> indices = { 1, 2, 3 };
        tensor[ indices ] = 9.9f;
        EXPECT_FLOAT_EQ( tensor[ indices ], 9.9f );
    }

    TEST( TensorElementAccessTest, VectorOperatorIndexConst ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float, Compute::HostMemoryResource> tensor( shape, 4.4f );

        const auto& const_tensor = tensor;
        std::vector<size_t> indices = { 1, 2 };
        EXPECT_FLOAT_EQ( const_tensor[ indices ], 4.4f );
    }

    // ====================================================================
    // at() Method Tests
    // ====================================================================

    TEST( TensorElementAccessTest, AtMethod ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float, Compute::HostMemoryResource> tensor( shape, 5.0f );
        auto val = tensor.at( { 0, 1 } );
        EXPECT_FLOAT_EQ( val, 5.0f );

        EXPECT_THROW( (tensor.at( { 2, 0 } )), std::out_of_range );
        EXPECT_THROW( (tensor.at( { 0, 3 } )), std::out_of_range );
    }

    TEST( TensorElementAccessTest, AtMethodBoundsChecking ) {
        std::vector<size_t> shape = { 3, 4, 5 };
        Tensor<float, Compute::HostMemoryResource> tensor( shape, 1.0f );

        EXPECT_NO_THROW( tensor.at( { 0, 0, 0 } ) );
        EXPECT_NO_THROW( tensor.at( { 2, 3, 4 } ) );
        EXPECT_NO_THROW( tensor.at( { 1, 2, 3 } ) );

        EXPECT_THROW( tensor.at( { 3, 0, 0 } ), std::out_of_range );
        EXPECT_THROW( tensor.at( { 0, 4, 0 } ), std::out_of_range );
        EXPECT_THROW( tensor.at( { 0, 0, 5 } ), std::out_of_range );
    }

    TEST( TensorElementAccessTest, AtMethodWrongDimensions ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float, Compute::HostMemoryResource> tensor( shape );

        EXPECT_THROW( tensor.at( { 0 } ), std::runtime_error );
        EXPECT_THROW( tensor.at( { 0, 1, 2 } ), std::runtime_error );
        EXPECT_THROW( tensor.at( {} ), std::runtime_error );
    }

    // ====================================================================
    // set() Method Tests
    // ====================================================================

    TEST( TensorElementAccessTest, SetMethod ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float, Compute::HostMemoryResource> tensor( shape );
        tensor.set( { 0, 1 }, 7.5f );
        EXPECT_FLOAT_EQ( (tensor[ 0, 1 ]), 7.5f );

        EXPECT_THROW( (tensor.set( { 2, 0 }, 7.5f )), std::out_of_range );
        EXPECT_THROW( (tensor.set( { 0, 3 }, 7.5f )), std::out_of_range );
    }

    TEST( TensorElementAccessTest, SetMethodBoundsChecking ) {
        std::vector<size_t> shape = { 3, 4 };
        Tensor<float, Compute::HostMemoryResource> tensor( shape );

        EXPECT_NO_THROW( tensor.set( { 0, 0 }, 1.0f ) );
        EXPECT_NO_THROW( tensor.set( { 2, 3 }, 2.0f ) );

        EXPECT_THROW( tensor.set( { 3, 0 }, 1.0f ), std::out_of_range );
        EXPECT_THROW( tensor.set( { 0, 4 }, 1.0f ), std::out_of_range );
    }

    TEST( TensorElementAccessTest, SetMethodWrongDimensions ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float, Compute::HostMemoryResource> tensor( shape );

        EXPECT_THROW( tensor.set( { 0 }, 1.0f ), std::runtime_error );
        EXPECT_THROW( tensor.set( { 0, 1, 2 }, 1.0f ), std::runtime_error );
    }

    // ====================================================================
    // Device Memory Access Tests
    // ====================================================================

    TEST( TensorElementAccessTest, DeviceMemoryDirectAccessThrows ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float, Compute::CudaMemoryResource> cuda_tensor( shape );

        EXPECT_THROW( (cuda_tensor[ 0, 0 ]), std::runtime_error );
        EXPECT_THROW( (cuda_tensor[ { 0, 1 } ]), std::runtime_error );
    }

    TEST( TensorElementAccessTest, AtAndSetWithCudaTensor ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float, Compute::CudaMemoryResource> tensor( shape, 3.0f );

        auto val = tensor.at( { 0, 1 } );
        EXPECT_FLOAT_EQ( val, 3.0f );

        tensor.set( { 0, 1 }, 9.0f );

        val = tensor.at( { 0, 1 } );
        EXPECT_FLOAT_EQ( val, 9.0f );
    }

    TEST( TensorElementAccessTest, CudaTensorBoundsChecking ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float, Compute::CudaMemoryResource> cuda_tensor( shape );

        EXPECT_THROW( cuda_tensor.at( { 2, 0 } ), std::out_of_range );
        EXPECT_THROW( cuda_tensor.at( { 0, 3 } ), std::out_of_range );
        EXPECT_THROW( cuda_tensor.set( { 2, 1 }, 1.0f ), std::out_of_range );
        EXPECT_THROW( cuda_tensor.set( { 1, 3 }, 1.0f ), std::out_of_range );
    }

    // ====================================================================
    // Different Data Types Tests
    // ====================================================================

    TEST( TensorElementAccessTest, IntegerTypes ) {
        std::vector<size_t> shape = { 2, 2 };

        {
            Tensor<int, Compute::HostMemoryResource> int_tensor( shape );
            int_tensor[ 0, 0 ] = 42;
            int_tensor.set( { 1, 1 }, 84 );
            EXPECT_EQ( (int_tensor[ 0, 0 ]), 42 );
            EXPECT_EQ( (int_tensor.at( { 1, 1 } ) ), 84 );
        }

        {
            Tensor<int16_t, Compute::HostMemoryResource> int16_tensor( shape );
            int16_tensor[ 0, 1 ] = -1000;
            int16_tensor.set( { 1, 0 }, 2000 );
            EXPECT_EQ( ( int16_tensor[ 0, 1 ] ), -1000 );
            EXPECT_EQ( ( int16_tensor.at( { 1, 0 } ) ), 2000 );
        }

        {
            Tensor<uint32_t, Compute::HostMemoryResource> uint32_tensor( shape );
            uint32_tensor[ 1, 0 ] = 4000000000u;
            uint32_tensor.set( { 0, 1 }, 1000000u );
            EXPECT_EQ( ( uint32_tensor[ 1, 0 ] ), 4000000000u );
            EXPECT_EQ( ( uint32_tensor.at( { 0, 1 } ) ), 1000000u );
        }
    }

    // ====================================================================
    // Memory Resource Access Tests
    // ====================================================================

    TEST( TensorElementAccessTest, PinnedMemoryAccess ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float, Compute::CudaPinnedMemoryResource> pinned_tensor( shape );

        pinned_tensor[ 0, 0 ] = 1.1f;
        pinned_tensor[ 1, 2 ] = 2.2f;
        pinned_tensor.set( { 0, 1 }, 3.3f );

        EXPECT_FLOAT_EQ( ( pinned_tensor[ 0, 0 ] ), 1.1f );
        EXPECT_FLOAT_EQ( ( pinned_tensor[ 1, 2 ] ), 2.2f );
        EXPECT_FLOAT_EQ( ( pinned_tensor.at( { 0, 1 } ) ), 3.3f );
    }

    TEST( TensorElementAccessTest, ManagedMemoryAccess ) {
        std::vector<size_t> shape = { 2, 2 };
        Tensor<float, Compute::CudaManagedMemoryResource> managed_tensor( shape );

        managed_tensor[ 0, 0 ] = 4.4f;
        managed_tensor[ 1, 1 ] = 5.5f;
        managed_tensor.set( { 0, 1 }, 6.6f );

        EXPECT_FLOAT_EQ( ( managed_tensor[ 0, 0 ] ), 4.4f );
        EXPECT_FLOAT_EQ( ( managed_tensor[ 1, 1 ] ), 5.5f );
        EXPECT_FLOAT_EQ( ( managed_tensor.at( { 0, 1 } ) ), 6.6f );
    }

    // ====================================================================
    // Edge Cases and Special Scenarios
    // ====================================================================

    TEST( TensorElementAccessTest, SingleElementTensor ) {
        std::vector<size_t> shape = { 1 };
        Tensor<float, Compute::HostMemoryResource> single_tensor( shape );

        single_tensor[ 0 ] = 99.0f;
        EXPECT_FLOAT_EQ( single_tensor[ 0 ], 99.0f );
        EXPECT_FLOAT_EQ( single_tensor.at( { 0 } ), 99.0f );

        single_tensor.set( { 0 }, 88.0f );
        EXPECT_FLOAT_EQ( single_tensor[ 0 ], 88.0f );
    }

    TEST( TensorElementAccessTest, MultiDimensionalSingleElement ) {
        std::vector<size_t> shape = { 1, 1, 1 };
        Tensor<float, Compute::HostMemoryResource> tensor( shape );

        tensor[ 0, 0, 0 ] = 77.0f;
        EXPECT_FLOAT_EQ( ( tensor[ 0, 0, 0 ] ), 77.0f );
        EXPECT_FLOAT_EQ( tensor.at( { 0, 0, 0 } ), 77.0f );

        tensor.set( { 0, 0, 0 }, 66.0f );
        EXPECT_FLOAT_EQ( ( tensor[ 0, 0, 0 ] ), 66.0f );
    }

    TEST( TensorElementAccessTest, LargeTensorAccess ) {
        std::vector<size_t> shape = { 100, 200 };
        Tensor<float, Compute::HostMemoryResource> large_tensor( shape, 1.0f );

        large_tensor[ 50, 100 ] = 2.0f;
        large_tensor[ 99, 199 ] = 3.0f;
        large_tensor.set( { 0, 0 }, 4.0f );

        EXPECT_FLOAT_EQ( ( large_tensor[ 50, 100 ] ), 2.0f );
        EXPECT_FLOAT_EQ( ( large_tensor[ 99, 199 ] ), 3.0f );
        EXPECT_FLOAT_EQ( ( large_tensor.at( { 0, 0 } ) ), 4.0f );
        EXPECT_FLOAT_EQ( ( large_tensor[ 25, 75 ] ), 1.0f );
    }

    // ====================================================================
    // Access Pattern Tests
    // ====================================================================

    TEST( TensorElementAccessTest, SequentialAccess ) {
        std::vector<size_t> shape = { 3, 4 };
        Tensor<float, Compute::HostMemoryResource> tensor( shape );

        float value = 1.0f;
        for ( size_t i = 0; i < shape[ 0 ]; ++i ) {
            for ( size_t j = 0; j < shape[ 1 ]; ++j ) {
                tensor[ i, j ] = value;
                value += 1.0f;
            }
        }

        value = 1.0f;
        for ( size_t i = 0; i < shape[ 0 ]; ++i ) {
            for ( size_t j = 0; j < shape[ 1 ]; ++j ) {
                EXPECT_FLOAT_EQ( ( tensor[ i, j ] ), value );
                EXPECT_FLOAT_EQ( ( tensor.at( { i, j } ) ), value );
                value += 1.0f;
            }
        }
    }

    TEST( TensorElementAccessTest, RandomAccess ) {
        std::vector<size_t> shape = { 5, 5 };
        Tensor<float, Compute::HostMemoryResource> tensor( shape, 0.0f );

        tensor[ 2, 3 ] = 1.0f;
        tensor[ 0, 4 ] = 2.0f;
        tensor[ 4, 0 ] = 3.0f;
        tensor[ 1, 1 ] = 4.0f;
        tensor.set( { 3, 2 }, 5.0f );

        EXPECT_FLOAT_EQ( ( tensor[ 2, 3 ] ), 1.0f );
        EXPECT_FLOAT_EQ( ( tensor[ 0, 4 ] ), 2.0f );
        EXPECT_FLOAT_EQ( ( tensor[ 4, 0 ] ), 3.0f );
        EXPECT_FLOAT_EQ( ( tensor[ 1, 1 ] ), 4.0f );
        EXPECT_FLOAT_EQ( ( tensor.at( { 3, 2 } )), 5.0f );

        EXPECT_FLOAT_EQ( ( tensor[ 0, 0 ] ), 0.0f );
        EXPECT_FLOAT_EQ( ( tensor[ 4, 4 ] ), 0.0f );
    }

    // ====================================================================
    // Const Correctness Tests
    // ====================================================================

    TEST( TensorElementAccessTest, ConstTensorAccess ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float, Compute::HostMemoryResource> tensor( shape, 7.0f );

        const auto& const_tensor = tensor;

        EXPECT_FLOAT_EQ( ( const_tensor[ 0, 0 ] ), 7.0f );
        EXPECT_FLOAT_EQ( ( const_tensor[ 1, 2 ] ), 7.0f );
        EXPECT_FLOAT_EQ( ( const_tensor.at( { 0, 1 } ) ), 7.0f );

        std::vector<size_t> indices = { 1, 1 };
        EXPECT_FLOAT_EQ( ( const_tensor[ indices ] ), 7.0f );
    }

    // ====================================================================
    // Error Message Validation Tests
    // ====================================================================

    TEST( TensorElementAccessTest, ErrorMessageValidation ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float, Compute::HostMemoryResource> tensor( shape );

        try {
            tensor[ 2, 0 ];
            FAIL() << "Expected std::out_of_range exception";
        }
        catch ( const std::out_of_range& e ) {
            std::string error_msg( e.what() );
            EXPECT_NE( error_msg.find( "operator[]" ), std::string::npos );
            EXPECT_NE( error_msg.find( "out of range" ), std::string::npos );
        }

        try {
            tensor.at( { 0, 3 } );
            FAIL() << "Expected std::out_of_range exception";
        }
        catch ( const std::out_of_range& e ) {
            std::string error_msg( e.what() );
            EXPECT_NE( error_msg.find( "at()" ), std::string::npos );
            EXPECT_NE( error_msg.find( "out of range" ), std::string::npos );
        }

        try {
            tensor.set( { 0 }, 1.0f );
            FAIL() << "Expected std::runtime_error exception";
        }
        catch ( const std::runtime_error& e ) {
            std::string error_msg( e.what() );
            EXPECT_NE( error_msg.find( "set()" ), std::string::npos );
            EXPECT_NE( error_msg.find( "match the tensor rank" ), std::string::npos );
        }
    }

    // ====================================================================
    // Performance and Stress Tests
    // ====================================================================

    TEST( TensorElementAccessTest, AccessPerformance ) {
        std::vector<size_t> shape = { 100, 100 };
        Tensor<float, Compute::HostMemoryResource> tensor( shape );

        for ( size_t i = 0; i < 100; ++i ) {
            for ( size_t j = 0; j < 100; ++j ) {
                tensor[ i, j ] = static_cast<float>( i * 100 + j );
            }
        }

        for ( size_t i = 0; i < 100; ++i ) {
            for ( size_t j = 0; j < 100; ++j ) {
                float expected = static_cast<float>( i * 100 + j );
                EXPECT_FLOAT_EQ( ( tensor[ i, j ] ), expected );
            }
        }
    }
}