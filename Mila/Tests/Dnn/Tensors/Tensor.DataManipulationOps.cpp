#include <gtest/gtest.h>
#include <vector>

import Mila;

namespace Tensors::Tests
{
    using namespace Mila::Dnn;

    class TensorDataManipulationTest : public testing::Test {
    protected:
        TensorDataManipulationTest() {}
    };


    // ====================================================================
    // Clone Operation Tests
    // ====================================================================

    TEST( TensorDataManipulationTest, Clone_BasicFunctionality ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float, Compute::HostMemoryResource> original( shape, 3.14f );
        original.setName( "original" );

        auto cloned_tensor = original.clone();

        // Should have same properties
        EXPECT_EQ( cloned_tensor.shape(), original.shape() );
        EXPECT_EQ( cloned_tensor.size(), original.size() );
        EXPECT_EQ( cloned_tensor.getName(), original.getName() );
        EXPECT_EQ( cloned_tensor.strides(), original.strides() );
        EXPECT_EQ( cloned_tensor.getDataType(), original.getDataType() );

        // Should have different UID (independent tensor)
        EXPECT_NE( cloned_tensor.getUId(), original.getUId() );

        // Should have same values
        for ( size_t i = 0; i < shape[ 0 ]; ++i ) {
            for ( size_t j = 0; j < shape[ 1 ]; ++j ) {
                EXPECT_FLOAT_EQ( (cloned_tensor[ i, j ]), (original[ i, j ]) );
            }
        }

        // Unlike shallow copy, modifying clone should not affect original
        cloned_tensor[ 0, 0 ] = 0.0f;
        EXPECT_FLOAT_EQ( (cloned_tensor[ 0, 0 ]), 0.0f );
        EXPECT_FLOAT_EQ( (original[ 0, 0 ]), 3.14f );
    }

    TEST( TensorDataManipulationTest, Clone_CudaTensor ) {
        std::vector<size_t> shape = { 2, 2 };
        Tensor<float, Compute::CudaMemoryResource> cuda_original( shape, 2.5f );
        cuda_original.setName( "cuda_original" );

        auto cuda_clone = cuda_original.clone();

        EXPECT_EQ( cuda_clone.shape(), cuda_original.shape() );
        EXPECT_EQ( cuda_clone.size(), cuda_original.size() );
        EXPECT_EQ( cuda_clone.getName(), cuda_original.getName() );
        EXPECT_NE( cuda_clone.getUId(), cuda_original.getUId() );

        // Verify independence through host copies
        auto original_host = cuda_original.toHost<Compute::HostMemoryResource>();
        auto clone_host = cuda_clone.toHost<Compute::HostMemoryResource>();

        for ( size_t i = 0; i < shape[ 0 ]; ++i ) {
            for ( size_t j = 0; j < shape[ 1 ]; ++j ) {
                EXPECT_FLOAT_EQ( (original_host[ i, j ]), (clone_host[ i, j ]) );
            }
        }

        // Modify clone and verify independence
        cuda_clone.set( { 0, 0 }, 999.0f );

        auto modified_clone_host = cuda_clone.toHost<Compute::HostMemoryResource>();
        auto unchanged_original_host = cuda_original.toHost<Compute::HostMemoryResource>();

        EXPECT_FLOAT_EQ( (modified_clone_host[ 0, 0 ]), 999.0f );
        EXPECT_FLOAT_EQ( (unchanged_original_host[ 0, 0 ]), 2.5f );
    }

    TEST( TensorDataManipulationTest, Clone_DifferentDataTypes ) {
        std::vector<size_t> shape = { 2, 2 };

        // Test with int
        {
            Tensor<int, Compute::HostMemoryResource> int_original( shape, 42 );
            int_original.setName( "int_tensor" );

            auto int_clone = int_original.clone();

            EXPECT_EQ( int_clone.shape(), int_original.shape() );
            EXPECT_EQ( int_clone.getName(), "int_tensor" );
            EXPECT_NE( int_clone.getUId(), int_original.getUId() );

            for ( size_t i = 0; i < shape[ 0 ]; ++i ) {
                for ( size_t j = 0; j < shape[ 1 ]; ++j ) {
                    EXPECT_EQ( (int_clone[ i, j ]), 42 );
                }
            }

            int_clone[ 0, 0 ] = 100;
            EXPECT_EQ( (int_clone[ 0, 0 ]), 100 );
            EXPECT_EQ( (int_original[ 0, 0 ]), 42 );
        }

        // Test with uint32_t
        {
            Tensor<uint32_t, Compute::HostMemoryResource> uint_original( shape, 1000u );
            uint_original.setName( "uint_tensor" );

            auto uint_clone = uint_original.clone();

            EXPECT_EQ( uint_clone.shape(), uint_original.shape() );
            EXPECT_EQ( uint_clone.getName(), "uint_tensor" );
            EXPECT_NE( uint_clone.getUId(), uint_original.getUId() );

            for ( size_t i = 0; i < shape[ 0 ]; ++i ) {
                for ( size_t j = 0; j < shape[ 1 ]; ++j ) {
                    EXPECT_EQ( (uint_clone[ i, j ]), 1000u );
                }
            }

            uint_clone[ 1, 1 ] = 2000u;
            EXPECT_EQ( (uint_clone[ 1, 1 ]), 2000u );
            EXPECT_EQ( (uint_original[ 1, 1 ]), 1000u );
        }
    }

    TEST( TensorDataManipulationTest, Clone_EmptyTensor ) {
        Tensor<float, Compute::HostMemoryResource> empty_original;
        EXPECT_TRUE( empty_original.empty() );

        auto empty_clone = empty_original.clone();

        EXPECT_TRUE( empty_clone.empty() );
        EXPECT_EQ( empty_clone.size(), 0 );
        EXPECT_EQ( empty_clone.rank(), 0 );
        EXPECT_NE( empty_clone.getUId(), empty_original.getUId() );
    }

    TEST( TensorDataManipulationTest, Clone_PinnedMemory ) {
        std::vector<size_t> shape = { 3, 2 };
        Tensor<float, Compute::CudaPinnedMemoryResource> pinned_original( shape, 5.5f );
        pinned_original.setName( "pinned_tensor" );

        auto pinned_clone = pinned_original.clone();

        EXPECT_EQ( pinned_clone.shape(), pinned_original.shape() );
        EXPECT_EQ( pinned_clone.getName(), "pinned_tensor" );
        EXPECT_NE( pinned_clone.getUId(), pinned_original.getUId() );
        EXPECT_TRUE( pinned_clone.is_host_accessible() );
        EXPECT_TRUE( pinned_clone.is_device_accessible() );

        for ( size_t i = 0; i < shape[ 0 ]; ++i ) {
            for ( size_t j = 0; j < shape[ 1 ]; ++j ) {
                EXPECT_FLOAT_EQ( (pinned_clone[ i, j ]), 5.5f );
            }
        }

        pinned_clone[ 0, 0 ] = 11.0f;
        EXPECT_FLOAT_EQ( (pinned_clone[ 0, 0 ]), 11.0f );
        EXPECT_FLOAT_EQ( (pinned_original[ 0, 0 ]), 5.5f );
    }

    TEST( TensorDataManipulationTest, Clone_LargeTensor ) {
        std::vector<size_t> large_shape = { 50, 100 };
        Tensor<float, Compute::HostMemoryResource> large_original( large_shape, 1.23f );
        large_original.setName( "large_tensor" );

        auto large_clone = large_original.clone();

        EXPECT_EQ( large_clone.shape(), large_original.shape() );
        EXPECT_EQ( large_clone.size(), 5000 );
        EXPECT_EQ( large_clone.getName(), "large_tensor" );
        EXPECT_NE( large_clone.getUId(), large_original.getUId() );

        // Spot check values
        EXPECT_FLOAT_EQ( (large_clone[ 0, 0 ]), 1.23f );
        EXPECT_FLOAT_EQ( (large_clone[ 25, 50 ]), 1.23f );
        EXPECT_FLOAT_EQ( (large_clone[ 49, 99 ]), 1.23f );

        large_clone[ 10, 20 ] = 9.87f;
        EXPECT_FLOAT_EQ( (large_clone[ 10, 20 ]), 9.87f );
        EXPECT_FLOAT_EQ( (large_original[ 10, 20 ]), 1.23f );
    }

    // ====================================================================
    // Combined Operations Tests
    // ====================================================================

    TEST( TensorDataManipulationTest, FillThenClone ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float, Compute::HostMemoryResource> tensor( shape, 0.0f );
        tensor.setName( "test_tensor" );

        tensor.fill( 9.99f );
        auto cloned_tensor = tensor.clone();

        // Verify clone has filled values
        for ( size_t i = 0; i < shape[ 0 ]; ++i ) {
            for ( size_t j = 0; j < shape[ 1 ]; ++j ) {
                EXPECT_FLOAT_EQ( (cloned_tensor[ i, j ]), 9.99f );
            }
        }

        // Verify independence
        tensor.fill( 1.11f );
        for ( size_t i = 0; i < shape[ 0 ]; ++i ) {
            for ( size_t j = 0; j < shape[ 1 ]; ++j ) {
                EXPECT_FLOAT_EQ( (tensor[ i, j ]), 1.11f );
                EXPECT_FLOAT_EQ( (cloned_tensor[ i, j ]), 9.99f );
            }
        }
    }

    TEST( TensorDataManipulationTest, CloneThenFill ) {
        std::vector<size_t> shape = { 2, 2 };
        Tensor<float, Compute::HostMemoryResource> original( shape, 5.0f );
        original.setName( "original" );

        auto clone = original.clone();
        clone.fill( 10.0f );

        // Verify original unchanged
        for ( size_t i = 0; i < shape[ 0 ]; ++i ) {
            for ( size_t j = 0; j < shape[ 1 ]; ++j ) {
                EXPECT_FLOAT_EQ( (original[ i, j ]), 5.0f );
                EXPECT_FLOAT_EQ( (clone[ i, j ]), 10.0f );
            }
        }
    }

    TEST( TensorDataManipulationTest, FillAfterTransfer ) {
        Tensor<float, Compute::HostMemoryResource> host_tensor( { 2, 2 }, 1.0f );

        auto cuda_tensor = host_tensor.toDevice<Compute::CudaMemoryResource>();
        cuda_tensor.fill( 7.5f );

        auto back_to_host = cuda_tensor.toHost<Compute::HostMemoryResource>();

        for ( size_t i = 0; i < 2; ++i ) {
            for ( size_t j = 0; j < 2; ++j ) {
                EXPECT_FLOAT_EQ( (back_to_host[ i, j ]), 7.5f );
            }
        }
    }

    // ====================================================================
    // Edge Cases and Error Conditions
    // ====================================================================

    TEST( TensorDataManipulationTest, MultipleOperations ) {
        std::vector<size_t> shape = { 3, 3 };
        Tensor<float, Compute::HostMemoryResource> tensor( shape, 0.0f );
        tensor.setName( "multi_op_tensor" );

        // Fill -> Clone -> Fill -> Clone chain
        tensor.fill( 1.0f );
        auto clone1 = tensor.clone();

        tensor.fill( 2.0f );
        auto clone2 = tensor.clone();

        clone1.fill( 3.0f );
        auto clone3 = clone1.clone();

        // Verify all tensors have correct values
        for ( size_t i = 0; i < shape[ 0 ]; ++i ) {
            for ( size_t j = 0; j < shape[ 1 ]; ++j ) {
                EXPECT_FLOAT_EQ( (tensor[ i, j ]), 2.0f );
                EXPECT_FLOAT_EQ( (clone1[ i, j ]), 3.0f );
                EXPECT_FLOAT_EQ( (clone2[ i, j ]), 2.0f );
                EXPECT_FLOAT_EQ( (clone3[ i, j ]), 3.0f );
            }
        }

        // Verify all have same metadata except UID
        EXPECT_EQ( clone1.getName(), "multi_op_tensor" );
        EXPECT_EQ( clone2.getName(), "multi_op_tensor" );
        EXPECT_EQ( clone3.getName(), "multi_op_tensor" );

        EXPECT_NE( tensor.getUId(), clone1.getUId() );
        EXPECT_NE( tensor.getUId(), clone2.getUId() );
        EXPECT_NE( clone1.getUId(), clone3.getUId() );
    }

    TEST( TensorDataManipulationTest, SpecialFloatValues ) {
        std::vector<size_t> shape = { 2, 2 };
        Tensor<float, Compute::HostMemoryResource> tensor( shape );

        // Test with infinity
        tensor.fill( std::numeric_limits<float>::infinity() );
        for ( size_t i = 0; i < shape[ 0 ]; ++i ) {
            for ( size_t j = 0; j < shape[ 1 ]; ++j ) {
                EXPECT_TRUE( std::isinf( (tensor[ i, j ]) ) );
            }
        }

        // Test with negative infinity
        tensor.fill( -std::numeric_limits<float>::infinity() );
        for ( size_t i = 0; i < shape[ 0 ]; ++i ) {
            for ( size_t j = 0; j < shape[ 1 ]; ++j ) {
                EXPECT_TRUE( std::isinf( (tensor[ i, j ]) ) );
                EXPECT_LT( (tensor[ i, j ]), 0.0f );
            }
        }

        // Test with NaN
        tensor.fill( std::numeric_limits<float>::quiet_NaN() );
        for ( size_t i = 0; i < shape[ 0 ]; ++i ) {
            for ( size_t j = 0; j < shape[ 1 ]; ++j ) {
                EXPECT_TRUE( std::isnan( (tensor[ i, j ]) ) );
            }
        }
    }
}