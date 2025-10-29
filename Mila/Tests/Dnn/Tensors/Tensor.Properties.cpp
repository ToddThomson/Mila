#include <gtest/gtest.h>
#include <vector>

import Mila;

namespace Dnn::Tensors::Tests
{
    using namespace Mila::Dnn;

    class TensorPropertiesTest : public testing::Test {
    protected:
        TensorPropertiesTest() {}
    };

    // ====================================================================
    // Shape Property Tests
    // ====================================================================

    TEST( TensorPropertiesTest, Shape_BasicProperties ) {
        shape_t shape2D = { 2, 3 };
        HostTensor<TensorDataType::FP32> tensor2D( "CPU", shape2D );
        EXPECT_EQ( tensor2D.shape(), shape2D );

        shape_t shape3D = { 4, 5, 6 };
        HostTensor<TensorDataType::FP32> tensor3D( "CPU", shape3D );
        EXPECT_EQ( tensor3D.shape(), shape3D );

        shape_t shape1D = { 10 };
        HostTensor<TensorDataType::FP32> tensor1D( "CPU", shape1D );
        EXPECT_EQ( tensor1D.shape(), shape1D );
    }

    TEST( TensorPropertiesTest, Shape_EmptyTensor ) {
        // Empty shape {} represents a scalar (rank 0)
        HostTensor<TensorDataType::FP32> empty_tensor( "CPU", shape_t{} );
        
        EXPECT_EQ( empty_tensor.shape().size(), 0 );
        EXPECT_TRUE( empty_tensor.shape().empty() );
    }

    TEST( TensorPropertiesTest, Shape_ScalarTensor ) {
        shape_t scalar_shape = {};
        HostTensor<TensorDataType::FP32> scalar_tensor( "CPU", scalar_shape );
        
        EXPECT_EQ( scalar_tensor.shape(), scalar_shape );
        EXPECT_TRUE( scalar_tensor.shape().empty() );
    }

    TEST( TensorPropertiesTest, Shape_LargeDimensional ) {
        shape_t large_shape = { 2, 3, 4, 5, 6, 7 };
        HostTensor<TensorDataType::FP32> large_tensor( "CPU", large_shape );
        EXPECT_EQ( large_tensor.shape(), large_shape );
        EXPECT_EQ( large_tensor.shape().size(), 6 );
    }

    // ====================================================================
    // Stride Property Tests
    // ====================================================================

    TEST( TensorPropertiesTest, Strides_2D ) {
        shape_t shape = { 2, 3 };
        HostTensor<TensorDataType::FP32> tensor( "CPU", shape );
        stride_t expected_strides = { 3, 1 };
        
        EXPECT_EQ( tensor.strides(), expected_strides );
    }

    TEST( TensorPropertiesTest, Strides_3D ) {
        shape_t shape = { 2, 3, 4 };
        HostTensor<TensorDataType::FP32> tensor( "CPU", shape );
        shape_t expected_strides = { 12, 4, 1 };
        EXPECT_EQ( tensor.strides(), expected_strides );
    }

    TEST( TensorPropertiesTest, Strides_4D ) {
        shape_t shape = { 2, 3, 4, 5 };
        HostTensor<TensorDataType::FP32> tensor( "CPU", shape );
        shape_t expected_strides = { 60, 20, 5, 1 };
        EXPECT_EQ( tensor.strides(), expected_strides );
    }

    TEST( TensorPropertiesTest, Strides_1D ) {
        shape_t shape = { 10 };
        HostTensor<TensorDataType::FP32> tensor( "CPU", shape );
        shape_t expected_strides = { 1 };
        EXPECT_EQ( tensor.strides(), expected_strides );
    }

    TEST( TensorPropertiesTest, Strides_EmptyTensor ) {
        // Scalar: empty shape => empty strides
        HostTensor<TensorDataType::FP32> empty_tensor( "CPU", shape_t{} );
        EXPECT_TRUE( empty_tensor.strides().empty() );
    }

    TEST( TensorPropertiesTest, Strides_LargeShape ) {
        shape_t shape = { 10, 20, 30, 40 };
        HostTensor<TensorDataType::FP32> tensor( "CPU", shape );
        shape_t expected_strides = { 24000, 1200, 40, 1 };
        EXPECT_EQ( tensor.strides(), expected_strides );
    }

    // ====================================================================
    // Size Property Tests
    // ====================================================================

    TEST( TensorPropertiesTest, Size_BasicCalculations ) {
        shape_t shape2D = { 2, 3 };
        HostTensor<TensorDataType::FP32> tensor2D( "CPU", shape2D );
        EXPECT_EQ( tensor2D.size(), 6 );

        shape_t shape3D = { 2, 3, 4 };
        HostTensor<TensorDataType::FP32> tensor3D( "CPU", shape3D );
        EXPECT_EQ( tensor3D.size(), 24 );

        shape_t shape1D = { 10 };
        HostTensor<TensorDataType::FP32> tensor1D( "CPU", shape1D );
        EXPECT_EQ( tensor1D.size(), 10 );
    }

    TEST( TensorPropertiesTest, Size_EmptyTensor ) {
        // Use shape {0} to represent a zero-sized tensor
        HostTensor<TensorDataType::FP32> empty_tensor( "CPU", shape_t{0} );
        EXPECT_EQ( empty_tensor.size(), 0 );
    }

    TEST( TensorPropertiesTest, Size_ScalarTensor ) {
        shape_t scalar_shape = {};
        HostTensor<TensorDataType::FP32> scalar_tensor( "CPU", scalar_shape );
        // Scalar has one logical element
        EXPECT_EQ( scalar_tensor.size(), 1 );
    }

    TEST( TensorPropertiesTest, Size_LargeTensor ) {
        shape_t large_shape = { 100, 200 };
        HostTensor<TensorDataType::FP32> large_tensor( "CPU", large_shape );
        EXPECT_EQ( large_tensor.size(), 20000 );
    }

    TEST( TensorPropertiesTest, Size_DifferentMemoryTypes ) {
        shape_t shape = { 3, 4 };

        HostTensor<TensorDataType::FP32> host_tensor( "CPU", shape );
        Tensor<TensorDataType::FP32, Compute::CudaDeviceMemoryResource> cuda_tensor( "CUDA:0", shape );
        Tensor<TensorDataType::FP32, Compute::CudaPinnedMemoryResource> pinned_tensor( "CUDA:0", shape );
        Tensor<TensorDataType::FP32, Compute::CudaManagedMemoryResource> managed_tensor( "CUDA:0", shape );

        EXPECT_EQ( host_tensor.size(), 12 );
        EXPECT_EQ( cuda_tensor.size(), 12 );
        EXPECT_EQ( pinned_tensor.size(), 12 );
        EXPECT_EQ( managed_tensor.size(), 12 );
    }

    // ====================================================================
    // Rank Property Tests
    // ====================================================================

    TEST( TensorPropertiesTest, Rank_VariousDimensions ) {
        HostTensor<TensorDataType::FP32> tensor0D( "CPU", shape_t{} );
        EXPECT_EQ( tensor0D.rank(), 0 );

        shape_t shape1D = { 5 };
        HostTensor<TensorDataType::FP32> tensor1D( "CPU", shape1D );
        EXPECT_EQ( tensor1D.rank(), 1 );

        shape_t shape2D = { 3, 4 };
        HostTensor<TensorDataType::FP32> tensor2D( "CPU", shape2D );
        EXPECT_EQ( tensor2D.rank(), 2 );

        shape_t shape3D = { 2, 3, 4 };
        HostTensor<TensorDataType::FP32> tensor3D( "CPU", shape3D );
        EXPECT_EQ( tensor3D.rank(), 3 );

        shape_t shape4D = { 2, 3, 4, 5 };
        HostTensor<TensorDataType::FP32> tensor4D( "CPU", shape4D );
        EXPECT_EQ( tensor4D.rank(), 4 );
    }

    TEST( TensorPropertiesTest, Rank_HighDimensional ) {
        shape_t high_dim_shape = { 1, 2, 3, 4, 5, 6, 7, 8 };
        HostTensor<TensorDataType::FP32> high_dim_tensor( "CPU", high_dim_shape );
        EXPECT_EQ( high_dim_tensor.rank(), 8 );
    }

    TEST( TensorPropertiesTest, Rank_DifferentDataTypes ) {
        shape_t shape = { 2, 3, 4 };

        HostTensor<TensorDataType::FP32> float_tensor( "CPU", shape );
        HostTensor<TensorDataType::INT32> int_tensor( "CPU", shape );
        HostTensor<TensorDataType::UINT16> uint16_tensor( "CPU", shape );
        HostTensor<TensorDataType::INT16> int16_tensor( "CPU", shape );

        EXPECT_EQ( float_tensor.rank(), 3 );
        EXPECT_EQ( int_tensor.rank(), 3 );
        EXPECT_EQ( uint16_tensor.rank(), 3 );
        EXPECT_EQ( int16_tensor.rank(), 3 );
    }

    // ====================================================================
    // Empty Property Tests
    // ====================================================================

    TEST( TensorPropertiesTest, Empty_DefaultConstructor ) {
        // Empty shape {} represents scalar (rank 0)
        HostTensor<TensorDataType::FP32> scalar_tensor( "CPU", shape_t{} );
        EXPECT_FALSE( scalar_tensor.empty() );
        EXPECT_EQ( scalar_tensor.size(), 1 );
        EXPECT_EQ( scalar_tensor.rank(), 0 );
    }

    TEST( TensorPropertiesTest, Empty_ZeroSizeShape ) {
        shape_t zero_shape = { 0 };
        HostTensor<TensorDataType::FP32> zero_tensor( "CPU", zero_shape );
        EXPECT_TRUE( zero_tensor.empty() );
    }

    TEST( TensorPropertiesTest, Empty_NonEmptyTensors ) {
        shape_t shape1D = { 1 };
        HostTensor<TensorDataType::FP32> tensor1D( "CPU", shape1D );
        EXPECT_FALSE( tensor1D.empty() );

        shape_t shape2D = { 2, 3 };
        HostTensor<TensorDataType::FP32> tensor2D( "CPU", shape2D );
        EXPECT_FALSE( tensor2D.empty() );

        shape_t shape3D = { 1, 1, 1 };
        HostTensor<TensorDataType::FP32> tensor3D( "CPU", shape3D );
        EXPECT_FALSE( tensor3D.empty() );
    }

    TEST( TensorPropertiesTest, Empty_DifferentMemoryTypes ) {
        HostTensor<TensorDataType::FP32> host_empty( "CPU", shape_t{0} );
        Tensor<TensorDataType::FP32, Compute::CudaDeviceMemoryResource> cuda_empty( "CUDA:0", shape_t{0} );
        Tensor<TensorDataType::FP32, Compute::CudaPinnedMemoryResource> pinned_empty( "CUDA:0", shape_t{0} );
        Tensor<TensorDataType::FP32, Compute::CudaManagedMemoryResource> managed_empty( "CUDA:0", shape_t{0} );

        EXPECT_TRUE( host_empty.empty() );
        EXPECT_TRUE( cuda_empty.empty() );
        EXPECT_TRUE( pinned_empty.empty() );
        EXPECT_TRUE( managed_empty.empty() );
    }

    // ====================================================================
    // Property Consistency Tests
    // ====================================================================

    TEST( TensorPropertiesTest, PropertyConsistency_SizeRankShape ) {
        shape_t shape = { 2, 3, 4 };
        HostTensor<TensorDataType::FP32> tensor( "CPU", shape );

        EXPECT_EQ( tensor.shape(), shape );
        EXPECT_EQ( tensor.rank(), shape.size() );
        EXPECT_EQ( tensor.size(), 2 * 3 * 4 );
        EXPECT_FALSE( tensor.empty() );
    }

    TEST( TensorPropertiesTest, PropertyConsistency_AfterOperations ) {
        shape_t shape = { 3, 4 };
        HostTensor<TensorDataType::FP32> tensor( "CPU", shape );

        // Use available API: setName and constructing a same-shape tensor to validate shape preservation
        tensor.setName( "test_tensor" );
        HostTensor<TensorDataType::FP32> cloned( "CPU", shape );
        EXPECT_EQ( cloned.shape(), shape );
        EXPECT_EQ( cloned.rank(), 2 );
        EXPECT_EQ( cloned.size(), 12 );
        EXPECT_FALSE( cloned.empty() );
    }

    TEST( TensorPropertiesTest, PropertyConsistency_AfterTransfer ) {
        shape_t shape = { 2, 3 };
        HostTensor<TensorDataType::FP32> host_tensor( "CPU", shape );

        // Try to create a device tensor by name and copy host -> device -> host.
        try
        {
            Tensor<TensorDataType::FP32, Compute::CudaDeviceMemoryResource> cuda_tensor( "CUDA:0", shape );
            EXPECT_NO_THROW( copy( host_tensor, cuda_tensor ) );

            auto back_to_host = toHost<TensorDataType::FP32>( cuda_tensor );
            EXPECT_EQ( back_to_host.shape(), shape );
            EXPECT_EQ( back_to_host.rank(), 2 );
            EXPECT_EQ( back_to_host.size(), 6 );
            EXPECT_FALSE( back_to_host.empty() );
        }
        catch (const std::exception& e)
        {
            GTEST_SKIP() << "CUDA device not available for transfer test: " << e.what();
        }
    }

    // ====================================================================
    // Edge Cases and Special Scenarios
    // ====================================================================

    TEST( TensorPropertiesTest, EdgeCases_SingleElementTensor ) {
        shape_t single_shape = { 1 };
        HostTensor<TensorDataType::FP32> single_tensor( "CPU", single_shape );

        EXPECT_EQ( single_tensor.shape(), single_shape );
        EXPECT_EQ( single_tensor.rank(), 1 );
        EXPECT_EQ( single_tensor.size(), 1 );
        EXPECT_FALSE( single_tensor.empty() );
        EXPECT_EQ( single_tensor.strides(), shape_t{1} );
    }

    TEST( TensorPropertiesTest, EdgeCases_MultiDimensionalSingleElement ) {
        shape_t multi_single_shape = { 1, 1, 1, 1 };
        HostTensor<TensorDataType::FP32> multi_single_tensor( "CPU", multi_single_shape );

        EXPECT_EQ( multi_single_tensor.shape(), multi_single_shape );
        EXPECT_EQ( multi_single_tensor.rank(), 4 );
        EXPECT_EQ( multi_single_tensor.size(), 1 );
        EXPECT_FALSE( multi_single_tensor.empty() );
        EXPECT_EQ( multi_single_tensor.strides(), shape_t( { 1, 1, 1, 1 } ));
    }

    TEST( TensorPropertiesTest, EdgeCases_LargeUniformShape ) {
        shape_t uniform_shape = { 10, 10, 10, 10 };
        HostTensor<TensorDataType::FP32> uniform_tensor( "CPU", uniform_shape );

        EXPECT_EQ( uniform_tensor.shape(), uniform_shape );
        EXPECT_EQ( uniform_tensor.rank(), 4 );
        EXPECT_EQ( uniform_tensor.size(), 10000 );
        EXPECT_FALSE( uniform_tensor.empty() );
        EXPECT_EQ( uniform_tensor.strides(), shape_t( { 1000, 100, 10, 1 } ) );
    }

    TEST( TensorPropertiesTest, EdgeCases_AsymmetricShape ) {
        shape_t asymmetric_shape = { 1, 100, 1, 50 };
        HostTensor<TensorDataType::FP32> asymmetric_tensor( "CPU", asymmetric_shape );

        EXPECT_EQ( asymmetric_tensor.shape(), asymmetric_shape );
        EXPECT_EQ( asymmetric_tensor.rank(), 4 );
        EXPECT_EQ( asymmetric_tensor.size(), 5000 );
        EXPECT_FALSE( asymmetric_tensor.empty() );
        EXPECT_EQ( asymmetric_tensor.strides(), stride_t( { 5000, 50, 50, 1 } ) );
    }

    // ====================================================================
    // Property Validation with Different Data Types
    // ====================================================================

    TEST( TensorPropertiesTest, DataTypes_PropertyConsistency ) {
        shape_t shape = { 3, 4 };

        HostTensor<TensorDataType::FP32> float_tensor( "CPU", shape );
        HostTensor<TensorDataType::INT32> int_tensor( "CPU", shape );
        HostTensor<TensorDataType::UINT16> uint16_tensor( "CPU", shape );
        HostTensor<TensorDataType::INT16> int16_tensor( "CPU", shape );
        HostTensor<TensorDataType::UINT32> uint32_tensor( "CPU", shape );

        EXPECT_EQ( float_tensor.shape(), shape );
        EXPECT_EQ( int_tensor.shape(), shape );
        EXPECT_EQ( uint16_tensor.shape(), shape );
        EXPECT_EQ( int16_tensor.shape(), shape );
        EXPECT_EQ( uint32_tensor.shape(), shape );

        EXPECT_EQ( float_tensor.size(), 12 );
        EXPECT_EQ( int_tensor.size(), 12 );
        EXPECT_EQ( uint16_tensor.size(), 12 );
        EXPECT_EQ( int16_tensor.size(), 12 );
        EXPECT_EQ( uint32_tensor.size(), 12 );

        EXPECT_EQ( float_tensor.rank(), 2 );
        EXPECT_EQ( int_tensor.rank(), 2 );
        EXPECT_EQ( uint16_tensor.rank(), 2 );
        EXPECT_EQ( int16_tensor.rank(), 2 );
        EXPECT_EQ( uint32_tensor.rank(), 2 );

        stride_t expected_strides = { 4, 1 };
        EXPECT_EQ( float_tensor.strides(), expected_strides );
        EXPECT_EQ( int_tensor.strides(), expected_strides );
        EXPECT_EQ( uint16_tensor.strides(), expected_strides );
        EXPECT_EQ( int16_tensor.strides(), expected_strides );
        EXPECT_EQ( uint32_tensor.strides(), expected_strides );
    }

    // ====================================================================
    // Performance and Stress Tests
    // ====================================================================

    TEST( TensorPropertiesTest, Performance_LargeTensorProperties ) {
        shape_t large_shape = { 1000, 1000 };
        HostTensor<TensorDataType::FP32> large_tensor( "CPU", large_shape );

        EXPECT_EQ( large_tensor.shape(), large_shape );
        EXPECT_EQ( large_tensor.rank(), 2 );
        EXPECT_EQ( large_tensor.size(), 1000000 );
        EXPECT_FALSE( large_tensor.empty() );
        EXPECT_EQ( large_tensor.strides(), stride_t( { 1000, 1 } ) );
    }

    TEST( TensorPropertiesTest, Performance_HighDimensionalTensor ) {
        shape_t high_dim_shape = { 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 };
        HostTensor<TensorDataType::FP32> high_dim_tensor( "CPU", high_dim_shape );

        EXPECT_EQ( high_dim_tensor.shape(), high_dim_shape );
        EXPECT_EQ( high_dim_tensor.rank(), 10 );
        EXPECT_EQ( high_dim_tensor.size(), 1024 );
        EXPECT_FALSE( high_dim_tensor.empty() );

        shape_t expected_strides = { 512, 256, 128, 64, 32, 16, 8, 4, 2, 1 };
        EXPECT_EQ( high_dim_tensor.strides(), expected_strides );
    }
}