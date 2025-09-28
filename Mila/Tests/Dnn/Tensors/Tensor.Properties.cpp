#include <gtest/gtest.h>
#include <vector>

import Mila;

namespace Tensors::Tests
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
        std::vector<size_t> shape2D = { 2, 3 };
        Tensor<float, Compute::HostMemoryResource> tensor2D( shape2D );
        EXPECT_EQ( tensor2D.shape(), shape2D );

        std::vector<size_t> shape3D = { 4, 5, 6 };
        Tensor<float, Compute::HostMemoryResource> tensor3D( shape3D );
        EXPECT_EQ( tensor3D.shape(), shape3D );

        std::vector<size_t> shape1D = { 10 };
        Tensor<float, Compute::HostMemoryResource> tensor1D( shape1D );
        EXPECT_EQ( tensor1D.shape(), shape1D );
    }

    TEST( TensorPropertiesTest, Shape_EmptyTensor ) {
        Tensor<float, Compute::HostMemoryResource> empty_tensor;
        EXPECT_EQ( empty_tensor.shape().size(), 0 );
        EXPECT_TRUE( empty_tensor.shape().empty() );
    }

    TEST( TensorPropertiesTest, Shape_ScalarTensor ) {
        std::vector<size_t> scalar_shape = {};
        Tensor<float, Compute::HostMemoryResource> scalar_tensor( scalar_shape );
        EXPECT_EQ( scalar_tensor.shape(), scalar_shape );
        EXPECT_TRUE( scalar_tensor.shape().empty() );
    }

    TEST( TensorPropertiesTest, Shape_LargeDimensional ) {
        std::vector<size_t> large_shape = { 2, 3, 4, 5, 6, 7 };
        Tensor<float, Compute::HostMemoryResource> large_tensor( large_shape );
        EXPECT_EQ( large_tensor.shape(), large_shape );
        EXPECT_EQ( large_tensor.shape().size(), 6 );
    }

    // ====================================================================
    // Stride Property Tests
    // ====================================================================

    TEST( TensorPropertiesTest, Strides_2D ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float, Compute::HostMemoryResource> tensor( shape );
        std::vector<size_t> expected_strides = { 3, 1 };
        EXPECT_EQ( tensor.strides(), expected_strides );
    }

    TEST( TensorPropertiesTest, Strides_3D ) {
        std::vector<size_t> shape = { 2, 3, 4 };
        Tensor<float, Compute::HostMemoryResource> tensor( shape );
        std::vector<size_t> expected_strides = { 12, 4, 1 };
        EXPECT_EQ( tensor.strides(), expected_strides );
    }

    TEST( TensorPropertiesTest, Strides_4D ) {
        std::vector<size_t> shape = { 2, 3, 4, 5 };
        Tensor<float, Compute::HostMemoryResource> tensor( shape );
        std::vector<size_t> expected_strides = { 60, 20, 5, 1 };
        EXPECT_EQ( tensor.strides(), expected_strides );
    }

    TEST( TensorPropertiesTest, Strides_1D ) {
        std::vector<size_t> shape = { 10 };
        Tensor<float, Compute::HostMemoryResource> tensor( shape );
        std::vector<size_t> expected_strides = { 1 };
        EXPECT_EQ( tensor.strides(), expected_strides );
    }

    TEST( TensorPropertiesTest, Strides_EmptyTensor ) {
        Tensor<float, Compute::HostMemoryResource> empty_tensor;
        EXPECT_TRUE( empty_tensor.strides().empty() );
    }

    TEST( TensorPropertiesTest, Strides_LargeShape ) {
        std::vector<size_t> shape = { 10, 20, 30, 40 };
        Tensor<float, Compute::HostMemoryResource> tensor( shape );
        std::vector<size_t> expected_strides = { 24000, 1200, 40, 1 };
        EXPECT_EQ( tensor.strides(), expected_strides );
    }

    // ====================================================================
    // Size Property Tests
    // ====================================================================

    TEST( TensorPropertiesTest, Size_BasicCalculations ) {
        std::vector<size_t> shape2D = { 2, 3 };
        Tensor<float, Compute::HostMemoryResource> tensor2D( shape2D );
        EXPECT_EQ( tensor2D.size(), 6 );

        std::vector<size_t> shape3D = { 2, 3, 4 };
        Tensor<float, Compute::HostMemoryResource> tensor3D( shape3D );
        EXPECT_EQ( tensor3D.size(), 24 );

        std::vector<size_t> shape1D = { 10 };
        Tensor<float, Compute::HostMemoryResource> tensor1D( shape1D );
        EXPECT_EQ( tensor1D.size(), 10 );
    }

    TEST( TensorPropertiesTest, Size_EmptyTensor ) {
        Tensor<float, Compute::HostMemoryResource> empty_tensor;
        EXPECT_EQ( empty_tensor.size(), 0 );
    }

    TEST( TensorPropertiesTest, Size_ScalarTensor ) {
        std::vector<size_t> scalar_shape = {};
        Tensor<float, Compute::HostMemoryResource> scalar_tensor( scalar_shape );
        EXPECT_EQ( scalar_tensor.size(), 0 );
    }

    TEST( TensorPropertiesTest, Size_LargeTensor ) {
        std::vector<size_t> large_shape = { 100, 200 };
        Tensor<float, Compute::HostMemoryResource> large_tensor( large_shape );
        EXPECT_EQ( large_tensor.size(), 20000 );
    }

    TEST( TensorPropertiesTest, Size_DifferentMemoryTypes ) {
        std::vector<size_t> shape = { 3, 4 };

        Tensor<float, Compute::HostMemoryResource> host_tensor( shape );
        Tensor<float, Compute::CudaMemoryResource> cuda_tensor( shape );
        Tensor<float, Compute::CudaPinnedMemoryResource> pinned_tensor( shape );
        Tensor<float, Compute::CudaManagedMemoryResource> managed_tensor( shape );

        EXPECT_EQ( host_tensor.size(), 12 );
        EXPECT_EQ( cuda_tensor.size(), 12 );
        EXPECT_EQ( pinned_tensor.size(), 12 );
        EXPECT_EQ( managed_tensor.size(), 12 );
    }

    // ====================================================================
    // Rank Property Tests
    // ====================================================================

    TEST( TensorPropertiesTest, Rank_VariousDimensions ) {
        Tensor<float, Compute::HostMemoryResource> tensor0D;
        EXPECT_EQ( tensor0D.rank(), 0 );

        std::vector<size_t> shape1D = { 5 };
        Tensor<float, Compute::HostMemoryResource> tensor1D( shape1D );
        EXPECT_EQ( tensor1D.rank(), 1 );

        std::vector<size_t> shape2D = { 3, 4 };
        Tensor<float, Compute::HostMemoryResource> tensor2D( shape2D );
        EXPECT_EQ( tensor2D.rank(), 2 );

        std::vector<size_t> shape3D = { 2, 3, 4 };
        Tensor<float, Compute::HostMemoryResource> tensor3D( shape3D );
        EXPECT_EQ( tensor3D.rank(), 3 );

        std::vector<size_t> shape4D = { 2, 3, 4, 5 };
        Tensor<float, Compute::HostMemoryResource> tensor4D( shape4D );
        EXPECT_EQ( tensor4D.rank(), 4 );
    }

    TEST( TensorPropertiesTest, Rank_HighDimensional ) {
        std::vector<size_t> high_dim_shape = { 1, 2, 3, 4, 5, 6, 7, 8 };
        Tensor<float, Compute::HostMemoryResource> high_dim_tensor( high_dim_shape );
        EXPECT_EQ( high_dim_tensor.rank(), 8 );
    }

    TEST( TensorPropertiesTest, Rank_DifferentDataTypes ) {
        std::vector<size_t> shape = { 2, 3, 4 };

        Tensor<float, Compute::HostMemoryResource> float_tensor( shape );
        Tensor<int, Compute::HostMemoryResource> int_tensor( shape );
        Tensor<uint16_t, Compute::HostMemoryResource> uint16_tensor( shape );
        Tensor<int16_t, Compute::HostMemoryResource> int16_tensor( shape );

        EXPECT_EQ( float_tensor.rank(), 3 );
        EXPECT_EQ( int_tensor.rank(), 3 );
        EXPECT_EQ( uint16_tensor.rank(), 3 );
        EXPECT_EQ( int16_tensor.rank(), 3 );
    }

    // ====================================================================
    // Empty Property Tests
    // ====================================================================

    TEST( TensorPropertiesTest, Empty_DefaultConstructor ) {
        Tensor<float, Compute::HostMemoryResource> empty_tensor;
        EXPECT_TRUE( empty_tensor.empty() );
        EXPECT_EQ( empty_tensor.size(), 0 );
        EXPECT_EQ( empty_tensor.rank(), 0 );
    }

    TEST( TensorPropertiesTest, Empty_ZeroSizeShape ) {
        std::vector<size_t> zero_shape = {};
        Tensor<float, Compute::HostMemoryResource> zero_tensor( zero_shape );
        EXPECT_TRUE( zero_tensor.empty() );
    }

    TEST( TensorPropertiesTest, Empty_NonEmptyTensors ) {
        std::vector<size_t> shape1D = { 1 };
        Tensor<float, Compute::HostMemoryResource> tensor1D( shape1D );
        EXPECT_FALSE( tensor1D.empty() );

        std::vector<size_t> shape2D = { 2, 3 };
        Tensor<float, Compute::HostMemoryResource> tensor2D( shape2D );
        EXPECT_FALSE( tensor2D.empty() );

        std::vector<size_t> shape3D = { 1, 1, 1 };
        Tensor<float, Compute::HostMemoryResource> tensor3D( shape3D );
        EXPECT_FALSE( tensor3D.empty() );
    }

    TEST( TensorPropertiesTest, Empty_DifferentMemoryTypes ) {
        Tensor<float, Compute::HostMemoryResource> host_empty;
        Tensor<float, Compute::CudaMemoryResource> cuda_empty;
        Tensor<float, Compute::CudaPinnedMemoryResource> pinned_empty;
        Tensor<float, Compute::CudaManagedMemoryResource> managed_empty;

        EXPECT_TRUE( host_empty.empty() );
        EXPECT_TRUE( cuda_empty.empty() );
        EXPECT_TRUE( pinned_empty.empty() );
        EXPECT_TRUE( managed_empty.empty() );
    }

    // ====================================================================
    // Property Consistency Tests
    // ====================================================================

    TEST( TensorPropertiesTest, PropertyConsistency_SizeRankShape ) {
        std::vector<size_t> shape = { 2, 3, 4 };
        Tensor<float, Compute::HostMemoryResource> tensor( shape );

        EXPECT_EQ( tensor.shape(), shape );
        EXPECT_EQ( tensor.rank(), shape.size() );
        EXPECT_EQ( tensor.size(), 2 * 3 * 4 );
        EXPECT_FALSE( tensor.empty() );
    }

    TEST( TensorPropertiesTest, PropertyConsistency_AfterOperations ) {
        std::vector<size_t> shape = { 3, 4 };
        Tensor<float, Compute::HostMemoryResource> tensor( shape, 1.0f );

        tensor.fill( 2.0f );
        EXPECT_EQ( tensor.shape(), shape );
        EXPECT_EQ( tensor.rank(), 2 );
        EXPECT_EQ( tensor.size(), 12 );
        EXPECT_FALSE( tensor.empty() );

        auto cloned = tensor.clone();
        EXPECT_EQ( cloned.shape(), shape );
        EXPECT_EQ( cloned.rank(), 2 );
        EXPECT_EQ( cloned.size(), 12 );
        EXPECT_FALSE( cloned.empty() );
    }

    TEST( TensorPropertiesTest, PropertyConsistency_AfterTransfer ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float, Compute::HostMemoryResource> host_tensor( shape, 1.0f );

        auto cuda_tensor = host_tensor.toDevice<Compute::CudaMemoryResource>();
        EXPECT_EQ( cuda_tensor.shape(), shape );
        EXPECT_EQ( cuda_tensor.rank(), 2 );
        EXPECT_EQ( cuda_tensor.size(), 6 );
        EXPECT_FALSE( cuda_tensor.empty() );

        auto back_to_host = cuda_tensor.toHost<Compute::HostMemoryResource>();
        EXPECT_EQ( back_to_host.shape(), shape );
        EXPECT_EQ( back_to_host.rank(), 2 );
        EXPECT_EQ( back_to_host.size(), 6 );
        EXPECT_FALSE( back_to_host.empty() );
    }

    // ====================================================================
    // Edge Cases and Special Scenarios
    // ====================================================================

    TEST( TensorPropertiesTest, EdgeCases_SingleElementTensor ) {
        std::vector<size_t> single_shape = { 1 };
        Tensor<float, Compute::HostMemoryResource> single_tensor( single_shape );

        EXPECT_EQ( single_tensor.shape(), single_shape );
        EXPECT_EQ( single_tensor.rank(), 1 );
        EXPECT_EQ( single_tensor.size(), 1 );
        EXPECT_FALSE( single_tensor.empty() );
        EXPECT_EQ( single_tensor.strides(), std::vector<size_t>{1} );
    }

    TEST( TensorPropertiesTest, EdgeCases_MultiDimensionalSingleElement ) {
        std::vector<size_t> multi_single_shape = { 1, 1, 1, 1 };
        Tensor<float, Compute::HostMemoryResource> multi_single_tensor( multi_single_shape );

        EXPECT_EQ( multi_single_tensor.shape(), multi_single_shape );
        EXPECT_EQ( multi_single_tensor.rank(), 4 );
        EXPECT_EQ( multi_single_tensor.size(), 1 );
        EXPECT_FALSE( multi_single_tensor.empty() );
        EXPECT_EQ( multi_single_tensor.strides(), std::vector<size_t>( { 1, 1, 1, 1 } ) );
    }

    TEST( TensorPropertiesTest, EdgeCases_LargeUniformShape ) {
        std::vector<size_t> uniform_shape = { 10, 10, 10, 10 };
        Tensor<float, Compute::HostMemoryResource> uniform_tensor( uniform_shape );

        EXPECT_EQ( uniform_tensor.shape(), uniform_shape );
        EXPECT_EQ( uniform_tensor.rank(), 4 );
        EXPECT_EQ( uniform_tensor.size(), 10000 );
        EXPECT_FALSE( uniform_tensor.empty() );
        EXPECT_EQ( uniform_tensor.strides(), std::vector<size_t>( { 1000, 100, 10, 1 } ) );
    }

    TEST( TensorPropertiesTest, EdgeCases_AsymmetricShape ) {
        std::vector<size_t> asymmetric_shape = { 1, 100, 1, 50 };
        Tensor<float, Compute::HostMemoryResource> asymmetric_tensor( asymmetric_shape );

        EXPECT_EQ( asymmetric_tensor.shape(), asymmetric_shape );
        EXPECT_EQ( asymmetric_tensor.rank(), 4 );
        EXPECT_EQ( asymmetric_tensor.size(), 5000 );
        EXPECT_FALSE( asymmetric_tensor.empty() );
        EXPECT_EQ( asymmetric_tensor.strides(), std::vector<size_t>( { 5000, 50, 50, 1 } ) );
    }

    // ====================================================================
    // Property Validation with Different Data Types
    // ====================================================================

    TEST( TensorPropertiesTest, DataTypes_PropertyConsistency ) {
        std::vector<size_t> shape = { 3, 4 };

        Tensor<float, Compute::HostMemoryResource> float_tensor( shape );
        Tensor<int, Compute::HostMemoryResource> int_tensor( shape );
        Tensor<uint16_t, Compute::HostMemoryResource> uint16_tensor( shape );
        Tensor<int16_t, Compute::HostMemoryResource> int16_tensor( shape );
        Tensor<uint32_t, Compute::HostMemoryResource> uint32_tensor( shape );

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

        std::vector<size_t> expected_strides = { 4, 1 };
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
        std::vector<size_t> large_shape = { 1000, 1000 };
        Tensor<float, Compute::HostMemoryResource> large_tensor( large_shape );

        EXPECT_EQ( large_tensor.shape(), large_shape );
        EXPECT_EQ( large_tensor.rank(), 2 );
        EXPECT_EQ( large_tensor.size(), 1000000 );
        EXPECT_FALSE( large_tensor.empty() );
        EXPECT_EQ( large_tensor.strides(), std::vector<size_t>( { 1000, 1 } ) );
    }

    TEST( TensorPropertiesTest, Performance_HighDimensionalTensor ) {
        std::vector<size_t> high_dim_shape = { 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 };
        Tensor<float, Compute::HostMemoryResource> high_dim_tensor( high_dim_shape );

        EXPECT_EQ( high_dim_tensor.shape(), high_dim_shape );
        EXPECT_EQ( high_dim_tensor.rank(), 10 );
        EXPECT_EQ( high_dim_tensor.size(), 1024 );
        EXPECT_FALSE( high_dim_tensor.empty() );

        std::vector<size_t> expected_strides = { 512, 256, 128, 64, 32, 16, 8, 4, 2, 1 };
        EXPECT_EQ( high_dim_tensor.strides(), expected_strides );
    }
}