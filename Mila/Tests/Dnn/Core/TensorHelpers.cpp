#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <cmath>
#include <cuda_fp16.h>

import Mila;

namespace Core::Tests
{
    using namespace Mila::Dnn;

    // Helper for half precision tests to convert half to float for comparisons
    float half_to_float( half h ) {
        return __half2float( h );
    }

    // Random Initialization Tests
    //----------------------------------------------------------------------------------------
    TEST( TensorHelpersTests, Random_Float_CPU ) {
        std::vector<size_t> shape = { 100, 100 };
        Tensor<float, Compute::HostMemoryResource> tensor( shape );

        float min_val = -1.0f;
        float max_val = 1.0f;

        random( tensor, min_val, max_val );

        // Check values are within range
        bool all_in_range = true;
        float actual_min = std::numeric_limits<float>::max();
        float actual_max = std::numeric_limits<float>::lowest();

        for ( size_t i = 0; i < tensor.shape()[ 0 ]; ++i ) {
            for ( size_t j = 0; j < tensor.shape()[ 1 ]; ++j ) {
                float val = tensor[ i, j ];
                if ( val < min_val || val > max_val ) {
                    all_in_range = false;
                }
                actual_min = std::min( actual_min, val );
                actual_max = std::max( actual_max, val );
            }
        }

        EXPECT_TRUE( all_in_range ) << "All values should be in range [" << min_val << ", " << max_val << "]";

        // Verify we're actually getting a range of values (not all the same)
        EXPECT_LT( actual_min, actual_max );

        // With enough samples, we should get close to the min/max bounds
        EXPECT_LT( actual_min, min_val + 0.2f );
        EXPECT_GT( actual_max, max_val - 0.2f );
    }

    TEST( TensorHelpersTests, Random_Float_CUDA ) {
        std::vector<size_t> shape = { 100, 100 };
        Tensor<float, Compute::CudaMemoryResource> tensor( shape );

        float min_val = -2.5f;
        float max_val = 2.5f;

        random( tensor, min_val, max_val );

        // Convert to CPU to check values
        auto cpu_tensor = tensor.to<Compute::HostMemoryResource>();

        // Check values are within range
        bool all_in_range = true;
        float actual_min = std::numeric_limits<float>::max();
        float actual_max = std::numeric_limits<float>::lowest();

        for ( size_t i = 0; i < cpu_tensor.shape()[ 0 ]; ++i ) {
            for ( size_t j = 0; j < cpu_tensor.shape()[ 1 ]; ++j ) {
                float val = cpu_tensor[ i, j ];
                if ( val < min_val || val > max_val ) {
                    all_in_range = false;
                }
                actual_min = std::min( actual_min, val );
                actual_max = std::max( actual_max, val );
            }
        }

        EXPECT_TRUE( all_in_range ) << "All values should be in range [" << min_val << ", " << max_val << "]";

        // Verify we're actually getting a range of values (not all the same)
        EXPECT_LT( actual_min, actual_max );

        // With enough samples, we should get close to the min/max bounds
        EXPECT_LT( actual_min, min_val + 0.3f );
        EXPECT_GT( actual_max, max_val - 0.3f );
    }

    TEST( TensorHelpersTests, Random_Half_CPU ) {
        std::vector<size_t> shape = { 100, 100 };
        Tensor<half, Compute::HostMemoryResource> tensor( shape );

        half min_val = __float2half( -1.0f );
        half max_val = __float2half( 1.0f );

        random( tensor, min_val, max_val );

        // Check values are within range
        bool all_in_range = true;
        float actual_min = std::numeric_limits<float>::max();
        float actual_max = std::numeric_limits<float>::lowest();

        for ( size_t i = 0; i < tensor.shape()[ 0 ]; ++i ) {
            for ( size_t j = 0; j < tensor.shape()[ 1 ]; ++j ) {
                float val = half_to_float( tensor[ i, j ] );
                if ( val < -1.0f || val > 1.0f ) {
                    all_in_range = false;
                }
                actual_min = std::min( actual_min, val );
                actual_max = std::max( actual_max, val );
            }
        }

        EXPECT_TRUE( all_in_range ) << "All values should be in range [-1.0, 1.0]";

        // Verify we're actually getting a range of values (not all the same)
        EXPECT_LT( actual_min, actual_max );

        // With enough samples, we should get close to the min/max bounds
        // Half precision has less precision, so we use a slightly wider margin
        EXPECT_LT( actual_min, -0.8f );
        EXPECT_GT( actual_max, 0.8f );
    }

    TEST( TensorHelpersTests, Random_Half_CUDA ) {
        std::vector<size_t> shape = { 100, 100 };
        Tensor<half, Compute::CudaMemoryResource> tensor( shape );

        half min_val = __float2half( -2.5f );
        half max_val = __float2half( 2.5f );

        random( tensor, min_val, max_val );

        // Convert to CPU to check values
        auto cpu_tensor = tensor.to<Compute::HostMemoryResource>();

        // Check values are within range
        bool all_in_range = true;
        float actual_min = std::numeric_limits<float>::max();
        float actual_max = std::numeric_limits<float>::lowest();

        for ( size_t i = 0; i < cpu_tensor.shape()[ 0 ]; ++i ) {
            for ( size_t j = 0; j < cpu_tensor.shape()[ 1 ]; ++j ) {
                float val = half_to_float( cpu_tensor[ i, j ] );
                if ( val < -2.5f || val > 2.5f ) {
                    all_in_range = false;
                }
                actual_min = std::min( actual_min, val );
                actual_max = std::max( actual_max, val );
            }
        }

        EXPECT_TRUE( all_in_range ) << "All values should be in range [-2.5, 2.5]";

        // Verify we're actually getting a range of values (not all the same)
        EXPECT_LT( actual_min, actual_max );

        // With enough samples, we should get close to the min/max bounds
        // Half precision has less precision, so we use a wider margin
        EXPECT_LT( actual_min, -2.0f );
        EXPECT_GT( actual_max, 2.0f );
    }

    // Xavier Initialization Tests
    //----------------------------------------------------------------------------------------
    TEST( TensorHelpersTests, Xavier_Float_CPU ) {
        std::vector<size_t> shape = { 100, 100 };
        Tensor<float, Compute::HostMemoryResource> tensor( shape );

        size_t input_size = 784;  // Typical MNIST input size
        size_t output_size = 10;  // Typical MNIST output size

        xavier( tensor, input_size, output_size );

        // Calculate expected limit for Xavier initialization
        float limit = std::sqrt( 6.0f / (input_size + output_size) );

        // Check values are within range
        bool all_in_range = true;
        float actual_min = std::numeric_limits<float>::max();
        float actual_max = std::numeric_limits<float>::lowest();

        for ( size_t i = 0; i < tensor.shape()[ 0 ]; ++i ) {
            for ( size_t j = 0; j < tensor.shape()[ 1 ]; ++j ) {
                float val = tensor[ i, j ];
                if ( val < -limit || val > limit ) {
                    all_in_range = false;
                }
                actual_min = std::min( actual_min, val );
                actual_max = std::max( actual_max, val );
            }
        }

        EXPECT_TRUE( all_in_range ) << "All values should be in range [-" << limit << ", " << limit << "]";

        // Verify we're actually getting a range of values (not all the same)
        EXPECT_LT( actual_min, actual_max );

        // With enough samples, we should get reasonably close to the limits
        EXPECT_LT( actual_min, -0.6f * limit );
        EXPECT_GT( actual_max, 0.6f * limit );
    }

    TEST( TensorHelpersTests, Xavier_Float_CUDA ) {
        std::vector<size_t> shape = { 100, 100 };
        Tensor<float, Compute::CudaMemoryResource> tensor( shape );

        size_t input_size = 1024;
        size_t output_size = 512;

        xavier( tensor, input_size, output_size );

        // Calculate expected limit for Xavier initialization
        float limit = std::sqrt( 6.0f / (input_size + output_size) );

        // Convert to CPU to check values
        auto cpu_tensor = tensor.to<Compute::HostMemoryResource>();

        // Check values are within range
        bool all_in_range = true;
        float actual_min = std::numeric_limits<float>::max();
        float actual_max = std::numeric_limits<float>::lowest();

        for ( size_t i = 0; i < cpu_tensor.shape()[ 0 ]; ++i ) {
            for ( size_t j = 0; j < cpu_tensor.shape()[ 1 ]; ++j ) {
                float val = cpu_tensor[ i, j ];
                if ( val < -limit || val > limit ) {
                    all_in_range = false;
                }
                actual_min = std::min( actual_min, val );
                actual_max = std::max( actual_max, val );
            }
        }

        EXPECT_TRUE( all_in_range ) << "All values should be in range [-" << limit << ", " << limit << "]";

        // Verify we're actually getting a range of values (not all the same)
        EXPECT_LT( actual_min, actual_max );

        // With enough samples, we should get reasonably close to the limits
        EXPECT_LT( actual_min, -0.6f * limit );
        EXPECT_GT( actual_max, 0.6f * limit );
    }

    TEST( TensorHelpersTests, Xavier_Half_CPU ) {
        std::vector<size_t> shape = { 100, 100 };
        Tensor<half, Compute::HostMemoryResource> tensor( shape );

        size_t input_size = 784;
        size_t output_size = 10;

        xavier( tensor, input_size, output_size );

        // Calculate expected limit for Xavier initialization
        float limit = std::sqrt( 6.0f / (input_size + output_size) );

        // Check values are within range
        bool all_in_range = true;
        float actual_min = std::numeric_limits<float>::max();
        float actual_max = std::numeric_limits<float>::lowest();

        for ( size_t i = 0; i < tensor.shape()[ 0 ]; ++i ) {
            for ( size_t j = 0; j < tensor.shape()[ 1 ]; ++j ) {
                float val = half_to_float( tensor[ i, j ] );
                if ( val < -limit || val > limit ) {
                    all_in_range = false;
                }
                actual_min = std::min( actual_min, val );
                actual_max = std::max( actual_max, val );
            }
        }

        EXPECT_TRUE( all_in_range ) << "All values should be in range [-" << limit << ", " << limit << "]";

        // Verify we're actually getting a range of values (not all the same)
        EXPECT_LT( actual_min, actual_max );

        // With enough samples and half precision, we should get reasonably close to the limits
        // Using a wider margin due to half precision
        EXPECT_LT( actual_min, -0.5f * limit );
        EXPECT_GT( actual_max, 0.5f * limit );
    }

    TEST( TensorHelpersTests, Xavier_Half_CUDA ) {
        std::vector<size_t> shape = { 100, 100 };
        Tensor<half, Compute::CudaMemoryResource> tensor( shape );

        size_t input_size = 1024;
        size_t output_size = 512;

        xavier( tensor, input_size, output_size );

        // Calculate expected limit for Xavier initialization
        float limit = std::sqrt( 6.0f / (input_size + output_size) );

        // Convert to CPU to check values
        auto cpu_tensor = tensor.to<Compute::HostMemoryResource>();

        // Check values are within range
        bool all_in_range = true;
        float actual_min = std::numeric_limits<float>::max();
        float actual_max = std::numeric_limits<float>::lowest();

        for ( size_t i = 0; i < cpu_tensor.shape()[ 0 ]; ++i ) {
            for ( size_t j = 0; j < cpu_tensor.shape()[ 1 ]; ++j ) {
                float val = half_to_float( cpu_tensor[ i, j ] );
                if ( val < -limit || val > limit ) {
                    all_in_range = false;
                }
                actual_min = std::min( actual_min, val );
                actual_max = std::max( actual_max, val );
            }
        }

        EXPECT_TRUE( all_in_range ) << "All values should be in range [-" << limit << ", " << limit << "]";

        // Verify we're actually getting a range of values (not all the same)
        EXPECT_LT( actual_min, actual_max );

        // With enough samples and half precision, we should get reasonably close to the limits
        // Using a wider margin due to half precision
        EXPECT_LT( actual_min, -0.5f * limit );
        EXPECT_GT( actual_max, 0.5f * limit );
    }
}