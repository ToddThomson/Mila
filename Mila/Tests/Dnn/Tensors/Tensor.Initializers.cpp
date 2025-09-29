#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <cmath>
#include <limits>

import Mila;

namespace Dnn::Tensors::Tests
{
    using namespace Mila::Dnn;

    // ====================================================================
    // Random Initialization Tests - CPU Compute Device
    // ====================================================================

    TEST( TensorInitializersTests, Random_FP32_CPU ) {
        std::vector<size_t> shape = { 100, 100 };
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> tensor( "CPU", shape );

        float min_val = -1.0f;
        float max_val = 1.0f;

        random( tensor, min_val, max_val );

        // Check values are within range
        bool all_in_range = true;
        float actual_min = std::numeric_limits<float>::max();
        float actual_max = std::numeric_limits<float>::lowest();

        for (size_t i = 0; i < tensor.shape()[0]; ++i) {
            for (size_t j = 0; j < tensor.shape()[1]; ++j) {
                auto val = tensor.at( { i, j } );
                if (val < min_val || val > max_val) {
                    all_in_range = false;
                }
                actual_min = std::min( actual_min, static_cast<float>( val ) );
                actual_max = std::max( actual_max, static_cast<float>( val ) );
            }
        }

        EXPECT_TRUE( all_in_range ) << "All values should be in range [" << min_val << ", " << max_val << "]";
        EXPECT_LT( actual_min, actual_max );
        EXPECT_LT( actual_min, min_val + 0.2f );
        EXPECT_GT( actual_max, max_val - 0.2f );
    }

    TEST( TensorInitializersTests, Random_INT32_CPU ) {
        std::vector<size_t> shape = { 50, 50 };
        Tensor<TensorDataType::INT32, Compute::CpuMemoryResource> tensor( "CPU", shape );

        int32_t min_val = -100;
        int32_t max_val = 100;

        random( tensor, min_val, max_val );

        // Verify all values are within integer range
        bool all_in_range = true;
        for (size_t i = 0; i < tensor.shape()[0]; ++i) {
            for (size_t j = 0; j < tensor.shape()[1]; ++j) {
                auto val = tensor.at( { i, j } );
                if (val < min_val || val > max_val) {
                    all_in_range = false;
                }
            }
        }

        EXPECT_TRUE( all_in_range );
    }

    TEST( TensorInitializersTests, Random_UINT32_CPU ) {
        std::vector<size_t> shape = { 30, 30 };
        Tensor<TensorDataType::UINT32, Compute::CpuMemoryResource> tensor( "CPU", shape );

        int32_t min_val = 0;
        int32_t max_val = 1000;

        random( tensor, min_val, max_val );

        // Verify all values are positive and within range
        bool all_in_range = true;
        for (size_t i = 0; i < tensor.shape()[0]; ++i) {
            for (size_t j = 0; j < tensor.shape()[1]; ++j) {
                auto val = tensor.at( { i, j } );
                if (val < 0 || val < static_cast<uint32_t>( min_val ) || val > static_cast<uint32_t>( max_val )) {
                    all_in_range = false;
                }
            }
        }

        EXPECT_TRUE( all_in_range );
    }

    // ====================================================================
    // Random Initialization Tests - CUDA Backend (CPU-only until CUDA backend ready)
    // ====================================================================

    TEST( TensorInitializersTests, Random_FP32_CUDA ) {
        std::vector<size_t> shape = { 100, 100 };

        // Skip if CUDA not available
        try {
            Tensor<TensorDataType::FP32, Compute::CudaMemoryResource> dev_tensor( "CUDA:0", shape );

            float min_val = -2.5f;
            float max_val = 2.5f;

            random( dev_tensor, min_val, max_val );

            // Transfer to CPU for validation
            auto cpu_tensor = dev_tensor.to<TensorDataType::FP32, Compute::CpuMemoryResource>();

            // Check values are within range
            bool all_in_range = true;
            float actual_min = std::numeric_limits<float>::max();
            float actual_max = std::numeric_limits<float>::lowest();

            for (size_t i = 0; i < cpu_tensor.shape()[0]; ++i) {
                for (size_t j = 0; j < cpu_tensor.shape()[1]; ++j) {
                    auto val = cpu_tensor.at( { i, j } );
                    if (val < min_val || val > max_val) {
                        all_in_range = false;
                    }
                    actual_min = std::min( actual_min, val );
                    actual_max = std::max( actual_max, val );
                }
            }

            EXPECT_TRUE( all_in_range ) << "All values should be in range [" << min_val << ", " << max_val << "]";
            EXPECT_LT( actual_min, actual_max );
            EXPECT_LT( actual_min, min_val + 0.3f );
            EXPECT_GT( actual_max, max_val - 0.3f );
        }
        catch (const std::exception&) {
            GTEST_SKIP() << "CUDA device not available, skipping CUDA test";
        }
    }

    TEST( TensorInitializersTests, Random_INT32_CUDA ) {
        std::vector<size_t> shape = { 60, 60 };

        try {
            Tensor<TensorDataType::INT32, Compute::CudaMemoryResource> dev_tensor( "CUDA:0", shape );

            int32_t min_val = -500;
            int32_t max_val = 500;

            random( dev_tensor, min_val, max_val );

            auto cpu_tensor = dev_tensor.to<TensorDataType::INT32, Compute::CpuMemoryResource>();

            bool all_in_range = true;
            for (size_t i = 0; i < cpu_tensor.shape()[0]; ++i) {
                for (size_t j = 0; j < cpu_tensor.shape()[1]; ++j) {
                    auto val = cpu_tensor.at( { i, j } );
                    if (val < min_val || val > max_val) {
                        all_in_range = false;
                    }
                }
            }

            EXPECT_TRUE( all_in_range );
        }
        catch (const std::exception&) {
            GTEST_SKIP() << "CUDA device not available, skipping CUDA test";
        }
    }

    // ====================================================================
    // Random Initialization Tests - Managed Memory
    // ====================================================================

    TEST( TensorInitializersTests, Random_FP32_ManagedMemory ) {
        try {
            std::vector<size_t> shape = { 50, 50 };
            Tensor<TensorDataType::FP32, Compute::CudaManagedMemoryResource> tensor( "CUDA:0", shape );

            float min_val = -1.0f;
            float max_val = 1.0f;

            random( tensor, min_val, max_val );

            // Managed memory allows direct host access
            bool all_in_range = true;
            for (size_t i = 0; i < tensor.shape()[0]; ++i) {
                for (size_t j = 0; j < tensor.shape()[1]; ++j) {
                    auto val = tensor.at( { i, j } );
                    if (val < min_val || val > max_val) {
                        all_in_range = false;
                    }
                }
            }

            EXPECT_TRUE( all_in_range );
            EXPECT_TRUE( tensor.is_host_accessible() );
            EXPECT_TRUE( tensor.is_device_accessible() );
        }
        catch (const std::exception&) {
            GTEST_SKIP() << "CUDA managed memory not available, skipping test";
        }
    }

    // ====================================================================
    // Xavier Initialization Tests - CPU Backend
    // ====================================================================

    TEST( TensorInitializersTests, Xavier_FP32_CPU ) {
        std::vector<size_t> shape = { 100, 100 };
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> tensor( "CPU", shape );

        size_t input_size = 784;   // Typical MNIST input size
        size_t output_size = 10;   // Typical MNIST output size

        xavier( tensor, input_size, output_size );

        // Calculate expected limit for Xavier initialization
        float limit = std::sqrt( 6.0f / (input_size + output_size) );

        // Check values are within range
        bool all_in_range = true;
        float actual_min = std::numeric_limits<float>::max();
        float actual_max = std::numeric_limits<float>::lowest();

        for (size_t i = 0; i < tensor.shape()[0]; ++i) {
            for (size_t j = 0; j < tensor.shape()[1]; ++j) {
                auto val = tensor.at( { i, j } );
                if (val < -limit || val > limit) {
                    all_in_range = false;
                }
                actual_min = std::min( actual_min, val );
                actual_max = std::max( actual_max, val );
            }
        }

        EXPECT_TRUE( all_in_range ) << "All values should be in range [-" << limit << ", " << limit << "]";
        EXPECT_LT( actual_min, actual_max );
        EXPECT_LT( actual_min, -0.6f * limit );
        EXPECT_GT( actual_max, 0.6f * limit );
    }

    TEST( TensorInitializersTests, Xavier_INT32_CPU ) {
        std::vector<size_t> shape = { 50, 50 };
        Tensor<TensorDataType::INT32, Compute::CpuMemoryResource> tensor( "CPU", shape );

        size_t input_size = 100;
        size_t output_size = 50;

        xavier( tensor, input_size, output_size );

        float limit = std::sqrt( 6.0f / (input_size + output_size) );
        int32_t int_limit = std::max( static_cast<int32_t>(std::round( limit )), 1 );

        bool all_in_range = true;
        for (size_t i = 0; i < tensor.shape()[0]; ++i) {
            for (size_t j = 0; j < tensor.shape()[1]; ++j) {
                auto val = tensor.at( { i, j } );
                if (val < -int_limit || val > int_limit) {
                    all_in_range = false;
                }
            }
        }

        EXPECT_TRUE( all_in_range );
    }

    // ====================================================================
    // Xavier Initialization Tests - CUDA Backend
    // ====================================================================

    TEST( TensorInitializersTests, Xavier_FP32_CUDA ) {
        try {
            std::vector<size_t> shape = { 100, 100 };
            Tensor<TensorDataType::FP32, Compute::CudaMemoryResource> tensor( "CUDA:0", shape );

            size_t input_size = 1024;
            size_t output_size = 512;

            xavier( tensor, input_size, output_size );

            // Calculate expected limit for Xavier initialization
            float limit = std::sqrt( 6.0f / (input_size + output_size) );

            // Transfer to CPU for validation
            auto cpu_tensor = tensor.to<TensorDataType::FP32, Compute::CpuMemoryResource>();

            // Check values are within range
            bool all_in_range = true;
            float actual_min = std::numeric_limits<float>::max();
            float actual_max = std::numeric_limits<float>::lowest();

            for (size_t i = 0; i < cpu_tensor.shape()[0]; ++i) {
                for (size_t j = 0; j < cpu_tensor.shape()[1]; ++j) {
                    auto val = cpu_tensor.at( { i, j } );
                    if (val < -limit || val > limit) {
                        all_in_range = false;
                    }
                    actual_min = std::min( actual_min, val );
                    actual_max = std::max( actual_max, val );
                }
            }

            EXPECT_TRUE( all_in_range ) << "All values should be in range [-" << limit << ", " << limit << "]";
            EXPECT_LT( actual_min, actual_max );
            EXPECT_LT( actual_min, -0.6f * limit );
            EXPECT_GT( actual_max, 0.6f * limit );
        }
        catch (const std::exception&) {
            GTEST_SKIP() << "CUDA device not available, skipping CUDA test";
        }
    }

    TEST( TensorInitializersTests, Xavier_ExtremeInputOutputSizes ) {
        std::vector<size_t> shape = { 20, 20 };
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> tensor( "CPU", shape );

        // Very large input size, small output size
        size_t input_size = 10000;
        size_t output_size = 1;

        xavier( tensor, input_size, output_size );

        float limit = std::sqrt( 6.0f / (input_size + output_size) );
        EXPECT_LT( limit, 0.1f ); // Should be very small

        bool all_in_range = true;
        for (size_t i = 0; i < tensor.shape()[0]; ++i) {
            for (size_t j = 0; j < tensor.shape()[1]; ++j) {
                auto val = tensor.at( { i, j } );
                if (val < -limit || val > limit) {
                    all_in_range = false;
                }
            }
        }

        EXPECT_TRUE( all_in_range );
    }

    // ====================================================================
    // Basic Initialization Tests - CPU Backend
    // ====================================================================

    TEST( TensorInitializersTests, Zeros_FP32_CPU ) {
        std::vector<size_t> shape = { 50, 50 };
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> tensor( "CPU", shape );

        zeros( tensor );

        // Verify all values are zero
        for (size_t i = 0; i < tensor.shape()[0]; ++i) {
            for (size_t j = 0; j < tensor.shape()[1]; ++j) {
                auto val = tensor.at( { i, j } );
                EXPECT_FLOAT_EQ( val, 0.0f );
            }
        }
    }

    TEST( TensorInitializersTests, Zeros_INT32_CPU ) {
        std::vector<size_t> shape = { 30, 30 };
        Tensor<TensorDataType::INT32, Compute::CpuMemoryResource> tensor( "CPU", shape );

        zeros( tensor );

        for (size_t i = 0; i < tensor.shape()[0]; ++i) {
            for (size_t j = 0; j < tensor.shape()[1]; ++j) {
                auto val = tensor.at( { i, j } );
                EXPECT_EQ( val, 0 );
            }
        }
    }

    TEST( TensorInitializersTests, Ones_FP32_CPU ) {
        std::vector<size_t> shape = { 50, 50 };
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> tensor( "CPU", shape );

        ones( tensor );

        // Verify all values are one
        for (size_t i = 0; i < tensor.shape()[0]; ++i) {
            for (size_t j = 0; j < tensor.shape()[1]; ++j) {
                auto val = tensor.at( { i, j } );
                EXPECT_FLOAT_EQ( val, 1.0f );
            }
        }
    }

    TEST( TensorInitializersTests, Ones_UINT32_CPU ) {
        std::vector<size_t> shape = { 20, 20 };
        Tensor<TensorDataType::UINT32, Compute::CpuMemoryResource> tensor( "CPU", shape );

        ones( tensor );

        for (size_t i = 0; i < tensor.shape()[0]; ++i) {
            for (size_t j = 0; j < tensor.shape()[1]; ++j) {
                auto val = tensor.at( { i, j } );
                EXPECT_EQ( val, 1u );
            }
        }
    }

    TEST( TensorInitializersTests, Fill_FP32_CPU ) {
        std::vector<size_t> shape = { 50, 50 };
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> tensor( "CPU", shape );

        float fill_value = 3.14f;
        fill( tensor, fill_value );

        // Verify all values match fill value
        for (size_t i = 0; i < tensor.shape()[0]; ++i) {
            for (size_t j = 0; j < tensor.shape()[1]; ++j) {
                auto val = tensor.at( { i, j } );
                EXPECT_FLOAT_EQ( val, fill_value );
            }
        }
    }

    TEST( TensorInitializersTests, Fill_NegativeValue_CPU ) {
        std::vector<size_t> shape = { 30, 30 };
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> tensor( "CPU", shape );

        float fill_value = -2.718f;
        fill( tensor, fill_value );

        for (size_t i = 0; i < tensor.shape()[0]; ++i) {
            for (size_t j = 0; j < tensor.shape()[1]; ++j) {
                auto val = tensor.at( { i, j } );
                EXPECT_FLOAT_EQ( val, fill_value );
            }
        }
    }

    // ====================================================================
    // Basic Initialization Tests - CUDA Backend
    // ====================================================================

    TEST( TensorInitializersTests, Zeros_FP32_CUDA ) {
        try {
            std::vector<size_t> shape = { 50, 50 };
            Tensor<TensorDataType::FP32, Compute::CudaMemoryResource> tensor( "CUDA:0", shape );

            zeros( tensor );

            // Transfer to CPU for validation
            auto cpu_tensor = tensor.to<TensorDataType::FP32, Compute::CpuMemoryResource>();

            // Verify all values are zero
            for (size_t i = 0; i < cpu_tensor.shape()[0]; ++i) {
                for (size_t j = 0; j < cpu_tensor.shape()[1]; ++j) {
                    auto val = cpu_tensor.at( { i, j } );
                    EXPECT_FLOAT_EQ( val, 0.0f );
                }
            }
        }
        catch (const std::exception&) {
            GTEST_SKIP() << "CUDA device not available, skipping CUDA test";
        }
    }

    TEST( TensorInitializersTests, Ones_FP32_CUDA ) {
        try {
            std::vector<size_t> shape = { 50, 50 };
            Tensor<TensorDataType::FP32, Compute::CudaMemoryResource> tensor( "CUDA:0", shape );

            ones( tensor );

            // Transfer to CPU for validation
            auto cpu_tensor = tensor.to<TensorDataType::FP32, Compute::CpuMemoryResource>();

            // Verify all values are one
            for (size_t i = 0; i < cpu_tensor.shape()[0]; ++i) {
                for (size_t j = 0; j < cpu_tensor.shape()[1]; ++j) {
                    auto val = cpu_tensor.at( { i, j } );
                    EXPECT_FLOAT_EQ( val, 1.0f );
                }
            }
        }
        catch (const std::exception&) {
            GTEST_SKIP() << "CUDA device not available, skipping CUDA test";
        }
    }

    TEST( TensorInitializersTests, Fill_FP32_CUDA ) {
        try {
            std::vector<size_t> shape = { 50, 50 };
            Tensor<TensorDataType::FP32, Compute::CudaMemoryResource> tensor( "CUDA:0", shape );

            float fill_value = 2.71f;
            fill( tensor, fill_value );

            // Transfer to CPU for validation
            auto cpu_tensor = tensor.to<TensorDataType::FP32, Compute::CpuMemoryResource>();

            // Verify all values match fill value
            for (size_t i = 0; i < cpu_tensor.shape()[0]; ++i) {
                for (size_t j = 0; j < cpu_tensor.shape()[1]; ++j) {
                    auto val = cpu_tensor.at( { i, j } );
                    EXPECT_FLOAT_EQ( val, fill_value );
                }
            }
        }
        catch (const std::exception&) {
            GTEST_SKIP() << "CUDA device not available, skipping CUDA test";
        }
    }

    TEST( TensorInitializersTests, Fill_INT32_CUDA ) {
        try {
            std::vector<size_t> shape = { 40, 40 };
            Tensor<TensorDataType::INT32, Compute::CudaMemoryResource> tensor( "CUDA:0", shape );

            int32_t fill_value = 42;
            fill( tensor, fill_value );

            auto cpu_tensor = tensor.to<TensorDataType::INT32, Compute::CpuMemoryResource>();

            for (size_t i = 0; i < cpu_tensor.shape()[0]; ++i) {
                for (size_t j = 0; j < cpu_tensor.shape()[1]; ++j) {
                    auto val = cpu_tensor.at( { i, j } );
                    EXPECT_EQ( val, 42 );
                }
            }
        }
        catch (const std::exception&) {
            GTEST_SKIP() << "CUDA device not available, skipping CUDA test";
        }
    }

    // ====================================================================
    // Edge Case Tests
    // ====================================================================

    TEST( TensorInitializersTests, InitializeEmptyTensor ) {
        // Create default-constructed empty tensor
        auto device_context = Compute::DeviceContext::create( "CPU" );
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> empty_tensor( device_context, {} );
        EXPECT_TRUE( empty_tensor.empty() );

        // These operations should not crash on empty tensors
        EXPECT_NO_THROW( zeros( empty_tensor ) );
        EXPECT_NO_THROW( ones( empty_tensor ) );
        EXPECT_NO_THROW( fill( empty_tensor, 5.0f ) );
        EXPECT_NO_THROW( random( empty_tensor, -1.0f, 1.0f ) );
        EXPECT_NO_THROW( xavier( empty_tensor, 100, 50 ) );

        EXPECT_TRUE( empty_tensor.empty() );
    }

    TEST( TensorInitializersTests, InitializeSingleElement ) {
        std::vector<size_t> shape = { 1 };
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> single_tensor( "CPU", shape );

        float fill_value = 7.5f;
        fill( single_tensor, fill_value );

        auto val = single_tensor.at( { 0 } );
        EXPECT_FLOAT_EQ( val, fill_value );

        // Test with random
        random( single_tensor, -10.0f, 10.0f );
        val = single_tensor.at( { 0 } );
        EXPECT_GE( val, -10.0f );
        EXPECT_LE( val, 10.0f );
    }

    TEST( TensorInitializersTests, Initialize1DTensor ) {
        std::vector<size_t> shape = { 100 };
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> tensor_1d( "CPU", shape );

        zeros( tensor_1d );

        for (size_t i = 0; i < tensor_1d.size(); ++i) {
            auto val = tensor_1d.at( { i } );
            EXPECT_FLOAT_EQ( val, 0.0f );
        }
    }

    TEST( TensorInitializersTests, Initialize3DTensor ) {
        std::vector<size_t> shape = { 10, 20, 30 };
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> tensor_3d( "CPU", shape );

        float fill_value = 1.23f;
        fill( tensor_3d, fill_value );

        // Test a few representative elements
        for (size_t i = 0; i < 5; ++i) {
            for (size_t j = 0; j < 5; ++j) {
                for (size_t k = 0; k < 5; ++k) {
                    auto val = tensor_3d.at( { i, j, k } );
                    EXPECT_FLOAT_EQ( val, fill_value );
                }
            }
        }
    }

    TEST( TensorInitializersTests, InitializeLargeTensor ) {
        std::vector<size_t> shape = { 500, 500 };
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> large_tensor( "CPU", shape );

        // Test that initialization doesn't fail with large tensors
        EXPECT_NO_THROW( zeros( large_tensor ) );
        EXPECT_NO_THROW( random( large_tensor, -1.0f, 1.0f ) );

        // Reset to zeros and spot check a few values
        zeros( large_tensor );
        EXPECT_FLOAT_EQ( large_tensor.at( { 0, 0 } ), 0.0f );
        EXPECT_FLOAT_EQ( large_tensor.at( { 499, 499 } ), 0.0f );
        EXPECT_FLOAT_EQ( large_tensor.at( { 250, 250 } ), 0.0f );
    }

    // ====================================================================
    // Parameter Validation Tests
    // ====================================================================

    TEST( TensorInitializersTests, Random_InvalidRange ) {
        std::vector<size_t> shape = { 10, 10 };
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> tensor( "CPU", shape );

        // Should handle min_val > max_val gracefully
        float min_val = 5.0f;
        float max_val = -5.0f;

        // The implementation should handle this case (may swap or throw)
        EXPECT_NO_THROW( random( tensor, min_val, max_val ) );
    }

    TEST( TensorInitializersTests, Random_EqualMinMax ) {
        std::vector<size_t> shape = { 20, 20 };
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> tensor( "CPU", shape );

        float value = 3.14f;
        random( tensor, value, value );

        // All values should be approximately equal to the single value
        for (size_t i = 0; i < tensor.shape()[0]; ++i) {
            for (size_t j = 0; j < tensor.shape()[1]; ++j) {
                auto val = tensor.at( { i, j } );
                EXPECT_NEAR( val, value, 0.01f );
            }
        }
    }

    TEST( TensorInitializersTests, Xavier_ZeroInputSize ) {
        std::vector<size_t> shape = { 10, 10 };
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> tensor( "CPU", shape );

        // Xavier with zero input size should not crash
        EXPECT_NO_THROW( xavier( tensor, 0, 100 ) );
    }

    TEST( TensorInitializersTests, Xavier_ZeroOutputSize ) {
        std::vector<size_t> shape = { 10, 10 };
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> tensor( "CPU", shape );

        // Xavier with zero output size should not crash
        EXPECT_NO_THROW( xavier( tensor, 100, 0 ) );
    }

    TEST( TensorInitializersTests, Xavier_BothZeroSizes ) {
        std::vector<size_t> shape = { 10, 10 };
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> tensor( "CPU", shape );

        // Xavier with both zero sizes - implementation should handle gracefully
        EXPECT_NO_THROW( xavier( tensor, 0, 0 ) );
    }

    // ====================================================================
    // Cross-Device Consistency Tests
    // ====================================================================

    TEST( TensorInitializersTests, CrossDevice_FillConsistency ) {
        try {
            std::vector<size_t> shape = { 50, 50 };

            Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> cpu_tensor( "CPU", shape );
            Tensor<TensorDataType::FP32, Compute::CudaMemoryResource> cuda_tensor( "CUDA:0", shape );

            float fill_value = 2.5f;

            fill( cpu_tensor, fill_value );
            fill( cuda_tensor, fill_value );

            auto cuda_to_cpu = cuda_tensor.to<TensorDataType::FP32, Compute::CpuMemoryResource>();

            // Verify both tensors have the same values
            for (size_t i = 0; i < shape[0]; ++i) {
                for (size_t j = 0; j < shape[1]; ++j) {
                    auto cpu_val = cpu_tensor.at( { i, j } );
                    auto cuda_val = cuda_to_cpu.at( { i, j } );
                    EXPECT_FLOAT_EQ( cpu_val, cuda_val );
                }
            }
        }
        catch (const std::exception&) {
            GTEST_SKIP() << "CUDA device not available, skipping cross-device test";
        }
    }

    TEST( TensorInitializersTests, CrossDevice_ZerosConsistency ) {
        try {
            std::vector<size_t> shape = { 30, 30 };

            Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> cpu_tensor( "CPU", shape );
            Tensor<TensorDataType::FP32, Compute::CudaMemoryResource> cuda_tensor( "CUDA:0", shape );

            zeros( cpu_tensor );
            zeros( cuda_tensor );

            auto cuda_to_cpu = cuda_tensor.to<TensorDataType::FP32, Compute::CpuMemoryResource>();

            for (size_t i = 0; i < shape[0]; ++i) {
                for (size_t j = 0; j < shape[1]; ++j) {
                    auto cpu_val = cpu_tensor.at( { i, j } );
                    auto cuda_val = cuda_to_cpu.at( { i, j } );
                    EXPECT_FLOAT_EQ( cpu_val, cuda_val );
                    EXPECT_FLOAT_EQ( cpu_val, 0.0f );
                }
            }
        }
        catch (const std::exception&) {
            GTEST_SKIP() << "CUDA device not available, skipping cross-device test";
        }
    }

    // ====================================================================
    // Reproducibility Tests
    // ====================================================================

    TEST( TensorInitializersTests, RandomSeedReproducibility ) {
        std::vector<size_t> shape = { 20, 20 };

        // Set specific seed
        Mila::Core::RandomGenerator::getInstance().setSeed( 12345 );

        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> tensor1( "CPU", shape );
        random( tensor1, -1.0f, 1.0f );

        // Reset to same seed
        Mila::Core::RandomGenerator::getInstance().setSeed( 12345 );

        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> tensor2( "CPU", shape );
        random( tensor2, -1.0f, 1.0f );

        // Tensors should have identical values
        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                auto val1 = tensor1.at( { i, j } );
                auto val2 = tensor2.at( { i, j } );
                EXPECT_FLOAT_EQ( val1, val2 );
            }
        }
    }

    TEST( TensorInitializersTests, DifferentSeedsDifferentValues ) {
        std::vector<size_t> shape = { 20, 20 };

        Mila::Core::RandomGenerator::getInstance().setSeed( 11111 );
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> tensor1( "CPU", shape );
        random( tensor1, -1.0f, 1.0f );

        Mila::Core::RandomGenerator::getInstance().setSeed( 22222 );
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> tensor2( "CPU", shape );
        random( tensor2, -1.0f, 1.0f );

        // Tensors should have different values
        bool found_difference = false;
        for (size_t i = 0; i < shape[0] && !found_difference; ++i) {
            for (size_t j = 0; j < shape[1] && !found_difference; ++j) {
                auto val1 = tensor1.at( { i, j } );
                auto val2 = tensor2.at( { i, j } );
                if (std::abs( val1 - val2 ) > 1e-6f) {
                    found_difference = true;
                }
            }
        }
        EXPECT_TRUE( found_difference ) << "Different seeds should produce different random values";
    }
}