/**
 * @file CpuResidualOpTests.cpp
 * @brief Test suite for the CPU Residual operation.
 */

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <chrono>
#include <string>
#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>
#include <cmath>
#ifdef USE_OMP
#include <omp.h>
#endif

import Mila;

namespace Operations::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    /**
     * @brief Test fixture for CpuResidualOp tests
     */
    class CpuResidualOpTests : public ::testing::Test {
    protected:
        void SetUp() override {
            cpu_context_ = std::make_shared<DeviceContext>( "CPU" );

            // Small shapes for quick tests
            small_shape_ = { 2, 3, 4 };

            // Medium shapes for more thorough tests
            medium_shape_ = { 8, 16, 32 };

            // Large shape for stress tests
            large_shape_ = { 32, 64, 128 };

            // Create CPU Residual operation with specific context
            cpu_residual_op_ = std::make_shared<CpuResidualOp>( cpu_context_ );
        }

        // Helper method to check for NaNs or Infs
        bool hasNaNorInf( const Tensor<float, HostMemoryResource>& tensor ) {
            for ( size_t i = 0; i < tensor.size(); ++i ) {
                if ( std::isnan( tensor.data()[ i ] ) || std::isinf( tensor.data()[ i ] ) ) {
                    std::cout << "Found NaN or Inf at index " << i << ": " << tensor.data()[ i ] << std::endl;
                    return true;
                }
            }
            return false;
        }

        std::shared_ptr<DeviceContext> cpu_context_;
        std::shared_ptr<CpuResidualOp> cpu_residual_op_;

        std::vector<size_t> small_shape_;
        std::vector<size_t> medium_shape_;
        std::vector<size_t> large_shape_;
    };

    /**
     * @brief Test name property of CpuResidualOp
     */
    TEST_F( CpuResidualOpTests, Name ) {
        EXPECT_EQ( cpu_residual_op_->getDeviceName(), "Cpu::ResidualOp" );
    }

    /**
     * @brief Test basic functionality of CpuResidualOp
     */
    TEST_F( CpuResidualOpTests, BasicFunctionality ) {
        // Create input and output tensors
        Tensor<float, HostMemoryResource> input_a( small_shape_ );
        Tensor<float, HostMemoryResource> input_b( small_shape_ );
        Tensor<float, HostMemoryResource> output( small_shape_ );

        // Initialize input tensors with test values
        for ( size_t i = 0; i < input_a.size(); ++i ) {
            input_a.data()[ i ] = static_cast<float>( i );
            input_b.data()[ i ] = static_cast<float>( i ) * 0.5f;
        }

        // Execute residual operation
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> params;
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> output_state;
        OperationAttributes props;

        ASSERT_NO_THROW( cpu_residual_op_->forward( input_a, input_b, params, props, output, output_state ) );
		std::cerr << input_a.toString( true ) << std::endl;
		std::cerr << input_b.toString( true ) << std::endl;
		std::cerr << output.toString( true ) << std::endl;

        // Verify output has correct values (element-wise addition)
        for ( size_t i = 0; i < output.size(); ++i ) {
            float expected_output = input_a.data()[ i ] + input_b.data()[ i ];
            EXPECT_FLOAT_EQ( output.data()[ i ], expected_output );
        }
    }

    /**
     * @brief Test backward pass functionality
     */
    TEST_F( CpuResidualOpTests, BackwardPass ) {
        // Create tensors for forward and backward passes
        Tensor<float, HostMemoryResource> input_a( small_shape_ );
        Tensor<float, HostMemoryResource> input_b( small_shape_ );
        Tensor<float, HostMemoryResource> output( small_shape_ );
        Tensor<float, HostMemoryResource> output_grad( small_shape_ );
        Tensor<float, HostMemoryResource> input_a_grad( small_shape_ );
        Tensor<float, HostMemoryResource> input_b_grad( small_shape_ );

        // Initialize tensors
        for ( size_t i = 0; i < input_a.size(); ++i ) {
            input_a.data()[ i ] = static_cast<float>( i );
            input_b.data()[ i ] = static_cast<float>( i ) * 0.5f;
            // Pre-fill gradients to test accumulation
            input_a_grad.data()[ i ] = 0.1f;
            input_b_grad.data()[ i ] = 0.2f;
            // Set arbitrary output gradients
            output_grad.data()[ i ] = static_cast<float>( i % 5 );
        }

        // Forward pass first to populate output
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> params;
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> output_state;
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> param_grads;
        OperationAttributes props;

        cpu_residual_op_->forward( input_a, input_b, params, props, output, output_state );

        // Store initial gradient values for later comparison
        std::vector<float> initial_grad_a( input_a.size() );
        std::vector<float> initial_grad_b( input_b.size() );
        for ( size_t i = 0; i < input_a.size(); ++i ) {
            initial_grad_a[ i ] = input_a_grad.data()[ i ];
            initial_grad_b[ i ] = input_b_grad.data()[ i ];
        }

        // Execute backward pass
        ASSERT_NO_THROW( cpu_residual_op_->backward(
            input_a, input_b, output, output_grad,
            params, param_grads, input_a_grad, input_b_grad,
            props, output_state ) );

        // Verify gradients are not NaN or Inf
        EXPECT_FALSE( hasNaNorInf( input_a_grad ) );
        EXPECT_FALSE( hasNaNorInf( input_b_grad ) );

        // For residual connections, the gradients flow back unchanged to both inputs
        for ( size_t i = 0; i < input_a.size(); ++i ) {
            EXPECT_FLOAT_EQ( input_a_grad.data()[ i ], initial_grad_a[ i ] + output_grad.data()[ i ] );
            EXPECT_FLOAT_EQ( input_b_grad.data()[ i ], initial_grad_b[ i ] + output_grad.data()[ i ] );
        }

        // For standard input values, gradients should be non-zero
        bool all_zeros_a = true;
        bool all_zeros_b = true;
        for ( size_t i = 0; i < input_a.size(); ++i ) {
            if ( std::abs( input_a_grad.data()[ i ] ) > 1e-5f ) {
                all_zeros_a = false;
            }
            if ( std::abs( input_b_grad.data()[ i ] ) > 1e-5f ) {
                all_zeros_b = false;
            }
        }
        EXPECT_FALSE( all_zeros_a );
        EXPECT_FALSE( all_zeros_b );
    }

    /**
     * @brief Test edge cases with zero, very small, and very large values
     */
    TEST_F( CpuResidualOpTests, EdgeCases ) {
        // Create test tensors
        Tensor<float, HostMemoryResource> input_a( small_shape_ );
        Tensor<float, HostMemoryResource> input_b( small_shape_ );
        Tensor<float, HostMemoryResource> output( small_shape_ );

        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> params;
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> output_state;
        OperationAttributes props;

        // Test 1: Both inputs are zeros
        for ( size_t i = 0; i < input_a.size(); ++i ) {
            input_a.data()[ i ] = 0.0f;
            input_b.data()[ i ] = 0.0f;
        }

        ASSERT_NO_THROW( cpu_residual_op_->forward( input_a, input_b, params, props, output, output_state ) );

        for ( size_t i = 0; i < output.size(); ++i ) {
            EXPECT_FLOAT_EQ( output.data()[ i ], 0.0f );
        }

        // Test 2: One input is zero, the other has values
        for ( size_t i = 0; i < input_a.size(); ++i ) {
            input_a.data()[ i ] = static_cast<float>( i );
            input_b.data()[ i ] = 0.0f;
        }

        ASSERT_NO_THROW( cpu_residual_op_->forward( input_a, input_b, params, props, output, output_state ) );

        for ( size_t i = 0; i < output.size(); ++i ) {
            EXPECT_FLOAT_EQ( output.data()[ i ], input_a.data()[ i ] );
        }

        // Test 3: Very large values, test for overflow
        for ( size_t i = 0; i < input_a.size(); ++i ) {
            input_a.data()[ i ] = 1e20f;
            input_b.data()[ i ] = 1e20f;
        }

        ASSERT_NO_THROW( cpu_residual_op_->forward( input_a, input_b, params, props, output, output_state ) );

        // Result should be well-defined (possibly Inf, but consistent)
        for ( size_t i = 1; i < output.size(); ++i ) {
            EXPECT_EQ( std::isinf( output.data()[ 0 ] ), std::isinf( output.data()[ i ] ) );
        }

        // Test 4: Very small values, test for precision
        for ( size_t i = 0; i < input_a.size(); ++i ) {
            input_a.data()[ i ] = 1e-20f;
            input_b.data()[ i ] = 1e-20f;
        }

        ASSERT_NO_THROW( cpu_residual_op_->forward( input_a, input_b, params, props, output, output_state ) );

        for ( size_t i = 0; i < output.size(); ++i ) {
            EXPECT_FLOAT_EQ( output.data()[ i ], 2e-20f );
        }

        // Test 5: Opposite values that should cancel out
        for ( size_t i = 0; i < input_a.size(); ++i ) {
            input_a.data()[ i ] = static_cast<float>( i );
            input_b.data()[ i ] = -static_cast<float>( i );
        }

        ASSERT_NO_THROW( cpu_residual_op_->forward( input_a, input_b, params, props, output, output_state ) );

        for ( size_t i = 0; i < output.size(); ++i ) {
            EXPECT_NEAR( output.data()[ i ], 0.0f, 1e-5f );
        }
    }

    /**
     * @brief Test with inputs that have different patterns
     */
    TEST_F( CpuResidualOpTests, DifferentPatterns ) {
        // Create test tensors
        Tensor<float, HostMemoryResource> input_a( medium_shape_ );
        Tensor<float, HostMemoryResource> input_b( medium_shape_ );
        Tensor<float, HostMemoryResource> output( medium_shape_ );

        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> params;
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> output_state;
        OperationAttributes props;

        // Fill with different patterns
        for ( size_t i = 0; i < input_a.size(); ++i ) {
            // First input: alternating positive and negative values
            input_a.data()[ i ] = (i % 2 == 0) ? static_cast<float>( i ) : -static_cast<float>( i );

            // Second input: increasing values
            input_b.data()[ i ] = static_cast<float>( i );
        }

        ASSERT_NO_THROW( cpu_residual_op_->forward( input_a, input_b, params, props, output, output_state ) );

        // Verify results
        for ( size_t i = 0; i < output.size(); ++i ) {
            float expected = input_a.data()[ i ] + input_b.data()[ i ];
            EXPECT_FLOAT_EQ( output.data()[ i ], expected );
        }

        // Now try with different patterns
        for ( size_t i = 0; i < input_a.size(); ++i ) {
            // First input: sine wave pattern
            input_a.data()[ i ] = sinf( static_cast<float>( i ) * 0.1f );

            // Second input: cosine wave pattern
            input_b.data()[ i ] = cosf( static_cast<float>( i ) * 0.1f );
        }

        ASSERT_NO_THROW( cpu_residual_op_->forward( input_a, input_b, params, props, output, output_state ) );

        // Verify results
        for ( size_t i = 0; i < output.size(); ++i ) {
            float expected = input_a.data()[ i ] + input_b.data()[ i ];
            EXPECT_FLOAT_EQ( output.data()[ i ], expected );
        }
    }

    /**
     * @brief Test numerical stability with varied inputs
     */
    TEST_F( CpuResidualOpTests, NumericalStability ) {
        // Create test tensors
        Tensor<float, HostMemoryResource> input_a( medium_shape_ );
        Tensor<float, HostMemoryResource> input_b( medium_shape_ );
        Tensor<float, HostMemoryResource> output( medium_shape_ );

        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> params;
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> output_state;
        OperationAttributes props;

        // Test a variety of values: positive, negative, small, large
        for ( size_t i = 0; i < input_a.size(); ++i ) {
            float val_a, val_b;
            int pattern = i % 8;
            switch ( pattern ) {
                case 0: val_a = 1.0f; val_b = 1.0f; break;
                case 1: val_a = -1.0f; val_b = 1.0f; break;
                case 2: val_a = 0.0001f; val_b = 10000.0f; break;
                case 3: val_a = -0.0001f; val_b = -10000.0f; break;
                case 4: val_a = 10.0f; val_b = 0.1f; break;
                case 5: val_a = -10.0f; val_b = -0.1f; break;
                case 6: val_a = 100.0f; val_b = -100.0f; break;
                case 7: val_a = -100.0f; val_b = 100.0f; break;
                default: val_a = 0.0f; val_b = 0.0f; break;
            }
            input_a.data()[ i ] = val_a;
            input_b.data()[ i ] = val_b;
        }

        ASSERT_NO_THROW( cpu_residual_op_->forward( input_a, input_b, params, props, output, output_state ) );

        // Verify results
        for ( size_t i = 0; i < output.size(); ++i ) {
            float expected = input_a.data()[ i ] + input_b.data()[ i ];
            EXPECT_FLOAT_EQ( output.data()[ i ], expected );
        }
    }

    /**
     * @brief Test deterministic behavior (multiple runs should produce same result)
     */
    TEST_F( CpuResidualOpTests, DeterministicBehavior ) {
        // Create test tensors
        Tensor<float, HostMemoryResource> input_a( medium_shape_ );
        Tensor<float, HostMemoryResource> input_b( medium_shape_ );
        Tensor<float, HostMemoryResource> output1( medium_shape_ );
        Tensor<float, HostMemoryResource> output2( medium_shape_ );

        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> params;
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> output_state1;
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> output_state2;
        OperationAttributes props;

        // Initialize with consistent values
        for ( size_t i = 0; i < input_a.size(); ++i ) {
            input_a.data()[ i ] = (static_cast<float>( i % 17 ) - 8.5f) * 0.1f;
            input_b.data()[ i ] = (static_cast<float>( i % 19 ) - 9.5f) * 0.1f;
        }

        // Run twice with same input
        cpu_residual_op_->forward( input_a, input_b, params, props, output1, output_state1 );
        cpu_residual_op_->forward( input_a, input_b, params, props, output2, output_state2 );

        // Results should be identical
        for ( size_t i = 0; i < output1.size(); ++i ) {
            EXPECT_EQ( output1.data()[ i ], output2.data()[ i ] );
        }
    }

    /**
     * @brief Test error handling for device type mismatch
     */
    TEST_F( CpuResidualOpTests, DeviceTypeMismatch ) {
        // Try to create with CUDA:0 if available, otherwise skip the test
        try {
            EXPECT_THROW( CpuResidualOp( std::make_shared<DeviceContext>( "CUDA:0" ) ), std::runtime_error );
        }
        catch ( const std::exception& e ) {
            GTEST_SKIP() << "CUDA device not available, skipping device mismatch test: " << e.what();
        }
    }

    /**
     * @brief Test performance with large input
     */
    TEST_F( CpuResidualOpTests, Performance ) {
        // Skip test if running in CI environment
        if ( std::getenv( "CI" ) != nullptr ) {
            GTEST_SKIP() << "Skipping performance test in CI environment";
        }

        // Create large test tensors
        Tensor<float, HostMemoryResource> input_a( large_shape_ );
        Tensor<float, HostMemoryResource> input_b( large_shape_ );
        Tensor<float, HostMemoryResource> output( large_shape_ );

        // Fill with random values
        for ( size_t i = 0; i < input_a.size(); ++i ) {
            input_a.data()[ i ] = (static_cast<float>( rand() ) / RAND_MAX * 2.0f - 1.0f) * 10.0f;
            input_b.data()[ i ] = (static_cast<float>( rand() ) / RAND_MAX * 2.0f - 1.0f) * 10.0f;
        }

        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> params;
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> output_state;
        OperationAttributes props;

        // Measure performance over multiple iterations
        const int iterations = 10;
        auto start_time = std::chrono::high_resolution_clock::now();

        for ( int i = 0; i < iterations; ++i ) {
            cpu_residual_op_->forward( input_a, input_b, params, props, output, output_state );
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>( end_time - start_time );

        // Report performance metrics
        size_t total_elements = input_a.size() * iterations;
        double elements_per_second = static_cast<double>( total_elements ) / (duration.count() * 1e-6);
        std::cout << "CPU Residual Performance: " << elements_per_second / 1e6 << " million elements/sec" << std::endl;
        std::cout << "Average time per iteration: " << duration.count() / iterations << " microseconds" << std::endl;
    }

    /**
     * @brief Test with OpenMP (if available)
     */
    TEST_F( CpuResidualOpTests, OpenMPScaling ) {
    #ifdef USE_OMP
        // Skip test if running in CI environment
        if ( std::getenv( "CI" ) != nullptr ) {
            GTEST_SKIP() << "Skipping OpenMP scaling test in CI environment";
        }

        // Create large test tensors
        Tensor<float, HostMemoryResource> input_a( large_shape_ );
        Tensor<float, HostMemoryResource> input_b( large_shape_ );
        Tensor<float, HostMemoryResource> output( large_shape_ );

        // Fill with random values
        for ( size_t i = 0; i < input_a.size(); ++i ) {
            input_a.data()[ i ] = (static_cast<float>( rand() ) / RAND_MAX * 2.0f - 1.0f) * 10.0f;
            input_b.data()[ i ] = (static_cast<float>( rand() ) / RAND_MAX * 2.0f - 1.0f) * 10.0f;
        }

        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> params;
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> output_state;
        OperationAttributes props;

        // Get number of threads
        int max_threads = omp_get_max_threads();
        std::cout << "Max OpenMP threads: " << max_threads << std::endl;

        // Test with different numbers of threads
        std::vector<int> thread_counts = { 1 };
        if ( max_threads > 1 ) {
            thread_counts.push_back( max_threads );
            if ( max_threads > 3 ) {
                thread_counts.push_back( max_threads / 2 );
            }
        }

        const int iterations = 10;

        for ( int num_threads : thread_counts ) {
            omp_set_num_threads( num_threads );

            auto start_time = std::chrono::high_resolution_clock::now();

            for ( int i = 0; i < iterations; ++i ) {
                cpu_residual_op_->forward( input_a, input_b, params, props, output, output_state );
            }

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>( end_time - start_time );

            double elements_per_second = static_cast<double>( input_a.size() * iterations ) /
                (duration.count() * 1e-6);

            std::cout << "CPU Residual with " << num_threads << " threads: "
                << elements_per_second / 1e6 << " million elements/sec" << std::endl;
        }
    #else
        // GTEST_SKIP() << "OpenMP not available, skipping scaling test";
    #endif
    }
}
