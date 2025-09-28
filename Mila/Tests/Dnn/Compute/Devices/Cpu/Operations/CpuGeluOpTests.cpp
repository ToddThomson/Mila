/**
 * @file CpuGeluOpTests.cpp
 * @brief Test suite for the CPU GELU operation.
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

namespace Compute::Cpu::Operations::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    /**
     * @brief Test fixture for CpuGeluOp tests
     */
    class CpuGeluOpTests : public ::testing::Test {
    protected:
        void SetUp() override {
            cpu_context_ = std::make_shared<DeviceContext>( "CPU" );

            // Small shapes for quick tests
            small_shape_ = { 2, 3, 4 };

            // Medium shapes for more thorough tests
            medium_shape_ = { 8, 16, 32 };

            // Large shape for stress tests
            large_shape_ = { 32, 64, 128 };

            // Create default GELU configuration
            config_ = GeluConfig();

            // Create CPU GELU operation with specific context and config
            cpu_gelu_op_ = std::make_shared<CpuGeluOp>( cpu_context_, config_ );
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

        // Helper method to compute the GELU function reference
        float geluReference( float x ) {
            const float sqrt_2_over_pi = sqrtf( 2.0f / M_PI );
            float x_cube = 0.044715f * x * x * x;
            return 0.5f * x * (1.0f + tanhf( sqrt_2_over_pi * (x + x_cube) ));
        }

        // Helper method to compute the GELU gradient reference
        float geluGradReference( float x ) {
            const float sqrt_2_over_pi = sqrtf( 2.0f / M_PI );
            float x_cube = 0.044715f * x * x * x;
            float tanh_arg = sqrt_2_over_pi * (x + x_cube);
            float tanh_out = tanhf( tanh_arg );
            float coshf_out = coshf( tanh_arg );
            float sech_out = 1.0f / (coshf_out * coshf_out);
            return 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * sqrt_2_over_pi * (1.0f + 3.0f * 0.044715f * x * x);
        }

        std::shared_ptr<DeviceContext> cpu_context_;
        std::shared_ptr<CpuGeluOp> cpu_gelu_op_;
        GeluConfig config_;

        std::vector<size_t> small_shape_;
        std::vector<size_t> medium_shape_;
        std::vector<size_t> large_shape_;
    };

    /**
     * @brief Test name property of CpuGeluOp
     */
    TEST_F( CpuGeluOpTests, Name ) {
        EXPECT_EQ( cpu_gelu_op_->getName(), "Cpu::GeluOp" );
    }

    /**
     * @brief Test basic functionality of CpuGeluOp
     */
    TEST_F( CpuGeluOpTests, BasicFunctionality ) {
        // Create input and output tensors
        Tensor<float, HostMemoryResource> input( small_shape_ );
        Tensor<float, HostMemoryResource> output( small_shape_ );

        // Initialize input with test values
        for ( size_t i = 0; i < input.size(); ++i ) {
            input.data()[ i ] = (static_cast<float>( i ) - 10.0f) / 10.0f; // Range from -1.0 to +1.0
        }

        // Execute GELU operation
        std::vector<std::shared_ptr<ITensorData>> parameters;
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> output_state;

        ASSERT_NO_THROW( cpu_gelu_op_->forward( input, parameters, output, output_state ) );

        // Verify output has correct values
        for ( size_t i = 0; i < output.size(); ++i ) {
            float x = input.data()[ i ];
            float expected_output = geluReference( x );
            EXPECT_NEAR( output.data()[ i ], expected_output, 1e-5f );
        }
    }

    /**
     * @brief Test backward pass functionality
     */
    TEST_F( CpuGeluOpTests, BackwardPass ) {
        // Create tensors for forward and backward passes
        Tensor<float, HostMemoryResource> input( small_shape_ );
        Tensor<float, HostMemoryResource> output( small_shape_ );
        Tensor<float, HostMemoryResource> output_grad( small_shape_ );
        Tensor<float, HostMemoryResource> input_grad( small_shape_ );

        // Zero out the input gradient tensor
        for ( size_t i = 0; i < input_grad.size(); ++i ) {
            input_grad.data()[ i ] = 0.0f;
        }

        // Initialize input with test values
        for ( size_t i = 0; i < input.size(); ++i ) {
            input.data()[ i ] = (static_cast<float>( i ) - 10.0f) / 10.0f;
            // Set all output gradients to 1.0 for easy verification
            output_grad.data()[ i ] = 1.0f;
        }

        // Forward pass first
        std::vector<std::shared_ptr<ITensorData>> parameters;
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> output_state;

        cpu_gelu_op_->forward( input, parameters, output, output_state );

        // Backward pass using the raw pointers - this is how CpuGeluOp's backward is designed to work
        ASSERT_NO_THROW( cpu_gelu_op_->backward(
            input_grad.data(),
            input.data(),
            output_grad.data(),
            input.size() ) );

        // Verify gradients are not NaN or Inf
        EXPECT_FALSE( hasNaNorInf( input_grad ) );

        // Verify gradients match the expected formula
        for ( size_t i = 0; i < input.size(); ++i ) {
            float x = input.data()[ i ];
            float expected_grad = geluGradReference( x );
            EXPECT_NEAR( input_grad.data()[ i ], expected_grad, 1e-4f );
        }

        // For standard input values, gradients should be non-zero
        bool all_zeros = true;
        for ( size_t i = 0; i < input_grad.size(); ++i ) {
            if ( std::abs( input_grad.data()[ i ] ) > 1e-5f ) {
                all_zeros = false;
                break;
            }
        }
        EXPECT_FALSE( all_zeros );
    }

    /**
     * @brief Test edge cases with zero, very small, and very large values
     */
    TEST_F( CpuGeluOpTests, EdgeCases ) {
        // Create test tensors
        Tensor<float, HostMemoryResource> input( small_shape_ );
        Tensor<float, HostMemoryResource> output( small_shape_ );

        std::vector<std::shared_ptr<ITensorData>> parameters;
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> output_state;

        // Test 1: All zeros
        for ( size_t i = 0; i < input.size(); ++i ) {
            input.data()[ i ] = 0.0f;
        }

        ASSERT_NO_THROW( cpu_gelu_op_->forward( input, parameters, output, output_state ) );

        for ( size_t i = 0; i < output.size(); ++i ) {
            EXPECT_NEAR( output.data()[ i ], 0.0f, 1e-5f );
        }

        // Test 2: Very large positive values (should approach x)
        for ( size_t i = 0; i < input.size(); ++i ) {
            input.data()[ i ] = 100.0f;
        }

        ASSERT_NO_THROW( cpu_gelu_op_->forward( input, parameters, output, output_state ) );

        for ( size_t i = 0; i < output.size(); ++i ) {
            EXPECT_NEAR( output.data()[ i ], input.data()[ i ], input.data()[ i ] * 0.01f ); // Within 1%
        }

        // Test 3: Very large negative values (should approach 0)
        for ( size_t i = 0; i < input.size(); ++i ) {
            input.data()[ i ] = -100.0f;
        }

        ASSERT_NO_THROW( cpu_gelu_op_->forward( input, parameters, output, output_state ) );

        for ( size_t i = 0; i < output.size(); ++i ) {
            EXPECT_NEAR( output.data()[ i ], 0.0f, 1e-4f );
        }

        // Test 4: Very small values (should be close to 0.5 * x)
        for ( size_t i = 0; i < input.size(); ++i ) {
            input.data()[ i ] = 1e-5f;
        }

        ASSERT_NO_THROW( cpu_gelu_op_->forward( input, parameters, output, output_state ) );

        for ( size_t i = 0; i < output.size(); ++i ) {
            EXPECT_NEAR( output.data()[ i ], 0.5f * input.data()[ i ], 1e-8f );
        }
    }

    /**
     * @brief Test numerical stability with varied inputs
     */
    TEST_F( CpuGeluOpTests, NumericalStability ) {
        // Create test tensors
        Tensor<float, HostMemoryResource> input( medium_shape_ );
        Tensor<float, HostMemoryResource> output( medium_shape_ );

        std::vector<std::shared_ptr<ITensorData>> parameters;
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> output_state;

        // Test a variety of values: positive, negative, small, large
        for ( size_t i = 0; i < input.size(); ++i ) {
            float val;
            int pattern = i % 8;
            switch ( pattern ) {
                case 0: val = 1.0f; break;
                case 1: val = -1.0f; break;
                case 2: val = 0.0001f; break;
                case 3: val = -0.0001f; break;
                case 4: val = 10.0f; break;
                case 5: val = -10.0f; break;
                case 6: val = 100.0f; break;
                case 7: val = -100.0f; break;
                default: val = 0.0f; break;
            }
            input.data()[ i ] = val;
        }

        ASSERT_NO_THROW( cpu_gelu_op_->forward( input, parameters, output, output_state ) );

        // Verify no NaN or Inf values
        EXPECT_FALSE( hasNaNorInf( output ) );
    }

    /**
     * @brief Test deterministic behavior (multiple runs should produce same result)
     */
    TEST_F( CpuGeluOpTests, DeterministicBehavior ) {
        // Create test tensors
        Tensor<float, HostMemoryResource> input( medium_shape_ );
        Tensor<float, HostMemoryResource> output1( medium_shape_ );
        Tensor<float, HostMemoryResource> output2( medium_shape_ );

        std::vector<std::shared_ptr<ITensorData>> parameters;
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> output_state1;
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> output_state2;

        // Initialize with consistent values
        for ( size_t i = 0; i < input.size(); ++i ) {
            input.data()[ i ] = (static_cast<float>( i % 17 ) - 8.5f) * 0.1f;
        }

        // Run twice with same input
        cpu_gelu_op_->forward( input, parameters, output1, output_state1 );
        cpu_gelu_op_->forward( input, parameters, output2, output_state2 );

        // Results should be identical
        for ( size_t i = 0; i < output1.size(); ++i ) {
            EXPECT_EQ( output1.data()[ i ], output2.data()[ i ] );
        }
    }

    /**
     * @brief Test error handling for device type mismatch
     */
    TEST_F( CpuGeluOpTests, DeviceTypeMismatch ) {
        // Try to create with CUDA:0 if available, otherwise skip the test
        try {
            GeluConfig cuda_config;
            EXPECT_THROW( CpuGeluOp( std::make_shared<DeviceContext>( "CUDA:0" ), cuda_config ), std::runtime_error );
        }
        catch ( const std::exception& e ) {
            GTEST_SKIP() << "CUDA device not available, skipping device mismatch test: " << e.what();
        }
    }

    /**
     * @brief Test constructors
     */
    TEST_F( CpuGeluOpTests, Constructors ) {
        GeluConfig test_config;

        // CPU operations should work with both constructor forms
        ASSERT_NO_THROW( ( CpuGeluOp( test_config ) ) );
        ASSERT_NO_THROW( CpuGeluOp( cpu_context_, test_config ) );
    }

    /**
     * @brief Test different approximation methods
     */
    TEST_F( CpuGeluOpTests, ApproximationMethods ) {
        // Create test tensors
        Tensor<float, HostMemoryResource> input( small_shape_ );
        Tensor<float, HostMemoryResource> output_tanh( small_shape_ );

        // Initialize input with test values
        for ( size_t i = 0; i < input.size(); ++i ) {
            input.data()[ i ] = (static_cast<float>( i ) - 10.0f) / 10.0f;
        }

        std::vector<std::shared_ptr<ITensorData>> parameters;
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> output_state;

        // Test Tanh approximation (only supported method)
        GeluConfig tanh_config = GeluConfig().withApproximationMethod( GeluConfig::ApproximationMethod::Tanh );
        auto tanh_op = std::make_shared<CpuGeluOp>( cpu_context_, tanh_config );

        ASSERT_NO_THROW( tanh_op->forward( input, parameters, output_tanh, output_state ) );

        // Verify output values match our reference implementation (which uses tanh)
        for ( size_t i = 0; i < output_tanh.size(); ++i ) {
            float x = input.data()[ i ];
            float expected_output = geluReference( x );
            EXPECT_NEAR( output_tanh.data()[ i ], expected_output, 1e-5f );
        }

        // Test unsupported approximation methods should fail validation
        GeluConfig exact_config = GeluConfig().withApproximationMethod( GeluConfig::ApproximationMethod::Exact );
        EXPECT_THROW( exact_config.validate(), std::invalid_argument );

        GeluConfig sigmoid_config = GeluConfig().withApproximationMethod( GeluConfig::ApproximationMethod::Sigmoid );
        EXPECT_THROW( sigmoid_config.validate(), std::invalid_argument );
    }

    /**
     * @brief Test performance with large input
     */
    TEST_F( CpuGeluOpTests, Performance ) {
        // Skip test if running in CI environment
        if ( std::getenv( "CI" ) != nullptr ) {
            GTEST_SKIP() << "Skipping performance test in CI environment";
        }

        // Create large test tensors
        Tensor<float, HostMemoryResource> input( large_shape_ );
        Tensor<float, HostMemoryResource> output( large_shape_ );

        // Fill with random values
        for ( size_t i = 0; i < input.size(); ++i ) {
            input.data()[ i ] = (static_cast<float>( rand() ) / RAND_MAX * 2.0f - 1.0f) * 10.0f;
        }

        std::vector<std::shared_ptr<ITensorData>> parameters;
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> output_state;

        // Measure performance over multiple iterations
        const int iterations = 10;
        auto start_time = std::chrono::high_resolution_clock::now();

        for ( int i = 0; i < iterations; ++i ) {
            cpu_gelu_op_->forward( input, parameters, output, output_state );
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>( end_time - start_time );

        // Report performance metrics
        size_t total_elements = input.size() * iterations;
        double elements_per_second = static_cast<double>( total_elements ) / (duration.count() * 1e-6);
        std::cout << "CPU GELU Performance: " << elements_per_second / 1e6 << " million elements/sec" << std::endl;
        std::cout << "Average time per iteration: " << duration.count() / iterations << " microseconds" << std::endl;
    }

    /**
     * @brief Test with OpenMP (if available)
     */
    TEST_F( CpuGeluOpTests, OpenMPScaling ) {
    #ifdef USE_OMP
        // Skip test if running in CI environment
        if ( std::getenv( "CI" ) != nullptr ) {
            GTEST_SKIP() << "Skipping OpenMP scaling test in CI environment";
        }

        // Create large test tensors
        Tensor<float, HostMemoryResource> input( large_shape_ );
        Tensor<float, HostMemoryResource> output( large_shape_ );

        // Fill with random values
        for ( size_t i = 0; i < input.size(); ++i ) {
            input.data()[ i ] = (static_cast<float>( rand() ) / RAND_MAX * 2.0f - 1.0f) * 10.0f;
        }

        std::vector<std::shared_ptr<ITensorData>> parameters;
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> output_state;

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
                cpu_gelu_op_->forward( input, parameters, output, output_state );
            }

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>( end_time - start_time );

            double elements_per_second = static_cast<double>( input.size() * iterations ) /
                (duration.count() * 1e-6);

            std::cout << "CPU GELU with " << num_threads << " threads: "
                << elements_per_second / 1e6 << " million elements/sec" << std::endl;
        }
    #else
        // GTEST_SKIP() << "OpenMP not available, skipping scaling test";
    #endif
    }
}