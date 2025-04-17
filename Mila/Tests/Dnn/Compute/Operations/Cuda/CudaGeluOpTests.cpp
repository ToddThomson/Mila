/**
 * @file CudaGeluOpTests.cpp
 * @brief Test suite for the CUDA GELU operation.
 */

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <chrono>
#include <string>
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

import Mila;

namespace Operations::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    /**
     * @brief Test fixture for CudaGeluOp tests
     */
    class CudaGeluOpTests : public ::testing::Test {
    protected:
        void SetUp() override {
            cuda_context_ = std::make_shared<DeviceContext>( "CUDA:0" );
            cpu_context_ = std::make_shared<DeviceContext>( "CPU" );

            // Small shapes for quick tests
            small_shape_ = { 2, 3, 4 };

            // Medium shapes for more thorough tests
            medium_shape_ = { 8, 16, 32 };

            // Large shape for stress tests
            large_shape_ = { 32, 64, 128 };

            // Create CUDA GELU operation with specific context
            cuda_gelu_op_ = std::make_shared<CudaGeluOp<float, float>>( cuda_context_ );

            // Get CPU GELU op for comparison
            auto cpu_op = OperationRegistry::instance().createOperation<float, float, DeviceType::Cpu>(
                "Cpu::GeluOp", cpu_context_ );
            cpu_gelu_op_ = std::static_pointer_cast<UnaryOperation<float, float, DeviceType::Cpu>>(cpu_op);
        }

        // Helper method to compare tensors with tolerance
        bool compareTensors( const Tensor<float, HostMemoryResource>& a,
            const Tensor<float, HostMemoryResource>& b,
            float epsilon = 1e-5f ) {
            if ( a.size() != b.size() ) return false;

            for ( size_t i = 0; i < a.size(); ++i ) {
                float diff = std::abs( a.data()[ i ] - b.data()[ i ] );
                if ( diff > epsilon ) {
                    std::cout << "Mismatch at index " << i << ": "
                        << a.data()[ i ] << " vs " << b.data()[ i ]
                        << " (diff = " << diff << ")" << std::endl;
                        return false;
                }
            }
            return true;
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

        std::shared_ptr<DeviceContext> cuda_context_;
        std::shared_ptr<DeviceContext> cpu_context_;
        std::shared_ptr<CudaGeluOp<float, float>> cuda_gelu_op_;
        std::shared_ptr<UnaryOperation<float, float, DeviceType::Cpu>> cpu_gelu_op_;

        std::vector<size_t> small_shape_;
        std::vector<size_t> medium_shape_;
        std::vector<size_t> large_shape_;
    };

    /**
     * @brief Test basic functionality of CudaGeluOp
     */
    TEST_F( CudaGeluOpTests, BasicFunctionality ) {
        // Create input and output tensors
        Tensor<float, DeviceMemoryResource> device_input( small_shape_ );
        Tensor<float, DeviceMemoryResource> device_output( small_shape_ );
        Tensor<float, HostMemoryResource> host_input( small_shape_ );
        Tensor<float, HostMemoryResource> host_output( small_shape_ );

        // Initialize input with test values
        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input.data()[ i ] = (static_cast<float>( i ) - 10.0f) / 10.0f; // Range from -1.0 to +1.0
        }

        // Copy to device
        device_input.copyFrom( host_input );

        // Execute GELU operation
        std::vector<std::shared_ptr<Tensor<float, DeviceMemoryResource>>> params;
        std::vector<std::shared_ptr<Tensor<float, DeviceMemoryResource>>> output_cache;
        OperationAttributes props;

        ASSERT_NO_THROW( cuda_gelu_op_->forward( device_input, params, props, device_output, output_cache ) );

        // Copy result back to host
        host_output.copyFrom( device_output );

        // Verify output has correct values
        for ( size_t i = 0; i < host_output.size(); ++i ) {
            float x = host_input.data()[ i ];
            // Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/?) * (x + 0.044715 * x^3)))
            const float sqrt_2_over_pi = 0.7978845608f; // sqrt(2/?)
            float x_cube = x * x * x;
            float gelu_approx = 0.5f * x * (1.0f + std::tanh( sqrt_2_over_pi * (x + 0.044715f * x_cube) ));

            EXPECT_NEAR( host_output.data()[ i ], gelu_approx, 1e-5f );
        }
    }

    /**
     * @brief Test that CUDA and CPU implementations produce equivalent results
     */
    TEST_F( CudaGeluOpTests, CudaCpuEquivalence ) {
        // Create input and output tensors
        Tensor<float, DeviceMemoryResource> cuda_input( medium_shape_ );
        Tensor<float, DeviceMemoryResource> cuda_output( medium_shape_ );
        Tensor<float, HostMemoryResource> cpu_input( medium_shape_ );
        Tensor<float, HostMemoryResource> cpu_output( medium_shape_ );
        Tensor<float, HostMemoryResource> cuda_output_host( medium_shape_ );

        // Initialize input with varied test values
        for ( size_t i = 0; i < cpu_input.size(); ++i ) {
            // Use a mix of positive, negative, and near-zero values
            cpu_input.data()[ i ] = (static_cast<float>( i % 101 ) - 50.0f) / 25.0f;
        }

        // Copy to CUDA device
        cuda_input.copyFrom( cpu_input );

        // Execute CUDA GELU operation
        std::vector<std::shared_ptr<Tensor<float, DeviceMemoryResource>>> cuda_params;
        std::vector<std::shared_ptr<Tensor<float, DeviceMemoryResource>>> cuda_output_cache;
        OperationAttributes cuda_props;

        cuda_gelu_op_->forward( cuda_input, cuda_params, cuda_props, cuda_output, cuda_output_cache );

        // Execute CPU GELU operation
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> cpu_params;
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> cpu_output_cache;
        OperationAttributes cpu_props;

        cpu_gelu_op_->forward( cpu_input, cpu_params, cpu_props, cpu_output, cpu_output_cache );

        // Copy CUDA result back to host
        cuda_output_host.copyFrom( cuda_output );

        // Compare results (with small tolerance for floating point differences)
        EXPECT_TRUE( compareTensors( cpu_output, cuda_output_host, 1e-5f ) );
    }

    /**
     * @brief Test backward pass functionality
     */
    TEST_F( CudaGeluOpTests, BackwardPass ) {
        // Create tensors for forward and backward passes
        Tensor<float, DeviceMemoryResource> input( small_shape_ );
        Tensor<float, DeviceMemoryResource> output( small_shape_ );
        Tensor<float, DeviceMemoryResource> output_grad( small_shape_ );
        Tensor<float, DeviceMemoryResource> input_grad( small_shape_ );

        Tensor<float, HostMemoryResource> host_input( small_shape_ );
        Tensor<float, HostMemoryResource> host_output_grad( small_shape_ );
        Tensor<float, HostMemoryResource> host_input_grad( small_shape_ );

        // Initialize input with test values
        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input.data()[ i ] = (static_cast<float>( i ) - 10.0f) / 10.0f;
            // Set all output gradients to 1.0 for easy verification
            host_output_grad.data()[ i ] = 1.0f;
        }

        // Copy to device
        input.copyFrom( host_input );
        output_grad.copyFrom( host_output_grad );

        // Forward pass first
        std::vector<std::shared_ptr<Tensor<float, DeviceMemoryResource>>> params;
        std::vector<std::shared_ptr<Tensor<float, DeviceMemoryResource>>> output_cache;
        OperationAttributes props;

        cuda_gelu_op_->forward( input, params, props, output, output_cache );

        // Backward pass
        std::vector<std::shared_ptr<Tensor<float, DeviceMemoryResource>>> param_grads;

        ASSERT_NO_THROW( cuda_gelu_op_->backward(
            input, output, output_grad, params, param_grads, input_grad, props, output_cache ) );

        // Copy result back to host
        host_input_grad.copyFrom( input_grad );

        // Verify gradients are not NaN or Inf
        EXPECT_FALSE( hasNaNorInf( host_input_grad ) );

        // The GELU gradient should be approximately:
        // dGELU/dx = 0.5 * (1 + tanh(sqrt(2/?) * (x + 0.044715 * x^3)))
        //           + 0.5 * x * (1 - tanh(sqrt(2/?) * (x + 0.044715 * x^3))^2) * sqrt(2/?) * (1 + 3 * 0.044715 * x^2)
        // For standard input values, this should be non-zero
        bool all_zeros = true;
        for ( size_t i = 0; i < host_input_grad.size(); ++i ) {
            if ( std::abs( host_input_grad.data()[ i ] ) > 1e-5f ) {
                all_zeros = false;
                break;
            }
        }
        EXPECT_FALSE( all_zeros );
    }

    /**
     * @brief Test edge cases with zero, very small, and very large values
     */
    TEST_F( CudaGeluOpTests, EdgeCases ) {
        // Create test tensors
        Tensor<float, DeviceMemoryResource> device_input( small_shape_ );
        Tensor<float, DeviceMemoryResource> device_output( small_shape_ );
        Tensor<float, HostMemoryResource> host_input( small_shape_ );
        Tensor<float, HostMemoryResource> host_output( small_shape_ );

        std::vector<std::shared_ptr<Tensor<float, DeviceMemoryResource>>> params;
        std::vector<std::shared_ptr<Tensor<float, DeviceMemoryResource>>> output_cache;
        OperationAttributes props;

        // Test 1: All zeros
        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input.data()[ i ] = 0.0f;
        }
        device_input.copyFrom( host_input );

        ASSERT_NO_THROW( cuda_gelu_op_->forward( device_input, params, props, device_output, output_cache ) );

        host_output.copyFrom( device_output );
        for ( size_t i = 0; i < host_output.size(); ++i ) {
            EXPECT_NEAR( host_output.data()[ i ], 0.0f, 1e-5f );
        }

        // Test 2: Very large positive values (should approach x)
        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input.data()[ i ] = 100.0f;
        }
        device_input.copyFrom( host_input );

        ASSERT_NO_THROW( cuda_gelu_op_->forward( device_input, params, props, device_output, output_cache ) );

        host_output.copyFrom( device_output );
        for ( size_t i = 0; i < host_output.size(); ++i ) {
            EXPECT_NEAR( host_output.data()[ i ], host_input.data()[ i ], host_input.data()[ i ] * 0.01f ); // Within 1%
        }

        // Test 3: Very large negative values (should approach 0)
        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input.data()[ i ] = -100.0f;
        }
        device_input.copyFrom( host_input );

        ASSERT_NO_THROW( cuda_gelu_op_->forward( device_input, params, props, device_output, output_cache ) );

        host_output.copyFrom( device_output );
        for ( size_t i = 0; i < host_output.size(); ++i ) {
            EXPECT_NEAR( host_output.data()[ i ], 0.0f, 1e-4f );
        }

        // Test 4: Very small values (should be close to 0.5 * x)
        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input.data()[ i ] = 1e-5f;
        }
        device_input.copyFrom( host_input );

        ASSERT_NO_THROW( cuda_gelu_op_->forward( device_input, params, props, device_output, output_cache ) );

        host_output.copyFrom( device_output );
        for ( size_t i = 0; i < host_output.size(); ++i ) {
            EXPECT_NEAR( host_output.data()[ i ], 0.5f * host_input.data()[ i ], 1e-8f );
        }
    }

    /**
     * @brief Test numerical stability with varied inputs
     */
    TEST_F( CudaGeluOpTests, NumericalStability ) {
        // Create test tensors
        Tensor<float, DeviceMemoryResource> device_input( medium_shape_ );
        Tensor<float, DeviceMemoryResource> device_output( medium_shape_ );
        Tensor<float, HostMemoryResource> host_input( medium_shape_ );
        Tensor<float, HostMemoryResource> host_output( medium_shape_ );

        std::vector<std::shared_ptr<Tensor<float, DeviceMemoryResource>>> params;
        std::vector<std::shared_ptr<Tensor<float, DeviceMemoryResource>>> output_cache;
        OperationAttributes props;

        // Test a variety of values: positive, negative, small, large
        for ( size_t i = 0; i < host_input.size(); ++i ) {
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
            host_input.data()[ i ] = val;
        }

        device_input.copyFrom( host_input );
        ASSERT_NO_THROW( cuda_gelu_op_->forward( device_input, params, props, device_output, output_cache ) );
        host_output.copyFrom( device_output );

        // Verify no NaN or Inf values
        EXPECT_FALSE( hasNaNorInf( host_output ) );
    }

    /**
     * @brief Test deterministic behavior (multiple runs should produce same result)
     */
    TEST_F( CudaGeluOpTests, DeterministicBehavior ) {
        // Create test tensors
        Tensor<float, DeviceMemoryResource> device_input( medium_shape_ );
        Tensor<float, DeviceMemoryResource> device_output1( medium_shape_ );
        Tensor<float, DeviceMemoryResource> device_output2( medium_shape_ );
        Tensor<float, HostMemoryResource> host_input( medium_shape_ );
        Tensor<float, HostMemoryResource> host_output1( medium_shape_ );
        Tensor<float, HostMemoryResource> host_output2( medium_shape_ );

        std::vector<std::shared_ptr<Tensor<float, DeviceMemoryResource>>> params;
        std::vector<std::shared_ptr<Tensor<float, DeviceMemoryResource>>> output_cache1;
        std::vector<std::shared_ptr<Tensor<float, DeviceMemoryResource>>> output_cache2;
        OperationAttributes props;

        // Initialize with consistent values
        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input.data()[ i ] = (static_cast<float>( i % 17 ) - 8.5f) * 0.1f;
        }

        device_input.copyFrom( host_input );

        // Run twice with same input
        cuda_gelu_op_->forward( device_input, params, props, device_output1, output_cache1 );
        cuda_gelu_op_->forward( device_input, params, props, device_output2, output_cache2 );

        // Copy results back to host
        host_output1.copyFrom( device_output1 );
        host_output2.copyFrom( device_output2 );

        // Results should be identical
        for ( size_t i = 0; i < host_output1.size(); ++i ) {
            EXPECT_EQ( host_output1.data()[ i ], host_output2.data()[ i ] );
        }
    }

    /**
     * @brief Test performance with large input
     */
    TEST_F( CudaGeluOpTests, Performance ) {
        // Skip test if running in CI environment without proper CUDA devices
        if ( std::getenv( "CI" ) != nullptr ) {
            GTEST_SKIP() << "Skipping performance test in CI environment";
        }

        // Create large test tensors
        Tensor<float, DeviceMemoryResource> device_input( large_shape_ );
        Tensor<float, DeviceMemoryResource> device_output( large_shape_ );
        Tensor<float, HostMemoryResource> host_input( large_shape_ );

        // Fill with random values
        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input.data()[ i ] = (static_cast<float>( rand() ) / RAND_MAX * 2.0f - 1.0f) * 10.0f;
        }

        device_input.copyFrom( host_input );

        std::vector<std::shared_ptr<Tensor<float, DeviceMemoryResource>>> params;
        std::vector<std::shared_ptr<Tensor<float, DeviceMemoryResource>>> output_cache;
        OperationAttributes props;

        // Make sure everything is ready
        cudaDeviceSynchronize();

        // Measure performance over multiple iterations
        const int iterations = 100;
        auto start_time = std::chrono::high_resolution_clock::now();

        for ( int i = 0; i < iterations; ++i ) {
            cuda_gelu_op_->forward( device_input, params, props, device_output, output_cache );
        }

        // Wait for all operations to complete
        cudaDeviceSynchronize();

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>( end_time - start_time );

        // Report performance metrics
        size_t total_elements = device_input.size() * iterations;
        double elements_per_second = static_cast<double>(total_elements) / (duration.count() * 1e-6);
        std::cout << "GELU Performance: " << elements_per_second / 1e9 << " billion elements/sec" << std::endl;
        std::cout << "Average time per iteration: " << duration.count() / iterations << " microseconds" << std::endl;

        // No assertion, just informational
    }
}
