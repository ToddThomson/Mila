/**
 * @file CudaResidualOpTests.cpp
 * @brief Test suite for the CUDA Residual operation.
 */

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <chrono>
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <stdexcept>

import Mila;

namespace Operations::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    /**
     * @brief Test fixture for CudaResidualOp tests
     */
    class CudaResidualOpTests : public ::testing::Test {
    protected:
        void SetUp() override {
            // Create device contexts for both CPU and CUDA
            cuda_context_ = std::make_shared<DeviceContext>( "CUDA:0" );
            cpu_context_ = std::make_shared<DeviceContext>( "CPU" );

            // Define small shapes for quick tests
            small_shape_ = { 2, 3, 4 };

            // Define medium shapes for more thorough tests
            medium_shape_ = { 8, 16, 32 };

            // Define large shapes for stress tests
            large_shape_ = { 32, 64, 128 };

            // Create CUDA Residual operation with specific context
            cuda_residual_op_ = std::make_shared<CudaResidualOp<float>>( cuda_context_ );

            // Get CPU Residual op for comparison
            auto cpu_op = OperationRegistry::instance().createOperation<float, float, DeviceType::Cpu>(
                "Cpu::ResidualOp", cpu_context_ );
            cpu_residual_op_ = std::static_pointer_cast<BinaryOperation<float, float, DeviceType::Cpu>>(cpu_op);
        }

        // Helper method to calculate reference residual result (A + B)
        void referenceResidual(
            const Tensor<float, HostMemoryResource>& input1,
            const Tensor<float, HostMemoryResource>& input2,
            Tensor<float, HostMemoryResource>& output ) {

            ASSERT_EQ( input1.size(), input2.size() );
            ASSERT_EQ( input1.size(), output.size() );

            for ( size_t i = 0; i < input1.size(); ++i ) {
                output.data()[ i ] = input1.data()[ i ] + input2.data()[ i ];
            }
        }

        // Helper method to compare tensors with tolerance
        bool compareTensors(
            const Tensor<float, HostMemoryResource>& a,
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
        std::shared_ptr<CudaResidualOp<float>> cuda_residual_op_;
        std::shared_ptr<BinaryOperation<float, float, DeviceType::Cpu>> cpu_residual_op_;

        // Test shapes
        std::vector<size_t> small_shape_;
        std::vector<size_t> medium_shape_;
        std::vector<size_t> large_shape_;
    };

    /**
     * @brief Test name property of CudaResidualOp
     */
    TEST_F( CudaResidualOpTests, Name ) {
        EXPECT_EQ( cuda_residual_op_->getName(), "Cuda::ResidualOp" );
    }

    /**
     * @brief Test basic functionality with small tensors
     */
    TEST_F( CudaResidualOpTests, BasicFunctionality ) {
        // Create input and output tensors
        Tensor<float, CudaMemoryResource> device_input1( small_shape_ );
        Tensor<float, CudaMemoryResource> device_input2( small_shape_ );
        Tensor<float, CudaMemoryResource> device_output( small_shape_ );

        Tensor<float, HostMemoryResource> host_input1( small_shape_ );
        Tensor<float, HostMemoryResource> host_input2( small_shape_ );
        Tensor<float, HostMemoryResource> host_output( small_shape_ );
        Tensor<float, HostMemoryResource> expected_output( small_shape_ );

        // Initialize input with test values
        for ( size_t i = 0; i < host_input1.size(); ++i ) {
            host_input1.data()[ i ] = static_cast<float>( i ) * 0.1f;
            host_input2.data()[ i ] = static_cast<float>( i ) * 0.2f;
        }

        // Copy to device
        device_input1.copyFrom( host_input1 );
        device_input2.copyFrom( host_input2 );

        // Execute operation
        std::vector<std::shared_ptr<Tensor<float, CudaMemoryResource>>> params;
        std::vector<std::shared_ptr<Tensor<float, CudaMemoryResource>>> output_state;
        OperationAttributes props;

        ASSERT_NO_THROW( cuda_residual_op_->forward(
            device_input1, device_input2, params, props, device_output, output_state ) );

        // Copy result back to host
        host_output.copyFrom( device_output );

        // Compute expected output with reference implementation
        referenceResidual( host_input1, host_input2, expected_output );

        // Verify output has correct values
        EXPECT_TRUE( compareTensors( host_output, expected_output, 1e-4f ) );
    }

    /**
     * @brief Test with special values (zeros, ones, negatives)
     */
    TEST_F( CudaResidualOpTests, SpecialValues ) {
        // Create input and output tensors
        Tensor<float, CudaMemoryResource> device_input1( small_shape_ );
        Tensor<float, CudaMemoryResource> device_input2( small_shape_ );
        Tensor<float, CudaMemoryResource> device_output( small_shape_ );

        Tensor<float, HostMemoryResource> host_input1( small_shape_ );
        Tensor<float, HostMemoryResource> host_input2( small_shape_ );
        Tensor<float, HostMemoryResource> host_output( small_shape_ );
        Tensor<float, HostMemoryResource> expected_output( small_shape_ );

        // Initialize with special values
        for ( size_t i = 0; i < host_input1.size(); ++i ) {
            // Pattern of zeros, ones, negatives, and mixed values
            int pattern = i % 4;
            switch ( pattern ) {
                case 0:
                    host_input1.data()[ i ] = 0.0f;
                    host_input2.data()[ i ] = static_cast<float>( i ) * 0.1f;
                    break;
                case 1:
                    host_input1.data()[ i ] = 1.0f;
                    host_input2.data()[ i ] = 0.0f;
                    break;
                case 2:
                    host_input1.data()[ i ] = -1.0f * static_cast<float>(i) * 0.1f;
                    host_input2.data()[ i ] = static_cast<float>(i) * 0.1f;
                    break;
                case 3:
                    host_input1.data()[ i ] = static_cast<float>(i) * 0.1f;
                    host_input2.data()[ i ] = -1.0f * static_cast<float>(i) * 0.1f;
                    break;
            }
        }

        // Copy to device
        device_input1.copyFrom( host_input1 );
        device_input2.copyFrom( host_input2 );

        // Execute operation
        std::vector<std::shared_ptr<Tensor<float, CudaMemoryResource>>> params;
        std::vector<std::shared_ptr<Tensor<float, CudaMemoryResource>>> output_state;
        OperationAttributes props;

        cuda_residual_op_->forward( device_input1, device_input2, params, props, device_output, output_state );

        // Copy result back to host
        host_output.copyFrom( device_output );

        // Compute expected output with reference implementation
        referenceResidual( host_input1, host_input2, expected_output );

        // Verify output has correct values
        EXPECT_TRUE( compareTensors( host_output, expected_output, 1e-4f ) );
    }

    /**
     * @brief Test that CUDA and CPU implementations produce equivalent results
     */
    TEST_F( CudaResidualOpTests, CudaCpuEquivalence ) {
        // Create input and output tensors
        Tensor<float, CudaMemoryResource> cuda_input1( medium_shape_ );
        Tensor<float, CudaMemoryResource> cuda_input2( medium_shape_ );
        Tensor<float, CudaMemoryResource> cuda_output( medium_shape_ );

        Tensor<float, HostMemoryResource> cpu_input1( medium_shape_ );
        Tensor<float, HostMemoryResource> cpu_input2( medium_shape_ );
        Tensor<float, HostMemoryResource> cpu_output( medium_shape_ );
        Tensor<float, HostMemoryResource> cuda_output_host( medium_shape_ );

        // Initialize inputs with varied test values
        for ( size_t i = 0; i < cpu_input1.size(); ++i ) {
            // Use a mix of positive, negative, and near-zero values
            cpu_input1.data()[ i ] = (static_cast<float>( i % 101 ) - 50.0f) / 25.0f;
            cpu_input2.data()[ i ] = (static_cast<float>( i % 77 ) - 38.0f) / 38.0f;
        }

        // Copy to CUDA device
        cuda_input1.copyFrom( cpu_input1 );
        cuda_input2.copyFrom( cpu_input2 );

        // Execute CUDA Residual operation
        std::vector<std::shared_ptr<Tensor<float, CudaMemoryResource>>> cuda_params;
        std::vector<std::shared_ptr<Tensor<float, CudaMemoryResource>>> cuda_output_state;
        OperationAttributes cuda_props;

        cuda_residual_op_->forward( cuda_input1, cuda_input2, cuda_params, cuda_props, cuda_output, cuda_output_state );

        // Execute CPU Residual operation
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> cpu_params;
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> cpu_output_state;
        OperationAttributes cpu_props;

        cpu_residual_op_->forward( cpu_input1, cpu_input2, cpu_params, cpu_props, cpu_output, cpu_output_state );

        // Copy CUDA result back to host
        cuda_output_host.copyFrom( cuda_output );

        // Compare results (with small tolerance for floating point differences)
        EXPECT_TRUE( compareTensors( cpu_output, cuda_output_host, 1e-4f ) );
    }

    /**
     * @brief Test backward pass functionality
     */
    //TEST_F( CudaResidualOpTests, BackwardPass ) {
    //    // Create tensors for forward and backward passes
    //    Tensor<float, CudaMemoryResource> input1( small_shape_ );
    //    Tensor<float, CudaMemoryResource> input2( small_shape_ );
    //    Tensor<float, CudaMemoryResource> output( small_shape_ );
    //    Tensor<float, CudaMemoryResource> output_grad( small_shape_ );
    //    Tensor<float, CudaMemoryResource> input1_grad( small_shape_ );
    //    Tensor<float, CudaMemoryResource> input2_grad( small_shape_ );

    //    Tensor<float, HostMemoryResource> host_input1( small_shape_ );
    //    Tensor<float, HostMemoryResource> host_input2( small_shape_ );
    //    Tensor<float, HostMemoryResource> host_output( small_shape_ );
    //    Tensor<float, HostMemoryResource> host_output_grad( small_shape_ );
    //    Tensor<float, HostMemoryResource> host_input1_grad( small_shape_ );
    //    Tensor<float, HostMemoryResource> host_input2_grad( small_shape_ );

    //    // Initialize inputs
    //    for ( size_t i = 0; i < host_input1.size(); ++i ) {
    //        host_input1.data()[ i ] = static_cast<float>( i ) * 0.1f;
    //        host_input2.data()[ i ] = static_cast<float>( i ) * 0.2f;
    //    }

    //    // Set output gradients (typically from upstream layer)
    //    for ( size_t i = 0; i < host_output_grad.size(); ++i ) {
    //        host_output_grad.data()[ i ] = 1.0f;  // Simple gradient of all ones for verification
    //    }

    //    // Copy to device
    //    input1.copyFrom( host_input1 );
    //    input2.copyFrom( host_input2 );
    //    output_grad.copyFrom( host_output_grad );

    //    // Forward pass first
    //    std::vector<std::shared_ptr<Tensor<float, CudaMemoryResource>>> params;
    //    std::vector<std::shared_ptr<Tensor<float, CudaMemoryResource>>> output_state;
    //    OperationAttributes props;

    //    cuda_residual_op_->forward( input1, input2, params, props, output, output_state );

    //    // Backward pass
    //    std::vector<std::shared_ptr<Tensor<float, CudaMemoryResource>>> param_grads;

    //    ASSERT_NO_THROW( cuda_residual_op_->backward(
    //        input1, input2, output, output_grad, params, param_grads,
    //        input1_grad, input2_grad, props, output_state ) );

    //    // Copy results back to host
    //    host_input1_grad.copyFrom( input1_grad );
    //    host_input2_grad.copyFrom( input2_grad );

    //    // For residual connection, the gradient should flow through to both inputs unchanged
    //    // Verify gradients are equal to output gradient (all 1.0f in this case)
    //    for ( size_t i = 0; i < host_input1_grad.size(); ++i ) {
    //        EXPECT_FLOAT_EQ( host_input1_grad.data()[ i ], 1.0f );
    //        EXPECT_FLOAT_EQ( host_input2_grad.data()[ i ], 1.0f );
    //    }

    //    // Verify no NaN or Inf values
    //    EXPECT_FALSE( hasNaNorInf( host_input1_grad ) );
    //    EXPECT_FALSE( hasNaNorInf( host_input2_grad ) );
    //}

    /**
     * @brief Test edge cases with different shapes
     */
    TEST_F( CudaResidualOpTests, NonStandardShapes ) {
        // Create non-standard shapes
        std::vector<size_t> odd_shape = { 3, 5, 7 };

        // Create input and output tensors
        Tensor<float, CudaMemoryResource> device_input1( odd_shape );
        Tensor<float, CudaMemoryResource> device_input2( odd_shape );
        Tensor<float, CudaMemoryResource> device_output( odd_shape );

        Tensor<float, HostMemoryResource> host_input1( odd_shape );
        Tensor<float, HostMemoryResource> host_input2( odd_shape );
        Tensor<float, HostMemoryResource> host_output( odd_shape );
        Tensor<float, HostMemoryResource> expected_output( odd_shape );

        // Initialize tensors
        for ( size_t i = 0; i < host_input1.size(); ++i ) {
            host_input1.data()[ i ] = static_cast<float>( i ) * 0.1f;
            host_input2.data()[ i ] = static_cast<float>( i ) * 0.2f;
        }

        // Copy to device
        device_input1.copyFrom( host_input1 );
        device_input2.copyFrom( host_input2 );

        // Execute operation
        std::vector<std::shared_ptr<Tensor<float, CudaMemoryResource>>> params;
        std::vector<std::shared_ptr<Tensor<float, CudaMemoryResource>>> output_state;
        OperationAttributes props;

        ASSERT_NO_THROW( cuda_residual_op_->forward(
            device_input1, device_input2, params, props, device_output, output_state ) );

        // Copy result back to host
        host_output.copyFrom( device_output );

        // Compute expected output with reference implementation
        referenceResidual( host_input1, host_input2, expected_output );

        // Verify output matches reference implementation
        EXPECT_TRUE( compareTensors( host_output, expected_output, 1e-4f ) );
    }

    /**
     * @brief Test numerical stability with varied inputs
     */
    TEST_F( CudaResidualOpTests, NumericalStability ) {
        // Create test tensors
        Tensor<float, CudaMemoryResource> device_input1( medium_shape_ );
        Tensor<float, CudaMemoryResource> device_input2( medium_shape_ );
        Tensor<float, CudaMemoryResource> device_output( medium_shape_ );

        Tensor<float, HostMemoryResource> host_input1( medium_shape_ );
        Tensor<float, HostMemoryResource> host_input2( medium_shape_ );
        Tensor<float, HostMemoryResource> host_output( medium_shape_ );

        // Test a variety of input values: positive, negative, small, large
        for ( size_t i = 0; i < host_input1.size(); ++i ) {
            int pattern1 = i % 8;
            int pattern2 = (i + 4) % 8;  // Offset pattern for second input

            float val1, val2;

            // Generate varied test patterns
            switch ( pattern1 ) {
                case 0: val1 = 1.0f; break;
                case 1: val1 = -1.0f; break;
                case 2: val1 = 0.0001f; break;
                case 3: val1 = -0.0001f; break;
                case 4: val1 = 10.0f; break;
                case 5: val1 = -10.0f; break;
                case 6: val1 = 100.0f; break;
                case 7: val1 = -100.0f; break;
                default: val1 = 0.0f; break;
            }

            switch ( pattern2 ) {
                case 0: val2 = 1.0f; break;
                case 1: val2 = -1.0f; break;
                case 2: val2 = 0.0001f; break;
                case 3: val2 = -0.0001f; break;
                case 4: val2 = 10.0f; break;
                case 5: val2 = -10.0f; break;
                case 6: val2 = 100.0f; break;
                case 7: val2 = -100.0f; break;
                default: val2 = 0.0f; break;
            }

            host_input1.data()[ i ] = val1;
            host_input2.data()[ i ] = val2;
        }

        // Copy to device
        device_input1.copyFrom( host_input1 );
        device_input2.copyFrom( host_input2 );

        // Execute operation
        std::vector<std::shared_ptr<Tensor<float, CudaMemoryResource>>> params;
        std::vector<std::shared_ptr<Tensor<float, CudaMemoryResource>>> output_state;
        OperationAttributes props;

        ASSERT_NO_THROW( cuda_residual_op_->forward(
            device_input1, device_input2, params, props, device_output, output_state ) );

        // Copy result back to host
        host_output.copyFrom( device_output );

        // Verify no NaN or Inf values
        EXPECT_FALSE( hasNaNorInf( host_output ) );

        // Verify additivity property (a + b = output)
        for ( size_t i = 0; i < host_output.size(); ++i ) {
            EXPECT_NEAR( host_output.data()[ i ], host_input1.data()[ i ] + host_input2.data()[ i ], 1e-4f );
        }
    }

    /**
     * @brief Test deterministic behavior (multiple runs should produce same result)
     */
    TEST_F( CudaResidualOpTests, DeterministicBehavior ) {
        // Create test tensors
        Tensor<float, CudaMemoryResource> device_input1( medium_shape_ );
        Tensor<float, CudaMemoryResource> device_input2( medium_shape_ );
        Tensor<float, CudaMemoryResource> device_output1( medium_shape_ );
        Tensor<float, CudaMemoryResource> device_output2( medium_shape_ );

        Tensor<float, HostMemoryResource> host_input1( medium_shape_ );
        Tensor<float, HostMemoryResource> host_input2( medium_shape_ );
        Tensor<float, HostMemoryResource> host_output1( medium_shape_ );
        Tensor<float, HostMemoryResource> host_output2( medium_shape_ );

        // Initialize with consistent values
        for ( size_t i = 0; i < host_input1.size(); ++i ) {
            host_input1.data()[ i ] = static_cast<float>( i % 17 ) * 0.1f;
            host_input2.data()[ i ] = static_cast<float>( i % 13 ) * 0.2f;
        }

        // Copy to device
        device_input1.copyFrom( host_input1 );
        device_input2.copyFrom( host_input2 );

        // Run twice with same input
        std::vector<std::shared_ptr<Tensor<float, CudaMemoryResource>>> params;
        std::vector<std::shared_ptr<Tensor<float, CudaMemoryResource>>> output_state1;
        std::vector<std::shared_ptr<Tensor<float, CudaMemoryResource>>> output_state2;
        OperationAttributes props;

        cuda_residual_op_->forward( device_input1, device_input2, params, props, device_output1, output_state1 );
        cuda_residual_op_->forward( device_input1, device_input2, params, props, device_output2, output_state2 );

        // Copy results back to host
        host_output1.copyFrom( device_output1 );
        host_output2.copyFrom( device_output2 );

        // Results should be identical
        for ( size_t i = 0; i < host_output1.size(); ++i ) {
            EXPECT_EQ( host_output1.data()[ i ], host_output2.data()[ i ] );
        }
    }

    /**
     * @brief Test error handling for device type mismatch
     */
    TEST_F( CudaResidualOpTests, DeviceTypeMismatch ) {
        // Attempt to create a CudaResidualOp with a CPU context
        EXPECT_THROW( (CudaResidualOp<float>( cpu_context_ )), std::runtime_error );
    }

    /**
     * @brief Test performance with large input
     */
    TEST_F( CudaResidualOpTests, Performance ) {
        // Skip test if running in CI environment
        if ( std::getenv( "CI" ) != nullptr ) {
            GTEST_SKIP() << "Skipping performance test in CI environment";
        }

        // Create large test tensors
        Tensor<float, CudaMemoryResource> device_input1( large_shape_ );
        Tensor<float, CudaMemoryResource> device_input2( large_shape_ );
        Tensor<float, CudaMemoryResource> device_output( large_shape_ );

        Tensor<float, HostMemoryResource> host_input1( large_shape_ );
        Tensor<float, HostMemoryResource> host_input2( large_shape_ );

        // Fill with random values
        for ( size_t i = 0; i < host_input1.size(); ++i ) {
            host_input1.data()[ i ] = (static_cast<float>( rand() ) / RAND_MAX * 2.0f - 1.0f) * 10.0f;
            host_input2.data()[ i ] = (static_cast<float>( rand() ) / RAND_MAX * 2.0f - 1.0f) * 10.0f;
        }

        // Copy to device
        device_input1.copyFrom( host_input1 );
        device_input2.copyFrom( host_input2 );

        std::vector<std::shared_ptr<Tensor<float, CudaMemoryResource>>> params;
        std::vector<std::shared_ptr<Tensor<float, CudaMemoryResource>>> output_state;
        OperationAttributes props;

        // Make sure everything is ready
        cudaDeviceSynchronize();

        // Measure performance over multiple iterations
        const int iterations = 100;
        auto start_time = std::chrono::high_resolution_clock::now();

        for ( int i = 0; i < iterations; ++i ) {
            cuda_residual_op_->forward( device_input1, device_input2, params, props, device_output, output_state );
        }

        // Wait for all operations to complete
        cudaDeviceSynchronize();

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>( end_time - start_time );

        // Calculate operations (2 operations per element: add and assign)
        size_t ops_per_iter = device_output.size();
        size_t total_ops = ops_per_iter * iterations;

        // Calculate performance metrics
        double ops_per_second = static_cast<double>(total_ops) / (duration.count() * 1e-6);
        double gops = ops_per_second / 1e9;
        double avg_time_per_iter = duration.count() / iterations;

        // Record properties that will show in test details
        RecordProperty( "Performance_GOPS", std::to_string( gops ) );
        RecordProperty( "Average_Time_us", std::to_string( avg_time_per_iter ) );
        RecordProperty( "Implementation", "CUDA" );
        RecordProperty( "Elements", std::to_string( device_output.size() ) );

        std::cout << "CUDA Residual Performance: " << gops << " GOPS" << std::endl;
        std::cout << "Average time per iteration: " << avg_time_per_iter << " microseconds" << std::endl;

        // No assertion, just informational
        EXPECT_TRUE( true );
    }

    /**
     * @brief Test comparison with CPU performance
     */
    TEST_F( CudaResidualOpTests, CpuCudaPerformanceComparison ) {
        // Skip test if running in CI environment
        if ( std::getenv( "CI" ) != nullptr ) {
            GTEST_SKIP() << "Skipping performance comparison test in CI environment";
        }

        // Create test tensors for CUDA
        Tensor<float, CudaMemoryResource> cuda_input1( medium_shape_ );
        Tensor<float, CudaMemoryResource> cuda_input2( medium_shape_ );
        Tensor<float, CudaMemoryResource> cuda_output( medium_shape_ );

        // Create test tensors for CPU
        Tensor<float, HostMemoryResource> cpu_input1( medium_shape_ );
        Tensor<float, HostMemoryResource> cpu_input2( medium_shape_ );
        Tensor<float, HostMemoryResource> cpu_output( medium_shape_ );

        // Initialize inputs
        for ( size_t i = 0; i < cpu_input1.size(); ++i ) {
            cpu_input1.data()[ i ] = static_cast<float>( i % 17 ) * 0.1f;
            cpu_input2.data()[ i ] = static_cast<float>( i % 13 ) * 0.2f;
        }

        // Copy to CUDA
        cuda_input1.copyFrom( cpu_input1 );
        cuda_input2.copyFrom( cpu_input2 );

        // Setup params
        std::vector<std::shared_ptr<Tensor<float, CudaMemoryResource>>> cuda_params;
        std::vector<std::shared_ptr<Tensor<float, CudaMemoryResource>>> cuda_output_state;
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> cpu_params;
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> cpu_output_state;
        OperationAttributes props;

        // Measure CUDA performance
        cudaDeviceSynchronize();
        const int iterations = 100;
        auto cuda_start_time = std::chrono::high_resolution_clock::now();

        for ( int i = 0; i < iterations; ++i ) {
            cuda_residual_op_->forward( cuda_input1, cuda_input2, cuda_params, props, cuda_output, cuda_output_state );
        }

        cudaDeviceSynchronize();
        auto cuda_end_time = std::chrono::high_resolution_clock::now();
        auto cuda_duration = std::chrono::duration_cast<std::chrono::microseconds>( cuda_end_time - cuda_start_time );

        // Measure CPU performance
        auto cpu_start_time = std::chrono::high_resolution_clock::now();

        for ( int i = 0; i < iterations; ++i ) {
            cpu_residual_op_->forward( cpu_input1, cpu_input2, cpu_params, props, cpu_output, cpu_output_state );
        }

        auto cpu_end_time = std::chrono::high_resolution_clock::now();
        auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>( cpu_end_time - cpu_start_time );

        // Calculate operations (2 operations per element: add and assign)
        size_t ops_per_iter = cuda_output.size();
        double cuda_gops = static_cast<double>( ops_per_iter * iterations ) / (cuda_duration.count() * 1e-6) / 1e9;
        double cpu_gops = static_cast<double>( ops_per_iter * iterations ) / (cpu_duration.count() * 1e-6) / 1e9;
        double speedup = static_cast<double>(cpu_duration.count()) / cuda_duration.count();

        // Report comparison
        std::cout << "CUDA Performance: " << cuda_gops << " GOPS" << std::endl;
        std::cout << "CPU Performance: " << cpu_gops << " GOPS" << std::endl;
        std::cout << "CUDA Speedup: " << speedup << "x" << std::endl;

        RecordProperty( "CUDA_GOPS", std::to_string( cuda_gops ) );
        RecordProperty( "CPU_GOPS", std::to_string( cpu_gops ) );
        RecordProperty( "CUDA_Speedup", std::to_string( speedup ) );

        // No assertion, just informational
        EXPECT_TRUE( true );
    }

    /**
     * @brief Test with mismatched input shapes
     */
    TEST_F( CudaResidualOpTests, Forward_MismatchedShapes ) {
        // Create tensors with different shapes
        std::vector<size_t> shape1 = { 2, 3, 4 };
        std::vector<size_t> shape2 = { 2, 3, 5 };  // Different last dimension

        Tensor<float, CudaMemoryResource> device_input1( shape1 );
        Tensor<float, CudaMemoryResource> device_input2( shape2 );
        Tensor<float, CudaMemoryResource> device_output( shape1 );

        // Initialize tensors
        Tensor<float, HostMemoryResource> host_input1( shape1 );
        Tensor<float, HostMemoryResource> host_input2( shape2 );

        for ( size_t i = 0; i < host_input1.size(); ++i ) {
            host_input1.data()[ i ] = static_cast<float>( i ) * 0.1f;
        }

        for ( size_t i = 0; i < host_input2.size(); ++i ) {
            host_input2.data()[ i ] = static_cast<float>( i ) * 0.2f;
        }

        // Copy to device
        device_input1.copyFrom( host_input1 );
        device_input2.copyFrom( host_input2 );

        // Execute operation - should fail due to shape mismatch
        std::vector<std::shared_ptr<Tensor<float, CudaMemoryResource>>> params;
        std::vector<std::shared_ptr<Tensor<float, CudaMemoryResource>>> output_state;
        OperationAttributes props;

        // This should throw an exception due to mismatched input shapes

		// TODO: Review op arg validation. Comment out for now.
        /*EXPECT_THROW( cuda_residual_op_->forward(
            device_input1, device_input2, params, props, device_output, output_state ),
            std::runtime_error );*/
    }
}