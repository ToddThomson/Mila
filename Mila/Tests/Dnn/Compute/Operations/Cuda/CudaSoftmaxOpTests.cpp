/**
 * @file CudaSoftmaxOpTests.cpp
 * @brief Test suite for the CUDA Softmax operation.
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
     * @brief Test fixture for CudaSoftmaxOp tests
     */
    class CudaSoftmaxOpTests : public ::testing::Test {
    protected:
        void SetUp() override {
            // Create device contexts for both CPU and CUDA
            cuda_context_ = std::make_shared<DeviceContext>( "CUDA:0" );
            cpu_context_ = std::make_shared<DeviceContext>( "CPU" );

            // Define shapes for testing
            // Small shape for quick tests (batch, sequence_length, vocabulary)
            small_batch_ = 2;
            small_seq_len_ = 3;
            small_vocab_size_ = 4;
            small_shape_ = { small_batch_, small_seq_len_, small_vocab_size_ };

            // Medium shape for more thorough tests
            medium_batch_ = 8;
            medium_seq_len_ = 16;
            medium_vocab_size_ = 32;
            medium_shape_ = { medium_batch_, medium_seq_len_, medium_vocab_size_ };

            // Large shape for stress tests
            large_batch_ = 16;
            large_seq_len_ = 32;
            large_vocab_size_ = 1024;
            large_shape_ = { large_batch_, large_seq_len_, large_vocab_size_ };

            // Create CUDA Softmax operation with specific context
            cuda_softmax_op_ = std::make_shared<CudaSoftmaxOp<float, float>>( cuda_context_ );

            // Get CPU Softmax op for comparison
            auto cpu_op = OperationRegistry::instance().createOperation<float, float, DeviceType::Cpu>(
                "Cpu::SoftmaxOp", cpu_context_ );
            cpu_softmax_op_ = std::static_pointer_cast<UnaryOperation<float, float, DeviceType::Cpu>>(cpu_op);
        }

        // Helper method to reference implementation of softmax operation
        void referenceSoftmax(
            const Tensor<float, HostMemoryResource>& input,
            Tensor<float, HostMemoryResource>& output,
            int axis ) {

            // Currently, we only support axis = 2 (softmax along the vocabulary dimension)
            ASSERT_EQ( axis, 2 );

            int B = input.shape()[ 0 ];   // Batch size
            int T = input.shape()[ 1 ];   // Sequence length
            int V = input.shape()[ 2 ];   // Vocabulary size

            // For each batch and sequence position
            for ( int b = 0; b < B; b++ ) {
                for ( int t = 0; t < T; t++ ) {
                    // First find the maximum value for numerical stability
                    float max_val = -std::numeric_limits<float>::infinity();
                    for ( int v = 0; v < V; v++ ) {
                        float val = input.data()[ (b * T + t) * V + v ];
                        max_val = std::max( max_val, val );
                    }

                    // Calculate the sum of exp(x - max_val)
                    float sum = 0.0f;
                    for ( int v = 0; v < V; v++ ) {
                        float val = input.data()[ (b * T + t) * V + v ];
                        float exp_val = std::exp( val - max_val );
                        output.data()[ (b * T + t) * V + v ] = exp_val;
                        sum += exp_val;
                    }

                    // Normalize by the sum
                    for ( int v = 0; v < V; v++ ) {
                        output.data()[ (b * T + t) * V + v ] /= sum;
                    }
                }
            }
        }

        // Helper method for calculating softmax gradient
        void referenceSoftmaxGradient(
            const Tensor<float, HostMemoryResource>& output,
            const Tensor<float, HostMemoryResource>& output_gradient,
            Tensor<float, HostMemoryResource>& input_gradient,
            int axis ) {

            // Currently, we only support axis = 2 (softmax along the vocabulary dimension)
            ASSERT_EQ( axis, 2 );

            int B = output.shape()[ 0 ];   // Batch size
            int T = output.shape()[ 1 ];   // Sequence length
            int V = output.shape()[ 2 ];   // Vocabulary size

            // For each batch and sequence position
            for ( int b = 0; b < B; b++ ) {
                for ( int t = 0; t < T; t++ ) {
                    // Calculate gradient for each vocabulary position
                    for ( int i = 0; i < V; i++ ) {
                        float grad_i = 0.0f;
                        float y_i = output.data()[ (b * T + t) * V + i ];

                        for ( int j = 0; j < V; j++ ) {
                            float y_j = output.data()[ (b * T + t) * V + j ];
                            float dy_j = output_gradient.data()[ (b * T + t) * V + j ];

                            // dL/dx_i = sum_j(dL/dy_j * (y_i * (kronecker_delta(i,j) - y_j)))
                            if ( i == j ) {
                                grad_i += dy_j * y_i * (1.0f - y_j);
                            }
                            else {
                                grad_i += dy_j * (-y_i * y_j);
                            }
                        }

                        input_gradient.data()[ (b * T + t) * V + i ] = grad_i;
                    }
                }
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

        // Helper method to check if softmax outputs sum to 1 along the specified axis
        bool checkSoftmaxProperties(
            const Tensor<float, HostMemoryResource>& output,
            int axis,
            float epsilon = 1e-5f ) {

            // Currently, we only support axis = 2
            if ( axis != 2 ) return false;

            int B = output.shape()[ 0 ];
            int T = output.shape()[ 1 ];
            int V = output.shape()[ 2 ];

            for ( int b = 0; b < B; b++ ) {
                for ( int t = 0; t < T; t++ ) {
                    float sum = 0.0f;
                    bool has_negative = false;

                    for ( int v = 0; v < V; v++ ) {
                        float val = output.data()[ (b * T + t) * V + v ];

                        // Check if any value is negative (shouldn't happen with softmax)
                        if ( val < 0.0f ) {
                            std::cout << "Found negative value: " << val << " at ["
                                << b << ", " << t << ", " << v << "]" << std::endl;
                            has_negative = true;
                        }

                        sum += val;
                    }

                    // Check if sum equals 1
                    if ( std::abs( sum - 1.0f ) > epsilon ) {
                        std::cout << "Sum not equal to 1: " << sum << " at ["
                            << b << ", " << t << "]" << std::endl;
                        return false;
                    }

                    if ( has_negative ) return false;
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
        std::shared_ptr<CudaSoftmaxOp<float, float>> cuda_softmax_op_;
        std::shared_ptr<UnaryOperation<float, float, DeviceType::Cpu>> cpu_softmax_op_;

        // Test dimensions
        size_t small_batch_, small_seq_len_, small_vocab_size_;
        size_t medium_batch_, medium_seq_len_, medium_vocab_size_;
        size_t large_batch_, large_seq_len_, large_vocab_size_;

        // Test shapes
        std::vector<size_t> small_shape_;
        std::vector<size_t> medium_shape_;
        std::vector<size_t> large_shape_;
    };

    /**
     * @brief Test name property of CudaSoftmaxOp
     */
    TEST_F( CudaSoftmaxOpTests, Name ) {
        EXPECT_EQ( cuda_softmax_op_->getName(), "Cuda::SoftmaxOp" );
    }

    /**
     * @brief Test basic functionality with small tensors
     */
    TEST_F( CudaSoftmaxOpTests, BasicFunctionality ) {
        // Create input and output tensors
        Tensor<float, DeviceMemoryResource> device_input( small_shape_ );
        Tensor<float, DeviceMemoryResource> device_output( small_shape_ );

        Tensor<float, HostMemoryResource> host_input( small_shape_ );
        Tensor<float, HostMemoryResource> host_output( small_shape_ );
        Tensor<float, HostMemoryResource> expected_output( small_shape_ );

        // Initialize input with test values
        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input.data()[ i ] = static_cast<float>( i % 7 ) - 3.0f;  // Range from -3 to 3
        }

        // Copy to device
        device_input.copyFrom( host_input );

        // Execute operation
        std::vector<std::shared_ptr<Tensor<float, DeviceMemoryResource>>> params;
        std::vector<std::shared_ptr<Tensor<float, DeviceMemoryResource>>> output_cache;
        OperationAttributes props;
        props.axis = 2;  // Softmax along vocabulary dimension

        ASSERT_NO_THROW( cuda_softmax_op_->forward(
            device_input, params, props, device_output, output_cache ) );

        // Copy result back to host
        host_output.copyFrom( device_output );

        // Compute expected output with reference implementation
        referenceSoftmax( host_input, expected_output, props.axis );

        // Verify output has correct values
        EXPECT_TRUE( compareTensors( host_output, expected_output, 1e-4f ) );

        // Verify softmax output properties (sums to 1, non-negative values)
        EXPECT_TRUE( checkSoftmaxProperties( host_output, props.axis ) );
    }

    /**
     * @brief Test with special input values
     */
    TEST_F( CudaSoftmaxOpTests, SpecialValues ) {
        // Create input and output tensors
        Tensor<float, DeviceMemoryResource> device_input( small_shape_ );
        Tensor<float, DeviceMemoryResource> device_output( small_shape_ );

        Tensor<float, HostMemoryResource> host_input( small_shape_ );
        Tensor<float, HostMemoryResource> host_output( small_shape_ );
        Tensor<float, HostMemoryResource> expected_output( small_shape_ );

        // Initialize input with special cases
        for ( int b = 0; b < small_batch_; b++ ) {
            for ( int t = 0; t < small_seq_len_; t++ ) {
                for ( int v = 0; v < small_vocab_size_; v++ ) {
                    int index = (b * small_seq_len_ + t) * small_vocab_size_ + v;

                    // Create different patterns for testing:
                    if ( b == 0 && t == 0 ) {
                        // Case 1: All equal values
                        host_input.data()[ index ] = 1.0f;
                    }
                    else if ( b == 0 && t == 1 ) {
                        // Case 2: One hot (one large value, others small)
                        host_input.data()[ index ] = (v == 2) ? 10.0f : 0.0f;
                    }
                    else if ( b == 1 && t == 0 ) {
                        // Case 3: Large negative values
                        host_input.data()[ index ] = -100.0f + v * 10.0f;
                    }
                    else if ( b == 1 && t == 1 ) {
                        // Case 4: Large positive values
                        host_input.data()[ index ] = 100.0f - v * 10.0f;
                    }
                    else {
                        // Case 5: Alternating positive/negative
                        host_input.data()[ index ] = (v % 2 == 0) ? v * 1.5f : -v * 1.5f;
                    }
                }
            }
        }

        // Copy to device
        device_input.copyFrom( host_input );

        // Execute operation
        std::vector<std::shared_ptr<Tensor<float, DeviceMemoryResource>>> params;
        std::vector<std::shared_ptr<Tensor<float, DeviceMemoryResource>>> output_cache;
        OperationAttributes props;
        props.axis = 2;  // Softmax along vocabulary dimension

        ASSERT_NO_THROW( cuda_softmax_op_->forward(
            device_input, params, props, device_output, output_cache ) );

        // Copy result back to host
        host_output.copyFrom( device_output );

        // Compute expected output with reference implementation
        referenceSoftmax( host_input, expected_output, props.axis );

        // Verify output has correct values
        EXPECT_TRUE( compareTensors( host_output, expected_output, 1e-4f ) );

        // Verify softmax output properties (sums to 1, non-negative values)
        EXPECT_TRUE( checkSoftmaxProperties( host_output, props.axis ) );

        // For case 1 (all equal inputs), verify uniform distribution
        float expected_uniform = 1.0f / small_vocab_size_;
        for ( int v = 0; v < small_vocab_size_; v++ ) {
            EXPECT_NEAR( host_output.data()[ v ], expected_uniform, 1e-5f );
        }

        // For case 2 (one hot), verify near one-hot output
        int hot_index = 2;
        for ( int v = 0; v < small_vocab_size_; v++ ) {
            int idx = small_vocab_size_ + v;  // Offset for case 2
            if ( v == hot_index ) {
                EXPECT_GT( host_output.data()[ idx ], 0.9f );  // Should be close to 1.0
            }
            else {
                EXPECT_LT( host_output.data()[ idx ], 0.1f );  // Should be close to 0.0
            }
        }
    }

    /**
     * @brief Test that CUDA and CPU implementations produce equivalent results
     */
    TEST_F( CudaSoftmaxOpTests, CudaCpuEquivalence ) {
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

        // Execute CUDA Softmax operation
        std::vector<std::shared_ptr<Tensor<float, DeviceMemoryResource>>> cuda_params;
        std::vector<std::shared_ptr<Tensor<float, DeviceMemoryResource>>> cuda_output_cache;
        OperationAttributes cuda_props;
        cuda_props.axis = 2;

        cuda_softmax_op_->forward( cuda_input, cuda_params, cuda_props, cuda_output, cuda_output_cache );

        // Execute CPU Softmax operation
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> cpu_params;
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> cpu_output_cache;
        OperationAttributes cpu_props;
        cpu_props.axis = 2;

        cpu_softmax_op_->forward( cpu_input, cpu_params, cpu_props, cpu_output, cpu_output_cache );

        // Copy CUDA result back to host
        cuda_output_host.copyFrom( cuda_output );

        // Compare results (with small tolerance for floating point differences)
        EXPECT_TRUE( compareTensors( cpu_output, cuda_output_host, 1e-4f ) );
    }

    /**
     * @brief Test backward pass functionality
     */
    TEST_F( CudaSoftmaxOpTests, BackwardPass ) {
        // Create tensors for forward and backward passes
        Tensor<float, DeviceMemoryResource> input( small_shape_ );
        Tensor<float, DeviceMemoryResource> output( small_shape_ );
        Tensor<float, DeviceMemoryResource> output_grad( small_shape_ );
        Tensor<float, DeviceMemoryResource> input_grad( small_shape_ );

        Tensor<float, HostMemoryResource> host_input( small_shape_ );
        Tensor<float, HostMemoryResource> host_output( small_shape_ );
        Tensor<float, HostMemoryResource> host_output_grad( small_shape_ );
        Tensor<float, HostMemoryResource> host_input_grad( small_shape_ );
        Tensor<float, HostMemoryResource> expected_input_grad( small_shape_ );

        // Initialize input
        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input.data()[ i ] = static_cast<float>( i % 7 ) - 3.0f;
        }

        // Initialize output gradient (normally from upstream layer)
        for ( size_t i = 0; i < host_output_grad.size(); ++i ) {
            host_output_grad.data()[ i ] = (static_cast<float>( i % 5 ) - 2.0f) * 0.5f;
        }

        // Copy to device
        input.copyFrom( host_input );
        output_grad.copyFrom( host_output_grad );

        // Forward pass first
        std::vector<std::shared_ptr<Tensor<float, DeviceMemoryResource>>> params;
        std::vector<std::shared_ptr<Tensor<float, DeviceMemoryResource>>> output_cache;
        OperationAttributes props;
        props.axis = 2;

        cuda_softmax_op_->forward( input, params, props, output, output_cache );

        // Copy output to host for reference calculation
        host_output.copyFrom( output );

        // Backward pass
        std::vector<std::shared_ptr<Tensor<float, DeviceMemoryResource>>> param_grads;

        ASSERT_NO_THROW( cuda_softmax_op_->backward(
            input, output, output_grad, params, param_grads,
            input_grad, props, output_cache ) );

        // Copy result back to host
        host_input_grad.copyFrom( input_grad );

        // Calculate expected input gradient using reference implementation
        referenceSoftmaxGradient( host_output, host_output_grad, expected_input_grad, props.axis );

        // Verify gradients match reference implementation
        EXPECT_TRUE( compareTensors( host_input_grad, expected_input_grad, 1e-4f ) );

        // Verify no NaN or Inf values
        EXPECT_FALSE( hasNaNorInf( host_input_grad ) );
    }

    /**
     * @brief Test numerical stability with varied inputs
     */
    TEST_F( CudaSoftmaxOpTests, NumericalStability ) {
        // Create test tensors
        Tensor<float, DeviceMemoryResource> device_input( medium_shape_ );
        Tensor<float, DeviceMemoryResource> device_output( medium_shape_ );

        Tensor<float, HostMemoryResource> host_input( medium_shape_ );
        Tensor<float, HostMemoryResource> host_output( medium_shape_ );

        // Test a variety of challenging input values
        for ( int b = 0; b < medium_batch_; b++ ) {
            for ( int t = 0; t < medium_seq_len_; t++ ) {
                for ( int v = 0; v < medium_vocab_size_; v++ ) {
                    int index = (b * medium_seq_len_ + t) * medium_vocab_size_ + v;

                    // Create different patterns based on position
                    int pattern = (b * 10 + t) % 8;
                    switch ( pattern ) {
                        case 0:  // Very large values
                            host_input.data()[ index ] = 1000.0f + v * 0.01f;
                            break;
                        case 1:  // Very small (negative) values
                            host_input.data()[ index ] = -1000.0f - v * 0.01f;
                            break;
                        case 2:  // Mix of large and small values
                            host_input.data()[ index ] = (v % 2 == 0) ? 100.0f : -100.0f;
                            break;
                        case 3:  // Very large differences
                            host_input.data()[ index ] = (v == medium_vocab_size_ / 2) ? 1000.0f : -1000.0f;
                            break;
                        case 4:  // Near zero values
                            host_input.data()[ index ] = (v - medium_vocab_size_ / 2) * 1e-5f;
                            break;
                        case 5:  // All identical
                            host_input.data()[ index ] = 42.0f;
                            break;
                        case 6:  // Alternating small increments
                            host_input.data()[ index ] = v * 0.001f;
                            break;
                        case 7:  // NaN and Inf tests (should be handled by kernel)
                            // Instead of actual NaN/Inf which would corrupt the test,
                            // use very large/small values that test the numerical stability
                            host_input.data()[ index ] = (v % 3 == 0) ? 1e30f : ((v % 3 == 1) ? -1e30f : 0.0f);
                            break;
                    }
                }
            }
        }

        // Copy to device
        device_input.copyFrom( host_input );

        // Execute operation
        std::vector<std::shared_ptr<Tensor<float, DeviceMemoryResource>>> params;
        std::vector<std::shared_ptr<Tensor<float, DeviceMemoryResource>>> output_cache;
        OperationAttributes props;
        props.axis = 2;

        ASSERT_NO_THROW( cuda_softmax_op_->forward(
            device_input, params, props, device_output, output_cache ) );

        // Copy result back to host
        host_output.copyFrom( device_output );

        // Verify no NaN or Inf values
        EXPECT_FALSE( hasNaNorInf( host_output ) );

        // Check softmax properties are maintained even with extreme inputs
        EXPECT_TRUE( checkSoftmaxProperties( host_output, props.axis ) );
    }

    /**
     * @brief Test deterministic behavior (multiple runs should produce same result)
     */
    TEST_F( CudaSoftmaxOpTests, DeterministicBehavior ) {
        // Create test tensors
        Tensor<float, DeviceMemoryResource> device_input( medium_shape_ );
        Tensor<float, DeviceMemoryResource> device_output1( medium_shape_ );
        Tensor<float, DeviceMemoryResource> device_output2( medium_shape_ );

        Tensor<float, HostMemoryResource> host_input( medium_shape_ );
        Tensor<float, HostMemoryResource> host_output1( medium_shape_ );
        Tensor<float, HostMemoryResource> host_output2( medium_shape_ );

        // Initialize with consistent values
        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input.data()[ i ] = (static_cast<float>( i % 17 ) - 8.5f) * 0.1f;
        }

        // Copy to device
        device_input.copyFrom( host_input );

        // Run twice with same input
        std::vector<std::shared_ptr<Tensor<float, DeviceMemoryResource>>> params;
        std::vector<std::shared_ptr<Tensor<float, DeviceMemoryResource>>> output_cache1;
        std::vector<std::shared_ptr<Tensor<float, DeviceMemoryResource>>> output_cache2;
        OperationAttributes props;
        props.axis = 2;

        cuda_softmax_op_->forward( device_input, params, props, device_output1, output_cache1 );
        cuda_softmax_op_->forward( device_input, params, props, device_output2, output_cache2 );

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
    TEST_F( CudaSoftmaxOpTests, DeviceTypeMismatch ) {
        // Attempt to create a CudaSoftmaxOp with a CPU context
        EXPECT_THROW( (CudaSoftmaxOp<float, float>( cpu_context_ )), std::runtime_error );
    }

    /**
     * @brief Test performance with large input
     */
    TEST_F( CudaSoftmaxOpTests, Performance ) {
        // Skip test if running in CI environment
        if ( std::getenv( "CI" ) != nullptr ) {
            GTEST_SKIP() << "Skipping performance test in CI environment";
        }

        // Create large test tensors
        Tensor<float, DeviceMemoryResource> device_input( large_shape_ );
        Tensor<float, DeviceMemoryResource> device_output( large_shape_ );

        Tensor<float, HostMemoryResource> host_input( large_shape_ );

        // Fill with random values
        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input.data()[ i ] = (static_cast<float>( rand() ) / RAND_MAX * 10.0f - 5.0f);
        }

        // Copy to device
        device_input.copyFrom( host_input );

        std::vector<std::shared_ptr<Tensor<float, DeviceMemoryResource>>> params;
        std::vector<std::shared_ptr<Tensor<float, DeviceMemoryResource>>> output_cache;
        OperationAttributes props;
        props.axis = 2;

        // Make sure everything is ready
        cudaDeviceSynchronize();

        // Measure performance over multiple iterations
        const int iterations = 100;
        auto start_time = std::chrono::high_resolution_clock::now();

        for ( int i = 0; i < iterations; ++i ) {
            cuda_softmax_op_->forward( device_input, params, props, device_output, output_cache );
        }

        // Wait for all operations to complete
        cudaDeviceSynchronize();

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>( end_time - start_time );

        // Calculate operations per softmax
        // This is approximate: find max (V ops), exp and sum (2*V ops), division (V ops)
        size_t ops_per_vector = large_vocab_size_ * 4;
        size_t vectors_per_iter = large_batch_ * large_seq_len_;
        size_t ops_per_iter = ops_per_vector * vectors_per_iter;
        size_t total_ops = ops_per_iter * iterations;

        // Calculate performance metrics
        double ops_per_second = static_cast<double>(total_ops) / (duration.count() * 1e-6);
        double gops = ops_per_second / 1e9;
        double avg_time_per_iter = duration.count() / iterations;

        // Record properties that will show in test details
        RecordProperty( "Performance_GOPS", std::to_string( gops ) );
        RecordProperty( "Average_Time_us", std::to_string( avg_time_per_iter ) );
        RecordProperty( "Implementation", "CUDA" );
        RecordProperty( "Batch_Size", std::to_string( large_batch_ ) );
        RecordProperty( "Sequence_Length", std::to_string( large_seq_len_ ) );
        RecordProperty( "Vocabulary_Size", std::to_string( large_vocab_size_ ) );

        std::cout << "CUDA Softmax Performance: " << gops << " GOPS" << std::endl;
        std::cout << "Average time per iteration: " << avg_time_per_iter << " microseconds" << std::endl;

        // No assertion, just informational
        EXPECT_TRUE( true );
    }

    /**
     * @brief Test comparison with CPU performance
     */
    TEST_F( CudaSoftmaxOpTests, CpuCudaPerformanceComparison ) {
        // Skip test if running in CI environment
        if ( std::getenv( "CI" ) != nullptr ) {
            GTEST_SKIP() << "Skipping performance comparison test in CI environment";
        }

        // Create test tensors for CUDA
        Tensor<float, DeviceMemoryResource> cuda_input( medium_shape_ );
        Tensor<float, DeviceMemoryResource> cuda_output( medium_shape_ );

        // Create test tensors for CPU
        Tensor<float, HostMemoryResource> cpu_input( medium_shape_ );
        Tensor<float, HostMemoryResource> cpu_output( medium_shape_ );

        // Initialize with random values
        for ( size_t i = 0; i < cpu_input.size(); ++i ) {
            cpu_input.data()[ i ] = (static_cast<float>( rand() ) / RAND_MAX * 10.0f - 5.0f);
        }

        // Copy to CUDA
        cuda_input.copyFrom( cpu_input );

        // Setup parameters
        std::vector<std::shared_ptr<Tensor<float, DeviceMemoryResource>>> cuda_params;
        std::vector<std::shared_ptr<Tensor<float, DeviceMemoryResource>>> cuda_output_cache;
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> cpu_params;
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> cpu_output_cache;
        OperationAttributes props;
        props.axis = 2;

        // Measure CUDA performance
        cudaDeviceSynchronize();
        const int iterations = 100;
        auto cuda_start_time = std::chrono::high_resolution_clock::now();

        for ( int i = 0; i < iterations; ++i ) {
            cuda_softmax_op_->forward( cuda_input, cuda_params, props, cuda_output, cuda_output_cache );
        }

        cudaDeviceSynchronize();
        auto cuda_end_time = std::chrono::high_resolution_clock::now();
        auto cuda_duration = std::chrono::duration_cast<std::chrono::microseconds>( cuda_end_time - cuda_start_time );

        // Measure CPU performance
        auto cpu_start_time = std::chrono::high_resolution_clock::now();

        for ( int i = 0; i < iterations; ++i ) {
            cpu_softmax_op_->forward( cpu_input, cpu_params, props, cpu_output, cpu_output_cache );
        }

        auto cpu_end_time = std::chrono::high_resolution_clock::now();
        auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>( cpu_end_time - cpu_start_time );

        // Calculate operations per softmax (same as Performance test)
        size_t ops_per_vector = medium_vocab_size_ * 4;
        size_t vectors_per_iter = medium_batch_ * medium_seq_len_;
        size_t ops_per_iter = ops_per_vector * vectors_per_iter;

        double cuda_gops = static_cast<double>(ops_per_iter * iterations) / (cuda_duration.count() * 1e-6) / 1e9;
        double cpu_gops = static_cast<double>(ops_per_iter * iterations) / (cpu_duration.count() * 1e-6) / 1e9;
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
}
