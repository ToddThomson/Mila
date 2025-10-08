/**
 * @file CpuSoftmaxOpTests.cpp
 * @brief Test suite for the CPU Softmax operation.
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
     * @brief Test fixture for CpuSoftmaxOp tests
     */
    class CpuSoftmaxOpTests : public ::testing::Test {
    protected:
        void SetUp() override {
            cpu_context_ = std::make_shared<DeviceContext>( "CPU" );

            // Small shapes for quick tests
            small_shape_ = { 2, 3, 4 };

            // Medium shapes for more thorough tests
            medium_shape_ = { 8, 16, 32 };

            // Large shape for stress tests
            large_shape_ = { 32, 64, 128 };

            // Create CPU Softmax operation with specific context
            cpu_softmax_op_ = std::make_shared<CpuSoftmaxOp>( cpu_context_ );
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

        // Helper method to compute softmax manually for verification
        std::vector<float> computeSoftmax( const std::vector<float>& input, size_t size ) {
            std::vector<float> output( size );

            // Find max value for numerical stability
            float max_val = *std::max_element( input.begin(), input.end() );

            // Compute exp(x - max_val) and sum
            float sum = 0.0f;
            for ( size_t i = 0; i < size; ++i ) {
                output[ i ] = std::exp( input[ i ] - max_val );
                sum += output[ i ];
            }

            // Normalize by sum
            for ( size_t i = 0; i < size; ++i ) {
                output[ i ] /= sum;
            }

            return output;
        }

        // Helper method to check if the output sums to 1.0
        bool checkProbabilityDistribution( const Tensor<float, HostMemoryResource>& tensor,
            int axis, float tolerance = 1e-5f ) {
            const int64_t ndim = tensor.shape().size();
            if ( axis < 0 ) {
                axis = ndim + axis;
            }

            // Calculate outer_size and inner_size similar to the op implementation
            int64_t outer_size = 1;
            for ( int64_t i = 0; i < axis; ++i ) {
                outer_size *= tensor.shape()[ i ];
            }

            const int64_t dim_size = tensor.shape()[ axis ];

            int64_t inner_size = 1;
            for ( int64_t i = axis + 1; i < ndim; ++i ) {
                inner_size *= tensor.shape()[ i ];
            }

            // Check that each distribution sums to 1.0
            for ( int64_t outer = 0; outer < outer_size; ++outer ) {
                for ( int64_t inner = 0; inner < inner_size; ++inner ) {
                    float sum = 0.0f;
                    for ( int64_t i = 0; i < dim_size; ++i ) {
                        int64_t idx = (outer * dim_size * inner_size) + (i * inner_size) + inner;
                        sum += static_cast<const float*>( tensor.data() )[ idx ];
                    }
                    if ( std::abs( sum - 1.0f ) > tolerance ) {
                        std::cout << "Distribution does not sum to 1.0. Sum: " << sum << std::endl;
                        return false;
                    }
                }
            }
            return true;
        }

        std::shared_ptr<DeviceContext> cpu_context_;
        std::shared_ptr<CpuSoftmaxOp> cpu_softmax_op_;

        std::vector<size_t> small_shape_;
        std::vector<size_t> medium_shape_;
        std::vector<size_t> large_shape_;
    };

    /**
     * @brief Test name property of CpuSoftmaxOp
     */
    TEST_F( CpuSoftmaxOpTests, Name ) {
        EXPECT_EQ( cpu_softmax_op_->getDeviceName(), "Cpu::SoftmaxOp" );
    }

    /**
     * @brief Test basic functionality of CpuSoftmaxOp with axis=2 (last dimension)
     */
    TEST_F( CpuSoftmaxOpTests, BasicFunctionalityLastAxis ) {
        // Create input and output tensors
        Tensor<float, HostMemoryResource> input( small_shape_ );
        Tensor<float, HostMemoryResource> output( small_shape_ );

        // Initialize input with test values
        for ( size_t i = 0; i < input.size(); ++i ) {
            input.data()[ i ] = (static_cast<float>( i ) - 10.0f) / 10.0f;
        }

        // Execute Softmax operation with axis=2 (last dimension)
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> params;
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> output_state;
        OperationAttributes props;
        props.axis = 2;  // Last dimension

        ASSERT_NO_THROW( cpu_softmax_op_->forward( input, params, props, output, output_state ) );

        // Verify output has no NaN or Inf values
        EXPECT_FALSE( hasNaNorInf( output ) );

        // Verify each probability distribution sums to 1.0
        EXPECT_TRUE( checkProbabilityDistribution( output, 2 ) );

        // Verify values against manual computation for a few samples
        for ( size_t b = 0; b < small_shape_[ 0 ]; ++b ) {
            for ( size_t t = 0; t < small_shape_[ 1 ]; ++t ) {
                // Extract the vector for this batch and sequence position
                std::vector<float> input_vec;
                std::vector<float> output_vec;
                for ( size_t c = 0; c < small_shape_[ 2 ]; ++c ) {
                    size_t idx = (b * small_shape_[ 1 ] * small_shape_[ 2 ]) + (t * small_shape_[ 2 ]) + c;
                    input_vec.push_back( input.data()[ idx ] );
                    output_vec.push_back( output.data()[ idx ] );
                }

                // Compute expected softmax result
                std::vector<float> expected = computeSoftmax( input_vec, small_shape_[ 2 ] );

                // Compare computed vs expected
                for ( size_t c = 0; c < small_shape_[ 2 ]; ++c ) {
                    EXPECT_NEAR( output_vec[ c ], expected[ c ], 1e-5f );
                }
            }
        }
    }

    /**
     * @brief Test softmax along the middle axis
     */
    TEST_F( CpuSoftmaxOpTests, MiddleAxisFunctionality ) {
        // Create input and output tensors
        Tensor<float, HostMemoryResource> input( small_shape_ );
        Tensor<float, HostMemoryResource> output( small_shape_ );

        // Initialize input with test values
        for ( size_t i = 0; i < input.size(); ++i ) {
            input.data()[ i ] = (static_cast<float>( i ) - 10.0f) / 10.0f;
        }

        // Execute Softmax operation with axis=1 (middle dimension)
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> params;
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> output_state;
        OperationAttributes props;
        props.axis = 1;  // Middle dimension

        ASSERT_NO_THROW( cpu_softmax_op_->forward( input, params, props, output, output_state ) );

        // Verify output has no NaN or Inf values
        EXPECT_FALSE( hasNaNorInf( output ) );

        // Verify each probability distribution sums to 1.0
        EXPECT_TRUE( checkProbabilityDistribution( output, 1 ) );
    }

    /**
     * @brief Test softmax with negative axis index
     */
    TEST_F( CpuSoftmaxOpTests, NegativeAxisIndex ) {
        // Create input and output tensors
        Tensor<float, HostMemoryResource> input( small_shape_ );
        Tensor<float, HostMemoryResource> output( small_shape_ );

        // Initialize input with test values
        for ( size_t i = 0; i < input.size(); ++i ) {
            input.data()[ i ] = (static_cast<float>( i ) - 10.0f) / 10.0f;
        }

        // Execute Softmax operation with axis=-1 (last dimension)
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> params;
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> output_state;
        OperationAttributes props;
        props.axis = -1;  // Last dimension

        ASSERT_NO_THROW( cpu_softmax_op_->forward( input, params, props, output, output_state ) );

        // Verify output has no NaN or Inf values
        EXPECT_FALSE( hasNaNorInf( output ) );

        // Verify each probability distribution sums to 1.0
        EXPECT_TRUE( checkProbabilityDistribution( output, -1 ) );
    }

    /**
     * @brief Test softmax with first axis
     */
    TEST_F( CpuSoftmaxOpTests, FirstAxisFunctionality ) {
        // Create input and output tensors
        Tensor<float, HostMemoryResource> input( small_shape_ );
        Tensor<float, HostMemoryResource> output( small_shape_ );

        // Initialize input with test values
        for ( size_t i = 0; i < input.size(); ++i ) {
            input.data()[ i ] = (static_cast<float>( i ) - 10.0f) / 10.0f;
        }

        // Execute Softmax operation with axis=0 (first dimension)
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> params;
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> output_state;
        OperationAttributes props;
        props.axis = 0;  // First dimension

        ASSERT_NO_THROW( cpu_softmax_op_->forward( input, params, props, output, output_state ) );

        // Verify output has no NaN or Inf values
        EXPECT_FALSE( hasNaNorInf( output ) );

        // Verify each probability distribution sums to 1.0
        EXPECT_TRUE( checkProbabilityDistribution( output, 0 ) );
    }

    /**
     * @brief Test backward pass functionality
     */
    //TEST_F( CpuSoftmaxOpTests, BackwardPass ) {
    //    // Create tensors for forward and backward passes
    //    Tensor<float, HostMemoryResource> input( small_shape_ );
    //    Tensor<float, HostMemoryResource> output( small_shape_ );
    //    Tensor<float, HostMemoryResource> output_grad( small_shape_ );
    //    Tensor<float, HostMemoryResource> input_grad( small_shape_ );

    //    // Initialize tensors
    //    for ( size_t i = 0; i < input.size(); ++i ) {
    //        input.data()[ i ] = (static_cast<float>( i ) - 10.0f) / 10.0f;
    //        input_grad.data()[ i ] = 0.0f;  // Initialize to zero
    //        // Set all output gradients to 1.0 for simple testing
    //        output_grad.data()[ i ] = 1.0f;
    //    }

    //    // Forward pass first
    //    std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> params;
    //    std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> output_state;
    //    std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> param_grads;
    //    OperationAttributes props;
    //    props.axis = 2;  // Last dimension

    //    cpu_softmax_op_->forward( input, params, props, output, output_state );

    //    // Execute backward pass
    //    ASSERT_NO_THROW( cpu_softmax_op_->backward(
    //        input, output, output_grad,
    //        params, param_grads, input_grad,
    //        props, output_state ) );

    //    // Verify gradients are not NaN or Inf
    //    EXPECT_FALSE( hasNaNorInf( input_grad ) );

    //    // For standard input values, gradients should be non-zero
    //    bool all_zeros = true;
    //    for ( size_t i = 0; i < input_grad.size(); ++i ) {
    //        if ( std::abs( input_grad.data()[ i ] ) > 1e-5f ) {
    //            all_zeros = false;
    //            break;
    //        }
    //    }
    //    EXPECT_FALSE( all_zeros );

    //    // For uniform output gradients, verify the gradients sum to zero within each softmax distribution
    //    // This is a characteristic of the softmax gradient when all output gradients are equal
    //    for ( size_t b = 0; b < small_shape_[ 0 ]; ++b ) {
    //        for ( size_t t = 0; t < small_shape_[ 1 ]; ++t ) {
    //            float sum = 0.0f;
    //            for ( size_t c = 0; c < small_shape_[ 2 ]; ++c ) {
    //                size_t idx = (b * small_shape_[ 1 ] * small_shape_[ 2 ]) + (t * small_shape_[ 2 ]) + c;
    //                sum += input_grad.data()[ idx ];
    //            }
    //            EXPECT_NEAR( sum, 0.0f, 1e-5f );
    //        }
    //    }
    //}

    /**
     * @brief Test edge cases with extreme input values
     */
    TEST_F( CpuSoftmaxOpTests, EdgeCases ) {
        // Create test tensors
        Tensor<float, HostMemoryResource> input( small_shape_ );
        Tensor<float, HostMemoryResource> output( small_shape_ );

        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> params;
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> output_state;
        OperationAttributes props;
        props.axis = 2;  // Last dimension

        // Test 1: All zeros
        for ( size_t i = 0; i < input.size(); ++i ) {
            input.data()[ i ] = 0.0f;
        }

        ASSERT_NO_THROW( cpu_softmax_op_->forward( input, params, props, output, output_state ) );

        // With all zeros input, softmax should produce uniform distributions
        for ( size_t b = 0; b < small_shape_[ 0 ]; ++b ) {
            for ( size_t t = 0; t < small_shape_[ 1 ]; ++t ) {
                float uniform_value = 1.0f / small_shape_[ 2 ];
                for ( size_t c = 0; c < small_shape_[ 2 ]; ++c ) {
                    size_t idx = (b * small_shape_[ 1 ] * small_shape_[ 2 ]) + (t * small_shape_[ 2 ]) + c;
                    EXPECT_FLOAT_EQ( output.data()[ idx ], uniform_value );
                }
            }
        }

        // Test 2: Very large positive values in some positions
        for ( size_t i = 0; i < input.size(); ++i ) {
            input.data()[ i ] = -1000.0f;  // Very negative value as default
        }

        // Set one position per distribution to a large positive value
        for ( size_t b = 0; b < small_shape_[ 0 ]; ++b ) {
            for ( size_t t = 0; t < small_shape_[ 1 ]; ++t ) {
                size_t hot_pos = (b + t) % small_shape_[ 2 ];  // Different position for each distribution
                size_t idx = (b * small_shape_[ 1 ] * small_shape_[ 2 ]) + (t * small_shape_[ 2 ]) + hot_pos;
                input.data()[ idx ] = 1000.0f;  // Very large positive value
            }
        }

        ASSERT_NO_THROW( cpu_softmax_op_->forward( input, params, props, output, output_state ) );

        // Output should be approximately one-hot encoded
        for ( size_t b = 0; b < small_shape_[ 0 ]; ++b ) {
            for ( size_t t = 0; t < small_shape_[ 1 ]; ++t ) {
                size_t hot_pos = (b + t) % small_shape_[ 2 ];
                for ( size_t c = 0; c < small_shape_[ 2 ]; ++c ) {
                    size_t idx = (b * small_shape_[ 1 ] * small_shape_[ 2 ]) + (t * small_shape_[ 2 ]) + c;
                    if ( c == hot_pos ) {
                        EXPECT_NEAR( output.data()[ idx ], 1.0f, 1e-5f );
                    }
                    else {
                        EXPECT_NEAR( output.data()[ idx ], 0.0f, 1e-5f );
                    }
                }
            }
        }

        // Test 3: All identical values
        for ( size_t i = 0; i < input.size(); ++i ) {
            input.data()[ i ] = 1.5f;
        }

        ASSERT_NO_THROW( cpu_softmax_op_->forward( input, params, props, output, output_state ) );

        // With identical inputs, softmax should produce uniform distributions
        for ( size_t b = 0; b < small_shape_[ 0 ]; ++b ) {
            for ( size_t t = 0; t < small_shape_[ 1 ]; ++t ) {
                float uniform_value = 1.0f / small_shape_[ 2 ];
                for ( size_t c = 0; c < small_shape_[ 2 ]; ++c ) {
                    size_t idx = (b * small_shape_[ 1 ] * small_shape_[ 2 ]) + (t * small_shape_[ 2 ]) + c;
                    EXPECT_NEAR( output.data()[ idx ], uniform_value, 1e-5f );
                }
            }
        }
    }

    /**
     * @brief Test numerical stability with large positive and negative values
     */
    TEST_F( CpuSoftmaxOpTests, NumericalStability ) {
        // Create test tensors
        Tensor<float, HostMemoryResource> input( medium_shape_ );
        Tensor<float, HostMemoryResource> output( medium_shape_ );

        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> params;
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> output_state;
        OperationAttributes props;
        props.axis = 2;  // Last dimension

        // Test a variety of values: positive, negative, small, large
        for ( size_t i = 0; i < input.size(); ++i ) {
            float val;
            int pattern = i % 8;
            switch ( pattern ) {
                case 0: val = 1.0f; break;
                case 1: val = -1.0f; break;
                case 2: val = 0.0001f; break;
                case 3: val = -0.0001f; break;
                case 4: val = 100.0f; break;
                case 5: val = -100.0f; break;
                case 6: val = 1000.0f; break;
                case 7: val = -1000.0f; break;
                default: val = 0.0f; break;
            }
            input.data()[ i ] = val;
        }

        ASSERT_NO_THROW( cpu_softmax_op_->forward( input, params, props, output, output_state ) );

        // Verify no NaN or Inf values
        EXPECT_FALSE( hasNaNorInf( output ) );

        // Verify each probability distribution sums to 1.0
        EXPECT_TRUE( checkProbabilityDistribution( output, 2 ) );

        // Output values should be between 0 and 1
        for ( size_t i = 0; i < output.size(); ++i ) {
            EXPECT_GE( output.data()[ i ], 0.0f );
            EXPECT_LE( output.data()[ i ], 1.0f );
        }
    }

    /**
     * @brief Test deterministic behavior (multiple runs should produce same result)
     */
    TEST_F( CpuSoftmaxOpTests, DeterministicBehavior ) {
        // Create test tensors
        Tensor<float, HostMemoryResource> input( medium_shape_ );
        Tensor<float, HostMemoryResource> output1( medium_shape_ );
        Tensor<float, HostMemoryResource> output2( medium_shape_ );

        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> params;
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> output_state1;
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> output_state2;
        OperationAttributes props;
        props.axis = 2;  // Last dimension

        // Initialize with consistent values
        for ( size_t i = 0; i < input.size(); ++i ) {
            input.data()[ i ] = (static_cast<float>( i % 17 ) - 8.5f) * 0.1f;
        }

        // Run twice with same input
        cpu_softmax_op_->forward( input, params, props, output1, output_state1 );
        cpu_softmax_op_->forward( input, params, props, output2, output_state2 );

        // Results should be identical
        for ( size_t i = 0; i < output1.size(); ++i ) {
            EXPECT_EQ( output1.data()[ i ], output2.data()[ i ] );
        }
    }

    /**
     * @brief Test error handling for device type mismatch
     */
    TEST_F( CpuSoftmaxOpTests, DeviceTypeMismatch ) {
        // Try to create with CUDA:0 if available, otherwise skip the test
        try {
            EXPECT_THROW( CpuSoftmaxOp( std::make_shared<DeviceContext>( "CUDA:0" ) ), std::runtime_error );
        }
        catch ( const std::exception& e ) {
            GTEST_SKIP() << "CUDA device not available, skipping device mismatch test: " << e.what();
        }
    }

    /**
     * @brief Test error handling for invalid axis
     */
    TEST_F( CpuSoftmaxOpTests, InvalidAxis ) {
        // Create test tensors
        Tensor<float, HostMemoryResource> input( small_shape_ );
        Tensor<float, HostMemoryResource> output( small_shape_ );

        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> params;
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> output_state;

        // Test with out-of-bounds positive axis
        OperationAttributes props_pos;
        props_pos.axis = 3;  // 3-dimensional tensor, so axis 3 is out of bounds
        EXPECT_THROW( cpu_softmax_op_->forward( input, params, props_pos, output, output_state ), std::runtime_error );

        // Test with out-of-bounds negative axis
        OperationAttributes props_neg;
        props_neg.axis = -4;  // 3-dimensional tensor, so axis -4 is out of bounds
        EXPECT_THROW( cpu_softmax_op_->forward( input, params, props_neg, output, output_state ), std::runtime_error );
    }

    /**
     * @brief Test performance with large input
     */
    TEST_F( CpuSoftmaxOpTests, Performance ) {
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

        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> params;
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> output_state;
        OperationAttributes props;
        props.axis = 2;  // Last dimension

        // Measure performance over multiple iterations
        const int iterations = 10;
        auto start_time = std::chrono::high_resolution_clock::now();

        for ( int i = 0; i < iterations; ++i ) {
            cpu_softmax_op_->forward( input, params, props, output, output_state );
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>( end_time - start_time );

        // Report performance metrics
        size_t total_elements = input.size() * iterations;
        double elements_per_second = static_cast<double>( total_elements ) / (duration.count() * 1e-6);
        std::cout << "CPU Softmax Performance: " << elements_per_second / 1e6 << " million elements/sec" << std::endl;
        std::cout << "Average time per iteration: " << duration.count() / iterations << " microseconds" << std::endl;
    }

    /**
     * @brief Test with OpenMP (if available)
     */
    TEST_F( CpuSoftmaxOpTests, OpenMPScaling ) {
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

        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> params;
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> output_state;
        OperationAttributes props;
        props.axis = 2;  // Last dimension

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
                cpu_softmax_op_->forward( input, params, props, output, output_state );
            }

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>( end_time - start_time );

            double elements_per_second = static_cast<double>( input.size() * iterations ) /
                (duration.count() * 1e-6);

            std::cout << "CPU Softmax with " << num_threads << " threads: "
                << elements_per_second / 1e6 << " million elements/sec" << std::endl;
        }
    #else
        // GTEST_SKIP() << "OpenMP not available, skipping scaling test";
    #endif
    }

    /**
     * @brief Test with 2D tensors (common use case)
     */
    TEST_F( CpuSoftmaxOpTests, TwoDimensionalTensor ) {
        // Create 2D tensor (batch_size x features)
        std::vector<size_t> shape_2d = { 32, 64 };
        Tensor<float, HostMemoryResource> input( shape_2d );
        Tensor<float, HostMemoryResource> output( shape_2d );

        // Fill with values
        for ( size_t i = 0; i < input.size(); ++i ) {
            input.data()[ i ] = (static_cast<float>( i % 10 ) - 5.0f) * 0.5f;
        }

        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> params;
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> output_state;
        OperationAttributes props;
        props.axis = 1;  // Features dimension (commonly used for classification)

        ASSERT_NO_THROW( cpu_softmax_op_->forward( input, params, props, output, output_state ) );

        // Verify output has no NaN or Inf values
        EXPECT_FALSE( hasNaNorInf( output ) );

        // Verify each probability distribution sums to 1.0
        EXPECT_TRUE( checkProbabilityDistribution( output, 1 ) );

        // Verify output values using manual computation for a few samples
        for ( size_t b = 0; b < 3; ++b ) {  // Check first 3 batches
            // Extract the vector for this batch
            std::vector<float> input_vec;
            std::vector<float> output_vec;
            for ( size_t f = 0; f < shape_2d[ 1 ]; ++f ) {
                size_t idx = b * shape_2d[ 1 ] + f;
                input_vec.push_back( input.data()[ idx ] );
                output_vec.push_back( output.data()[ idx ] );
            }

            // Compute expected softmax result
            std::vector<float> expected = computeSoftmax( input_vec, shape_2d[ 1 ] );

            // Compare computed vs expected
            for ( size_t f = 0; f < shape_2d[ 1 ]; ++f ) {
                EXPECT_NEAR( output_vec[ f ], expected[ f ], 1e-5f );
            }
        }
    }
}

