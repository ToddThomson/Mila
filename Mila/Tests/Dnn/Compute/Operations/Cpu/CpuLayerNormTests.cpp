/**
 * @file CpuLayerNormTests.cpp
 * @brief Test suite for the CPU LayerNorm operation.
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
     * @brief Test fixture for CpuLayerNormOp tests
     */
    class CpuLayerNormOpTests : public ::testing::Test {
    protected:
        void SetUp() override {
            cpu_context_ = std::make_shared<DeviceContext>( "CPU" );

            // Small shapes for quick tests
            small_shape_ = { 2, 3, 4 };

            // Medium shapes for more thorough tests
            medium_shape_ = { 8, 16, 32 };

            // Large shape for stress tests
            large_shape_ = { 32, 64, 128 };

            // Create CPU LayerNorm operation with specific context
            cpu_layernorm_op_ = std::make_shared<CpuLayerNormOp>( cpu_context_ );

            // Initialize parameters for LayerNorm operation
            small_parameters_ = createParameters( small_shape_[ 2 ] );
            medium_parameters_ = createParameters( medium_shape_[ 2 ] );
            large_parameters_ = createParameters( large_shape_[ 2 ] );
        }

        // Helper method to create weight and bias parameters for LayerNorm
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> createParameters( size_t feature_dim ) {
            std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> params;

            // Weight tensor (gamma)
            auto weight = std::make_shared<Tensor<float, HostMemoryResource>>( std::vector<size_t>{feature_dim} );
            for ( size_t i = 0; i < feature_dim; ++i ) {
                weight->data()[ i ] = 1.0f; // Initialize with ones
            }
            params.push_back( weight );

            // Bias tensor (beta)
            auto bias = std::make_shared<Tensor<float, HostMemoryResource>>( std::vector<size_t>{feature_dim} );
            for ( size_t i = 0; i < feature_dim; ++i ) {
                bias->data()[ i ] = 0.0f; // Initialize with zeros
            }
            params.push_back( bias );

            return params;
        }

        // Helper method to prepare output state tensors
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> createOutputState( size_t batch_size, size_t sequence_length ) {
            std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> output_state;

            // Mean tensor
            auto mean = std::make_shared<Tensor<float, HostMemoryResource>>( std::vector<size_t>{batch_size, sequence_length} );
            output_state.push_back( mean );

            // Reciprocal standard deviation tensor
            auto rstd = std::make_shared<Tensor<float, HostMemoryResource>>( std::vector<size_t>{batch_size, sequence_length} );
            output_state.push_back( rstd );

            return output_state;
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

        // Reference implementation of LayerNorm for validation
        void layerNormReference(
            const Tensor<float, HostMemoryResource>& input,
            const std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>>& params,
            float epsilon,
            Tensor<float, HostMemoryResource>& output,
            std::vector<float>& means,
            std::vector<float>& rstds ) {

            int B = input.shape()[ 0 ]; // batch size
            int T = input.shape()[ 1 ]; // sequence length
            int C = input.shape()[ 2 ]; // feature dimension

            const float* weight = params[ 0 ]->raw_data();
            const float* bias = params[ 1 ]->raw_data();

            means.resize( B * T );
            rstds.resize( B * T );

            for ( int b = 0; b < B; b++ ) {
                for ( int t = 0; t < T; t++ ) {
                    // Calculate mean
                    float mean = 0.0f;
                    int input_offset = b * T * C + t * C;
                    for ( int i = 0; i < C; i++ ) {
                        mean += input.raw_data()[ input_offset + i ];
                    }
                    mean /= C;

                    // Calculate variance
                    float variance = 0.0f;
                    for ( int i = 0; i < C; i++ ) {
                        float diff = input.raw_data()[ input_offset + i ] - mean;
                        variance += diff * diff;
                    }
                    variance /= C;

                    // Calculate reciprocal standard deviation
                    float rstd = 1.0f / sqrtf( variance + epsilon );

                    // Store mean and rstd for validation
                    means[ b * T + t ] = mean;
                    rstds[ b * T + t ] = rstd;

                    // Normalize, scale, and shift
                    for ( int i = 0; i < C; i++ ) {
                        float normalized = (input.raw_data()[ input_offset + i ] - mean) * rstd;
                        output.data()[ input_offset + i ] = normalized * weight[ i ] + bias[ i ];
                    }
                }
            }
        }

        std::shared_ptr<DeviceContext> cpu_context_;
        std::shared_ptr<CpuLayerNormOp> cpu_layernorm_op_;

        std::vector<size_t> small_shape_;
        std::vector<size_t> medium_shape_;
        std::vector<size_t> large_shape_;

        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> small_parameters_;
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> medium_parameters_;
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> large_parameters_;
    };

    /**
     * @brief Test name property of CpuLayerNormOp
     */
    TEST_F( CpuLayerNormOpTests, Name ) {
        EXPECT_EQ( cpu_layernorm_op_->getName(), "Cpu::LayerNormOp" );
    }

    /**
     * @brief Test basic functionality of CpuLayerNormOp
     */
    TEST_F( CpuLayerNormOpTests, BasicFunctionality ) {
        // Create input and output tensors
        Tensor<float, HostMemoryResource> input( small_shape_ );
        Tensor<float, HostMemoryResource> output( small_shape_ );

        // Initialize input with test values
        for ( size_t i = 0; i < input.size(); ++i ) {
            input.data()[ i ] = (static_cast<float>( i ) - 10.0f) / 10.0f; // Range from -1.0 to +1.0
        }

        // Create reference output and cache for mean/rstd
        Tensor<float, HostMemoryResource> expected_output( small_shape_ );
        std::vector<float> expected_means;
        std::vector<float> expected_rstds;

        // Calculate reference output
        float epsilon = 1e-5f;
        layerNormReference( input, small_parameters_, epsilon, expected_output, expected_means, expected_rstds );

        // Execute LayerNorm operation
        auto output_state = createOutputState( small_shape_[ 0 ], small_shape_[ 1 ] );
        OperationAttributes props;
        props.set( "epsilon", epsilon );

        ASSERT_NO_THROW( cpu_layernorm_op_->forward( input, small_parameters_, props, output, output_state ) );

        // Verify output has correct values
        for ( size_t i = 0; i < output.size(); ++i ) {
            EXPECT_NEAR( output.data()[ i ], expected_output.data()[ i ], 1e-5f );
        }

        // Verify cached mean and rstd values
        float* mean_data = output_state[ 0 ]->data();
        float* rstd_data = output_state[ 1 ]->data();

        for ( size_t i = 0; i < small_shape_[ 0 ] * small_shape_[ 1 ]; ++i ) {
            EXPECT_NEAR( mean_data[ i ], expected_means[ i ], 1e-5f );
            EXPECT_NEAR( rstd_data[ i ], expected_rstds[ i ], 1e-5f );
        }
    }

    /**
     * @brief Test backward pass functionality
     */
    //TEST_F( CpuLayerNormOpTests, BackwardPass ) {
    //    // Create tensors
    //    Tensor<float, HostMemoryResource> input( small_shape_ );
    //    Tensor<float, HostMemoryResource> output( small_shape_ );
    //    Tensor<float, HostMemoryResource> output_grad( small_shape_ );
    //    Tensor<float, HostMemoryResource> input_grad( small_shape_ );

    //    // Initialize tensors
    //    for ( size_t i = 0; i < input.size(); ++i ) {
    //        input.data()[ i ] = (static_cast<float>( i ) - 10.0f) / 10.0f;
    //        input_grad.data()[ i ] = 0.0f;
    //        output_grad.data()[ i ] = 1.0f;  // Set all output gradients to 1.0
    //    }

    //    // Forward pass to populate output_state
    //    auto output_state = createOutputState( small_shape_[ 0 ], small_shape_[ 1 ] );
    //    OperationAttributes props;
    //    props.set( "epsilon", 1e-5f );

    //    cpu_layernorm_op_->forward( input, small_parameters_, props, output, output_state );

    //    // Initialize tensors to accumulate gradients for weight and bias
    //    Tensor<float, HostMemoryResource> dweight( std::vector<size_t>{small_shape_[ 2 ]} );
    //    Tensor<float, HostMemoryResource> dbias( std::vector<size_t>{small_shape_[ 2 ]} );

    //    for ( size_t i = 0; i < small_shape_[ 2 ]; ++i ) {
    //        dweight.data()[ i ] = 0.0f;
    //        dbias.data()[ i ] = 0.0f;
    //    }

    //    // Execute backward pass
    //    ASSERT_NO_THROW( cpu_layernorm_op_->backward(
    //        input_grad.data(),
    //        dweight.data(), dbias.data(),
    //        output_grad.data(),
    //        input.data(), small_parameters_[ 0 ]->data(),
    //        output_state[ 0 ]->data(), output_state[ 1 ]->data(),
    //        small_shape_[ 0 ], small_shape_[ 1 ], small_shape_[ 2 ] ) );

    //    // Verify gradients are not NaN or Inf
    //    EXPECT_FALSE( hasNaNorInf( input_grad ) );
    //    EXPECT_FALSE( hasNaNorInf( dweight ) );
    //    EXPECT_FALSE( hasNaNorInf( dbias ) );

    //    // Bias gradient should be the sum of output gradients across batch and sequence dims
    //    for ( size_t i = 0; i < small_shape_[ 2 ]; ++i ) {
    //        float expected_dbias = small_shape_[ 0 ] * small_shape_[ 1 ];  // Sum of 1.0's
    //        EXPECT_NEAR( dbias.data()[ i ], expected_dbias, 1e-5f );
    //    }

    //    // For standard input values, gradients should be non-zero
    //    bool all_zeros = true;
    //    for ( size_t i = 0; i < input_grad.size(); ++i ) {
    //        if ( std::abs( input_grad.data()[ i ] ) > 1e-5f ) {
    //            all_zeros = false;
    //            break;
    //        }
    //    }
    //    EXPECT_FALSE( all_zeros );

    //    all_zeros = true;
    //    for ( size_t i = 0; i < dweight.size(); ++i ) {
    //        if ( std::abs( dweight.data()[ i ] ) > 1e-5f ) {
    //            all_zeros = false;
    //            break;
    //        }
    //    }
    //    EXPECT_FALSE( all_zeros );
    //}

    /**
     * @brief Test different epsilon values for numerical stability
     */
    TEST_F( CpuLayerNormOpTests, EpsilonValues ) {
        // Create input and output tensors
        Tensor<float, HostMemoryResource> input( small_shape_ );
        Tensor<float, HostMemoryResource> output( small_shape_ );

        // Initialize input with test values
        for ( size_t i = 0; i < input.size(); ++i ) {
            input.data()[ i ] = (static_cast<float>( i ) - 10.0f) / 10.0f;
        }

        auto output_state = createOutputState( small_shape_[ 0 ], small_shape_[ 1 ] );
        OperationAttributes props;

        // Test with different epsilon values
        std::vector<float> epsilon_values = { 1e-3f, 1e-5f, 1e-7f, 1e-9f };

        for ( float epsilon : epsilon_values ) {
            props.set( "epsilon", epsilon );

            ASSERT_NO_THROW( cpu_layernorm_op_->forward( input, small_parameters_, props, output, output_state ) );

            // Calculate reference output
            Tensor<float, HostMemoryResource> expected_output( small_shape_ );
            std::vector<float> expected_means;
            std::vector<float> expected_rstds;
            layerNormReference( input, small_parameters_, epsilon, expected_output, expected_means, expected_rstds );

            // Verify outputs match reference
            for ( size_t i = 0; i < output.size(); ++i ) {
                EXPECT_NEAR( output.data()[ i ], expected_output.data()[ i ], 1e-5f );
            }

            // Verify no NaN or Inf values
            EXPECT_FALSE( hasNaNorInf( output ) );
        }
    }

    /**
     * @brief Test edge cases with zero, very small, and very large values
     */
    TEST_F( CpuLayerNormOpTests, EdgeCases ) {
        // Create test tensors
        Tensor<float, HostMemoryResource> input( small_shape_ );
        Tensor<float, HostMemoryResource> output( small_shape_ );

        auto output_state = createOutputState( small_shape_[ 0 ], small_shape_[ 1 ] );
        OperationAttributes props;
        props.set( "epsilon", 1e-5f );

        // Test 1: All zeros
        for ( size_t i = 0; i < input.size(); ++i ) {
            input.data()[ i ] = 0.0f;
        }

        ASSERT_NO_THROW( cpu_layernorm_op_->forward( input, small_parameters_, props, output, output_state ) );

        // With all zeros input, all outputs should be zero
        for ( size_t i = 0; i < output.size(); ++i ) {
            EXPECT_NEAR( output.data()[ i ], 0.0f, 1e-5f );
        }

        // Test 2: All identical values
        for ( size_t i = 0; i < input.size(); ++i ) {
            input.data()[ i ] = 1.0f;
        }

        ASSERT_NO_THROW( cpu_layernorm_op_->forward( input, small_parameters_, props, output, output_state ) );

        // With all identical inputs, normalized values should be zero before scale and shift
        // After adding the bias (which is 0), outputs should be 0
        for ( size_t i = 0; i < output.size(); ++i ) {
            EXPECT_NEAR( output.data()[ i ], 0.0f, 1e-5f );
        }

        // Test 3: Very large values
        for ( size_t i = 0; i < input.size(); ++i ) {
            input.data()[ i ] = 1e6f + static_cast<float>( i );  // large but slightly different
        }

        ASSERT_NO_THROW( cpu_layernorm_op_->forward( input, small_parameters_, props, output, output_state ) );
        EXPECT_FALSE( hasNaNorInf( output ) );

        // Test 4: Very small values
        for ( size_t i = 0; i < input.size(); ++i ) {
            input.data()[ i ] = 1e-6f * static_cast<float>( i + 1 );
        }

        ASSERT_NO_THROW( cpu_layernorm_op_->forward( input, small_parameters_, props, output, output_state ) );
        EXPECT_FALSE( hasNaNorInf( output ) );
    }

    /**
     * @brief Test with custom weights and biases
     */
    TEST_F( CpuLayerNormOpTests, CustomWeightsBiases ) {
        // Create test tensors
        Tensor<float, HostMemoryResource> input( small_shape_ );
        Tensor<float, HostMemoryResource> output( small_shape_ );

        // Initialize input with test values
        for ( size_t i = 0; i < input.size(); ++i ) {
            input.data()[ i ] = (static_cast<float>( i ) - 10.0f) / 10.0f;
        }

        // Create custom weights and biases
        auto custom_params = createParameters( small_shape_[ 2 ] );
        for ( size_t i = 0; i < small_shape_[ 2 ]; ++i ) {
            custom_params[ 0 ]->data()[ i ] = 2.0f;  // Custom weight
            custom_params[ 1 ]->data()[ i ] = 1.0f;  // Custom bias
        }

        auto output_state = createOutputState( small_shape_[ 0 ], small_shape_[ 1 ] );
        OperationAttributes props;
        props.set( "epsilon", 1e-5f );

        ASSERT_NO_THROW( cpu_layernorm_op_->forward( input, custom_params, props, output, output_state ) );

        // Calculate reference output with custom weights/biases
        Tensor<float, HostMemoryResource> expected_output( small_shape_ );
        std::vector<float> expected_means;
        std::vector<float> expected_rstds;
        layerNormReference( input, custom_params, 1e-5f, expected_output, expected_means, expected_rstds );

        // Verify outputs match reference
        for ( size_t i = 0; i < output.size(); ++i ) {
            EXPECT_NEAR( output.data()[ i ], expected_output.data()[ i ], 1e-5f );
        }
    }

    /**
     * @brief Test numerical stability with varied inputs
     */
    TEST_F( CpuLayerNormOpTests, NumericalStability ) {
        // Create test tensors
        Tensor<float, HostMemoryResource> input( medium_shape_ );
        Tensor<float, HostMemoryResource> output( medium_shape_ );

        auto output_state = createOutputState( medium_shape_[ 0 ], medium_shape_[ 1 ] );
        OperationAttributes props;
        props.set( "epsilon", 1e-5f );

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

        ASSERT_NO_THROW( cpu_layernorm_op_->forward( input, medium_parameters_, props, output, output_state ) );

        // Verify no NaN or Inf values
        EXPECT_FALSE( hasNaNorInf( output ) );
    }

    /**
     * @brief Test deterministic behavior (multiple runs should produce same result)
     */
    TEST_F( CpuLayerNormOpTests, DeterministicBehavior ) {
        // Create test tensors
        Tensor<float, HostMemoryResource> input( medium_shape_ );
        Tensor<float, HostMemoryResource> output1( medium_shape_ );
        Tensor<float, HostMemoryResource> output2( medium_shape_ );

        // Initialize with consistent values
        for ( size_t i = 0; i < input.size(); ++i ) {
            input.data()[ i ] = (static_cast<float>( i % 17 ) - 8.5f) * 0.1f;
        }

        auto output_state1 = createOutputState( medium_shape_[ 0 ], medium_shape_[ 1 ] );
        auto output_state2 = createOutputState( medium_shape_[ 0 ], medium_shape_[ 1 ] );
        OperationAttributes props;
        props.set( "epsilon", 1e-5f );

        // Run twice with same input
        cpu_layernorm_op_->forward( input, medium_parameters_, props, output1, output_state1 );
        cpu_layernorm_op_->forward( input, medium_parameters_, props, output2, output_state2 );

        // Results should be identical
        for ( size_t i = 0; i < output1.size(); ++i ) {
            EXPECT_EQ( output1.data()[ i ], output2.data()[ i ] );
        }
    }

    /**
     * @brief Test error handling for device type mismatch
     */
    TEST_F( CpuLayerNormOpTests, DeviceTypeMismatch ) {
        // Try to create with CUDA:0 if available, otherwise skip the test
        try {
            EXPECT_THROW( CpuLayerNormOp( std::make_shared<DeviceContext>( "CUDA:0" ) ), std::runtime_error );
        }
        catch ( const std::exception& e ) {
            GTEST_SKIP() << "CUDA device not available, skipping device mismatch test: " << e.what();
        }
    }

    /**
     * @brief Test performance with large input
     */
    TEST_F( CpuLayerNormOpTests, Performance ) {
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

        auto output_state = createOutputState( large_shape_[ 0 ], large_shape_[ 1 ] );
        OperationAttributes props;
        props.set( "epsilon", 1e-5f );

        // Measure performance over multiple iterations
        const int iterations = 10;
        auto start_time = std::chrono::high_resolution_clock::now();

        for ( int i = 0; i < iterations; ++i ) {
            cpu_layernorm_op_->forward( input, large_parameters_, props, output, output_state );
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>( end_time - start_time );

        // Report performance metrics
        size_t total_elements = input.size() * iterations;
        double elements_per_second = static_cast<double>( total_elements ) / (duration.count() * 1e-6);
        std::cout << "CPU LayerNorm Performance: " << elements_per_second / 1e6 << " million elements/sec" << std::endl;
        std::cout << "Average time per iteration: " << duration.count() / iterations << " microseconds" << std::endl;
    }

    /**
     * @brief Test with OpenMP (if available)
     */
    TEST_F( CpuLayerNormOpTests, OpenMPScaling ) {
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

        auto output_state = createOutputState( large_shape_[ 0 ], large_shape_[ 1 ] );
        OperationAttributes props;
        props.set( "epsilon", 1e-5f );

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
                cpu_layernorm_op_->forward( input, large_parameters_, props, output, output_state );
            }

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>( end_time - start_time );

            double elements_per_second = static_cast<double>( input.size() * iterations ) /
                (duration.count() * 1e-6);

            std::cout << "CPU LayerNorm with " << num_threads << " threads: "
                << elements_per_second / 1e6 << " million elements/sec" << std::endl;
        }
    #else
        // GTEST_SKIP() << "OpenMP not available, skipping scaling test";
    #endif
    }
}
