/**
 * @file CpuFullyConnectedOpTests.cpp
 * @brief Test suite for the CPU Fully Connected operation.
 */

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <chrono>
#include <string>
#include <iostream>
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
     * @brief Test fixture for CpuFullyConnectedOp tests
     */
    class CpuFullyConnectedOpTests : public ::testing::Test {
    protected:
        void SetUp() override {
            cpu_context_ = std::make_shared<DeviceContext>( "CPU" );

            // Small shapes for quick tests
            small_batch_ = 2;
            small_seq_len_ = 3;
            small_in_features_ = 4;
            small_out_features_ = 5;
            small_input_shape_ = { small_batch_, small_seq_len_, small_in_features_ };
            small_output_shape_ = { small_batch_, small_seq_len_, small_out_features_ };
            small_weight_shape_ = { small_out_features_, small_in_features_ };
            small_bias_shape_ = { small_out_features_ };

            // Medium shapes for more thorough tests
            medium_batch_ = 8;
            medium_seq_len_ = 16;
            medium_in_features_ = 32;
            medium_out_features_ = 24;
            medium_input_shape_ = { medium_batch_, medium_seq_len_, medium_in_features_ };
            medium_output_shape_ = { medium_batch_, medium_seq_len_, medium_out_features_ };
            medium_weight_shape_ = { medium_out_features_, medium_in_features_ };
            medium_bias_shape_ = { medium_out_features_ };

            // Large shapes for stress tests
            large_batch_ = 32;
            large_seq_len_ = 64;
            large_in_features_ = 128;
            large_out_features_ = 96;
            large_input_shape_ = { large_batch_, large_seq_len_, large_in_features_ };
            large_output_shape_ = { large_batch_, large_seq_len_, large_out_features_ };
            large_weight_shape_ = { large_out_features_, large_in_features_ };
            large_bias_shape_ = { large_out_features_ };

            // Create CPU FullyConnected operation with specific context
            cpu_fc_op_ = std::make_shared<CpuFullyConnectedOp>( cpu_context_ );
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

        // Helper method for reference implementation of fully connected operation
        void referenceFullyConnected(
            const Tensor<float, HostMemoryResource>& input,
            const Tensor<float, HostMemoryResource>& weights,
            const Tensor<float, HostMemoryResource>* bias,
            Tensor<float, HostMemoryResource>& output ) {

            int B = input.shape()[ 0 ];   // Batch size
            int T = input.shape()[ 1 ];   // Sequence length
            int C = input.shape()[ 2 ];   // Input features
            int OC = output.shape()[ 2 ]; // Output features

            // For each batch and time step
            for ( int b = 0; b < B; b++ ) {
                for ( int t = 0; t < T; t++ ) {
                    int bt = b * T + t;
                    // For each output feature
                    for ( int o = 0; o < OC; o++ ) {
                        float val = bias ? bias->data()[ o ] : 0.0f;
                        // Multiply input features with weights
                        for ( int i = 0; i < C; i++ ) {
                            val += input.data()[ bt * C + i ] * weights.data()[ o * C + i ];
                        }
                        output.data()[ bt * OC + o ] = val;
                    }
                }
            }
        }

        std::shared_ptr<DeviceContext> cpu_context_;
        std::shared_ptr<CpuFullyConnectedOp> cpu_fc_op_;

        // Test dimensions
        size_t small_batch_, small_seq_len_, small_in_features_, small_out_features_;
        size_t medium_batch_, medium_seq_len_, medium_in_features_, medium_out_features_;
        size_t large_batch_, large_seq_len_, large_in_features_, large_out_features_;

        // Test shapes
        std::vector<size_t> small_input_shape_, small_output_shape_, small_weight_shape_, small_bias_shape_;
        std::vector<size_t> medium_input_shape_, medium_output_shape_, medium_weight_shape_, medium_bias_shape_;
        std::vector<size_t> large_input_shape_, large_output_shape_, large_weight_shape_, large_bias_shape_;
    };

    /**
     * @brief Test name property of CpuFullyConnectedOp
     */
    TEST_F( CpuFullyConnectedOpTests, Name ) {
        EXPECT_EQ( cpu_fc_op_->getName(), "Cpu::FullyConnectedOp" );
    }

    /**
     * @brief Test basic functionality of CpuFullyConnectedOp without bias
     */
    TEST_F( CpuFullyConnectedOpTests, BasicFunctionalityWithoutBias ) {
        // Create input, weight, and output tensors
        Tensor<float, HostMemoryResource> input( small_input_shape_ );
        auto weights = std::make_shared<Tensor<float, HostMemoryResource>>( small_weight_shape_ );
        Tensor<float, HostMemoryResource> output( small_output_shape_ );
        Tensor<float, HostMemoryResource> expected_output( small_output_shape_ );

        // Initialize input with test values
        for ( size_t i = 0; i < input.size(); ++i ) {
            input.data()[ i ] = (static_cast<float>( i ) - 10.0f) / 10.0f; // Range from -1.0 to +1.0
        }

        // Initialize weights
        for ( size_t i = 0; i < weights->size(); ++i ) {
            weights->data()[ i ] = (static_cast<float>( i % 7 ) - 3.0f) / 3.0f;
        }

        // Execute operation
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> params = { weights };
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> output_state;
        OperationAttributes props;

        ASSERT_NO_THROW( cpu_fc_op_->forward( input, params, props, output, output_state ) );

        // Compute expected output with reference implementation
        referenceFullyConnected( input, *weights, nullptr, expected_output );

        // Verify output has correct values
        for ( size_t i = 0; i < output.size(); ++i ) {
            EXPECT_NEAR( output.data()[ i ], expected_output.data()[ i ], 1e-5f );
        }
    }

    /**
     * @brief Test basic functionality of CpuFullyConnectedOp with bias
     */
    TEST_F( CpuFullyConnectedOpTests, BasicFunctionalityWithBias ) {
        // Create input, weight, bias, and output tensors
        Tensor<float, HostMemoryResource> input( small_input_shape_ );
        auto weights = std::make_shared<Tensor<float, HostMemoryResource>>( small_weight_shape_ );
        auto bias = std::make_shared<Tensor<float, HostMemoryResource>>( small_bias_shape_ );
        Tensor<float, HostMemoryResource> output( small_output_shape_ );
        Tensor<float, HostMemoryResource> expected_output( small_output_shape_ );

        // Initialize input with test values
        for ( size_t i = 0; i < input.size(); ++i ) {
            input.data()[ i ] = (static_cast<float>( i ) - 10.0f) / 10.0f;
        }

        // Initialize weights and bias
        for ( size_t i = 0; i < weights->size(); ++i ) {
            weights->data()[ i ] = (static_cast<float>( i % 7 ) - 3.0f) / 3.0f;
        }

        for ( size_t i = 0; i < bias->size(); ++i ) {
            bias->data()[ i ] = (static_cast<float>( i ) - 2.0f) / 5.0f;
        }

        // Execute operation
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> params = { weights, bias };
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> output_state;
        OperationAttributes props;

        ASSERT_NO_THROW( cpu_fc_op_->forward( input, params, props, output, output_state ) );

        // Compute expected output with reference implementation
        referenceFullyConnected( input, *weights, bias.get(), expected_output );

        // Verify output has correct values
        for ( size_t i = 0; i < output.size(); ++i ) {
            EXPECT_NEAR( output.data()[ i ], expected_output.data()[ i ], 1e-5f );
        }
    }

    /**
     * @brief Test backward pass functionality
     */
    //TEST_F( CpuFullyConnectedOpTests, BackwardPass ) {
    //    // Create tensors for forward and backward passes
    //    Tensor<float, HostMemoryResource> input( small_input_shape_ );
    //    Tensor<float, HostMemoryResource> weights( small_weight_shape_ );
    //    Tensor<float, HostMemoryResource> bias( small_bias_shape_ );

    //    Tensor<float, HostMemoryResource> input_grad( small_input_shape_ );
    //    Tensor<float, HostMemoryResource> weight_grad( small_weight_shape_ );
    //    Tensor<float, HostMemoryResource> bias_grad( small_bias_shape_ );
    //    Tensor<float, HostMemoryResource> output_grad( small_output_shape_ );

    //    // Initialize tensors with test values
    //    for ( size_t i = 0; i < input.size(); ++i ) {
    //        input.data()[ i ] = (static_cast<float>( i ) - 10.0f) / 10.0f;
    //        input_grad.data()[ i ] = 0.0f;
    //    }

    //    for ( size_t i = 0; i < weights.size(); ++i ) {
    //        weights.data()[ i ] = (static_cast<float>( i % 7 ) - 3.0f) / 3.0f;
    //        weight_grad.data()[ i ] = 0.0f;
    //    }

    //    for ( size_t i = 0; i < bias.size(); ++i ) {
    //        bias.data()[ i ] = (static_cast<float>( i ) - 2.0f) / 5.0f;
    //        bias_grad.data()[ i ] = 0.0f;
    //    }

    //    // Set all output gradients to 1.0 for easier verification
    //    for ( size_t i = 0; i < output_grad.size(); ++i ) {
    //        output_grad.data()[ i ] = 1.0f;
    //    }

    //    // Call backward method
    //    int B = small_batch_;
    //    int T = small_seq_len_;
    //    int C = small_in_features_;
    //    int OC = small_out_features_;

    //    ASSERT_NO_THROW( cpu_fc_op_->backward(
    //        input_grad.data(),
    //        weight_grad.data(),
    //        bias_grad.data(),
    //        output_grad.data(),
    //        input.data(),
    //        weights.data(),
    //        B, T, C, OC ) );

    //    // Verify gradients are not NaN or Inf
    //    EXPECT_FALSE( hasNaNorInf( input_grad ) );
    //    EXPECT_FALSE( hasNaNorInf( weight_grad ) );
    //    EXPECT_FALSE( hasNaNorInf( bias_grad ) );

    //    // For bias_grad, verify that each element is equal to B*T (sum of all 1.0's over batch and time)
    //    for ( size_t o = 0; o < bias_grad.size(); ++o ) {
    //        EXPECT_FLOAT_EQ( bias_grad.data()[ o ], B * T );
    //    }

    //    // For weight_grad, verify using a manual calculation
    //    for ( int o = 0; o < OC; o++ ) {
    //        for ( int i = 0; i < C; i++ ) {
    //            float expected_grad = 0.0f;
    //            for ( int b = 0; b < B; b++ ) {
    //                for ( int t = 0; t < T; t++ ) {
    //                    expected_grad += input.data()[ (b * T + t) * C + i ];
    //                }
    //            }
    //            EXPECT_NEAR( weight_grad.data()[ o * C + i ], expected_grad, 1e-5f );
    //        }
    //    }

    //    // For input_grad, verify using a manual calculation
    //    for ( int b = 0; b < B; b++ ) {
    //        for ( int t = 0; t < T; t++ ) {
    //            int bt = b * T + t;
    //            for ( int i = 0; i < C; i++ ) {
    //                float expected_grad = 0.0f;
    //                for ( int o = 0; o < OC; o++ ) {
    //                    expected_grad += weights.data()[ o * C + i ] * 1.0f; // output_grad is all 1's
    //                }
    //                EXPECT_NEAR( input_grad.data()[ bt * C + i ], expected_grad, 1e-5f );
    //            }
    //        }
    //    }
    //}

    /**
     * @brief Test edge case with non-standard batch size that doesn't divide by LOOP_UNROLL
     */
    TEST_F( CpuFullyConnectedOpTests, NonStandardBatchSize ) {
        // Create a shape that won't divide evenly by the LOOP_UNROLL factor (which is 8)
        std::vector<size_t> odd_input_shape = { 3, 5, small_in_features_ };
        std::vector<size_t> odd_output_shape = { 3, 5, small_out_features_ };

        Tensor<float, HostMemoryResource> input( odd_input_shape );
        auto weights = std::make_shared<Tensor<float, HostMemoryResource>>( small_weight_shape_ );
        auto bias = std::make_shared<Tensor<float, HostMemoryResource>>( small_bias_shape_ );
        Tensor<float, HostMemoryResource> output( odd_output_shape );
        Tensor<float, HostMemoryResource> expected_output( odd_output_shape );

        // Initialize tensors
        for ( size_t i = 0; i < input.size(); ++i ) {
            input.data()[ i ] = (static_cast<float>( i ) - 10.0f) / 10.0f;
        }

        for ( size_t i = 0; i < weights->size(); ++i ) {
            weights->data()[ i ] = (static_cast<float>( i % 7 ) - 3.0f) / 3.0f;
        }

        for ( size_t i = 0; i < bias->size(); ++i ) {
            bias->data()[ i ] = (static_cast<float>( i ) - 2.0f) / 5.0f;
        }

        // Execute operation
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> params = { weights, bias };
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> output_state;
        OperationAttributes props;

        ASSERT_NO_THROW( cpu_fc_op_->forward( input, params, props, output, output_state ) );

        // Compute expected output with reference implementation
        referenceFullyConnected( input, *weights, bias.get(), expected_output );

        // Verify output matches reference implementation
        for ( size_t i = 0; i < output.size(); ++i ) {
            EXPECT_NEAR( output.data()[ i ], expected_output.data()[ i ], 1e-5f );
        }
    }

    /**
     * @brief Test numerical stability with varied inputs
     */
    TEST_F( CpuFullyConnectedOpTests, NumericalStability ) {
        // Create test tensors
        Tensor<float, HostMemoryResource> input( medium_input_shape_ );
        auto weights = std::make_shared<Tensor<float, HostMemoryResource>>( medium_weight_shape_ );
        auto bias = std::make_shared<Tensor<float, HostMemoryResource>>( medium_bias_shape_ );
        Tensor<float, HostMemoryResource> output( medium_output_shape_ );

        // Test a variety of input values: positive, negative, small, large
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

        // Test a variety of weight values
        for ( size_t i = 0; i < weights->size(); ++i ) {
            float val;
            int pattern = i % 6;
            switch ( pattern ) {
                case 0: val = 0.1f; break;
                case 1: val = -0.1f; break;
                case 2: val = 0.001f; break;
                case 3: val = -0.001f; break;
                case 4: val = 1.0f; break;
                case 5: val = -1.0f; break;
                default: val = 0.0f; break;
            }
            weights->data()[ i ] = val;
        }

        // Initialize bias
        for ( size_t i = 0; i < bias->size(); ++i ) {
            bias->data()[ i ] = (i % 2 == 0) ? 0.5f : -0.5f;
        }

        // Execute operation
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> params = { weights, bias };
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> output_state;
        OperationAttributes props;

        ASSERT_NO_THROW( cpu_fc_op_->forward( input, params, props, output, output_state ) );

        // Verify no NaN or Inf values
        EXPECT_FALSE( hasNaNorInf( output ) );
    }

    /**
     * @brief Test deterministic behavior (multiple runs should produce same result)
     */
    TEST_F( CpuFullyConnectedOpTests, DeterministicBehavior ) {
        // Create test tensors
        Tensor<float, HostMemoryResource> input( medium_input_shape_ );
        auto weights = std::make_shared<Tensor<float, HostMemoryResource>>( medium_weight_shape_ );
        auto bias = std::make_shared<Tensor<float, HostMemoryResource>>( medium_bias_shape_ );
        Tensor<float, HostMemoryResource> output1( medium_output_shape_ );
        Tensor<float, HostMemoryResource> output2( medium_output_shape_ );

        // Initialize with consistent values
        for ( size_t i = 0; i < input.size(); ++i ) {
            input.data()[ i ] = (static_cast<float>( i % 17 ) - 8.5f) * 0.1f;
        }

        for ( size_t i = 0; i < weights->size(); ++i ) {
            weights->data()[ i ] = (static_cast<float>( i % 13 ) - 6.5f) * 0.05f;
        }

        for ( size_t i = 0; i < bias->size(); ++i ) {
            bias->data()[ i ] = (static_cast<float>( i % 7 ) - 3.5f) * 0.1f;
        }

        // Run twice with same input
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> params = { weights, bias };
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> output_state1;
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> output_state2;
        OperationAttributes props;

        cpu_fc_op_->forward( input, params, props, output1, output_state1 );
        cpu_fc_op_->forward( input, params, props, output2, output_state2 );

        // Results should be identical
        for ( size_t i = 0; i < output1.size(); ++i ) {
            EXPECT_EQ( output1.data()[ i ], output2.data()[ i ] );
        }
    }

    /**
     * @brief Test error handling for device type mismatch
     */
    TEST_F( CpuFullyConnectedOpTests, DeviceTypeMismatch ) {
        // Try to create with CUDA:0 if available, otherwise skip the test
        try {
            EXPECT_THROW( CpuFullyConnectedOp( std::make_shared<DeviceContext>( "CUDA:0" ) ), std::runtime_error );
        }
        catch ( const std::exception& e ) {
            GTEST_SKIP() << "CUDA device not available, skipping device mismatch test: " << e.what();
        }
    }

    /**
     * @brief Test performance with large input
     */
    TEST_F( CpuFullyConnectedOpTests, Performance ) {
        // Skip test if running in CI environment
        if ( std::getenv( "CI" ) != nullptr ) {
            GTEST_SKIP() << "Skipping performance test in CI environment";
        }

        // Create large test tensors
        Tensor<float, HostMemoryResource> input( large_input_shape_ );
        auto weights = std::make_shared<Tensor<float, HostMemoryResource>>( large_weight_shape_ );
        auto bias = std::make_shared<Tensor<float, HostMemoryResource>>( large_bias_shape_ );
        Tensor<float, HostMemoryResource> output( large_output_shape_ );

        // Fill with random values
        for ( size_t i = 0; i < input.size(); ++i ) {
            input.data()[ i ] = (static_cast<float>( rand() ) / RAND_MAX * 2.0f - 1.0f) * 10.0f;
        }

        for ( size_t i = 0; i < weights->size(); ++i ) {
            weights->data()[ i ] = (static_cast<float>( rand() ) / RAND_MAX * 2.0f - 1.0f) * 0.1f;
        }

        for ( size_t i = 0; i < bias->size(); ++i ) {
            bias->data()[ i ] = (static_cast<float>( rand() ) / RAND_MAX * 2.0f - 1.0f);
        }

        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> params = { weights, bias };
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> output_state;
        OperationAttributes props;

        // Measure performance over multiple iterations
        const int iterations = 10;
        auto start_time = std::chrono::high_resolution_clock::now();

        for ( int i = 0; i < iterations; ++i ) {
            cpu_fc_op_->forward( input, params, props, output, output_state );
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>( end_time - start_time );
        
        // Calculate FLOPs (floating point operations)
        // Each output element requires C multiply-adds, so 2*C FLOPs
        size_t flops_per_iter = large_batch_ * large_seq_len_ * large_out_features_ * (2 * large_in_features_);
        size_t total_flops = flops_per_iter * iterations;

        // Calculate performance metrics
        double flops_per_second = static_cast<double>(total_flops) / (duration.count() * 1e-6);
        double gflops = flops_per_second / 1e9;
        double avg_time_per_iter = duration.count() / iterations;

        // Record properties that will show in test details
        RecordProperty( "Performance_GFLOPS", std::to_string( gflops ) );
        RecordProperty( "Average_Time_us", std::to_string( avg_time_per_iter ) );
        RecordProperty( "Implementation", "CPU" );
        RecordProperty( "Batch_Size", std::to_string( large_batch_ ) );
        RecordProperty( "Sequence_Length", std::to_string( large_seq_len_ ) );
        RecordProperty( "Input_Features", std::to_string( large_in_features_ ) );
        RecordProperty( "Output_Features", std::to_string( large_out_features_ ) );

        // No assertion, just informational
        EXPECT_TRUE( true );
		std::cout << "Performance: " << gflops << " GFLOPS, "
			<< "Average Time: " << avg_time_per_iter << " us" << std::endl;
    }

    /**
     * @brief Test with OpenMP (if available)
     */
    TEST_F( CpuFullyConnectedOpTests, OpenMPScaling ) {
    #ifdef USE_OMP
        // Skip test if running in CI environment
        if ( std::getenv( "CI" ) != nullptr ) {
            GTEST_SKIP() << "Skipping OpenMP scaling test in CI environment";
        }

        // Create large test tensors
        Tensor<float, HostMemoryResource> input( large_input_shape_ );
        auto weights = std::make_shared<Tensor<float, HostMemoryResource>>( large_weight_shape_ );
        auto bias = std::make_shared<Tensor<float, HostMemoryResource>>( large_bias_shape_ );
        Tensor<float, HostMemoryResource> output( large_output_shape_ );

        // Fill with random values
        for ( size_t i = 0; i < input.size(); ++i ) {
            input.data()[ i ] = (static_cast<float>( rand() ) / RAND_MAX * 2.0f - 1.0f) * 10.0f;
        }

        for ( size_t i = 0; i < weights->size(); ++i ) {
            weights->data()[ i ] = (static_cast<float>( rand() ) / RAND_MAX * 2.0f - 1.0f) * 0.1f;
        }

        for ( size_t i = 0; i < bias->size(); ++i ) {
            bias->data()[ i ] = (static_cast<float>( rand() ) / RAND_MAX * 2.0f - 1.0f);
        }

        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> params = { weights, bias };
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
        size_t flops_per_iter = large_batch_ * large_seq_len_ * large_out_features_ * (2 * large_in_features_);

        for ( int num_threads : thread_counts ) {
            omp_set_num_threads( num_threads );

            auto start_time = std::chrono::high_resolution_clock::now();

            for ( int i = 0; i < iterations; ++i ) {
                cpu_fc_op_->forward( input, params, props, output, output_state );
            }

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>( end_time - start_time );

            double flops_per_second = static_cast<double>( flops_per_iter * iterations ) /
                (duration.count() * 1e-6);

            std::cout << "CPU FullyConnected with " << num_threads << " threads: "
                << flops_per_second / 1e9 << " GFLOPS" << std::endl;
        }
    #else
        // GTEST_SKIP() << "OpenMP not available, skipping scaling test";
    #endif
    }
}