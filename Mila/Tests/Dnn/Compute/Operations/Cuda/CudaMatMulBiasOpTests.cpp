/**
 * @file CudaFullyConnectedOpTests.cpp
 * @brief Test suite for the CUDA Fully Connected operation.
 */

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <chrono>
#include <string>
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
     * @brief Test fixture for CudaFullyConnectedOp tests
     */
    class CudaFullyConnectedOpTests : public ::testing::Test {
    protected:
        void SetUp() override {
            cuda_context_ = std::make_shared<DeviceContext>( "CUDA:0" );
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

            // Create CUDA FullyConnected operation with specific context
            cuda_fc_op_ = std::make_shared<CudaFullyConnectedOp<float>>( cuda_context_ );

            // Get CPU FullyConnected op for comparison
            auto cpu_op = OperationRegistry::instance().createOperation<float, float, DeviceType::Cpu>(
                "Cpu::FullyConnectedOp", cpu_context_ );
            cpu_fc_op_ = std::static_pointer_cast<UnaryOperation<float, float, DeviceType::Cpu>>(cpu_op);
        }

        // Helper method to reference implementation of fully connected operation
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
        std::shared_ptr<CudaFullyConnectedOp<float>> cuda_fc_op_;
        std::shared_ptr<UnaryOperation<float, float, DeviceType::Cpu>> cpu_fc_op_;

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
     * @brief Test name property of CudaFullyConnectedOp
     */
    TEST_F( CudaFullyConnectedOpTests, Name ) {
        EXPECT_EQ( cuda_fc_op_->getName(), "Cuda::FullyConnectedOp" );
    }

    /**
     * @brief Test basic functionality of CudaFullyConnectedOp without bias
     */
    TEST_F( CudaFullyConnectedOpTests, BasicFunctionalityWithoutBias ) {
        // Create input, weight, and output tensors
        Tensor<float, CudaMemoryResource> device_input( small_input_shape_ );
        auto device_weights = std::make_shared<Tensor<float, CudaMemoryResource>>( small_weight_shape_ );
        Tensor<float, CudaMemoryResource> device_output( small_output_shape_ );

        Tensor<float, HostMemoryResource> host_input( small_input_shape_ );
        Tensor<float, HostMemoryResource> host_weights( small_weight_shape_ );
        Tensor<float, HostMemoryResource> host_output( small_output_shape_ );
        Tensor<float, HostMemoryResource> expected_output( small_output_shape_ );

        // Initialize input with test values
        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input.data()[ i ] = (static_cast<float>( i ) - 10.0f) / 10.0f; // Range from -1.0 to +1.0
        }

        // Initialize weights
        for ( size_t i = 0; i < host_weights.size(); ++i ) {
            host_weights.data()[ i ] = (static_cast<float>( i % 7 ) - 3.0f) / 3.0f;
        }

        // Copy to device
        device_input.copyFrom( host_input );
        device_weights->copyFrom( host_weights );

        // Execute operation
        std::vector<std::shared_ptr<Tensor<float, CudaMemoryResource>>> params = { device_weights };
        std::vector<std::shared_ptr<Tensor<float, CudaMemoryResource>>> output_state;
        OperationAttributes props;

        ASSERT_NO_THROW( cuda_fc_op_->forward( device_input, params, props, device_output, output_state ) );

        // Copy result back to host
        host_output.copyFrom( device_output );
        //host_output.toString( true );

        // Compute expected output with reference implementation
        referenceFullyConnected( host_input, host_weights, nullptr, expected_output );

        // Verify output has correct values
        EXPECT_TRUE( compareTensors( host_output, expected_output, 1e-4f ) );
    }

    /**
     * @brief Test basic functionality of CudaFullyConnectedOp with bias
     */
    TEST_F( CudaFullyConnectedOpTests, BasicFunctionalityWithBias ) {
        // Create input, weight, bias, and output tensors
        Tensor<float, CudaMemoryResource> device_input( small_input_shape_ );
        auto device_weights = std::make_shared<Tensor<float, CudaMemoryResource>>( small_weight_shape_ );
        auto device_bias = std::make_shared<Tensor<float, CudaMemoryResource>>( small_bias_shape_ );
        Tensor<float, CudaMemoryResource> device_output( small_output_shape_ );

        Tensor<float, HostMemoryResource> host_input( small_input_shape_ );
        Tensor<float, HostMemoryResource> host_weights( small_weight_shape_ );
        Tensor<float, HostMemoryResource> host_bias( small_bias_shape_ );
        Tensor<float, HostMemoryResource> host_output( small_output_shape_ );
        Tensor<float, HostMemoryResource> expected_output( small_output_shape_ );

        // Initialize input with test values
        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input.data()[ i ] = (static_cast<float>( i ) - 10.0f) / 10.0f;
        }

        // Initialize weights and bias
        for ( size_t i = 0; i < host_weights.size(); ++i ) {
            host_weights.data()[ i ] = (static_cast<float>( i % 7 ) - 3.0f) / 3.0f;
        }

        for ( size_t i = 0; i < host_bias.size(); ++i ) {
            host_bias.data()[ i ] = (static_cast<float>( i ) - 2.0f) / 5.0f;
        }

        // Copy to device
        device_input.copyFrom( host_input );
        device_weights->copyFrom( host_weights );
        device_bias->copyFrom( host_bias );

        // Execute operation
        std::vector<std::shared_ptr<Tensor<float, CudaMemoryResource>>> params = { device_weights, device_bias };
        std::vector<std::shared_ptr<Tensor<float, CudaMemoryResource>>> output_state;
        OperationAttributes props;

        ASSERT_NO_THROW( cuda_fc_op_->forward( device_input, params, props, device_output, output_state ) );

        // Copy result back to host
        host_output.copyFrom( device_output );

        // Compute expected output with reference implementation
        referenceFullyConnected( host_input, host_weights, &host_bias, expected_output );

        // Verify output has correct values
        EXPECT_TRUE( compareTensors( host_output, expected_output, 1e-4f ) );
    }

    /**
     * @brief Test that CUDA and CPU implementations produce equivalent results
     */
    TEST_F( CudaFullyConnectedOpTests, CudaCpuEquivalence ) {
        // Create input and output tensors
        Tensor<float, CudaMemoryResource> cuda_input( medium_input_shape_ );
        auto cuda_weights = std::make_shared<Tensor<float, CudaMemoryResource>>( medium_weight_shape_ );
        auto cuda_bias = std::make_shared<Tensor<float, CudaMemoryResource>>( medium_bias_shape_ );
        Tensor<float, CudaMemoryResource> cuda_output( medium_output_shape_ );

        Tensor<float, HostMemoryResource> cpu_input( medium_input_shape_ );
        auto cpu_weights = std::make_shared<Tensor<float, HostMemoryResource>>( medium_weight_shape_ );
        auto cpu_bias = std::make_shared<Tensor<float, HostMemoryResource>>( medium_bias_shape_ );
        Tensor<float, HostMemoryResource> cpu_output( medium_output_shape_ );

        Tensor<float, HostMemoryResource> cuda_output_host( medium_output_shape_ );

        // Initialize input with varied test values
        for ( size_t i = 0; i < cpu_input.size(); ++i ) {
            // Use a mix of positive, negative, and near-zero values
            cpu_input.data()[ i ] = (static_cast<float>( i % 101 ) - 50.0f) / 25.0f;
        }

        // Initialize weights with varied test values
        for ( size_t i = 0; i < cpu_weights->size(); ++i ) {
            cpu_weights->data()[ i ] = (static_cast<float>( i % 77 ) - 38.0f) / 38.0f;
        }

        // Initialize bias 
        for ( size_t i = 0; i < cpu_bias->size(); ++i ) {
            cpu_bias->data()[ i ] = (static_cast<float>( i % 13 ) - 6.0f) / 6.0f;
        }

        // Copy to CUDA device
        cuda_input.copyFrom( cpu_input );
        cuda_weights->copyFrom( *cpu_weights );
        cuda_bias->copyFrom( *cpu_bias );

        // Execute CUDA Fully Connected operation
        std::vector<std::shared_ptr<Tensor<float, CudaMemoryResource>>> cuda_params = { cuda_weights, cuda_bias };
        std::vector<std::shared_ptr<Tensor<float, CudaMemoryResource>>> cuda_output_state;
        OperationAttributes cuda_props;

        cuda_fc_op_->forward( cuda_input, cuda_params, cuda_props, cuda_output, cuda_output_state );

        // Execute CPU Fully Connected operation
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> cpu_params = { cpu_weights, cpu_bias };
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> cpu_output_state;
        OperationAttributes cpu_props;

        cpu_fc_op_->forward( cpu_input, cpu_params, cpu_props, cpu_output, cpu_output_state );

        // Copy CUDA result back to host
        cuda_output_host.copyFrom( cuda_output );

        // Compare results (with small tolerance for floating point differences)
        EXPECT_TRUE( compareTensors( cpu_output, cuda_output_host, 1e-4f ) );
    }

    /**
     * @brief Test backward pass functionality
     */
    TEST_F( CudaFullyConnectedOpTests, BackwardPass ) {
        // Note: This test is commented out as the backward functionality is marked with FIXME in CudaFullyConnectedOp.ixx
        // Uncomment and complete once the backward functionality is implemented
        /*
        // Create tensors for forward and backward passes
        Tensor<float, CudaMemoryResource> input( small_input_shape_ );
        Tensor<float, CudaMemoryResource> output( small_output_shape_ );
        Tensor<float, CudaMemoryResource> output_grad( small_output_shape_ );
        Tensor<float, CudaMemoryResource> input_grad( small_input_shape_ );

        auto weights = std::make_shared<Tensor<float, CudaMemoryResource>>( small_weight_shape_ );
        auto bias = std::make_shared<Tensor<float, CudaMemoryResource>>( small_bias_shape_ );
        auto weight_grad = std::make_shared<Tensor<float, CudaMemoryResource>>( small_weight_shape_ );
        auto bias_grad = std::make_shared<Tensor<float, CudaMemoryResource>>( small_bias_shape_ );

        Tensor<float, HostMemoryResource> host_input( small_input_shape_ );
        Tensor<float, HostMemoryResource> host_weights( small_weight_shape_ );
        Tensor<float, HostMemoryResource> host_bias( small_bias_shape_ );
        Tensor<float, HostMemoryResource> host_output_grad( small_output_shape_ );
        Tensor<float, HostMemoryResource> host_input_grad( small_input_shape_ );
        Tensor<float, HostMemoryResource> host_weight_grad( small_weight_shape_ );
        Tensor<float, HostMemoryResource> host_bias_grad( small_bias_shape_ );

        // Initialize input and parameters
        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input.data()[ i ] = (static_cast<float>( i ) - 10.0f) / 10.0f;
        }

        for ( size_t i = 0; i < host_weights.size(); ++i ) {
            host_weights.data()[ i ] = (static_cast<float>( i % 7 ) - 3.0f) / 3.0f;
        }

        for ( size_t i = 0; i < host_bias.size(); ++i ) {
            host_bias.data()[ i ] = (static_cast<float>( i ) - 2.0f) / 5.0f;
        }

        // Set all output gradients to 1.0 for easier verification
        for ( size_t i = 0; i < host_output_grad.size(); ++i ) {
            host_output_grad.data()[ i ] = 1.0f;
        }

        // Copy to device
        input.copyFrom( host_input );
        weights->copyFrom( host_weights );
        bias->copyFrom( host_bias );
        output_grad.copyFrom( host_output_grad );

        // Forward pass first
        std::vector<std::shared_ptr<Tensor<float, CudaMemoryResource>>> params = { weights, bias };
        std::vector<std::shared_ptr<Tensor<float, CudaMemoryResource>>> output_state;
        std::vector<std::shared_ptr<Tensor<float, CudaMemoryResource>>> param_grads = { weight_grad, bias_grad };
        OperationAttributes props;

        cuda_fc_op_->forward( input, params, props, output, output_state );

        // Backward pass
        ASSERT_NO_THROW( cuda_fc_op_->backward(
            input, output, output_grad, params, param_grads, input_grad, props, output_state ) );

        // Copy result back to host
        host_input_grad.copyFrom( input_grad );
        host_weight_grad.copyFrom( *weight_grad );
        host_bias_grad.copyFrom( *bias_grad );

        // Verify gradients are not NaN or Inf
        EXPECT_FALSE( hasNaNorInf( host_input_grad ) );
        EXPECT_FALSE( hasNaNorInf( host_weight_grad ) );
        EXPECT_FALSE( hasNaNorInf( host_bias_grad ) );

        // For bias_grad, verify that each element is equal to B*T (sum of all 1.0's over batch and time)
        for ( size_t o = 0; o < host_bias_grad.size(); ++o ) {
            EXPECT_FLOAT_EQ( host_bias_grad.data()[ o ], small_batch_ * small_seq_len_ );
        }

        // For weight_grad, verify using a manual calculation
        for ( int o = 0; o < small_out_features_; o++ ) {
            for ( int i = 0; i < small_in_features_; i++ ) {
                float expected_grad = 0.0f;
                for ( int b = 0; b < small_batch_; b++ ) {
                    for ( int t = 0; t < small_seq_len_; t++ ) {
                        expected_grad += host_input.data()[ (b * small_seq_len_ + t) * small_in_features_ + i ];
                    }
                }
                EXPECT_NEAR( host_weight_grad.data()[ o * small_in_features_ + i ], expected_grad, 1e-4f );
            }
        }

        // For input_grad, verify using a manual calculation
        for ( int b = 0; b < small_batch_; b++ ) {
            for ( int t = 0; t < small_seq_len_; t++ ) {
                int bt = b * small_seq_len_ + t;
                for ( int i = 0; i < small_in_features_; i++ ) {
                    float expected_grad = 0.0f;
                    for ( int o = 0; o < small_out_features_; o++ ) {
                        expected_grad += host_weights.data()[ o * small_in_features_ + i ];
                    }
                    EXPECT_NEAR( host_input_grad.data()[ bt * small_in_features_ + i ], expected_grad, 1e-4f );
                }
            }
        }
        */
    }

    /**
     * @brief Test edge cases with non-standard shapes
     */
    TEST_F( CudaFullyConnectedOpTests, NonStandardShapes ) {
        // Create a shape that might not align well with warp size or memory access patterns
        std::vector<size_t> odd_input_shape = { 3, 5, 17 };
        std::vector<size_t> odd_output_shape = { 3, 5, 11 };
        std::vector<size_t> odd_weight_shape = { 11, 17 };
        std::vector<size_t> odd_bias_shape = { 11 };

        Tensor<float, CudaMemoryResource> device_input( odd_input_shape );
        auto device_weights = std::make_shared<Tensor<float, CudaMemoryResource>>( odd_weight_shape );
        auto device_bias = std::make_shared<Tensor<float, CudaMemoryResource>>( odd_bias_shape );
        Tensor<float, CudaMemoryResource> device_output( odd_output_shape );

        Tensor<float, HostMemoryResource> host_input( odd_input_shape );
        Tensor<float, HostMemoryResource> host_weights( odd_weight_shape );
        Tensor<float, HostMemoryResource> host_bias( odd_bias_shape );
        Tensor<float, HostMemoryResource> host_output( odd_output_shape );
        Tensor<float, HostMemoryResource> expected_output( odd_output_shape );

        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input.data()[ i ] = (static_cast<float>( i ) - 10.0f) / 10.0f;
        }

        for ( size_t i = 0; i < host_weights.size(); ++i ) {
            host_weights.data()[ i ] = (static_cast<float>( i % 7 ) - 3.0f) / 3.0f;
        }

        for ( size_t i = 0; i < host_bias.size(); ++i ) {
            host_bias.data()[ i ] = (static_cast<float>( i ) - 2.0f) / 5.0f;
        }

        // Copy to device
        device_input.copyFrom( host_input );
        device_weights->copyFrom( host_weights );
        device_bias->copyFrom( host_bias );

        // Execute operation
        std::vector<std::shared_ptr<Tensor<float, CudaMemoryResource>>> params = { device_weights, device_bias };
        std::vector<std::shared_ptr<Tensor<float, CudaMemoryResource>>> output_state;
        OperationAttributes props;

        ASSERT_NO_THROW( cuda_fc_op_->forward( device_input, params, props, device_output, output_state ) );

        // Copy result back to host
        host_output.copyFrom( device_output );

        // Compute expected output with reference implementation
        referenceFullyConnected( host_input, host_weights, &host_bias, expected_output );

        // Verify output matches reference implementation
        EXPECT_TRUE( compareTensors( host_output, expected_output, 1e-4f ) );
    }

    /**
     * @brief Test numerical stability with varied inputs
     */
    TEST_F( CudaFullyConnectedOpTests, NumericalStability ) {
        // Create test tensors
        Tensor<float, CudaMemoryResource> device_input( medium_input_shape_ );
        auto device_weights = std::make_shared<Tensor<float, CudaMemoryResource>>( medium_weight_shape_ );
        auto device_bias = std::make_shared<Tensor<float, CudaMemoryResource>>( medium_bias_shape_ );
        Tensor<float, CudaMemoryResource> device_output( medium_output_shape_ );

        Tensor<float, HostMemoryResource> host_input( medium_input_shape_ );
        Tensor<float, HostMemoryResource> host_weights( medium_weight_shape_ );
        Tensor<float, HostMemoryResource> host_bias( medium_bias_shape_ );
        Tensor<float, HostMemoryResource> host_output( medium_output_shape_ );

        // Test a variety of input values: positive, negative, small, large
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

        // Test a variety of weight values
        for ( size_t i = 0; i < host_weights.size(); ++i ) {
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
            host_weights.data()[ i ] = val;
        }

        // Initialize bias
        for ( size_t i = 0; i < host_bias.size(); ++i ) {
            host_bias.data()[ i ] = (i % 2 == 0) ? 0.5f : -0.5f;
        }

        // Copy to device
        device_input.copyFrom( host_input );
        device_weights->copyFrom( host_weights );
        device_bias->copyFrom( host_bias );

        // Execute operation
        std::vector<std::shared_ptr<Tensor<float, CudaMemoryResource>>> params = { device_weights, device_bias };
        std::vector<std::shared_ptr<Tensor<float, CudaMemoryResource>>> output_state;
        OperationAttributes props;

        ASSERT_NO_THROW( cuda_fc_op_->forward( device_input, params, props, device_output, output_state ) );

        // Copy result back to host
        host_output.copyFrom( device_output );

        // Verify no NaN or Inf values
        EXPECT_FALSE( hasNaNorInf( host_output ) );
    }

    /**
     * @brief Test deterministic behavior (multiple runs should produce same result)
     */
    TEST_F( CudaFullyConnectedOpTests, DeterministicBehavior ) {
        // Create test tensors
        Tensor<float, CudaMemoryResource> device_input( medium_input_shape_ );
        auto device_weights = std::make_shared<Tensor<float, CudaMemoryResource>>( medium_weight_shape_ );
        auto device_bias = std::make_shared<Tensor<float, CudaMemoryResource>>( medium_bias_shape_ );
        Tensor<float, CudaMemoryResource> device_output1( medium_output_shape_ );
        Tensor<float, CudaMemoryResource> device_output2( medium_output_shape_ );

        Tensor<float, HostMemoryResource> host_input( medium_input_shape_ );
        Tensor<float, HostMemoryResource> host_weights( medium_weight_shape_ );
        Tensor<float, HostMemoryResource> host_bias( medium_bias_shape_ );
        Tensor<float, HostMemoryResource> host_output1( medium_output_shape_ );
        Tensor<float, HostMemoryResource> host_output2( medium_output_shape_ );

        // Initialize with consistent values
        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input.data()[ i ] = (static_cast<float>( i % 17 ) - 8.5f) * 0.1f;
        }

        for ( size_t i = 0; i < host_weights.size(); ++i ) {
            host_weights.data()[ i ] = (static_cast<float>( i % 13 ) - 6.5f) * 0.05f;
        }

        for ( size_t i = 0; i < host_bias.size(); ++i ) {
            host_bias.data()[ i ] = (static_cast<float>( i % 7 ) - 3.5f) * 0.1f;
        }

        // Copy to device
        device_input.copyFrom( host_input );
        device_weights->copyFrom( host_weights );
        device_bias->copyFrom( host_bias );

        // Run twice with same input
        std::vector<std::shared_ptr<Tensor<float, CudaMemoryResource>>> params = { device_weights, device_bias };
        std::vector<std::shared_ptr<Tensor<float, CudaMemoryResource>>> output_state1;
        std::vector<std::shared_ptr<Tensor<float, CudaMemoryResource>>> output_state2;
        OperationAttributes props;

        cuda_fc_op_->forward( device_input, params, props, device_output1, output_state1 );
        cuda_fc_op_->forward( device_input, params, props, device_output2, output_state2 );

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
    TEST_F( CudaFullyConnectedOpTests, DeviceTypeMismatch ) {
        // Attempt to create a CudaFullyConnectedOp with a CPU context
        EXPECT_THROW( (CudaFullyConnectedOp<float>( cpu_context_ )), std::runtime_error );

        // Create tensors on CPU memory
        Tensor<float, HostMemoryResource> cpu_input( small_input_shape_ );
        auto cpu_weights = std::make_shared<Tensor<float, HostMemoryResource>>( small_weight_shape_ );
        Tensor<float, HostMemoryResource> cpu_output( small_output_shape_ );

        // Attempt to run on CPU memory (should throw)
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> cpu_params = { cpu_weights };
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> cpu_output_state;
        OperationAttributes cpu_props;

        // This should fail with a compile-time error, so we don't directly test it:
        // EXPECT_THROW( cuda_fc_op_->forward( cpu_input, cpu_params, cpu_props, cpu_output, cpu_output_state ), std::runtime_error );
    }

    /**
     * @brief Test performance with large input
     */
    TEST_F( CudaFullyConnectedOpTests, Performance ) {
        // Skip test if running in CI environment
        if ( std::getenv( "CI" ) != nullptr ) {
            GTEST_SKIP() << "Skipping performance test in CI environment";
        }

        // Create large test tensors
        Tensor<float, CudaMemoryResource> device_input( large_input_shape_ );
        auto device_weights = std::make_shared<Tensor<float, CudaMemoryResource>>( large_weight_shape_ );
        auto device_bias = std::make_shared<Tensor<float, CudaMemoryResource>>( large_bias_shape_ );
        Tensor<float, CudaMemoryResource> device_output( large_output_shape_ );

        Tensor<float, HostMemoryResource> host_input( large_input_shape_ );
        Tensor<float, HostMemoryResource> host_weights( large_weight_shape_ );
        Tensor<float, HostMemoryResource> host_bias( large_bias_shape_ );

        // Fill with random values
        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input.data()[ i ] = (static_cast<float>( rand() ) / RAND_MAX * 2.0f - 1.0f) * 10.0f;
        }

        for ( size_t i = 0; i < host_weights.size(); ++i ) {
            host_weights.data()[ i ] = (static_cast<float>( rand() ) / RAND_MAX * 2.0f - 1.0f) * 0.1f;
        }

        for ( size_t i = 0; i < host_bias.size(); ++i ) {
            host_bias.data()[ i ] = (static_cast<float>( rand() ) / RAND_MAX * 2.0f - 1.0f);
        }

        // Copy to device
        device_input.copyFrom( host_input );
        device_weights->copyFrom( host_weights );
        device_bias->copyFrom( host_bias );

        std::vector<std::shared_ptr<Tensor<float, CudaMemoryResource>>> params = { device_weights, device_bias };
        std::vector<std::shared_ptr<Tensor<float, CudaMemoryResource>>> output_state;
        OperationAttributes props;

        // Make sure everything is ready
        cudaDeviceSynchronize();

        // Measure performance over multiple iterations
        const int iterations = 100;
        auto start_time = std::chrono::high_resolution_clock::now();

        for ( int i = 0; i < iterations; ++i ) {
            cuda_fc_op_->forward( device_input, params, props, device_output, output_state );
        }

        // Wait for all operations to complete
        cudaDeviceSynchronize();

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>( end_time - start_time );

        // Calculate FLOPs (floating point operations)
        // Each output element requires C multiply-adds, so 2*C FLOPs per output element
        size_t flops_per_iter = large_batch_ * large_seq_len_ * large_out_features_ * (2 * large_in_features_);
        size_t total_flops = flops_per_iter * iterations;

        // Calculate performance metrics
        double flops_per_second = static_cast<double>(total_flops) / (duration.count() * 1e-6);
        double gflops = flops_per_second / 1e9;
        double avg_time_per_iter = duration.count() / iterations;

        // Record properties that will show in test details
        // With this pattern (in performance tests):
        SCOPED_TRACE( ::testing::Message() << "Performance_GFLOPS: " << gflops );
        RecordProperty( "Performance_GFLOPS", std::to_string( gflops ) );
        RecordProperty( "Average_Time_us", std::to_string( avg_time_per_iter ) );
        RecordProperty( "Implementation", "CUDA" );
        RecordProperty( "Batch_Size", std::to_string( large_batch_ ) );
        RecordProperty( "Sequence_Length", std::to_string( large_seq_len_ ) );
        RecordProperty( "Input_Features", std::to_string( large_in_features_ ) );
        RecordProperty( "Output_Features", std::to_string( large_out_features_ ) );

        std::cout << "CUDA FullyConnected Performance: " << gflops << " GFLOPS" << std::endl;
        std::cout << "Average time per iteration: " << avg_time_per_iter << " microseconds" << std::endl;

        // No assertion, just informational
        EXPECT_TRUE( true );
    }

    /**
     * @brief Test comparison with CPU performance
     */
    TEST_F( CudaFullyConnectedOpTests, CpuCudaPerformanceComparison ) {
        // Skip test if running in CI environment
        if ( std::getenv( "CI" ) != nullptr ) {
            GTEST_SKIP() << "Skipping performance comparison test in CI environment";
        }

        // Create test tensors for CUDA
        Tensor<float, CudaMemoryResource> cuda_input( medium_input_shape_ );
        auto cuda_weights = std::make_shared<Tensor<float, CudaMemoryResource>>( medium_weight_shape_ );
        auto cuda_bias = std::make_shared<Tensor<float, CudaMemoryResource>>( medium_bias_shape_ );
        Tensor<float, CudaMemoryResource> cuda_output( medium_output_shape_ );

        // Create test tensors for CPU
        Tensor<float, HostMemoryResource> cpu_input( medium_input_shape_ );
        auto cpu_weights = std::make_shared<Tensor<float, HostMemoryResource>>( medium_weight_shape_ );
        auto cpu_bias = std::make_shared<Tensor<float, HostMemoryResource>>( medium_bias_shape_ );
        Tensor<float, HostMemoryResource> cpu_output( medium_output_shape_ );

        // Fill with random values
        for ( size_t i = 0; i < cpu_input.size(); ++i ) {
            cpu_input.data()[ i ] = (static_cast<float>( rand() ) / RAND_MAX * 2.0f - 1.0f);
        }

        for ( size_t i = 0; i < cpu_weights->size(); ++i ) {
            cpu_weights->data()[ i ] = (static_cast<float>( rand() ) / RAND_MAX * 2.0f - 1.0f) * 0.1f;
        }

        for ( size_t i = 0; i < cpu_bias->size(); ++i ) {
            cpu_bias->data()[ i ] = (static_cast<float>( rand() ) / RAND_MAX * 2.0f - 1.0f) * 0.1f;
        }

        // Copy to CUDA
        cuda_input.copyFrom( cpu_input );
        cuda_weights->copyFrom( *cpu_weights );
        cuda_bias->copyFrom( *cpu_bias );

        // Setup parameters
        std::vector<std::shared_ptr<Tensor<float, CudaMemoryResource>>> cuda_params = { cuda_weights, cuda_bias };
        std::vector<std::shared_ptr<Tensor<float, CudaMemoryResource>>> cuda_output_state;

        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> cpu_params = { cpu_weights, cpu_bias };
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> cpu_output_state;

        OperationAttributes props;

        // Measure CUDA performance
        cudaDeviceSynchronize();
        const int iterations = 100;
        auto cuda_start_time = std::chrono::high_resolution_clock::now();

        for ( int i = 0; i < iterations; ++i ) {
            cuda_fc_op_->forward( cuda_input, cuda_params, props, cuda_output, cuda_output_state );
        }
        cudaDeviceSynchronize();
        auto cuda_end_time = std::chrono::high_resolution_clock::now();
        auto cuda_duration = std::chrono::duration_cast<std::chrono::microseconds>( cuda_end_time - cuda_start_time );

        // Measure CPU performance
        auto cpu_start_time = std::chrono::high_resolution_clock::now();
        for ( int i = 0; i < iterations; ++i ) {
            cpu_fc_op_->forward( cpu_input, cpu_params, props, cpu_output, cpu_output_state );
        }
        auto cpu_end_time = std::chrono::high_resolution_clock::now();
        auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>( cpu_end_time - cpu_start_time );

        // Calculate FLOPs
        size_t flops_per_iter = medium_batch_ * medium_seq_len_ * medium_out_features_ * (2 * medium_in_features_);
        double cuda_gflops = static_cast<double>( flops_per_iter * iterations ) / (cuda_duration.count() * 1e-6) / 1e9;
        double cpu_gflops = static_cast<double>( flops_per_iter * iterations ) / (cpu_duration.count() * 1e-6) / 1e9;
        double speedup = static_cast<double>( cpu_duration.count() ) / cuda_duration.count();

        // Report comparison
        std::cout << "CUDA Performance: " << cuda_gflops << " GFLOPS" << std::endl;
        std::cout << "CPU Performance: " << cpu_gflops << " GFLOPS" << std::endl;
        std::cout << "CUDA Speedup: " << speedup << "x" << std::endl;

        RecordProperty( "CUDA_GFLOPS", std::to_string( cuda_gflops ) );
        RecordProperty( "CPU_GFLOPS", std::to_string( cpu_gflops ) );
        RecordProperty( "CUDA_Speedup", std::to_string( speedup ) );

        // No assertion, just informational
        EXPECT_TRUE( true );
    }
}
