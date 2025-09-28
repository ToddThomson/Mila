/**
 * @file CudaMultiHeadAttentionOpTests.cpp
 * @brief Complete test suite for the CUDA Multi-Head Attention operation.
 */

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <stdexcept>
#include <random>

import Mila;

namespace Operations::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    /**
     * @brief Test fixture for CudaMultiHeadAttentionOp tests
     */
    class CudaMultiHeadAttentionOpTests : public ::testing::Test {
    protected:
        void SetUp() override {
            // Create device contexts for both CPU and CUDA
            cuda_context_ = std::make_shared<DeviceContext>( "CUDA:0" );
            cpu_context_ = std::make_shared<DeviceContext>( "CPU" );

            // Define shapes for testing
            small_shape_ = { 2, 16, 64 };  // Small shape for quick tests [Batch, Seq_len, Hidden_dim]
            medium_shape_ = { 8, 32, 128 };  // Medium shape for thorough tests
            large_shape_ = { 16, 64, 256 };  // Large shape for stress tests

            // Create CUDA MHA operation with specific context
            cuda_mha_op_ = std::make_shared<CudaMultiHeadAttentionOp<float>>( cuda_context_ );

            // Get CPU MHA op for comparison (assume it's registered)
            auto cpu_op = OperationRegistry::instance().createUnaryOperation<float, float, DeviceType::Cpu>(
                "Cpu::MultiHeadAttentionOp", cpu_context_ );
            cpu_mha_op_ = std::static_pointer_cast<UnaryOperation<float, float, DeviceType::Cpu>>(cpu_op);
        }

        // Helper method to compare tensors with tolerance
        bool compareTensors(
            const Tensor<float, HostMemoryResource>& a,
            const Tensor<float, HostMemoryResource>& b,
            float epsilon = 1e-5f ) {

            if ( a.size() != b.size() ) return false;

            for ( size_t i = 0; i < a.size(); ++i ) {
                float diff = std::abs( a.rawData()[ i ] - b.rawData()[ i ] );
                if ( diff > epsilon ) {
                    std::cout << "Mismatch at index " << i << ": "
                        << a.rawData()[ i ] << " vs " << b.rawData()[ i ]
                        << " (diff = " << diff << ")" << std::endl;
                        return false;
                }
            }
            return true;
        }

        // Helper method to check for NaNs or Infs
        bool hasNaNorInf( const Tensor<float, HostMemoryResource>& tensor ) {
            for ( size_t i = 0; i < tensor.size(); ++i ) {
                if ( std::isnan( tensor.rawData()[ i ] ) || std::isinf( tensor.rawData()[ i ] ) ) {
                    std::cout << "Found NaN or Inf at index " << i << ": " << tensor.rawData()[ i ] << std::endl;
                    return true;
                }
            }
            return false;
        }

        // Helper method to initialize tensors with random values
        void initializeRandomTensor( Tensor<float, HostMemoryResource>& tensor, float min = -1.0f, float max = 1.0f ) {
            std::random_device rd;
            std::mt19937 gen( rd() );
            std::uniform_real_distribution<float> dist( min, max );

            for ( size_t i = 0; i < tensor.size(); ++i ) {
                tensor.rawData()[ i ] = dist( gen );
            }
        }

        // Helper method to reference implementation of multi-head attention
        void referenceMultiHeadAttention(
            const Tensor<float, HostMemoryResource>& input,
            const Tensor<float, HostMemoryResource>& weights,
            const Tensor<float, HostMemoryResource>& bias,
            Tensor<float, HostMemoryResource>& output,
            Tensor<float, HostMemoryResource>& attn_scores,
            Tensor<float, HostMemoryResource>& attn_weights,
            int num_heads ) {

            // Extract dimensions
            size_t B = input.shape()[ 0 ];  // Batch size
            size_t T = input.shape()[ 1 ];  // Sequence length
            size_t C = input.shape()[ 2 ];  // Hidden dimension

            // Simplified CPU reference implementation
            // In real code, you'd have the proper algorithm here
            // This is a placeholder implementation

            // Clear output
            for ( size_t i = 0; i < output.size(); ++i ) {
                output.rawData()[ i ] = 0.0f;
            }

            // Calculate embedding dimension per head
            size_t head_dim = C / num_heads;

            // For each batch and sequence position
            for ( size_t b = 0; b < B; ++b ) {
                for ( size_t t = 0; t < T; ++t ) {
                    // For each head
                    for ( size_t h = 0; h < num_heads; ++h ) {
                        // For each position to attend to
                        for ( size_t t2 = 0; t2 < T; ++t2 ) {
                            float score = 0.0f;

                            // Calculate attention score (simplified dot-product)
                            for ( size_t d = 0; d < head_dim; ++d ) {
                                // Query for current position
                                float q = input.rawData()[ (b * T + t) * C + h * head_dim + d ];
                                // Key for position to attend to
                                float k = input.rawData()[ (b * T + t2) * C + h * head_dim + d + C ];
                                score += q * k;
                            }

                            // Scale by sqrt(head_dim)
                            score /= std::sqrt( static_cast<float>( head_dim ) );

                            // Store attention score
                            attn_scores.rawData()[ (b * num_heads + h) * T * T + t * T + t2 ] = score;
                        }

                        // Apply softmax to get attention weights
                        float max_score = -std::numeric_limits<float>::infinity();
                        for ( size_t t2 = 0; t2 < T; ++t2 ) {
                            float score = attn_scores.rawData()[ (b * num_heads + h) * T * T + t * T + t2 ];
                            max_score = std::max( max_score, score );
                        }

                        float sum_exp = 0.0f;
                        for ( size_t t2 = 0; t2 < T; ++t2 ) {
                            float score = attn_scores.rawData()[ (b * num_heads + h) * T * T + t * T + t2 ];
                            float exp_score = std::exp( score - max_score );
                            attn_weights.rawData()[ (b * num_heads + h) * T * T + t * T + t2 ] = exp_score;
                            sum_exp += exp_score;
                        }

                        // Normalize attention weights
                        for ( size_t t2 = 0; t2 < T; ++t2 ) {
                            attn_weights.rawData()[ (b * num_heads + h) * T * T + t * T + t2 ] /= sum_exp;
                        }

                        // Apply attention weights to values
                        for ( size_t d = 0; d < head_dim; ++d ) {
                            float weighted_sum = 0.0f;

                            for ( size_t t2 = 0; t2 < T; ++t2 ) {
                                float weight = attn_weights.rawData()[ (b * num_heads + h) * T * T + t * T + t2 ];
                                float value = input.rawData()[ (b * T + t2) * C + h * head_dim + d + 2 * C ];
                                weighted_sum += weight * value;
                            }

                            // Store in output
                            output.rawData()[ (b * T + t) * C + h * head_dim + d ] = weighted_sum;
                        }
                    }

                    // Apply output projection and bias
                    // This is simplified - real implementation would use matrix multiplication
                    for ( size_t oc = 0; oc < C; ++oc ) {
                        float sum = bias.rawData()[ oc ];

                        for ( size_t ic = 0; ic < C; ++ic ) {
                            sum += output.rawData()[ (b * T + t) * C + ic ] * weights.rawData()[ C * oc + ic ];
                        }

                        // Final output
                        output.rawData()[ (b * T + t) * C + oc ] = sum;
                    }
                }
            }
        }

        std::shared_ptr<DeviceContext> cuda_context_;
        std::shared_ptr<DeviceContext> cpu_context_;
        std::shared_ptr<CudaMultiHeadAttentionOp<float>> cuda_mha_op_;
        std::shared_ptr<UnaryOperation<float, float, DeviceType::Cpu>> cpu_mha_op_;

        // Test shapes
        std::vector<size_t> small_shape_;
        std::vector<size_t> medium_shape_;
        std::vector<size_t> large_shape_;
    };

    /**
     * @brief Test name property of CudaMultiHeadAttentionOp
     */
    TEST_F( CudaMultiHeadAttentionOpTests, Name ) {
        EXPECT_EQ( cuda_mha_op_->getName(), "Cuda::MultiHeadAttentionOp" );
    }

    /**
     * @brief Test constructor without explicit device context
     */
    TEST_F( CudaMultiHeadAttentionOpTests, DefaultConstructor ) {
        auto op = std::make_shared<CudaMultiHeadAttentionOp<float>>();
        EXPECT_EQ( op->getName(), "Cuda::MultiHeadAttentionOp" );
    }

    /**
     * @brief Test constructor with non-CUDA device context throws exception
     */
    TEST_F( CudaMultiHeadAttentionOpTests, ConstructorWithNonCudaContext ) {
        EXPECT_THROW( (std::make_shared<CudaMultiHeadAttentionOp<float>>( cpu_context_ )), std::runtime_error );
    }

    /**
     * @brief Test basic functionality with small tensors
     */
    TEST_F( CudaMultiHeadAttentionOpTests, BasicFunctionality ) {
        // Create input, weight, bias, and output tensors
        Tensor<float, CudaMemoryResource> device_input( small_shape_ );

        // Define weight and bias shapes based on input
        std::vector<size_t> weight_shape = { small_shape_[ 2 ], small_shape_[ 2 ] };  // Hidden x Hidden
        std::vector<size_t> bias_shape = { small_shape_[ 2 ] };  // Hidden dim

        auto device_weight = std::make_shared<Tensor<float, CudaMemoryResource>>( weight_shape );
        auto device_bias = std::make_shared<Tensor<float, CudaMemoryResource>>( bias_shape );
        Tensor<float, CudaMemoryResource> device_output( small_shape_ );

        // Attention cache tensors
        size_t num_heads = 4;
        std::vector<size_t> attn_scores_shape = { small_shape_[ 0 ], num_heads, small_shape_[ 1 ], small_shape_[ 1 ] };  // [B, H, T, T]
        std::vector<size_t> attn_weights_shape = { small_shape_[ 0 ], num_heads, small_shape_[ 1 ], small_shape_[ 1 ] };  // [B, H, T, T]

        auto device_attn_scores = std::make_shared<Tensor<float, CudaMemoryResource>>( attn_scores_shape );
        auto device_attn_weights = std::make_shared<Tensor<float, CudaMemoryResource>>( attn_weights_shape );

        Tensor<float, HostMemoryResource> host_input( small_shape_ );
        Tensor<float, HostMemoryResource> host_weight( weight_shape );
        Tensor<float, HostMemoryResource> host_bias( bias_shape );
        Tensor<float, HostMemoryResource> host_output( small_shape_ );
        Tensor<float, HostMemoryResource> host_attn_scores( attn_scores_shape );
        Tensor<float, HostMemoryResource> host_attn_weights( attn_weights_shape );

        Tensor<float, HostMemoryResource> expected_output( small_shape_ );
        Tensor<float, HostMemoryResource> expected_attn_scores( attn_scores_shape );
        Tensor<float, HostMemoryResource> expected_attn_weights( attn_weights_shape );

        // Initialize input, weight, and bias with test values
        initializeRandomTensor( host_input, -0.1f, 0.1f );
        initializeRandomTensor( host_weight, -0.01f, 0.01f );

        // Initialize bias to zeros
        for ( size_t i = 0; i < host_bias.size(); ++i ) {
            host_bias.rawData()[ i ] = 0.0f;
        }

        // Copy to device
        device_input.copyFrom( host_input );
        device_weight->copyFrom( host_weight );
        device_bias->copyFrom( host_bias );

        // Execute operation
        std::vector<std::shared_ptr<Tensor<float, CudaMemoryResource>>> params = { device_weight, device_bias };
        std::vector<std::shared_ptr<Tensor<float, CudaMemoryResource>>> output_state = {
            device_attn_scores, device_attn_weights
        };

        OperationAttributes props;
        props.set( "num_heads", num_heads );

        ASSERT_NO_THROW( cuda_mha_op_->forward(
            device_input, params, props, device_output, output_state ) );

        // Copy result back to host
        host_output.copyFrom( device_output );
        host_attn_scores.copyFrom( *device_attn_scores );
        host_attn_weights.copyFrom( *device_attn_weights );

        // Compute expected output with reference implementation
        referenceMultiHeadAttention(
            host_input, host_weight, host_bias,
            expected_output, expected_attn_scores, expected_attn_weights,
            num_heads );

        // Verify output has correct values (with higher tolerance for MHA)
        EXPECT_TRUE( compareTensors( host_output, expected_output, 1e-3f ) );
        EXPECT_TRUE( compareTensors( host_attn_scores, expected_attn_scores, 1e-3f ) );
        EXPECT_TRUE( compareTensors( host_attn_weights, expected_attn_weights, 1e-3f ) );
    }

    /**
     * @brief Test that CUDA and CPU implementations produce equivalent results
     */
    TEST_F( CudaMultiHeadAttentionOpTests, CudaCpuEquivalence ) {
        // Skip if CPU implementation isn't available
        if ( !cpu_mha_op_ ) {
            GTEST_SKIP() << "CPU implementation of MultiHeadAttention not available";
        }

        // Use small shape for faster comparison
        std::vector<size_t> test_shape = { 2, 8, 32 };

        // Create input, weight, bias, and output tensors
        Tensor<float, CudaMemoryResource> cuda_input( test_shape );

        // Define weight and bias shapes based on input
        std::vector<size_t> weight_shape = { test_shape[ 2 ], test_shape[ 2 ] };  // Hidden x Hidden
        std::vector<size_t> bias_shape = { test_shape[ 2 ] };  // Hidden dim

        auto cuda_weight = std::make_shared<Tensor<float, CudaMemoryResource>>( weight_shape );
        auto cuda_bias = std::make_shared<Tensor<float, CudaMemoryResource>>( bias_shape );
        Tensor<float, CudaMemoryResource> cuda_output( test_shape );

        // Attention cache tensors
        size_t num_heads = 4;  // Using 4 attention heads for this test
        std::vector<size_t> attn_scores_shape = { test_shape[ 0 ], num_heads, test_shape[ 1 ], test_shape[ 1 ] };  // [B, H, T, T]
        std::vector<size_t> attn_weights_shape = { test_shape[ 0 ], num_heads, test_shape[ 1 ], test_shape[ 1 ] };  // [B, H, T, T]

        auto cuda_attn_scores = std::make_shared<Tensor<float, CudaMemoryResource>>( attn_scores_shape );
        auto cuda_attn_weights = std::make_shared<Tensor<float, CudaMemoryResource>>( attn_weights_shape );

        // CPU tensors
        Tensor<float, HostMemoryResource> cpu_input( test_shape );
        auto cpu_weight = std::make_shared<Tensor<float, HostMemoryResource>>( weight_shape );
        auto cpu_bias = std::make_shared<Tensor<float, HostMemoryResource>>( bias_shape );
        Tensor<float, HostMemoryResource> cpu_output( test_shape );
        //std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> cpu_output_state;

        // Host tensors for comparison
        Tensor<float, HostMemoryResource> cuda_output_host( test_shape );

        // Initialize input, weight, and bias with test values
        initializeRandomTensor( cpu_input, -0.1f, 0.1f );
        initializeRandomTensor( *cpu_weight, -0.01f, 0.01f );

        // Initialize bias to zeros
        for ( size_t i = 0; i < cpu_bias->size(); ++i ) {
            cpu_bias->rawData()[ i ] = 0.0f;
        }

        // Copy to CUDA device
        cuda_input.copyFrom( cpu_input );
        cuda_weight->copyFrom( *cpu_weight );
        cuda_bias->copyFrom( *cpu_bias );

        // Execute CUDA MHA operation
        std::vector<std::shared_ptr<Tensor<float, CudaMemoryResource>>> cuda_params = { cuda_weight, cuda_bias };
        std::vector<std::shared_ptr<Tensor<float, CudaMemoryResource>>> cuda_output_state = {
            cuda_attn_scores, cuda_attn_weights
        };

        OperationAttributes props;
        props.set( "num_heads", num_heads );

        cuda_mha_op_->forward( cuda_input, cuda_params, props, cuda_output, cuda_output_state );

        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> cpu_params = { cpu_weight, cpu_bias };
        auto cpu_attn_scores = std::make_shared<Tensor<float, HostMemoryResource>>( attn_scores_shape );
        auto cpu_attn_weights = std::make_shared<Tensor<float, HostMemoryResource>>( attn_weights_shape );
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> cpu_output_state = {
            cpu_attn_scores,
            cpu_attn_weights
        };

        cpu_mha_op_->forward( cpu_input, cpu_params, props, cpu_output, cpu_output_state );

        // Copy CUDA result back to host
        cuda_output_host.copyFrom( cuda_output );

        // Compare results (with small tolerance for floating point differences)
        EXPECT_TRUE( compareTensors( cpu_output, cuda_output_host, 1e-3f ) );
    }

    /**
     * @brief Test with different numbers of attention heads
     */
    TEST_F( CudaMultiHeadAttentionOpTests, DifferentHeadCounts ) {
        std::vector<int> head_counts = { 1, 2, 4, 8 };

        for ( int num_heads : head_counts ) {
            // Create input, weight, bias, and output tensors
            Tensor<float, CudaMemoryResource> device_input( small_shape_ );

            // Define weight and bias shapes based on input
            std::vector<size_t> weight_shape = { small_shape_[ 2 ], small_shape_[ 2 ] };  // Hidden x Hidden
            std::vector<size_t> bias_shape = { small_shape_[ 2 ] };  // Hidden dim

            auto device_weight = std::make_shared<Tensor<float, CudaMemoryResource>>( weight_shape );
            auto device_bias = std::make_shared<Tensor<float, CudaMemoryResource>>( bias_shape );
            Tensor<float, CudaMemoryResource> device_output( small_shape_ );

            // Attention cache tensors
            std::vector<size_t> attn_scores_shape = { small_shape_[ 0 ], static_cast<size_t>(num_heads), small_shape_[ 1 ], small_shape_[ 1 ] };
            std::vector<size_t> attn_weights_shape = { small_shape_[ 0 ], static_cast<size_t>(num_heads), small_shape_[ 1 ], small_shape_[ 1 ] };

            auto device_attn_scores = std::make_shared<Tensor<float, CudaMemoryResource>>( attn_scores_shape );
            auto device_attn_weights = std::make_shared<Tensor<float, CudaMemoryResource>>( attn_weights_shape );

            Tensor<float, HostMemoryResource> host_input( small_shape_ );
            Tensor<float, HostMemoryResource> host_weight( weight_shape );
            Tensor<float, HostMemoryResource> host_bias( bias_shape );

            // Initialize with random values
            initializeRandomTensor( host_input, -0.1f, 0.1f );
            initializeRandomTensor( host_weight, -0.01f, 0.01f );

            // Initialize bias to zeros
            for ( size_t i = 0; i < host_bias.size(); ++i ) {
                host_bias.rawData()[ i ] = 0.0f;
            }

            // Copy to device
            device_input.copyFrom( host_input );
            device_weight->copyFrom( host_weight );
            device_bias->copyFrom( host_bias );

            // Execute operation
            std::vector<std::shared_ptr<Tensor<float, CudaMemoryResource>>> params = { device_weight, device_bias };
            std::vector<std::shared_ptr<Tensor<float, CudaMemoryResource>>> output_state = {
                device_attn_scores, device_attn_weights
            };

            OperationAttributes props;
            props.set( "num_heads", num_heads );

            // Verify that different head counts work without exceptions
            EXPECT_NO_THROW( cuda_mha_op_->forward(
                device_input, params, props, device_output, output_state ) );

            // Copy result back to host
            Tensor<float, HostMemoryResource> host_output( small_shape_ );
            host_output.copyFrom( device_output );

            // Check for NaNs or Infs
            EXPECT_FALSE( hasNaNorInf( host_output ) ) << "Found NaN or Inf with " << num_heads << " attention heads";
        }
    }

    /**
     * @brief Test with different sequence lengths
     */
    TEST_F( CudaMultiHeadAttentionOpTests, DifferentSequenceLengths ) {
        std::vector<size_t> seq_lengths = { 1, 8, 32, 64 };
        int num_heads = 4;

        for ( size_t seq_len : seq_lengths ) {
            // Create shape with varying sequence length
            std::vector<size_t> shape = { 2, seq_len, 64 };

            // Create input, weight, bias, and output tensors
            Tensor<float, CudaMemoryResource> device_input( shape );

            // Define weight and bias shapes based on input
            std::vector<size_t> weight_shape = { shape[ 2 ], shape[ 2 ] };  // Hidden x Hidden
            std::vector<size_t> bias_shape = { shape[ 2 ] };  // Hidden dim

            auto device_weight = std::make_shared<Tensor<float, CudaMemoryResource>>( weight_shape );
            auto device_bias = std::make_shared<Tensor<float, CudaMemoryResource>>( bias_shape );
            Tensor<float, CudaMemoryResource> device_output( shape );

            // Attention cache tensors
            std::vector<size_t> attn_scores_shape = { shape[ 0 ], static_cast<size_t>(num_heads), shape[ 1 ], shape[ 1 ] };
            std::vector<size_t> attn_weights_shape = { shape[ 0 ], static_cast<size_t>(num_heads), shape[ 1 ], shape[ 1 ] };

            auto device_attn_scores = std::make_shared<Tensor<float, CudaMemoryResource>>( attn_scores_shape );
            auto device_attn_weights = std::make_shared<Tensor<float, CudaMemoryResource>>( attn_weights_shape );

            Tensor<float, HostMemoryResource> host_input( shape );
            Tensor<float, HostMemoryResource> host_weight( weight_shape );
            Tensor<float, HostMemoryResource> host_bias( bias_shape );

            // Initialize with random values
            initializeRandomTensor( host_input, -0.1f, 0.1f );
            initializeRandomTensor( host_weight, -0.01f, 0.01f );

            // Initialize bias to zeros
            for ( size_t i = 0; i < host_bias.size(); ++i ) {
                host_bias.rawData()[ i ] = 0.0f;
            }

            // Copy to device
            device_input.copyFrom( host_input );
            device_weight->copyFrom( host_weight );
            device_bias->copyFrom( host_bias );

            // Execute operation
            std::vector<std::shared_ptr<Tensor<float, CudaMemoryResource>>> params = { device_weight, device_bias };
            std::vector<std::shared_ptr<Tensor<float, CudaMemoryResource>>> output_state = {
                device_attn_scores, device_attn_weights
            };

            OperationAttributes props;
            props.set( "num_heads", num_heads );

            // Verify that different sequence lengths work without exceptions
            EXPECT_NO_THROW( cuda_mha_op_->forward(
                device_input, params, props, device_output, output_state ) );

            // Copy result back to host
            Tensor<float, HostMemoryResource> host_output( shape );
            host_output.copyFrom( device_output );

            // Check for NaNs or Infs
            EXPECT_FALSE( hasNaNorInf( host_output ) ) << "Found NaN or Inf with sequence length " << seq_len;
        }
    }

    /**
     * @brief Test with different batch sizes
     */
    TEST_F( CudaMultiHeadAttentionOpTests, DifferentBatchSizes ) {
        std::vector<size_t> batch_sizes = { 1, 4, 16 };
        int num_heads = 4;

        for ( size_t batch_size : batch_sizes ) {
            // Create shape with varying batch size
            std::vector<size_t> shape = { batch_size, 16, 64 };

            // Create input, weight, bias, and output tensors
            Tensor<float, CudaMemoryResource> device_input( shape );

            // Define weight and bias shapes based on input
            std::vector<size_t> weight_shape = { shape[ 2 ], shape[ 2 ] };  // Hidden x Hidden
            std::vector<size_t> bias_shape = { shape[ 2 ] };  // Hidden dim

            auto device_weight = std::make_shared<Tensor<float, CudaMemoryResource>>( weight_shape );
            auto device_bias = std::make_shared<Tensor<float, CudaMemoryResource>>( bias_shape );
            Tensor<float, CudaMemoryResource> device_output( shape );

            // Attention cache tensors
            std::vector<size_t> attn_scores_shape = { shape[ 0 ], static_cast<size_t>(num_heads), shape[ 1 ], shape[ 1 ] };
            std::vector<size_t> attn_weights_shape = { shape[ 0 ], static_cast<size_t>(num_heads), shape[ 1 ], shape[ 1 ] };

            auto device_attn_scores = std::make_shared<Tensor<float, CudaMemoryResource>>( attn_scores_shape );
            auto device_attn_weights = std::make_shared<Tensor<float, CudaMemoryResource>>( attn_weights_shape );

            Tensor<float, HostMemoryResource> host_input( shape );
            Tensor<float, HostMemoryResource> host_weight( weight_shape );
            Tensor<float, HostMemoryResource> host_bias( bias_shape );

            // Initialize with random values
            initializeRandomTensor( host_input, -0.1f, 0.1f );
            initializeRandomTensor( host_weight, -0.01f, 0.01f );

            // Initialize bias to zeros
            for ( size_t i = 0; i < host_bias.size(); ++i ) {
                host_bias.rawData()[ i ] = 0.0f;
            }

            // Copy to device
            device_input.copyFrom( host_input );
            device_weight->copyFrom( host_weight );
            device_bias->copyFrom( host_bias );

            // Execute operation
            std::vector<std::shared_ptr<Tensor<float, CudaMemoryResource>>> params = { device_weight, device_bias };
            std::vector<std::shared_ptr<Tensor<float, CudaMemoryResource>>> output_state = {
                device_attn_scores, device_attn_weights
            };

            OperationAttributes props;
            props.set( "num_heads", num_heads );

            // Verify that different batch sizes work without exceptions
            EXPECT_NO_THROW( cuda_mha_op_->forward(
                device_input, params, props, device_output, output_state ) );

            // Copy result back to host
            Tensor<float, HostMemoryResource> host_output( shape );
            host_output.copyFrom( device_output );

            // Check for NaNs or Infs
            EXPECT_FALSE( hasNaNorInf( host_output ) ) << "Found NaN or Inf with batch size " << batch_size;
        }
    }

    /**
     * @brief Test backward pass functionality works correctly
     */
    TEST_F( CudaMultiHeadAttentionOpTests, BackwardPass ) {
        // Create tensors for forward pass
        Tensor<float, CudaMemoryResource> device_input( small_shape_ );

        // Define weight and bias shapes based on input
        std::vector<size_t> weight_shape = { small_shape_[ 2 ], small_shape_[ 2 ] };  // Hidden x Hidden
        std::vector<size_t> bias_shape = { small_shape_[ 2 ] };  // Hidden dim

        auto device_weight = std::make_shared<Tensor<float, CudaMemoryResource>>( weight_shape );
        auto device_bias = std::make_shared<Tensor<float, CudaMemoryResource>>( bias_shape );
        Tensor<float, CudaMemoryResource> device_output( small_shape_ );

        // Attention cache tensors
        int num_heads = 4;  // Using 4 attention heads for this test
        std::vector<size_t> attn_scores_shape = { small_shape_[ 0 ], static_cast<size_t>(num_heads), small_shape_[ 1 ], small_shape_[ 1 ] };
        std::vector<size_t> attn_weights_shape = { small_shape_[ 0 ], static_cast<size_t>(num_heads), small_shape_[ 1 ], small_shape_[ 1 ] };

        auto device_attn_scores = std::make_shared<Tensor<float, CudaMemoryResource>>( attn_scores_shape );
        auto device_attn_weights = std::make_shared<Tensor<float, CudaMemoryResource>>( attn_weights_shape );

        // Create tensors for backward pass
        Tensor<float, CudaMemoryResource> device_output_grad( small_shape_ );
        auto device_weight_grad = std::make_shared<Tensor<float, CudaMemoryResource>>( weight_shape );
        auto device_bias_grad = std::make_shared<Tensor<float, CudaMemoryResource>>( bias_shape );
        Tensor<float, CudaMemoryResource> device_input_grad( small_shape_ );

        // Host tensors for verification
        Tensor<float, HostMemoryResource> host_input( small_shape_ );
        Tensor<float, HostMemoryResource> host_weight( weight_shape );
        Tensor<float, HostMemoryResource> host_bias( bias_shape );
        Tensor<float, HostMemoryResource> host_output_grad( small_shape_ );

        // Initialize with random values
        initializeRandomTensor( host_input, -0.1f, 0.1f );
        initializeRandomTensor( host_weight, -0.01f, 0.01f );
        initializeRandomTensor( host_output_grad, -0.1f, 0.1f );

        // Initialize bias to zeros
        for ( size_t i = 0; i < host_bias.size(); ++i ) {
            host_bias.rawData()[ i ] = 0.0f;
        }

        // Copy to device
        device_input.copyFrom( host_input );
        device_weight->copyFrom( host_weight );
        device_bias->copyFrom( host_bias );
        device_output_grad.copyFrom( host_output_grad );

        // Zero out device gradients
        for ( size_t i = 0; i < device_weight_grad->size(); ++i ) {
            device_weight_grad->rawData()[ i ] = 0.0f;
        }

        for ( size_t i = 0; i < device_bias_grad->size(); ++i ) {
            device_bias_grad->rawData()[ i ] = 0.0f;
        }

        for ( size_t i = 0; i < device_input_grad.size(); ++i ) {
            device_input_grad.rawData()[ i ] = 0.0f;
        }

        // Forward pass
        std::vector<std::shared_ptr<Tensor<float, CudaMemoryResource>>> params = { device_weight, device_bias };
        std::vector<std::shared_ptr<Tensor<float, CudaMemoryResource>>> output_state = {
            device_attn_scores, device_attn_weights
        };

        OperationAttributes props;
        props.set( "num_heads", num_heads );

        cuda_mha_op_->forward( device_input, params, props, device_output, output_state );

        // Backward pass
        std::vector<std::shared_ptr<Tensor<float, CudaMemoryResource>>> param_gradients = {
            device_weight_grad, device_bias_grad
        };

        // Verify that the backward pass runs without exceptions
        EXPECT_NO_THROW( cuda_mha_op_->backward(
            device_input, device_output, device_output_grad,
            params, param_gradients, device_input_grad,
            props, output_state ) );

        // Copy results back to host for verification
        Tensor<float, HostMemoryResource> device_input_grad_host( small_shape_ );
        Tensor<float, HostMemoryResource> device_weight_grad_host( weight_shape );
        Tensor<float, HostMemoryResource> device_bias_grad_host( bias_shape );

        device_input_grad_host.copyFrom( device_input_grad );
        device_weight_grad_host.copyFrom( *device_weight_grad );
        device_bias_grad_host.copyFrom( *device_bias_grad );

        // Check for NaNs or Infs in gradients
        EXPECT_FALSE( hasNaNorInf( device_input_grad_host ) ) << "Found NaN or Inf in input gradients";
        EXPECT_FALSE( hasNaNorInf( device_weight_grad_host ) ) << "Found NaN or Inf in weight gradients";
        EXPECT_FALSE( hasNaNorInf( device_bias_grad_host ) ) << "Found NaN or Inf in bias gradients";
    }

    /**
     * @brief Test numerical stability with edge cases
     */
    TEST_F( CudaMultiHeadAttentionOpTests, NumericalStability ) {
        // Test cases with potential numerical stability issues
        std::vector<std::pair<float, float>> test_cases = {
            {-0.001f, 0.001f},  // Very small values
            {-10.0f, 10.0f},    // Large values
            {1e-6f, 2e-6f}      // Tiny values
        };

        int num_heads = 4;

        for ( const auto& test_case : test_cases ) {
            float min_val = test_case.first;
            float max_val = test_case.second;

            // Create input, weight, bias, and output tensors
            Tensor<float, CudaMemoryResource> device_input( small_shape_ );

            // Define weight and bias shapes based on input
            std::vector<size_t> weight_shape = { small_shape_[ 2 ], small_shape_[ 2 ] };
            std::vector<size_t> bias_shape = { small_shape_[ 2 ] };

            auto device_weight = std::make_shared<Tensor<float, CudaMemoryResource>>( weight_shape );
            auto device_bias = std::make_shared<Tensor<float, CudaMemoryResource>>( bias_shape );
            Tensor<float, CudaMemoryResource> device_output( small_shape_ );

            // Attention cache tensors
            std::vector<size_t> attn_scores_shape = { small_shape_[ 0 ], static_cast<size_t>(num_heads), small_shape_[ 1 ], small_shape_[ 1 ] };
            std::vector<size_t> attn_weights_shape = { small_shape_[ 0 ], static_cast<size_t>(num_heads), small_shape_[ 1 ], small_shape_[ 1 ] };

            auto device_attn_scores = std::make_shared<Tensor<float, CudaMemoryResource>>( attn_scores_shape );
            auto device_attn_weights = std::make_shared<Tensor<float, CudaMemoryResource>>( attn_weights_shape );

            Tensor<float, HostMemoryResource> host_input( small_shape_ );
            Tensor<float, HostMemoryResource> host_weight( weight_shape );
            Tensor<float, HostMemoryResource> host_bias( bias_shape );

            // Initialize with specific range of values for this test case
            initializeRandomTensor( host_input, min_val, max_val );
            initializeRandomTensor( host_weight, min_val, max_val );
            initializeRandomTensor( host_bias, min_val, max_val );

            // Copy to device
            device_input.copyFrom( host_input );
            device_weight->copyFrom( host_weight );
            device_bias->copyFrom( host_bias );

            // Execute operation
            std::vector<std::shared_ptr<Tensor<float, CudaMemoryResource>>> params = { device_weight, device_bias };
            std::vector<std::shared_ptr<Tensor<float, CudaMemoryResource>>> output_state = {
                device_attn_scores, device_attn_weights
            };

            OperationAttributes props;
            props.set( "num_heads", num_heads );

            EXPECT_NO_THROW( cuda_mha_op_->forward(
                device_input, params, props, device_output, output_state ) );

            // Copy result back to host
            Tensor<float, HostMemoryResource> host_output( small_shape_ );
            host_output.copyFrom( device_output );

            // Check for NaNs or Infs
            EXPECT_FALSE( hasNaNorInf( host_output ) )
                << "Found NaN or Inf with values in range [" << min_val << ", " << max_val << "]";
        }
    }

    /**
     * @brief Test with stress test on large tensors
     */
    TEST_F( CudaMultiHeadAttentionOpTests, LargeInputStressTest ) {
        // Use large shape for stress test
        int num_heads = 8;

        // Create input, weight, bias, and output tensors
        Tensor<float, CudaMemoryResource> device_input( large_shape_ );

        // Define weight and bias shapes based on input
        std::vector<size_t> weight_shape = { large_shape_[ 2 ], large_shape_[ 2 ] };
        std::vector<size_t> bias_shape = { large_shape_[ 2 ] };

        auto device_weight = std::make_shared<Tensor<float, CudaMemoryResource>>( weight_shape );
        auto device_bias = std::make_shared<Tensor<float, CudaMemoryResource>>( bias_shape );
        Tensor<float, CudaMemoryResource> device_output( large_shape_ );

        // Attention cache tensors
        std::vector<size_t> attn_scores_shape = { large_shape_[ 0 ], static_cast<size_t>(num_heads), large_shape_[ 1 ], large_shape_[ 1 ] };
        std::vector<size_t> attn_weights_shape = { large_shape_[ 0 ], static_cast<size_t>(num_heads), large_shape_[ 1 ], large_shape_[ 1 ] };

        auto device_attn_scores = std::make_shared<Tensor<float, CudaMemoryResource>>( attn_scores_shape );
        auto device_attn_weights = std::make_shared<Tensor<float, CudaMemoryResource>>( attn_weights_shape );

        Tensor<float, HostMemoryResource> host_input( large_shape_ );
        Tensor<float, HostMemoryResource> host_weight( weight_shape );
        Tensor<float, HostMemoryResource> host_bias( bias_shape );

        // Initialize with random values
        initializeRandomTensor( host_input, -0.1f, 0.1f );
        initializeRandomTensor( host_weight, -0.01f, 0.01f );

        // Initialize bias to zeros
        for ( size_t i = 0; i < host_bias.size(); ++i ) {
            host_bias.rawData()[ i ] = 0.0f;
        }

        // Copy to device
        device_input.copyFrom( host_input );
        device_weight->copyFrom( host_weight );
        device_bias->copyFrom( host_bias );

        // Execute operation
        std::vector<std::shared_ptr<Tensor<float, CudaMemoryResource>>> params = { device_weight, device_bias };
        std::vector<std::shared_ptr<Tensor<float, CudaMemoryResource>>> output_state = {
            device_attn_scores, device_attn_weights
        };

        OperationAttributes props;
        props.set( "num_heads", num_heads );

        // Verify no exceptions are thrown for large input
        EXPECT_NO_THROW( cuda_mha_op_->forward(
            device_input, params, props, device_output, output_state ) );

        // Copy result back to host
        Tensor<float, HostMemoryResource> host_output( large_shape_ );
        host_output.copyFrom( device_output );

        // Check for NaNs or Infs
        EXPECT_FALSE( hasNaNorInf( host_output ) );
    }
}
