/**
 * @file CudaLayerNormOpTests.cpp
 * @brief Complete test suite for the CUDA Layer Normalization operation.
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
     * @brief Test fixture for CudaLayerNormOp tests
     */
    class CudaLayerNormOpTests : public ::testing::Test {
    protected:
        void SetUp() override {
            // Create device contexts for both CPU and CUDA
            cuda_context_ = std::make_shared<DeviceContext>( "CUDA:0" );
            cpu_context_ = std::make_shared<DeviceContext>( "CPU" );

            // Define shapes for testing
            small_shape_ = { 2, 3, 4 };  // Small shape for quick tests
            medium_shape_ = { 8, 16, 32 };  // Medium shape for thorough tests
            large_shape_ = { 16, 32, 128 };  // Large shape for stress tests

            // Create CUDA LayerNorm operation with specific context
            cuda_layernorm_op_ = std::make_shared<CudaLayerNormOp<float>>( cuda_context_ );

            // Get CPU LayerNorm op for comparison
            auto cpu_op = OperationRegistry::instance().createOperation<float, float, DeviceType::Cpu>(
                "Cpu::LayerNormOp", cpu_context_ );
            cpu_layernorm_op_ = std::static_pointer_cast<UnaryOperation<float, float, DeviceType::Cpu>>(cpu_op);
        }

        // Helper method to calculate reference LayerNorm result
        void referenceLayerNorm(
            const Tensor<float, HostMemoryResource>& input,
            const Tensor<float, HostMemoryResource>& weight,
            const Tensor<float, HostMemoryResource>& bias,
            Tensor<float, HostMemoryResource>& output,
            Tensor<float, HostMemoryResource>& mean,
            Tensor<float, HostMemoryResource>& rstd,
            float epsilon ) {

            int B = input.shape()[ 0 ];
            int T = input.shape()[ 1 ];
            int C = input.shape()[ 2 ];

            for ( int b = 0; b < B; b++ ) {
                for ( int t = 0; t < T; t++ ) {
                    float sum = 0.0f;
                    float sum_sq = 0.0f;

                    // Compute mean and variance
                    for ( int c = 0; c < C; c++ ) {
                        float val = input.data()[ (b * T + t) * C + c ];
                        sum += val;
                        sum_sq += val * val;
                    }
                    float mean_val = sum / C;
                    float var_val = sum_sq / C - mean_val * mean_val;
                    float rstd_val = 1.0f / std::sqrt( var_val + epsilon );

                    mean.data()[ b * T + t ] = mean_val;
                    rstd.data()[ b * T + t ] = rstd_val;

                    // Normalize and apply scale/shift
                    for ( int c = 0; c < C; c++ ) {
                        float val = input.data()[ (b * T + t) * C + c ];
                        float norm_val = (val - mean_val) * rstd_val;
                        output.data()[ (b * T + t) * C + c ] = norm_val * weight.data()[ c ] + bias.data()[ c ];
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

        // Helper method to initialize tensors with random values
        void initializeRandomTensor( Tensor<float, HostMemoryResource>& tensor, float min = -1.0f, float max = 1.0f ) {
            std::random_device rd;
            std::mt19937 gen( rd() );
            std::uniform_real_distribution<float> dist( min, max );

            for ( size_t i = 0; i < tensor.size(); ++i ) {
                tensor.data()[ i ] = dist( gen );
            }
        }

        // Helper method to compute numerical gradient
        void computeNumericalGradient(
            const Tensor<float, HostMemoryResource>& input,
            const Tensor<float, HostMemoryResource>& weight,
            const Tensor<float, HostMemoryResource>& bias,
            const Tensor<float, HostMemoryResource>& output_grad,
            Tensor<float, HostMemoryResource>& input_grad,
            Tensor<float, HostMemoryResource>& weight_grad,
            Tensor<float, HostMemoryResource>& bias_grad,
            float epsilon = 1e-5f,
            float delta = 1e-4f ) {

            size_t B = input.shape()[ 0 ];
            size_t T = input.shape()[ 1 ];
            size_t C = input.shape()[ 2 ];

            // Temporary tensors for forward pass
            Tensor<float, HostMemoryResource> output( input.shape() );
            Tensor<float, HostMemoryResource> mean( { B * T } );
            Tensor<float, HostMemoryResource> rstd( { B * T } );
            Tensor<float, HostMemoryResource> perturbed_input( input.shape() );
            Tensor<float, HostMemoryResource> perturbed_output( input.shape() );

            // Copy input to perturbed input
            std::copy( input.raw_data(), input.raw_data() + input.size(), perturbed_input.raw_data() );

            // Compute input gradients
            for ( size_t i = 0; i < input.size(); ++i ) {
                // Compute output with original input
                referenceLayerNorm( input, weight, bias, output, mean, rstd, epsilon );
                float original_loss = 0.0f;
                for ( size_t j = 0; j < output.size(); ++j ) {
                    original_loss += output.raw_data()[ j ] * output_grad.raw_data()[ j ];
                }

                // Perturb input
                perturbed_input.data()[ i ] = input.data()[ i ] + delta;

                // Compute output with perturbed input
                referenceLayerNorm( perturbed_input, weight, bias, perturbed_output, mean, rstd, epsilon );
                float perturbed_loss = 0.0f;
                for ( size_t j = 0; j < perturbed_output.size(); ++j ) {
                    perturbed_loss += perturbed_output.data()[ j ] * output_grad.data()[ j ];
                }

                // Restore perturbed input
                perturbed_input.data()[ i ] = input.data()[ i ];

                // Compute numerical gradient
                input_grad.data()[ i ] = (perturbed_loss - original_loss) / delta;
            }

            // Compute weight gradients
            Tensor<float, HostMemoryResource> perturbed_weight( weight.shape() );
            std::copy( weight.raw_data(), weight.raw_data() + weight.size(), perturbed_weight.raw_data() );

            for ( size_t i = 0; i < weight.size(); ++i ) {
                // Compute output with original weights
                referenceLayerNorm( input, weight, bias, output, mean, rstd, epsilon );
                float original_loss = 0.0f;
                for ( size_t j = 0; j < output.size(); ++j ) {
                    original_loss += output.data()[ j ] * output_grad.data()[ j ];
                }

                // Perturb weight
                perturbed_weight.data()[ i ] = weight.data()[ i ] + delta;

                // Compute output with perturbed weight
                referenceLayerNorm( input, perturbed_weight, bias, perturbed_output, mean, rstd, epsilon );
                float perturbed_loss = 0.0f;
                for ( size_t j = 0; j < perturbed_output.size(); ++j ) {
                    perturbed_loss += perturbed_output.data()[ j ] * output_grad.data()[ j ];
                }

                // Restore perturbed weight
                perturbed_weight.data()[ i ] = weight.data()[ i ];

                // Compute numerical gradient
                weight_grad.data()[ i ] = (perturbed_loss - original_loss) / delta;
            }

            // Compute bias gradients
            Tensor<float, HostMemoryResource> perturbed_bias( bias.shape() );
            std::copy( bias.raw_data(), bias.raw_data() + bias.size(), perturbed_bias.raw_data() );

            for ( size_t i = 0; i < bias.size(); ++i ) {
                // Compute output with original bias
                referenceLayerNorm( input, weight, bias, output, mean, rstd, epsilon );
                float original_loss = 0.0f;
                for ( size_t j = 0; j < output.size(); ++j ) {
                    original_loss += output.data()[ j ] * output_grad.data()[ j ];
                }

                // Perturb bias
                perturbed_bias.data()[ i ] = bias.data()[ i ] + delta;

                // Compute output with perturbed bias
                referenceLayerNorm( input, weight, perturbed_bias, perturbed_output, mean, rstd, epsilon );
                float perturbed_loss = 0.0f;
                for ( size_t j = 0; j < perturbed_output.size(); ++j ) {
                    perturbed_loss += perturbed_output.data()[ j ] * output_grad.data()[ j ];
                }

                // Restore perturbed bias
                perturbed_bias.data()[ i ] = bias.data()[ i ];

                // Compute numerical gradient
                bias_grad.data()[ i ] = (perturbed_loss - original_loss) / delta;
            }
        }

        std::shared_ptr<DeviceContext> cuda_context_;
        std::shared_ptr<DeviceContext> cpu_context_;
        std::shared_ptr<CudaLayerNormOp<float>> cuda_layernorm_op_;
        std::shared_ptr<UnaryOperation<float, float, DeviceType::Cpu>> cpu_layernorm_op_;

        // Test shapes
        std::vector<size_t> small_shape_;
        std::vector<size_t> medium_shape_;
        std::vector<size_t> large_shape_;
    };

    /**
     * @brief Test name property of CudaLayerNormOp
     */
    TEST_F( CudaLayerNormOpTests, Name ) {
        EXPECT_EQ( cuda_layernorm_op_->getName(), "Cuda::LayerNormOp" );
    }

    /**
     * @brief Test constructor without explicit device context
     */
    TEST_F( CudaLayerNormOpTests, DefaultConstructor ) {
        auto op = std::make_shared<CudaLayerNormOp<float>>();
        EXPECT_EQ( op->getName(), "Cuda::LayerNormOp" );
    }

    /**
     * @brief Test constructor with non-CUDA device context throws exception
     */
    TEST_F( CudaLayerNormOpTests, ConstructorWithNonCudaContext ) {
        EXPECT_THROW( ( std::make_shared<CudaLayerNormOp<float>>( cpu_context_ ) ), std::runtime_error );
    }

    /**
     * @brief Test basic functionality with small tensors
     */
    TEST_F( CudaLayerNormOpTests, BasicFunctionality ) {
        // Create input, weight, bias, and output tensors
        Tensor<float, DeviceMemoryResource> device_input( small_shape_ );
        auto device_weight = std::make_shared<Tensor<float, DeviceMemoryResource>>( std::vector<size_t>{small_shape_[ 2 ]} );
        auto device_bias = std::make_shared<Tensor<float, DeviceMemoryResource>>( std::vector<size_t>{small_shape_[ 2 ]} );
        Tensor<float, DeviceMemoryResource> device_output( small_shape_ );
        auto device_mean = std::make_shared<Tensor<float, DeviceMemoryResource>>( std::vector<size_t>{small_shape_[ 0 ] * small_shape_[ 1 ]} );
        auto device_rstd = std::make_shared<Tensor<float, DeviceMemoryResource>>( std::vector<size_t>{small_shape_[ 0 ] * small_shape_[ 1 ]} );

        Tensor<float, HostMemoryResource> host_input( small_shape_ );
        Tensor<float, HostMemoryResource> host_weight( std::vector<size_t>{small_shape_[ 2 ]} );
        Tensor<float, HostMemoryResource> host_bias( std::vector<size_t>{small_shape_[ 2 ]} );
        Tensor<float, HostMemoryResource> host_output( small_shape_ );
        Tensor<float, HostMemoryResource> host_mean( std::vector<size_t>{small_shape_[ 0 ] * small_shape_[ 1 ]} );
        Tensor<float, HostMemoryResource> host_rstd( std::vector<size_t>{small_shape_[ 0 ] * small_shape_[ 1 ]} );
        Tensor<float, HostMemoryResource> expected_output( small_shape_ );
        Tensor<float, HostMemoryResource> expected_mean( std::vector<size_t>{small_shape_[ 0 ] * small_shape_[ 1 ]} );
        Tensor<float, HostMemoryResource> expected_rstd( std::vector<size_t>{small_shape_[ 0 ] * small_shape_[ 1 ]} );

        // Initialize input, weight, and bias with test values
        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input.data()[ i ] = static_cast<float>( i % 7 ) - 3.0f;  // Range from -3 to 3
        }
        for ( size_t i = 0; i < host_weight.size(); ++i ) {
            host_weight.data()[ i ] = 1.0f;  // Scale of 1.0
        }
        for ( size_t i = 0; i < host_bias.size(); ++i ) {
            host_bias.data()[ i ] = 0.0f;  // No shift
        }

        // Copy to device
        device_input.copyFrom( host_input );
        device_weight->copyFrom( host_weight );
        device_bias->copyFrom( host_bias );

        // Execute operation
        std::vector<std::shared_ptr<Tensor<float, DeviceMemoryResource>>> params = { device_weight, device_bias };
        std::vector<std::shared_ptr<Tensor<float, DeviceMemoryResource>>> output_cache = { device_mean, device_rstd };
        OperationAttributes props;
        props.epsilon = 1e-5f;

        ASSERT_NO_THROW( cuda_layernorm_op_->forward(
            device_input, params, props, device_output, output_cache ) );

        // Copy result back to host
        host_output.copyFrom( device_output );
        host_mean.copyFrom( *device_mean );
        host_rstd.copyFrom( *device_rstd );

        // Compute expected output with reference implementation
        referenceLayerNorm( host_input, host_weight, host_bias, expected_output, expected_mean, expected_rstd, props.epsilon );

        // Verify output has correct values
        EXPECT_TRUE( compareTensors( host_output, expected_output, 1e-4f ) );
        EXPECT_TRUE( compareTensors( host_mean, expected_mean, 1e-4f ) );
        EXPECT_TRUE( compareTensors( host_rstd, expected_rstd, 1e-4f ) );
    }

    /**
     * @brief Test that CUDA and CPU implementations produce equivalent results
     */
    TEST_F( CudaLayerNormOpTests, CudaCpuEquivalence ) {
        // Create input, weight, bias, and output tensors
        Tensor<float, DeviceMemoryResource> cuda_input( medium_shape_ );
        auto cuda_weight = std::make_shared<Tensor<float, DeviceMemoryResource>>( std::vector<size_t>{medium_shape_[ 2 ]} );
        auto cuda_bias = std::make_shared<Tensor<float, DeviceMemoryResource>>( std::vector<size_t>{medium_shape_[ 2 ]} );
        Tensor<float, DeviceMemoryResource> cuda_output( medium_shape_ );
        auto cuda_mean = std::make_shared<Tensor<float, DeviceMemoryResource>>( std::vector<size_t>{medium_shape_[ 0 ] * medium_shape_[ 1 ]} );
        auto cuda_rstd = std::make_shared<Tensor<float, DeviceMemoryResource>>( std::vector<size_t>{medium_shape_[ 0 ] * medium_shape_[ 1 ]} );

        Tensor<float, HostMemoryResource> cpu_input( medium_shape_ );
        Tensor<float, HostMemoryResource> cpu_weight( std::vector<size_t>{medium_shape_[ 2 ]} );
        Tensor<float, HostMemoryResource> cpu_bias( std::vector<size_t>{medium_shape_[ 2 ]} );
        Tensor<float, HostMemoryResource> cpu_output( medium_shape_ );
        Tensor<float, HostMemoryResource> cuda_output_host( medium_shape_ );

        // Initialize input, weight, and bias with test values
        for ( size_t i = 0; i < cpu_input.size(); ++i ) {
            cpu_input.data()[ i ] = static_cast<float>( i % 7 ) - 3.0f;  // Range from -3 to 3
        }
        for ( size_t i = 0; i < cpu_weight.size(); ++i ) {
            cpu_weight.data()[ i ] = 1.0f;  // Scale of 1.0
        }
        for ( size_t i = 0; i < cpu_bias.size(); ++i ) {
            cpu_bias.data()[ i ] = 0.0f;  // No shift
        }

        // Copy to CUDA device
        cuda_input.copyFrom( cpu_input );
        cuda_weight->copyFrom( cpu_weight );
        cuda_bias->copyFrom( cpu_bias );

        // Execute CUDA LayerNorm operation
        std::vector<std::shared_ptr<Tensor<float, DeviceMemoryResource>>> cuda_params = { cuda_weight, cuda_bias };
        std::vector<std::shared_ptr<Tensor<float, DeviceMemoryResource>>> cuda_output_cache = { cuda_mean, cuda_rstd };
        OperationAttributes cuda_props;
        cuda_props.epsilon = 1e-5f;

        cuda_layernorm_op_->forward( cuda_input, cuda_params, cuda_props, cuda_output, cuda_output_cache );

        // Execute CPU LayerNorm operation
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> cpu_params = { 
            std::make_shared<Tensor<float, HostMemoryResource>>( cpu_weight ),
            std::make_shared<Tensor<float, HostMemoryResource>>( cpu_bias ) };
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> cpu_output_cache;
        OperationAttributes cpu_props;
        cpu_props.epsilon = 1e-5f;

        cpu_layernorm_op_->forward( cpu_input, cpu_params, cpu_props, cpu_output, cpu_output_cache );

        // Copy CUDA result back to host
        cuda_output_host.copyFrom( cuda_output );

        // Compare results (with small tolerance for floating point differences)
        EXPECT_TRUE( compareTensors( cpu_output, cuda_output_host, 1e-4f ) );
    }

    /**
     * @brief Test backward pass functionality works correctly
     */
    TEST_F( CudaLayerNormOpTests, BackwardPass ) {
        // Create tensors for forward pass
        Tensor<float, DeviceMemoryResource> device_input( small_shape_ );
        auto device_weight = std::make_shared<Tensor<float, DeviceMemoryResource>>( std::vector<size_t>{small_shape_[ 2 ]} );
        auto device_bias = std::make_shared<Tensor<float, DeviceMemoryResource>>( std::vector<size_t>{small_shape_[ 2 ]} );
        Tensor<float, DeviceMemoryResource> device_output( small_shape_ );
        auto device_mean = std::make_shared<Tensor<float, DeviceMemoryResource>>( std::vector<size_t>{small_shape_[ 0 ] * small_shape_[ 1 ]} );
        auto device_rstd = std::make_shared<Tensor<float, DeviceMemoryResource>>( std::vector<size_t>{small_shape_[ 0 ] * small_shape_[ 1 ]} );

        // Create tensors for backward pass
        Tensor<float, DeviceMemoryResource> device_output_grad( small_shape_ );
        auto device_weight_grad = std::make_shared<Tensor<float, DeviceMemoryResource>>( std::vector<size_t>{small_shape_[ 2 ]} );
        auto device_bias_grad = std::make_shared<Tensor<float, DeviceMemoryResource>>( std::vector<size_t>{small_shape_[ 2 ]} );
        Tensor<float, DeviceMemoryResource> device_input_grad( small_shape_ );

        // Host tensors for verification
        Tensor<float, HostMemoryResource> host_input( small_shape_ );
        Tensor<float, HostMemoryResource> host_weight( std::vector<size_t>{small_shape_[ 2 ]} );
        Tensor<float, HostMemoryResource> host_bias( std::vector<size_t>{small_shape_[ 2 ]} );
        Tensor<float, HostMemoryResource> host_output_grad( small_shape_ );
        Tensor<float, HostMemoryResource> host_input_grad( small_shape_ );
        Tensor<float, HostMemoryResource> host_weight_grad( std::vector<size_t>{small_shape_[ 2 ]} );
        Tensor<float, HostMemoryResource> host_bias_grad( std::vector<size_t>{small_shape_[ 2 ]} );
        Tensor<float, HostMemoryResource> device_input_grad_host( small_shape_ );
        Tensor<float, HostMemoryResource> device_weight_grad_host( std::vector<size_t>{small_shape_[ 2 ]} );
        Tensor<float, HostMemoryResource> device_bias_grad_host( std::vector<size_t>{small_shape_[ 2 ]} );

        // Initialize with random values
        initializeRandomTensor( host_input, -1.0f, 1.0f );
        initializeRandomTensor( host_weight, 0.5f, 1.5f );
        initializeRandomTensor( host_bias, -0.5f, 0.5f );
        initializeRandomTensor( host_output_grad, -0.1f, 0.1f );

        // Zero out gradients
        for ( size_t i = 0; i < host_input_grad.size(); ++i ) {
            host_input_grad.data()[ i ] = 0.0f;
        }
        for ( size_t i = 0; i < host_weight_grad.size(); ++i ) {
            host_weight_grad.data()[ i ] = 0.0f;
        }
        for ( size_t i = 0; i < host_bias_grad.size(); ++i ) {
            host_bias_grad.data()[ i ] = 0.0f;
        }

        // Copy data to device
        device_input.copyFrom( host_input );
        device_weight->copyFrom( host_weight );
        device_bias->copyFrom( host_bias );
        device_output_grad.copyFrom( host_output_grad );
        device_weight_grad->copyFrom( host_weight_grad );
        device_bias_grad->copyFrom( host_bias_grad );
        device_input_grad.copyFrom( host_input_grad );

        // Forward pass
        std::vector<std::shared_ptr<Tensor<float, DeviceMemoryResource>>> params = { device_weight, device_bias };
        std::vector<std::shared_ptr<Tensor<float, DeviceMemoryResource>>> output_cache = { device_mean, device_rstd };
        OperationAttributes props;
        props.epsilon = 1e-5f;

        cuda_layernorm_op_->forward( device_input, params, props, device_output, output_cache );

        // Backward pass
        std::vector<std::shared_ptr<Tensor<float, DeviceMemoryResource>>> param_gradients = {
            device_weight_grad, device_bias_grad
        };

        cuda_layernorm_op_->backward(
            device_input, device_output, device_output_grad,
            params, param_gradients, device_input_grad,
            props, output_cache );

        // Copy results back to host for verification
        device_input_grad_host.copyFrom( device_input_grad );
        device_weight_grad_host.copyFrom( *device_weight_grad );
        device_bias_grad_host.copyFrom( *device_bias_grad );

        // Compute numerical gradients for verification
        Tensor<float, HostMemoryResource> numerical_input_grad( small_shape_ );
        Tensor<float, HostMemoryResource> numerical_weight_grad( std::vector<size_t>{small_shape_[ 2 ]} );
        Tensor<float, HostMemoryResource> numerical_bias_grad( std::vector<size_t>{small_shape_[ 2 ]} );

        computeNumericalGradient(
            host_input, host_weight, host_bias, host_output_grad,
            numerical_input_grad, numerical_weight_grad, numerical_bias_grad,
            props.epsilon, 1e-4f );

        // Compare analytical gradients with numerical approximations
        // Use higher tolerance for complex operations like LayerNorm
        EXPECT_TRUE( compareTensors( device_input_grad_host, numerical_input_grad, 1e-3f ) );
        EXPECT_TRUE( compareTensors( device_weight_grad_host, numerical_weight_grad, 1e-3f ) );
        EXPECT_TRUE( compareTensors( device_bias_grad_host, numerical_bias_grad, 1e-3f ) );
    }

    /**
     * @brief Test with non-unity weights and non-zero biases
     */
    TEST_F( CudaLayerNormOpTests, NonDefaultWeightsBias ) {
        // Create input, weight, bias, and output tensors
        Tensor<float, DeviceMemoryResource> device_input( small_shape_ );
        auto device_weight = std::make_shared<Tensor<float, DeviceMemoryResource>>( std::vector<size_t>{small_shape_[ 2 ]} );
        auto device_bias = std::make_shared<Tensor<float, DeviceMemoryResource>>( std::vector<size_t>{small_shape_[ 2 ]} );
        Tensor<float, DeviceMemoryResource> device_output( small_shape_ );
        auto device_mean = std::make_shared<Tensor<float, DeviceMemoryResource>>( std::vector<size_t>{small_shape_[ 0 ] * small_shape_[ 1 ]} );
        auto device_rstd = std::make_shared<Tensor<float, DeviceMemoryResource>>( std::vector<size_t>{small_shape_[ 0 ] * small_shape_[ 1 ]} );

        Tensor<float, HostMemoryResource> host_input( small_shape_ );
        Tensor<float, HostMemoryResource> host_weight( std::vector<size_t>{small_shape_[ 2 ]} );
        Tensor<float, HostMemoryResource> host_bias( std::vector<size_t>{small_shape_[ 2 ]} );
        Tensor<float, HostMemoryResource> host_output( small_shape_ );
        Tensor<float, HostMemoryResource> host_mean( std::vector<size_t>{small_shape_[ 0 ] * small_shape_[ 1 ]} );
        Tensor<float, HostMemoryResource> host_rstd( std::vector<size_t>{small_shape_[ 0 ] * small_shape_[ 1 ]} );
        Tensor<float, HostMemoryResource> expected_output( small_shape_ );
        Tensor<float, HostMemoryResource> expected_mean( std::vector<size_t>{small_shape_[ 0 ] * small_shape_[ 1 ]} );
        Tensor<float, HostMemoryResource> expected_rstd( std::vector<size_t>{small_shape_[ 0 ] * small_shape_[ 1 ]} );

        // Initialize input with test values
        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input.data()[ i ] = static_cast<float>( i % 7 ) - 3.0f;  // Range from -3 to 3
        }

        // Initialize non-unity weights and non-zero biases
        for ( size_t i = 0; i < host_weight.size(); ++i ) {
            host_weight.data()[ i ] = 2.0f - (i % 3) * 0.5f;  // Range from 1.0 to 2.0
        }
        for ( size_t i = 0; i < host_bias.size(); ++i ) {
            host_bias.data()[ i ] = 0.5f - (i % 5) * 0.2f;  // Range from -0.3 to 0.5
        }

        // Copy to device
        device_input.copyFrom( host_input );
        device_weight->copyFrom( host_weight );
        device_bias->copyFrom( host_bias );

        // Execute operation
        std::vector<std::shared_ptr<Tensor<float, DeviceMemoryResource>>> params = { device_weight, device_bias };
        std::vector<std::shared_ptr<Tensor<float, DeviceMemoryResource>>> output_cache = { device_mean, device_rstd };
        OperationAttributes props;
        props.epsilon = 1e-5f;

        cuda_layernorm_op_->forward(
            device_input, params, props, device_output, output_cache );

        // Copy result back to host
        host_output.copyFrom( device_output );
        host_mean.copyFrom( *device_mean );
        host_rstd.copyFrom( *device_rstd );

        // Compute expected output with reference implementation
        referenceLayerNorm( host_input, host_weight, host_bias, expected_output, expected_mean, expected_rstd, props.epsilon );

        // Verify output has correct values
        EXPECT_TRUE( compareTensors( host_output, expected_output, 1e-4f ) );
        EXPECT_TRUE( compareTensors( host_mean, expected_mean, 1e-4f ) );
        EXPECT_TRUE( compareTensors( host_rstd, expected_rstd, 1e-4f ) );
    }

    /**
     * @brief Test with different epsilon values
     */
    TEST_F( CudaLayerNormOpTests, EpsilonVariations ) {
        std::vector<float> epsilon_values = { 1e-3f, 1e-5f, 1e-7f };

        for ( float epsilon : epsilon_values ) {
            // Create input, weight, bias, and output tensors
            Tensor<float, DeviceMemoryResource> device_input( small_shape_ );
            auto device_weight = std::make_shared<Tensor<float, DeviceMemoryResource>>( std::vector<size_t>{small_shape_[ 2 ]} );
            auto device_bias = std::make_shared<Tensor<float, DeviceMemoryResource>>( std::vector<size_t>{small_shape_[ 2 ]} );
            Tensor<float, DeviceMemoryResource> device_output( small_shape_ );
            auto device_mean = std::make_shared<Tensor<float, DeviceMemoryResource>>( std::vector<size_t>{small_shape_[ 0 ] * small_shape_[ 1 ]} );
            auto device_rstd = std::make_shared<Tensor<float, DeviceMemoryResource>>( std::vector<size_t>{small_shape_[ 0 ] * small_shape_[ 1 ]} );

            Tensor<float, HostMemoryResource> host_input( small_shape_ );
            Tensor<float, HostMemoryResource> host_weight( std::vector<size_t>{small_shape_[ 2 ]} );
            Tensor<float, HostMemoryResource> host_bias( std::vector<size_t>{small_shape_[ 2 ]} );
            Tensor<float, HostMemoryResource> host_output( small_shape_ );
            Tensor<float, HostMemoryResource> expected_output( small_shape_ );
            Tensor<float, HostMemoryResource> expected_mean( std::vector<size_t>{small_shape_[ 0 ] * small_shape_[ 1 ]} );
            Tensor<float, HostMemoryResource> expected_rstd( std::vector<size_t>{small_shape_[ 0 ] * small_shape_[ 1 ]} );

            // Initialize with random values
            initializeRandomTensor( host_input, -1.0f, 1.0f );
            initializeRandomTensor( host_weight, 0.8f, 1.2f );
            initializeRandomTensor( host_bias, -0.2f, 0.2f );

            // Copy to device
            device_input.copyFrom( host_input );
            device_weight->copyFrom( host_weight );
            device_bias->copyFrom( host_bias );

            // Execute operation
            std::vector<std::shared_ptr<Tensor<float, DeviceMemoryResource>>> params = { device_weight, device_bias };
            std::vector<std::shared_ptr<Tensor<float, DeviceMemoryResource>>> output_cache = { device_mean, device_rstd };
            OperationAttributes props;
            props.epsilon = epsilon;

            cuda_layernorm_op_->forward(
                device_input, params, props, device_output, output_cache );

            // Copy result back to host
            host_output.copyFrom( device_output );

            // Compute expected output with reference implementation
            referenceLayerNorm( host_input, host_weight, host_bias, expected_output, expected_mean, expected_rstd, epsilon );

            // Verify output has correct values
            EXPECT_TRUE( compareTensors( host_output, expected_output, 1e-4f ) );
        }
    }

    /**
     * @brief Test with tensors of different sizes
     */
    TEST_F( CudaLayerNormOpTests, DifferentSizes ) {
        std::vector<std::vector<size_t>> test_shapes = {
            {1, 1, 4},    // Single element, tiny layer
            {1, 5, 16},   // Single batch, small layer
            {4, 1, 32},   // Small batch, single sequence, medium layer
            {16, 24, 64}  // Medium batch and sequence, large layer
        };

        for ( const auto& shape : test_shapes ) {
            // Create input, weight, bias, and output tensors
            Tensor<float, DeviceMemoryResource> device_input( shape );
            auto device_weight = std::make_shared<Tensor<float, DeviceMemoryResource>>( std::vector<size_t>{shape[ 2 ]} );
            auto device_bias = std::make_shared<Tensor<float, DeviceMemoryResource>>( std::vector<size_t>{shape[ 2 ]} );
            Tensor<float, DeviceMemoryResource> device_output( shape );
            auto device_mean = std::make_shared<Tensor<float, DeviceMemoryResource>>( std::vector<size_t>{shape[ 0 ] * shape[ 1 ]} );
            auto device_rstd = std::make_shared<Tensor<float, DeviceMemoryResource>>( std::vector<size_t>{shape[ 0 ] * shape[ 1 ]} );

            Tensor<float, HostMemoryResource> host_input( shape );
            Tensor<float, HostMemoryResource> host_weight( std::vector<size_t>{shape[ 2 ]} );
            Tensor<float, HostMemoryResource> host_bias( std::vector<size_t>{shape[ 2 ]} );
            Tensor<float, HostMemoryResource> host_output( shape );
            Tensor<float, HostMemoryResource> expected_output( shape );
            Tensor<float, HostMemoryResource> expected_mean( std::vector<size_t>{shape[ 0 ] * shape[ 1 ]} );
            Tensor<float, HostMemoryResource> expected_rstd( std::vector<size_t>{shape[ 0 ] * shape[ 1 ]} );

            // Initialize with random values
            initializeRandomTensor( host_input, -1.0f, 1.0f );
            initializeRandomTensor( host_weight, 0.8f, 1.2f );
            initializeRandomTensor( host_bias, -0.2f, 0.2f );

            // Copy to device
            device_input.copyFrom( host_input );
            device_weight->copyFrom( host_weight );
            device_bias->copyFrom( host_bias );

            // Execute operation
            std::vector<std::shared_ptr<Tensor<float, DeviceMemoryResource>>> params = { device_weight, device_bias };
            std::vector<std::shared_ptr<Tensor<float, DeviceMemoryResource>>> output_cache = { device_mean, device_rstd };
            OperationAttributes props;
            props.epsilon = 1e-5f;

            cuda_layernorm_op_->forward(
                device_input, params, props, device_output, output_cache );

            // Copy result back to host
            host_output.copyFrom( device_output );

            // Compute expected output with reference implementation
            referenceLayerNorm( host_input, host_weight, host_bias, expected_output, expected_mean, expected_rstd, props.epsilon );

            // Verify output has correct values (use larger tolerance for larger tensors)
            EXPECT_TRUE( compareTensors( host_output, expected_output, 1e-4f ) );
        }
    }

    /**
     * @brief Test with stress test on large tensors
     */
    TEST_F( CudaLayerNormOpTests, LargeInputStressTest ) {
        // Create input, weight, bias, and output tensors
        Tensor<float, DeviceMemoryResource> device_input( large_shape_ );
        auto device_weight = std::make_shared<Tensor<float, DeviceMemoryResource>>( std::vector<size_t>{large_shape_[ 2 ]} );
        auto device_bias = std::make_shared<Tensor<float, DeviceMemoryResource>>( std::vector<size_t>{large_shape_[ 2 ]} );
        Tensor<float, DeviceMemoryResource> device_output( large_shape_ );
        auto device_mean = std::make_shared<Tensor<float, DeviceMemoryResource>>( std::vector<size_t>{large_shape_[ 0 ] * large_shape_[ 1 ]} );
        auto device_rstd = std::make_shared<Tensor<float, DeviceMemoryResource>>( std::vector<size_t>{large_shape_[ 0 ] * large_shape_[ 1 ]} );

        Tensor<float, HostMemoryResource> host_input( large_shape_ );
        Tensor<float, HostMemoryResource> host_weight( std::vector<size_t>{large_shape_[ 2 ]} );
        Tensor<float, HostMemoryResource> host_bias( std::vector<size_t>{large_shape_[ 2 ]} );
        Tensor<float, HostMemoryResource> host_output( large_shape_ );

        // Initialize with random values
        initializeRandomTensor( host_input, -1.0f, 1.0f );
        initializeRandomTensor( host_weight, 0.8f, 1.2f );
        initializeRandomTensor( host_bias, -0.2f, 0.2f );

        // Copy to device
        device_input.copyFrom( host_input );
        device_weight->copyFrom( host_weight );
        device_bias->copyFrom( host_bias );

        // Execute operation
        std::vector<std::shared_ptr<Tensor<float, DeviceMemoryResource>>> params = { device_weight, device_bias };
        std::vector<std::shared_ptr<Tensor<float, DeviceMemoryResource>>> output_cache = { device_mean, device_rstd };
        OperationAttributes props;
        props.epsilon = 1e-5f;

        // Verify no exceptions are thrown for large input
        EXPECT_NO_THROW( cuda_layernorm_op_->forward(
            device_input, params, props, device_output, output_cache ) );

        // Copy result back to host
        host_output.copyFrom( device_output );

        // Check for NaNs or Infs
        EXPECT_FALSE( hasNaNorInf( host_output ) );
    }
}
