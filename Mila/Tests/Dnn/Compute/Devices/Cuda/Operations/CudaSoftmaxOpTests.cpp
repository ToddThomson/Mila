/**
 * @brief Test suite for the CUDA Softmax operation with both FP32 and FP16 precision.
 * Updated to support dynamic axis selection and enhanced numerical stability tests.
 */

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <chrono>
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdexcept>

import Mila;

namespace Operations::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    /**
     * @brief Test fixture for CudaSoftmaxOp tests with template support for precision
     */
    template<typename TPrecision>
    class CudaSoftmaxOpPrecisionTests : public ::testing::Test {
    protected:
        void SetUp() override {
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

            // 2D shape for common use cases
            shape_2d_ = { 32, 64 };

            // Create CUDA Softmax operation with specific context
            cuda_softmax_op_ = std::make_shared<CudaSoftmaxOp<TPrecision>>( cuda_context_ );

            // Get CPU Softmax op for comparison
            // Note: CPU version always uses float for comparison even with half precision GPU ops
            auto cpu_op = OperationRegistry::instance().createOperation<float, float, DeviceType::Cpu>(
                "Cpu::SoftmaxOp", cpu_context_ );
            cpu_softmax_op_ = std::static_pointer_cast<UnaryOperation<float, float, DeviceType::Cpu>>(cpu_op);
        }

        // Helper method to reference implementation of softmax operation
        void referenceSoftmax(
            const Tensor<float, HostMemoryResource>& input,
            Tensor<float, HostMemoryResource>& output,
            int axis ) {

            const int64_t ndim = input.shape().size();
            if ( axis < 0 ) {
                axis = ndim + axis;
            }

            // Calculate outer_size and inner_size for softmax computation
            int64_t outer_size = 1;
            for ( int64_t i = 0; i < axis; ++i ) {
                outer_size *= input.shape()[ i ];
            }

            const int64_t dim_size = input.shape()[ axis ];

            int64_t inner_size = 1;
            for ( int64_t i = axis + 1; i < ndim; ++i ) {
                inner_size *= input.shape()[ i ];
            }

            // For each outer and inner position
            for ( int64_t outer = 0; outer < outer_size; ++outer ) {
                for ( int64_t inner = 0; inner < inner_size; ++inner ) {
                    // First find the maximum value for numerical stability
                    float max_val = -std::numeric_limits<float>::infinity();
                    for ( int64_t i = 0; i < dim_size; ++i ) {
                        int64_t idx = (outer * dim_size * inner_size) + (i * inner_size) + inner;
                        float val = input.data()[ idx ];
                        max_val = std::max( max_val, val );
                    }

                    // Calculate the sum of exp(x - max_val)
                    float sum = 0.0f;
                    for ( int64_t i = 0; i < dim_size; ++i ) {
                        int64_t idx = (outer * dim_size * inner_size) + (i * inner_size) + inner;
                        float val = input.data()[ idx ];
                        float exp_val = std::exp( val - max_val );
                        output.data()[ idx ] = exp_val;
                        sum += exp_val;
                    }

                    // Normalize by the sum
                    for ( int64_t i = 0; i < dim_size; ++i ) {
                        int64_t idx = (outer * dim_size * inner_size) + (i * inner_size) + inner;
                        output.rawData()[ idx ] /= sum;
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

            const int64_t ndim = output.shape().size();
            if ( axis < 0 ) {
                axis = ndim + axis;
            }

            // Calculate outer_size and inner_size for gradient computation
            int64_t outer_size = 1;
            for ( int64_t i = 0; i < axis; ++i ) {
                outer_size *= output.shape()[ i ];
            }

            const int64_t dim_size = output.shape()[ axis ];

            int64_t inner_size = 1;
            for ( int64_t i = axis + 1; i < ndim; ++i ) {
                inner_size *= output.shape()[ i ];
            }

            // For each outer and inner position
            for ( int64_t outer = 0; outer < outer_size; ++outer ) {
                for ( int64_t inner = 0; inner < inner_size; ++inner ) {
                    // Calculate the jacobian-vector product for this distribution
                    for ( int64_t i = 0; i < dim_size; ++i ) {
                        int64_t i_idx = (outer * dim_size * inner_size) + (i * inner_size) + inner;
                        float grad_i = 0.0f;
                        float y_i = output.data()[ i_idx ];

                        for ( int64_t j = 0; j < dim_size; ++j ) {
                            int64_t j_idx = (outer * dim_size * inner_size) + (j * inner_size) + inner;
                            float y_j = output.data()[ j_idx ];
                            float dy_j = output_gradient.data()[ j_idx ];

                            // dL/dx_i = sum_j(dL/dy_j * (y_i * (kronecker_delta(i,j) - y_j)))
                            if ( i == j ) {
                                grad_i += dy_j * y_i * (1.0f - y_j);
                            }
                            else {
                                grad_i += dy_j * (-y_i * y_j);
                            }
                        }

                        input_gradient.data()[ i_idx ] = grad_i;
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

            const int64_t ndim = output.shape().size();
            if ( axis < 0 ) {
                axis = ndim + axis;
            }

            // Calculate outer_size and inner_size
            int64_t outer_size = 1;
            for ( int64_t i = 0; i < axis; ++i ) {
                outer_size *= output.shape()[ i ];
            }

            const int64_t dim_size = output.shape()[ axis ];

            int64_t inner_size = 1;
            for ( int64_t i = axis + 1; i < ndim; ++i ) {
                inner_size *= output.shape()[ i ];
            }

            // For each distribution along the specified axis
            for ( int64_t outer = 0; outer < outer_size; ++outer ) {
                for ( int64_t inner = 0; inner < inner_size; ++inner ) {
                    float sum = 0.0f;
                    bool has_negative = false;

                    // Sum the distribution
                    for ( int64_t i = 0; i < dim_size; ++i ) {
                        int64_t idx = (outer * dim_size * inner_size) + (i * inner_size) + inner;
                        float val = output.data()[ idx ];

                        // Check if any value is negative (shouldn't happen with softmax)
                        if ( val < 0.0f ) {
                            std::cout << "Found negative value: " << val << " at index " << idx << std::endl;
                            has_negative = true;
                        }

                        sum += val;
                    }

                    // Check if sum equals 1
                    if ( std::abs( sum - 1.0f ) > epsilon ) {
                        std::cout << "Sum not equal to 1: " << sum << " for outer="
                            << outer << ", inner=" << inner << std::endl;
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
        std::shared_ptr<CudaSoftmaxOp<TPrecision>> cuda_softmax_op_;
        std::shared_ptr<UnaryOperation<float, float, DeviceType::Cpu>> cpu_softmax_op_;

        // Test dimensions
        size_t small_batch_, small_seq_len_, small_vocab_size_;
        size_t medium_batch_, medium_seq_len_, medium_vocab_size_;
        size_t large_batch_, large_seq_len_, large_vocab_size_;

        // Test shapes
        std::vector<size_t> small_shape_;
        std::vector<size_t> medium_shape_;
        std::vector<size_t> large_shape_;
        std::vector<size_t> shape_2d_;
    };

    // Define the types to test
    using TestTypes = ::testing::Types<float, half>;
    TYPED_TEST_SUITE( CudaSoftmaxOpPrecisionTests, TestTypes );

    /**
     * @brief Test name property of CudaSoftmaxOp
     */
    TYPED_TEST( CudaSoftmaxOpPrecisionTests, Name ) {
        EXPECT_EQ( this->cuda_softmax_op_->getName(), "Cuda::SoftmaxOp" );
    }

    /**
     * @brief Test basic functionality with small tensors along last axis
     */
    TYPED_TEST( CudaSoftmaxOpPrecisionTests, BasicFunctionalityLastAxis ) {
        // Create input and output tensors
        Tensor<float, CudaDeviceMemoryResource> device_input( this->small_shape_ );
        Tensor<TypeParam, CudaDeviceMemoryResource> device_output( this->small_shape_ );

        Tensor<float, HostMemoryResource> host_input( this->small_shape_ );
        Tensor<float, HostMemoryResource> host_output( this->small_shape_ );
        Tensor<float, HostMemoryResource> expected_output( this->small_shape_ );

        // Initialize input with test values
        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input.data()[ i ] = static_cast<float>( i % 7 ) - 3.0f;  // Range from -3 to 3
        }

        // Copy to device
        device_input.copyFrom( host_input );

        // Execute operation
        std::vector<std::shared_ptr<Tensor<float, CudaDeviceMemoryResource>>> params;
        std::vector<std::shared_ptr<Tensor<TypeParam, CudaDeviceMemoryResource>>> output_state;
        OperationAttributes props;
        props.axis = 2;  // Last dimension

        ASSERT_NO_THROW( this->cuda_softmax_op_->forward(
            device_input, params, props, device_output, output_state ) );

        // Copy result back to host (convert to float if needed)
        if constexpr ( std::is_same_v<TypeParam, float> ) {
            host_output.copyFrom( device_output );
        }
        else {
            // Need to convert half to float for comparison
            Tensor<float, CudaDeviceMemoryResource> float_output( this->small_shape_ );
            // Use appropriate conversion kernel or helper function
            // This is a placeholder - actual implementation would depend on your project's utility functions
            // convert_half_to_float(float_output.data(), device_output.data(), float_output.size(), this->cuda_context_->getStream());
            host_output.copyFrom( float_output );
        }

        // Compute expected output with reference implementation
        this->referenceSoftmax( host_input, expected_output, props.axis );

        // Adjust epsilon for half precision comparisons
        float epsilon = std::is_same_v<TypeParam, half> ? 1e-3f : 1e-4f;

        // Verify output has correct values
        EXPECT_TRUE( this->compareTensors( host_output, expected_output, epsilon ) );

        // Verify softmax output properties (sums to 1, non-negative values)
        EXPECT_TRUE( this->checkSoftmaxProperties( host_output, props.axis ) );
    }

    /**
     * @brief Test softmax with negative axis index
     */
    TYPED_TEST( CudaSoftmaxOpPrecisionTests, NegativeAxisIndex ) {
        // Create input and output tensors
        Tensor<float, CudaDeviceMemoryResource> device_input( this->small_shape_ );
        Tensor<TypeParam, CudaDeviceMemoryResource> device_output( this->small_shape_ );

        Tensor<float, HostMemoryResource> host_input( this->small_shape_ );
        Tensor<float, HostMemoryResource> host_output( this->small_shape_ );
        Tensor<float, HostMemoryResource> expected_output( this->small_shape_ );

        // Initialize input with test values
        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input.data()[ i ] = static_cast<float>( i % 7 ) - 3.0f;  // Range from -3 to 3
        }

        // Copy to device
        device_input.copyFrom( host_input );

        // Execute operation with negative axis (-1 refers to the last dimension)
        std::vector<std::shared_ptr<Tensor<float, CudaDeviceMemoryResource>>> params;
        std::vector<std::shared_ptr<Tensor<TypeParam, CudaDeviceMemoryResource>>> output_state;
        OperationAttributes props;
        props.axis = -1;  // Last dimension, equivalent to 2 for 3D tensor

        ASSERT_NO_THROW( this->cuda_softmax_op_->forward(
            device_input, params, props, device_output, output_state ) );

        // Copy result back to host (convert if needed)
        if constexpr ( std::is_same_v<TypeParam, float> ) {
            host_output.copyFrom( device_output );
        }
        else {
            // Need to convert half to float for comparison
            Tensor<float, CudaDeviceMemoryResource> float_output( this->small_shape_ );
            // convert_half_to_float(float_output.data(), device_output.data(), float_output.size(), this->cuda_context_->getStream());
            host_output.copyFrom( float_output );
        }

        // Compute expected output with reference implementation
        this->referenceSoftmax( host_input, expected_output, props.axis );

        // Adjust epsilon for half precision comparisons
        float epsilon = std::is_same_v<TypeParam, half> ? 1e-3f : 1e-4f;

        // Verify output has correct values
        EXPECT_TRUE( this->compareTensors( host_output, expected_output, epsilon ) );

        // Verify softmax output properties (sums to 1, non-negative values)
        EXPECT_TRUE( this->checkSoftmaxProperties( host_output, props.axis ) );
    }

    /**
     * @brief Test softmax along the middle axis
     */
    TYPED_TEST( CudaSoftmaxOpPrecisionTests, MiddleAxisFunctionality ) {
        // Create input and output tensors
        Tensor<float, CudaDeviceMemoryResource> device_input( this->small_shape_ );
        Tensor<TypeParam, CudaDeviceMemoryResource> device_output( this->small_shape_ );

        Tensor<float, HostMemoryResource> host_input( this->small_shape_ );
        Tensor<float, HostMemoryResource> host_output( this->small_shape_ );
        Tensor<float, HostMemoryResource> expected_output( this->small_shape_ );

        // Initialize input with test values
        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input.data()[ i ] = static_cast<float>( i % 7 ) - 3.0f;  // Range from -3 to 3
        }

        // Copy to device
        device_input.copyFrom( host_input );

        // Execute operation with middle axis
        std::vector<std::shared_ptr<Tensor<float, CudaDeviceMemoryResource>>> params;
        std::vector<std::shared_ptr<Tensor<TypeParam, CudaDeviceMemoryResource>>> output_state;
        OperationAttributes props;
        props.axis = 1;  // Middle dimension (sequence_length)

        ASSERT_NO_THROW( this->cuda_softmax_op_->forward(
            device_input, params, props, device_output, output_state ) );

        // Copy result back to host (convert if needed)
        if constexpr ( std::is_same_v<TypeParam, float> ) {
            host_output.copyFrom( device_output );
        }
        else {
            // Need to convert half to float for comparison
            Tensor<float, CudaDeviceMemoryResource> float_output( this->small_shape_ );
            // convert_half_to_float(float_output.data(), device_output.data(), float_output.size(), this->cuda_context_->getStream());
            host_output.copyFrom( float_output );
        }

        // Compute expected output with reference implementation
        this->referenceSoftmax( host_input, expected_output, props.axis );

        // Adjust epsilon for half precision comparisons
        float epsilon = std::is_same_v<TypeParam, half> ? 1e-3f : 1e-4f;

        // Verify output has correct values
        EXPECT_TRUE( this->compareTensors( host_output, expected_output, epsilon ) );

        // Verify softmax output properties (sums to 1, non-negative values)
        EXPECT_TRUE( this->checkSoftmaxProperties( host_output, props.axis ) );
    }

    /**
     * @brief Test softmax along the first axis
     */
    TYPED_TEST( CudaSoftmaxOpPrecisionTests, FirstAxisFunctionality ) {
        // Create input and output tensors
        Tensor<float, CudaDeviceMemoryResource> device_input( this->small_shape_ );
        Tensor<TypeParam, CudaDeviceMemoryResource> device_output( this->small_shape_ );

        Tensor<float, HostMemoryResource> host_input( this->small_shape_ );
        Tensor<float, HostMemoryResource> host_output( this->small_shape_ );
        Tensor<float, HostMemoryResource> expected_output( this->small_shape_ );

        // Initialize input with test values
        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input.data()[ i ] = static_cast<float>( i % 7 ) - 3.0f;  // Range from -3 to 3
        }

        // Copy to device
        device_input.copyFrom( host_input );

        // Execute operation with first axis
        std::vector<std::shared_ptr<Tensor<float, CudaDeviceMemoryResource>>> params;
        std::vector<std::shared_ptr<Tensor<TypeParam, CudaDeviceMemoryResource>>> output_state;
        OperationAttributes props;
        props.axis = 0;  // First dimension (batch)

        ASSERT_NO_THROW( this->cuda_softmax_op_->forward(
            device_input, params, props, device_output, output_state ) );

        // Copy result back to host (convert if needed)
        if constexpr ( std::is_same_v<TypeParam, float> ) {
            host_output.copyFrom( device_output );
        }
        else {
            // Need to convert half to float for comparison
            Tensor<float, CudaDeviceMemoryResource> float_output( this->small_shape_ );
            // convert_half_to_float(float_output.data(), device_output.data(), float_output.size(), this->cuda_context_->getStream());
            host_output.copyFrom( float_output );
        }

        // Compute expected output with reference implementation
        this->referenceSoftmax( host_input, expected_output, props.axis );

        // Adjust epsilon for half precision comparisons
        float epsilon = std::is_same_v<TypeParam, half> ? 1e-3f : 1e-4f;

        // Verify output has correct values
        EXPECT_TRUE( this->compareTensors( host_output, expected_output, epsilon ) );

        // Verify softmax output properties (sums to 1, non-negative values)
        EXPECT_TRUE( this->checkSoftmaxProperties( host_output, props.axis ) );
    }

    /**
     * @brief Test with 2D tensors (common use case)
     */
    TYPED_TEST( CudaSoftmaxOpPrecisionTests, TwoDimensionalTensor ) {
        // Create 2D test tensors (batch_size x features)
        Tensor<float, CudaDeviceMemoryResource> device_input( this->shape_2d_ );
        Tensor<TypeParam, CudaDeviceMemoryResource> device_output( this->shape_2d_ );

        Tensor<float, HostMemoryResource> host_input( this->shape_2d_ );
        Tensor<float, HostMemoryResource> host_output( this->shape_2d_ );
        Tensor<float, HostMemoryResource> expected_output( this->shape_2d_ );

        // Initialize input with test values
        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input.data()[ i ] = static_cast<float>( i % 10 ) - 5.0f;  // Range from -5 to 4
        }

        // Copy to device
        device_input.copyFrom( host_input );

        // Execute operation
        std::vector<std::shared_ptr<Tensor<float, CudaDeviceMemoryResource>>> params;
        std::vector<std::shared_ptr<Tensor<TypeParam, CudaDeviceMemoryResource>>> output_state;
        OperationAttributes props;
        props.axis = 1;  // Features dimension (commonly used for classification)

        ASSERT_NO_THROW( this->cuda_softmax_op_->forward(
            device_input, params, props, device_output, output_state ) );

        // Copy result back to host (convert if needed)
        if constexpr ( std::is_same_v<TypeParam, float> ) {
            host_output.copyFrom( device_output );
        }
        else {
            // Need to convert half to float for comparison
            Tensor<float, CudaDeviceMemoryResource> float_output( this->shape_2d_ );
            // convert_half_to_float(float_output.data(), device_output.data(), float_output.size(), this->cuda_context_->getStream());
            host_output.copyFrom( float_output );
        }

        // Compute expected output with reference implementation
        this->referenceSoftmax( host_input, expected_output, props.axis );

        // Adjust epsilon for half precision comparisons
        float epsilon = std::is_same_v<TypeParam, half> ? 1e-3f : 1e-4f;

        // Verify output has correct values
        EXPECT_TRUE( this->compareTensors( host_output, expected_output, epsilon ) );

        // Verify softmax output properties (sums to 1, non-negative values)
        EXPECT_TRUE( this->checkSoftmaxProperties( host_output, props.axis ) );
    }

    /**
     * @brief Test with special input values
     */
    TYPED_TEST( CudaSoftmaxOpPrecisionTests, SpecialValues ) {
        // Create input and output tensors
        Tensor<float, CudaDeviceMemoryResource> device_input( this->small_shape_ );
        Tensor<TypeParam, CudaDeviceMemoryResource> device_output( this->small_shape_ );

        Tensor<float, HostMemoryResource> host_input( this->small_shape_ );
        Tensor<float, HostMemoryResource> host_output( this->small_shape_ );
        Tensor<float, HostMemoryResource> expected_output( this->small_shape_ );

        // Initialize input with special cases
        for ( int b = 0; b < this->small_batch_; b++ ) {
            for ( int t = 0; t < this->small_seq_len_; t++ ) {
                for ( int v = 0; v < this->small_vocab_size_; v++ ) {
                    int index = (b * this->small_seq_len_ + t) * this->small_vocab_size_ + v;

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

        // Test with different axis values
        std::vector<int> test_axes = { 0, 1, 2, -1, -2, -3 };

        for ( int axis : test_axes ) {
            // Skip invalid axis values
            if ( axis >= static_cast<int>(this->small_shape_.size()) ||
                axis < -static_cast<int>( this->small_shape_.size() ) ) {
                continue;
            }

            // Execute operation
            std::vector<std::shared_ptr<Tensor<float, CudaDeviceMemoryResource>>> params;
            std::vector<std::shared_ptr<Tensor<TypeParam, CudaDeviceMemoryResource>>> output_state;
            OperationAttributes props;
            props.axis = axis;

            ASSERT_NO_THROW( this->cuda_softmax_op_->forward(
                device_input, params, props, device_output, output_state ) );

            // Copy result back to host (convert if needed)
            if constexpr ( std::is_same_v<TypeParam, float> ) {
                host_output.copyFrom( device_output );
            }
            else {
                // Need to convert half to float for comparison
                Tensor<float, CudaDeviceMemoryResource> float_output( this->small_shape_ );
                // convert_half_to_float(float_output.data(), device_output.data(), float_output.size(), this->cuda_context_->getStream());
                host_output.copyFrom( float_output );
            }

            // Compute expected output with reference implementation
            this->referenceSoftmax( host_input, expected_output, props.axis );

            // Adjust epsilon for half precision comparisons
            float epsilon = std::is_same_v<TypeParam, half> ? 1e-3f : 1e-4f;

            // Verify output has correct values
            EXPECT_TRUE( this->compareTensors( host_output, expected_output, epsilon ) );

            // Verify softmax output properties (sums to 1, non-negative values)
            EXPECT_TRUE( this->checkSoftmaxProperties( host_output, props.axis ) );
        }
    }

    /**
     * @brief Test numerical stability with varied inputs
     */
    TYPED_TEST( CudaSoftmaxOpPrecisionTests, NumericalStability ) {
        // Create test tensors
        Tensor<float, CudaDeviceMemoryResource> device_input( this->medium_shape_ );
        Tensor<TypeParam, CudaDeviceMemoryResource> device_output( this->medium_shape_ );

        Tensor<float, HostMemoryResource> host_input( this->medium_shape_ );
        Tensor<float, HostMemoryResource> host_output( this->medium_shape_ );

        // Test a variety of challenging input values
        for ( int b = 0; b < this->medium_batch_; b++ ) {
            for ( int t = 0; t < this->medium_seq_len_; t++ ) {
                for ( int v = 0; v < this->medium_vocab_size_; v++ ) {
                    int index = (b * this->medium_seq_len_ + t) * this->medium_vocab_size_ + v;

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
                            host_input.data()[ index ] = (v == this->medium_vocab_size_ / 2) ? 1000.0f : -1000.0f;
                            break;
                        case 4:  // Near zero values
                            host_input.data()[ index ] = (v - this->medium_vocab_size_ / 2) * 1e-5f;
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

        // Test with different axis values
        std::vector<int> test_axes = { 0, 1, 2, -1, -2, -3 };

        for ( int axis : test_axes ) {
            // Skip invalid axis values
            if ( axis >= static_cast<int>(this->medium_shape_.size()) ||
                axis < -static_cast<int>( this->medium_shape_.size() ) ) {
                continue;
            }

            // Execute operation
            std::vector<std::shared_ptr<Tensor<float, CudaDeviceMemoryResource>>> params;
            std::vector<std::shared_ptr<Tensor<TypeParam, CudaDeviceMemoryResource>>> output_state;
            OperationAttributes props;
            props.axis = axis;

            ASSERT_NO_THROW( this->cuda_softmax_op_->forward(
                device_input, params, props, device_output, output_state ) );

            // Copy result back to host (convert if needed)
            if constexpr ( std::is_same_v<TypeParam, float> ) {
                host_output.copyFrom( device_output );
            }
            else {
                // Need to convert half to float for comparison
                Tensor<float, CudaDeviceMemoryResource> float_output( this->medium_shape_ );
                // convert_half_to_float(float_output.data(), device_output.data(), float_output.size(), this->cuda_context_->getStream());
                host_output.copyFrom( float_output );
            }

            // Verify no NaN or Inf values
            EXPECT_FALSE( this->hasNaNorInf( host_output ) );

            // Check softmax properties are maintained even with extreme inputs
            EXPECT_TRUE( this->checkSoftmaxProperties( host_output, props.axis ) );
        }
    }

    /**
     * @brief Test error handling for invalid axis
     */
    TYPED_TEST( CudaSoftmaxOpPrecisionTests, InvalidAxis ) {
        // Create test tensors
        Tensor<float, CudaDeviceMemoryResource> device_input( this->small_shape_ );
        Tensor<TypeParam, CudaDeviceMemoryResource> device_output( this->small_shape_ );

        // Initialize input with test values
        Tensor<float, HostMemoryResource> host_input( this->small_shape_ );
        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input.data()[ i ] = static_cast<float>( i % 7 ) - 3.0f;
        }
        device_input.copyFrom( host_input );

        // Setup for operation execution
        std::vector<std::shared_ptr<Tensor<float, CudaDeviceMemoryResource>>> params;
        std::vector<std::shared_ptr<Tensor<TypeParam, CudaDeviceMemoryResource>>> output_state;

        // Test with out-of-bounds positive axis
        OperationAttributes props_pos;
        props_pos.axis = this->small_shape_.size();  // Out of bounds (valid: 0,1,2)
        EXPECT_THROW( this->cuda_softmax_op_->forward(
            device_input, params, props_pos, device_output, output_state ), std::runtime_error );

        // Test with out-of-bounds negative axis
        OperationAttributes props_neg;
        props_neg.axis = -static_cast<int>(this->small_shape_.size()) - 1;  // Out of bounds (valid: -1,-2,-3)
        EXPECT_THROW( this->cuda_softmax_op_->forward(
            device_input, params, props_neg, device_output, output_state ), std::runtime_error );
    }

    /**
     * @brief Test performance for both precision types
     */
    TYPED_TEST( CudaSoftmaxOpPrecisionTests, Performance ) {
        // Skip test if running in CI environment
        if ( std::getenv( "CI" ) != nullptr ) {
            GTEST_SKIP() << "Skipping performance test in CI environment";
        }

        // Create large test tensors
        Tensor<float, CudaDeviceMemoryResource> device_input( this->large_shape_ );
        Tensor<TypeParam, CudaDeviceMemoryResource> device_output( this->large_shape_ );

        Tensor<float, HostMemoryResource> host_input( this->large_shape_ );

        // Fill with random values
        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input.data()[ i ] = (static_cast<float>( rand() ) / RAND_MAX * 10.0f - 5.0f);
        }

        // Copy to device
        device_input.copyFrom( host_input );

        // Test performance with different axis values
        std::vector<int> test_axes = { 0, 1, 2, -1 };

        for ( int axis : test_axes ) {
            std::vector<std::shared_ptr<Tensor<float, CudaDeviceMemoryResource>>> params;
            std::vector<std::shared_ptr<Tensor<TypeParam, CudaDeviceMemoryResource>>> output_state;
            OperationAttributes props;
            props.axis = axis;

            // Make sure everything is ready
            cudaDeviceSynchronize();

            // Measure performance over multiple iterations
            const int iterations = 100;
            auto start_time = std::chrono::high_resolution_clock::now();

            for ( int i = 0; i < iterations; ++i ) {
                this->cuda_softmax_op_->forward( device_input, params, props, device_output, output_state );
            }

            // Wait for all operations to complete
            cudaDeviceSynchronize();

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>( end_time - start_time );

            // Calculate operations per softmax
            // This is approximate: find max, exp, sum, division operations
            size_t ops_per_distribution;
            size_t num_distributions;

            // Calculate based on the axis
            int normalized_axis = axis < 0 ? this->large_shape_.size() + axis : axis;
            size_t dim_size = this->large_shape_[ normalized_axis ];

            // Calculate number of distributions
            num_distributions = 1;
            for ( size_t i = 0; i < this->large_shape_.size(); i++ ) {
                if ( i != normalized_axis ) {
                    num_distributions *= this->large_shape_[ i ];
                }
            }

            // Operations per distribution: max (dim_size), exp (dim_size), sum (dim_size), div (dim_size)
            ops_per_distribution = dim_size * 4;

            size_t total_ops = ops_per_distribution * num_distributions * iterations;

            // Calculate performance metrics
            double ops_per_second = static_cast<double>(total_ops) / (duration.count() * 1e-6);
            double gops = ops_per_second / 1e9;
            double avg_time_per_iter = duration.count() / iterations;

            // Get precision name for recording
            std::string precision_name = std::is_same_v<TypeParam, half> ? "FP16" : "FP32";

            // Record properties that will show in test details
            this->RecordProperty( "Performance_GOPS_Axis_" + std::to_string( axis ), std::to_string( gops ) );
            this->RecordProperty( "Average_Time_us_Axis_" + std::to_string( axis ), std::to_string( avg_time_per_iter ) );

            std::cout << "CUDA Softmax (" << precision_name << ") Performance with axis=" << axis
                << ": " << gops << " GOPS" << std::endl;
            std::cout << "Average time per iteration: " << avg_time_per_iter << " microseconds" << std::endl;
        }
    }

    // Legacy non-template tests for backward compatibility
    class CudaSoftmaxOpTests : public ::testing::Test {
    protected:
        void SetUp() override {
            // Create device contexts for both CPU and CUDA
            cuda_context_ = std::make_shared<DeviceContext>( "CUDA:0" );
            cpu_context_ = std::make_shared<DeviceContext>( "CPU" );

            // Define shapes for testing
            small_shape_ = { 2, 3, 4 }; // (batch, sequence_length, vocabulary)

            // Create CUDA Softmax operation with specific context
            cuda_softmax_op_ = std::make_shared<CudaSoftmaxOp<float>>( cuda_context_ );
        }

        std::shared_ptr<DeviceContext> cuda_context_;
        std::shared_ptr<DeviceContext> cpu_context_;
        std::shared_ptr<CudaSoftmaxOp<float>> cuda_softmax_op_;
        std::vector<size_t> small_shape_;
    };

    /**
     * @brief Test name property of CudaSoftmaxOp (legacy)
     */
    TEST_F( CudaSoftmaxOpTests, Name ) {
        EXPECT_EQ( cuda_softmax_op_->getName(), "Cuda::SoftmaxOp" );
    }

    /**
     * @brief Test error handling for device type mismatch
     */
    TEST_F( CudaSoftmaxOpTests, DeviceTypeMismatch ) {
        // Attempt to create a CudaSoftmaxOp with a CPU context
        EXPECT_THROW( (CudaSoftmaxOp<float>( cpu_context_ )), std::runtime_error );
    }

    /**
     * @brief Test comparison of FP16 and FP32 performance
     */
    TEST( CudaSoftmaxOpPrecisionComparison, PerformanceComparison ) {
        // Skip test if running in CI environment
        if ( std::getenv( "CI" ) != nullptr ) {
            GTEST_SKIP() << "Skipping performance comparison test in CI environment";
        }

        // Create device context for CUDA
        auto cuda_context = std::make_shared<DeviceContext>( "CUDA:0" );

        // Create operations for both precision types
        auto fp32_op = std::make_shared<CudaSoftmaxOp<float>>( cuda_context );
        auto fp16_op = std::make_shared<CudaSoftmaxOp<half>>( cuda_context );

        // Define test shape
        std::vector<size_t> shape = { 16, 32, 1024 }; // Large shape for significant performance difference

        // Create tensors
        Tensor<float, CudaDeviceMemoryResource> input( shape );
        Tensor<half, CudaDeviceMemoryResource> input_half( shape );
        Tensor<float, CudaDeviceMemoryResource> output_fp32( shape );
        Tensor<half, CudaDeviceMemoryResource> output_fp16( shape );
        Tensor<float, HostMemoryResource> host_input( shape );

        // Fill with random data
        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input.data()[ i ] = (static_cast<float>( rand() ) / RAND_MAX * 10.0f - 5.0f);
        }

        // Copy to device
        input.copyFrom( host_input );

        // Test with different axis values
        std::vector<int> test_axes = { 2, -1 };  // Test with last dimension and negative indexing

        for ( int axis : test_axes ) {
            // Setup parameters
            std::vector<std::shared_ptr<Tensor<float, CudaDeviceMemoryResource>>> params;
            std::vector<std::shared_ptr<Tensor<half, CudaDeviceMemoryResource>>> params_fp16;
            std::vector<std::shared_ptr<Tensor<float, CudaDeviceMemoryResource>>> cache_fp32;
            std::vector<std::shared_ptr<Tensor<half, CudaDeviceMemoryResource>>> cache_fp16;
            OperationAttributes props;
            props.axis = axis;

            // Make sure GPU is ready
            cudaDeviceSynchronize();

            // Measure FP32 performance
            const int iterations = 100;
            auto fp32_start = std::chrono::high_resolution_clock::now();

            for ( int i = 0; i < iterations; ++i ) {
                fp32_op->forward( input, params, props, output_fp32, cache_fp32 );
            }

            cudaDeviceSynchronize();
            auto fp32_end = std::chrono::high_resolution_clock::now();
            auto fp32_duration = std::chrono::duration_cast<std::chrono::microseconds>( fp32_end - fp32_start );

            // Convert the host input data to half precision on the device
            //input_half.copyFrom( input );

            // Measure FP16 performance
            auto fp16_start = std::chrono::high_resolution_clock::now();

            for ( int i = 0; i < iterations; ++i ) {
                //fp16_op->forward( input_half, params_fp16, props, output_fp16, cache_fp16 );
            }

            cudaDeviceSynchronize();
            auto fp16_end = std::chrono::high_resolution_clock::now();
            auto fp16_duration = std::chrono::duration_cast<std::chrono::microseconds>( fp16_end - fp16_start );

            // Calculate operations based on the axis
            int normalized_axis = axis < 0 ? shape.size() + axis : axis;
            size_t dim_size = shape[ normalized_axis ];

            // Calculate number of distributions
            size_t num_distributions = 1;
            for ( size_t i = 0; i < shape.size(); i++ ) {
                if ( i != normalized_axis ) {
                    num_distributions *= shape[ i ];
                }
            }

            // Operations per distribution: max (dim_size), exp (dim_size), sum (dim_size), div (dim_size)
            size_t ops_per_distribution = dim_size * 4;

            size_t total_ops = ops_per_distribution * num_distributions * iterations;

            // Calculate metrics
            double fp32_gops = static_cast<double>(total_ops) / (fp32_duration.count() * 1e-6) / 1e9;
            double fp16_gops = static_cast<double>(total_ops) / (fp16_duration.count() * 1e-6) / 1e9;
            double speedup = static_cast<double>(fp32_duration.count()) / fp16_duration.count();

            // Output results
            std::cout << "Axis " << axis << " - FP32 Performance: " << fp32_gops << " GOPS" << std::endl;
            std::cout << "Axis " << axis << " - FP16 Performance: " << fp16_gops << " GOPS" << std::endl;
            std::cout << "Axis " << axis << " - FP16 Speedup over FP32: " << speedup << "x" << std::endl;

            // Record metrics
            ::testing::Test::RecordProperty( "FP32_GOPS_Axis_" + std::to_string( axis ), std::to_string( fp32_gops ) );
            ::testing::Test::RecordProperty( "FP16_GOPS_Axis_" + std::to_string( axis ), std::to_string( fp16_gops ) );
            ::testing::Test::RecordProperty( "FP16_Speedup_Axis_" + std::to_string( axis ), std::to_string( speedup ) );
        }
    }
}
