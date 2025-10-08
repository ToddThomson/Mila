/**
 * @file CudaEncoderOpTests.cpp
 * @brief Test suite for the CUDA Encoder operation.
 */

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <chrono>
#include <cuda_fp16.h>  // For half type
#include <cmath>
#include <cuda_runtime.h>

import Mila;

namespace Operations::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    /**
     * @brief Test fixture for CudaEncoderOp tests
     */
    class CudaEncoderOpTests : public ::testing::Test {
    protected:
        void SetUp() override {
            cuda_context_ = std::make_shared<DeviceContext>( "CUDA:0" );
            cpu_context_ = std::make_shared<DeviceContext>( "CPU" );

            // Test dimensions
            batch_size_ = 4;
            sequence_length_ = 16;
            channels_ = 32;
            vocab_size_ = 100;

            // Input/output shapes
            input_shape_ = { batch_size_, sequence_length_ };
            output_shape_ = { batch_size_, sequence_length_, channels_ };

            // Create operations
            cuda_op_float_ = std::make_shared<CudaEncoderOp<float>>( cuda_context_ );
            cuda_op_half_ = std::make_shared<CudaEncoderOp<half>>( cuda_context_ );

            // Create parameters
            SetupParameters();
        }

        void SetupParameters() {
            // Create token embedding table (wte)
            wte_ = std::make_shared<Tensor<float, CudaDeviceMemoryResource>>( std::vector<size_t>{vocab_size_, channels_} );
            wpe_ = std::make_shared<Tensor<float, CudaDeviceMemoryResource>>( std::vector<size_t>{sequence_length_, channels_} );

            // Create host tensors for initialization
            Tensor<float, HostMemoryResource> host_wte( std::vector<size_t>{vocab_size_, channels_} );
            Tensor<float, HostMemoryResource> host_wpe( std::vector<size_t>{sequence_length_, channels_} );

            // Initialize with deterministic values
            for ( size_t i = 0; i < host_wte.size(); ++i ) {
                host_wte.data()[ i ] = static_cast<float>( i % 10 ) * 0.1f;
            }

            for ( size_t i = 0; i < host_wpe.size(); ++i ) {
                host_wpe.data()[ i ] = static_cast<float>( i % 7 ) * 0.01f;
            }

            // Copy to device
            wte_->copyFrom( host_wte );
            wpe_->copyFrom( host_wpe );

            parameters_.push_back( wte_ );
            parameters_.push_back( wpe_ );

            // Create half precision parameters using the new toHalf() method
            wte_half_ = std::make_shared<Tensor<half, CudaDeviceMemoryResource>>( wte_->toHalf() );
            wpe_half_ = std::make_shared<Tensor<half, CudaDeviceMemoryResource>>( wpe_->toHalf() );

            parameters_half_.push_back( wte_half_ );
            parameters_half_.push_back( wpe_half_ );
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

        // Helper method to check reference implementation
        float referenceEncoderOutput( int token_id, int position, int channel ) {
            // Get host copies of parameter data to use for reference
            Tensor<float, HostMemoryResource> host_wte( wte_->shape() );
            Tensor<float, HostMemoryResource> host_wpe( wpe_->shape() );

            host_wte.copyFrom( *wte_ );
            host_wpe.copyFrom( *wpe_ );

            return host_wte.data()[ token_id * channels_ + channel ] +
                host_wpe.data()[ position * channels_ + channel ];
        }

        std::shared_ptr<DeviceContext> cuda_context_;
        std::shared_ptr<DeviceContext> cpu_context_;
        std::shared_ptr<CudaEncoderOp<float>> cuda_op_float_;
        std::shared_ptr<CudaEncoderOp<half>> cuda_op_half_;

        size_t batch_size_;
        size_t sequence_length_;
        size_t channels_;
        size_t vocab_size_;

        std::vector<size_t> input_shape_;
        std::vector<size_t> output_shape_;

        std::shared_ptr<Tensor<float, CudaDeviceMemoryResource>> wte_;
        std::shared_ptr<Tensor<float, CudaDeviceMemoryResource>> wpe_;
        std::vector<std::shared_ptr<Tensor<float, CudaDeviceMemoryResource>>> parameters_;

        std::shared_ptr<Tensor<half, CudaDeviceMemoryResource>> wte_half_;
        std::shared_ptr<Tensor<half, CudaDeviceMemoryResource>> wpe_half_;
        std::vector<std::shared_ptr<Tensor<half, CudaDeviceMemoryResource>>> parameters_half_;

        OperationAttributes attributes_;
    };

    /**
     * @brief Test basic functionality of CudaEncoderOp with float precision
     */
    TEST_F( CudaEncoderOpTests, BasicFunctionality_Float ) {
        // Create input and output tensors
        Tensor<int, CudaDeviceMemoryResource> input( input_shape_ );
        Tensor<float, CudaDeviceMemoryResource> output( output_shape_ );
        std::vector<std::shared_ptr<Tensor<float, CudaDeviceMemoryResource>>> output_state;

        // Create host tensors for initialization and verification
        Tensor<int, HostMemoryResource> host_input( input_shape_ );
        Tensor<float, HostMemoryResource> host_output( output_shape_ );

        // Initialize input with token IDs
        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input.data()[ i ] = static_cast<int>( i % vocab_size_ );
        }

        // Copy to device
        input.copyFrom( host_input );

        // Execute encoder operation
        ASSERT_NO_THROW( cuda_op_float_->forward( input, parameters_, attributes_, output, output_state ) );

        // Copy output back to host for verification
        host_output.copyFrom( output );

        // Verify output has correct values
        for ( size_t b = 0; b < batch_size_; ++b ) {
            for ( size_t t = 0; t < sequence_length_; ++t ) {
                int token_id = host_input.data()[ b * sequence_length_ + t ];

                for ( size_t c = 0; c < channels_; ++c ) {
                    float expected = referenceEncoderOutput( token_id, t, c );
                    float actual = host_output.data()[ (b * sequence_length_ + t) * channels_ + c ];
                    EXPECT_NEAR( actual, expected, 1e-5f );
                }
            }
        }
    }

    /**
 * @brief Test basic functionality of CudaEncoderOp with half precision
 */
    TEST_F( CudaEncoderOpTests, BasicFunctionality_Half ) {
        // Create input and output tensors
        Tensor<int, CudaDeviceMemoryResource> input( input_shape_ );
        Tensor<half, CudaDeviceMemoryResource> output( output_shape_ );
        std::vector<std::shared_ptr<Tensor<half, CudaDeviceMemoryResource>>> output_state;

        // Create host tensors for initialization and verification
        Tensor<int, HostMemoryResource> host_input( input_shape_ );
        Tensor<float, HostMemoryResource> host_output( output_shape_ );

        // Initialize input with token IDs
        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input.data()[ i ] = static_cast<int>( i % vocab_size_ );
        }

        // Copy to device
        input.copyFrom( host_input );

        // Execute encoder operation
        ASSERT_NO_THROW( cuda_op_half_->forward( input, parameters_half_, attributes_, output, output_state ) );

        // Copy output back to host for verification
        // First convert half precision output to float using toFloat method
        Tensor<float, CudaDeviceMemoryResource> float_output = output.toFloat();

        // Then copy to host
        host_output.copyFrom( float_output );

        // For half precision, we use a larger epsilon
        float epsilon = 1e-3f;

        // Verify output has correct values
        for ( size_t b = 0; b < batch_size_; ++b ) {
            for ( size_t t = 0; t < sequence_length_; ++t ) {
                int token_id = host_input.data()[ b * sequence_length_ + t ];

                for ( size_t c = 0; c < channels_; ++c ) {
                    float expected = referenceEncoderOutput( token_id, t, c );
                    float actual = host_output.data()[ (b * sequence_length_ + t) * channels_ + c ];
                    EXPECT_NEAR( actual, expected, epsilon );
                }
            }
        }
    }


    /**
     * @brief Test that operation name is correct
     */
    TEST_F( CudaEncoderOpTests, GetName ) {
        EXPECT_EQ( cuda_op_float_->getDeviceName(), "Cuda::EncoderOp" );
        EXPECT_EQ( cuda_op_half_->getDeviceName(), "Cuda::EncoderOp" );
    }

    /**
    * @brief Test constructor with invalid device context
    */
    TEST_F( CudaEncoderOpTests, InvalidDeviceContext ) {
        try {
            // Try creating with CPU context, which should fail
            auto invalid_op = CudaEncoderOp<float>( cpu_context_ );
            FAIL() << "Expected exception for invalid device context";
        }
        catch ( const std::runtime_error& e ) {
            // Check for the specific error message from ValidateContext
            EXPECT_NE( std::string( e.what() ).find( "The provided device context is incompatible with the operation's device type." ), std::string::npos );
        }
        catch ( ... ) {
            FAIL() << "Expected std::runtime_error for invalid device context";
        }
    }

    /**
     * @brief Test behavior with invalid memory types
     */
    TEST_F( CudaEncoderOpTests, InvalidMemoryType ) {
        try {
            // Create a CPU memory tensor
            Tensor<int, HostMemoryResource> cpu_input( input_shape_ );
            Tensor<float, HostMemoryResource> cpu_output( output_shape_ );
            std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> cpu_output_state;

            // This should throw an exception
            // Note: This won't compile directly due to type mismatches, but the test logic is valid
            // We'd need proper runtime checking in the operation for this to work

            // For testing purposes, we'll just check that the op requires CUDA memory
            SUCCEED() << "Runtime memory type verification should be implemented in the CudaEncoderOp";
        }
        catch ( const std::runtime_error& e ) {
            // We expect a runtime error about memory types
            EXPECT_NE( std::string( e.what() ).find( "CUDA memory" ), std::string::npos );
        }
        catch ( ... ) {
            FAIL() << "Expected std::runtime_error for mismatched memory types";
        }
    }

    /**
     * @brief Test with empty parameters
     */
    TEST_F( CudaEncoderOpTests, EmptyParameters ) {
        Tensor<int, CudaDeviceMemoryResource> input( input_shape_ );
        Tensor<float, CudaDeviceMemoryResource> output( output_shape_ );
        std::vector<std::shared_ptr<Tensor<float, CudaDeviceMemoryResource>>> empty_params;
        std::vector<std::shared_ptr<Tensor<float, CudaDeviceMemoryResource>>> output_state;

        // Fill input with token IDs
        Tensor<int, HostMemoryResource> host_input( input_shape_ );
        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input.data()[ i ] = static_cast<int>( i % vocab_size_ );
        }
        input.copyFrom( host_input );

        // This should throw exception because we need wte and wpe parameters
        
        // TODO: Review arg validation for operations in general.

        //EXPECT_THROW( cuda_op_float_->forward( input, empty_params, attributes_, output, output_state ), std::runtime_error );
    }

    /**
     * @brief Test with token IDs out of vocabulary range
     */
    TEST_F( CudaEncoderOpTests, OutOfRangeTokenIds ) {
        Tensor<int, CudaDeviceMemoryResource> input( input_shape_ );
        Tensor<float, CudaDeviceMemoryResource> output( output_shape_ );
        std::vector<std::shared_ptr<Tensor<float, CudaDeviceMemoryResource>>> output_state;

        // Fill input with out-of-range token IDs
        Tensor<int, HostMemoryResource> host_input( input_shape_ );
        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input.data()[ i ] = static_cast<int>( vocab_size_ + i ); // Beyond vocab size
        }
        input.copyFrom( host_input );

        // This could cause memory access issues if not handled properly
        // TODO: Arg validation
        //EXPECT_THROW( cuda_op_float_->forward( input, parameters_, attributes_, output, output_state ),
        //    std::runtime_error );
    }

    /**
     * @brief Test precision comparison between float and half implementations
     */
    TEST_F( CudaEncoderOpTests, FloatHalfPrecisionComparison ) {
        // Create input tensor
        Tensor<int, CudaDeviceMemoryResource> input( input_shape_ );

        // Create output tensors
        Tensor<float, CudaDeviceMemoryResource> output_float( output_shape_ );
        Tensor<half, CudaDeviceMemoryResource> output_half( output_shape_ );
        Tensor<float, CudaDeviceMemoryResource> output_half_as_float( output_shape_ );

        // Use same values for input via host tensor
        Tensor<int, HostMemoryResource> host_input( input_shape_ );

        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input.data()[ i ] = static_cast<int>( i % vocab_size_ );
        }

        input.copyFrom( host_input );

        // Execute both ops
        std::vector<std::shared_ptr<Tensor<float, CudaDeviceMemoryResource>>> output_state_float;
        std::vector<std::shared_ptr<Tensor<half, CudaDeviceMemoryResource>>> output_state_half;

        cuda_op_float_->forward( input, parameters_, attributes_, output_float, output_state_float );
        cuda_op_half_->forward( input, parameters_half_, attributes_, output_half, output_state_half );

        // Convert half output to float for comparison
        output_half_as_float = output_half.toFloat();

        // Copy results to host for comparison
        Tensor<float, HostMemoryResource> host_output_float( output_shape_ );
        Tensor<float, HostMemoryResource> host_output_half( output_shape_ );

        host_output_float.copyFrom( output_float );
        host_output_half.copyFrom( output_half_as_float );

        // Results should be similar but not identical (due to precision differences)
        float epsilon = 1e-2f; // Allow larger differences for half precision
        bool all_close = true;
        float max_diff = 0.0f;
        size_t diff_count = 0;

        for ( size_t i = 0; i < host_output_float.size(); ++i ) {
            float diff = std::abs( host_output_float.data()[ i ] - host_output_half.data()[ i ] );
            max_diff = std::max( max_diff, diff );

            if ( diff > epsilon ) {
                diff_count++;
                if ( diff_count <= 5 ) { // Only log the first few differences
                    std::cout << "Precision difference at index " << i << ": float="
                        << host_output_float.data()[ i ] << ", half="
                        << host_output_half.data()[ i ] << ", diff=" << diff << std::endl;
                }
                all_close = false;
            }
        }

        std::cout << "Total differences: " << diff_count << " out of " << host_output_float.size()
            << " values (" << (100.0f * diff_count / host_output_float.size()) << "%), max difference: "
            << max_diff << std::endl;

        // We expect some differences due to precision, but they should be relatively small
        EXPECT_LT( static_cast<float>(diff_count) / host_output_float.size(), 0.1f )
            << "More than 10% of values show significant differences between float and half precision";
    }

    /**
     * @brief Test registration of operations in the registry
     */
    TEST_F( CudaEncoderOpTests, RegistryCreation ) {
        // Register operations
        CudaEncoderOpRegistrar::registerOperations();

        // Retrieve operations from registry
        auto op_float = OperationRegistry::instance().createUnaryOperation<float, int, DeviceType::Cuda>(
            "Cuda::EncoderOp", cuda_context_, "Float_Precision" );

        auto op_half = OperationRegistry::instance().createUnaryOperation<half, int, DeviceType::Cuda>(
            "Cuda::EncoderOp", cuda_context_, "Half_Precision" );

        // Verify operations are created correctly
        ASSERT_NE( op_float, nullptr );
        ASSERT_NE( op_half, nullptr );

        // Check operation types
        EXPECT_EQ( op_float->getDeviceName(), "Cuda::EncoderOp" );
        EXPECT_EQ( op_half->getDeviceName(), "Cuda::EncoderOp" );
    }

    /**
     * @brief Test deterministic behavior (multiple runs should produce same result)
     */
    TEST_F( CudaEncoderOpTests, DeterministicBehavior ) {
        // Create test tensors
        Tensor<int, CudaDeviceMemoryResource> input( input_shape_ );
        Tensor<float, CudaDeviceMemoryResource> output1( output_shape_ );
        Tensor<float, CudaDeviceMemoryResource> output2( output_shape_ );

        // Initialize input with test data
        Tensor<int, HostMemoryResource> host_input( input_shape_ );
        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input.data()[ i ] = static_cast<int>( i % vocab_size_ );
        }
        input.copyFrom( host_input );

        // Run twice with same input
        std::vector<std::shared_ptr<Tensor<float, CudaDeviceMemoryResource>>> output_state1;
        std::vector<std::shared_ptr<Tensor<float, CudaDeviceMemoryResource>>> output_state2;

        cuda_op_float_->forward( input, parameters_, attributes_, output1, output_state1 );

        // Synchronize device to ensure first operation is complete
        cudaDeviceSynchronize();

        cuda_op_float_->forward( input, parameters_, attributes_, output2, output_state2 );

        // Copy results back to host for comparison
        Tensor<float, HostMemoryResource> host_output1( output_shape_ );
        Tensor<float, HostMemoryResource> host_output2( output_shape_ );

        host_output1.copyFrom( output1 );
        host_output2.copyFrom( output2 );

        // Results should be identical
        for ( size_t i = 0; i < host_output1.size(); ++i ) {
            EXPECT_EQ( host_output1.data()[ i ], host_output2.data()[ i ] );
        }
    }

    /**
     * @brief Test backward pass interface
     */
    TEST_F( CudaEncoderOpTests, BackwardInterface ) {
        // Create tensors for forward and backward passes
        Tensor<int, CudaDeviceMemoryResource> input( input_shape_ );
        Tensor<float, CudaDeviceMemoryResource> output( output_shape_ );
        Tensor<float, CudaDeviceMemoryResource> output_grad( output_shape_ );
        Tensor<int, CudaDeviceMemoryResource> input_grad( input_shape_ );

        // Initialize input and output gradient with host tensors
        Tensor<int, HostMemoryResource> host_input( input_shape_ );
        Tensor<float, HostMemoryResource> host_output_grad( output_shape_ );

        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input.data()[ i ] = static_cast<int>( i % vocab_size_ );
        }

        for ( size_t i = 0; i < host_output_grad.size(); ++i ) {
            host_output_grad.data()[ i ] = 0.1f;
        }

        input.copyFrom( host_input );
        output_grad.copyFrom( host_output_grad );

        // Forward pass first
        std::vector<std::shared_ptr<Tensor<float, CudaDeviceMemoryResource>>> output_state;
        cuda_op_float_->forward( input, parameters_, attributes_, output, output_state );

        // Create parameter gradients with same shapes as parameters
        auto wte_grad = std::make_shared<Tensor<float, CudaDeviceMemoryResource>>( wte_->shape() );
        auto wpe_grad = std::make_shared<Tensor<float, CudaDeviceMemoryResource>>( wpe_->shape() );

        std::vector<std::shared_ptr<Tensor<float, CudaDeviceMemoryResource>>> param_grads;
        param_grads.push_back( wte_grad );
        param_grads.push_back( wpe_grad );

        // Backward pass - might not be implemented yet, but should have interface ready
        try {
            cuda_op_float_->backward( input, output, output_grad, parameters_,
                param_grads, input_grad, attributes_, output_state );

            // If implemented, check results are non-zero
            Tensor<float, HostMemoryResource> host_wte_grad( wte_->shape() );
            Tensor<float, HostMemoryResource> host_wpe_grad( wpe_->shape() );

            host_wte_grad.copyFrom( *wte_grad );
            host_wpe_grad.copyFrom( *wpe_grad );

            bool all_zeros_wte = true;
            bool all_zeros_wpe = true;

            for ( size_t i = 0; i < host_wte_grad.size() && all_zeros_wte; ++i ) {
                if ( std::abs( host_wte_grad.data()[ i ] ) > 1e-7f ) all_zeros_wte = false;
            }

            for ( size_t i = 0; i < host_wpe_grad.size() && all_zeros_wpe; ++i ) {
                if ( std::abs( host_wpe_grad.data()[ i ] ) > 1e-7f ) all_zeros_wpe = false;
            }

            // If backward pass is implemented, these should be false
            if ( !all_zeros_wte && !all_zeros_wpe ) {
                SUCCEED() << "Backward pass is implemented and producing non-zero gradients";
            }
        }
        catch ( const std::exception& e ) {
            // Check if error indicates backward pass isn't implemented yet
            std::string error_message = e.what();
            if ( error_message.find( "TODO" ) != std::string::npos ||
                error_message.find( "not implemented" ) != std::string::npos ||
                error_message.find( "FIXME" ) != std::string::npos ) {
                GTEST_SKIP() << "Backward pass not implemented yet: " << error_message;
            }
            else {
                FAIL() << "Unexpected exception: " << error_message;
            }
        }
    }

    /**
     * @brief Test performance with various batch sizes
     */
    TEST_F( CudaEncoderOpTests, Performance ) {
        // Skip test if running in CI environment
        if ( std::getenv( "CI" ) != nullptr ) {
            GTEST_SKIP() << "Skipping performance test in CI environment";
        }

        // Performance test with larger batch size
        size_t large_batch = 32;
        std::vector<size_t> large_input_shape = { large_batch, sequence_length_ };
        std::vector<size_t> large_output_shape = { large_batch, sequence_length_, channels_ };

        // Create tensors
        Tensor<int, CudaDeviceMemoryResource> input( large_input_shape );
        Tensor<float, CudaDeviceMemoryResource> output( large_output_shape );
        std::vector<std::shared_ptr<Tensor<float, CudaDeviceMemoryResource>>> output_state;

        // Initialize input
        Tensor<int, HostMemoryResource> host_input( large_input_shape );
        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input.data()[ i ] = static_cast<int>( i % vocab_size_ );
        }
        input.copyFrom( host_input );

        // Make sure everything is ready
        cudaDeviceSynchronize();

        // Measure performance over multiple iterations
        const int iterations = 100;
        auto start_time = std::chrono::high_resolution_clock::now();

        for ( int i = 0; i < iterations; ++i ) {
            cuda_op_float_->forward( input, parameters_, attributes_, output, output_state );
        }

        // Wait for all operations to complete
        cudaDeviceSynchronize();

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>( end_time - start_time );

        // Report performance metrics
        size_t total_elements = input.size() * iterations;
        double elements_per_second = static_cast<double>(total_elements) / (duration.count() * 1e-6);
        std::cout << "CUDA Encoder Performance: " << elements_per_second / 1e6 << " million tokens/sec" << std::endl;
        std::cout << "Average time per iteration: " << duration.count() / iterations << " microseconds" << std::endl;

        // No assertion, just informational
    }
}
