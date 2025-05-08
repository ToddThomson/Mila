/**
 * @file CpuEncoderOpTests.cpp
 * @brief Test suite for the CPU Encoder operation.
 */

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <cmath>

import Mila;

namespace Operations::Cpu::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    /**
     * @brief Test fixture for CpuEncoderOp tests
     */
    class CpuEncoderOpTests : public ::testing::Test {
    protected:
        void SetUp() override {
            cpu_context_ = std::make_shared<DeviceContext>( "CPU" );

            // Test dimensions
            batch_size_ = 4;
            sequence_length_ = 16;
            channels_ = 32;
            vocab_size_ = 100;

            // Input/output shapes
            input_shape_ = { batch_size_, sequence_length_ };
            output_shape_ = { batch_size_, sequence_length_, channels_ };

            // Create operation
            cpu_op_ = std::make_shared<CpuEncoderOp>( cpu_context_ );

            // Create parameters
            SetupParameters();
        }

        void SetupParameters() {
            // Create token embedding table (wte)
            wte_ = std::make_shared<Tensor<float, HostMemoryResource>>( std::vector<size_t>{vocab_size_, channels_} );
            wpe_ = std::make_shared<Tensor<float, HostMemoryResource>>( std::vector<size_t>{sequence_length_, channels_} );

            // Initialize with deterministic values
            for ( size_t i = 0; i < wte_->size(); ++i ) {
                wte_->data()[ i ] = static_cast<float>( i % 10 ) * 0.1f;
            }

            for ( size_t i = 0; i < wpe_->size(); ++i ) {
                wpe_->data()[ i ] = static_cast<float>( i % 7 ) * 0.01f;
            }

            parameters_.push_back( wte_ );
            parameters_.push_back( wpe_ );
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
            return wte_->data()[ token_id * channels_ + channel ] +
                wpe_->data()[ position * channels_ + channel ];
        }

        std::shared_ptr<DeviceContext> cpu_context_;
        std::shared_ptr<CpuEncoderOp> cpu_op_;

        size_t batch_size_;
        size_t sequence_length_;
        size_t channels_;
        size_t vocab_size_;

        std::vector<size_t> input_shape_;
        std::vector<size_t> output_shape_;

        std::shared_ptr<Tensor<float, HostMemoryResource>> wte_;
        std::shared_ptr<Tensor<float, HostMemoryResource>> wpe_;
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> parameters_;

        OperationAttributes attributes_;
    };

    /**
     * @brief Test basic functionality of CpuEncoderOp with int inputs
     */
    TEST_F( CpuEncoderOpTests, BasicFunctionality_Int ) {
        // Create input and output tensors
        Tensor<int, HostMemoryResource> input( input_shape_ );
        Tensor<float, HostMemoryResource> output( output_shape_ );
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> output_state;

        // Initialize input with token IDs
        for ( size_t i = 0; i < input.size(); ++i ) {
            input.data()[ i ] = static_cast<int>( i % vocab_size_ );
        }

        // Execute encoder operation
        ASSERT_NO_THROW( cpu_op_->forward( input, parameters_, attributes_, output, output_state ) );

        // Verify output has correct values
        for ( size_t b = 0; b < batch_size_; ++b ) {
            for ( size_t t = 0; t < sequence_length_; ++t ) {
                int token_id = input.data()[ b * sequence_length_ + t ];

                for ( size_t c = 0; c < channels_; ++c ) {
                    float expected = referenceEncoderOutput( token_id, t, c );
                    float actual = output.data()[ (b * sequence_length_ + t) * channels_ + c ];
                    EXPECT_NEAR( actual, expected, 1e-5f );
                }
            }
        }
    }

    /**
     * @brief Test that operation name is correct
     */
    TEST_F( CpuEncoderOpTests, GetName ) {
        EXPECT_EQ( cpu_op_->getName(), "Cpu::EncoderOp" );
    }

    /**
    * @brief Test constructor with invalid device context
    */
    TEST_F( CpuEncoderOpTests, InvalidDeviceContext ) {
        try {
            // Try creating with CUDA context, which should fail
            auto cuda_context = std::make_shared<DeviceContext>( "CUDA:0" );
            auto invalid_op = CpuEncoderOp( cuda_context );
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
     * @brief Test with empty parameters
     */
    TEST_F( CpuEncoderOpTests, EmptyParameters ) {
        Tensor<int, HostMemoryResource> input( input_shape_ );
        Tensor<float, HostMemoryResource> output( output_shape_ );
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> empty_params;
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> output_state;

        // Fill input with token IDs
        for ( size_t i = 0; i < input.size(); ++i ) {
            input.data()[ i ] = static_cast<int>( i % vocab_size_ );
        }

        // This should throw exception because we need wte and wpe parameters
        // TODO: Arg validation
        // EXPECT_THROW( cpu_op_->forward( input, empty_params, attributes_, output, output_state ), std::runtime_error );
    }

    /**
     * @brief Test with token IDs out of vocabulary range
     */
    TEST_F( CpuEncoderOpTests, OutOfRangeTokenIds ) {
        Tensor<int, HostMemoryResource> input( input_shape_ );
        Tensor<float, HostMemoryResource> output( output_shape_ );
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> output_state;

        // Fill input with out-of-range token IDs
        for ( size_t i = 0; i < input.size(); ++i ) {
            input.data()[ i ] = static_cast<int>( vocab_size_ + i ); // Beyond vocab size
        }

        // This could cause memory access issues if not handled properly
        // TODO: Arg validation
        // EXPECT_THROW( cpu_op_->forward( input, parameters_, attributes_, output, output_state ),
        //    std::runtime_error );
    }

    /**
     * @brief Test registration of operations in the registry
     */
    TEST_F( CpuEncoderOpTests, RegistryCreation ) {
        // Register operations
        CpuEncoderOpRegistrar::registerOperations();

        // Retrieve operation from registry
        auto op_int = OperationRegistry::instance().createUnaryOperation<float, int, DeviceType::Cpu>(
            "Cpu::EncoderOp", cpu_context_, "float" );

        // Verify operation is created correctly
        ASSERT_NE( op_int, nullptr );

        // Check operation type
        EXPECT_EQ( op_int->getName(), "Cpu::EncoderOp" );
    }

    /**
     * @brief Test deterministic behavior (multiple runs should produce same result)
     */
    TEST_F( CpuEncoderOpTests, DeterministicBehavior ) {
        // Create test tensors
        Tensor<int, HostMemoryResource> input( input_shape_ );
        Tensor<float, HostMemoryResource> output1( output_shape_ );
        Tensor<float, HostMemoryResource> output2( output_shape_ );

        // Initialize with consistent values
        for ( size_t i = 0; i < input.size(); ++i ) {
            input.data()[ i ] = static_cast<int>( i % vocab_size_ );
        }

        // Run twice with same input
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> output_state1;
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> output_state2;

        cpu_op_->forward( input, parameters_, attributes_, output1, output_state1 );
        cpu_op_->forward( input, parameters_, attributes_, output2, output_state2 );

        // Results should be identical
        for ( size_t i = 0; i < output1.size(); ++i ) {
            EXPECT_EQ( output1.data()[ i ], output2.data()[ i ] );
        }
    }

    /**
     * @brief Test backward pass interface
     */
    TEST_F( CpuEncoderOpTests, BackwardInterface ) {
        // Create tensors for forward and backward passes
        Tensor<int, HostMemoryResource> input( input_shape_ );
        Tensor<float, HostMemoryResource> output( output_shape_ );
        Tensor<float, HostMemoryResource> output_grad( output_shape_ );
        Tensor<int, HostMemoryResource> input_grad( input_shape_ );

        // Initialize input and output gradient
        for ( size_t i = 0; i < input.size(); ++i ) {
            input.data()[ i ] = static_cast<int>( i % vocab_size_ );
        }

        for ( size_t i = 0; i < output_grad.size(); ++i ) {
            output_grad.data()[ i ] = 0.1f;
        }

        // Forward pass first
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> output_state;
        cpu_op_->forward( input, parameters_, attributes_, output, output_state );

        // Create parameter gradients with same shapes as parameters
        auto wte_grad = std::make_shared<Tensor<float, HostMemoryResource>>( wte_->shape() );
        auto wpe_grad = std::make_shared<Tensor<float, HostMemoryResource>>( wpe_->shape() );

        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> param_grads;
        param_grads.push_back( wte_grad );
        param_grads.push_back( wpe_grad );

        // Backward pass - might not be implemented yet, but should have interface ready
        try {
            cpu_op_->backward( input, output, output_grad, parameters_,
                param_grads, input_grad, attributes_, output_state );

            // If implemented, check results are non-zero
            bool all_zeros_wte = true;
            bool all_zeros_wpe = true;

            for ( size_t i = 0; i < wte_grad->size() && all_zeros_wte; ++i ) {
                if ( std::abs( wte_grad->data()[ i ] ) > 1e-7f ) all_zeros_wte = false;
            }

            for ( size_t i = 0; i < wpe_grad->size() && all_zeros_wpe; ++i ) {
                if ( std::abs( wpe_grad->data()[ i ] ) > 1e-7f ) all_zeros_wpe = false;
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
                error_message.find( "not implemented" ) != std::string::npos ) {
                GTEST_SKIP() << "Backward pass not implemented yet: " << error_message;
            }
            else {
                FAIL() << "Unexpected exception: " << error_message;
            }
        }
    }
}
