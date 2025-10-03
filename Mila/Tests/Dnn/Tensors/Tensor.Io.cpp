#include <gtest/gtest.h>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

import Mila;

namespace Dnn::Tensors::Tests
{
    using namespace Mila::Dnn;

    class TensorIoTest : public ::testing::Test {
    protected:
        void SetUp() override {
            // Test tensors with different shapes and properties using host-compatible types
            scalar_tensor_ = std::make_unique<HostTensor<TensorDataType::FP32>>( "CPU", std::vector<size_t>{} );
            vector_tensor_ = std::make_unique<HostTensor<TensorDataType::FP32>>( "CPU", std::vector<size_t>{5} );
            matrix_tensor_ = std::make_unique<HostTensor<TensorDataType::INT32>>( "CPU", std::vector<size_t>{3, 4} );
            tensor_3d_ = std::make_unique<HostTensor<TensorDataType::INT8>>( "CPU", std::vector<size_t>{2, 3, 4} );
            tensor_4d_ = std::make_unique<HostTensor<TensorDataType::UINT8>>( "CPU", std::vector<size_t>{2, 3, 4, 5} );

            // Set descriptive names for testing
            vector_tensor_->setName( "test_vector" );
            matrix_tensor_->setName( "test_matrix" );
            tensor_3d_->setName( "conv_weights" );
            tensor_4d_->setName( "batch_data" );
        }

        std::unique_ptr<HostTensor<TensorDataType::FP32>> scalar_tensor_;
        std::unique_ptr<HostTensor<TensorDataType::FP32>> vector_tensor_;
        std::unique_ptr<HostTensor<TensorDataType::INT32>> matrix_tensor_;
        std::unique_ptr<HostTensor<TensorDataType::INT8>> tensor_3d_;
        std::unique_ptr<HostTensor<TensorDataType::UINT8>> tensor_4d_;
    };

    // =============================================================================
    // toString() Method Tests
    // =============================================================================

    TEST_F( TensorIoTest, ToStringBasicFormatting ) {
        std::string result = scalar_tensor_->toString();

        // Verify basic structure elements are present
        EXPECT_TRUE( result.find( "Tensor: tensor_" ) != std::string::npos );
        EXPECT_TRUE( result.find( "TensorData(shape=[]" ) != std::string::npos );
        EXPECT_TRUE( result.find( "strides=[]" ) != std::string::npos );
        EXPECT_TRUE( result.find( "format=RowMajor" ) != std::string::npos );
        EXPECT_TRUE( result.find( "size=1" ) != std::string::npos );
        EXPECT_TRUE( result.find( "Type: FP32" ) != std::string::npos );
        EXPECT_TRUE( result.find( "Device: CPU" ) != std::string::npos );
    }

    TEST_F( TensorIoTest, ToStringWithName ) {
        std::string result = vector_tensor_->toString();

        // Should include tensor name in the output
        EXPECT_TRUE( result.find( "::test_vector" ) != std::string::npos );
        EXPECT_TRUE( result.find( "Type: FP32" ) != std::string::npos );
    }

    TEST_F( TensorIoTest, ToStringWithoutName ) {
        std::string result = scalar_tensor_->toString();

        // Should not have double colons when no name is set
        EXPECT_FALSE( result.find( "::" ) != std::string::npos );
    }

    TEST_F( TensorIoTest, ToStringDifferentDataTypes ) {
        std::string fp32_result = vector_tensor_->toString();
        std::string int32_result = matrix_tensor_->toString();
        std::string int8_result = tensor_3d_->toString();
        std::string uint8_result = tensor_4d_->toString();

        EXPECT_TRUE( fp32_result.find( "Type: FP32" ) != std::string::npos );
        EXPECT_TRUE( int32_result.find( "Type: INT32" ) != std::string::npos );
        EXPECT_TRUE( int8_result.find( "Type: INT8" ) != std::string::npos );
        EXPECT_TRUE( uint8_result.find( "Type: UINT8" ) != std::string::npos );
    }

    TEST_F( TensorIoTest, ToStringDifferentShapes ) {
        std::string scalar_result = scalar_tensor_->toString();
        std::string vector_result = vector_tensor_->toString();
        std::string matrix_result = matrix_tensor_->toString();
        std::string tensor_3d_result = tensor_3d_->toString();
        std::string tensor_4d_result = tensor_4d_->toString();

        // Check shape representations
        EXPECT_TRUE( scalar_result.find( "shape=[]" ) != std::string::npos );
        EXPECT_TRUE( vector_result.find( "shape=[5]" ) != std::string::npos );
        EXPECT_TRUE( matrix_result.find( "shape=[3,4]" ) != std::string::npos );
        EXPECT_TRUE( tensor_3d_result.find( "shape=[2,3,4]" ) != std::string::npos );
        EXPECT_TRUE( tensor_4d_result.find( "shape=[2,3,4,5]" ) != std::string::npos );

        // Check stride representations
        EXPECT_TRUE( scalar_result.find( "strides=[]" ) != std::string::npos );
        EXPECT_TRUE( vector_result.find( "strides=[1]" ) != std::string::npos );
        EXPECT_TRUE( matrix_result.find( "strides=[4,1]" ) != std::string::npos );
        EXPECT_TRUE( tensor_3d_result.find( "strides=[12,4,1]" ) != std::string::npos );
        EXPECT_TRUE( tensor_4d_result.find( "strides=[60,20,5,1]" ) != std::string::npos );

        // Check size calculations
        EXPECT_TRUE( scalar_result.find( "size=1" ) != std::string::npos );
        EXPECT_TRUE( vector_result.find( "size=5" ) != std::string::npos );
        EXPECT_TRUE( matrix_result.find( "size=12" ) != std::string::npos );
        EXPECT_TRUE( tensor_3d_result.find( "size=24" ) != std::string::npos );
        EXPECT_TRUE( tensor_4d_result.find( "size=120" ) != std::string::npos );
    }

    TEST_F( TensorIoTest, ToStringCudaDevice ) {
        auto cuda_tensor = DeviceTensor<TensorDataType::FP32>( "CUDA:0", { 10, 10 } );
        std::string result = cuda_tensor.toString();

        EXPECT_TRUE( result.find( "Device: CUDA:0" ) != std::string::npos );
        EXPECT_TRUE( result.find( "Type: FP32" ) != std::string::npos );
    }

    TEST_F( TensorIoTest, ToStringCudaDeviceOnlyTypes ) {
        // Test device-only types with CUDA memory resources
        auto fp16_tensor = DeviceTensor<TensorDataType::FP16>( "CUDA:0", { 5, 5 } );
        auto bf16_tensor = DeviceTensor<TensorDataType::BF16>( "CUDA:0", { 3, 3 } );

        std::string fp16_result = fp16_tensor.toString();
        std::string bf16_result = bf16_tensor.toString();

        EXPECT_TRUE( fp16_result.find( "Type: FP16" ) != std::string::npos );
        EXPECT_TRUE( fp16_result.find( "Device: CUDA:0" ) != std::string::npos );
        EXPECT_TRUE( bf16_result.find( "Type: BF16" ) != std::string::npos );
        EXPECT_TRUE( bf16_result.find( "Device: CUDA:0" ) != std::string::npos );
    }

    TEST_F( TensorIoTest, ToStringWithBuffer ) {
        // Test with showBuffer = true (though buffer content display is not implemented for abstract types)
        std::string result = vector_tensor_->toString( true );

        // Should contain the message about buffer content not being implemented
        EXPECT_TRUE( result.find( "Buffer content display not implemented for abstract data types" ) != std::string::npos );
    }

    TEST_F( TensorIoTest, ToStringBufferNonHostAccessible ) {
        auto device_tensor = DeviceTensor<TensorDataType::FP32>( "CUDA:0", { 5, 5 } );
        std::string result = device_tensor.toString( true );

        // Should indicate tensor is not host-accessible
        EXPECT_TRUE( result.find( "Tensor is not host-accessible. Cannot output buffer contents." ) != std::string::npos );
    }

    TEST_F( TensorIoTest, ToStringUniqueIdentifiers ) {
        auto tensor1 = HostTensor<TensorDataType::FP32>( "CPU", { 2, 2 } );
        auto tensor2 = HostTensor<TensorDataType::FP32>( "CPU", { 2, 2 } );

        std::string result1 = tensor1.toString();
        std::string result2 = tensor2.toString();

        // Each tensor should have a unique identifier
        EXPECT_NE( result1, result2 );

        // Both should contain "tensor_" followed by different numbers
        EXPECT_TRUE( result1.find( "tensor_" ) != std::string::npos );
        EXPECT_TRUE( result2.find( "tensor_" ) != std::string::npos );
    }

    // =============================================================================
    // Stream Operator (<<) Tests
    // =============================================================================

    TEST_F( TensorIoTest, StreamOperatorBasic ) {
        std::ostringstream oss;
        oss << *vector_tensor_;

        std::string result = oss.str();
        std::string direct_toString = vector_tensor_->toString();

        // Stream operator should produce same output as toString()
        EXPECT_EQ( result, direct_toString );
    }

    TEST_F( TensorIoTest, StreamOperatorChaining ) {
        std::ostringstream oss;
        oss << "Tensor 1: " << *vector_tensor_ << "\nTensor 2: " << *matrix_tensor_;

        std::string result = oss.str();

        // Should contain both tensor descriptions
        EXPECT_TRUE( result.find( "Tensor 1:" ) != std::string::npos );
        EXPECT_TRUE( result.find( "Tensor 2:" ) != std::string::npos );
        EXPECT_TRUE( result.find( "test_vector" ) != std::string::npos );
        EXPECT_TRUE( result.find( "test_matrix" ) != std::string::npos );
    }

    TEST_F( TensorIoTest, StreamOperatorDifferentMemoryResources ) {
        std::ostringstream oss;

        auto host_tensor = HostTensor<TensorDataType::FP32>( "CPU", { 3, 3 } );
        host_tensor.setName( "host_test" );

        auto pinned_tensor = PinnedTensor<TensorDataType::FP32>( "CUDA:0", { 3, 3 } );
        pinned_tensor.setName( "pinned_test" );

        auto managed_tensor = UniversalTensor<TensorDataType::FP32>( "CUDA:0", { 3, 3 } );
        managed_tensor.setName( "managed_test" );

        oss << host_tensor << "\n---\n" << pinned_tensor << "\n---\n" << managed_tensor;

        std::string result = oss.str();

        EXPECT_TRUE( result.find( "host_test" ) != std::string::npos );
        EXPECT_TRUE( result.find( "Type: FP32" ) != std::string::npos );
        EXPECT_TRUE( result.find( "pinned_test" ) != std::string::npos );
        EXPECT_TRUE( result.find( "managed_test" ) != std::string::npos );
    }

    TEST_F( TensorIoTest, StreamOperatorMultipleDataTypes ) {
        std::ostringstream oss;

        auto fp32_tensor = HostTensor<TensorDataType::FP32>( "CPU", { 2 } );
        auto int32_tensor = HostTensor<TensorDataType::INT32>( "CPU", { 2 } );
        auto int8_tensor = HostTensor<TensorDataType::INT8>( "CPU", { 2 } );
        auto uint8_tensor = HostTensor<TensorDataType::UINT8>( "CPU", { 2 } );

        oss << fp32_tensor << "\n" << int32_tensor << "\n" << int8_tensor << "\n" << uint8_tensor;

        std::string result = oss.str();

        EXPECT_TRUE( result.find( "Type: FP32" ) != std::string::npos );
        EXPECT_TRUE( result.find( "Type: INT32" ) != std::string::npos );
        EXPECT_TRUE( result.find( "Type: INT8" ) != std::string::npos );
        EXPECT_TRUE( result.find( "Type: UINT8" ) != std::string::npos );
    }

    TEST_F( TensorIoTest, StreamOperatorDeviceOnlyTypes ) {
        std::ostringstream oss;

        auto fp16_tensor = DeviceTensor<TensorDataType::FP16>( "CUDA:0", { 2 } );
        auto bf16_tensor = DeviceTensor<TensorDataType::BF16>( "CUDA:0", { 2 } );

        oss << fp16_tensor << "\n" << bf16_tensor;

        std::string result = oss.str();

        EXPECT_TRUE( result.find( "Type: FP16" ) != std::string::npos );
        EXPECT_TRUE( result.find( "Type: BF16" ) != std::string::npos );
    }

    // =============================================================================
    // Edge Cases and Error Handling
    // =============================================================================

    TEST_F( TensorIoTest, EmptyTensorString ) {
        auto empty_tensor = HostTensor<TensorDataType::FP32>( "CPU", { 0 } );
        std::string result = empty_tensor.toString();

        EXPECT_TRUE( result.find( "size=0" ) != std::string::npos );
        EXPECT_TRUE( result.find( "shape=[0]" ) != std::string::npos );
    }

    TEST_F( TensorIoTest, LargeTensorString ) {
        auto large_tensor = HostTensor<TensorDataType::FP32>( "CPU", { 1000, 1000 } );
        std::string result = large_tensor.toString();

        EXPECT_TRUE( result.find( "size=1000000" ) != std::string::npos );
        EXPECT_TRUE( result.find( "shape=[1000,1000]" ) != std::string::npos );
        EXPECT_TRUE( result.find( "strides=[1000,1]" ) != std::string::npos );
    }

    TEST_F( TensorIoTest, ToStringConsistency ) {
        // toString() should be consistent across multiple calls
        std::string result1 = matrix_tensor_->toString();
        std::string result2 = matrix_tensor_->toString();

        EXPECT_EQ( result1, result2 );
    }

    TEST_F( TensorIoTest, StreamOperatorConsistency ) {
        // Stream operator should be consistent with toString()
        std::ostringstream oss;
        oss << *matrix_tensor_;

        std::string stream_result = oss.str();
        std::string toString_result = matrix_tensor_->toString();

        EXPECT_EQ( stream_result, toString_result );
    }

    // =============================================================================
    // Memory Resource Specific Tests
    // =============================================================================

    TEST_F( TensorIoTest, HostTensorAlias ) {
        auto host_tensor = HostTensor<TensorDataType::FP32>( "CPU", { 5, 5 } );
        std::string result = host_tensor.toString();

        EXPECT_TRUE( result.find( "Device: CPU" ) != std::string::npos );
        EXPECT_TRUE( result.find( "Type: FP32" ) != std::string::npos );
    }

    TEST_F( TensorIoTest, DeviceTensorAlias ) {
        auto device_tensor = DeviceTensor<TensorDataType::FP32>( "CUDA:0", { 5, 5 } );
        std::string result = device_tensor.toString();

        EXPECT_TRUE( result.find( "Device: CUDA:0" ) != std::string::npos );
        EXPECT_TRUE( result.find( "Type: FP32" ) != std::string::npos );
    }

    TEST_F( TensorIoTest, PinnedTensorAlias ) {
        auto pinned_tensor = PinnedTensor<TensorDataType::FP32>( "CUDA:0", { 5, 5 } );  // Changed from "CPU" to "CUDA:0"
        std::string result = pinned_tensor.toString();

        EXPECT_TRUE( result.find( "Device: CUDA:0" ) != std::string::npos );  // Changed from "CPU" to "CUDA:0"
        EXPECT_TRUE( result.find( "Type: FP32" ) != std::string::npos );
    }

    TEST_F( TensorIoTest, UniversalTensorAlias ) {
        auto universal_tensor = UniversalTensor<TensorDataType::FP32>( "CUDA:0", { 5, 5 } );
        std::string result = universal_tensor.toString();

        EXPECT_TRUE( result.find( "Device: CUDA:0" ) != std::string::npos );
        EXPECT_TRUE( result.find( "Type: FP32" ) != std::string::npos );
    }

    // =============================================================================
    // Formatting and Layout Tests
    // =============================================================================

    TEST_F( TensorIoTest, OutputLayoutFormatting ) {
        // Test that the outputLayout() method produces correct formatting
        std::string result = tensor_3d_->toString();

        // Should contain proper bracketing and comma separation
        EXPECT_TRUE( result.find( "shape=[2,3,4]" ) != std::string::npos );
        EXPECT_TRUE( result.find( "strides=[12,4,1]" ) != std::string::npos );
        EXPECT_TRUE( result.find( "format=RowMajor" ) != std::string::npos );

        // Should not have extra spaces or malformed brackets
        EXPECT_FALSE( result.find( "shape=[ 2, 3, 4 ]" ) != std::string::npos );
        EXPECT_FALSE( result.find( "strides=[ 12, 4, 1 ]" ) != std::string::npos );
    }

    TEST_F( TensorIoTest, NewlineHandling ) {
        std::string result = matrix_tensor_->toString();

        // Should end with a newline
        EXPECT_TRUE( result.back() == '\n' );

        // Should have exactly one newline at the end
        size_t newline_count = 0;
        for (char c : result) {
            if (c == '\n') newline_count++;
        }
        EXPECT_EQ( newline_count, 1u );
    }
}