/**
 * @file Math.Cuda.cpp
 * @brief Unit tests for CUDA tensor mathematical operations.
 *
 * Tests element-wise operations (add, subtract, multiply, divide) on CUDA tensors
 * using the ExecutionContext design with raw pointer borrowing semantics.
 * Tests both synchronous and asynchronous execution patterns.
 */

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <algorithm>
#include <cmath>

import Mila;

namespace Dnn::Tensors::TensorOps::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    /**
     * @brief Test fixture for CUDA tensor math operations.
     */
    class CudaTensorMathTest : public ::testing::Test {
    protected:
        void SetUp() override {
            // Create CUDA ExecutionContext with device ID 0
            exec_ctx_ = std::make_unique<ExecutionContext<DeviceType::Cuda>>( 0 );
        }

        void TearDown() override {
            if (exec_ctx_)
            {
                exec_ctx_->synchronize();
                exec_ctx_.reset();
            }
        }

        std::unique_ptr<ExecutionContext<DeviceType::Cuda>> exec_ctx_;
    };

    /**
     * @brief Helper to create a CUDA tensor with values.
     *
     * @tparam TDataType Tensor data type
     * @param device_name Device name string
     * @param shape Tensor shape
     * @param values Initial values (optional)
     * @return Tensor with specified shape and values
     */
    template<TensorDataType TDataType>
    Tensor<TDataType, CudaDeviceMemoryResource> makeCudaTensor(
        const std::string& device_name,
        const std::vector<size_t>& shape,
        const std::vector<typename TensorHostTypeMap<TDataType>::host_type>& values = {} )
    {
        Tensor<TDataType, CudaDeviceMemoryResource> tensor( device_name, shape );

        if (!values.empty() && !tensor.empty() && tensor.size() == values.size())
        {
            // Use fill operation to initialize tensor with host values
            // Since CUDA tensors aren't host-accessible, we need to use fill operations
            fill( tensor, std::span{ values } );
        }

        return tensor;
    }

    /**
     * @brief Helper to copy CUDA tensor to host for verification.
     *
     * @tparam TDataType Tensor data type
     * @param cuda_tensor CUDA tensor to copy
     * @return CPU tensor with copied values
     */
    template<TensorDataType TDataType>
    Tensor<TDataType, CpuMemoryResource> copyToHost(
        const Tensor<TDataType, CudaDeviceMemoryResource>& cuda_tensor )
    {
        // Create CPU tensor with same shape
        Tensor<TDataType, CpuMemoryResource> cpu_tensor( "CPU", cuda_tensor.shape() );

        // TODO: Implement transfer operations to copy from CUDA to CPU
        // For now, this is a placeholder implementation
        // transfer(cuda_tensor, cpu_tensor);

        return cpu_tensor;
    }

    // ============================================================================
    // Addition Tests
    // ============================================================================

    TEST_F( CudaTensorMathTest, Add_SameShape_Int32 )
    {
        auto device_name = exec_ctx_->getDeviceName();

        auto a = Tensor<TensorDataType::INT32, CudaDeviceMemoryResource>( device_name, { 2, 2 } );
        auto b = Tensor<TensorDataType::INT32, CudaDeviceMemoryResource>( device_name, { 2, 2 } );

        // Fill tensors with test data
        std::vector<int32_t> values_a = { 1, 2, 3, 4 };
        std::vector<int32_t> values_b = { 5, 6, 7, 8 };

        fill( a, std::span{ values_a }, exec_ctx_.get() );
        fill( b, std::span{ values_b }, exec_ctx_.get() );

        // Pre-allocate result tensor
        auto result = Tensor<TensorDataType::INT32, CudaDeviceMemoryResource>( device_name, { 2, 2 } );

        // Perform operation with ExecutionContext
        add( a, b, result, exec_ctx_.get() );

        // Synchronize to ensure completion
        exec_ctx_->synchronize();

        // Verify shape
        EXPECT_EQ( result.shape(), a.shape() );

        // TODO: Add verification by copying to host and checking values
        // Expected results: [6, 8, 10, 12]
    }

    TEST_F( CudaTensorMathTest, Add_SameShape_Float )
    {
        auto device_name = exec_ctx_->getDeviceName();

        auto a = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>( device_name, { 2, 2 } );
        auto b = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>( device_name, { 2, 2 } );

        std::vector<float> values_a = { 1.5f, 2.5f, 3.5f, 4.5f };
        std::vector<float> values_b = { 0.5f, 1.5f, 2.5f, 3.5f };

        fill( a, std::span{ values_a }, exec_ctx_.get() );
        fill( b, std::span{ values_b }, exec_ctx_.get() );

        auto result = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>( device_name, { 2, 2 } );

        add( a, b, result, exec_ctx_.get() );
        exec_ctx_->synchronize();

        EXPECT_EQ( result.shape(), a.shape() );
        // Expected results: [2.0f, 4.0f, 6.0f, 8.0f]
    }

    TEST_F( CudaTensorMathTest, Add_WithoutExecutionContext )
    {
        auto device_name = exec_ctx_->getDeviceName();

        auto a = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>( device_name, { 3 } );
        auto b = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>( device_name, { 3 } );

        std::vector<float> values_a = { 1.0f, 2.0f, 3.0f };
        std::vector<float> values_b = { 4.0f, 5.0f, 6.0f };

        fill( a, std::span{ values_a } );
        fill( b, std::span{ values_b } );

        auto result = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>( device_name, { 3 } );

        // Call without ExecutionContext (uses default stream and synchronizes)
        add( a, b, result );

        // No need to synchronize - operation is already synchronized
        // Expected results: [5.0f, 7.0f, 9.0f]
    }

    TEST_F( CudaTensorMathTest, Add_LargeArray_Float )
    {
        auto device_name = exec_ctx_->getDeviceName();
        const size_t size = 10000;

        std::vector<float> values_a( size );
        std::vector<float> values_b( size );

        for (size_t i = 0; i < size; ++i)
        {
            values_a[i] = static_cast<float>( i );
            values_b[i] = static_cast<float>( i * 2 );
        }

        auto a = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>( device_name, { size } );
        auto b = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>( device_name, { size } );
        auto result = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>( device_name, { size } );

        fill( a, std::span{ values_a }, exec_ctx_.get() );
        fill( b, std::span{ values_b }, exec_ctx_.get() );

        add( a, b, result, exec_ctx_.get() );
        exec_ctx_->synchronize();

        EXPECT_EQ( result.size(), size );
        // Expected: result[i] = i + i*2 = i*3
    }

    TEST_F( CudaTensorMathTest, Add_DifferentShape_Throws )
    {
        auto device_name = exec_ctx_->getDeviceName();

        auto a = Tensor<TensorDataType::INT32, CudaDeviceMemoryResource>( device_name, { 2, 2 } );
        auto b = Tensor<TensorDataType::INT32, CudaDeviceMemoryResource>( device_name, { 2, 1 } );
        auto result = Tensor<TensorDataType::INT32, CudaDeviceMemoryResource>( device_name, { 2, 2 } );

        EXPECT_THROW( add( a, b, result, exec_ctx_.get() ), std::invalid_argument );
    }

    TEST_F( CudaTensorMathTest, Add_ResultShapeMismatch_Throws )
    {
        auto device_name = exec_ctx_->getDeviceName();

        auto a = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>( device_name, { 2, 2 } );
        auto b = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>( device_name, { 2, 2 } );
        auto result = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>( device_name, { 3 } );

        EXPECT_THROW( add( a, b, result, exec_ctx_.get() ), std::invalid_argument );
    }

    TEST_F( CudaTensorMathTest, Add_Scalar_Tensors )
    {
        auto device_name = exec_ctx_->getDeviceName();

        // Create scalar tensors (rank 0)
        auto a = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>( device_name, {} );
        auto b = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>( device_name, {} );
        auto result = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>( device_name, {} );

        fill( a, 5.0f, exec_ctx_.get() );
        fill( b, 3.0f, exec_ctx_.get() );

        EXPECT_TRUE( a.isScalar() );
        EXPECT_TRUE( b.isScalar() );

        add( a, b, result, exec_ctx_.get() );
        exec_ctx_->synchronize();

        EXPECT_TRUE( result.isScalar() );
        // Expected result: 8.0f
    }

    TEST_F( CudaTensorMathTest, Add_UsingOperator )
    {
        auto device_name = exec_ctx_->getDeviceName();

        auto a = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>( device_name, { 3 } );
        auto b = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>( device_name, { 3 } );

        std::vector<float> values_a = { 1.0f, 2.0f, 3.0f };
        std::vector<float> values_b = { 4.0f, 5.0f, 6.0f };

        fill( a, std::span{ values_a } );
        fill( b, std::span{ values_b } );

        // Operator automatically allocates result and uses default execution
        auto result = a + b;

        // Expected results: [5.0f, 7.0f, 9.0f]
    }

    // ============================================================================
    // Subtraction Tests
    // ============================================================================

    TEST_F( CudaTensorMathTest, Subtract_SameShape_Int32 )
    {
        auto device_name = exec_ctx_->getDeviceName();

        auto a = Tensor<TensorDataType::INT32, CudaDeviceMemoryResource>( device_name, { 2, 2 } );
        auto b = Tensor<TensorDataType::INT32, CudaDeviceMemoryResource>( device_name, { 2, 2 } );

        std::vector<int32_t> values_a = { 10, 20, 30, 40 };
        std::vector<int32_t> values_b = { 1, 2, 3, 4 };

        fill( a, std::span{ values_a }, exec_ctx_.get() );
        fill( b, std::span{ values_b }, exec_ctx_.get() );

        auto result = Tensor<TensorDataType::INT32, CudaDeviceMemoryResource>( device_name, { 2, 2 } );

        subtract( a, b, result, exec_ctx_.get() );
        exec_ctx_->synchronize();

        EXPECT_EQ( result.shape(), a.shape() );
        // Expected results: [9, 18, 27, 36]
    }

    TEST_F( CudaTensorMathTest, Subtract_SameShape_Float )
    {
        auto device_name = exec_ctx_->getDeviceName();

        auto a = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>( device_name, { 3 } );
        auto b = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>( device_name, { 3 } );

        std::vector<float> values_a = { 5.5f, 10.0f, 15.5f };
        std::vector<float> values_b = { 1.5f, 2.0f, 3.5f };

        fill( a, std::span{ values_a }, exec_ctx_.get() );
        fill( b, std::span{ values_b }, exec_ctx_.get() );

        auto result = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>( device_name, { 3 } );

        subtract( a, b, result, exec_ctx_.get() );
        exec_ctx_->synchronize();

        // Expected results: [4.0f, 8.0f, 12.0f]
    }

    TEST_F( CudaTensorMathTest, Subtract_Negative_Results )
    {
        auto device_name = exec_ctx_->getDeviceName();

        auto a = Tensor<TensorDataType::INT32, CudaDeviceMemoryResource>( device_name, { 2 } );
        auto b = Tensor<TensorDataType::INT32, CudaDeviceMemoryResource>( device_name, { 2 } );

        std::vector<int32_t> values_a = { 5, 10 };
        std::vector<int32_t> values_b = { 10, 5 };

        fill( a, std::span{ values_a } );
        fill( b, std::span{ values_b } );

        auto result = Tensor<TensorDataType::INT32, CudaDeviceMemoryResource>( device_name, { 2 } );

        subtract( a, b, result );  // Without context

        // Expected results: [-5, 5]
    }

    TEST_F( CudaTensorMathTest, Subtract_UsingOperator )
    {
        auto device_name = exec_ctx_->getDeviceName();

        auto a = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>( device_name, { 2 } );
        auto b = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>( device_name, { 2 } );

        std::vector<float> values_a = { 10.0f, 20.0f };
        std::vector<float> values_b = { 3.0f, 5.0f };

        fill( a, std::span{ values_a } );
        fill( b, std::span{ values_b } );

        auto result = a - b;

        // Expected results: [7.0f, 15.0f]
    }

    // ============================================================================
    // Multiplication Tests
    // ============================================================================

    TEST_F( CudaTensorMathTest, Multiply_SameShape_Int32 )
    {
        auto device_name = exec_ctx_->getDeviceName();

        auto a = Tensor<TensorDataType::INT32, CudaDeviceMemoryResource>( device_name, { 2, 2 } );
        auto b = Tensor<TensorDataType::INT32, CudaDeviceMemoryResource>( device_name, { 2, 2 } );

        std::vector<int32_t> values_a = { 2, 3, 4, 5 };
        std::vector<int32_t> values_b = { 10, 10, 10, 10 };

        fill( a, std::span{ values_a }, exec_ctx_.get() );
        fill( b, std::span{ values_b }, exec_ctx_.get() );

        auto result = Tensor<TensorDataType::INT32, CudaDeviceMemoryResource>( device_name, { 2, 2 } );

        multiply( a, b, result, exec_ctx_.get() );
        exec_ctx_->synchronize();

        // Expected results: [20, 30, 40, 50]
    }

    TEST_F( CudaTensorMathTest, Multiply_SameShape_Float )
    {
        auto device_name = exec_ctx_->getDeviceName();

        auto a = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>( device_name, { 3 } );
        auto b = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>( device_name, { 3 } );

        std::vector<float> values_a = { 2.5f, 3.0f, 4.5f };
        std::vector<float> values_b = { 2.0f, 2.0f, 2.0f };

        fill( a, std::span{ values_a }, exec_ctx_.get() );
        fill( b, std::span{ values_b }, exec_ctx_.get() );

        auto result = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>( device_name, { 3 } );

        multiply( a, b, result, exec_ctx_.get() );
        exec_ctx_->synchronize();

        // Expected results: [5.0f, 6.0f, 9.0f]
    }

    TEST_F( CudaTensorMathTest, Multiply_WithZeros )
    {
        auto device_name = exec_ctx_->getDeviceName();

        auto a = Tensor<TensorDataType::INT32, CudaDeviceMemoryResource>( device_name, { 3 } );
        auto b = Tensor<TensorDataType::INT32, CudaDeviceMemoryResource>( device_name, { 3 } );

        std::vector<int32_t> values_a = { 5, 0, 10 };
        std::vector<int32_t> values_b = { 2, 3, 0 };

        fill( a, std::span{ values_a } );
        fill( b, std::span{ values_b } );

        auto result = Tensor<TensorDataType::INT32, CudaDeviceMemoryResource>( device_name, { 3 } );

        multiply( a, b, result );

        // Expected results: [10, 0, 0]
    }

    TEST_F( CudaTensorMathTest, Multiply_UsingOperator )
    {
        auto device_name = exec_ctx_->getDeviceName();

        auto a = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>( device_name, { 2 } );
        auto b = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>( device_name, { 2 } );

        std::vector<float> values_a = { 3.0f, 4.0f };
        std::vector<float> values_b = { 5.0f, 6.0f };

        fill( a, std::span{ values_a } );
        fill( b, std::span{ values_b } );

        auto result = a * b;

        // Expected results: [15.0f, 24.0f]
    }

    // ============================================================================
    // Division Tests (Note: divide operation may not be implemented yet)
    // ============================================================================

    /*
    TEST_F( CudaTensorMathTest, Divide_SameShape_Int32 )
    {
        auto device_name = exec_ctx_->getDeviceName();

        auto a = Tensor<TensorDataType::INT32, CudaDeviceMemoryResource>( device_name, { 3 } );
        auto b = Tensor<TensorDataType::INT32, CudaDeviceMemoryResource>( device_name, { 3 } );

        std::vector<int32_t> values_a = { 20, 30, 40 };
        std::vector<int32_t> values_b = { 2, 3, 4 };

        fill( a, std::span{ values_a }, exec_ctx_.get() );
        fill( b, std::span{ values_b }, exec_ctx_.get() );

        auto result = Tensor<TensorDataType::INT32, CudaDeviceMemoryResource>( device_name, { 3 } );

        divide( a, b, result, exec_ctx_.get() );
        exec_ctx_->synchronize();

        // Expected results: [10, 10, 10]
    }

    TEST_F( CudaTensorMathTest, Divide_SameShape_Float )
    {
        auto device_name = exec_ctx_->getDeviceName();

        auto a = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>( device_name, { 3 } );
        auto b = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>( device_name, { 3 } );

        std::vector<float> values_a = { 10.0f, 20.0f, 30.0f };
        std::vector<float> values_b = { 2.0f, 4.0f, 5.0f };

        fill( a, std::span{ values_a }, exec_ctx_.get() );
        fill( b, std::span{ values_b }, exec_ctx_.get() );

        auto result = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>( device_name, { 3 } );

        divide( a, b, result, exec_context_.get() );
        exec_context_->synchronize();

        // Expected results: [5.0f, 5.0f, 6.0f]
    }

    TEST_F( CudaTensorMathTest, Divide_ByZero_Float_InfNaN )
    {
        auto device_name = exec_ctx_->getDeviceName();

        auto a = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>( device_name, { 2 } );
        auto b = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>( device_name, { 2 } );

        std::vector<float> values_a = { 10.0f, 0.0f };
        std::vector<float> values_b = { 0.0f, 0.0f };

        fill( a, std::span{ values_a } );
        fill( b, std::span{ values_b } );

        auto result = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>( device_name, { 2 } );

        // Float division by zero follows IEEE 754 (should produce inf/nan)
        divide( a, b, result );

        // Expected: result[0] = inf, result[1] = nan
        // Need to copy to host to verify with std::isinf and std::isnan
    }

    TEST_F( CudaTensorMathTest, Divide_UsingOperator )
    {
        auto device_name = exec_ctx_->getDeviceName();

        auto a = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>( device_name, { 2 } );
        auto b = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>( device_name, { 2 } );

        std::vector<float> values_a = { 20.0f, 30.0f };
        std::vector<float> values_b = { 4.0f, 5.0f };

        fill( a, std::span{ values_a } );
        fill( b, std::span{ values_b } );

        auto result = a / b;

        // Expected results: [5.0f, 6.0f]
    }
    */

    // ============================================================================
    // Sum Reduction Tests
    // ============================================================================

    TEST_F( CudaTensorMathTest, Sum_SmallTensor )
    {
        auto device_name = exec_ctx_->getDeviceName();

        auto tensor = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>( device_name, { 4 } );
        std::vector<float> values = { 1.0f, 2.0f, 3.0f, 4.0f };

        fill( tensor, std::span{ values }, exec_ctx_.get() );

        float result = sum( tensor, exec_ctx_.get() );

        EXPECT_FLOAT_EQ( result, 10.0f );
    }

    TEST_F( CudaTensorMathTest, Sum_LargeTensor )
    {
        auto device_name = exec_ctx_->getDeviceName();
        const size_t size = 10000;

        auto tensor = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>( device_name, { size } );
        std::vector<float> values( size );

        for (size_t i = 0; i < size; ++i)
        {
            values[i] = 1.0f;
        }

        fill( tensor, std::span{ values }, exec_ctx_.get() );

        float result = sum( tensor, exec_ctx_.get() );

        EXPECT_FLOAT_EQ( result, static_cast<float>( size ) );
    }

    TEST_F( CudaTensorMathTest, Sum_WithoutExecutionContext )
    {
        auto device_name = exec_ctx_->getDeviceName();

        auto tensor = Tensor<TensorDataType::INT32, CudaDeviceMemoryResource>( device_name, { 3 } );
        std::vector<int32_t> values = { 10, 20, 30 };

        fill( tensor, std::span{ values } );

        float result = sum( tensor );  // No ExecutionContext

        EXPECT_FLOAT_EQ( result, 60.0f );
    }

    TEST_F( CudaTensorMathTest, Sum_EmptyTensor )
    {
        auto device_name = exec_ctx_->getDeviceName();

        auto tensor = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>( device_name, { 0 } );

        float result = sum( tensor, exec_ctx_.get() );

        EXPECT_FLOAT_EQ( result, 0.0f );
    }

    TEST_F( CudaTensorMathTest, Sum_ScalarTensor )
    {
        auto device_name = exec_ctx_->getDeviceName();

        auto tensor = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>( device_name, {} );
        fill( tensor, 42.0f, exec_ctx_.get() );

        float result = sum( tensor, exec_ctx_.get() );

        EXPECT_FLOAT_EQ( result, 42.0f );
    }

    // ============================================================================
    // Multi-dimensional Tensor Tests
    // ============================================================================

    TEST_F( CudaTensorMathTest, Add_3D_Tensors )
    {
        auto device_name = exec_ctx_->getDeviceName();

        std::vector<float> values_a( 24 );
        std::vector<float> values_b( 24 );

        for (size_t i = 0; i < 24; ++i)
        {
            values_a[i] = static_cast<float>( i );
            values_b[i] = static_cast<float>( i * 2 );
        }

        auto a = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>( device_name, { 2, 3, 4 } );
        auto b = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>( device_name, { 2, 3, 4 } );
        auto result = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>( device_name, { 2, 3, 4 } );

        fill( a, std::span{ values_a }, exec_ctx_.get() );
        fill( b, std::span{ values_b }, exec_ctx_.get() );

        add( a, b, result, exec_ctx_.get() );
        exec_ctx_->synchronize();

        EXPECT_EQ( result.shape(), a.shape() );
        EXPECT_EQ( result.size(), 24u );

        // Expected: result[i] = i + i*2 = i*3 for each element
    }

    // ============================================================================
    // Edge Case Tests
    // ============================================================================

    TEST_F( CudaTensorMathTest, Operations_SingleElement )
    {
        auto device_name = exec_ctx_->getDeviceName();

        auto a = Tensor<TensorDataType::INT32, CudaDeviceMemoryResource>( device_name, { 1 } );
        auto b = Tensor<TensorDataType::INT32, CudaDeviceMemoryResource>( device_name, { 1 } );

        fill( a, 10, exec_ctx_.get() );
        fill( b, 5, exec_ctx_.get() );

        auto add_result = Tensor<TensorDataType::INT32, CudaDeviceMemoryResource>( device_name, { 1 } );
        auto sub_result = Tensor<TensorDataType::INT32, CudaDeviceMemoryResource>( device_name, { 1 } );
        auto mul_result = Tensor<TensorDataType::INT32, CudaDeviceMemoryResource>( device_name, { 1 } );

        add( a, b, add_result, exec_ctx_.get() );
        subtract( a, b, sub_result, exec_ctx_.get() );
        multiply( a, b, mul_result, exec_ctx_.get() );

        exec_ctx_->synchronize();

        // Expected: add=15, sub=5, mul=50
        // Need to copy to host to verify actual values
    }

    // ============================================================================
    // Chained Operations Test
    // ============================================================================

    TEST_F( CudaTensorMathTest, ChainedOperations_WithContext )
    {
        auto device_name = exec_ctx_->getDeviceName();

        auto a = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>( device_name, { 3 } );
        auto b = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>( device_name, { 3 } );
        auto c = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>( device_name, { 3 } );

        std::vector<float> values_a = { 10.0f, 20.0f, 30.0f };
        std::vector<float> values_b = { 2.0f, 3.0f, 4.0f };
        std::vector<float> values_c = { 1.0f, 1.0f, 1.0f };

        fill( a, std::span{ values_a }, exec_ctx_.get() );
        fill( b, std::span{ values_b }, exec_ctx_.get() );
        fill( c, std::span{ values_c }, exec_ctx_.get() );

        auto temp = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>( device_name, { 3 } );
        auto result = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>( device_name, { 3 } );

        // Compute: (a + b) * c
        add( a, b, temp, exec_ctx_.get() );
        multiply( temp, c, result, exec_ctx_.get() );

        exec_ctx_->synchronize();

        // Expected results: [12.0f, 23.0f, 34.0f]
    }

    TEST_F( CudaTensorMathTest, ChainedOperations_UsingOperators )
    {
        auto device_name = exec_ctx_->getDeviceName();

        auto a = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>( device_name, { 2 } );
        auto b = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>( device_name, { 2 } );
        auto c = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>( device_name, { 2 } );

        std::vector<float> values_a = { 5.0f, 10.0f };
        std::vector<float> values_b = { 2.0f, 3.0f };
        std::vector<float> values_c = { 1.0f, 1.0f };

        fill( a, std::span{ values_a } );
        fill( b, std::span{ values_b } );
        fill( c, std::span{ values_c } );

        // Compute: (a + b) * c - b
        auto result = (a + b) * c - b;

        // Expected results: [5.0f, 10.0f]  // (5+2)*1-2 = 5, (10+3)*1-3 = 10
    }

    // ============================================================================
    // Asynchronous Execution Tests
    // ============================================================================

    TEST_F( CudaTensorMathTest, AsyncExecution_MultipleOperations )
    {
        auto device_name = exec_ctx_->getDeviceName();

        auto a = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>( device_name, { 1000 } );
        auto b = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>( device_name, { 1000 } );
        auto c = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>( device_name, { 1000 } );

        std::vector<float> values( 1000, 1.0f );
        fill( a, std::span{ values }, exec_ctx_.get() );
        fill( b, std::span{ values }, exec_ctx_.get() );

        auto result1 = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>( device_name, { 1000 } );
        auto result2 = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>( device_name, { 1000 } );

        // Queue multiple async operations on same stream
        add( a, b, result1, exec_ctx_.get() );       // Don't sync yet
        multiply( result1, a, result2, exec_ctx_.get() );  // Don't sync yet

        // All operations complete when we synchronize
        exec_ctx_->synchronize();

        // Verify operations completed
        float sum_result = sum( result2, exec_ctx_.get() );
        EXPECT_FLOAT_EQ( sum_result, 2000.0f );  // (1+1)*1 = 2 for each of 1000 elements
    }

    // ============================================================================
    // Error Handling Tests
    // ============================================================================

    TEST_F( CudaTensorMathTest, Operations_EmptyTensors )
    {
        auto device_name = exec_ctx_->getDeviceName();

        auto a = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>( device_name, { 0 } );
        auto b = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>( device_name, { 0 } );
        auto result = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>( device_name, { 0 } );

        // Operations on empty tensors should not throw
        EXPECT_NO_THROW( add( a, b, result, exec_ctx_.get() ) );
        EXPECT_NO_THROW( subtract( a, b, result, exec_ctx_.get() ) );
        EXPECT_NO_THROW( multiply( a, b, result, exec_ctx_.get() ) );

        exec_ctx_->synchronize();
    }
}