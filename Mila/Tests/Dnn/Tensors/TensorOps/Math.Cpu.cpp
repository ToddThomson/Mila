/**
 * @file Math.Cpu.cpp
 * @brief Unit tests for CPU tensor mathematical operations.
 *
 * Tests element-wise operations (add, subtract, multiply, divide) on CPU tensors
 * using the ExecutionContext design with raw pointer borrowing semantics.
 */

#include <gtest/gtest.h>
#include <memory>
#include <vector>

import Mila;

namespace Dnn::Tensors::TensorOps::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    /**
     * @brief Test fixture for CPU tensor math operations.
     */
    class CpuTensorMathTest : public ::testing::Test {
    protected:
        void SetUp() override {
            // CPU ExecutionContext doesn't require parameters
            exec_ctx_ = std::make_unique<ExecutionContext<DeviceType::Cpu>>();
        }

        void TearDown() override {
            exec_ctx_.reset();
        }

        std::unique_ptr<ExecutionContext<DeviceType::Cpu>> exec_ctx_;
    };

    /**
     * @brief Helper to create a CPU tensor with values.
     *
     * @tparam TDataType Tensor data type
     * @param device_name Device name string
     * @param shape Tensor shape
     * @param values Initial values (optional)
     * @return Tensor with specified shape and values
     */
    template<TensorDataType TDataType>
    Tensor<TDataType, CpuMemoryResource> makeCpuTensor(
        const std::string& device_name,
        const std::vector<size_t>& shape,
        const std::vector<typename TensorHostTypeMap<TDataType>::host_type>& values = {} )
    {
        Tensor<TDataType, CpuMemoryResource> tensor( device_name, shape );

        if (!values.empty() && !tensor.empty() && tensor.size() == values.size())
        {
            auto data = tensor.data();
            for (size_t i = 0; i < values.size(); ++i)
            {
                data[i] = values[i];
            }
        }

        return tensor;
    }

    // ============================================================================
    // Addition Tests
    // ============================================================================

    TEST_F( CpuTensorMathTest, Add_SameShape_Int32 )
    {
        auto device_name = exec_ctx_->getDeviceName();

        auto a = Tensor<TensorDataType::INT32, CpuMemoryResource>( device_name, { 2, 2 } );
        auto data_a = a.data();
        for (size_t i = 0; i < a.size(); ++i)
        {
            data_a[i] = static_cast<int32_t>( i + 1 );
        }

        auto b = Tensor<TensorDataType::INT32, CpuMemoryResource>( device_name, { 2, 2 } );
        auto data_b = b.data();
        for (size_t i = 0; i < b.size(); ++i)
        {
            data_b[i] = static_cast<int32_t>( i + 5 );
        }

        // Pre-allocate result tensor
        auto result = Tensor<TensorDataType::INT32, CpuMemoryResource>( device_name, { 2, 2 } );

        // Perform operation with ExecutionContext
        add( a, b, result, exec_ctx_.get() );

        EXPECT_EQ( result.shape(), a.shape() );
        auto data_result = result.data();
        EXPECT_EQ( data_result[0], 6 );
        EXPECT_EQ( data_result[1], 8 );
        EXPECT_EQ( data_result[2], 10 );
        EXPECT_EQ( data_result[3], 12 );
    }

    TEST_F( CpuTensorMathTest, Add_SameShape_Float )
    {
        auto device_name = exec_ctx_->getDeviceName();

        auto a = makeCpuTensor<TensorDataType::FP32>(
            device_name, { 2, 2 }, { 1.5f, 2.5f, 3.5f, 4.5f } );
        auto b = makeCpuTensor<TensorDataType::FP32>(
            device_name, { 2, 2 }, { 0.5f, 1.5f, 2.5f, 3.5f } );

        auto result = Tensor<TensorDataType::FP32, CpuMemoryResource>( device_name, { 2, 2 } );

        add( a, b, result, exec_ctx_.get() );

        EXPECT_EQ( result.shape(), a.shape() );
        auto data = result.data();
        EXPECT_FLOAT_EQ( data[0], 2.0f );
        EXPECT_FLOAT_EQ( data[1], 4.0f );
        EXPECT_FLOAT_EQ( data[2], 6.0f );
        EXPECT_FLOAT_EQ( data[3], 8.0f );
    }

    TEST_F( CpuTensorMathTest, Add_WithoutExecutionContext )
    {
        auto device_name = exec_ctx_->getDeviceName();

        auto a = makeCpuTensor<TensorDataType::FP32>(
            device_name, { 3 }, { 1.0f, 2.0f, 3.0f } );
        auto b = makeCpuTensor<TensorDataType::FP32>(
            device_name, { 3 }, { 4.0f, 5.0f, 6.0f } );

        auto result = Tensor<TensorDataType::FP32, CpuMemoryResource>( device_name, { 3 } );

        // Call without ExecutionContext (nullptr)
        add( a, b, result );

        auto data = result.data();
        EXPECT_FLOAT_EQ( data[0], 5.0f );
        EXPECT_FLOAT_EQ( data[1], 7.0f );
        EXPECT_FLOAT_EQ( data[2], 9.0f );
    }

    TEST_F( CpuTensorMathTest, Add_LargeArray_Float )
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

        auto a = makeCpuTensor<TensorDataType::FP32>( device_name, { size }, values_a );
        auto b = makeCpuTensor<TensorDataType::FP32>( device_name, { size }, values_b );
        auto result = Tensor<TensorDataType::FP32, CpuMemoryResource>( device_name, { size } );

        add( a, b, result, exec_ctx_.get() );

        EXPECT_EQ( result.size(), size );
        auto data = result.data();

        // Verify a few samples
        EXPECT_FLOAT_EQ( data[0], 0.0f );
        EXPECT_FLOAT_EQ( data[100], 300.0f );
        EXPECT_FLOAT_EQ( data[size - 1], static_cast<float>((size - 1) * 3) );
    }

    TEST_F( CpuTensorMathTest, Add_DifferentShape_Throws )
    {
        auto device_name = exec_ctx_->getDeviceName();

        auto a = makeCpuTensor<TensorDataType::INT32>( device_name, { 2, 2 }, { 1, 2, 3, 4 } );
        auto b = makeCpuTensor<TensorDataType::INT32>( device_name, { 2, 1 }, { 5, 6 } );
        auto result = Tensor<TensorDataType::INT32, CpuMemoryResource>( device_name, { 2, 2 } );

        EXPECT_THROW( add( a, b, result, exec_ctx_.get() ), std::invalid_argument );
    }

    TEST_F( CpuTensorMathTest, Add_ResultShapeMismatch_Throws )
    {
        auto device_name = exec_ctx_->getDeviceName();

        auto a = makeCpuTensor<TensorDataType::FP32>( device_name, { 2, 2 }, { 1.0f, 2.0f, 3.0f, 4.0f } );
        auto b = makeCpuTensor<TensorDataType::FP32>( device_name, { 2, 2 }, { 5.0f, 6.0f, 7.0f, 8.0f } );
        auto result = Tensor<TensorDataType::FP32, CpuMemoryResource>( device_name, { 3 } );

        EXPECT_THROW( add( a, b, result, exec_ctx_.get() ), std::invalid_argument );
    }

    TEST_F( CpuTensorMathTest, Add_Scalar_Tensors )
    {
        auto device_name = exec_ctx_->getDeviceName();

        // Create scalar tensors (rank 0)
        auto a = Tensor<TensorDataType::FP32, CpuMemoryResource>( device_name, {} );
        auto b = Tensor<TensorDataType::FP32, CpuMemoryResource>( device_name, {} );
        auto result = Tensor<TensorDataType::FP32, CpuMemoryResource>( device_name, {} );

        a.item() = 5.0f;
        b.item() = 3.0f;

        EXPECT_TRUE( a.isScalar() );
        EXPECT_TRUE( b.isScalar() );

        add( a, b, result, exec_ctx_.get() );

        EXPECT_TRUE( result.isScalar() );
        EXPECT_FLOAT_EQ( result.item(), 8.0f );
    }

    TEST_F( CpuTensorMathTest, Add_UsingOperator )
    {
        auto device_name = exec_ctx_->getDeviceName();

        auto a = makeCpuTensor<TensorDataType::FP32>( device_name, { 3 }, { 1.0f, 2.0f, 3.0f } );
        auto b = makeCpuTensor<TensorDataType::FP32>( device_name, { 3 }, { 4.0f, 5.0f, 6.0f } );

        // Operator automatically allocates result and uses default execution
        auto result = a + b;

        auto data = result.data();
        EXPECT_FLOAT_EQ( data[0], 5.0f );
        EXPECT_FLOAT_EQ( data[1], 7.0f );
        EXPECT_FLOAT_EQ( data[2], 9.0f );
    }

    // ============================================================================
    // Subtraction Tests
    // ============================================================================

    TEST_F( CpuTensorMathTest, Subtract_SameShape_Int32 )
    {
        auto device_name = exec_ctx_->getDeviceName();

        auto a = makeCpuTensor<TensorDataType::INT32>( device_name, { 2, 2 }, { 10, 20, 30, 40 } );
        auto b = makeCpuTensor<TensorDataType::INT32>( device_name, { 2, 2 }, { 1, 2, 3, 4 } );
        auto result = Tensor<TensorDataType::INT32, CpuMemoryResource>( device_name, { 2, 2 } );

        subtract( a, b, result, exec_ctx_.get() );

        EXPECT_EQ( result.shape(), a.shape() );
        auto data = result.data();
        EXPECT_EQ( data[0], 9 );
        EXPECT_EQ( data[1], 18 );
        EXPECT_EQ( data[2], 27 );
        EXPECT_EQ( data[3], 36 );
    }

    TEST_F( CpuTensorMathTest, Subtract_SameShape_Float )
    {
        auto device_name = exec_ctx_->getDeviceName();

        auto a = makeCpuTensor<TensorDataType::FP32>( device_name, { 3 }, { 5.5f, 10.0f, 15.5f } );
        auto b = makeCpuTensor<TensorDataType::FP32>( device_name, { 3 }, { 1.5f, 2.0f, 3.5f } );
        auto result = Tensor<TensorDataType::FP32, CpuMemoryResource>( device_name, { 3 } );

        subtract( a, b, result, exec_ctx_.get() );

        auto data = result.data();
        EXPECT_FLOAT_EQ( data[0], 4.0f );
        EXPECT_FLOAT_EQ( data[1], 8.0f );
        EXPECT_FLOAT_EQ( data[2], 12.0f );
    }

    TEST_F( CpuTensorMathTest, Subtract_Negative_Results )
    {
        auto device_name = exec_ctx_->getDeviceName();

        auto a = makeCpuTensor<TensorDataType::INT32>( device_name, { 2 }, { 5, 10 } );
        auto b = makeCpuTensor<TensorDataType::INT32>( device_name, { 2 }, { 10, 5 } );
        auto result = Tensor<TensorDataType::INT32, CpuMemoryResource>( device_name, { 2 } );

        subtract( a, b, result );  // Without context

        auto data = result.data();
        EXPECT_EQ( data[0], -5 );
        EXPECT_EQ( data[1], 5 );
    }

    TEST_F( CpuTensorMathTest, Subtract_UsingOperator )
    {
        auto device_name = exec_ctx_->getDeviceName();

        auto a = makeCpuTensor<TensorDataType::FP32>( device_name, { 2 }, { 10.0f, 20.0f } );
        auto b = makeCpuTensor<TensorDataType::FP32>( device_name, { 2 }, { 3.0f, 5.0f } );

        auto result = a - b;

        auto data = result.data();
        EXPECT_FLOAT_EQ( data[0], 7.0f );
        EXPECT_FLOAT_EQ( data[1], 15.0f );
    }

    // ============================================================================
    // Multiplication Tests
    // ============================================================================

    TEST_F( CpuTensorMathTest, Multiply_SameShape_Int32 )
    {
        auto device_name = exec_ctx_->getDeviceName();

        auto a = makeCpuTensor<TensorDataType::INT32>( device_name, { 2, 2 }, { 2, 3, 4, 5 } );
        auto b = makeCpuTensor<TensorDataType::INT32>( device_name, { 2, 2 }, { 10, 10, 10, 10 } );
        auto result = Tensor<TensorDataType::INT32, CpuMemoryResource>( device_name, { 2, 2 } );

        multiply( a, b, result, exec_ctx_.get() );

        auto data = result.data();
        EXPECT_EQ( data[0], 20 );
        EXPECT_EQ( data[1], 30 );
        EXPECT_EQ( data[2], 40 );
        EXPECT_EQ( data[3], 50 );
    }

    TEST_F( CpuTensorMathTest, Multiply_SameShape_Float )
    {
        auto device_name = exec_ctx_->getDeviceName();

        auto a = makeCpuTensor<TensorDataType::FP32>( device_name, { 3 }, { 2.5f, 3.0f, 4.5f } );
        auto b = makeCpuTensor<TensorDataType::FP32>( device_name, { 3 }, { 2.0f, 2.0f, 2.0f } );
        auto result = Tensor<TensorDataType::FP32, CpuMemoryResource>( device_name, { 3 } );

        multiply( a, b, result, exec_ctx_.get() );

        auto data = result.data();
        EXPECT_FLOAT_EQ( data[0], 5.0f );
        EXPECT_FLOAT_EQ( data[1], 6.0f );
        EXPECT_FLOAT_EQ( data[2], 9.0f );
    }

    TEST_F( CpuTensorMathTest, Multiply_WithZeros )
    {
        auto device_name = exec_ctx_->getDeviceName();

        auto a = makeCpuTensor<TensorDataType::INT32>( device_name, { 3 }, { 5, 0, 10 } );
        auto b = makeCpuTensor<TensorDataType::INT32>( device_name, { 3 }, { 2, 3, 0 } );
        auto result = Tensor<TensorDataType::INT32, CpuMemoryResource>( device_name, { 3 } );

        multiply( a, b, result );

        auto data = result.data();
        EXPECT_EQ( data[0], 10 );
        EXPECT_EQ( data[1], 0 );
        EXPECT_EQ( data[2], 0 );
    }

    TEST_F( CpuTensorMathTest, Multiply_UsingOperator )
    {
        auto device_name = exec_ctx_->getDeviceName();

        auto a = makeCpuTensor<TensorDataType::FP32>( device_name, { 2 }, { 3.0f, 4.0f } );
        auto b = makeCpuTensor<TensorDataType::FP32>( device_name, { 2 }, { 5.0f, 6.0f } );

        auto result = a * b;

        auto data = result.data();
        EXPECT_FLOAT_EQ( data[0], 15.0f );
        EXPECT_FLOAT_EQ( data[1], 24.0f );
    }

    // ============================================================================
    // Division Tests
    // ============================================================================

    TEST_F( CpuTensorMathTest, Divide_SameShape_Int32 )
    {
        auto device_name = exec_ctx_->getDeviceName();

        auto a = makeCpuTensor<TensorDataType::INT32>( device_name, { 3 }, { 20, 30, 40 } );
        auto b = makeCpuTensor<TensorDataType::INT32>( device_name, { 3 }, { 2, 3, 4 } );
        auto result = Tensor<TensorDataType::INT32, CpuMemoryResource>( device_name, { 3 } );

        divide( a, b, result, exec_ctx_.get() );

        auto data = result.data();
        EXPECT_EQ( data[0], 10 );
        EXPECT_EQ( data[1], 10 );
        EXPECT_EQ( data[2], 10 );
    }

    TEST_F( CpuTensorMathTest, Divide_SameShape_Float )
    {
        auto device_name = exec_ctx_->getDeviceName();

        auto a = makeCpuTensor<TensorDataType::FP32>( device_name, { 3 }, { 10.0f, 20.0f, 30.0f } );
        auto b = makeCpuTensor<TensorDataType::FP32>( device_name, { 3 }, { 2.0f, 4.0f, 5.0f } );
        auto result = Tensor<TensorDataType::FP32, CpuMemoryResource>( device_name, { 3 } );

        divide( a, b, result, exec_ctx_.get() );

        auto data = result.data();
        EXPECT_FLOAT_EQ( data[0], 5.0f );
        EXPECT_FLOAT_EQ( data[1], 5.0f );
        EXPECT_FLOAT_EQ( data[2], 6.0f );
    }

    TEST_F( CpuTensorMathTest, Divide_ByZero_Int32_Throws )
    {
        auto device_name = exec_ctx_->getDeviceName();

        auto a = makeCpuTensor<TensorDataType::INT32>( device_name, { 2 }, { 10, 20 } );
        auto b = makeCpuTensor<TensorDataType::INT32>( device_name, { 2 }, { 0, 5 } );
        auto result = Tensor<TensorDataType::INT32, CpuMemoryResource>( device_name, { 2 } );

        EXPECT_THROW( divide( a, b, result, exec_ctx_.get() ), std::runtime_error );
    }

    TEST_F( CpuTensorMathTest, Divide_ByZero_Float_InfNaN )
    {
        auto device_name = exec_ctx_->getDeviceName();

        auto a = makeCpuTensor<TensorDataType::FP32>( device_name, { 2 }, { 10.0f, 0.0f } );
        auto b = makeCpuTensor<TensorDataType::FP32>( device_name, { 2 }, { 0.0f, 0.0f } );
        auto result = Tensor<TensorDataType::FP32, CpuMemoryResource>( device_name, { 2 } );

        // Float division by zero follows IEEE 754
        divide( a, b, result );

        auto data = result.data();
        EXPECT_TRUE( std::isinf( data[0] ) );  // 10.0 / 0.0 = inf
        EXPECT_TRUE( std::isnan( data[1] ) );  // 0.0 / 0.0 = nan
    }

    TEST_F( CpuTensorMathTest, Divide_UsingOperator )
    {
        auto device_name = exec_ctx_->getDeviceName();

        auto a = makeCpuTensor<TensorDataType::FP32>( device_name, { 2 }, { 20.0f, 30.0f } );
        auto b = makeCpuTensor<TensorDataType::FP32>( device_name, { 2 }, { 4.0f, 5.0f } );

        auto result = a / b;

        auto data = result.data();
        EXPECT_FLOAT_EQ( data[0], 5.0f );
        EXPECT_FLOAT_EQ( data[1], 6.0f );
    }

    // ============================================================================
    // Multi-dimensional Tensor Tests
    // ============================================================================

    TEST_F( CpuTensorMathTest, Add_3D_Tensors )
    {
        auto device_name = exec_ctx_->getDeviceName();

        std::vector<float> values_a( 24 );
        std::vector<float> values_b( 24 );

        for (size_t i = 0; i < 24; ++i)
        {
            values_a[i] = static_cast<float>( i );
            values_b[i] = static_cast<float>( i * 2 );
        }

        auto a = makeCpuTensor<TensorDataType::FP32>( device_name, { 2, 3, 4 }, values_a );
        auto b = makeCpuTensor<TensorDataType::FP32>( device_name, { 2, 3, 4 }, values_b );
        auto result = Tensor<TensorDataType::FP32, CpuMemoryResource>( device_name, { 2, 3, 4 } );

        add( a, b, result, exec_ctx_.get() );

        EXPECT_EQ( result.shape(), a.shape() );
        EXPECT_EQ( result.size(), 24u );

        auto data = result.data();
        for (size_t i = 0; i < 24; ++i)
        {
            EXPECT_FLOAT_EQ( data[i], static_cast<float>( i * 3 ) );
        }
    }

    // ============================================================================
    // Edge Case Tests
    // ============================================================================

    TEST_F( CpuTensorMathTest, Operations_SingleElement )
    {
        auto device_name = exec_ctx_->getDeviceName();

        auto a = makeCpuTensor<TensorDataType::INT32>( device_name, { 1 }, { 10 } );
        auto b = makeCpuTensor<TensorDataType::INT32>( device_name, { 1 }, { 5 } );

        auto add_result = Tensor<TensorDataType::INT32, CpuMemoryResource>( device_name, { 1 } );
        auto sub_result = Tensor<TensorDataType::INT32, CpuMemoryResource>( device_name, { 1 } );
        auto mul_result = Tensor<TensorDataType::INT32, CpuMemoryResource>( device_name, { 1 } );
        auto div_result = Tensor<TensorDataType::INT32, CpuMemoryResource>( device_name, { 1 } );

        add( a, b, add_result, exec_ctx_.get() );
        subtract( a, b, sub_result, exec_ctx_.get() );
        multiply( a, b, mul_result, exec_ctx_.get() );
        divide( a, b, div_result, exec_ctx_.get() );

        EXPECT_EQ( add_result.data()[0], 15 );
        EXPECT_EQ( sub_result.data()[0], 5 );
        EXPECT_EQ( mul_result.data()[0], 50 );
        EXPECT_EQ( div_result.data()[0], 2 );
    }

    // ============================================================================
    // Chained Operations Test
    // ============================================================================

    TEST_F( CpuTensorMathTest, ChainedOperations_WithContext )
    {
        auto device_name = exec_ctx_->getDeviceName();

        auto a = makeCpuTensor<TensorDataType::FP32>( device_name, { 3 }, { 10.0f, 20.0f, 30.0f } );
        auto b = makeCpuTensor<TensorDataType::FP32>( device_name, { 3 }, { 2.0f, 3.0f, 4.0f } );
        auto c = makeCpuTensor<TensorDataType::FP32>( device_name, { 3 }, { 1.0f, 1.0f, 1.0f } );

        auto temp = Tensor<TensorDataType::FP32, CpuMemoryResource>( device_name, { 3 } );
        auto result = Tensor<TensorDataType::FP32, CpuMemoryResource>( device_name, { 3 } );

        // Compute: (a + b) * c
        add( a, b, temp, exec_ctx_.get() );
        multiply( temp, c, result, exec_ctx_.get() );

        auto data = result.data();
        EXPECT_FLOAT_EQ( data[0], 12.0f );
        EXPECT_FLOAT_EQ( data[1], 23.0f );
        EXPECT_FLOAT_EQ( data[2], 34.0f );
    }

    TEST_F( CpuTensorMathTest, ChainedOperations_UsingOperators )
    {
        auto device_name = exec_ctx_->getDeviceName();

        auto a = makeCpuTensor<TensorDataType::FP32>( device_name, { 2 }, { 5.0f, 10.0f } );
        auto b = makeCpuTensor<TensorDataType::FP32>( device_name, { 2 }, { 2.0f, 3.0f } );
        auto c = makeCpuTensor<TensorDataType::FP32>( device_name, { 2 }, { 1.0f, 1.0f } );

        // Compute: (a + b) * c - b
        auto result = (a + b) * c - b;

        auto data = result.data();
        EXPECT_FLOAT_EQ( data[0], 5.0f );   // (5+2)*1-2 = 5
        EXPECT_FLOAT_EQ( data[1], 10.0f );  // (10+3)*1-3 = 10
    }
}