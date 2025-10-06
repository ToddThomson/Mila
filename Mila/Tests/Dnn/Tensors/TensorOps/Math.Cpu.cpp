#include <gtest/gtest.h>
#include <memory>

import Mila;

namespace Dnn::Tensors::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    // Helper to create a CPU tensor with context and values
    template<TensorDataType TDataType>
    Tensor<TDataType, CpuMemoryResource> makeCpuTensor( 
        const std::vector<size_t>& shape, 
        const std::vector<typename TensorHostTypeMap<TDataType>::host_type>& values = {} )
    {
        auto cpu_context = std::make_shared<Compute::CpuDeviceContext>();
        Tensor<TDataType, CpuMemoryResource> tensor( cpu_context, shape );

        if (!values.empty() && !tensor.empty() && tensor.size() == values.size()) {
            auto data = tensor.data();
            for (size_t i = 0; i < values.size(); ++i) {
                data[i] = values[i];
            }
        }

        return tensor;
    }

    TEST( TensorOpsMathCpu, Add_SameShape_Int32 )
    {
        auto cpu_context = std::make_shared<CpuDeviceContext>();
        auto a = Tensor<TensorDataType::INT32, CpuMemoryResource>( cpu_context, { 2, 2 } );

        auto data_a = a.data();
        for (size_t i = 0; i < a.size(); ++i) {
            data_a[i] = static_cast<int32_t>( i + 1 );
        }

        auto b = Tensor<TensorDataType::INT32, CpuMemoryResource>( cpu_context, { 2, 2 } );
        auto data_b = b.data();
        for (size_t i = 0; i < b.size(); ++i) {
            data_b[i] = static_cast<int32_t>( i + 5 );
        }

        auto result = add( a, b );

        EXPECT_EQ( result.shape(), a.shape() );
        auto data_result = result.data();
        ASSERT_EQ( data_result[0], 6 );
        ASSERT_EQ( data_result[1], 8 );
        ASSERT_EQ( data_result[2], 10 );
        ASSERT_EQ( data_result[3], 12 );
    }

    TEST( TensorOpsMathCpu, Add_SameShape_Float )
    {
        auto a = makeCpuTensor<TensorDataType::FP32>( { 2, 2 }, { 1.5f, 2.5f, 3.5f, 4.5f } );
        auto b = makeCpuTensor<TensorDataType::FP32>( { 2, 2 }, { 0.5f, 1.5f, 2.5f, 3.5f } );

        auto result = add( a, b );

        EXPECT_EQ( result.shape(), a.shape() );
        auto data = result.data();
        ASSERT_FLOAT_EQ( data[0], 2.0f );
        ASSERT_FLOAT_EQ( data[1], 4.0f );
        ASSERT_FLOAT_EQ( data[2], 6.0f );
        ASSERT_FLOAT_EQ( data[3], 8.0f );
    }

    TEST( TensorOpsMathCpu, Add_DifferentShape_Throws )
    {
        auto a = makeCpuTensor<TensorDataType::INT32>( { 2, 2 }, { 1, 2, 3, 4 } );
        auto b = makeCpuTensor<TensorDataType::INT32>( { 2, 1 }, { 5, 6 } );

        EXPECT_THROW( add( a, b ), std::invalid_argument );
    }

    TEST( TensorOpsMathCpu, Add_Scalar_Tensors )
    {
        auto cpu_context = std::make_shared<CpuDeviceContext>();

        // Create scalar tensors (rank 0)
        auto a = Tensor<TensorDataType::FP32, CpuMemoryResource>( cpu_context, {} );
        auto b = Tensor<TensorDataType::FP32, CpuMemoryResource>( cpu_context, {} );

        a.item() = 5.0f;
        b.item() = 3.0f;

        EXPECT_TRUE( a.isScalar() );
        EXPECT_TRUE( b.isScalar() );

        auto result = add( a, b );

        EXPECT_TRUE( result.isScalar() );
        EXPECT_FLOAT_EQ( result.item(), 8.0f );
    }

    TEST( TensorOpsMathCpu, Add_Empty_Tensors )
    {
        auto a = makeCpuTensor<TensorDataType::FP32>( { 0 } );
        auto b = makeCpuTensor<TensorDataType::FP32>( { 0 } );

        EXPECT_TRUE( a.empty() );
        EXPECT_TRUE( b.empty() );

        auto result = add( a, b );

        EXPECT_TRUE( result.empty() );
        EXPECT_EQ( result.size(), 0u );
    }
}