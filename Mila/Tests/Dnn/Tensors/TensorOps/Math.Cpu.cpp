#include <gtest/gtest.h>
#include <memory>

import Mila;

namespace Dnn::Tensors::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    // Helper to create a CPU tensor with context and values
    template<TensorDataType T>
    Tensor<T, CpuMemoryResource> makeCpuTensor( const std::vector<size_t>& shape/*, const std::vector<T>& values*/ )
    {
        auto cpu_context = std::make_shared<Compute::CpuDeviceContext>();
        Tensor<T, CpuMemoryResource> tensor( cpu_context, shape );
        
        /*if (!tensor.empty() && tensor.size() == values.size()) {
            T* data = static_cast<T*>(tensor.rawData());
            for (size_t i = 0; i < values.size(); ++i)
                data[i] = values[i];
        }*/
        
        return tensor;
    }

    TEST( TensorOpsMathCpu, Add_SameShape_Int32 )
    {
        auto cpu_context_ = std::make_shared<CpuDeviceContext>();
		auto a = Tensor<TensorDataType::INT32, CpuMemoryResource>( cpu_context_, { 2, 2 } );
        
        auto data_a = static_cast<int32_t*>(a.rawData());
        for (size_t i = 0; i < a.size(); ++i)
            data_a[i] = 22;


		auto b = Tensor<TensorDataType::INT32, CpuMemoryResource>( cpu_context_, { 2, 2 } );

        auto result = add( a, b );

        //EXPECT_EQ( result.shape(), a.shape() );
        //auto* data = static_cast<const int32_t*>(result.rawData());
        //ASSERT_EQ( data[0], 6 );
        //ASSERT_EQ( data[1], 8 );
        //ASSERT_EQ( data[2], 10 );
        //ASSERT_EQ( data[3], 12 );
    }

    /*TEST( TensorOpsMathCpu, Add_SameShape_Float )
    {
        auto a = makeCpuTensor<float>( { 2, 2 }, { 1.5f, 2.5f, 3.5f, 4.5f } );
        auto b = makeCpuTensor<float>( { 2, 2 }, { 0.5f, 1.5f, 2.5f, 3.5f } );

        auto result = add( a, b );

        EXPECT_EQ( result.shape(), a.shape() );
        auto* data = static_cast<const float*>(result.rawData());
        ASSERT_FLOAT_EQ( data[0], 2.0f );
        ASSERT_FLOAT_EQ( data[1], 4.0f );
        ASSERT_FLOAT_EQ( data[2], 6.0f );
        ASSERT_FLOAT_EQ( data[3], 8.0f );
    }

    TEST( TensorOpsMathCpu, Add_DifferentShape_Throws )
    {
        auto a = makeCpuTensor<int32_t>( { 2, 2 }, { 1, 2, 3, 4 } );
        auto b = makeCpuTensor<int32_t>( { 2, 1 }, { 5, 6 } );

        EXPECT_THROW( add( a, b ), std::invalid_argument );
    }*/
}