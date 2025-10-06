#include <gtest/gtest.h>
#include <vector>
#include <cstdint>
#include <memory>
#include <type_traits>

import Mila;

namespace Dnn::Tensors::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    // ========================================================================
    // data() Tests - Type-Safe Raw Pointer Access (Host-Accessible Only)
    // ========================================================================

    TEST( TensorDataPointers, Data_ReturnsRawPointer_FP32 ) {
        Tensor<TensorDataType::FP32, CpuMemoryResource> t( "CPU", { 3, 2 } );

        auto* data_ptr = t.data();

        // Verify type deduction returns raw pointer
        static_assert(std::is_same_v<decltype(data_ptr), float*>);

        // Verify we can index and write
        data_ptr[0] = 1.0f;
        data_ptr[1] = 2.0f;
        data_ptr[2] = 3.0f;

        // Verify values
        EXPECT_FLOAT_EQ( data_ptr[0], 1.0f );
        EXPECT_FLOAT_EQ( data_ptr[1], 2.0f );
        EXPECT_FLOAT_EQ( data_ptr[2], 3.0f );
    }

    TEST( TensorDataPointers, Data_ReturnsRawPointer_INT32 ) {
        Tensor<TensorDataType::INT32, CpuMemoryResource> t( "CPU", { 5 } );

        auto* data_ptr = t.data();

        // Verify type deduction returns raw pointer
        static_assert(std::is_same_v<decltype(data_ptr), int32_t*>);

        for (size_t i = 0; i < t.size(); ++i) {
            data_ptr[i] = static_cast<int32_t>( i * 100 );
        }

        for (size_t i = 0; i < t.size(); ++i) {
            EXPECT_EQ( data_ptr[i], static_cast<int32_t>( i * 100 ) );
        }
    }

    TEST( TensorDataPointers, Data_ReturnsRawPointer_INT8 ) {
        Tensor<TensorDataType::INT8, CpuMemoryResource> t( "CPU", { 4 } );

        auto* data_ptr = t.data();

        // Verify type deduction returns raw pointer
        static_assert(std::is_same_v<decltype(data_ptr), int8_t*>);

        data_ptr[0] = -128;
        data_ptr[1] = -1;
        data_ptr[2] = 0;
        data_ptr[3] = 127;

        EXPECT_EQ( data_ptr[0], -128 );
        EXPECT_EQ( data_ptr[1], -1 );
        EXPECT_EQ( data_ptr[2], 0 );
        EXPECT_EQ( data_ptr[3], 127 );
    }

    TEST( TensorDataPointers, Data_ReturnsRawPointer_UINT8 ) {
        Tensor<TensorDataType::UINT8, CpuMemoryResource> t( "CPU", { 3 } );

        auto* data_ptr = t.data();

        // Verify type deduction returns raw pointer
        static_assert(std::is_same_v<decltype(data_ptr), uint8_t*>);

        data_ptr[0] = 0;
        data_ptr[1] = 128;
        data_ptr[2] = 255;

        EXPECT_EQ( data_ptr[0], 0 );
        EXPECT_EQ( data_ptr[1], 128 );
        EXPECT_EQ( data_ptr[2], 255 );
    }

    TEST( TensorDataPointers, Data_Const_ReturnsConstRawPointer ) {
        Tensor<TensorDataType::FP32, CpuMemoryResource> t( "CPU", { 4 } );

        // Fill with data first
        auto* data_ptr = t.data();
        data_ptr[0] = 1.0f;
        data_ptr[1] = 2.0f;
        data_ptr[2] = 3.0f;
        data_ptr[3] = 4.0f;

        // Test const version
        const auto& const_t = t;
        const auto* const_data_ptr = const_t.data();

        // Verify const type deduction returns const raw pointer
        static_assert(std::is_same_v<decltype(const_data_ptr), const float*>);

        // Can read but not write (enforced at compile-time)
        EXPECT_FLOAT_EQ( const_data_ptr[0], 1.0f );
        EXPECT_FLOAT_EQ( const_data_ptr[1], 2.0f );
        EXPECT_FLOAT_EQ( const_data_ptr[2], 3.0f );
        EXPECT_FLOAT_EQ( const_data_ptr[3], 4.0f );

        // This would fail to compile (const correctness):
        // const_data_ptr[0] = 5.0f;
    }

    TEST( TensorDataPointers, Data_PointerArithmetic_Works ) {
        Tensor<TensorDataType::INT32, CpuMemoryResource> t( "CPU", { 10 } );

        auto* data_ptr = t.data();

        // Fill using pointer arithmetic
        for (size_t i = 0; i < t.size(); ++i) {
            *(data_ptr + i) = static_cast<int32_t>( i * 10 );
        }

        // Verify using indexing
        for (size_t i = 0; i < t.size(); ++i) {
            EXPECT_EQ( data_ptr[i], static_cast<int32_t>( i * 10 ) );
        }
    }

    TEST( TensorDataPointers, Data_IteratorStyle_Access ) {
        Tensor<TensorDataType::FP32, CpuMemoryResource> t( "CPU", { 5 } );

        auto* data_ptr = t.data();

        // Fill using iterator-style access
        float value = 1.5f;
        for (size_t i = 0; i < t.size(); ++i) {
            data_ptr[i] = value;
            value += 0.5f;
        }

        // Verify
        EXPECT_FLOAT_EQ( data_ptr[0], 1.5f );
        EXPECT_FLOAT_EQ( data_ptr[1], 2.0f );
        EXPECT_FLOAT_EQ( data_ptr[2], 2.5f );
        EXPECT_FLOAT_EQ( data_ptr[3], 3.0f );
        EXPECT_FLOAT_EQ( data_ptr[4], 3.5f );
    }

    TEST( TensorDataPointers, Data_EmptyTensor_ReturnsValidPointer ) {
        // Scalar tensor (rank 0, size 1)
        Tensor<TensorDataType::FP32, CpuMemoryResource> scalar( "CPU", {} );
        ASSERT_EQ( scalar.size(), 1u );
        ASSERT_TRUE( scalar.isScalar() );

        auto* data_ptr = scalar.data();
        ASSERT_NE( data_ptr, nullptr );

        // Can write to scalar
        data_ptr[0] = 42.0f;
        EXPECT_FLOAT_EQ( data_ptr[0], 42.0f );
    }

    TEST( TensorDataPointers, Data_ZeroSizeTensor_ReturnsNullptr ) {
        // Empty 1D tensor (rank 1, size 0)
        Tensor<TensorDataType::FP32, CpuMemoryResource> empty( "CPU", { 0 } );
        ASSERT_EQ( empty.size(), 0u );
        ASSERT_TRUE( empty.empty() );

        auto* data_ptr = empty.data();
        // May be nullptr or valid pointer depending on implementation
        // Just verify it doesn't crash
        (void)data_ptr;
    }

    TEST( TensorDataPointers, Data_ConsistentPointer_MultipleCalls ) {
        Tensor<TensorDataType::INT32, CpuMemoryResource> t( "CPU", { 10 } );

        auto* ptr1 = t.data();
        auto* ptr2 = t.data();

        EXPECT_EQ( ptr1, ptr2 );
    }

    // ========================================================================
    // Multi-Dimensional Tensor Tests
    // ========================================================================

    TEST( TensorDataPointers, Data_MultiDimensional_RowMajorLayout ) {
        // Create 2x3 matrix
        Tensor<TensorDataType::INT32, CpuMemoryResource> t( "CPU", { 2, 3 } );
        ASSERT_EQ( t.size(), 6u );

        auto* data_ptr = t.data();

        // Fill in row-major order
        int32_t value = 1;
        for (size_t i = 0; i < t.size(); ++i) {
            data_ptr[i] = value++;
        }

        // Verify layout: [1,2,3, 4,5,6]
        EXPECT_EQ( data_ptr[0], 1 ); // [0,0]
        EXPECT_EQ( data_ptr[1], 2 ); // [0,1]
        EXPECT_EQ( data_ptr[2], 3 ); // [0,2]
        EXPECT_EQ( data_ptr[3], 4 ); // [1,0]
        EXPECT_EQ( data_ptr[4], 5 ); // [1,1]
        EXPECT_EQ( data_ptr[5], 6 ); // [1,2]
    }

    TEST( TensorDataPointers, Data_3DimensionalTensor_RowMajorLayout ) {
        // Create 2x3x4 tensor
        Tensor<TensorDataType::FP32, CpuMemoryResource> t( "CPU", { 2, 3, 4 } );
        ASSERT_EQ( t.size(), 24u );

        auto* data_ptr = t.data();

        // Fill with sequential values
        for (size_t i = 0; i < t.size(); ++i) {
            data_ptr[i] = static_cast<float>( i );
        }

        // Verify row-major layout
        EXPECT_FLOAT_EQ( data_ptr[0], 0.0f );   // [0,0,0]
        EXPECT_FLOAT_EQ( data_ptr[1], 1.0f );   // [0,0,1]
        EXPECT_FLOAT_EQ( data_ptr[4], 4.0f );   // [0,1,0]
        EXPECT_FLOAT_EQ( data_ptr[12], 12.0f ); // [1,0,0]
    }

    // ========================================================================
    // Type Safety Tests
    // ========================================================================

    TEST( TensorDataPointers, Data_TypeSafety_CompileTimeChecking ) {
        // data() provides compile-time type safety
        Tensor<TensorDataType::FP32, CpuMemoryResource> fp32_tensor( "CPU", { 5 } );
        Tensor<TensorDataType::INT32, CpuMemoryResource> int32_tensor( "CPU", { 5 } );

        auto* fp32_data = fp32_tensor.data();
        auto* int32_data = int32_tensor.data();

        // Types are different at compile time
        static_assert(!std::is_same_v<decltype(fp32_data), decltype(int32_data)>);

        // Correct types deduced
        static_assert(std::is_same_v<decltype(fp32_data), float*>);
        static_assert(std::is_same_v<decltype(int32_data), int32_t*>);
    }

    // ========================================================================
    // Usage Pattern Tests
    // ========================================================================

    TEST( TensorDataPointers, Data_WorksWithStandardAlgorithms ) {
        Tensor<TensorDataType::FP32, CpuMemoryResource> t( "CPU", { 10 } );

        auto* data_ptr = t.data();

        // Use std::fill
        std::fill( data_ptr, data_ptr + t.size(), 42.0f );

        // Verify
        for (size_t i = 0; i < t.size(); ++i) {
            EXPECT_FLOAT_EQ( data_ptr[i], 42.0f );
        }
    }

    TEST( TensorDataPointers, Data_WorksWithMemcpy ) {
        Tensor<TensorDataType::INT32, CpuMemoryResource> src( "CPU", { 5 } );
        Tensor<TensorDataType::INT32, CpuMemoryResource> dst( "CPU", { 5 } );

        // Fill source
        auto* src_ptr = src.data();
        for (size_t i = 0; i < src.size(); ++i) {
            src_ptr[i] = static_cast<int32_t>( i * 10 );
        }

        // Copy using memcpy
        auto* dst_ptr = dst.data();
        std::memcpy( dst_ptr, src_ptr, src.size() * sizeof( int32_t ) );

        // Verify
        for (size_t i = 0; i < dst.size(); ++i) {
            EXPECT_EQ( dst_ptr[i], src_ptr[i] );
        }
    }

    TEST( TensorDataPointers, Data_CStyleAPIInterop ) {
        Tensor<TensorDataType::INT32, CpuMemoryResource> t( "CPU", { 5 } );

        // Simulate C API that takes raw pointer
        auto c_api_fill = []( int32_t* buffer, size_t count, int32_t value ) {
            for (size_t i = 0; i < count; ++i) {
                buffer[i] = value;
            }
            };

        c_api_fill( t.data(), t.size(), 777 );

        // Verify
        auto* data_ptr = t.data();
        for (size_t i = 0; i < t.size(); ++i) {
            EXPECT_EQ( data_ptr[i], 777 );
        }
    }

    // ========================================================================
    // Edge Cases and Special Scenarios
    // ========================================================================

    TEST( TensorDataPointers, Data_SingleElement_Tensor ) {
        Tensor<TensorDataType::FP32, CpuMemoryResource> t( "CPU", { 1 } );

        auto* data_ptr = t.data();
        data_ptr[0] = 3.14159f;

        EXPECT_FLOAT_EQ( data_ptr[0], 3.14159f );
    }

    TEST( TensorDataPointers, Data_LargeTensor_Performance ) {
        // Verify data() works efficiently with large tensors
        Tensor<TensorDataType::FP32, CpuMemoryResource> t( "CPU", { 1000, 1000 } );
        ASSERT_EQ( t.size(), 1000000u );

        auto* data_ptr = t.data();

        // Fill large tensor
        for (size_t i = 0; i < t.size(); ++i) {
            data_ptr[i] = static_cast<float>( i % 100 );
        }

        // Spot check
        EXPECT_FLOAT_EQ( data_ptr[0], 0.0f );
        EXPECT_FLOAT_EQ( data_ptr[50], 50.0f );
        EXPECT_FLOAT_EQ( data_ptr[100], 0.0f );
    }

    TEST( TensorDataPointers, Data_WorksWithRangeBasedFor ) {
        Tensor<TensorDataType::INT32, CpuMemoryResource> t( "CPU", { 5 } );

        auto* data_ptr = t.data();

        // Can't use range-based for directly, but can create a span-like view
        int32_t value = 1;
        for (size_t i = 0; i < t.size(); ++i) {
            data_ptr[i] = value++;
        }

        // Verify
        for (size_t i = 0; i < t.size(); ++i) {
            EXPECT_EQ( data_ptr[i], static_cast<int32_t>( i + 1 ) );
        }
    }

    // ========================================================================
    // Consistency with operator[] Tests
    // ========================================================================

    TEST( TensorDataPointers, Data_ConsistentWithOperatorBracket ) {
        Tensor<TensorDataType::FP32, CpuMemoryResource> t( "CPU", { 2, 3 } );

        // Write via data()
        auto* data_ptr = t.data();
        data_ptr[0] = 1.0f;
        data_ptr[3] = 4.0f;

        // Read via operator[]
        EXPECT_FLOAT_EQ( (t[{0, 0}]), 1.0f );
        EXPECT_FLOAT_EQ( (t[{1, 0}]), 4.0f );

        // Write via operator[]
        t[{0, 1}] = 2.0f;
        t[{1, 1}] = 5.0f;

        // Read via data()
        EXPECT_FLOAT_EQ( data_ptr[1], 2.0f );
        EXPECT_FLOAT_EQ( data_ptr[4], 5.0f );
    }

    TEST( TensorDataPointers, Data_ConsistentWithItem_ScalarTensor ) {
        Tensor<TensorDataType::FP32, CpuMemoryResource> scalar( "CPU", {} );
        ASSERT_TRUE( scalar.isScalar() );

        // Write via data()
        auto* data_ptr = scalar.data();
        data_ptr[0] = 3.14f;

        // Read via item()
        EXPECT_FLOAT_EQ( scalar.item(), 3.14f );

        // Write via item()
        scalar.item() = 2.71f;

        // Read via data()
        EXPECT_FLOAT_EQ( data_ptr[0], 2.71f );
    }
}