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
    // rawData() Tests - Type-Erased Void Pointer Access
    // ========================================================================

    TEST( TensorDataPointers, RawData_NonConst_ReturnsValidPointer ) {
        Tensor<TensorDataType::FP32, CpuMemoryResource> t( "CPU", { 2, 3 } );

        void* raw = t.rawData();
        ASSERT_NE( raw, nullptr );

        // Verify we can write through raw pointer
        float* typed = static_cast<float*>(raw);
        typed[0] = 3.14f;
        typed[1] = 2.71f;

        // Verify data was written correctly
        auto data_ptr = t.data();
        EXPECT_FLOAT_EQ( data_ptr[0], 3.14f );
        EXPECT_FLOAT_EQ( data_ptr[1], 2.71f );
    }

    TEST( TensorDataPointers, RawData_Const_ReturnsValidPointer ) {
        Tensor<TensorDataType::INT32, CpuMemoryResource> t( "CPU", { 5 } );

        // Fill with data first
        auto data_ptr = t.data();
        for (size_t i = 0; i < t.size(); ++i) {
            data_ptr[i] = static_cast<int32_t>( i * 10 );
        }

        // Test const version
        const auto& const_t = t;
        const void* raw = const_t.rawData();
        ASSERT_NE( raw, nullptr );

        // Verify we can read through const raw pointer
        const int32_t* typed = static_cast<const int32_t*>( raw );
        for (size_t i = 0; i < t.size(); ++i) {
            EXPECT_EQ( typed[i], static_cast<int32_t>( i * 10 ) );
        }
    }

    TEST( TensorDataPointers, RawData_EmptyTensor_BehavesCorrectly ) {
        Tensor<TensorDataType::FP32, CpuMemoryResource> t( "CPU", std::vector<size_t>{} );
        ASSERT_EQ( t.size(), 1u );

        // rawData() behavior for empty tensors is implementation-defined
        // (may be nullptr or valid pointer to zero-sized allocation)
        // Just verify it doesn't crash
        void* raw = t.rawData();
        (void)raw; // Suppress unused warning

        const auto& const_t = t;
        const void* const_raw = const_t.rawData();
        (void)const_raw; // Suppress unused warning
    }

    TEST( TensorDataPointers, RawData_DifferentDataTypes_ReturnsSamePointer ) {
        // rawData() should return the same underlying pointer regardless of type
        Tensor<TensorDataType::INT32, CpuMemoryResource> t( "CPU", { 10 } );

        void* raw1 = t.rawData();
        void* raw2 = t.rawData();

        EXPECT_EQ( raw1, raw2 );
    }

    TEST( TensorDataPointers, RawData_RequiresManualTypeCasting ) {
        Tensor<TensorDataType::UINT8, CpuMemoryResource> t( "CPU", { 4 } );

        void* raw = t.rawData();
        ASSERT_NE( raw, nullptr );

        // Manual type casting is required - no type safety
        uint8_t* typed = static_cast<uint8_t*>(raw);
        typed[0] = 255;
        typed[1] = 128;
        typed[2] = 64;
        typed[3] = 0;

        // Verify via data()
        auto data_ptr = t.data();
        EXPECT_EQ( data_ptr[0], 255 );
        EXPECT_EQ( data_ptr[1], 128 );
        EXPECT_EQ( data_ptr[2], 64 );
        EXPECT_EQ( data_ptr[3], 0 );
    }

    // ========================================================================
    // data() Tests - Type-Safe TensorPtr Access (Host-Accessible Only)
    // ========================================================================

    TEST( TensorDataPointers, Data_ReturnsTypedPointer_FP32 ) {
        Tensor<TensorDataType::FP32, CpuMemoryResource> t( "CPU", { 3, 2 } );

        auto data_ptr = t.data();

        // Verify type deduction works correctly
        static_assert(std::is_same_v<decltype(data_ptr), TensorPtr<float>>);

        // Verify we can index and write
        data_ptr[0] = 1.0f;
        data_ptr[1] = 2.0f;
        data_ptr[2] = 3.0f;

        // Verify values
        EXPECT_FLOAT_EQ( data_ptr[0], 1.0f );
        EXPECT_FLOAT_EQ( data_ptr[1], 2.0f );
        EXPECT_FLOAT_EQ( data_ptr[2], 3.0f );
    }

    TEST( TensorDataPointers, Data_ReturnsTypedPointer_INT32 ) {
        Tensor<TensorDataType::INT32, CpuMemoryResource> t( "CPU", { 5 } );

        auto data_ptr = t.data();

        // Verify type deduction
        static_assert(std::is_same_v<decltype(data_ptr), TensorPtr<int32_t>>);

        for (size_t i = 0; i < t.size(); ++i) {
            data_ptr[i] = static_cast<int32_t>( i * 100 );
        }

        for (size_t i = 0; i < t.size(); ++i) {
            EXPECT_EQ( data_ptr[i], static_cast<int32_t>( i * 100 ) );
        }
    }

    TEST( TensorDataPointers, Data_ReturnsTypedPointer_INT8 ) {
        Tensor<TensorDataType::INT8, CpuMemoryResource> t( "CPU", { 4 } );

        auto data_ptr = t.data();

        // Verify type deduction
        static_assert(std::is_same_v<decltype(data_ptr), TensorPtr<int8_t>>);

        data_ptr[0] = -128;
        data_ptr[1] = -1;
        data_ptr[2] = 0;
        data_ptr[3] = 127;

        EXPECT_EQ( data_ptr[0], -128 );
        EXPECT_EQ( data_ptr[1], -1 );
        EXPECT_EQ( data_ptr[2], 0 );
        EXPECT_EQ( data_ptr[3], 127 );
    }

    TEST( TensorDataPointers, Data_ReturnsTypedPointer_UINT8 ) {
        Tensor<TensorDataType::UINT8, CpuMemoryResource> t( "CPU", { 3 } );

        auto data_ptr = t.data();

        // Verify type deduction
        static_assert(std::is_same_v<decltype(data_ptr), TensorPtr<uint8_t>>);

        data_ptr[0] = 0;
        data_ptr[1] = 128;
        data_ptr[2] = 255;

        EXPECT_EQ( data_ptr[0], 0 );
        EXPECT_EQ( data_ptr[1], 128 );
        EXPECT_EQ( data_ptr[2], 255 );
    }

    TEST( TensorDataPointers, Data_Const_ReturnsConstTypedPointer ) {
        Tensor<TensorDataType::FP32, CpuMemoryResource> t( "CPU", { 4 } );

        // Fill with data first
        auto data_ptr = t.data();
        data_ptr[0] = 1.0f;
        data_ptr[1] = 2.0f;
        data_ptr[2] = 3.0f;
        data_ptr[3] = 4.0f;

        // Test const version
        const auto& const_t = t;
        auto const_data_ptr = const_t.data();

        // Verify const type deduction
        static_assert(std::is_same_v<decltype(const_data_ptr), TensorPtr<const float>>);

        // Can read but not write (enforced at compile-time)
        EXPECT_FLOAT_EQ( const_data_ptr[0], 1.0f );
        EXPECT_FLOAT_EQ( const_data_ptr[1], 2.0f );
        EXPECT_FLOAT_EQ( const_data_ptr[2], 3.0f );
        EXPECT_FLOAT_EQ( const_data_ptr[3], 4.0f );

        // This would fail to compile (const correctness):
        // const_data_ptr[0] = 5.0f; // ERROR: assignment to const
    }

    TEST( TensorDataPointers, Data_PointerArithmetic_Works ) {
        Tensor<TensorDataType::INT32, CpuMemoryResource> t( "CPU", { 10 } );

        auto data_ptr = t.data();

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

        auto data_ptr = t.data();

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

    // ========================================================================
    // Interoperability Tests - data() vs rawData()
    // ========================================================================

    TEST( TensorDataPointers, DataAndRawData_PointToSameMemory ) {
        Tensor<TensorDataType::FP32, CpuMemoryResource> t( "CPU", { 6 } );

        void* raw = t.rawData();
        auto data_ptr = t.data();

        // They should point to the same underlying memory
        EXPECT_EQ( raw, static_cast<void*>(&data_ptr[0]) );

        // Write through data(), read through rawData()
        data_ptr[0] = 42.0f;
        float* typed_raw = static_cast<float*>(raw);
        EXPECT_FLOAT_EQ( typed_raw[0], 42.0f );

        // Write through rawData(), read through data()
        typed_raw[1] = 99.0f;
        EXPECT_FLOAT_EQ( data_ptr[1], 99.0f );
    }

    TEST( TensorDataPointers, DataAndRawData_SamePointerValue ) {
        Tensor<TensorDataType::INT32, CpuMemoryResource> t( "CPU", { 8 } );

        void* raw = t.rawData();
        auto data_ptr = t.data();

        // Cast to uintptr_t to compare addresses
        EXPECT_EQ( reinterpret_cast<uintptr_t>(raw),
            reinterpret_cast<uintptr_t>(static_cast<int32_t*>(data_ptr)) );
    }

    // ========================================================================
    // Multi-Dimensional Tensor Tests
    // ========================================================================

    TEST( TensorDataPointers, Data_MultiDimensional_RowMajorLayout ) {
        // Create 2x3 matrix
        Tensor<TensorDataType::INT32, CpuMemoryResource> t( "CPU", { 2, 3 } );
        ASSERT_EQ( t.size(), 6u );

        auto data_ptr = t.data();

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

    TEST( TensorDataPointers, RawData_MultiDimensional_RowMajorLayout ) {
        // Create 3x2 matrix
        Tensor<TensorDataType::FP32, CpuMemoryResource> t( "CPU", { 3, 2 } );
        ASSERT_EQ( t.size(), 6u );

        void* raw = t.rawData();
        float* typed = static_cast<float*>(raw);

        // Fill in row-major order
        float value = 1.0f;
        for (size_t i = 0; i < t.size(); ++i) {
            typed[i] = value;
            value += 1.0f;
        }

        // Verify layout: [1,2, 3,4, 5,6]
        EXPECT_FLOAT_EQ( typed[0], 1.0f ); // [0,0]
        EXPECT_FLOAT_EQ( typed[1], 2.0f ); // [0,1]
        EXPECT_FLOAT_EQ( typed[2], 3.0f ); // [1,0]
        EXPECT_FLOAT_EQ( typed[3], 4.0f ); // [1,1]
        EXPECT_FLOAT_EQ( typed[4], 5.0f ); // [2,0]
        EXPECT_FLOAT_EQ( typed[5], 6.0f ); // [2,1]
    }

    // ========================================================================
    // Type Safety Comparison Tests
    // ========================================================================

    TEST( TensorDataPointers, Data_TypeSafety_CompileTimeChecking ) {
        // data() provides compile-time type safety
        Tensor<TensorDataType::FP32, CpuMemoryResource> fp32_tensor( "CPU", { 5 } );
        Tensor<TensorDataType::INT32, CpuMemoryResource> int32_tensor( "CPU", { 5 } );

        auto fp32_data = fp32_tensor.data();
        auto int32_data = int32_tensor.data();

        // Types are different at compile time
        static_assert(!std::is_same_v<decltype(fp32_data), decltype(int32_data)>);

        // Correct types deduced
        static_assert(std::is_same_v<decltype(fp32_data), TensorPtr<float>>);
        static_assert(std::is_same_v<decltype(int32_data), TensorPtr<int32_t>>);
    }

    TEST( TensorDataPointers, RawData_NoTypeSafety_RuntimeOnly ) {
        // rawData() returns void* with no type information
        Tensor<TensorDataType::FP32, CpuMemoryResource> fp32_tensor( "CPU", { 5 } );
        Tensor<TensorDataType::INT32, CpuMemoryResource> int32_tensor( "CPU", { 5 } );

        void* fp32_raw = fp32_tensor.rawData();
        void* int32_raw = int32_tensor.rawData();

        // Same type at compile time - no type safety!
        static_assert(std::is_same_v<decltype(fp32_raw), decltype(int32_raw)>);

        // Both are void*
        static_assert(std::is_same_v<decltype(fp32_raw), void*>);
        static_assert(std::is_same_v<decltype(int32_raw), void*>);
    }

    // ========================================================================
    // Usage Pattern Tests
    // ========================================================================

    TEST( TensorDataPointers, Data_PreferredForHostAccessibleMemory ) {
        // data() is the preferred method for host-accessible tensors
        Tensor<TensorDataType::FP32, CpuMemoryResource> t( "CPU", { 10 } );

        // Type-safe, no casting needed
        auto data_ptr = t.data();
        for (size_t i = 0; i < t.size(); ++i) {
            data_ptr[i] = static_cast<float>( i ) * 1.5f;
        }

        // Verify
        for (size_t i = 0; i < t.size(); ++i) {
            EXPECT_FLOAT_EQ( data_ptr[i], static_cast<float>( i ) * 1.5f );
        }
    }

    TEST( TensorDataPointers, RawData_UsedForLowLevelInterop ) {
        // rawData() is useful for interfacing with C APIs
        Tensor<TensorDataType::INT32, CpuMemoryResource> t( "CPU", { 5 } );

        // Simulate C API that takes void*
        auto c_api_fill = []( void* buffer, size_t count, int32_t value ) {
            int32_t* typed = static_cast<int32_t*>(buffer);
            for (size_t i = 0; i < count; ++i) {
                typed[i] = value;
            }
            };

        c_api_fill( t.rawData(), t.size(), 777 );

        // Verify via data()
        auto data_ptr = t.data();
        for (size_t i = 0; i < t.size(); ++i) {
            EXPECT_EQ( data_ptr[i], 777 );
        }
    }

    // ========================================================================
    // Edge Cases and Special Scenarios
    // ========================================================================

    TEST( TensorDataPointers, Data_SingleElement_Tensor ) {
        Tensor<TensorDataType::FP32, CpuMemoryResource> t( "CPU", { 1 } );

        auto data_ptr = t.data();
        data_ptr[0] = 3.14159f;

        EXPECT_FLOAT_EQ( data_ptr[0], 3.14159f );
    }

    TEST( TensorDataPointers, RawData_SingleElement_Tensor ) {
        Tensor<TensorDataType::INT32, CpuMemoryResource> t( "CPU", { 1 } );

        void* raw = t.rawData();
        int32_t* typed = static_cast<int32_t*>(raw);
        typed[0] = 42;

        EXPECT_EQ( typed[0], 42 );
    }

    TEST( TensorDataPointers, Data_LargeTensor_Performance ) {
        // Verify data() works efficiently with large tensors
        Tensor<TensorDataType::FP32, CpuMemoryResource> t( "CPU", { 1000, 1000 } );
        ASSERT_EQ( t.size(), 1000000u );

        auto data_ptr = t.data();

        // Fill large tensor
        for (size_t i = 0; i < t.size(); ++i) {
            data_ptr[i] = static_cast<float>( i % 100 );
        }

        // Spot check
        EXPECT_FLOAT_EQ( data_ptr[0], 0.0f );
        EXPECT_FLOAT_EQ( data_ptr[50], 50.0f );
        EXPECT_FLOAT_EQ( data_ptr[100], 0.0f );
    }
}