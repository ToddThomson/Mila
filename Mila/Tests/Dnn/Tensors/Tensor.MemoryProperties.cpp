/**
 * @file Tensor.MemoryProperties.cpp
 * @brief CPU type-information / interface-compliance tests for Tensor.ixx.
 *
 * Area file of the Tensor value-type archetype (see Specifications/Testing.Tensors.md).
 * Covers the ITensor type-info surface — elementSize, getStorageSize, getDataType,
 * getDataTypeName, getDeviceType, getMemoryResource — and the host/device
 * accessibility statics, for CPU memory resources. The TensorDataTypeTraits
 * compile-time rows are data-type metadata (no memory resource), so they live here.
 * CUDA memory-resource accessibility and the sub-byte packed storage path are in
 * Tensor.MemoryProperties.Cuda.cpp.
 */

#include <gtest/gtest.h>

import Mila;

namespace Mila::Tests::Dnn::Tensors
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    class TensorMemoryPropertiesCpuTests : public testing::Test {};

    // ====================================================================
    // Element size & storage size
    // ====================================================================

    TEST_F( TensorMemoryPropertiesCpuTests, ElementSize_MatchesDataTypeWidth ) {
        Tensor<TensorDataType::FP32, CpuMemoryResource> fp32( Device::Cpu(), shape_t{ 2, 3 } );
        Tensor<TensorDataType::INT8, CpuMemoryResource> int8( Device::Cpu(), shape_t{ 2, 3 } );
        Tensor<TensorDataType::INT16, CpuMemoryResource> int16( Device::Cpu(), shape_t{ 2, 3 } );

        EXPECT_EQ( fp32.elementSize(), 4u );
        EXPECT_EQ( int8.elementSize(), 1u );
        EXPECT_EQ( int16.elementSize(), 2u );
    }

    TEST_F( TensorMemoryPropertiesCpuTests, StorageSize_IsElementCountTimesWidth ) {
        Tensor<TensorDataType::FP32, CpuMemoryResource> fp32( Device::Cpu(), shape_t{ 2, 3 } );
        EXPECT_EQ( fp32.getStorageSize(), 6u * 4u );

        Tensor<TensorDataType::INT16, CpuMemoryResource> int16( Device::Cpu(), shape_t{ 4 } );
        EXPECT_EQ( int16.getStorageSize(), 4u * 2u );
    }

    TEST_F( TensorMemoryPropertiesCpuTests, StorageSize_ZeroWhenNoBuffer ) {
        // A zero in the shape allocates no buffer -> zero storage.
        Tensor<TensorDataType::FP32, CpuMemoryResource> empty( Device::Cpu(), shape_t{ 0 } );
        EXPECT_EQ( empty.getStorageSize(), 0u );
    }

    // ====================================================================
    // Data type information
    // ====================================================================

    TEST_F( TensorMemoryPropertiesCpuTests, DataType_NameAndEnumForHostTypes ) {
        shape_t shape = { 2, 2 };

        Tensor<TensorDataType::FP32, CpuMemoryResource> fp32( Device::Cpu(), shape );
        Tensor<TensorDataType::INT8, CpuMemoryResource> int8( Device::Cpu(), shape );
        Tensor<TensorDataType::UINT32, CpuMemoryResource> uint32( Device::Cpu(), shape );

        EXPECT_EQ( fp32.getDataType(), TensorDataType::FP32 );
        EXPECT_EQ( fp32.getDataTypeName(), "FP32" );
        EXPECT_EQ( int8.getDataTypeName(), "INT8" );
        EXPECT_EQ( uint32.getDataTypeName(), "UINT32" );
    }

    // ====================================================================
    // Accessibility statics & device type (CPU)
    // ====================================================================

    TEST_F( TensorMemoryPropertiesCpuTests, Accessibility_CpuIsHostOnly ) {
        using CpuFp32 = Tensor<TensorDataType::FP32, CpuMemoryResource>;

        static_assert( CpuFp32::is_host_accessible() );
        static_assert( !CpuFp32::is_device_accessible() );

        CpuFp32 tensor( Device::Cpu(), shape_t{ 2, 2 } );
        EXPECT_TRUE( tensor.is_host_accessible() );
        EXPECT_FALSE( tensor.is_device_accessible() );
        EXPECT_EQ( tensor.getDeviceType(), DeviceType::Cpu );
    }

    TEST_F( TensorMemoryPropertiesCpuTests, MemoryResource_NonNullWithBuffer ) {
        Tensor<TensorDataType::FP32, CpuMemoryResource> tensor( Device::Cpu(), shape_t{ 2, 2 } );
        EXPECT_NE( tensor.getMemoryResource(), nullptr );

        Tensor<TensorDataType::FP32, CpuMemoryResource> empty( Device::Cpu(), shape_t{ 0 } );
        EXPECT_EQ( empty.getMemoryResource(), nullptr );
    }

    // ====================================================================
    // TensorDataTypeTraits (compile-time metadata, all data types)
    // ====================================================================

    TEST_F( TensorMemoryPropertiesCpuTests, DataTypeTraits_WidthAndCategory ) {
        static_assert( TensorDataTypeTraits<TensorDataType::FP32>::is_float_type );
        static_assert( !TensorDataTypeTraits<TensorDataType::FP32>::is_integer_type );
        static_assert( !TensorDataTypeTraits<TensorDataType::FP32>::is_device_only );
        static_assert( TensorDataTypeTraits<TensorDataType::FP32>::size_in_bytes == 4 );

        static_assert( TensorDataTypeTraits<TensorDataType::FP16>::is_device_only );
        static_assert( TensorDataTypeTraits<TensorDataType::FP16>::size_in_bytes == 2 );
        static_assert( TensorDataTypeTraits<TensorDataType::BF16>::is_device_only );
        static_assert( TensorDataTypeTraits<TensorDataType::BF16>::size_in_bytes == 2 );

        static_assert( TensorDataTypeTraits<TensorDataType::FP8_E4M3>::is_device_only );
        static_assert( TensorDataTypeTraits<TensorDataType::FP8_E4M3>::size_in_bytes == 1 );

        static_assert( TensorDataTypeTraits<TensorDataType::INT8>::is_integer_type );
        static_assert( !TensorDataTypeTraits<TensorDataType::INT8>::is_device_only );
        static_assert( TensorDataTypeTraits<TensorDataType::INT32>::size_in_bytes == 4 );

        SUCCEED();
    }
}
