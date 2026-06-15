/**
 * @file Tensor.Constructors.cpp
 * @brief CPU construction/lifetime tests for Tensor<TDataType, CpuMemoryResource>.
 *
 * Area file of the Tensor value-type / god-module archetype (see
 * Specifications/Testing.Tensors.md). Covers the construction, move, deleted-copy,
 * unique-id, and CPU data-type surface of Tensor.ixx. CPU memory resource only, so
 * this rides the MILA_ENABLE_CUDA=OFF CI gate; device-tensor construction
 * (DeviceTensor/Pinned/Universal), the device/memory-resource mismatch throw, and
 * the cross-resource accessibility checks live in Tensor.Constructors.Cuda.cpp.
 */

#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <type_traits>

import Mila;

namespace Mila::Tests::Dnn::Tensors
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace
    {
        template<TensorDataType TDataType>
        using CpuTensor = Tensor<TDataType, CpuMemoryResource>;

        // Copy is deleted by design (move-only); clone()/move are the supported paths.
        static_assert( !std::is_copy_constructible_v<CpuTensor<TensorDataType::FP32>> );
        static_assert( !std::is_copy_assignable_v<CpuTensor<TensorDataType::FP32>> );
        static_assert( std::is_move_constructible_v<CpuTensor<TensorDataType::FP32>> );
        static_assert( std::is_move_assignable_v<CpuTensor<TensorDataType::FP32>> );
    }

    class TensorConstructorsCpuTests : public testing::Test {};

    // ====================================================================
    // A. Construction & Validation
    // ====================================================================

    TEST_F( TensorConstructorsCpuTests, Construct_MultiDimShape ) {
        shape_t shape = { 2, 3 };
        CpuTensor<TensorDataType::FP32> tensor( Device::Cpu(), shape );

        EXPECT_FALSE( tensor.empty() );
        EXPECT_EQ( tensor.size(), 6 );
        EXPECT_EQ( tensor.rank(), 2 );
        EXPECT_EQ( tensor.shape(), shape );
        EXPECT_EQ( tensor.getDeviceType(), DeviceType::Cpu );
    }

    TEST_F( TensorConstructorsCpuTests, Construct_ScalarFromEmptyShape ) {
        CpuTensor<TensorDataType::FP32> tensor( Device::Cpu(), shape_t{} );

        EXPECT_FALSE( tensor.empty() );   // scalars are NOT empty
        EXPECT_EQ( tensor.size(), 1 );    // scalars have size 1
        EXPECT_EQ( tensor.rank(), 0 );
        EXPECT_EQ( tensor.strides().size(), 0 );
        EXPECT_TRUE( tensor.isScalar() );
    }

    TEST_F( TensorConstructorsCpuTests, Construct_EmptyFromZeroInShape ) {
        CpuTensor<TensorDataType::FP32> tensor( Device::Cpu(), shape_t{ 0 } );

        EXPECT_TRUE( tensor.empty() );    // a zero in the shape yields an empty tensor
        EXPECT_EQ( tensor.size(), 0 );
        EXPECT_EQ( tensor.rank(), 1 );    // still rank 1
        EXPECT_FALSE( tensor.isScalar() );
    }

    TEST_F( TensorConstructorsCpuTests, Construct_SingleDimension ) {
        CpuTensor<TensorDataType::INT32> tensor( Device::Cpu(), shape_t{ 42 } );

        EXPECT_EQ( tensor.size(), 42 );
        EXPECT_EQ( tensor.rank(), 1 );
        EXPECT_FALSE( tensor.empty() );
    }

    TEST_F( TensorConstructorsCpuTests, Construct_LargeShape ) {
        shape_t large_shape = { 100, 200, 50 };
        CpuTensor<TensorDataType::FP32> tensor( Device::Cpu(), large_shape );

        EXPECT_EQ( tensor.shape(), large_shape );
        EXPECT_EQ( tensor.size(), 1000000 );
        EXPECT_EQ( tensor.rank(), 3 );
    }

    TEST_F( TensorConstructorsCpuTests, Construct_WithName ) {
        CpuTensor<TensorDataType::FP32> tensor( Device::Cpu(), shape_t{ 2, 3 }, "activations" );

        EXPECT_EQ( tensor.getName(), "activations" );
    }

    // ====================================================================
    // B. Move semantics (move-only type)
    // ====================================================================

    TEST_F( TensorConstructorsCpuTests, Move_Construct_TransfersStateAndInvalidatesSource ) {
        shape_t shape = { 2, 3 };
        CpuTensor<TensorDataType::FP32> original( Device::Cpu(), shape );
        std::string original_uid = original.getUId();

        CpuTensor<TensorDataType::FP32> moved( std::move( original ) );

        EXPECT_EQ( moved.shape(), shape );
        EXPECT_EQ( moved.size(), 6 );
        EXPECT_EQ( moved.getUId(), original_uid );

        EXPECT_TRUE( original.empty() );
        EXPECT_EQ( original.size(), 0 );
    }

    TEST_F( TensorConstructorsCpuTests, Move_Assign_TransfersStateAndInvalidatesSource ) {
        shape_t shape = { 2, 3 };
        CpuTensor<TensorDataType::FP32> original( Device::Cpu(), shape );
        std::string original_uid = original.getUId();

        CpuTensor<TensorDataType::FP32> moved( Device::Cpu(), shape_t{} );
        moved = std::move( original );

        EXPECT_EQ( moved.shape(), shape );
        EXPECT_EQ( moved.size(), 6 );
        EXPECT_EQ( moved.getUId(), original_uid );

        EXPECT_TRUE( original.empty() );
        EXPECT_EQ( original.size(), 0 );
    }

    TEST_F( TensorConstructorsCpuTests, Move_Assign_SelfMoveIsSafe ) {
        shape_t shape = { 2, 3 };
        CpuTensor<TensorDataType::FP32> tensor( Device::Cpu(), shape );
        std::string original_uid = tensor.getUId();

        // Self-move guard: must leave the tensor in a valid, unchanged state.
        tensor = std::move( tensor );

        EXPECT_EQ( tensor.getUId(), original_uid );
        EXPECT_EQ( tensor.shape(), shape );
        EXPECT_EQ( tensor.size(), 6 );
    }

    // ====================================================================
    // C. Unique identity
    // ====================================================================

    TEST_F( TensorConstructorsCpuTests, UniqueId_DistinctPerInstance ) {
        CpuTensor<TensorDataType::FP32> tensor1( Device::Cpu(), shape_t{ 2, 3 } );
        CpuTensor<TensorDataType::FP32> tensor2( Device::Cpu(), shape_t{ 2, 3 } );

        EXPECT_NE( tensor1.getUId(), tensor2.getUId() );
    }

    TEST_F( TensorConstructorsCpuTests, UniqueId_TransfersWithMove ) {
        CpuTensor<TensorDataType::FP32> original( Device::Cpu(), shape_t{ 2, 3 } );
        std::string original_uid = original.getUId();

        CpuTensor<TensorDataType::FP32> moved( std::move( original ) );
        EXPECT_EQ( moved.getUId(), original_uid );

        CpuTensor<TensorDataType::FP32> fresh( Device::Cpu(), shape_t{ 2, 3 } );
        EXPECT_NE( fresh.getUId(), original_uid );
    }

    // ====================================================================
    // J. CPU-supported data types (getDataType / getDataTypeName / size)
    // ====================================================================

    TEST_F( TensorConstructorsCpuTests, DataTypes_AllCpuSupported ) {
        shape_t shape = { 2, 3 };

        CpuTensor<TensorDataType::FP32> fp32( Device::Cpu(), shape );
        CpuTensor<TensorDataType::INT8> int8( Device::Cpu(), shape );
        CpuTensor<TensorDataType::INT16> int16( Device::Cpu(), shape );
        CpuTensor<TensorDataType::INT32> int32( Device::Cpu(), shape );
        CpuTensor<TensorDataType::UINT8> uint8( Device::Cpu(), shape );
        CpuTensor<TensorDataType::UINT16> uint16( Device::Cpu(), shape );
        CpuTensor<TensorDataType::UINT32> uint32( Device::Cpu(), shape );

        EXPECT_EQ( fp32.getDataType(), TensorDataType::FP32 );
        EXPECT_EQ( int8.getDataType(), TensorDataType::INT8 );
        EXPECT_EQ( int16.getDataType(), TensorDataType::INT16 );
        EXPECT_EQ( int32.getDataType(), TensorDataType::INT32 );
        EXPECT_EQ( uint8.getDataType(), TensorDataType::UINT8 );
        EXPECT_EQ( uint16.getDataType(), TensorDataType::UINT16 );
        EXPECT_EQ( uint32.getDataType(), TensorDataType::UINT32 );

        EXPECT_EQ( fp32.getDataTypeName(), "FP32" );
        EXPECT_EQ( int8.getDataTypeName(), "INT8" );
        EXPECT_EQ( int16.getDataTypeName(), "INT16" );
        EXPECT_EQ( int32.getDataTypeName(), "INT32" );
        EXPECT_EQ( uint8.getDataTypeName(), "UINT8" );
        EXPECT_EQ( uint16.getDataTypeName(), "UINT16" );
        EXPECT_EQ( uint32.getDataTypeName(), "UINT32" );

        EXPECT_EQ( fp32.size(), 6 );
        EXPECT_EQ( int8.size(), 6 );
        EXPECT_EQ( uint32.size(), 6 );

        EXPECT_TRUE( fp32.is_host_accessible() );
        EXPECT_FALSE( fp32.is_device_accessible() );
    }
}
