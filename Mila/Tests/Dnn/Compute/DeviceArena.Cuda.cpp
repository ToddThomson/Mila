/**
 * @file DeviceArena.Cuda.cpp
 * @brief The parameter arena a block's weights are placed in (MemoryFootprint.md 11.8.1).
 *
 * Placement, the price a placed tensor reports, and the arena's own exactness check.
 */

#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>

import Mila;

namespace Mila::Tests::Dnn::Compute
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace
    {
        bool hasCuda()
        {
            return DeviceRegistry::instance().hasDeviceType( DeviceType::Cuda );
        }

        std::shared_ptr<DeviceArena> arenaOf( std::size_t capacity )
        {
            return std::make_shared<DeviceArena>( std::make_unique<CudaDeviceMemoryResource>( 0 ), capacity, "test" );
        }
    }

    TEST( DeviceArenaCudaTests, PlacedSizeRoundsUpTo256Bytes )
    {
        EXPECT_EQ( arenaPlacedBytes( 0 ), 0u );
        EXPECT_EQ( arenaPlacedBytes( 1 ), 256u );
        EXPECT_EQ( arenaPlacedBytes( 256 ), 256u );
        EXPECT_EQ( arenaPlacedBytes( 257 ), 512u );
    }

    TEST( DeviceArenaCudaTests, TensorsArePlacedBackToBackAt256ByteBoundaries )
    {
        if ( !hasCuda() )
        {
            GTEST_SKIP() << "CUDA device not available.";
        }

        // 100 FP32 = 400 bytes, 3 BF16 = 6 bytes, 1024 UINT8 = 1024 bytes.
        const std::size_t capacity = arenaPlacedBytes( 400 ) + arenaPlacedBytes( 6 ) + arenaPlacedBytes( 1024 );
        auto arena = arenaOf( capacity );

        Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> first( Device::Cuda( 0 ), shape_t{ 100 }, "first", arena );
        Tensor<TensorDataType::BF16, CudaDeviceMemoryResource> second( Device::Cuda( 0 ), shape_t{ 3 }, "second", arena );
        Tensor<TensorDataType::UINT8, CudaDeviceMemoryResource> third( Device::Cuda( 0 ), shape_t{ 1024 }, "third", arena );

        const auto address = []( const auto& tensor ) { return reinterpret_cast<std::uintptr_t>( tensor.rawData() ); };

        EXPECT_EQ( address( first ) % kArenaPlacementAlignmentBytes, 0u );
        EXPECT_EQ( address( second ) - address( first ), 512u );
        EXPECT_EQ( address( third ) - address( second ), 256u );
        EXPECT_TRUE( arena->isFull() );
    }

    TEST( DeviceArenaCudaTests, PlacedTensorReportsItsPlacedSize )
    {
        if ( !hasCuda() )
        {
            GTEST_SKIP() << "CUDA device not available.";
        }

        auto arena = arenaOf( arenaPlacedBytes( 3 * 1024 * 1024 + 2 ) );

        Tensor<TensorDataType::BF16, CudaDeviceMemoryResource> placed(
            Device::Cuda( 0 ), shape_t{ 1536 * 1024 + 1 }, "placed", arena );
        Tensor<TensorDataType::BF16, CudaDeviceMemoryResource> alone( Device::Cuda( 0 ), shape_t{ 1536 * 1024 + 1 }, "alone" );

        EXPECT_TRUE( placed.isPlacedInArena() );
        EXPECT_FALSE( alone.isPlacedInArena() );

        // Placed: its bytes to the next 256. Alone: its own allocation, rounded to the 2 MiB granule.
        EXPECT_EQ( occupiedTensorBytes( placed ), arenaPlacedBytes( placed.getStorageSize() ) );
        EXPECT_EQ( occupiedTensorBytes( alone ), 4u * 1024 * 1024 );
    }

    TEST( DeviceArenaCudaTests, OverfullArenaNamesTheTensor )
    {
        if ( !hasCuda() )
        {
            GTEST_SKIP() << "CUDA device not available.";
        }

        auto arena = arenaOf( 256 );

        Tensor<TensorDataType::UINT8, CudaDeviceMemoryResource> fits( Device::Cuda( 0 ), shape_t{ 200 }, "fits", arena );

        EXPECT_THROW(
            ( Tensor<TensorDataType::UINT8, CudaDeviceMemoryResource>( Device::Cuda( 0 ), shape_t{ 1 }, "spills", arena ) ),
            std::logic_error );
    }

    TEST( DeviceArenaCudaTests, PlacedTensorKeepsTheArenaAlive )
    {
        if ( !hasCuda() )
        {
            GTEST_SKIP() << "CUDA device not available.";
        }

        auto arena = arenaOf( 256 );
        const std::weak_ptr<DeviceArena> watch = arena;

        auto tensor = std::make_unique<Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>>(
            Device::Cuda( 0 ), shape_t{ 64 }, "kept", arena );

        arena.reset();
        EXPECT_FALSE( watch.expired() );

        tensor.reset();
        EXPECT_TRUE( watch.expired() );
    }
}
