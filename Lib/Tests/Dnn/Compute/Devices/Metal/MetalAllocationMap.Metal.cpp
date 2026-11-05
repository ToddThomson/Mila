/**
 * @file MetalAllocationMap.Metal.cpp
 * @brief Addresses to buffers and offsets. The buffers are tokens, never dereferenced, so no GPU is touched.
 */

#include <gtest/gtest.h>

namespace MTL
{
    class Buffer;
}

import std;
import Compute.MetalAllocationMap;
import Compute.MetalBufferLocation;

namespace Mila::Tests::Dnn::Compute
{
    using namespace Mila::Dnn::Compute;

    namespace
    {
        const void* addressAt( std::uintptr_t address )
        {
            return reinterpret_cast<const void*>( address );
        }

        MTL::Buffer* bufferNamed( std::uintptr_t token )
        {
            return reinterpret_cast<MTL::Buffer*>( token );
        }
    }

    TEST( MetalAllocationMap, TheBaseAndAnInteriorAddressResolveWithTheirOffset )
    {
        MetalAllocationMap map;
        map.add( addressAt( 0x1000 ), 0x100, bufferNamed( 1 ) );

        EXPECT_EQ( map.locate( addressAt( 0x1000 ) ).buffer, bufferNamed( 1 ) );
        EXPECT_EQ( map.locate( addressAt( 0x1000 ) ).offset, 0u );
        EXPECT_EQ( map.locate( addressAt( 0x1040 ) ).offset, 0x40u );
        EXPECT_EQ( map.locate( addressAt( 0x10FF ) ).offset, 0xFFu );
    }

    TEST( MetalAllocationMap, OnePastTheEndIsNotInTheAllocation )
    {
        MetalAllocationMap map;
        map.add( addressAt( 0x1000 ), 0x100, bufferNamed( 1 ) );

        EXPECT_THROW( (void)map.locate( addressAt( 0x1100 ) ), std::invalid_argument );
    }

    TEST( MetalAllocationMap, AnAddressBeforeEveryAllocationOrInAGapIsRefused )
    {
        MetalAllocationMap map;
        map.add( addressAt( 0x1000 ), 0x100, bufferNamed( 1 ) );
        map.add( addressAt( 0x3000 ), 0x100, bufferNamed( 2 ) );

        EXPECT_THROW( (void)map.locate( addressAt( 0x0FFF ) ), std::invalid_argument );
        EXPECT_THROW( (void)map.locate( addressAt( 0x2000 ) ), std::invalid_argument );
    }

    TEST( MetalAllocationMap, EachAddressFindsItsOwnAllocation )
    {
        MetalAllocationMap map;
        map.add( addressAt( 0x1000 ), 0x100, bufferNamed( 1 ) );
        map.add( addressAt( 0x1100 ), 0x100, bufferNamed( 2 ) );

        EXPECT_EQ( map.locate( addressAt( 0x10FF ) ).buffer, bufferNamed( 1 ) );
        EXPECT_EQ( map.locate( addressAt( 0x1100 ) ).buffer, bufferNamed( 2 ) );
        EXPECT_EQ( map.locate( addressAt( 0x1100 ) ).offset, 0u );
    }

    TEST( MetalAllocationMap, ARemovedAllocationIsGoneAndItsBufferReturned )
    {
        MetalAllocationMap map;
        map.add( addressAt( 0x1000 ), 0x100, bufferNamed( 1 ) );

        EXPECT_EQ( map.remove( addressAt( 0x1000 ) ), bufferNamed( 1 ) );
        EXPECT_EQ( map.count(), 0u );
        EXPECT_THROW( (void)map.locate( addressAt( 0x1000 ) ), std::invalid_argument );
    }

    TEST( MetalAllocationMap, RemovingAnAddressThatStartsNoAllocationRemovesNothing )
    {
        MetalAllocationMap map;
        map.add( addressAt( 0x1000 ), 0x100, bufferNamed( 1 ) );

        EXPECT_EQ( map.remove( addressAt( 0x1040 ) ), nullptr );
        EXPECT_EQ( map.count(), 1u );
    }
}
