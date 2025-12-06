#include <gtest/gtest.h>
#include <string>
#include <vector>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <set>
#include <functional>
#include <stdexcept>

import Compute.Device;
import Compute.DeviceId;
import Compute.DeviceType;

namespace Dnn::Compute::Tests
{
	using namespace Mila::Dnn::Compute;

	class DeviceIdTest : public ::testing::Test
	{
	protected:
		void SetUp() override
		{
		}

		void TearDown() override
		{
		}
	};

	TEST_F( DeviceIdTest, DefaultConstructorAndHelpers )
	{
		auto cpu = Device::Cpu();
		EXPECT_EQ( cpu.type, DeviceType::Cpu );
		EXPECT_EQ( cpu.index, 0 );

		auto cuda3 = Device::Cuda( 3 );
		EXPECT_EQ( cuda3.type, DeviceType::Cuda );
		EXPECT_EQ( cuda3.index, 3 );

		DeviceId explicitId{ DeviceType::Cpu, -1 };
		EXPECT_EQ( explicitId.type, DeviceType::Cpu );
		EXPECT_EQ( explicitId.index, -1 );
	}

	TEST_F( DeviceIdTest, ComparisonAndOrdering )
	{
		std::vector<DeviceId> ids{
			Device::Cuda( 2 ),
			Device::Cpu(),
			Device::Cuda( 0 ),
			Device::Cuda( 1 ),
			DeviceId{ DeviceType::Cpu, 5 }
		};

		std::sort( ids.begin(), ids.end() );

		// Ordering compares `type` first, then `index`.
		// DeviceType::Cpu (0) comes before DeviceType::Cuda (1).
		EXPECT_TRUE( ids.front().type == DeviceType::Cpu );
		EXPECT_TRUE( ids.back().type == DeviceType::Cuda );

		// Within same type Cuda, indices should be ascending
		auto it = std::find_if( ids.begin(), ids.end(), []( const DeviceId& d ) { return d.type == DeviceType::Cuda; } );
		ASSERT_NE( it, ids.end() );
		std::vector<int> cudaIndices;
		for ( ; it != ids.end(); ++it )
		{
			ASSERT_EQ( it->type, DeviceType::Cuda );
			cudaIndices.push_back( it->index );
		}

		EXPECT_EQ( cudaIndices, std::vector<int>( { 0,1,2 } ) );
	}

	TEST_F( DeviceIdTest, ToStringAndParseRoundTrip )
	{
		std::vector<DeviceId> cases{
			Device::Cpu(),
			Device::Cuda( 0 ),
			Device::Cuda( 7 ),
			DeviceId{ DeviceType::Metal, 3 },
			DeviceId{ DeviceType::Rocm, -1 }
		};

		for ( const auto& id : cases )
		{
			std::string s = id.toString();

			// Round-trip via parse should produce an equivalent DeviceId
			DeviceId parsed = DeviceId::parse( s );
			EXPECT_EQ( parsed, id ) << "Round-trip failed for: " << s;
		}

		// Also test parse without explicit index (interpreted as index 0)
		auto p = DeviceId::parse( "cuda" );
		EXPECT_EQ( p, Device::Cuda( 0 ) );

		p = DeviceId::parse( "CPU" );
		EXPECT_EQ( p, Device::Cpu() );

		p = DeviceId::parse( "Metal:2" );
		EXPECT_EQ( p, Device::Metal(2) );
	}

	TEST_F( DeviceIdTest, ParseFailures )
	{
		EXPECT_THROW( DeviceId::parse( "cuda:abc" ), std::invalid_argument );

		EXPECT_THROW( DeviceId::parse( "cuda:" ), std::invalid_argument );

		EXPECT_THROW( DeviceId::parse( "INVALID:0" ), std::invalid_argument );

		EXPECT_THROW( DeviceId::parse( "" ), std::invalid_argument );
	}

	TEST_F( DeviceIdTest, HashAndUnorderedContainers )
	{
		DeviceId a = Device::Cuda( 0 );
		DeviceId b = Device::Cuda( 1 );
		DeviceId a2 = Device::Cuda( 0 );

		// Equal objects must have equal hashes
		std::hash<DeviceId> hasher;
		EXPECT_EQ( hasher( a ), hasher( a2 ) );

		// Use as keys in unordered_set and unordered_map
		std::unordered_set<DeviceId> set;
		set.insert( a );
		set.insert( b );
		set.insert( a2 ); // duplicate

		EXPECT_EQ( set.size(), 2u );
		EXPECT_NE( set.find( a ), set.end() );
		EXPECT_NE( set.find( b ), set.end() );

		std::unordered_map<DeviceId, std::string> map;
		map[a] = "cuda0";
		map[b] = "cuda1";

		EXPECT_EQ( map[a], "cuda0" );
		EXPECT_EQ( map[b], "cuda1" );

		// Negative index allowed and round-trips through toString/parse
		DeviceId neg{ DeviceType::Cpu, -5 };
		std::string negStr = neg.toString();
		DeviceId negParsed = DeviceId::parse( negStr );
		EXPECT_EQ( negParsed, neg );
	}

	TEST_F( DeviceIdTest, UsabilityInOrderedContainers )
	{
		// Verify DeviceId provides total ordering usable by ordered containers
		std::set<DeviceId> s;
		s.insert( Device::Cuda( 2 ) );
		s.insert( Device::Cpu() );
		s.insert( Device::Cuda( 1 ) );

		// First element should be CPU
		ASSERT_FALSE( s.empty() );
		EXPECT_EQ( *s.begin(), Device::Cpu() );

		// Ensure Cuda indices are ordered inside the set
		auto it = std::find_if( s.begin(), s.end(), []( const DeviceId& d ) { return d.type == DeviceType::Cuda; } );
		ASSERT_NE( it, s.end() );
		EXPECT_EQ( it->index, 1 );
	}
}