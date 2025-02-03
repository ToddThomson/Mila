#include <gtest/gtest.h>

import Mila;

int main( int argc, char** argv ) {
	// TODO: Static device registration should be automatic
	Mila::Initialize();
	
	::testing::InitGoogleTest( &argc, argv );
	return RUN_ALL_TESTS();
}	
