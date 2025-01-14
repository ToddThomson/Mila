// MilaTest/test_example.cpp
#include <gtest/gtest.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

import Mila;

using namespace Mila::Dnn::Compute;

int main( int argc, char** argv ) {
	// TODO: Static device registration should be automatic
	Mila::Initialize();
	
	::testing::InitGoogleTest( &argc, argv );
	return RUN_ALL_TESTS();
}	
