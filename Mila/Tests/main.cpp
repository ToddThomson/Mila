// MilaTest/test_example.cpp
#include <gtest/gtest.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

//import Mila;

int main( int argc, char** argv ) {
	::testing::InitGoogleTest( &argc, argv );
	return RUN_ALL_TESTS();
}	
