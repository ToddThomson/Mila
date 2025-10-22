#include <benchmark/benchmark.h>

import Mila;

int main( int argc, char** argv )
{
	::benchmark::Initialize( &argc, argv );

	// Initialize Mila
	Mila::initialize();
	
	if (::benchmark::ReportUnrecognizedArguments( argc, argv )) 
		return 1;

	::benchmark::RunSpecifiedBenchmarks();

	return 0;
}