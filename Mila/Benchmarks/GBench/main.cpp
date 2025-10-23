#include <benchmark/benchmark.h>
#include <iostream>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#endif

import Mila;

int main( int argc, char** argv )
{
	// If running under the debugger (Visual Studio) and no explicit benchmark filter was provided,
	// print a short reminder showing how to run a specific benchmark via command line / project settings.
	bool has_filter = false;
	for ( int i = 1; i < argc; ++i )
	{
		if ( std::strncmp( argv[i], "--benchmark_filter", 17 ) == 0 ||
			 std::strncmp( argv[i], "--benchmark_list_tests", 21 ) == 0 )
		{
			has_filter = true;
			break;
		}
	}

#ifdef _WIN32
	if ( IsDebuggerPresent() && !has_filter )
	{
		std::cout << "No --benchmark_filter provided and debugger is attached.\n\n";
		std::cout << "To run a specific benchmark from Visual Studio:\n";
		std::cout << "  1) Open the project properties for the benchmark target.\n";
		std::cout << "  2) Go to __Debugging__ and set __Command Arguments__ to:\n";
		std::cout << "       --benchmark_filter=BM_CudaLinearForward\n";
		std::cout << "     or use a substring/regex, e.g.:\n";
		std::cout << "       --benchmark_filter=Linear.*\n\n";
		std::cout << "To list available benchmarks from the command line, run:\n";
		std::cout << "  ./GBench --benchmark_list_tests=true\n\n";
		std::cout << "Exiting so you can set the desired command arguments in Visual Studio.\n";
		return 0;
	}
#endif

	::benchmark::Initialize( &argc, argv );

	Mila::initialize();
	
	if (::benchmark::ReportUnrecognizedArguments( argc, argv )) 
		return 1;

	::benchmark::RunSpecifiedBenchmarks();

	return 0;
}