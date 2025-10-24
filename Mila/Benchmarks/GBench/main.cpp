#include <benchmark/benchmark.h>
#include <iostream>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#endif

import Mila;

//// Helper function to calculate theoretical peak FLOPS
//double calculateTheoreticalPeakFP32( const Mila::Dnn::Compute::CudaDeviceProps& props )
//{
//	int cores_per_sm = 0;
//
//	// Determine cores per SM based on compute capability
//	if (props.major == 8)
//	{  // Ampere
//		if (props.minor == 0)
//		{
//			cores_per_sm = 64;  // A100
//		}
//		else if (props.minor == 6)
//		{
//			cores_per_sm = 128; // RTX 30 series (GA102, GA104)
//		}
//		else if (props.minor == 9)
//		{
//			cores_per_sm = 128; // RTX 40 series (Ada - 4070, 4060)
//		}
//	}
//	else if (props.major == 7)
//	{  // Turing
//		cores_per_sm = 64;  // RTX 20 series
//	}
//	else if (props.major == 9)
//	{  // Ada Lovelace
//		if (props.minor == 0)
//		{
//			cores_per_sm = 128; // RTX 4090, 4080
//		}
//	}
//	else
//	{
//		// Default approximation
//		cores_per_sm = 64;
//	}
//
//	// Calculate peak TFLOPS
//	// Peak = SM_count * cores_per_SM * clock_rate_GHz * 2 (for FMA)
//	double total_cores = props.multiProcessorCount * cores_per_sm;
//	double clock_ghz = props.clockRate / 1e6;  // Convert kHz to GHz
//	double peak_tflops = (total_cores * clock_ghz * 2) / 1000.0;  // 2 for FMA
//
//	return peak_tflops;
//}

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
		std::cout << "  1) Open the Debug and Launch Settings for the Mila benchmark target.\n";
		std::cout << "  2) Set ""args"" to:\n";
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

	using namespace Mila::Dnn::Compute;

	// Print GPU information before running benchmarks
	try
	{
		auto& registry = DeviceRegistry::instance();

		// Create CPU device via getDevice
		auto dev = registry.getDevice( "CUDA:0" );
		auto cuda_device = std::dynamic_pointer_cast<CudaDevice>(dev);
		auto props = cuda_device->getProperties();

		std::cout << "\n" << std::string( 80, '=' ) << std::endl;
		std::cout << "GPU Benchmark Configuration" << std::endl;
		std::cout << std::string( 80, '=' ) << std::endl;
		std::cout << "Device:              " << props.name << std::endl;
		std::cout << "Compute Capability:  " << props.major << "." << props.minor << std::endl;
		std::cout << "Global Memory:       " << (props.totalGlobalMem / (1024 * 1024 * 1024)) << " GB" << std::endl;
		std::cout << "Multiprocessors:     " << props.multiProcessorCount << std::endl;
		std::cout << "Clock Rate:          " << (props.clockRate / 1000) << " MHz" << std::endl;
		std::cout << "Memory Clock:        " << (props.memoryClockRate / 1000) << " MHz" << std::endl;
		std::cout << "Memory Bus Width:    " << props.memoryBusWidth << " bits" << std::endl;
		std::cout << "L2 Cache Size:       " << (props.l2CacheSize / (1024 * 1024)) << " MB" << std::endl;

		// Calculate theoretical peak performance
		/*double theoretical_fp32_tflops = calculateTheoreticalPeakFP32( props );
		std::cout << "Theoretical Peak:    " << std::fixed << std::setprecision( 2 )
			<< theoretical_fp32_tflops << " TFLOPS (FP32)" << std::endl;*/

		/*std::cout << "CUDA Version:        " << (CUDART_VERSION / 1000) << "."
			<< ((CUDART_VERSION % 1000) / 10) << std::endl;
		std::cout << std::string( 80, '=' ) << std::endl << std::endl;*/

	}
	catch (const std::exception& e)
	{
		std::cerr << "Warning: Could not query GPU properties: " << e.what() << std::endl;
	}
	
	if (::benchmark::ReportUnrecognizedArguments( argc, argv )) 
		return 1;

	::benchmark::RunSpecifiedBenchmarks();

	return 0;
}