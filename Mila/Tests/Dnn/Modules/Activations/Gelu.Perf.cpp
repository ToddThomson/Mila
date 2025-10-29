#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstdint>

import Mila;

namespace Modules::Activations::PerfTests
{
	using namespace Mila::Dnn;
	using namespace Mila::Dnn::Compute;

	using MR = CudaDeviceMemoryResource;
	using GeluCudaModule = Gelu<DeviceType::Cuda, TensorDataType::FP32>;

	// Simple statistics helpers
	static double mean( const std::vector<double>& v )
	{
		if (v.empty()) return 0.0;
		return std::accumulate( v.begin(), v.end(), 0.0 ) / static_cast<double>(v.size());
	}

	static double median( std::vector<double> v )
	{
		if (v.empty()) return 0.0;
		std::sort( v.begin(), v.end() );
		size_t n = v.size();
		if (n % 2 == 1) return v[n / 2];
		return 0.5 * (v[n / 2 - 1] + v[n / 2]);
	}

	// Run an end-to-end performance measurement for a single shape.
	static void RunGeluEndToEndPerf( 
		std::shared_ptr<ExecutionContext<DeviceType::Cuda>> ctx,
		const std::vector<int64_t>& shape,
		size_t warmup_iters = 5,
		size_t timed_iters = 50 )
	{
		GeluConfig config;
		auto gelu = std::make_shared<GeluCudaModule>( ctx, config );

		// Allocate device tensors
		Tensor<TensorDataType::FP32, MR> input( ctx->getDevice(), shape );
		Tensor<TensorDataType::FP32, MR> output( ctx->getDevice(), shape );

		// Prepare host input and copy
		auto cpu_ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
		auto cpu_device = cpu_ctx->getDevice();
		Tensor<TensorDataType::FP32, CpuMemoryResource> host_input( cpu_device, shape );

		for (size_t i = 0; i < host_input.size(); ++i)
			host_input.data()[i] = static_cast<float>( i ) / host_input.size() * 4.0f - 2.0f;

		copy( host_input, input );

		// Warm-up runs
		for (size_t i = 0; i < warmup_iters; ++i)
		{
			gelu->forward( input, output );
			gelu->synchronize();
			
			// Ensure work completes by copying a small portion back (toHost will synchronize)
			auto tmp = toHost<TensorDataType::FP32>( output );
			(void)tmp.data();
		}

		// Timed runs
		std::vector<double> durations_ms;
		durations_ms.reserve( timed_iters );

		for (size_t i = 0; i < timed_iters; ++i)
		{
			auto t0 = std::chrono::steady_clock::now();

			gelu->forward( input, output );

			// Force synchronization and transfer to host to ensure kernel completed
			auto out_host = toHost<TensorDataType::FP32>( output );
			(void)out_host.data();

			auto t1 = std::chrono::steady_clock::now();
			std::chrono::duration<double, std::milli> ms = t1 - t0;
			durations_ms.push_back( ms.count() );
		}

		double avg = mean( durations_ms );
		double med = median( durations_ms );

		std::cout << "GELU Perf - shape=[";
		for (size_t i = 0; i < shape.size(); ++i)
		{
			if (i) std::cout << ",";
			std::cout << shape[i];
		}
		std::cout << "] : mean(ms)=" << avg << ", median(ms)=" << med << "\n";
	}

	// Disabled by default so this does not run in regular CI. Enable explicitly to run perf tests.
	TEST( GeluPerf, GeluCuda_EndToEndPerf )
	{
		if (!isOperationRegistered<DeviceType::Cuda, TensorDataType::FP32>( "GeluOp" ))
		{
			GTEST_SKIP() << "CUDA GeluOp not registered - skipping perf test.";
		}

		auto ctx = std::make_shared<ExecutionContext<DeviceType::Cuda>>( 0 );

		// A set of realistic sizes: small, medium, large, very large
		std::vector<std::vector<int64_t>> shapes = {
			{1,128,768}, // typical small transformer batch
			{8,256,1024}, // medium
			{16,512,2048}, // large
			{ 64,512,2048 } // V large
		};

		for (const auto& s : shapes)
		{
			RunGeluEndToEndPerf( ctx, s, /*warmup=*/5, /*timed=*/50 );
		}
	}
}