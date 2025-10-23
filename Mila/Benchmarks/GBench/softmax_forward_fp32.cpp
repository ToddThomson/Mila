#include <benchmark/benchmark.h>
#include <vector>
#include <memory>
#include <iostream>

import Mila;

namespace GBench::GeluBenchmarks
{
	using namespace Mila::Dnn;
	using namespace Mila::Dnn::Compute;

	static void BM_CudaSoftmaxForward( benchmark::State& state )
	{
		// Prefer explicit triple: [batch, seq, channels]
		// Backward-compatible: if only a single non-zero arg is provided treat it as total elements.
		size_t batch = 0;
		size_t seq = 1;
		size_t channels = 0;

		// If caller passed three non-zero ranges interpret them directly.
		if (state.range( 2 ) != 0)
		{
			batch = static_cast<size_t>(state.range( 0 ));
			seq = static_cast<size_t>(state.range( 1 ));
			channels = static_cast<size_t>(state.range( 2 ));
		}
		else
		{
			// Fallback: single-arg mode (total elements). Keep channels default to 1024 to match prior behavior.
			size_t total = static_cast<size_t>(state.range( 0 ));
			channels = 1024;
			seq = 1;
			batch = total / (seq * channels);
			if (batch == 0) batch = 1;
		}

		// build shape [batch, seq, channels]
		std::vector<size_t> shape = { batch, seq, channels };

		// Create CUDA execution context (device0)
		auto ctx = std::make_shared<ExecutionContext<DeviceType::Cuda>>( 0 );

		// Create Gelu op via registry
		SoftmaxConfig config;

		// Use operation registry to create op
		auto op = Mila::Dnn::Compute::OperationRegistry::instance().createUnaryOperation<DeviceType::Cuda, TensorDataType::FP32>(
			"SoftmaxOp", ctx, config );

		if (!op)
		{
			state.SkipWithError( "GeluOp not registered" );
			return;
		}

		// Create tensors
		Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> input( ctx->getDevice(), shape );
		Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> output( ctx->getDevice(), shape );

		// Prepare host input
		auto cpu_ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>( -1 );
		auto cpu_dev = cpu_ctx->getDevice();
		Tensor<TensorDataType::FP32, CpuMemoryResource> host_input( cpu_dev, shape );

		for (size_t i = 0; i < host_input.size(); ++i)
			host_input.data()[i] = static_cast<float>( i ) / host_input.size();

		copy( host_input, input );

		// Prepare empty parameters and output_state to match UnaryOperation::forward signature
		std::vector<std::shared_ptr<Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>>> params;
		std::vector<std::shared_ptr<Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>>> out_state;

		// Warm up
		for (int i = 0; i < 5; ++i)
		{
			op->forward( input, params, output, out_state );
			ctx->synchronize();
		}

		// Benchmark loop
		for (auto _ : state)
		{
			op->forward( input, params, output, out_state );
			ctx->synchronize();

			// report bytes (approx)
			state.SetBytesProcessed( static_cast<int64_t>(input.size() * sizeof( float )) );
		}
	}

	// Register benchmarks using explicit triples: (batch, seq, channels)
	BENCHMARK( BM_CudaSoftmaxForward )->Args( { 4, 128, 768 } )->Args( { 16, 512, 1024 } );
}