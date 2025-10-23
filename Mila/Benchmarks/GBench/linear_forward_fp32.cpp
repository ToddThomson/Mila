#include <benchmark/benchmark.h>
#include <vector>
#include <memory>
#include <iostream>

import Mila;

namespace GBench::GeluBenchmarks
{
	using namespace Mila::Dnn;
	using namespace Mila::Dnn::Compute;

	// Helper to create a 4-tuple Args vector with named semantics:
	// (batch, sequence_length, input_channels, output_channels)
	static std::vector<int64_t> Arg4( size_t batch, size_t seq, size_t in_ch, size_t out_ch )
	{
		return std::vector<int64_t>{ static_cast<int64_t>(batch),
									 static_cast<int64_t>(seq),
									 static_cast<int64_t>(in_ch),
									 static_cast<int64_t>(out_ch) };
	}

	static void BM_CudaLinearForward( benchmark::State& state )
	{
		// Interpret args as explicit quadruple: [batch, seq, input_features, output_features]
		// No fallbacks: this benchmark requires exactly four arguments to be registered.
		if ( state.range( 3 ) == 0 )
		{
			state.SkipWithError( "BM_CudaLinearForward requires four args: (batch, sequence_length, input_features, output_features)" );
			return;
		}

		size_t batch = static_cast<size_t>( state.range( 0 ) );
		size_t seq = static_cast<size_t>( state.range( 1 ) );
		size_t input_features = static_cast<size_t>( state.range( 2 ) );
		size_t output_features = static_cast<size_t>( state.range( 3 ) );

		// build shapes
		std::vector<size_t> input_shape  = { batch, seq, input_features };
		std::vector<size_t> output_shape = { batch, seq, output_features };

		// Create CUDA execution context (device0)
		auto ctx = std::make_shared<ExecutionContext<DeviceType::Cuda>>( 0 );

		// Create Linear config from parsed args
		LinearConfig config( input_features, output_features );

		// Use operation registry to create op
		auto op = Mila::Dnn::Compute::OperationRegistry::instance().createUnaryOperation<DeviceType::Cuda, TensorDataType::FP32>(
			"LinearOp", ctx, config );

		if ( !op )
		{
			state.SkipWithError( "LinearOp not registered" );
			return;
		}

		// Create tensors (device)
		Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> input( ctx->getDevice(), input_shape );
		Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> output( ctx->getDevice(), output_shape );

		// Prepare host input
		auto cpu_ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>( -1 );
		auto cpu_dev = cpu_ctx->getDevice();
		Tensor<TensorDataType::FP32, CpuMemoryResource> host_input( cpu_dev, input_shape );

		// Use tensor ops / initializers instead of manual loops:
		// Fill host input with deterministic constant values (1.0f) for reproducible benchmarks.
		fill<TensorDataType::FP32, CpuMemoryResource>( host_input, 1.0f );

		copy( host_input, input );

		// Initialize and populate weight (and optional bias) parameters required by LinearOp.
		// weight shape = [output_features, input_features], bias shape = [output_features]
		std::vector<size_t> weight_shape = { output_features, input_features };
		std::vector<size_t> bias_shape = { output_features };

		// Create host parameter tensors and initialize using available initializers:
		// Xavier for weights, zeros for bias.
		Tensor<TensorDataType::FP32, CpuMemoryResource> host_weight( cpu_dev, weight_shape );
		xavier<TensorDataType::FP32, CpuMemoryResource>( host_weight, input_features, output_features );

		std::shared_ptr<Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>> device_weight =
			std::make_shared<Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>>( ctx->getDevice(), weight_shape );
		device_weight->setName( "bench.linear.weight" );
		copy( host_weight, *device_weight );

		std::shared_ptr<Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>> device_bias = nullptr;
		if ( config.hasBias() )
		{
			Tensor<TensorDataType::FP32, CpuMemoryResource> host_bias( cpu_dev, bias_shape );
			zeros<TensorDataType::FP32, CpuMemoryResource>( host_bias );

			device_bias = std::make_shared<Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>>( ctx->getDevice(), bias_shape );
			device_bias->setName( "bench.linear.bias" );
			copy( host_bias, *device_bias );
		}

		// Prepare parameters vector for the operation (weight first, optional bias second)
		std::vector<std::shared_ptr<Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>>> params;
		params.emplace_back( device_weight );
		if ( device_bias ) params.emplace_back( device_bias );

		// Prepare empty output_state to match UnaryOperation::forward signature
		std::vector<std::shared_ptr<Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>>> out_state;

		// Warm up
		for ( int i = 0; i < 5; ++i )
		{
			op->forward( input, params, output, out_state );
			ctx->synchronize();
		}

		// Benchmark loop
		for ( auto _ : state )
		{
			op->forward( input, params, output, out_state );
			ctx->synchronize();

			// report bytes (approx) using input bytes processed
			state.SetBytesProcessed( static_cast<int64_t>( input.size() * sizeof( float ) ) );
		}
	}

	// Register benchmarks using explicit quadruples: (batch, sequence_length, input_features, output_features)
	BENCHMARK( BM_CudaLinearForward )->Args( Arg4( 4, 128, 1024, 4096 ) )->Args( Arg4( 64, 1024, 768, 50304 ) );
}