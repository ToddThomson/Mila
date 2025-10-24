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
		if (state.range( 3 ) == 0)
		{
			state.SkipWithError( "BM_CudaLinearForward requires four args: (batch, sequence_length, input_features, output_features)" );
			return;
		}

		const size_t batch = static_cast<size_t>(state.range( 0 ));
		const size_t seq = static_cast<size_t>(state.range( 1 ));
		const size_t input_features = static_cast<size_t>(state.range( 2 ));
		const size_t output_features = static_cast<size_t>(state.range( 3 ));

		// Calculate memory requirements upfront
		const size_t input_bytes = batch * seq * input_features * sizeof( float );
		const size_t weight_bytes = output_features * input_features * sizeof( float );
		const size_t output_bytes = batch * seq * output_features * sizeof( float );
		const size_t total_bytes = input_bytes + weight_bytes + output_bytes;

		// Check GPU memory availability
		//size_t free_mem, total_mem;
		//cudaMemGetInfo( &free_mem, &total_mem );

		//if (total_bytes > free_mem * 0.9)
		//{  // Leave 10% headroom
		//	std::ostringstream ss;
		//	ss << "Insufficient GPU memory: need " << (total_bytes / (1024 * 1024))
		//		<< " MB, have " << (free_mem / (1024 * 1024)) << " MB free";
		//	state.SkipWithError( ss.str() );
		//	return;
		//}

		// Build shapes
		std::vector<size_t> input_shape = { batch, seq, input_features };
		std::vector<size_t> output_shape = { batch, seq, output_features };

		try
		{
			// Create CUDA execution context (device0)
			auto ctx = std::make_shared<ExecutionContext<DeviceType::Cuda>>( 0 );

			// Create Linear config from parsed args
			LinearConfig config( input_features, output_features );

			// Use operation registry to create op
			auto op = Mila::Dnn::Compute::OperationRegistry::instance().createUnaryOperation<DeviceType::Cuda, TensorDataType::FP32>(
				"LinearOp", ctx, config );

			if (!op)
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

			fill<TensorDataType::FP32, CpuMemoryResource>( host_input, 3.1415926f );
			copy( host_input, input );

			// Initialize weights and bias
			std::vector<size_t> weight_shape = { output_features, input_features };
			std::vector<size_t> bias_shape = { output_features };

			Tensor<TensorDataType::FP32, CpuMemoryResource> host_weight( cpu_dev, weight_shape );
			xavier<TensorDataType::FP32, CpuMemoryResource>( host_weight, input_features, output_features );

			std::shared_ptr<Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>> device_weight =
				std::make_shared<Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>>( ctx->getDevice(), weight_shape );
			device_weight->setName( "bench.linear.weight" );
			copy( host_weight, *device_weight );

			std::shared_ptr<Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>> device_bias = nullptr;
			if (config.hasBias())
			{
				Tensor<TensorDataType::FP32, CpuMemoryResource> host_bias( cpu_dev, bias_shape );
				zeros<TensorDataType::FP32, CpuMemoryResource>( host_bias );

				device_bias = std::make_shared<Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>>( ctx->getDevice(), bias_shape );
				device_bias->setName( "bench.linear.bias" );
				copy( host_bias, *device_bias );
			}

			// Prepare parameters vector
			std::vector<std::shared_ptr<Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>>> params;
			params.emplace_back( device_weight );
			if (device_bias) params.emplace_back( device_bias );

			std::vector<std::shared_ptr<Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>>> out_state;

			// Warm up (single synchronization after all warmup iterations)
			for (int i = 0; i < 5; ++i)
			{
				op->forward( input, params, output, out_state );
			}
			ctx->synchronize();

			// Benchmark loop
			for (auto _ : state)
			{
				op->forward( input, params, output, out_state );
				ctx->synchronize();
			}

			// Calculate metrics AFTER the loop (Google Benchmark handles accumulation)

			// FLOPs: Matrix multiply is 2*M*N*K (multiply-add = 2 ops)
			// For linear: output = input @ weight^T
			// Shape: [batch*seq, input_features] @ [input_features, output_features]
			const int64_t M = batch * seq;
			const int64_t K = input_features;
			const int64_t N = output_features;
			const int64_t flops_per_iter = 2LL * M * N * K;

			state.SetItemsProcessed( state.iterations() * flops_per_iter );

			// Memory traffic: Read input, read weight, write output
			// (Ignoring bias for simplicity - it's negligible)
			const int64_t bytes_per_iter =
				(M * K +      // input read
					K * N +      // weight read
					M * N) *     // output write
				sizeof( float );

			state.SetBytesProcessed( state.iterations() * bytes_per_iter );

			// FLOPS counter - will automatically show as TFLOPS when > 1000 GFLOPS
			state.counters["FLOPS"] = benchmark::Counter(
				static_cast<double>(flops_per_iter),
				benchmark::Counter::kIsRate,
				benchmark::Counter::kIs1000
			);

			// Arithmetic intensity (FLOPs per byte) - indicates compute vs memory bound
			state.counters["AI"] = benchmark::Counter(
				static_cast<double>(flops_per_iter) / static_cast<double>(bytes_per_iter),
				benchmark::Counter::kAvgThreads
			);

			// Optional: Efficiency vs theoretical peak (adjust for your GPU)
			// RTX 3090: ~35.6 TFLOPS FP32
			// RTX 4090: ~82.6 TFLOPS FP32
			// A100: ~19.5 TFLOPS FP32 (156 TFLOPS with sparsity)
			//constexpr double theoretical_tflops = 35.6;  // Change based on your GPU
			//const double achieved_tflops = (flops_per_iter / 1e12) /
			//	(state.iterations() * benchmark::CPUInfo::Get().cycles_per_second * 1e-9 *
			//		state.min_time / state.iterations());
			//state.counters["PeakUtil%"] = benchmark::Counter(
			//	100.0 * (flops_per_iter / 1e12) * state.iterations() /
			//	(theoretical_tflops * (state.iterations() * state. .min_time)),
			//	benchmark::Counter::kAvgThreads
			//);

			// Add dimension labels for clarity
			state.SetLabel(
				"B=" + std::to_string( batch ) +
				",S=" + std::to_string( seq ) +
				",I=" + std::to_string( input_features ) +
				",O=" + std::to_string( output_features )
			);

		}
		catch (const std::exception& e)
		{
			state.SkipWithError( std::string( "Benchmark failed: " ) + e.what() );
			return;
		}
	}

	// Register benchmarks using explicit quadruples: (batch, sequence_length, input_features, output_features)
	BENCHMARK( BM_CudaLinearForward )
		->Args( Arg4( 4, 128, 1024, 4096 ) )      // 27 MB - Small batch
		->Args( Arg4( 1, 1024, 768, 50304 ) )     // 267 MB - Single batch vocab
		->Args( Arg4( 32, 512, 768, 3072 ) )      // 260 MB - FFN layer
		->Args( Arg4( 16, 2048, 4096, 4096 ) )    // 1.14 GB - Large model
		->Args( Arg4( 8, 512, 4096, 4096 ) )      // 604 MB - Medium model
		->Args( Arg4( 1, 1, 4096, 4096 ) )        // 67 MB - Single token
		->UseRealTime()
		//->MinIterations( 50 )                     // More iterations for stable measurements
		->Unit( benchmark::kMillisecond );          // Display in milliseconds
}