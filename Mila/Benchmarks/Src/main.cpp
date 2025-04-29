#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <stdexcept>
#include <sstream>
#include <cuda_runtime.h>

import Mila;

import Mila.Benchmark;
import Mila.BenchmarkManager;
import Mila.Benchmark.ModuleBenchmark;
import Mila.Benchmark.BlockModuleBenchmark;
import Mila.Benchmark.BinaryModuleBenchmark; // New import for binary module benchmarks
import Mila.Benchmark.OperationBenchmark;
import Mila.Benchmark.KernelBenchmark;

using namespace Mila::Dnn;
using namespace Mila::Dnn::Compute;
using namespace Mila::Benchmark;

int main( int argc, char* argv[] ) {
    try {
        std::cout << "Mila Benchmarking Tool" << std::endl;
        std::cout << "====================" << std::endl << std::endl;

        Mila::initialize();

        std::cout << "Mila version: " << Mila::getAPIVersion().ToString() << std::endl;

        auto devices = Compute::list_devices();
        std::cout << "Available devices: ";
        for ( const auto& device : devices ) {
            std::cout << device << " ";
        }
        std::cout << std::endl;

        // Parse command line arguments
        size_t iterations = 100;
        if ( argc > 1 ) {
            iterations = std::stoi( argv[ 1 ] );
        }

        // Create device contexts
        auto cpuContext = std::make_shared<DeviceContext>( "CPU" );
        auto cudaContext = std::make_shared<DeviceContext>( "CUDA:0" );

        BenchmarkManager manager;

        // Define benchmark shapes (small, medium, large)
        std::vector<std::vector<size_t>> benchmarkShapes = {
            {32, 128},              // Small
            {64, 1024},             // Medium
            {128, 4096},            // Large
            {8, 32, 128}            // 3D shape (batch, seq, hidden)
        };

        // Benchmark Module: Gelu
        for ( const auto& shape : benchmarkShapes ) {
            // CPU Gelu Module
            auto cpuGelu = std::make_shared<Gelu<float, DeviceType::Cpu>>(
                "Cpu::Gelu", cpuContext );

            // Create benchmark with direct template instantiation
            auto cpuGeluBench = std::make_unique<ModuleBenchmark<float, float, DeviceType::Cpu>>(
                cpuGelu,       // Pass the concrete module object
                shape,         // Input shape
                shape,         // Output shape is same as input for Gelu
                cpuContext
            );
            manager.addBenchmark( std::move( cpuGeluBench ) );

            // CUDA Gelu Module
            auto cudaGelu = std::make_shared<Gelu<float, DeviceType::Cuda>>(
                "Cuda::Gelu", cudaContext );

            // Create benchmark with direct template instantiation
            auto cudaGeluBench = std::make_unique<ModuleBenchmark<float, float, DeviceType::Cuda>>(
                cudaGelu,      // Pass the concrete module object
                shape,         // Input shape
                shape,         // Output shape is same as input for Gelu
                cudaContext
            );
            manager.addBenchmark( std::move( cudaGeluBench ) );
        }

        // Benchmark Operation: GeluOp
        for ( const auto& shape : benchmarkShapes ) {
            // CPU GeluOp
            auto cpuGeluOp = std::static_pointer_cast<OperationBase<float, float, DeviceType::Cpu>>(
                std::make_shared<CpuGeluOp>( cpuContext ));
            manager.addBenchmark( std::make_unique<OperationBenchmark<float, DeviceType::Cpu>>(
                cpuGeluOp, "CpuGeluOp", shape, cpuContext ) );

            // CUDA GeluOp
            auto cudaGeluOp = std::static_pointer_cast<OperationBase<float, float, DeviceType::Cuda>>(
                std::make_shared<CudaGeluOp<float>>( cudaContext ));
            manager.addBenchmark( std::make_unique<OperationBenchmark<float, DeviceType::Cuda>>(
                cudaGeluOp, "CudaGeluOp", shape, cudaContext ) );
        }

        // Benchmark CUDA Kernel: Gelu
        for ( const auto& shape : benchmarkShapes ) {
            // CUDA Gelu Kernel
            /*auto geluKernel = []( float* Y, const float* X, int N, cudaStream_t stream ) {
                cuda_gelu_forward_fp32( Y, X, N, stream );
                };

            manager.addBenchmark( std::make_unique<KernelBenchmark<float>>(
                geluKernel, "cuda_gelu_forward_fp32", shape, cudaContext ) );*/
        }

        // Benchmark BlockModule: MLP
        for ( const auto& shape : benchmarkShapes ) {
            if ( shape.size() >= 2 ) {  // MLP requires at least 2D input
                // Create MLP module
                size_t input_features = shape.back();
                size_t hidden_features = input_features * 4;  // Common ratio in transformers

                // CPU MLP
                auto cpuMLP = std::make_shared<MLP<float, DeviceType::Cpu>>(
                    "MLPCpu", cpuContext, shape, hidden_features, true, false );

                // Create benchmark with direct template instantiation
                auto cpuMLPBench = std::make_unique<BlockModuleBenchmark<float, DeviceType::Cpu>>(
                    cpuMLP,        // Pass the concrete block module object
                    shape,         // Input shape
                    cpuContext
                );
                manager.addBenchmark( std::move( cpuMLPBench ) );

                // CUDA MLP
                auto cudaMLP = std::make_shared<MLP<float, DeviceType::Cuda>>(
                    "MLPCuda", cudaContext, shape, hidden_features, true, false );

                // Create benchmark with direct template instantiation
                auto cudaMLPBench = std::make_unique<BlockModuleBenchmark<float, DeviceType::Cuda>>(
                    cudaMLP,       // Pass the concrete block module object
                    shape,         // Input shape
                    cudaContext
                );
                manager.addBenchmark( std::move( cudaMLPBench ) );
            }
        }

        // NEW: Benchmark BinaryModule: Residual
        for ( const auto& shape : benchmarkShapes ) {
            // CPU Residual
            auto cpuResidual = std::make_shared<Residual<float, DeviceType::Cpu>>(
                "ResidualCpu", cpuContext );

            // Create benchmark with direct template instantiation for binary module
            auto cpuResidualBench = std::make_unique<BinaryModuleBenchmark<float, DeviceType::Cpu>>(
                cpuResidual,    // Pass the concrete residual module object
                shape,          // Input shape
                cpuContext
            );
            manager.addBenchmark( std::move( cpuResidualBench ) );

            // CUDA Residual
            auto cudaResidual = std::make_shared<Residual<float, DeviceType::Cuda>>(
                "ResidualCuda", cudaContext );

            // Create benchmark with direct template instantiation for binary module
            auto cudaResidualBench = std::make_unique<BinaryModuleBenchmark<float, DeviceType::Cuda>>(
                cudaResidual,   // Pass the concrete residual module object
                shape,          // Input shape
                cudaContext
            );
            manager.addBenchmark( std::move( cudaResidualBench ) );
        }

        // Run all benchmarks
        manager.runAll( iterations );

        return 0;
    }
    catch ( const std::exception& e ) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
