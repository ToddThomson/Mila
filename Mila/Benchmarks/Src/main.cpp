#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <stdexcept>
#include <sstream>
#include <algorithm>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>

import Mila;

import Mila.Benchmark;
import Mila.BenchmarkManager;
import Mila.Benchmark.ModuleBenchmark;
import Mila.Benchmark.BlockModuleBenchmark;
import Mila.Benchmark.BinaryModuleBenchmark;
import Mila.Benchmark.OperationBenchmark;
import Mila.Benchmark.KernelBenchmark;

using namespace Mila::Dnn;
using namespace Mila::Dnn::Compute;
using namespace Mila::Benchmark;

namespace fs = std::filesystem;

// Define benchmark configuration struct
struct BenchmarkConfig {
    size_t iterations = 1000;
    DeviceType deviceType = DeviceType::Cuda;
    bool runCpu = false; // true;
    bool runCuda = true;
    bool runModuleBenchmarks = true;
    bool runOperationBenchmarks = true;
    bool runKernelBenchmarks = false;
    std::string outputFile = "";
    std::vector<std::vector<size_t>> customShapes = {};
    bool useDefaultShapes = true;
    bool verbose = false;
    std::string reportFormat = "text"; // Options: text, csv, json
    ComputePrecision::Policy precisionPolicy = ComputePrecision::Policy::Auto;
};

void printUsage() {
    std::cout << "Usage: mila_benchmark [options]\n";
    std::cout << "Options:\n";
    std::cout << "  --iterations <int>     Number of iterations for each benchmark (default: 100)\n";
    std::cout << "  --cpu-only             Only run CPU benchmarks\n";
    std::cout << "  --cuda-only            Only run CUDA benchmarks\n";
    std::cout << "  --no-modules           Skip module benchmarks\n";
    std::cout << "  --no-operations        Skip operation benchmarks\n";
    std::cout << "  --enable-kernels       Enable direct kernel benchmarks\n";
    std::cout << "  --output <file>        Write benchmark results to file\n";
    std::cout << "  --format <format>      Output format: text, csv, json (default: text)\n";
    std::cout << "  --precision <policy>   Precision policy: auto, performance, accuracy, disabled (default: auto)\n";
    std::cout << "  --shape <dims>         Custom shape to benchmark (comma-separated, e.g. 32,128)\n";
    std::cout << "                         Can be specified multiple times for different shapes\n";
    std::cout << "  --no-default-shapes    Don't use default benchmark shapes\n";
    std::cout << "  --verbose              Enable verbose output\n";
    std::cout << "  --help                 Show this help message\n";
}

bool parseCommandLine( int argc, char* argv[], BenchmarkConfig& config ) {
    for ( int i = 1; i < argc; i++ ) {
        std::string arg = argv[ i ];

        if ( arg == "--help" ) {
            printUsage();
            return false;
        }
        else if ( arg == "--iterations" && i + 1 < argc ) {
            config.iterations = std::stoi( argv[ ++i ] );
        }
        else if ( arg == "--cpu-only" ) {
            config.runCpu = true;
            config.runCuda = false;
        }
        else if ( arg == "--cuda-only" ) {
            config.runCpu = false;
            config.runCuda = true;
        }
        else if ( arg == "--no-modules" ) {
            config.runModuleBenchmarks = false;
        }
        else if ( arg == "--no-operations" ) {
            config.runOperationBenchmarks = false;
        }
        else if ( arg == "--enable-kernels" ) {
            config.runKernelBenchmarks = true;
        }
        else if ( arg == "--output" && i + 1 < argc ) {
            config.outputFile = argv[ ++i ];
        }
        else if ( arg == "--format" && i + 1 < argc ) {
            std::string format = argv[ ++i ];
            if ( format == "text" || format == "csv" || format == "json" ) {
                config.reportFormat = format;
            }
            else {
                std::cerr << "Unknown format: " << format << ". Using default: text" << std::endl;
            }
        }
        else if ( arg == "--precision" && i + 1 < argc ) {
            std::string precision = argv[ ++i ];
            if ( precision == "auto" ) {
                config.precisionPolicy = ComputePrecision::Policy::Auto;
            }
            else if ( precision == "performance" ) {
                config.precisionPolicy = ComputePrecision::Policy::Performance;
            }
            else if ( precision == "accuracy" ) {
                config.precisionPolicy = ComputePrecision::Policy::Accuracy;
            }
            else if ( precision == "disabled" ) {
                config.precisionPolicy = ComputePrecision::Policy::Disabled;
            }
            else {
                std::cerr << "Unknown precision policy: " << precision << ". Using default: auto" << std::endl;
            }
        }
        else if ( arg == "--shape" && i + 1 < argc ) {
            std::string shapeStr = argv[ ++i ];
            std::vector<size_t> shape;
            std::stringstream ss( shapeStr );
            std::string dim;

            while ( std::getline( ss, dim, ',' ) ) {
                shape.push_back( std::stoi( dim ) );
            }

            if ( !shape.empty() ) {
                config.customShapes.push_back( shape );
            }
        }
        else if ( arg == "--no-default-shapes" ) {
            config.useDefaultShapes = false;
        }
        else if ( arg == "--verbose" ) {
            config.verbose = true;
        }
        else if ( arg.substr( 0, 2 ) == "--" ) {
            std::cerr << "Unknown option: " << arg << std::endl;
            printUsage();
            return false;
        }
    }

    if ( config.verbose ) {
        std::cout << "Configuration:" << std::endl;
        std::cout << "  Iterations: " << config.iterations << std::endl;
        std::cout << "  Run CPU benchmarks: " << (config.runCpu ? "Yes" : "No") << std::endl;
        std::cout << "  Run CUDA benchmarks: " << (config.runCuda ? "Yes" : "No") << std::endl;
        std::cout << "  Run Module benchmarks: " << (config.runModuleBenchmarks ? "Yes" : "No") << std::endl;
        std::cout << "  Run Operation benchmarks: " << (config.runOperationBenchmarks ? "Yes" : "No") << std::endl;
        std::cout << "  Run Kernel benchmarks: " << (config.runKernelBenchmarks ? "Yes" : "No") << std::endl;

        // Print precision policy
        std::cout << "  Precision policy: ";
        switch ( config.precisionPolicy ) {
            case ComputePrecision::Policy::Auto:
                std::cout << "Auto";
                break;
            case ComputePrecision::Policy::Performance:
                std::cout << "Performance";
                break;
            case ComputePrecision::Policy::Accuracy:
                std::cout << "Accuracy";
                break;
            case ComputePrecision::Policy::Disabled:
                std::cout << "Disabled";
                break;
            default:
                std::cout << "Unknown";
        }
        std::cout << std::endl;

        if ( !config.outputFile.empty() ) {
            std::cout << "  Output file: " << config.outputFile << std::endl;
            std::cout << "  Output format: " << config.reportFormat << std::endl;
        }

        std::cout << "  Custom shapes: ";
        if ( config.customShapes.empty() ) {
            std::cout << "None";
        }
        else {
            for ( const auto& shape : config.customShapes ) {
                std::cout << "[";
                for ( size_t i = 0; i < shape.size(); ++i ) {
                    std::cout << shape[ i ];
                    if ( i < shape.size() - 1 ) std::cout << ",";
                }
                std::cout << "] ";
            }
        }
        std::cout << std::endl;
        std::cout << "  Use default shapes: " << (config.useDefaultShapes ? "Yes" : "No") << std::endl;
        std::cout << "  Note: CPU benchmarks will use only small shapes" << std::endl;
        std::cout << std::endl;
    }

    return true;
}

void writeResultsToFile( const std::vector<BenchmarkResult>& results, const std::string& filename,
    const std::string& format ) {
    std::ofstream file( filename );
    if ( !file.is_open() ) {
        throw std::runtime_error( "Failed to open output file: " + filename );
    }

    if ( format == "csv" ) {
        // Write CSV header
        file << BenchmarkResult::getCsvHeader() << std::endl;

        // Write each result as a CSV row
        for ( const auto& result : results ) {
            file << result.toCsv() << std::endl;
        }
    }
    else if ( format == "json" ) {
        // Write JSON array opening
        file << "[" << std::endl;

        // Write each result as a JSON object
        for ( size_t i = 0; i < results.size(); ++i ) {
            file << "  {" << std::endl;
            file << "    \"name\": " << std::quoted( results[ i ].name ) << "," << std::endl;
            file << "    \"time_ms\": " << results[ i ].time_ms << "," << std::endl;
            file << "    \"throughput_elements\": " << results[ i ].throughput_elements << "," << std::endl;
            file << "    \"throughput_gflops\": " << results[ i ].throughput_gflops << "," << std::endl;
            file << "    \"element_count\": " << results[ i ].elementCount << "," << std::endl;
            file << "    \"iterations\": " << results[ i ].iterations << "," << std::endl;
            file << "    \"device\": " << std::quoted( results[ i ].deviceName ) << "," << std::endl;
            file << "    \"notes\": " << std::quoted( results[ i ].notes ) << std::endl;
            file << "  }" << (i < results.size() - 1 ? "," : "") << std::endl;
        }

        // Write JSON array closing
        file << "]" << std::endl;
    }
    else {
        // Default to text format
        for ( const auto& result : results ) {
            file << result.toString() << std::endl;
        }
    }
}

int main( int argc, char* argv[] ) {
    try {
        std::cout << "Mila Benchmarking Tool" << std::endl;
        std::cout << "======================" << std::endl << std::endl;

        BenchmarkConfig config;
        if ( !parseCommandLine( argc, argv, config ) ) {
            return 1;
        }

        Mila::initialize();

        std::cout << "Mila version: " << Mila::getAPIVersion().ToString() << std::endl;

        auto devices = Compute::list_devices();
        std::cout << "Available devices: ";
        for ( const auto& device : devices ) {
            std::cout << device << " ";
        }
        std::cout << std::endl;

        std::shared_ptr<DeviceContext> cpuContext;
        std::shared_ptr<DeviceContext> cudaContext;

        if ( config.runCpu ) {
            cpuContext = std::make_shared<DeviceContext>( "CPU" );
        }

        if ( config.runCuda ) {
            try {
                cudaContext = std::make_shared<DeviceContext>( "CUDA:0" );
            }
            catch ( const std::exception& e ) {
                std::cerr << "CUDA error: " << e.what() << std::endl;
                if ( !config.runCpu ) {
                    std::cerr << "No valid device available. Exiting." << std::endl;
                    return 1;
                }
                config.runCuda = false;
                std::cerr << "Falling back to CPU only." << std::endl;
            }
        }

        BenchmarkManager manager;


        // Define benchmark shapes - separate shapes for CPU and CUDA
        std::vector<std::vector<size_t>> cpuBenchmarkShapes;
        std::vector<std::vector<size_t>> cudaBenchmarkShapes;

        // Add default shapes if requested
        if ( config.useDefaultShapes ) {
            // CPU gets only small shapes
            cpuBenchmarkShapes = {
                {32, 128},              // Small
                {8, 32, 128}            // Small 3D shape
            };

            // CUDA gets the full range of shapes
            cudaBenchmarkShapes = {
                {32, 128},              // Small
                {64, 1024},             // Medium
                {128, 4096},            // Large
                {8, 32, 128}            // 3D shape (batch, seq, hidden)
            };
        }

        // Add custom shapes if any - only add small custom shapes to CPU
        if ( !config.customShapes.empty() ) {
            for ( const auto& shape : config.customShapes ) {
                // For CPU, only add shapes that are "small enough"
                // Simple heuristic: total elements < 50,000
                size_t totalElements = 1;
                for ( size_t dim : shape ) {
                    totalElements *= dim;
                }

                // Add to CPU shapes if small enough
                if ( totalElements < 50000 ) {
                    cpuBenchmarkShapes.push_back( shape );
                }

                // Always add to CUDA shapes
                cudaBenchmarkShapes.push_back( shape );
            }
        }

        // Check if we have valid shapes
        if ( (config.runCpu && cpuBenchmarkShapes.empty()) ||
            (config.runCuda && cudaBenchmarkShapes.empty()) ) {
            std::cerr << "No valid benchmark shapes defined. Please specify shapes or enable default shapes." << std::endl;
            return 1;
        }

        std::cout << "Using " << cpuBenchmarkShapes.size() << " shapes for CPU benchmarks" << std::endl;
        if ( config.verbose && config.runCpu ) {
            std::cout << "CPU shapes: ";
            for ( const auto& shape : cpuBenchmarkShapes ) {
                std::cout << "[";
                for ( size_t i = 0; i < shape.size(); ++i ) {
                    std::cout << shape[ i ];
                    if ( i < shape.size() - 1 ) std::cout << ",";
                }
                std::cout << "] ";
            }
            std::cout << std::endl;
        }

        // Add benchmarks based on configuration
        if ( config.runModuleBenchmarks ) {
            // Module benchmarks (Gelu, etc.)
            if ( config.runCpu ) {
                for ( const auto& shape : cpuBenchmarkShapes ) {
                    // CPU Gelu Module
                    auto cpuGelu = std::make_shared<CpuGelu<float>>(
                        "Cpu::Gelu", cpuContext, false, config.precisionPolicy );

                    auto cpuGeluBench = std::make_unique<ModuleBenchmark<DeviceType::Cpu, float>>(
                        cpuGelu,
                        shape,
                        shape,
                        cpuContext
                    );
                    manager.addBenchmark( std::move( cpuGeluBench ) );
                }
            }

            if ( config.runCuda ) {
                for ( const auto& shape : cudaBenchmarkShapes ) {
                    auto cudaGelu = std::make_shared<CudaGelu<float>>(
                        "Cuda::Gelu", cudaContext, false, config.precisionPolicy );

                    auto cudaGeluBench = std::make_unique<ModuleBenchmark<DeviceType::Cuda, float>>(
                        cudaGelu,
                        shape,
                        shape,
                        cudaContext
                    );
                    manager.addBenchmark( std::move( cudaGeluBench ) );
                }
            }

            // MLP Module benchmarks
            if ( config.runCpu ) {
                for ( const auto& shape : cpuBenchmarkShapes ) {
                    if ( shape.size() >= 2 ) {  // MLP requires at least 2D input
                        size_t input_features = shape.back();
                        size_t hidden_features = input_features * 4;  // Common ratio in transformers

                        // CPU MLP
                        auto cpuMLP = std::make_shared<CpuMLP<float>>(
                            "Cpu::MLP", cpuContext, shape, hidden_features,
                            true, false, config.precisionPolicy );

                        auto cpuMLPBench = std::make_unique<BlockModuleBenchmark<DeviceType::Cpu, float>>(
                            cpuMLP,
                            shape,
                            cpuContext
                        );
                        manager.addBenchmark( std::move( cpuMLPBench ) );
                    }
                }
            }

            if ( config.runCuda ) {
                for ( const auto& shape : cudaBenchmarkShapes ) {
                    if ( shape.size() >= 2 ) {  // MLP requires at least 2D input
                        size_t input_features = shape.back();
                        size_t hidden_features = input_features * 4;  // Common ratio in transformers

                        // CUDA MLP
                        auto cudaMLP = std::make_shared<CudaMLP<float>>(
                            "Cuda::MLP", cudaContext, shape, hidden_features,
                            true, false, config.precisionPolicy );

                        auto cudaMLPBench = std::make_unique<BlockModuleBenchmark<DeviceType::Cuda, float>>(
                            cudaMLP,
                            shape,
                            cudaContext
                        );
                        manager.addBenchmark( std::move( cudaMLPBench ) );
                    }
                }
            }

            // Residual benchmarks
            if ( config.runCpu ) {
                for ( const auto& shape : cpuBenchmarkShapes ) {
                    // CPU Residual
                    auto cpuResidual = std::make_shared<Residual<DeviceType::Cpu, float>>(
                        "Cpu::Residual", cpuContext, false, config.precisionPolicy );

                    auto cpuResidualBench = std::make_unique<BinaryModuleBenchmark<DeviceType::Cpu, float>>(
                        cpuResidual,
                        shape,
                        cpuContext
                    );
                    manager.addBenchmark( std::move( cpuResidualBench ) );
                }
            }

            if ( config.runCuda ) {
                for ( const auto& shape : cudaBenchmarkShapes ) {
                    // CUDA Residual
                    auto cudaResidual = std::make_shared<Residual<DeviceType::Cuda, float>>(
                        "Cuda::Residual", cudaContext, false, config.precisionPolicy );

                    auto cudaResidualBench = std::make_unique<BinaryModuleBenchmark<DeviceType::Cuda, float>>(
                        cudaResidual,
                        shape,
                        cudaContext
                    );
                    manager.addBenchmark( std::move( cudaResidualBench ) );
                }
            }
        }

        if ( config.runOperationBenchmarks ) {
            // Operation benchmarks (GeluOp, etc.)
            if ( config.runCpu ) {
                for ( const auto& shape : cpuBenchmarkShapes ) {
                    auto cpuGeluOp = std::static_pointer_cast<OperationBase<DeviceType::Cpu, float>>(
                        std::make_shared<CpuGeluOp>( cpuContext, config.precisionPolicy ));

                    manager.addBenchmark( std::make_unique<OperationBenchmark<DeviceType::Cpu, float>>(
                        cpuGeluOp, "Cpu::GeluOp", shape, cpuContext, config.precisionPolicy ) );
                }
            }

            if ( config.runCuda ) {
                for ( const auto& shape : cudaBenchmarkShapes ) {
                    // CUDA GeluOp
                    auto cudaGeluOp = std::static_pointer_cast<OperationBase<DeviceType::Cuda, float>>(
                        std::make_shared<CudaGeluOp<float>>( cudaContext, config.precisionPolicy ));

                    manager.addBenchmark( std::make_unique<OperationBenchmark<DeviceType::Cuda, float>>(
                        cudaGeluOp, "Cuda::GeluOp", shape, cudaContext, config.precisionPolicy ) );
                }
            }
        }

        if ( config.runKernelBenchmarks && config.runCuda ) {
            // Direct CUDA kernel benchmarks
            for ( const auto& shape : cudaBenchmarkShapes ) {
                // CUDA Gelu Kernel
                /*auto geluKernel = [](float* Y, const float* X, int N, cudaStream_t stream) {
                    cuda_gelu_forward_fp32(Y, X, N, stream);
                };

                manager.addBenchmark(std::make_unique<KernelBenchmark<float>>(
                    geluKernel, "cuda_gelu_forward_fp32", shape, cudaContext));*/
            }
        }

        auto results = manager.runAll( config.iterations );

        if ( !config.outputFile.empty() ) {
            std::cout << "Writing results to " << config.outputFile << std::endl;
            writeResultsToFile( results, config.outputFile, config.reportFormat );
        }

        return 0;
    }
    catch ( const std::exception& e ) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}