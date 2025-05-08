module;
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>

export module Mila.BenchmarkDefinitions;

import Mila;

import Mila.Benchmark;
import Mila.BenchmarkManager;
import Mila.Benchmark.ModuleBenchmark;
import Mila.Benchmark.BlockModuleBenchmark;
import Mila.Benchmark.BinaryModuleBenchmark;
import Mila.Benchmark.OperationBenchmark;
import Mila.Benchmark.KernelBenchmark;

namespace Mila::Benchmark
{
    using json = nlohmann::json;

    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Benchmark;

    // Define a structure to hold benchmark definitions from JSON
    struct BenchmarkDefinition {
        std::string type;            // "module", "operation", or "kernel"
        std::string name;            // Name of the benchmark
        std::string moduleType;      // For modules: "gelu", "mlp", "residual", etc.
        std::string operationType;   // For operations: "geluOp", etc.
        std::string kernelType;      // For kernels: specific kernel name
        std::vector<int> parameters; // Additional parameters (e.g., hidden features for MLP)
    };

    // Parse shapes from JSON array
    std::vector<size_t> parseShapeFromJson( const json& shapeJson ) {
        std::vector<size_t> shape;
        for ( const auto& dim : shapeJson ) {
            shape.push_back( dim.get<size_t>() );
        }
        return shape;
    }

    // Load benchmark definitions from JSON file
    std::vector<BenchmarkDefinition> loadBenchmarkDefinitions( const std::string& filename ) {
        std::vector<BenchmarkDefinition> definitions;

        try {
            std::ifstream file( filename );
            if ( !file.is_open() ) {
                throw std::runtime_error( "Could not open benchmark definition file: " + filename );
            }

            json benchmarkJson;
            file >> benchmarkJson;

            for ( const auto& item : benchmarkJson[ "benchmarks" ] ) {
                BenchmarkDefinition def;
                def.type = item[ "type" ].get<std::string>();
                def.name = item[ "name" ].get<std::string>();

                if ( def.type == "module" ) {
                    def.moduleType = item[ "moduleType" ].get<std::string>();

                    // Load optional parameters if present
                    if ( item.contains( "parameters" ) ) {
                        for ( const auto& param : item[ "parameters" ] ) {
                            def.parameters.push_back( param.get<int>() );
                        }
                    }
                }
                else if ( def.type == "operation" ) {
                    def.operationType = item[ "operationType" ].get<std::string>();
                }
                else if ( def.type == "kernel" ) {
                    def.kernelType = item[ "kernelType" ].get<std::string>();
                }

                definitions.push_back( def );
            }
        }
        catch ( const std::exception& e ) {
            std::cerr << "Error loading benchmark definitions: " << e.what() << std::endl;
            throw;
        }

        return definitions;
    }

    // Add benchmarks from JSON definitions
    void addBenchmarksFromDefinitions(
        BenchmarkManager& manager,
        const std::vector<BenchmarkDefinition>& definitions,
        const std::vector<std::vector<size_t>>& cpuShapes,
        const std::vector<std::vector<size_t>>& cudaShapes,
        std::shared_ptr<DeviceContext> cpuContext,
        std::shared_ptr<DeviceContext> cudaContext,
        bool runCpu,
        bool runCuda
    ) {
        for ( const auto& def : definitions ) {
            if ( def.type == "module" ) {
                // Module benchmarks
                if ( def.moduleType == "gelu" ) {
                    if ( runCpu ) {
                        for ( const auto& shape : cpuShapes ) {
                            auto cpuGelu = std::make_shared<Gelu<float, DeviceType::Cpu>>(
                                "Cpu::" + def.name, cpuContext );

                            auto cpuGeluBench = std::make_unique<ModuleBenchmark<float, float, DeviceType::Cpu>>(
                                cpuGelu,
                                shape,
                                shape,
                                cpuContext
                            );
                            manager.addBenchmark( std::move( cpuGeluBench ) );
                        }
                    }

                    if ( runCuda ) {
                        for ( const auto& shape : cudaShapes ) {
                            auto cudaGelu = std::make_shared<Gelu<float, DeviceType::Cuda>>(
                                "Cuda::" + def.name, cudaContext );

                            auto cudaGeluBench = std::make_unique<ModuleBenchmark<float, float, DeviceType::Cuda>>(
                                cudaGelu,
                                shape,
                                shape,
                                cudaContext
                            );
                            manager.addBenchmark( std::move( cudaGeluBench ) );
                        }
                    }
                }
                else if ( def.moduleType == "mlp" ) {
                    // Get hidden feature size from parameters, default to 4x input size if not provided
                    int hiddenFeaturesMultiplier = def.parameters.size() > 0 ? def.parameters[ 0 ] : 4;
                    bool useGelu = def.parameters.size() > 1 ? (def.parameters[ 1 ] != 0) : true;
                    bool useBias = def.parameters.size() > 2 ? (def.parameters[ 2 ] != 0) : false;

                    if ( runCpu ) {
                        for ( const auto& shape : cpuShapes ) {
                            if ( shape.size() >= 2 ) {  // MLP requires at least 2D input
                                size_t input_features = shape.back();
                                size_t hidden_features = input_features * hiddenFeaturesMultiplier;

                                auto cpuMLP = std::make_shared<MLP<float, DeviceType::Cpu>>(
                                    "Cpu::" + def.name, cpuContext, shape, hidden_features, useGelu, useBias );

                                auto cpuMLPBench = std::make_unique<BlockModuleBenchmark<float, DeviceType::Cpu>>(
                                    cpuMLP,
                                    shape,
                                    cpuContext
                                );
                                manager.addBenchmark( std::move( cpuMLPBench ) );
                            }
                        }
                    }

                    if ( runCuda ) {
                        for ( const auto& shape : cudaShapes ) {
                            if ( shape.size() >= 2 ) {  // MLP requires at least 2D input
                                size_t input_features = shape.back();
                                size_t hidden_features = input_features * hiddenFeaturesMultiplier;

                                auto cudaMLP = std::make_shared<MLP<float, DeviceType::Cuda>>(
                                    "Cuda::" + def.name, cudaContext, shape, hidden_features, useGelu, useBias );

                                auto cudaMLPBench = std::make_unique<BlockModuleBenchmark<float, DeviceType::Cuda>>(
                                    cudaMLP,
                                    shape,
                                    cudaContext
                                );
                                manager.addBenchmark( std::move( cudaMLPBench ) );
                            }
                        }
                    }
                }
                else if ( def.moduleType == "residual" ) {
                    if ( runCpu ) {
                        for ( const auto& shape : cpuShapes ) {
                            auto cpuResidual = std::make_shared<Residual<float, DeviceType::Cpu>>(
                                "Cpu::" + def.name, cpuContext );

                            auto cpuResidualBench = std::make_unique<BinaryModuleBenchmark<float, DeviceType::Cpu>>(
                                cpuResidual,
                                shape,
                                cpuContext
                            );
                            manager.addBenchmark( std::move( cpuResidualBench ) );
                        }
                    }

                    if ( runCuda ) {
                        for ( const auto& shape : cudaShapes ) {
                            auto cudaResidual = std::make_shared<Residual<float, DeviceType::Cuda>>(
                                "Cuda::" + def.name, cudaContext );

                            auto cudaResidualBench = std::make_unique<BinaryModuleBenchmark<float, DeviceType::Cuda>>(
                                cudaResidual,
                                shape,
                                cudaContext
                            );
                            manager.addBenchmark( std::move( cudaResidualBench ) );
                        }
                    }
                }
                // Add other module types as needed
            }
            else if ( def.type == "operation" ) {
                if ( def.operationType == "geluOp" ) {
                    if ( runCpu ) {
                        for ( const auto& shape : cpuShapes ) {
                            auto cpuGeluOp = std::static_pointer_cast<OperationBase<float, float, DeviceType::Cpu>>(
                                std::make_shared<CpuGeluOp>( cpuContext ));
                            manager.addBenchmark( std::make_unique<OperationBenchmark<float, DeviceType::Cpu>>(
                                cpuGeluOp, "Cpu::" + def.name, shape, cpuContext ) );
                        }
                    }

                    if ( runCuda ) {
                        for ( const auto& shape : cudaShapes ) {
                            auto cudaGeluOp = std::static_pointer_cast<OperationBase<float, float, DeviceType::Cuda>>(
                                std::make_shared<CudaGeluOp<float>>( cudaContext ));
                            manager.addBenchmark( std::make_unique<OperationBenchmark<float, DeviceType::Cuda>>(
                                cudaGeluOp, "Cuda::" + def.name, shape, cudaContext ) );
                        }
                    }
                }
                // Add other operations as needed
            }
            else if ( def.type == "kernel" && runCuda ) {
                // CUDA kernel benchmarks - add implementations as needed
                /*
                if (def.kernelType == "gelu_forward") {
                    for (const auto& shape : cudaShapes) {
                        auto geluKernel = [](float* Y, const float* X, int N, cudaStream_t stream) {
                            cuda_gelu_forward_fp32(Y, X, N, stream);
                        };

                        manager.addBenchmark(std::make_unique<KernelBenchmark<float>>(
                            geluKernel, def.name, shape, cudaContext));
                    }
                }
                */
            }
        }
    }

}