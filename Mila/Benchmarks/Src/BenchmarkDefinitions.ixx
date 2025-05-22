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
        std::string precision = "auto"; // Precision policy: "auto", "performance", "accuracy", "disabled"
    };

    // Parse shapes from JSON array
    std::vector<size_t> parseShapeFromJson( const json& shapeJson ) {
        std::vector<size_t> shape;
        for ( const auto& dim : shapeJson ) {
            shape.push_back( dim.get<size_t>() );
        }
        return shape;
    }

    // Convert string precision policy to enum
    ComputePrecision::Policy stringToPrecisionPolicy( const std::string& precision ) {
        if ( precision == "performance" ) return ComputePrecision::Policy::Performance;
        if ( precision == "accuracy" ) return ComputePrecision::Policy::Accuracy;
        if ( precision == "disabled" ) return ComputePrecision::Policy::Disabled;
        return ComputePrecision::Policy::Auto; // Default
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

                // Load precision policy if present
                if ( item.contains( "precision" ) ) {
                    def.precision = item[ "precision" ].get<std::string>();
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
        bool runCuda,
        ComputePrecision::Policy defaultPrecision = ComputePrecision::Policy::Auto
    ) {
        for ( const auto& def : definitions ) {
            // Get precision policy from definition or use default
            ComputePrecision::Policy precisionPolicy =
                def.precision == "auto" ? defaultPrecision : stringToPrecisionPolicy( def.precision );

            if ( def.type == "module" ) {
                // Module benchmarks
                if ( def.moduleType == "gelu" ) {
                    if ( runCpu ) {
                        for ( const auto& shape : cpuShapes ) {
                            // Updated constructor order: name, context, precision, is_training
                            auto cpuGelu = std::make_shared<CpuGelu<float>>(
                                "Cpu::" + def.name, cpuContext,
                                precisionPolicy, // Precision before is_training
                                false );

                            auto cpuGeluBench = std::make_unique<ModuleBenchmark<DeviceType::Cpu, float>>(
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
                            // Updated constructor order: name, context, precision, is_training
                            auto cudaGelu = std::make_shared<CudaGelu<float>>(
                                "Cuda::" + def.name, cudaContext,
                                precisionPolicy, // Precision before is_training
                                false );

                            auto cudaGeluBench = std::make_unique<ModuleBenchmark<DeviceType::Cuda, float>>(
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

                                // Updated constructor order: name, context, input_shape, output_channels, has_bias, precision, is_training
                                auto cpuMLP = std::make_shared<CpuMLP<float>>(
                                    "Cpu::" + def.name, cpuContext, shape, hidden_features,
                                    useBias, // has_bias
                                    precisionPolicy, // Precision before is_training
                                    false );

                                auto cpuMLPBench = std::make_unique<BlockModuleBenchmark<DeviceType::Cpu, float>>(
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

                                // Updated constructor order: name, context, input_shape, output_channels, has_bias, precision, is_training
                                auto cudaMLP = std::make_shared<CudaMLP<float>>(
                                    "Cuda::" + def.name, cudaContext, shape, hidden_features,
                                    useBias, // has_bias
                                    precisionPolicy, // Precision before is_training
                                    false );

                                auto cudaMLPBench = std::make_unique<BlockModuleBenchmark<DeviceType::Cuda, float>>(
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
                            // Updated constructor order: name, context, precision, is_training
                            auto cpuResidual = std::make_shared<Residual<DeviceType::Cpu, float>>(
                                "Cpu::" + def.name, cpuContext,
                                precisionPolicy, // Precision before is_training
                                false );

                            auto cpuResidualBench = std::make_unique<BinaryModuleBenchmark<DeviceType::Cpu, float>>(
                                cpuResidual,
                                shape,
                                cpuContext
                            );
                            manager.addBenchmark( std::move( cpuResidualBench ) );
                        }
                    }

                    if ( runCuda ) {
                        for ( const auto& shape : cudaShapes ) {
                            // Updated constructor order: name, context, precision, is_training
                            auto cudaResidual = std::make_shared<Residual<DeviceType::Cuda, float>>(
                                "Cuda::" + def.name, cudaContext,
                                precisionPolicy, // Precision before is_training
                                false );

                            auto cudaResidualBench = std::make_unique<BinaryModuleBenchmark<DeviceType::Cuda, float>>(
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
                            auto cpuGeluOp = std::static_pointer_cast<OperationBase<DeviceType::Cpu, float>>(
                                std::make_shared<CpuGeluOp>( cpuContext, precisionPolicy ));

                            manager.addBenchmark( std::make_unique<OperationBenchmark<DeviceType::Cpu, float>>(
                                cpuGeluOp, "Cpu::" + def.name, shape, cpuContext, precisionPolicy ) );
                        }
                    }

                    if ( runCuda ) {
                        for ( const auto& shape : cudaShapes ) {
                            auto cudaGeluOp = std::static_pointer_cast<OperationBase<DeviceType::Cuda, float>>(
                                std::make_shared<CudaGeluOp<float>>( cudaContext, precisionPolicy ));

                            manager.addBenchmark( std::make_unique<OperationBenchmark<DeviceType::Cuda, float>>(
                                cudaGeluOp, "Cuda::" + def.name, shape, cudaContext, precisionPolicy ) );
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