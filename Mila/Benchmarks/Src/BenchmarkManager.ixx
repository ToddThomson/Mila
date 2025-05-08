module;
#include <chrono>
#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <cuda_runtime.h>

export module Mila.BenchmarkManager;

import Mila;
import Mila.Benchmark;

namespace Mila::Benchmark
{
    export class BenchmarkManager {
    public:
        void addBenchmark( std::unique_ptr<Benchmark> benchmark ) {
            benchmarks_.push_back( std::move( benchmark ) );
        }

        void runAll( size_t iterations ) {
            std::vector<BenchmarkResult> results;

            std::cout << "Running " << benchmarks_.size() << " benchmarks with "
                << iterations << " iterations each..." << std::endl << std::endl;

            // Header
            std::cout << std::left << std::setw( 40 ) << "Benchmark" << " | "
                << std::right << std::setw( 8 ) << "Time (ms)" << " | "
                << std::setw( 14 ) << "Throughput" << " | "
                << "Device" << std::endl;

            std::cout << std::string( 80, '-' ) << std::endl;

            // Run each benchmark
            for ( auto& benchmark : benchmarks_ ) {
                auto result = benchmark->run( iterations );
                results.push_back( result );

                // Print result
                std::cout << result.toString() << std::endl;
            }

            std::cout << std::endl;
        }

    private:
        std::vector<std::unique_ptr<Benchmark>> benchmarks_;
    };
}
