// File: Mila/Benchmarks/Benchmark.ixx
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
#include <map>
#include <any>

export module Mila.Benchmark;

import Mila;

namespace Mila::Benchmark
{
    export struct BenchmarkResult {
        std::string name;
        double time_ms = 0.0;                 // Execution time in milliseconds
        double throughput_elements = 0.0;     // Elements processed per second
        double throughput_gflops = 0.0;       // GFLOPS (if applicable)
        size_t elementCount = 0;              // Number of elements processed
        size_t iterations = 0;                // Number of iterations performed
        std::string deviceName;               // Device used for benchmark
        double totalRunTimeMs = 0.0;          // Total run time in milliseconds
        std::string notes;                    // Additional notes about the benchmark
        std::map<std::string, std::any> properties; // Additional properties

        std::string toString() const {
            std::ostringstream oss;
            oss << std::left << std::setw( 40 ) << name << " | "
                << std::right << std::setw( 8 ) << std::fixed << std::setprecision( 3 ) << time_ms << " ms | "
                << std::setw( 10 ) << std::fixed << std::setprecision( 2 );

            if ( throughput_gflops > 0 )
                oss << throughput_gflops << " GFLOPS";
            else
                oss << throughput_elements / 1e6 << " M elem/s";

            oss << " | " << deviceName;
            if ( iterations != 0 ) {
                oss << " | " << iterations << " iters";
            }

            if ( !notes.empty() ) {
                oss << " | " << notes;
            }

            return oss.str();
        }

        // Returns a CSV-formatted string with headers
        static std::string getCsvHeader() {
            return "Name,Time(ms),Elements,GFLOPS,ElementsPerSec,Device,Iterations,Notes";
        }

        // Returns a CSV-formatted row for this result
        std::string toCsv() const {
            std::ostringstream oss;
            oss << std::quoted( name ) << ","
                << time_ms << ","
                << elementCount << ","
                << throughput_gflops << ","
                << throughput_elements << ","
                << std::quoted( deviceName ) << ","
                << iterations << ","
                << std::quoted( notes );
            return oss.str();
        }
    };

    export class Benchmark {
    public:
        virtual ~Benchmark() = default;

        virtual BenchmarkResult run( size_t requestedIterations ) = 0;

        virtual std::string name() const = 0;

        // Returns true if this benchmark should run on the current system
        // (e.g., CUDA benchmarks should not run on systems without CUDA)
        virtual bool isApplicable() const {
            return true;
        }

    protected:
        template<typename Func>
        double measureExecutionTime( Func&& func, size_t iterations ) {
            // Warm-up phase to avoid initial delays
            for ( size_t i = 0; i < 3; i++ ) {
                func();
            }

            if ( deviceContext_->getDevice()->getDeviceType() == Dnn::Compute::DeviceType::Cuda ) {
                cudaDeviceSynchronize();
            }

            // Actual measurement phase
            auto start = std::chrono::high_resolution_clock::now();

            for ( size_t i = 0; i < iterations; i++ ) {
                func();
            }

            // Synchronize device if CUDA
            if ( deviceContext_->getDevice()->getDeviceType() == Dnn::Compute::DeviceType::Cuda ) {
                cudaDeviceSynchronize();
            }

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>( end - start );

            return duration.count() / 1000.0 / static_cast<double>(iterations); // Return time in ms
        }

        std::shared_ptr<Dnn::Compute::DeviceContext> deviceContext_;
    };
}