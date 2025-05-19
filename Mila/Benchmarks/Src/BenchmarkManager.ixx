module;
#include <chrono>
#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <any>

export module Mila.BenchmarkManager;

import Mila;
import Mila.Benchmark;

namespace Mila::Benchmark
{
    /**
     * @brief Manager class for running multiple benchmarks.
     *
     * This class manages a collection of benchmarks, runs them with specified parameters,
     * and collects their results.
     */
    export class BenchmarkManager {
    public:
        /**
         * @brief Adds a benchmark to the manager.
         *
         * @param benchmark The benchmark to add.
         */
        void addBenchmark( std::unique_ptr<Benchmark> benchmark ) {
            benchmarks_.push_back( std::move( benchmark ) );
        }

        /**
         * @brief Runs all benchmarks and returns their results.
         *
         * @param iterations The number of iterations to run each benchmark.
         * @return std::vector<BenchmarkResult> The results of all benchmarks.
         */
        std::vector<BenchmarkResult> runAll( size_t iterations ) {
            std::vector<BenchmarkResult> results;
            results.reserve( benchmarks_.size() );

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
                // Skip if not applicable to this system
                if ( !benchmark->isApplicable() ) {
                    std::cout << "Skipping " << benchmark->name() << " - not applicable on this system" << std::endl;
                    continue;
                }

                auto result = benchmark->run( iterations );
                results.push_back( result );

                // Print result
                std::cout << result.toString() << std::endl;
            }

            std::cout << std::endl;

            // Store the results for later retrieval
            lastResults_ = results;

            return results;
        }

        /**
         * @brief Runs all benchmarks and writes results to a file.
         *
         * @param iterations The number of iterations to run each benchmark.
         * @param outputFile Path to the output file.
         * @param format The output format: "text", "csv", or "json".
         * @return std::vector<BenchmarkResult> The results of all benchmarks.
         */
        std::vector<BenchmarkResult> runAllToFile( size_t iterations, const std::string& outputFile, const std::string& format ) {
            std::vector<BenchmarkResult> results = runAll( iterations );

            // Write results to file
            writeResultsToFile( results, outputFile, format );

            return results;
        }

        /**
         * @brief Gets the results from the most recent benchmark run.
         *
         * @return const std::vector<BenchmarkResult>& The benchmark results.
         */
        const std::vector<BenchmarkResult>& getLastResults() const {
            return lastResults_;
        }

        /**
         * @brief Writes benchmark results to a file.
         *
         * @param results The benchmark results to write.
         * @param filename The path to the output file.
         * @param format The output format: "text", "csv", or "json".
         */
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

                    // Include notes if not empty
                    if ( !results[ i ].notes.empty() ) {
                        file << "    \"notes\": " << std::quoted( results[ i ].notes ) << "," << std::endl;
                    }

                    // Include precision policy if present in properties
                    if ( results[ i ].properties.find( "precision_policy" ) != results[ i ].properties.end() ) {
                        int policy = std::any_cast<int>(results[ i ].properties.at( "precision_policy" ));
                        file << "    \"precision_policy\": " << policy << std::endl;
                    }
                    else {
                        file << "    \"precision_policy\": 0" << std::endl;  // Auto precision
                    }

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

        /**
         * @brief Gets the number of benchmarks in the manager.
         *
         * @return size_t The number of benchmarks.
         */
        size_t benchmarkCount() const {
            return benchmarks_.size();
        }

    private:
        std::vector<std::unique_ptr<Benchmark>> benchmarks_;
        std::vector<BenchmarkResult> lastResults_; ///< Results from the most recent run
    };
}