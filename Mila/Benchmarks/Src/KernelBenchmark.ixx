// File: Mila/Benchmarks/KernelBenchmarks.ixx
module;
#include <vector>
#include <memory>
#include <string>
#include <functional>
#include <stdexcept>
#include <cuda_runtime.h>

export module Mila.Benchmark.KernelBenchmark;

import Mila;
import Mila.Benchmark;

namespace Mila::Benchmark
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    // For CUDA kernels, we'll create a specialized benchmark
    // This will directly measure kernel performance without the operation overhead
    export template<typename TPrecision>
        class KernelBenchmark : public Benchmark {
        public:
            using KernelFunction = std::function<void( TPrecision*, const TPrecision*, int, cudaStream_t )>;

            KernelBenchmark( KernelFunction kernelFunc,
                std::string kernelName,
                std::vector<size_t> inputShape,
                std::shared_ptr<DeviceContext> context )
                : kernelFunc_( kernelFunc ), kernelName_( kernelName ), inputShape_( inputShape ) {

                this->deviceContext_ = context;

                // Ensure CUDA device context
                if ( context->getDevice()->getDeviceType() != DeviceType::Cuda ) {
                    throw std::runtime_error( "KernelBenchmark requires CUDA device context" );
                }

                // Calculate total elements
                size_t totalElements = 1;
                for ( auto dim : inputShape ) {
                    totalElements *= dim;
                }

                // Allocate device memory
                cudaMalloc( &d_input_, sizeof( TPrecision ) * totalElements );
                cudaMalloc( &d_output_, sizeof( TPrecision ) * totalElements );

                // Create host tensor for initialization
                std::vector<TPrecision> h_input( totalElements );

                // Initialize with random values
                for ( size_t i = 0; i < totalElements; ++i ) {
                    h_input[ i ] = static_cast<TPrecision>( rand() ) / RAND_MAX * 2.0f - 1.0f;
                }

                // Copy to device
                cudaMemcpy( d_input_, h_input.data(), sizeof( TPrecision ) * totalElements, cudaMemcpyHostToDevice );

                // Store element count
                elementCount_ = totalElements;
            }

            ~KernelBenchmark() {
                if ( d_input_ ) cudaFree( d_input_ );
                if ( d_output_ ) cudaFree( d_output_ );
            }

            BenchmarkResult run( size_t iterations ) override {
                BenchmarkResult result;
                result.name = name();
                result.iterations = iterations;
                result.elementCount = elementCount_;
                result.deviceName = "CUDA";

                // Get CUDA stream from device context
                cudaStream_t stream = deviceContext_->getStream();

                // Measure time
                result.time_ms = measureExecutionTime( [this, stream]() {
                    kernelFunc_( d_output_, d_input_, elementCount_, stream );
                    }, iterations );

                // Calculate throughput metrics
                result.throughput_elements = static_cast<double>(elementCount_) / (result.time_ms / 1000.0);

                // Estimate FLOPs for specific kernels
                double flops_per_element = 0;
                if ( kernelName_.find( "gelu" ) != std::string::npos ) {
                    flops_per_element = 15;  // Approximate FLOPs for GELU
                }
                else if ( kernelName_.find( "softmax" ) != std::string::npos ) {
                    flops_per_element = 5;   // Approximate FLOPs for Softmax
                }

                if ( flops_per_element > 0 ) {
                    result.throughput_gflops = static_cast<double>(elementCount_ * flops_per_element) /
                        (result.time_ms / 1000.0) / 1e9;
                }

                return result;
            }

            std::string name() const override {
                std::ostringstream oss;
                oss << "Kernel::" << kernelName_ << " [";
                for ( size_t i = 0; i < inputShape_.size(); ++i ) {
                    oss << inputShape_[ i ];
                    if ( i < inputShape_.size() - 1 ) {
                        oss << "×";
                    }
                }
                oss << "]";
                return oss.str();
            }

        private:
            KernelFunction kernelFunc_;
            std::string kernelName_;
            std::vector<size_t> inputShape_;
            TPrecision* d_input_ = nullptr;
            TPrecision* d_output_ = nullptr;
            size_t elementCount_;
    };
}
