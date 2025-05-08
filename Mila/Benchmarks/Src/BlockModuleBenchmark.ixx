// Fix for Mila/Benchmarks/Src/BlockModuleBenchmark.ixx

module;
#include <vector>
#include <memory>
#include <string>
#include <functional>

export module Mila.Benchmark.BlockModuleBenchmark;

import Mila;
import Mila.Benchmark;

namespace Mila::Benchmark
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    // For block modules (like MLP), create a specialized benchmark
    export template<typename TPrecision, DeviceType TDeviceType = DeviceType::Cuda>
        class BlockModuleBenchmark : public Benchmark {
        public:
            template <typename TBlockModule>
            BlockModuleBenchmark( std::shared_ptr<TBlockModule> blockModule,
                std::vector<size_t> inputShape,
                std::shared_ptr<DeviceContext> context )
                : inputShape_( inputShape ), moduleName_( blockModule->getName() ) {

                this->deviceContext_ = context;

                // Store the forward function as a lambda that captures the module
                forwardFunc_ = [blockModule]( const auto& input, auto& output ) {
                    blockModule->forward( input, output );
                    };

                // Store module name for MLP check
                isMLP_ = (moduleName_.find( "MLP" ) != std::string::npos);

                // Create input and output tensors
                if constexpr ( TDeviceType == DeviceType::Cuda ) {
                    input_ = Tensor<TPrecision, CudaMemoryResource>( inputShape_ );
                    output_ = Tensor<TPrecision, CudaMemoryResource>( inputShape_ );

                    // Create host tensor for initialization
                    Tensor<TPrecision, HostMemoryResource> hostInput( inputShape_ );

                    // Initialize with random values
                    for ( size_t i = 0; i < hostInput.size(); ++i ) {
                        hostInput.data()[ i ] = static_cast<TPrecision>( rand() ) / RAND_MAX * 2.0f - 1.0f;
                    }

                    // Copy to device
                    input_.copyFrom( hostInput );
                }
                else {
                    input_ = Tensor<TPrecision, HostMemoryResource>( inputShape_ );
                    output_ = Tensor<TPrecision, HostMemoryResource>( inputShape_ );

                    // Initialize with random values
                    for ( size_t i = 0; i < input_.size(); ++i ) {
                        input_.data()[ i ] = static_cast<TPrecision>( rand() ) / RAND_MAX * 2.0f - 1.0f;
                    }
                }
            }

            BenchmarkResult run( size_t iterations ) override {
                BenchmarkResult result;
                result.name = name();
                result.iterations = iterations;
                result.elementCount = input_.size();
                result.deviceName = deviceToString( deviceContext_->getDevice()->getDeviceType() );

                // Measure time
                result.time_ms = measureExecutionTime( [this]() {
                    forwardFunc_( input_, output_ );
                    }, iterations );

                // Calculate throughput metrics
                result.throughput_elements = static_cast<double>(input_.size()) / (result.time_ms / 1000.0);

                // For MLP, we can approximately estimate FLOPS based on operations within
                // For example: 2 FC layers + GELU activation
                if ( isMLP_ ) {
                    // Assuming size of FC layers and operations within
                    size_t input_features = inputShape_.back();
                    size_t hidden_features = input_features * 4; // Common ratio in transformers

                    // FLOPs for first FC: 2*M*N (M=input size, N=hidden size)
                    // FLOPs for GELU: ~15*N
                    // FLOPs for second FC: 2*N*M
                    double flops_per_forward = 2.0 * input_.size() / input_features * input_features * hidden_features +
                        15.0 * input_.size() / input_features * hidden_features +
                        2.0 * input_.size() / input_features * hidden_features * input_features;

                    result.throughput_gflops = flops_per_forward / (result.time_ms / 1000.0) / 1e9;
                }

                return result;
            }

            std::string name() const override {
                std::ostringstream oss;
                oss << moduleName_ << " [";
                for ( size_t i = 0; i < inputShape_.size(); ++i ) {
                    oss << inputShape_[ i ];
                    if ( i < inputShape_.size() - 1 ) {
                        oss << "x";
                    }
                }
                oss << "]";
                return oss.str();
            }

        private:
            using InputTensor = std::conditional_t<TDeviceType == DeviceType::Cuda,
                Tensor<TPrecision, CudaMemoryResource>,
                Tensor<TPrecision, HostMemoryResource>>;

            // Type-safe function to call forward
            std::function<void( const InputTensor&, InputTensor& )> forwardFunc_;

            std::string moduleName_; // Store the module name as a string
            bool isMLP_; // Flag to check if this is an MLP module
            std::vector<size_t> inputShape_;
            InputTensor input_;
            InputTensor output_;
    };
}
