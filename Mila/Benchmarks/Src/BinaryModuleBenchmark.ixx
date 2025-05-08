// Fix for Mila/Benchmarks/Src/BinaryModuleBenchmark.ixx

module;
#include <vector>
#include <memory>
#include <string>
#include <functional>

export module Mila.Benchmark.BinaryModuleBenchmark;

import Mila;
import Mila.Benchmark;

namespace Mila::Benchmark
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    // For modules that require two inputs (like Residual)
    export template<typename TPrecision, DeviceType TDeviceType = DeviceType::Cuda>
        class BinaryModuleBenchmark : public Benchmark {
        public:
            template <typename TModule>
            BinaryModuleBenchmark( std::shared_ptr<TModule> module,
                std::vector<size_t> inputShape,
                std::shared_ptr<DeviceContext> context )
                : inputShape_( inputShape ), moduleName_( module->getName() ) {

                this->deviceContext_ = context;

                // Store the forward function as a lambda that captures the module
                forwardFunc_ = [module]( const auto& input1, const auto& input2, auto& output ) {
                    module->forward( input1, input2, output );
                    };

                // Create input and output tensors
                if constexpr ( TDeviceType == DeviceType::Cuda ) {
                    input1_ = Tensor<TPrecision, CudaMemoryResource>( inputShape_ );
                    input2_ = Tensor<TPrecision, CudaMemoryResource>( inputShape_ );
                    output_ = Tensor<TPrecision, CudaMemoryResource>( inputShape_ );

                    // Create host tensor for initialization
                    Tensor<TPrecision, HostMemoryResource> hostInput1( inputShape_ );
                    Tensor<TPrecision, HostMemoryResource> hostInput2( inputShape_ );

                    // Initialize with random values
                    for ( size_t i = 0; i < hostInput1.size(); ++i ) {
                        hostInput1.data()[ i ] = static_cast<TPrecision>( rand() ) / RAND_MAX * 2.0f - 1.0f;
                        hostInput2.data()[ i ] = static_cast<TPrecision>( rand() ) / RAND_MAX * 2.0f - 1.0f;
                    }

                    // Copy to device
                    input1_.copyFrom( hostInput1 );
                    input2_.copyFrom( hostInput2 );
                }
                else {
                    input1_ = Tensor<TPrecision, HostMemoryResource>( inputShape_ );
                    input2_ = Tensor<TPrecision, HostMemoryResource>( inputShape_ );
                    output_ = Tensor<TPrecision, HostMemoryResource>( inputShape_ );

                    // Initialize with random values
                    for ( size_t i = 0; i < input1_.size(); ++i ) {
                        input1_.data()[ i ] = static_cast<TPrecision>( rand() ) / RAND_MAX * 2.0f - 1.0f;
                        input2_.data()[ i ] = static_cast<TPrecision>( rand() ) / RAND_MAX * 2.0f - 1.0f;
                    }
                }
            }

            BenchmarkResult run( size_t iterations ) override {
                BenchmarkResult result;
                result.name = name();
                result.iterations = iterations;
                result.elementCount = input1_.size();
                result.deviceName = deviceToString( deviceContext_->getDevice()->getDeviceType() );

                // Measure time
                result.time_ms = measureExecutionTime( [this]() {
                    forwardFunc_( input1_, input2_, output_ );
                    }, iterations );

                // Calculate throughput metrics
                result.throughput_elements = static_cast<double>(input1_.size()) / (result.time_ms / 1000.0);

                // For binary operations like residual connection, we count the operation as 1 FLOP per element
                result.throughput_gflops = static_cast<double>(input1_.size()) / (result.time_ms / 1000.0) / 1e9;

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
            std::function<void( const InputTensor&, const InputTensor&, InputTensor& )> forwardFunc_;

            std::string moduleName_; // Store the module name as a string
            std::vector<size_t> inputShape_;
            InputTensor input1_;
            InputTensor input2_;
            InputTensor output_;
    };
}
