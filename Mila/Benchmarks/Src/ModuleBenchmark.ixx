// Fix for Mila/Benchmarks/Src/ModuleBenchmark.ixx

module;
#include <vector>
#include <memory>
#include <string>
#include <functional>

export module Mila.Benchmark.ModuleBenchmark;

import Mila;
import Mila.Benchmark;

namespace Mila::Benchmark
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    export template<typename TPrecision, typename TInput = TPrecision, DeviceType TDeviceType = DeviceType::Cuda>
        class ModuleBenchmark : public Benchmark {
        public:
            template <typename TModule>
            ModuleBenchmark( std::shared_ptr<TModule> module,
                std::vector<size_t> inputShape,
                std::vector<size_t> outputShape,
                std::shared_ptr<DeviceContext> context )
                : inputShape_( inputShape ), outputShape_( outputShape ), moduleName_( module->getName() ) {

                this->deviceContext_ = context;

                // Store the forward function as a lambda that captures the module
                forwardFunc_ = [module]( const auto& input, auto& output ) {
                    module->forward( input, output );
                    };

                // Create input and output tensors
                if constexpr ( TDeviceType == DeviceType::Cuda ) {
                    input_ = Tensor<TInput, CudaMemoryResource>( inputShape_ );
                    output_ = Tensor<TPrecision, CudaMemoryResource>( outputShape_ );

                    // Create host tensor for initialization
                    Tensor<TInput, HostMemoryResource> hostInput( inputShape_ );

                    // Initialize with random values
                    for ( size_t i = 0; i < hostInput.size(); ++i ) {
                        hostInput.data()[ i ] = static_cast<TInput>( rand() ) / RAND_MAX * 2.0f - 1.0f;
                    }

                    // Copy to device
                    input_.copyFrom( hostInput );
                }
                else {
                    input_ = Tensor<TInput, HostMemoryResource>( inputShape_ );
                    output_ = Tensor<TPrecision, HostMemoryResource>( outputShape_ );

                    // Initialize with random values
                    for ( size_t i = 0; i < input_.size(); ++i ) {
                        input_.data()[ i ] = static_cast<TInput>( rand() ) / RAND_MAX * 2.0f - 1.0f;
                    }
                }
            }

            // Constructor overload for when input and output shapes are the same
            template <typename TModule>
            ModuleBenchmark( std::shared_ptr<TModule> module,
                std::vector<size_t> inputShape,
                std::shared_ptr<DeviceContext> context )
                : ModuleBenchmark( module, inputShape, inputShape, context ) {}

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

                // Calculate FLOPS based on the module type
                calculateFlops( result );

                return result;
            }

            std::string name() const override {
                std::ostringstream oss;
                oss << moduleName_ << " [";
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
            using InputTensor = std::conditional_t<TDeviceType == DeviceType::Cuda,
                Tensor<TInput, CudaMemoryResource>,
                Tensor<TInput, HostMemoryResource>>;

            using OutputTensor = std::conditional_t<TDeviceType == DeviceType::Cuda,
                Tensor<TPrecision, CudaMemoryResource>,
                Tensor<TPrecision, HostMemoryResource>>;

            // Type-safe function to call forward
            std::function<void( const InputTensor&, OutputTensor& )> forwardFunc_;

            std::string moduleName_; // Store the module name as a string
            std::vector<size_t> inputShape_;
            std::vector<size_t> outputShape_;
            InputTensor input_;
            OutputTensor output_;

            // Calculate FLOPs based on the module type and shape
            void calculateFlops( BenchmarkResult& result ) {
                // Use the stored module name instead of calling getName() on a type-erased pointer
                const std::string& moduleName = moduleName_;

                // GELU activation - approximately 15 FLOPs per element
                if ( moduleName.find( "Gelu" ) != std::string::npos ) {
                    constexpr int flops_per_element = 15;
                    result.throughput_gflops = static_cast<double>(input_.size() * flops_per_element) /
                        (result.time_ms / 1000.0) / 1e9;
                }
                // Linear/FC layers - 2*M*N FLOPs (M=input_features, N=output_features)
                else if ( moduleName.find( "Linear" ) != std::string::npos ||
                    moduleName.find( "FC" ) != std::string::npos ) {
                    size_t input_features = inputShape_.back();
                    size_t output_features = outputShape_.back();
                    size_t batch_elements = input_.size() / input_features;
                    double flops = 2.0 * batch_elements * input_features * output_features;
                    result.throughput_gflops = flops / (result.time_ms / 1000.0) / 1e9;
                }
                // Attention - complex calculation based on sequence length and embedding dim
                else if ( moduleName.find( "Attention" ) != std::string::npos ) {
                    // Approximate FLOP calculation for attention mechanism
                    if ( inputShape_.size() >= 3 ) {
                        size_t batch_size = inputShape_[ 0 ];
                        size_t seq_len = inputShape_[ 1 ];
                        size_t embed_dim = inputShape_[ 2 ];

                        // Q,K,V projections + attention computation + output projection
                        double flops = 4.0 * batch_size * seq_len * embed_dim * embed_dim +
                            2.0 * batch_size * seq_len * seq_len * embed_dim;
                        result.throughput_gflops = flops / (result.time_ms / 1000.0) / 1e9;
                    }
                }
                // Default conservative estimate for other modules
                else {
                    constexpr int default_flops_per_element = 10;
                    result.throughput_gflops = static_cast<double>(input_.size() * default_flops_per_element) /
                        (result.time_ms / 1000.0) / 1e9;
                }
            }
    };
}
