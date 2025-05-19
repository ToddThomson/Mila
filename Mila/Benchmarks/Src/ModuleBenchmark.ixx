// File: Mila/Benchmarks/Src/ModuleBenchmark.ixx

module;
#include <vector>
#include <memory>
#include <string>
#include <functional>
#include <sstream>

export module Mila.Benchmark.ModuleBenchmark;

import Mila;
import Mila.Benchmark;

namespace Mila::Benchmark
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    /**
     * @brief Benchmark implementation for Neural Network Modules.
     *
     * This benchmark class measures the performance of module execution
     * with configurable precision settings.
     *
     * @tparam TDeviceType The device type (CPU or CUDA) on which to perform computations.
     * @tparam TDataType The data type used for tensor elements throughout the module.
     */
    export template<DeviceType TDeviceType = DeviceType::Cuda, typename TDataType = float>
        class ModuleBenchmark : public Benchmark {
        public:
            using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaMemoryResource, HostMemoryResource>;

            /**
             * @brief Constructs a new ModuleBenchmark with different input and output shapes.
             *
             * @param module The module to benchmark.
             * @param inputShape The shape of the input tensor.
             * @param outputShape The shape of the output tensor.
             * @param context The device context to use.
             */
            template <typename TModule>
            ModuleBenchmark( std::shared_ptr<TModule> module,
                std::vector<size_t> inputShape,
                std::vector<size_t> outputShape,
                std::shared_ptr<DeviceContext> context )
                : inputShape_( inputShape ), outputShape_( outputShape ), moduleName_( module->getName() ) {

                this->deviceContext_ = context;

                // Store the precision policy if available from the module
                if constexpr ( requires { module->getComputePrecision().getPolicy(); } ) {
                    precisionPolicy_ = module->getComputePrecision().getPolicy();
                }

                // Store the forward function as a lambda that captures the module
                forwardFunc_ = [module]( const auto& input, auto& output ) {
                    module->forward( input, output );
                    };

                // Create input and output tensors
                if constexpr ( TDeviceType == DeviceType::Cuda ) {
                    input_ = Tensor<TDataType, CudaMemoryResource>( inputShape_ );
                    output_ = Tensor<TDataType, CudaMemoryResource>( outputShape_ );

                    // Create host tensor for initialization
                    Tensor<TDataType, HostMemoryResource> hostInput( inputShape_ );

                    // Initialize with random values
                    for ( size_t i = 0; i < hostInput.size(); ++i ) {
                        hostInput.data()[ i ] = static_cast<TDataType>( rand() ) / RAND_MAX * 2.0f - 1.0f;
                    }

                    // Copy to device
                    input_.copyFrom( hostInput );
                }
                else {
                    input_ = Tensor<TDataType, HostMemoryResource>( inputShape_ );
                    output_ = Tensor<TDataType, HostMemoryResource>( outputShape_ );

                    // Initialize with random values
                    for ( size_t i = 0; i < input_.size(); ++i ) {
                        input_.data()[ i ] = static_cast<TDataType>( rand() ) / RAND_MAX * 2.0f - 1.0f;
                    }
                }
            }

            /**
             * @brief Constructor overload for when input and output shapes are the same.
             *
             * @param module The module to benchmark.
             * @param inputShape The shape of both input and output tensors.
             * @param context The device context to use.
             */
            template <typename TModule>
            ModuleBenchmark( std::shared_ptr<TModule> module,
                std::vector<size_t> inputShape,
                std::shared_ptr<DeviceContext> context )
                : ModuleBenchmark( module, inputShape, inputShape, context ) {}

            /**
             * @brief Runs the benchmark for the specified number of iterations.
             *
             * @param iterations The number of times to run the module.
             * @return BenchmarkResult The results of the benchmark.
             */
            BenchmarkResult run( size_t iterations ) override {
                BenchmarkResult result;
                result.name = name();
                result.iterations = iterations;
                result.elementCount = input_.size();
                result.deviceName = deviceToString( deviceContext_->getDevice()->getDeviceType() );

                // Include precision policy information
                result.properties[ "precision_policy" ] = static_cast<int>(precisionPolicy_);

                // Measure time
                result.time_ms = measureExecutionTime( [this]() {
                    forwardFunc_( input_, output_ );
                    }, iterations );

                // Calculate throughput metrics
                result.throughput_elements = static_cast<double>(input_.size()) / (result.time_ms / 1000.0);

                // Calculate FLOPS based on the module type
                calculateFlops( result );

                // Add precision mode to the benchmark result
                std::string precisionStr;
                switch ( precisionPolicy_ ) {
                    case ComputePrecision::Policy::Auto:
                        precisionStr = "Auto";
                        break;
                    case ComputePrecision::Policy::Performance:
                        precisionStr = "Performance";
                        break;
                    case ComputePrecision::Policy::Accuracy:
                        precisionStr = "Accuracy";
                        break;
                    case ComputePrecision::Policy::Disabled:
                        precisionStr = "Disabled";
                        break;
                    default:
                        precisionStr = "Unknown";
                        break;
                }

                result.notes = "Precision: " + precisionStr;

                return result;
            }

            /**
             * @brief Gets the formatted name of the benchmark.
             *
             * @return std::string The name of the benchmark.
             */
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

                // Add precision info to the name
                switch ( precisionPolicy_ ) {
                    case ComputePrecision::Policy::Performance:
                        oss << " (Perf)";
                        break;
                    case ComputePrecision::Policy::Accuracy:
                        oss << " (Accu)";
                        break;
                    case ComputePrecision::Policy::Disabled:
                        oss << " (Dis)";
                        break;
                    case ComputePrecision::Policy::Auto:
                        oss << " (Auto)";
                        break;
                }

                return oss.str();
            }

        private:
            using InputOutputTensor = std::conditional_t<TDeviceType == DeviceType::Cuda,
                Tensor<TDataType, CudaMemoryResource>,
                Tensor<TDataType, HostMemoryResource>>;

            // Type-safe function to call forward
            std::function<void( const InputOutputTensor&, InputOutputTensor& )> forwardFunc_;

            std::string moduleName_; // Store the module name as a string
            std::vector<size_t> inputShape_;
            std::vector<size_t> outputShape_;
            InputOutputTensor input_;
            InputOutputTensor output_;
            ComputePrecision::Policy precisionPolicy_ = ComputePrecision::Policy::Auto; // Default precision policy

            /**
             * @brief Calculate FLOPs based on the module type and shape.
             *
             * @param result The benchmark result to update with FLOP calculations.
             */
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
                // MLP modules - compute based on its structure
                else if ( moduleName.find( "MLP" ) != std::string::npos ) {
                    // Estimate FLOPs for MLP as 2 linear layers + activation
                    if ( inputShape_.size() >= 2 ) {
                        size_t batch_elements = input_.size() / inputShape_.back();
                        size_t input_features = inputShape_.back();
                        size_t hidden_features = 4 * input_features; // Common expansion factor

                        // 2 Linear layers + GELU: 2*(2*batch*in*hidden + 2*batch*hidden*out) + 15*batch*hidden
                        double flops = 2.0 * batch_elements * input_features * hidden_features +
                            2.0 * batch_elements * hidden_features * input_features +
                            15.0 * batch_elements * hidden_features;
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