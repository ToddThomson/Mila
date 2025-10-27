// File: Mila/Benchmarks/Src/BinaryModuleBenchmark.ixx

module;
#include <vector>
#include <memory>
#include <string>
#include <functional>
#include <sstream>

export module Mila.Benchmark.BinaryModuleBenchmark;

import Mila;
import Mila.Benchmark;

namespace Mila::Benchmark
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    /**
     * @brief Benchmark implementation for binary modules that take two inputs.
     *
     * This benchmark class measures the performance of modules that require two input tensors,
     * such as Residual connections and other binary operations.
     *
     * @tparam TDeviceType The device type (CPU or CUDA) on which to perform computations.
     * @tparam TDataType The data type used for tensor elements throughout the module.
     */
    export template<DeviceType TDeviceType = DeviceType::Cuda, typename TDataType = float>
        class BinaryModuleBenchmark : public Benchmark {
        public:
            using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaDeviceMemoryResource, HostMemoryResource>;

            /**
             * @brief Constructs a new BinaryModuleBenchmark.
             *
             * @param module The binary module to benchmark.
             * @param inputShape The shape of both input tensors.
             * @param context The device context to use.
             */
            template <typename TModule>
            BinaryModuleBenchmark( std::shared_ptr<TModule> module,
                std::vector<size_t> inputShape,
                std::shared_ptr<DeviceContext> context )
                : inputShape_( inputShape ), moduleName_( module->getDeviceName() ) {

                this->deviceContext_ = context;

                // Store the precision policy if available from the module
                if constexpr ( requires { module->getComputePrecision().getPolicy(); } ) {
                    precisionPolicy_ = module->getComputePrecision().getPolicy();
                }

                // Store the forward function as a lambda that captures the module
                forwardFunc_ = [module]( const auto& input1, const auto& input2, auto& output ) {
                    module->forward( input1, input2, output );
                    };

                // Check if this module is a residual connection or other specific type
                isResidual_ = (moduleName_.find( "Residual" ) != std::string::npos);

                // Create input and output tensors
                if constexpr ( TDeviceType == DeviceType::Cuda ) {
                    input1_ = Tensor<TDataType, CudaDeviceMemoryResource>( inputShape_ );
                    input2_ = Tensor<TDataType, CudaDeviceMemoryResource>( inputShape_ );
                    output_ = Tensor<TDataType, CudaDeviceMemoryResource>( inputShape_ );

                    // Create host tensor for initialization
                    Tensor<TDataType, HostMemoryResource> hostInput1( inputShape_ );
                    Tensor<TDataType, HostMemoryResource> hostInput2( inputShape_ );

                    // Initialize with random values
                    for ( size_t i = 0; i < hostInput1.size(); ++i ) {
                        hostInput1.data()[ i ] = static_cast<TDataType>( rand() ) / RAND_MAX * 2.0f - 1.0f;
                        hostInput2.data()[ i ] = static_cast<TDataType>( rand() ) / RAND_MAX * 2.0f - 1.0f;
                    }

                    // Copy to device
                    input1_.copyFrom( hostInput1 );
                    input2_.copyFrom( hostInput2 );
                }
                else {
                    input1_ = Tensor<TDataType, HostMemoryResource>( inputShape_ );
                    input2_ = Tensor<TDataType, HostMemoryResource>( inputShape_ );
                    output_ = Tensor<TDataType, HostMemoryResource>( inputShape_ );

                    // Initialize with random values
                    for ( size_t i = 0; i < input1_.size(); ++i ) {
                        input1_.data()[ i ] = static_cast<TDataType>( rand() ) / RAND_MAX * 2.0f - 1.0f;
                        input2_.data()[ i ] = static_cast<TDataType>( rand() ) / RAND_MAX * 2.0f - 1.0f;
                    }
                }
            }

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
                result.elementCount = input1_.size();
                result.deviceName = deviceTypeToString( deviceContext_->getDevice()->getDeviceType() );

                // Include precision policy in the result
                result.properties[ "precision_policy" ] = static_cast<int>(precisionPolicy_);

                // Measure time
                result.time_ms = measureExecutionTime( [this]() {
                    forwardFunc_( input1_, input2_, output_ );
                    }, iterations );

                // Calculate throughput metrics
                result.throughput_elements = static_cast<double>(input1_.size()) / (result.time_ms / 1000.0);

                // Calculate FLOPS based on operation type
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
                    case ComputePrecision::Policy::Native:
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
                    case ComputePrecision::Policy::Native:
                        oss << " (Dis)";
                        break;
                    case ComputePrecision::Policy::Auto:
                        oss << " (Auto)";
                        break;
                }

                return oss.str();
            }

        private:
            using InputTensor = std::conditional_t<TDeviceType == DeviceType::Cuda,
                Tensor<TDataType, CudaDeviceMemoryResource>,
                Tensor<TDataType, HostMemoryResource>>;

            // Type-safe function to call forward
            std::function<void( const InputTensor&, const InputTensor&, InputTensor& )> forwardFunc_;

            std::string moduleName_;          // Store the module name as a string
            bool isResidual_ = false;         // Flag to check if this is a residual connection
            std::vector<size_t> inputShape_;
            InputTensor input1_;
            InputTensor input2_;
            InputTensor output_;
            ComputePrecision::Policy precisionPolicy_ = ComputePrecision::Policy::Auto; // Default precision policy

            /**
             * @brief Calculate FLOPs based on the module type and shape.
             *
             * @param result The benchmark result to update with FLOP calculations.
             */
            void calculateFlops( BenchmarkResult& result ) {
                // For residual connections, count 1 FLOP per element (addition operation)
                if ( isResidual_ ) {
                    result.throughput_gflops = static_cast<double>(input1_.size()) / (result.time_ms / 1000.0) / 1e9;
                }
                // For other binary operations, use a conservative estimate
                else {
                    constexpr int flops_per_element = 2; // For typical binary ops like add, multiply, etc.
                    result.throughput_gflops = static_cast<double>(input1_.size() * flops_per_element) /
                        (result.time_ms / 1000.0) / 1e9;
                }
            }
    };
}