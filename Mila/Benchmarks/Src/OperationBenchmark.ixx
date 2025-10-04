// File: Mila/Benchmarks/OperationBenchmarks.ixx
module;
#include <vector>
#include <memory>
#include <string>
#include <stdexcept>
#include <type_traits>
#include <sstream>

export module Mila.Benchmark.OperationBenchmark;

import Mila;
import Mila.Benchmark;

namespace Mila::Benchmark
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    /**
     * @brief Benchmark implementation for Operation layers.
     *
     * This benchmark class measures the performance of operation execution
     * with configurable precision settings.
     *
     * @tparam TDeviceType The device type (CPU or CUDA) on which to perform computations.
     * @tparam TDataType The data type used for tensor elements throughout the operation.
     */
    export template<DeviceType TDeviceType = DeviceType::Cuda, typename TDataType = float>
        class OperationBenchmark : public Benchmark {
        public:
            using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaDeviceMemoryResource, HostMemoryResource>;

            /**
             * @brief Constructs a new OperationBenchmark.
             *
             * @param operation The operation to benchmark.
             * @param opName The name of the operation.
             * @param inputShape The shape of the input tensor.
             * @param context The device context to use.
             * @param precision The compute precision policy to use (defaults to Auto).
             */
            OperationBenchmark(
                std::shared_ptr<OperationBase<TDeviceType, TDataType, TDataType, TDataType>> operation,
                std::string opName,
                std::vector<size_t> inputShape,
                std::shared_ptr<DeviceContext> context,
                ComputePrecision::Policy precision = ComputePrecision::Policy::Auto )
                : operation_( operation ), opName_( opName ), inputShape_( inputShape ) {

                this->deviceContext_ = context;

                // Update the operation's precision policy
                operation_->setPrecisionPolicy( precision );

                // Store the precision policy in properties
                properties_.set( "precision_policy", static_cast<int>(precision) );

                // Create input and output tensors based on device type
                if constexpr ( TDeviceType == DeviceType::Cuda ) {
                    input_ = Tensor<TDataType, CudaDeviceMemoryResource>( inputShape_ );
                    output_ = Tensor<TDataType, CudaDeviceMemoryResource>( inputShape_ );

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
                    output_ = Tensor<TDataType, HostMemoryResource>( inputShape_ );

                    // Initialize with random values
                    for ( size_t i = 0; i < input_.size(); ++i ) {
                        input_.data()[ i ] = static_cast<TDataType>( rand() ) / RAND_MAX * 2.0f - 1.0f;
                    }
                }

                // Create second input for binary operations (same shape as first input)
                if constexpr ( TDeviceType == DeviceType::Cuda ) {
                    input2_ = Tensor<TDataType, CudaDeviceMemoryResource>( inputShape_ );

                    // Create host tensor for initialization
                    Tensor<TDataType, HostMemoryResource> hostInput2( inputShape_ );

                    // Initialize with random values
                    for ( size_t i = 0; i < hostInput2.size(); ++i ) {
                        hostInput2.data()[ i ] = static_cast<TDataType>( rand() ) / RAND_MAX * 2.0f - 1.0f;
                    }

                    // Copy to device
                    input2_.copyFrom( hostInput2 );
                }
                else {
                    input2_ = Tensor<TDataType, HostMemoryResource>( inputShape_ );

                    // Initialize with random values
                    for ( size_t i = 0; i < input2_.size(); ++i ) {
                        input2_.data()[ i ] = static_cast<TDataType>( rand() ) / RAND_MAX * 2.0f - 1.0f;
                    }
                }

                // Initialize parameters and state vectors
                parameters_ = std::vector<std::shared_ptr<Tensor<TDataType, MR>>>();
                output_state_ = std::vector<std::shared_ptr<Tensor<TDataType, MR>>>();
            }

            /**
             * @brief Runs the benchmark for the specified number of iterations.
             *
             * @param iterations The number of times to run the operation.
             * @return BenchmarkResult The results of the benchmark.
             */
            BenchmarkResult run( size_t iterations ) override {
                BenchmarkResult result;
                result.name = name();
                result.iterations = iterations;
                result.elementCount = input_.size();
                result.deviceName = deviceToString( deviceContext_->getDevice()->getDeviceType() );

                // Include precision policy directly from the operation
                ComputePrecision::Policy precisionPolicy = operation_->getPrecisionPolicy();
                result.properties[ "precision_policy" ] = static_cast<int>(precisionPolicy);
                result.properties[ "mixed_precision_enabled" ] = operation_->isMixedPrecisionEnabled();

                // Determine if operation is UnaryOperation or BinaryOperation
                auto unaryOp = std::dynamic_pointer_cast<UnaryOperation<TDeviceType, TDataType, TDataType>>(operation_);
                auto binaryOp = std::dynamic_pointer_cast<BinaryOperation<TDeviceType, TDataType, TDataType>>(operation_);

                // Measure time
                result.time_ms = measureExecutionTime( [this, unaryOp, binaryOp]() {
                    if ( unaryOp ) {
                        // Call UnaryOperation::forward
                        unaryOp->forward( input_, parameters_, properties_, output_, output_state_ );
                    }
                    else if ( binaryOp ) {
                        // Call BinaryOperation::forward
                        binaryOp->forward( input_, input2_, parameters_, properties_, output_, output_state_ );
                    }
                    else {
                        throw std::runtime_error( "Operation is neither UnaryOperation nor BinaryOperation" );
                    }
                    }, iterations );

                // Calculate throughput metrics
                result.throughput_elements = static_cast<double>(input_.size()) / (result.time_ms / 1000.0);

                // Estimate FLOPs for specific operations
                double flops_per_element = 0;
                if ( opName_.find( "Gelu" ) != std::string::npos ) {
                    flops_per_element = 15;  // Approximate FLOPs for GELU
                }
                else if ( opName_.find( "Softmax" ) != std::string::npos ) {
                    flops_per_element = 5;   // Approximate FLOPs for Softmax
                }

                if ( flops_per_element > 0 ) {
                    result.throughput_gflops = static_cast<double>(input_.size() * flops_per_element) /
                        (result.time_ms / 1000.0) / 1e9;
                }

                // Add precision mode to the benchmark result
                std::string precisionStr;
                switch ( precisionPolicy ) {
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

                result.notes = "Precision: " + precisionStr +
                    (operation_->isMixedPrecisionEnabled() ? " (Mixed Precision Enabled)" : " (Mixed Precision Disabled)");

                return result;
            }

            /**
             * @brief Gets the formatted name of the benchmark.
             *
             * @return std::string The name of the benchmark.
             */
            std::string name() const override {
                std::ostringstream oss;
                oss << opName_ << " [";
                for ( size_t i = 0; i < inputShape_.size(); ++i ) {
                    oss << inputShape_[ i ];
                    if ( i < inputShape_.size() - 1 ) {
                        oss << "x";
                    }
                }
                oss << "]";

                // Add precision info directly from the operation
                ComputePrecision::Policy precisionPolicy = operation_->getPrecisionPolicy();
                switch ( precisionPolicy ) {
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

            std::shared_ptr<OperationBase<TDeviceType, TDataType, TDataType, TDataType>> operation_;
            std::string opName_;
            std::vector<size_t> inputShape_;
            InputTensor input_;
            InputTensor input2_;  // Second input tensor for binary operations
            InputTensor output_;
            std::vector<std::shared_ptr<Tensor<TDataType, MR>>> parameters_;
            std::vector<std::shared_ptr<Tensor<TDataType, MR>>> output_state_;
    };
}