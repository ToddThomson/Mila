// File: Mila/Benchmarks/OperationBenchmarks.ixx
module;
#include <vector>
#include <memory>
#include <string>
#include <stdexcept>
#include <type_traits>

export module Mila.Benchmark.OperationBenchmark;

import Mila;
import Mila.Benchmark;

namespace Mila::Benchmark
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    // Benchmark implementation for Operation layer
    export template<typename TPrecision, DeviceType TDeviceType = DeviceType::Cuda>
        class OperationBenchmark : public Benchmark {
        public:

            using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaMemoryResource, HostMemoryResource>;

            OperationBenchmark( std::shared_ptr<OperationBase<TPrecision, TPrecision, TPrecision, TDeviceType>> operation,
                std::string opName,
                std::vector<size_t> inputShape,
                std::shared_ptr<DeviceContext> context )
                : operation_( operation ), opName_( opName ), inputShape_( inputShape ) {

                this->deviceContext_ = context;

                // Create input and output tensors based on device type
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

                // Create second input for binary operations (same shape as first input)
                if constexpr ( TDeviceType == DeviceType::Cuda ) {
                    input2_ = Tensor<TPrecision, CudaMemoryResource>( inputShape_ );

                    // Create host tensor for initialization
                    Tensor<TPrecision, HostMemoryResource> hostInput2( inputShape_ );

                    // Initialize with random values
                    for ( size_t i = 0; i < hostInput2.size(); ++i ) {
                        hostInput2.data()[ i ] = static_cast<TPrecision>( rand() ) / RAND_MAX * 2.0f - 1.0f;
                    }

                    // Copy to device
                    input2_.copyFrom( hostInput2 );
                }
                else {
                    input2_ = Tensor<TPrecision, HostMemoryResource>( inputShape_ );

                    // Initialize with random values
                    for ( size_t i = 0; i < input2_.size(); ++i ) {
                        input2_.data()[ i ] = static_cast<TPrecision>( rand() ) / RAND_MAX * 2.0f - 1.0f;
                    }
                }

                // Initialize parameters and state vectors
                parameters_ = std::vector<std::shared_ptr<Tensor<TPrecision,
                    typename std::conditional_t<TDeviceType == DeviceType::Cuda,
                    CudaMemoryResource, HostMemoryResource>>>>();
                output_state_ = std::vector<std::shared_ptr<Tensor<TPrecision,
                    typename std::conditional_t<TDeviceType == DeviceType::Cuda,
                    CudaMemoryResource, HostMemoryResource>>>>();
            }

            BenchmarkResult run( size_t iterations ) override {
                BenchmarkResult result;
                result.name = name();
                result.iterations = iterations;
                result.elementCount = input_.size();
                result.deviceName = deviceToString( deviceContext_->getDevice()->getDeviceType() );

                // Determine if operation is UnaryOperation or BinaryOperation
                auto unaryOp = std::dynamic_pointer_cast<UnaryOperation<TPrecision, TPrecision, TDeviceType>>(operation_);
                auto binaryOp = std::dynamic_pointer_cast<BinaryOperation<TPrecision, TPrecision, TPrecision, TDeviceType>>(operation_);

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

                return result;
            }

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
                return oss.str();
            }

        private:
            using InputTensor = std::conditional_t<TDeviceType == DeviceType::Cuda,
                Tensor<TPrecision, CudaMemoryResource>,
                Tensor<TPrecision, HostMemoryResource>>;

            std::shared_ptr<OperationBase<TPrecision, TPrecision, TPrecision, TDeviceType>> operation_;
            std::string opName_;
            std::vector<size_t> inputShape_;
            InputTensor input_;
            InputTensor input2_;  // Second input tensor for binary operations
            InputTensor output_;
            std::vector<std::shared_ptr<Tensor<TPrecision, typename std::conditional_t<TDeviceType == DeviceType::Cuda,
                CudaMemoryResource, HostMemoryResource>>>> parameters_;
            std::vector<std::shared_ptr<Tensor<TPrecision, typename std::conditional_t<TDeviceType == DeviceType::Cuda,
                CudaMemoryResource, HostMemoryResource>>>> output_state_;
            OperationAttributes properties_;
    };
}
