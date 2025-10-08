// File: Mila/Benchmarks/Src/BlockModuleBenchmark.ixx

module;
#include <vector>
#include <memory>
#include <string>
#include <functional>
#include <sstream>

export module Mila.Benchmark.BlockModuleBenchmark;

import Mila;
import Mila.Benchmark;

namespace Mila::Benchmark
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    /**
     * @brief Benchmark implementation for Block Module types like MLPs.
     *
     * This benchmark class measures the performance of complex neural network blocks
     * with configurable precision settings.
     *
     * @tparam TDeviceType The device type (CPU or CUDA) on which to perform computations.
     * @tparam TDataType The data type used for tensor elements throughout the block.
     */
    export template<DeviceType TDeviceType = DeviceType::Cuda, typename TDataType = float>
        class BlockModuleBenchmark : public Benchmark {
        public:
            using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaDeviceMemoryResource, HostMemoryResource>;

            /**
             * @brief Constructs a new BlockModuleBenchmark.
             *
             * @param blockModule The block module to benchmark.
             * @param inputShape The shape of the input tensor.
             * @param context The device context to use.
             */
            template <typename TBlockModule>
            BlockModuleBenchmark( std::shared_ptr<TBlockModule> blockModule,
                std::vector<size_t> inputShape,
                std::shared_ptr<DeviceContext> context )
                : inputShape_( inputShape ), moduleName_( blockModule->getDeviceName() ) {

                this->deviceContext_ = context;

                // Store the precision policy if available from the module
                if constexpr ( requires { blockModule->getComputePrecision().getPolicy(); } ) {
                    precisionPolicy_ = blockModule->getComputePrecision().getPolicy();
                }

                // Store the forward function as a lambda that captures the module
                forwardFunc_ = [blockModule]( const auto& input, auto& output ) {
                    blockModule->forward( input, output );
                    };

                // Store module name for specific block type identification
                isMLP_ = (moduleName_.find( "MLP" ) != std::string::npos);
                isTransformer_ = (moduleName_.find( "Transformer" ) != std::string::npos);

                // Create input and output tensors
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
                result.elementCount = input_.size();
                result.deviceName = deviceToString( deviceContext_->getDevice()->getDeviceType() );

                // Include precision policy in the result
                result.properties[ "precision_policy" ] = static_cast<int>(precisionPolicy_);

                // Measure time
                result.time_ms = measureExecutionTime( [this]() {
                    forwardFunc_( input_, output_ );
                    }, iterations );

                // Calculate throughput metrics
                result.throughput_elements = static_cast<double>(input_.size()) / (result.time_ms / 1000.0);

                // Block-specific FLOP calculations
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
            std::function<void( const InputTensor&, InputTensor& )> forwardFunc_;

            std::string moduleName_; // Store the module name as a string
            bool isMLP_ = false;     // Flag to check if this is an MLP module
            bool isTransformer_ = false; // Flag to check if this is a Transformer module
            std::vector<size_t> inputShape_;
            InputTensor input_;
            InputTensor output_;
            ComputePrecision::Policy precisionPolicy_ = ComputePrecision::Policy::Auto; // Default precision policy

            /**
             * @brief Calculate FLOPs based on the module type and shape.
             *
             * @param result The benchmark result to update with FLOP calculations.
             */
            void calculateFlops( BenchmarkResult& result ) {
                // For MLP, we can approximately estimate FLOPS based on operations within
                if ( isMLP_ ) {
                    size_t input_features = inputShape_.back();
                    size_t hidden_features = input_features * 4; // Common ratio in transformers
                    size_t batch_elements = input_.size() / input_features;

                    // FLOPs for first FC: 2*batch*input*hidden
                    // FLOPs for GELU: ~15*batch*hidden
                    // FLOPs for second FC: 2*batch*hidden*input
                    double flops_per_forward =
                        2.0 * batch_elements * input_features * hidden_features +
                        15.0 * batch_elements * hidden_features +
                        2.0 * batch_elements * hidden_features * input_features;

                    result.throughput_gflops = flops_per_forward / (result.time_ms / 1000.0) / 1e9;
                }
                // For Transformer, provide separate FLOP calculation
                else if ( isTransformer_ ) {
                    if ( inputShape_.size() >= 3 ) {
                        size_t batch_size = inputShape_[ 0 ];
                        size_t seq_len = inputShape_[ 1 ];
                        size_t embed_dim = inputShape_[ 2 ];

                        // Estimate FLOPS for:
                        // 1. Self-attention (QKV projections, attention matrix, softmax, output projection)
                        // 2. MLP (two FC layers + activation)
                        // 3. Layer norms
                        double attn_flops =
                            4.0 * batch_size * seq_len * embed_dim * embed_dim + // QKV + output projections
                            2.0 * batch_size * seq_len * seq_len * embed_dim;    // Attention computation

                        double mlp_flops =
                            2.0 * batch_size * seq_len * embed_dim * (4 * embed_dim) + // First FC
                            15.0 * batch_size * seq_len * (4 * embed_dim) +            // GELU
                            2.0 * batch_size * seq_len * (4 * embed_dim) * embed_dim;  // Second FC

                        double layernorm_flops =
                            10.0 * batch_size * seq_len * embed_dim * 2; // Two layer norms

                        double total_flops = attn_flops + mlp_flops + layernorm_flops;

                        result.throughput_gflops = total_flops / (result.time_ms / 1000.0) / 1e9;
                    }
                }
                // Default case for other block types
                else {
                    constexpr int default_flops_per_element = 50; // Conservative estimate
                    result.throughput_gflops = static_cast<double>(input_.size() * default_flops_per_element) /
                        (result.time_ms / 1000.0) / 1e9;
                }
            }
    };
}