/**
 * @file CpuSoftmaxOp.ixx
 * @brief Implementation of the CPU-based softmax operation for neural networks.
 */

module;
#include <string>  
#include <memory>  
#include <vector>  
#include <cmath>  
#ifdef USE_OMP
#include <omp.h>
#endif  

export module Compute.CpuSoftmaxOp;

import Dnn.Tensor;  
import Compute.DeviceType;  
import Compute.OperationBase;  
import Compute.UnaryOperation;
import Compute.OperationRegistry;  
import Compute.OperationType;  
import Compute.MemoryResource;  
import Compute.CpuDevice;  

namespace Mila::Dnn::Compute
{
    /**
     * @brief CPU implementation of the softmax operation for neural networks.
     *
     * This class provides a CPU-based implementation of the softmax operation,
     * which converts a vector of real numbers into a probability distribution.
     * The softmax function is commonly used in classification tasks as the
     * final activation function of a neural network.
     *
     * @tparam TInput The data type of the input tensor elements.
     * @tparam TCompute The data type used for computation and output (defaults to the input type).
     */
    export
        template <typename TInput, typename TCompute = TInput>
    class CpuSoftmaxOp final : public UnaryOperation<float, float, DeviceType::Cpu> {
    public:
        /**
         * @brief Constructs a new CPU Softmax operation.
         *
         * Initializes the operation with the CPU device type and SoftmaxOp operation type.
         */
        CpuSoftmaxOp() : UnaryOperation<float, float, DeviceType::Cpu>( DeviceType::Cpu, OperationType::SoftmaxOp ) {}

        /**
         * @brief Performs the forward pass of the softmax operation.
         *
         * Converts input logits into a probability distribution by taking the
         * exponential of each element and normalizing by their sum.
         *
         * @param input Input tensor containing logits.
         * @param parameters Additional input parameters (not used in this operation).
         * @param properties Additional attributes for the operation.
         * @param output Output tensor to store the resulting probability distribution.
         * @param output_cache Cache for storing intermediate results (used in backward pass).
         */
        void forward(
            const Tensor<float, HostMemoryResource>& input,
            const std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>>& parameters,
            const OperationAttributes& properties,
            Tensor<float, HostMemoryResource>& output,
            std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>>& output_cache ) const override {

            const float* logits = input.data();
            float* probs = output.data();

            int B = input.shape()[ 0 ];
            int T = input.shape()[ 1 ];
            int V = input.shape()[ 2 ];  // 50257;
            int Vp = input.shape()[ 2 ];

            for ( int b = 0; b < B; b++ ) {
                for ( int t = 0; t < T; t++ ) {
                    // probs <- softmax(logits)  
                    const float* logits_bt = logits + b * T * Vp + t * Vp;
                    float* probs_bt = probs + b * T * Vp + t * Vp;

                    // maxval is only calculated and subtracted for numerical stability
                    float maxval = -10000.0f; // TODO something better
                    for ( int i = 0; i < V; i++ ) {
                        if ( logits_bt[ i ] > maxval ) {
                            maxval = logits_bt[ i ];
                        }
                    }

                    float sum = 0.0f;
                    for ( int i = 0; i < V; i++ ) {
                        probs_bt[ i ] = expf( logits_bt[ i ] - maxval );
                        sum += probs_bt[ i ];
                    }

                    // note we only loop to V, leaving the padded dimensions  
                    for ( int i = 0; i < V; i++ ) {
                        probs_bt[ i ] /= sum;
                    }

                    // for extra super safety we may wish to include this too,  
                    // forcing the probabilities here to be zero, but it shouldn't matter  
                    for ( int i = V; i < Vp; i++ ) {
                        probs_bt[ i ] = 0.0f;
                    }
                }
            }
        }

        /**
         * @brief Gets the name of this operation.
         *
         * @return std::string The name of the operation ("Cpu::SoftmaxOp").
         */
        std::string getName() const override {
            return "Cpu::SoftmaxOp";
        }

        /**
         * @brief Gets the class name of this operation.
         *
         * @return const std::string& The class name of the operation.
         */
        static const std::string& className() {
            static std::string name = "Cpu::SoftmaxOp";
            return name;
        }
    };

    /**
     * @brief Class responsible for registering the CpuSoftmaxOp operation.
     *
     * The CpuSoftmaxOpRegistrar class registers the CpuSoftmaxOp operation with the OperationRegistry.
     * It associates the operation name "Cpu::SoftmaxOp" with a factory function that creates instances of CpuSoftmaxOp.
     */
    export class CpuSoftmaxOpRegistrar {
    public:
        /**
         * @brief Registers the CpuSoftmaxOp operation with the OperationRegistry.
         *
         * This function registers the CpuSoftmaxOp operation for the CPU device type
         * with the OperationRegistry. It associates the operation name "Cpu::SoftmaxOp"
         * with a factory function that creates instances of CpuSoftmaxOp.
         */
        static void registerOperations() {
            const std::string opName = "Cpu::SoftmaxOp";

            OperationRegistry::instance().registerOperation<float, float, DeviceType::Cpu>(
                opName,
                []() -> std::shared_ptr<OperationBase<float, float, DeviceType::Cpu>> {
                    return std::make_shared<CpuSoftmaxOp<float, float>>();
                }
            );
        }

        /**
         * @brief Self-registration mechanism that registers the operation during startup.
         *
         * This static member ensures the operation is registered when the program starts
         * without requiring explicit registration calls.
         */
        static inline bool isRegistered = []() {
            registerOperations();
            return true;
            }();
    };
}
