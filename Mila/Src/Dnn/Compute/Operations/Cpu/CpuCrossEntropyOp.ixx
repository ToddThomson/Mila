/**
 * @file CpuCrossEntropyOp.ixx
 * @brief Implementation of the CPU-based cross entropy operation for neural networks.
 */

module;
#include <math.h>
#include <string>
#include <memory>
#include <vector>
#include <stdexcept>
#ifdef USE_OMP
#include <omp.h>
#endif
#include <cmath>

export module Compute.CpuCrossEntropyOp;

import Dnn.Tensor;
import Compute.Precision;
import Compute.OperationBase;
import Compute.UnaryOperation;
import Compute.OperationRegistry;
import Compute.DeviceType;
import Compute.DeviceContext;
import Compute.OperationType;
import Compute.OperationAttributes;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.CpuDevice;

using namespace Mila::Dnn;

namespace Mila::Dnn::Compute
{
    /**
     * @brief CPU implementation of the cross entropy loss operation for neural networks.
     *
     * This class provides a CPU-based implementation of the cross entropy loss function,
     * which is commonly used in classification tasks. It computes the negative log likelihood
     * of the correct class given the predicted probabilities.
     *
     * @tparam TInput The data type of the input tensor elements (typically int for class indices).
     * @tparam TDataType The data type used for computation and output (typically float).
     */
    export class CpuCrossEntropyOp : public UnaryOperation<DeviceType::Cpu, int, float> {
    public:
        using MR = typename CpuDevice::MR;
		using OperationBase = UnaryOperation<DeviceType::Cpu, int, float>;
        /**
         * @brief Constructs a new CPU Cross Entropy operation with the default device context.
         *
         * Initializes the operation with a CPU device context.
         */
        CpuCrossEntropyOp() 
            : OperationBase( OperationType::CrossEntropyOp, ComputePrecision::Policy::Disabled ) {
        }

        /**
         * @brief Constructs a new CPU Cross Entropy operation with a specific device context.
         *
         * @param context The device context to use for this operation.
         * @throws std::runtime_error If the context is not for a CPU device.
         */
        CpuCrossEntropyOp( std::shared_ptr<DeviceContext> context )
            : OperationBase( OperationType::CrossEntropyOp, context, ComputePrecision::Policy::Disabled ) {

        }

        /**
         * @brief Performs the forward pass of the cross entropy operation.
         *
         * Computes the negative log likelihood of the correct class for each sample.
         *
         * @param input Input tensor containing target class indices of shape [B, TDataType].
         * @param parameters Parameters tensor containing probabilities of shape [B, TDataType, V].
         * @param attributes Additional attributes for the operation.
         * @param output Output tensor to store the cross entropy losses of shape [B, TDataType].
         * @param output_state Cache for storing intermediate results (used in backward pass).
         */
        void forward(
            const Tensor<int, MR>& input,
            const std::vector<std::shared_ptr<Tensor<float, MR>>>& parameters,
            const OperationAttributes& attributes,
            Tensor<float, MR>& output,
            std::vector<std::shared_ptr<Tensor<float, MR>>>& output_state ) const override {

            auto B = input.shape()[ 0 ];
            auto T = input.shape()[ 1 ];
            auto Vp = parameters[ 0 ]->shape()[ 2 ];

            auto losses = output.raw_data();
            auto probs = parameters[ 0 ]->raw_data();
            auto targets = input.raw_data();

            const float epsilon = 1e-10f; // Small value to prevent log(0)

        #pragma omp parallel for collapse(2) if(B * T > 100)
            for ( int b = 0; b < B; b++ ) {
                for ( int t = 0; t < T; t++ ) {
                    // loss = -log(probs[target])
                    float* probs_bt = probs + b * T * Vp + t * Vp;
                    int ix = targets[ b * T + t ];
                    // Ensure index is valid and add epsilon to avoid log(0)
                    if ( ix >= 0 && ix < Vp ) {
                        losses[ b * T + t ] = -logf( probs_bt[ ix ] + epsilon );
                    }
                    else {
                        // Invalid index, set to zero or handle as needed
                        losses[ b * T + t ] = 0.0f;
                    }
                }
            }
        }

        /**
         * @brief Performs the backward pass of the cross entropy operation.
         *
         * Computes gradients with respect to inputs and probabilities.
         *
         * @param input Input tensor from the forward pass (target indices).
         * @param output Output tensor from the forward pass (loss values).
         * @param output_gradient Gradient of the loss with respect to the output.
         * @param parameters Parameters tensor from forward pass (probabilities).
         * @param parameter_gradients Gradients for parameters (probabilities).
         * @param input_gradient Gradient of the loss with respect to the input (unused for integer targets).
         * @param attributes Additional attributes for the operation.
         * @param output_state Cache tensors from forward pass.
         */
        void backward(
            const Tensor<int, MR>& input,
            const Tensor<float, MR>& output,
            const Tensor<float, MR>& output_gradient,
            const std::vector<std::shared_ptr<Tensor<float, MR>>>& parameters,
            std::vector<std::shared_ptr<Tensor<float, MR>>>& parameter_gradients,
            Tensor<int, MR>& input_gradient,
            const OperationAttributes& attributes,
            const std::vector<std::shared_ptr<Tensor<float, MR>>>& output_state ) const {

            // For combined softmax-crossentropy backward pass
            if ( parameter_gradients.size() > 0 ) {
                auto B = input.shape()[ 0 ];
                auto T = input.shape()[ 1 ];
                auto Vp = parameters[ 0 ]->shape()[ 2 ];
                auto V = attributes.get<int>( "vocab_size", Vp ); // Actual vocabulary size (without padding)

                float* dlogits = parameter_gradients[ 0 ]->raw_data();
                const float* dlosses = output_gradient.raw_data();
                const float* probs = parameters[ 0 ]->raw_data();

                backward_impl( dlogits, dlosses, probs, input, B, T, V, Vp );
            }
        }

        /**
         * @brief Helper method for the backward pass implementation.
         *
         * Computes gradients for the combined softmax and cross entropy operation.
         *
         * @param dlogits Gradient buffer for logits/probabilities.
         * @param dlosses Gradient buffer from output loss.
         * @param probs Original probability values.
         * @param targets Target class indices.
         * @param B Batch size.
         * @param TDataType Sequence length.
         * @param V Vocabulary size (without padding).
         * @param Vp Padded vocabulary size.
         */
        void backward_impl(
            float* dlogits,
            const float* dlosses,
            const float* probs,
            const Tensor<int, CpuMemoryResource>& targets,
            int B, int T, int V, int Vp ) const {

            // Backwards through both softmax and crossentropy
        #pragma omp parallel for collapse(2) if(B * T > 100)
            for ( int b = 0; b < B; b++ ) {
                for ( int t = 0; t < T; t++ ) {
                    float* dlogits_bt = dlogits + b * T * Vp + t * Vp;
                    const float* probs_bt = probs + b * T * Vp + t * Vp;
                    float dloss = dlosses[ b * T + t ];
                    int ix = targets.raw_data()[ b * T + t ];

                    // Only process valid indices
                    if ( ix >= 0 && ix < V ) {
                        // Note we only loop to V, leaving the padded dimensions
                        // of dlogits untouched, so gradient there stays at zero
                        for ( int i = 0; i < V; i++ ) {
                            float p = probs_bt[ i ];
                            float indicator = i == ix ? 1.0f : 0.0f;
                            dlogits_bt[ i ] += (p - indicator) * dloss;
                        }
                    }
                }
            }
        }

        /**
         * @brief Gets the name of this operation.
         *
         * @return std::string The name of the operation ("Cpu::CrossEntropyOp").
         */
        std::string getName() const override {
            return "Cpu::CrossEntropyOp";
        }
    };

    /**
     * @brief Class responsible for registering the CpuCrossEntropyOp operation.
     *
     * The CpuCrossEntropyOpRegistrar class registers the CpuCrossEntropyOp operation with the OperationRegistry.
     * It associates the operation name "Cpu::CrossEntropyOp" with a factory function that creates
     * instances of CpuCrossEntropyOp.
     */
    export class CpuCrossEntropyOpRegistrar {
    public:
        /**
         * @brief Registers the CpuCrossEntropyOp operation with the OperationRegistry.
         *
         * This function registers the CpuCrossEntropyOp operation for the CPU device type
         * with the OperationRegistry. It associates the operation name "Cpu::CrossEntropyOp"
         * with a factory function that creates instances of CpuCrossEntropyOp.
         */
        static void registerOperations() {
            const std::string opName = "Cpu::CrossEntropyOp";

            // Updated to use device context-aware registration
            OperationRegistry::instance().registerUnaryOperation<DeviceType::Cpu, int, float>(
                opName,
                []( std::shared_ptr<DeviceContext> context, ComputePrecision::Policy precision_policy ) -> std::shared_ptr<UnaryOperation<DeviceType::Cpu, int, float>> {
                    return context ? std::make_shared<CpuCrossEntropyOp>( context )
                        : std::make_shared<CpuCrossEntropyOp>();
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

