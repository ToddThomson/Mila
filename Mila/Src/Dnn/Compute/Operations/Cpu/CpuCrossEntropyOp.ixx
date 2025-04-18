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
     * @tparam TPrecision The data type used for computation and output (typically float).
     */
    export
        template<typename TInput = int, typename TPrecision = float>
    class CpuCrossEntropyOp : public UnaryOperation<TInput, TPrecision, DeviceType::Cpu> {
    public:
        using MR = typename CpuDevice::MR;
        /**
         * @brief Constructs a new CPU Cross Entropy operation with the default device context.
         *
         * Initializes the operation with a CPU device context.
         */
        CpuCrossEntropyOp() : UnaryOperation<TInput, TPrecision, DeviceType::Cpu>( OperationType::CrossEntropyOp ) {

        }

        /**
         * @brief Constructs a new CPU Cross Entropy operation with a specific device context.
         *
         * @param context The device context to use for this operation.
         * @throws std::runtime_error If the context is not for a CPU device.
         */
        CpuCrossEntropyOp( std::shared_ptr<DeviceContext> context )
            : UnaryOperation<TInput, TPrecision, DeviceType::Cpu>( OperationType::CrossEntropyOp, context ) {
            if ( !context->isDeviceType( DeviceType::Cpu ) ) {
                throw std::runtime_error( "CpuCrossEntropyOp requires a CPU device context." );
            }
        }

        /**
         * @brief Performs the forward pass of the cross entropy operation.
         *
         * Computes the negative log likelihood of the correct class for each sample.
         *
         * @param input Input tensor containing target class indices of shape [B, T].
         * @param parameters Parameters tensor containing probabilities of shape [B, T, V].
         * @param attributes Additional attributes for the operation.
         * @param output Output tensor to store the cross entropy losses of shape [B, T].
         * @param output_cache Cache for storing intermediate results (used in backward pass).
         */
        void forward(
            const Tensor<TInput, MR>& input,
            const std::vector<std::shared_ptr<Tensor<TPrecision, MR>>>& parameters,
            const OperationAttributes& attributes,
            Tensor<TPrecision, MR>& output,
            std::vector<std::shared_ptr<Tensor<TPrecision, MR>>>& output_cache ) const override {

            // Verify we're operating on CPU memory
            if ( !this->getDeviceContext()->isDeviceType( DeviceType::Cpu ) ) {
                throw std::runtime_error( "CpuCrossEntropyOp::forward can only be executed on CPU memory" );
            }

            auto B = input.shape()[ 0 ];
            auto T = input.shape()[ 1 ];
            auto Vp = parameters[ 0 ]->shape()[ 2 ];

            auto losses = output.raw_data();
            auto probs = parameters[ 0 ]->raw_data();
            auto targets = input.raw_data();

            const TPrecision epsilon = 1e-10f; // Small value to prevent log(0)

        #pragma omp parallel for collapse(2) if(B * T > 100)
            for ( int b = 0; b < B; b++ ) {
                for ( int t = 0; t < T; t++ ) {
                    // loss = -log(probs[target])
                    TPrecision* probs_bt = probs + b * T * Vp + t * Vp;
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
         * @param output_cache Cache tensors from forward pass.
         */
        void backward(
            const Tensor<TInput, MR>& input,
            const Tensor<TPrecision, MR>& output,
            const Tensor<TPrecision, MR>& output_gradient,
            const std::vector<std::shared_ptr<Tensor<TPrecision, MR>>>& parameters,
            std::vector<std::shared_ptr<Tensor<TPrecision, MR>>>& parameter_gradients,
            Tensor<TInput, MR>& input_gradient,
            const OperationAttributes& attributes,
            const std::vector<std::shared_ptr<Tensor<TPrecision, MR>>>& output_cache ) const {

            // Verify we're operating on CPU memory
            if ( !this->getDeviceContext()->isDeviceType( DeviceType::Cpu ) ) {
                throw std::runtime_error( "CpuCrossEntropyOp::backward can only be executed on CPU memory" );
            }

            // For combined softmax-crossentropy backward pass
            if ( parameter_gradients.size() > 0 ) {
                auto B = input.shape()[ 0 ];
                auto T = input.shape()[ 1 ];
                auto Vp = parameters[ 0 ]->shape()[ 2 ];
                auto V = attributes.get<int>( "vocab_size", Vp ); // Actual vocabulary size (without padding)

                TPrecision* dlogits = parameter_gradients[ 0 ]->raw_data();
                const TPrecision* dlosses = output_gradient.raw_data();
                const TPrecision* probs = parameters[ 0 ]->raw_data();

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
         * @param T Sequence length.
         * @param V Vocabulary size (without padding).
         * @param Vp Padded vocabulary size.
         */
        void backward_impl(
            TPrecision* dlogits,
            const TPrecision* dlosses,
            const TPrecision* probs,
            const Tensor<TInput, HostMemoryResource>& targets,
            int B, int T, int V, int Vp ) const {

            // Backwards through both softmax and crossentropy
        #pragma omp parallel for collapse(2) if(B * T > 100)
            for ( int b = 0; b < B; b++ ) {
                for ( int t = 0; t < T; t++ ) {
                    TPrecision* dlogits_bt = dlogits + b * T * Vp + t * Vp;
                    const TPrecision* probs_bt = probs + b * T * Vp + t * Vp;
                    TPrecision dloss = dlosses[ b * T + t ];
                    int ix = targets.raw_data()[ b * T + t ];

                    // Only process valid indices
                    if ( ix >= 0 && ix < V ) {
                        // Note we only loop to V, leaving the padded dimensions
                        // of dlogits untouched, so gradient there stays at zero
                        for ( int i = 0; i < V; i++ ) {
                            TPrecision p = probs_bt[ i ];
                            TPrecision indicator = i == ix ? 1.0f : 0.0f;
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
            OperationRegistry::instance().registerOperation<int, float, DeviceType::Cpu>(
                opName,
                "Default",  // Default empty variant for backward compatibility
                []( std::shared_ptr<DeviceContext> context ) -> std::shared_ptr<OperationBase<int, float, DeviceType::Cpu>> {
                    return context ? std::make_shared<CpuCrossEntropyOp<int, float>>( context )
                        : std::make_shared<CpuCrossEntropyOp<int, float>>();
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

