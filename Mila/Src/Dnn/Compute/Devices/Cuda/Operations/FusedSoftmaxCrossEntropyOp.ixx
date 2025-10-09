/**
 * @file FusedSoftmaxCrossEntropyOp.ixx
 * @brief Implementation of the CUDA-based fused softmax and cross entropy operation for neural networks.
 */

module;
#include <vector>
#include <iostream>
#include <memory>
#include "Kernels/CudaOps.h"
#include <type_traits>

export module Compute.FusedSoftmaxCrossEntropyOp;

import Dnn.Tensor;
import Compute.OperationBase;
import Compute.BinaryOperation;
import Compute.OperationRegistry;
import Compute.DeviceType;
import Compute.DeviceContext;
import Compute.OperationType;
import Compute.OperationAttributes;
import Compute.MemoryResource;
import Compute.CudaDeviceMemoryResource;
import Compute.CudaDevice;

namespace Mila::Dnn::Compute
{
    using namespace Mila::Dnn;

    /**
     * @brief Namespace for CUDA fused softmax cross entropy implementation details.
     *
     * This namespace contains the implementation details for the CUDA fused softmax cross entropy operation,
     * including specialized templates for different data types (float, half).
     */
    namespace Detail
    {
        // Primary template - will cause a compile error if no specialization exists
        template <typename T>
        struct cuda_softmax_crossentropy_impl;

        // Specialization for float
        template <>
        struct cuda_softmax_crossentropy_impl<float> {
            static inline void forward(
                float* losses, float* probs, const float* logits, const int* targets,
                int batch_size, int seq_len, int vocab_size,
                cudaStream_t stream ) {
                cuda_softmax_crossentropy_forward<float>(
                    losses, probs, logits, targets, batch_size, seq_len, vocab_size, stream );
            }

            static inline void backward(
                float* dlogits, const float* dlosses, const float* probs, const int* targets,
                int batch_size, int seq_len, int vocab_size,
                cudaStream_t stream ) {
                cuda_softmax_crossentropy_backward<float>(
                    dlogits, dlosses, probs, targets, batch_size, seq_len, vocab_size, stream );
            }
        };

        // Specialization for half
        template <>
        struct cuda_softmax_crossentropy_impl<half> {
            static inline void forward(
                half* losses, half* probs, const half* logits, const int* targets,
                int batch_size, int seq_len, int vocab_size,
                cudaStream_t stream ) {
                cuda_softmax_crossentropy_forward<half>(
                    losses, probs, logits, targets, batch_size, seq_len, vocab_size, stream );
            }

            static inline void backward(
                half* dlogits, const half* dlosses, const half* probs, const int* targets,
                int batch_size, int seq_len, int vocab_size,
                cudaStream_t stream ) {
                cuda_softmax_crossentropy_backward<half>(
                    dlogits, dlosses, probs, targets, batch_size, seq_len, vocab_size, stream );
            }
        };
    }

    /**
     * @brief CUDA implementation of the fused softmax and cross entropy operation for neural networks.
     *
     * This class provides a CUDA-based implementation of the fused softmax and cross entropy operation,
     * which combines two commonly used operations in neural networks to improve computational efficiency.
     * First, the softmax function converts a vector of real numbers (logits) into a probability
     * distribution. Then, the cross entropy computes the negative log likelihood of the correct class given
     * the predicted probabilities.
     *
     * The implementation is optimized for NVIDIA GPUs using CUDA for high-performance
     * computation, especially for large vocabulary sizes typical in language models.
     *
     * @tparam TPrecision The data type used for computation (float or half).
     */
    export template<typename TPrecision>
        requires (std::is_same_v<TPrecision, float> || std::is_same_v<TPrecision, half>)
    class FusedSoftmaxCrossEntropyOp : public BinaryOperation<TPrecision, int, TPrecision, DeviceType::Cuda> {
    public:
        using MR = typename CudaDevice::MR;

        /**
         * @brief Constructs a new CUDA Fused Softmax Cross Entropy operation with the default device context.
         *
         * Initializes the operation with a CUDA device context (defaults to CUDA:0).
         */
        FusedSoftmaxCrossEntropyOp() : BinaryOperation<TPrecision, int, TPrecision, DeviceType::Cuda>( OperationType::FusedOp ) {}

        /**
         * @brief Constructs a new CUDA Fused Softmax Cross Entropy operation with a specific device context.
         *
         * @param context The device context to use for this operation.
         * @throws std::runtime_error If the context is not for a CUDA device.
         */
        FusedSoftmaxCrossEntropyOp( std::shared_ptr<DeviceContext> context )
            : BinaryOperation<TPrecision, int, TPrecision, DeviceType::Cuda>( OperationType::FusedOp, context ) {}

        /**
         * @brief Performs the forward pass of the fused softmax cross entropy operation on CUDA.
         *
         * Converts input logits into a probability distribution using softmax and then
         * computes the cross entropy loss between the predicted probabilities and target classes.
         *
         * @param logits Input tensor containing logits of shape [B, S, V], where B is batch size,
         *               S is sequence length, and V is vocabulary size.
         * @param input2 Input tensor containing target class indices of shape [B, S].
         * @param parameters Additional parameters (not used in this operation).
         * @param properties Additional attributes for the operation.
         * @param output Output tensor of shape [B, S] to store the resulting cross entropy loss values.
         * @param output_state Cache for intermediate results, stores the computed probabilities for the backward pass.
         */
        void forward(
            const Tensor<TPrecision, MR>& logits,
            const Tensor<int, MR>& targets,
            const std::vector<std::shared_ptr<Tensor<TPrecision, MR>>>& parameters,
            Tensor<TPrecision, MR>& losses,
            std::vector<std::shared_ptr<Tensor<TPrecision, MR>>>& output_state ) const override {

            // Extract dimensions
            auto batch_size = logits.shape()[ 0 ];
            auto seq_len = logits.shape()[ 1 ];
            auto vocab_size = logits.shape()[ 2 ];

            // Get pointers to raw data
            const TPrecision* logits = logits.data();
            const int* targets = targets.data();
            TPrecision* losses = losses.data();

            // Ensure output_state has a tensor to store probabilities
            if ( output_state.empty() ) {
                output_state.push_back( std::make_shared<Tensor<TPrecision, MR>>( logits.shape(), this->getDeviceContext() ) );
            }
            else if ( output_state[ 0 ]->shape() != logits.shape() ) {
                output_state[ 0 ]->reshape( logits.shape() );
            }

            TPrecision* probs = output_state[ 0 ]->data();

            // Get CUDA stream from device context
            cudaStream_t stream = this->getDeviceContext()->getStream();

            // Call the implementation
            Detail::cuda_softmax_crossentropy_impl<TPrecision>::forward(
                losses, probs, logits, targets, batch_size, seq_len, vocab_size, stream );
        }

        /**
         * @brief Performs the backward pass of the fused softmax cross entropy operation.
         *
         * Computes gradients with respect to the logits by combining the gradients
         * from both the softmax and cross entropy operations.
         *
         * @param input1 Input tensor from the forward pass (logits).
         * @param input2 Input tensor from the forward pass (target indices).
         * @param output Output tensor from the forward pass (loss values).
         * @param output_gradient Gradient of the loss with respect to the output.
         * @param parameters Parameters tensor from forward pass (not used).
         * @param parameter_gradients Gradients for parameters (not used).
         * @param input1_gradient Gradient of the loss with respect to the logits.
         * @param input2_gradient Not used as we don't compute gradients for integer indices.
         * @param properties Additional attributes for the operation.
         * @param output_state Cache tensors from forward pass containing computed probabilities.
         */
        void backward(
            const Tensor<TPrecision, MR>& input1,  // logits
            const Tensor<int, MR>& input2,         // target indices
            const Tensor<TPrecision, MR>& output,  // loss values
            const Tensor<TPrecision, MR>& output_gradient,
            const std::vector<std::shared_ptr<Tensor<TPrecision, MR>>>& parameters,
            std::vector<std::shared_ptr<Tensor<TPrecision, MR>>>& parameter_gradients,
            Tensor<TPrecision, MR>& input1_gradient,  // dlogits
            Tensor<int, MR>& input2_gradient,
            const std::vector<std::shared_ptr<Tensor<TPrecision, MR>>>& output_state ) const override {

            // No gradients for integer target indices

            // Extract dimensions
            auto batch_size = input1.shape()[ 0 ];
            auto seq_len = input1.shape()[ 1 ];
            auto vocab_size = input1.shape()[ 2 ];

            // Get pointers to raw data
            const int* targets = input2.data();
            const TPrecision* dlosses = output_gradient.data();
            TPrecision* dlogits = input1_gradient.data();

            // Get probabilities from output_state
            if ( output_state.empty() || output_state[ 0 ]->shape() != input1.shape() ) {
                throw std::runtime_error( "Missing or invalid probabilities in output_state" );
            }
            const TPrecision* probs = output_state[ 0 ]->data();

            // Get CUDA stream from device context
            cudaStream_t stream = this->getDeviceContext()->getStream();

            // Call the implementation
            Detail::cuda_softmax_crossentropy_impl<TPrecision>::backward(
                dlogits, dlosses, probs, targets, batch_size, seq_len, vocab_size, stream );
        }

        /**
         * @brief Gets the name of this operation.
         *
         * @return std::string The name of the operation ("Cuda::FusedSoftmaxCrossEntropyOp").
         */
        std::string getName() const override {
            return "Cuda::FusedSoftmaxCrossEntropyOp";
        }
    };

    /**
     * @brief Class responsible for registering the FusedSoftmaxCrossEntropyOp operation.
     *
     * The FusedSoftmaxCrossEntropyOpRegistrar class registers the FusedSoftmaxCrossEntropyOp operation
     * with the OperationRegistry. It associates the operation name "Cuda::FusedSoftmaxCrossEntropyOp"
     * with a factory function that creates instances of FusedSoftmaxCrossEntropyOp.
     */
    export class FusedSoftmaxCrossEntropyOpRegistrar {
    public:
        /**
         * @brief Registers the FusedSoftmaxCrossEntropyOp operation with the OperationRegistry.
         *
         * This function registers the FusedSoftmaxCrossEntropyOp operation for the CUDA device type
         * with the OperationRegistry. It associates the operation name "Cuda::FusedSoftmaxCrossEntropyOp"
         * with a factory function that creates instances of FusedSoftmaxCrossEntropyOp.
         */
        static void registerOperations() {
            const std::string opName = "Cuda::FusedSoftmaxCrossEntropyOp";

            // Register float version
            OperationRegistry::instance().registerBinaryOperation<float, int, float, DeviceType::Cuda>(
                opName,
                "Default",
                []( std::shared_ptr<DeviceContext> context ) -> std::shared_ptr<BinaryOperation<float, int, float, DeviceType::Cuda>> {
                    return context ? std::make_shared<FusedSoftmaxCrossEntropyOp<float>>( context )
                        : std::make_shared<FusedSoftmaxCrossEntropyOp<float>>();
                }
            );

            // Register half version
            OperationRegistry::instance().registerBinaryOperation<half, int, half, DeviceType::Cuda>(
                opName,
                "Default",
                []( std::shared_ptr<DeviceContext> context ) -> std::shared_ptr<BinaryOperation<half, int, half, DeviceType::Cuda>> {
                    return context ? std::make_shared<FusedSoftmaxCrossEntropyOp<half>>( context )
                        : std::make_shared<FusedSoftmaxCrossEntropyOp<half>>();
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
