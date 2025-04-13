/**
 * @file CudaSoftmaxOp.ixx
 * @brief Implementation of the CUDA-based softmax operation for neural networks.
 */

module;
#include <vector>
#include <iostream>
#include <memory>
#include "Kernels/Cuda.Ops.h"

export module Compute.CudaSoftmaxOp;

import Dnn.Tensor;
import Compute.OperationBase;
import Compute.UnaryOperation;
import Compute.OperationRegistry;
import Compute.DeviceType;
import Compute.DeviceContext;
import Compute.OperationType;
import Compute.OperationAttributes;
import Compute.MemoryResource;
import Compute.CudaMemoryResource;
import Compute.CudaDevice;

using namespace Mila::Dnn;

namespace Mila::Dnn::Compute
{
    /**
     * @brief CUDA implementation of the softmax operation for neural networks.
     *
     * This class provides a CUDA-based implementation of the softmax operation,
     * which converts a vector of real numbers into a probability distribution.
     * The softmax function is commonly used in classification tasks as the
     * final activation function of a neural network, and in attention mechanisms
     * within transformer architectures.
     *
     * The implementation is optimized for NVIDIA GPUs using CUDA for high-performance
     * computation, especially for large vocabulary sizes typical in language models.
     *
     * @tparam TInput The data type of the input tensor elements.
     * @tparam TPrecision The data type of the output tensor elements (defaults to the input type).
     */
    export
        template<typename TInput, typename TPrecision = TInput>
    class CudaSoftmaxOp : public UnaryOperation<TInput, TPrecision, DeviceType::Cuda> {
    public:
        using MR = typename CudaDevice::MR;
        /**
         * @brief Constructs a new CUDA Softmax operation with the default device context.
         *
         * Initializes the operation with a CUDA device context (defaults to CUDA:0).
         */
        CudaSoftmaxOp() : UnaryOperation<TInput, TPrecision, DeviceType::Cuda>( OperationType::SoftmaxOp ) {}

        /**
         * @brief Constructs a new CUDA Softmax operation with a specific device context.
         *
         * @param context The device context to use for this operation.
         * @throws std::runtime_error If the context is not for a CUDA device.
         */
        CudaSoftmaxOp( std::shared_ptr<DeviceContext> context )
            : UnaryOperation<TInput, TPrecision, DeviceType::Cuda>( OperationType::SoftmaxOp, context ) {
            if ( !context->isDeviceType( DeviceType::Cuda ) ) {
                throw std::runtime_error( "CudaSoftmaxOp requires a CUDA device context" );
            }
        }

        /**
         * @brief Performs the forward pass of the softmax operation on CUDA.
         *
         * Converts input logits into a probability distribution by taking the
         * exponential of each element and normalizing by their sum. The computation
         * is performed on the GPU using CUDA kernels for optimal performance.
         *
         * The implementation includes numerical stability improvements by subtracting
         * the maximum value before applying the exponential function.
         *
         * @param input Input tensor containing logits of shape [B, T, V], where B is batch size,
         *              T is sequence length, and V is vocabulary size.
         * @param parameters Additional parameters (not used in this operation).
         * @param properties Additional attributes for the operation.
         * @param output Output tensor of shape [B, T, V] to store the resulting probability distribution.
         * @param output_cache Cache for intermediate results (not used in this operation).
         */
        void forward(
            const Tensor<TInput, MR>& input,
            const std::vector<std::shared_ptr<Tensor<TInput, MR>>>& parameters,
            const OperationAttributes& properties,
            Tensor<TPrecision, MR>& output,
            std::vector<std::shared_ptr<Tensor<TPrecision, MR>>>& output_cache ) const override {

            // Verify we're operating on CUDA memory
            if ( !this->getDeviceContext()->isDeviceType( DeviceType::Cuda ) ) {
                throw std::runtime_error( "CudaSoftmaxOp::forward can only be executed on CUDA memory" );
            }

            auto X = input.data();
            auto Y = output.data();
            int N = input.size();

            int axis = properties.axis;

            // Get CUDA stream from device context
            cudaStream_t stream = this->getDeviceContext()->getStream();

            // Call CUDA kernel with stream
            cuda_softmax_forward( Y, X, N, axis, stream );
        }

        /**
         * @brief Performs the backward pass of the softmax operation.
         *
         * Computes gradients with respect to inputs based on the output gradient.
         * For softmax: dL/dx_i = ?_j (dL/dy_j * (y_i * (?_ij - y_j)))
         * where ?_ij is the Kronecker delta.
         *
         * @param input Input tensor from the forward pass.
         * @param output Output tensor from the forward pass (softmax probabilities).
         * @param output_gradient Gradient of the loss with respect to the output.
         * @param parameters Parameters tensor from forward pass (not used).
         * @param parameter_gradients Gradients for parameters (not used).
         * @param input_gradient Gradient of the loss with respect to the input.
         * @param properties Additional attributes for the operation.
         * @param output_cache Cache tensors from forward pass.
         */
        void backward(
            const Tensor<TInput, MR>& input,
            const Tensor<TPrecision, MR>& output,
            const Tensor<TPrecision, MR>& output_gradient,
            const std::vector<std::shared_ptr<Tensor<TInput, MR>>>& parameters,
            std::vector<std::shared_ptr<Tensor<TInput, MR>>>& parameter_gradients,
            Tensor<TInput, MR>& input_gradient,
            const OperationAttributes& properties,
            const std::vector<std::shared_ptr<Tensor<TPrecision, MR>>>& output_cache ) const {

            // Verify we're operating on CUDA memory
            if ( !this->getDeviceContext()->isDeviceType( DeviceType::Cuda ) ) {
                throw std::runtime_error( "CudaSoftmaxOp::backward can only be executed on CUDA memory" );
            }

            // Extract tensors
            const TPrecision* Y = output.data();
            const TPrecision* dY = output_gradient.data();
            TInput* dX = input_gradient.data();
            int N = input.size();

            // Get the axis parameter from properties
            int axis = properties.axis;

            // Get CUDA stream from device context
            cudaStream_t stream = this->getDeviceContext()->getStream();

            // Call CUDA kernel with stream
            cuda_softmax_backward( dX, dY, Y, N, axis, stream );
        }

        /**
         * @brief Gets the name of this operation.
         *
         * @return std::string The name of the operation ("Cuda::SoftmaxOp").
         */
        std::string getName() const override {
            return "Cuda::SoftmaxOp";
        }
    };

    /**
     * @brief Class responsible for registering the CudaSoftmaxOp operation.
     *
     * The CudaSoftmaxOpRegistrar class registers the CudaSoftmaxOp operation with the OperationRegistry.
     * It associates the operation name "Cuda::SoftmaxOp" with a factory function that creates
     * instances of CudaSoftmaxOp.
     */
    export class CudaSoftmaxOpRegistrar {
    public:
        /**
         * @brief Registers the CudaSoftmaxOp operation with the OperationRegistry.
         *
         * This function registers the CudaSoftmaxOp operation for the CUDA device type
         * with the OperationRegistry. It associates the operation name "Cuda::SoftmaxOp"
         * with a factory function that creates instances of CudaSoftmaxOp.
         */
        static void registerOperations() {
            const std::string opName = "Cuda::SoftmaxOp";

            // Updated to use device context-aware registration
            OperationRegistry::instance().registerOperation<float, float, DeviceType::Cuda>(
                opName,
                "Default",  // Default empty variant for backward compatibility
                []( std::shared_ptr<DeviceContext> context ) -> std::shared_ptr<OperationBase<float, float, DeviceType::Cuda>> {
                    return context ? std::make_shared<CudaSoftmaxOp<float, float>>( context )
                        : std::make_shared<CudaSoftmaxOp<float, float>>();
                }
            );

            // Add additional precision variants if needed, for example:
            /*
            OperationRegistry::instance().registerOperation<float, half, DeviceType::Cuda>(
                opName,
                "half_precision",
                []( std::shared_ptr<DeviceContext> context ) -> std::shared_ptr<OperationBase<float, half, DeviceType::Cuda>> {
                    return context ? std::make_shared<CudaSoftmaxOp<float, half>>( context )
                        : std::make_shared<CudaSoftmaxOp<float, half>>();
                }
            );
            */
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