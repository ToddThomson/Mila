/**
 * @file CudaLayerNormOp.ixx
 * @brief Implementation of the CUDA-based Layer Normalization operation for neural networks.
 */

module;
#include <vector>
#include <memory>
#include <string>
#include <stdexcept>
#include "Kernels/Cuda.Ops.h"

export module Compute.CudaLayerNormOp;

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
     * @brief CUDA implementation of the Layer Normalization operation for neural networks.
     *
     * This class provides a CUDA-based implementation of the Layer Normalization operation,
     * which normalizes the activations of a layer for each example in a batch, usually applied
     * before the activation function. Layer normalization helps stabilize the learning process
     * and reduce the training time required to learn the parameters of neural networks.
     *
     * The normalization is applied across the last dimension (feature dimension) and includes
     * learnable scale (gamma) and shift (beta) parameters. The implementation is optimized for
     * NVIDIA GPUs using CUDA for high-performance computation.
     *
     * @tparam TInput The data type of the input tensor elements.
     * @tparam TDataType The data type of the output tensor elements (defaults to the input type).
     */
    export
        template<typename TInput, typename TPrecision = TInput>
    class CudaLayerNormOp : public UnaryOperation<TInput, TPrecision, DeviceType::Cuda> {
    public:
        using MR = typename CudaDevice::MR;
        /**
         * @brief Constructs a new CUDA Layer Normalization operation with the default device context.
         *
         * Initializes the operation with a CUDA device context (defaults to CUDA:0).
         */
        CudaLayerNormOp() : UnaryOperation<TInput, TPrecision, DeviceType::Cuda>( OperationType::LayerNormOp ) {}

        /**
         * @brief Constructs a new CUDA Layer Normalization operation with a specific device context.
         *
         * @param context The device context to use for this operation.
         * @throws std::runtime_error If the context is not for a CUDA device.
         */
        CudaLayerNormOp( std::shared_ptr<DeviceContext> context )
            : UnaryOperation<TInput, TPrecision, DeviceType::Cuda>( OperationType::LayerNormOp, context ) {
            if ( !context->isDeviceType( DeviceType::Cuda ) ) {
                throw std::runtime_error( "CudaLayerNormOp requires a CUDA device context" );
            }
        }

        /**
         * @brief Performs the forward pass of the Layer Normalization operation on CUDA.
         *
         * Normalizes the input tensor across the feature dimension (last dimension) by:
         * 1. Computing the mean and standard deviation of each sample
         * 2. Normalizing the values using these statistics
         * 3. Applying learnable scale and shift parameters
         *
         * The computation is performed on the GPU using CUDA kernels for optimal performance.
         *
         * @param input Input tensor of shape [B, TDataType, C] to be normalized, where B is batch size,
         *              TDataType is sequence length, and C is feature dimension.
         * @param parameters Vector of parameter tensors [weight, bias] where weight (gamma) and
         *                   bias (beta) are both of shape [C].
         * @param properties Additional attributes for the operation.
         * @param output Output tensor of shape [B, TDataType, C] containing the normalized values.
         * @param output_cache Vector containing tensors for intermediate results [mean, rstd],
         *                     where mean is the mean values and rstd is the reciprocal of standard deviation.
         */
        void forward(
            const Tensor<TInput, MR>& input,
            const std::vector<std::shared_ptr<Tensor<TInput, MR>>>& parameters,
            const OperationAttributes& properties,
            Tensor<TPrecision, MR>& output,
            std::vector<std::shared_ptr<Tensor<TPrecision, MR>>>& output_cache ) const override {

            const float* X = input.data();
            float* Y = output.data();

            const float* weight = parameters[ 0 ]->data();
            const float* bias = parameters[ 1 ]->data();

            float* mean = output_cache[ 0 ]->data();
            float* rstd = output_cache[ 1 ]->data();

            int B = input.shape()[ 0 ];
            int T = input.shape()[ 1 ];
            int C = input.shape()[ 2 ];
            float epsilon = properties.epsilon;

            cudaStream_t stream = this->getDeviceContext()->getStream();

            cuda_layernorm_forward( Y, mean, rstd, X, weight, bias, B, T, C, /* TODO epsilon, */ stream );
        }

        /**
         * @brief Performs the backward pass of the Layer Normalization operation.
         *
         * Computes gradients with respect to inputs, weights, and biases.
         *
         * @param input Input tensor from the forward pass.
         * @param output Output tensor from the forward pass.
         * @param output_gradient Gradient of the loss with respect to the output.
         * @param parameters Parameters tensor from forward pass [weight, bias].
         * @param parameter_gradients Gradients for parameters [d_weight, d_bias].
         * @param input_gradient Gradient of the loss with respect to the input.
         * @param properties Additional attributes for the operation.
         * @param output_cache Cache tensors from forward pass [mean, rstd].
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
                throw std::runtime_error( "CudaLayerNormOp::backward can only be executed on CUDA memory" );
            }

            // Extract dimensions
            int B = input.shape()[ 0 ];
            int T = input.shape()[ 1 ];
            int C = input.shape()[ 2 ];

            const float* X = input.data();
            const float* dY = output_gradient.data();
            float* dX = input_gradient.data();

            const float* weight = parameters[ 0 ]->data();
            float* dweight = parameter_gradients[ 0 ]->data();
            float* dbias = parameter_gradients[ 1 ]->data();

            const float* mean = output_cache[ 0 ]->data();
            const float* rstd = output_cache[ 1 ]->data();
            float epsilon = properties.epsilon;

            cudaStream_t stream = this->getDeviceContext()->getStream();

            //cuda_layernorm_backward( dX, dweight, dbias, dY, X, weight, mean, rstd, B, TDataType, C, stream );
        }

        /**
         * @brief Gets the name of this operation.
         *
         * @return std::string The name of the operation ("Cuda::LayerNormOp").
         */
        std::string getName() const override {
            return "Cuda::LayerNormOp";
        }
    };

    /**
     * @brief Class responsible for registering the CudaLayerNormOp operation.
     *
     * The CudaLayerNormOpRegistrar class registers the CudaLayerNormOp operation with the OperationRegistry.
     * It associates the operation name "Cuda::LayerNormOp" with a factory function that creates
     * instances of CudaLayerNormOp.
     */
    export class CudaLayerNormOpRegistrar {
    public:
        /**
         * @brief Registers the CudaLayerNormOp operation with the OperationRegistry.
         *
         * This function registers the CudaLayerNormOp operation for the CUDA device type
         * with the OperationRegistry. It associates the operation name "Cuda::LayerNormOp"
         * with a factory function that creates instances of CudaLayerNormOp.
         */
        static void registerOperations() {
            const std::string opName = "Cuda::LayerNormOp";

            // Updated to use device context-aware registration
            OperationRegistry::instance().registerOperation<float, float, DeviceType::Cuda>(
                opName,
                "Default",  // Default empty variant for backward compatibility
                []( std::shared_ptr<DeviceContext> context ) -> std::shared_ptr<OperationBase<float, float, DeviceType::Cuda>> {
                    return context ? std::make_shared<CudaLayerNormOp<float, float>>( context )
                        : std::make_shared<CudaLayerNormOp<float, float>>();
                }
            );

            // Add additional precision variants if needed, for example:
            /*
            OperationRegistry::instance().registerOperation<float, half, DeviceType::Cuda>(
                opName,
                "half_precision",
                []( std::shared_ptr<DeviceContext> context ) -> std::shared_ptr<OperationBase<float, half, DeviceType::Cuda>> {
                    return context ? std::make_shared<CudaLayerNormOp<float, half>>( context )
                        : std::make_shared<CudaLayerNormOp<float, half>>();
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