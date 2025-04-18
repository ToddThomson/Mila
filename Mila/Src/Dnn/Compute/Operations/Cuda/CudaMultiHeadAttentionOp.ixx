/**
 * @file CudaMultiHeadAttentionOp.ixx
 * @brief Implementation of the CUDA-based Multi-Head Attention operation for transformer models.
 */

module;
#include <vector>
#include <memory>
#include <string>
#include <stdexcept>

#include "Kernels/Cuda.Ops.h"

export module Compute.CudaMHAOp;

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
     * @brief CUDA implementation of the Multi-Head Attention operation for transformer models.
     *
     * This class provides a CUDA-based implementation of the Multi-Head Attention operation,
     * which is a key component of transformer architectures. The operation allows the model to
     * jointly attend to information from different representation subspaces at different positions.
     *
     * Multi-Head Attention consists of several attention mechanisms operating in parallel:
     * 1. Linear projections of the input into query, key, and value vectors
     * 2. Scaled dot-product attention computation between queries and keys
     * 3. Applying attention weights to values
     * 4. Concatenation of attention outputs from different heads
     *
     * The implementation is optimized for NVIDIA GPUs using CUDA for high-performance computation.
     *
     * @tparam TInput The data type of the input tensor elements.
     * @tparam TPrecision The data type of the output tensor elements (defaults to the input type).
     */
    export
        template<typename TInput, typename TPrecision = TInput>
    class CudaMultiHeadAttentionOp : public UnaryOperation<TInput, TPrecision, DeviceType::Cuda> {
    public:
        using MR = typename CudaDevice::MR;
        /**
         * @brief Constructs a new CUDA Multi-Head Attention operation with the default device context.
         *
         * Initializes the operation with a CUDA device context (defaults to CUDA:0).
         */
        CudaMultiHeadAttentionOp() : UnaryOperation<TInput, TPrecision, DeviceType::Cuda>( OperationType::MultiHeadAttentionOp ) {}

        /**
         * @brief Constructs a new CUDA Multi-Head Attention operation with a specific device context.
         *
         * @param context The device context to use for this operation.
         * @throws std::runtime_error If the context is not for a CUDA device.
         */
        CudaMultiHeadAttentionOp( std::shared_ptr<DeviceContext> context )
            : UnaryOperation<TInput, TPrecision, DeviceType::Cuda>( OperationType::MultiHeadAttentionOp, context ) {
            if ( !context->isDeviceType( DeviceType::Cuda ) ) {
                throw std::runtime_error( "CudaMultiHeadAttentionOp requires a CUDA device context." );
            }
        }

        /**
         * @brief Performs the forward pass of the Multi-Head Attention operation on CUDA.
         *
         * Computes attention scores, applies softmax to get attention weights, and uses these
         * weights to compute a weighted sum of value vectors. This process is performed in
         * parallel for multiple attention heads, then outputs are concatenated and projected.
         *
         * The computation is performed on the GPU using CUDA kernels for optimal performance.
         *
         * @param input Input tensor of shape [B, T, C] containing the input sequence, where B is batch size,
         *              T is sequence length, and C is the input feature dimension.
         * @param parameters Vector of parameter tensors [weight, bias], where weight contains the query, key,
         *                   value projections and output projection, and bias contains the corresponding biases.
         * @param properties Additional attributes for the operation, such as number of attention heads.
         * @param output Output tensor of shape [B, T, OC] containing the attention output, where OC is the
         *               output feature dimension.
         * @param output_cache Cache for intermediate results like attention scores and weights for
         *                     potential use in backward pass or visualization.
         */
        void forward(
            const Tensor<TInput, MR>& input,
            const std::vector<std::shared_ptr<Tensor<TInput, MR>>>& parameters,
            const OperationAttributes& properties,
            Tensor<TPrecision, MR>& output,
            std::vector<std::shared_ptr<Tensor<TPrecision, MR>>>& output_cache ) const override {

            // Verify we're operating on CUDA memory
            if ( !this->getDeviceContext()->isDeviceType( DeviceType::Cuda ) ) {
                throw std::runtime_error( "CudaMultiHeadAttentionOp::forward can only be executed on CUDA memory" );
            }

            auto X = input.data();
            auto Y = output.data();

            auto weight = parameters[ 0 ]->data();
            auto bias = parameters[ 1 ]->data();

            int B = input.shape()[ 0 ];
            int T = input.shape()[ 1 ];
            int C = input.shape()[ 2 ];
            int OC = output.shape()[ 2 ];

            int num_heads = properties.get<int>( "num_heads", 1 );

            float* attn_scores = nullptr;
            float* attn_weights = nullptr;

            if ( output_cache.size() >= 2 ) {
                attn_scores = output_cache[ 0 ]->data();
                attn_weights = output_cache[ 1 ]->data();
            }

            // Get CUDA stream from device context
            cudaStream_t stream = this->getDeviceContext()->getStream();

            // Call CUDA kernel with stream
            // cuda_mha_forward(Y, X, weight, bias, attn_scores, attn_weights, B, T, C, OC, num_heads, stream);
        }

        /**
         * @brief Performs the backward pass of the Multi-Head Attention operation.
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
         * @param output_cache Cache tensors from forward pass (attention scores and weights).
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
                throw std::runtime_error( "CudaMultiHeadAttentionOp::backward can only be executed on CUDA memory" );
            }

            // Extract dimensions
            int B = input.shape()[ 0 ];
            int T = input.shape()[ 1 ];
            int C = input.shape()[ 2 ];
            int OC = output.shape()[ 2 ];

            // Get the number of attention heads from properties
            int num_heads = properties.get<int>( "num_heads", 1 );

            // Extract tensors
            const TInput* X = input.data();
            const TPrecision* dY = output_gradient.data();
            TInput* dX = input_gradient.data();

            const TInput* W = parameters[ 0 ]->data();
            const TInput* bias = parameters[ 1 ]->data();
            TInput* dW = parameter_gradients[ 0 ]->data();
            TInput* dBias = parameter_gradients[ 1 ]->data();

            // Get cached attention data if available
            const TPrecision* attn_scores = nullptr;
            const TPrecision* attn_weights = nullptr;
            if ( output_cache.size() >= 2 ) {
                attn_scores = output_cache[ 0 ]->data();
                attn_weights = output_cache[ 1 ]->data();
            }

            // Get CUDA stream from device context
            cudaStream_t stream = this->getDeviceContext()->getStream();

            // Call CUDA backward kernels with stream
            // cuda_mha_backward(dX, dW, dBias, X, W, bias, dY, attn_scores, attn_weights, 
            //                   B, T, C, OC, num_heads, stream);
        }

        /**
         * @brief Gets the name of this operation.
         *
         * @return std::string The name of the operation ("Cuda::MultiHeadAttentionOp").
         */
        std::string getName() const override {
            return "Cuda::MultiHeadAttentionOp";
        }
    };

    /**
     * @brief Class responsible for registering the CudaMultiHeadAttentionOp operation.
     *
     * The CudaMultiHeadAttentionOpRegistrar class registers the CudaMultiHeadAttentionOp operation
     * with the OperationRegistry. It associates the operation name "Cuda::MultiHeadAttentionOp"
     * with a factory function that creates instances of CudaMultiHeadAttentionOp.
     */
    export class CudaMultiHeadAttentionOpRegistrar {
    public:
        /**
         * @brief Registers the CudaMultiHeadAttentionOp operation with the OperationRegistry.
         *
         * This function registers the CudaMultiHeadAttentionOp operation for the CUDA device type
         * with the OperationRegistry. It associates the operation name "Cuda::MultiHeadAttentionOp"
         * with a factory function that creates instances of CudaMultiHeadAttentionOp.
         */
        static void registerOperations() {
            const std::string opName = "Cuda::MultiHeadAttentionOp";

            // Updated to use device context-aware registration
            OperationRegistry::instance().registerOperation<float, float, DeviceType::Cuda>(
                opName,
                "Default",  // Default empty variant for backward compatibility
                []( std::shared_ptr<DeviceContext> context ) -> std::shared_ptr<OperationBase<float, float, DeviceType::Cuda>> {
                    return context ? std::make_shared<CudaMultiHeadAttentionOp<float, float>>( context )
                        : std::make_shared<CudaMultiHeadAttentionOp<float, float>>();
                }
            );

            // Add additional precision variants if needed, for example:
            /*
            OperationRegistry::instance().registerOperation<float, half, DeviceType::Cuda>(
                opName,
                "half_precision",
                []( std::shared_ptr<DeviceContext> context ) -> std::shared_ptr<OperationBase<float, half, DeviceType::Cuda>> {
                    return context ? std::make_shared<CudaMultiHeadAttentionOp<float, half>>( context )
                        : std::make_shared<CudaMultiHeadAttentionOp<float, half>>();
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

