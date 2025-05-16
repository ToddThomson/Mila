/**
 * @file CudaMultiHeadAttentionOp.ixx
 * @brief Implementation of the CUDA-based Multi-Head Attention operation for transformer models.
 */

module;
#include <vector>
#include <memory>
#include <string>
#include <stdexcept>
#include "Kernels/CudaOps.h"

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
    namespace Detail
    {
        /**
         * @brief Implementation details for CUDA-based Multi-Head Attention operations.
         *
         * This namespace contains specialized implementations of Multi-Head Attention
         * operations for different data types (float, half) using CUDA kernels.
         * The implementations are optimized for NVIDIA GPUs for high-performance computation
         * of attention mechanisms in transformer architectures.
         */

         // Primary template - will cause a compile error if no specialization exists
        template <typename T>
        struct cuda_mha_impl;

        // Specialization for float
        template <>
        struct cuda_mha_impl<float> {
            static inline void forward( float* out,
                float* qkvr, float* att,
                const float* inp,
                int B, int T, int C, int NH,
                cudaStream_t stream ) {
                
                cuda_mha_forward_fp32( out, qkvr, att, inp, B, T, C, NH, stream );
            }

            static inline void backward(/* Parameters to be added when backward implementation is ready */ ) {
                // Implementation will be added when cuda_attention_backward is available
            }
        };

        // Specialization for half
        template <>
        struct cuda_mha_impl<half> {
            static inline void forward( half* out,
                half* qkvr, half* att,
                const half* inp,
                int B, int T, int C, int NH,
                cudaStream_t stream ) {
                // Assuming there's a half-precision version available
                // cuda_attention_forward_fp16(out, qkvr, att, inp, B, T, C, NH, stream);

                // If no half-precision specific implementation exists yet:
                // TODO: Implement half-precision version or add a conversion wrapper
            }

            static inline void backward(/* Parameters to be added when backward implementation is ready */ ) {
                // Implementation will be added when cuda_attention_backward is available for half precision
            }
        };
    }

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
     * @tparam TPrecision The data type of the input tensor elements.
     * @tparam TDataType The data type of the output tensor elements (defaults to the input type).
     */
    export template<typename TInput = float, typename TOutput = TInput, typename TPrecision = TOutput>
		requires ValidFloatTensorType<TPrecision>
    class CudaMultiHeadAttentionOp : public UnaryOperation<DeviceType::Cuda, TInput, TOutput, TPrecision> {
    public:
        using MR = typename CudaDevice::MR;
		using UnaryOperationBase = UnaryOperation<DeviceType::Cuda, TInput, TOutput, TPrecision>;
        /**
         * @brief Constructs a new CUDA Multi-Head Attention operation with the default device context.
         *
         * Initializes the operation with a CUDA device context (defaults to CUDA:0).
         */
        CudaMultiHeadAttentionOp() : UnaryOperationBase( OperationType::MultiHeadAttentionOp ) {}

        /**
         * @brief Constructs a new CUDA Multi-Head Attention operation with a specific device context.
         *
         * @param context The device context to use for this operation.
         * @throws std::runtime_error If the context is not for a CUDA device.
         */
        CudaMultiHeadAttentionOp( std::shared_ptr<DeviceContext> context )
            : UnaryOperationBase( OperationType::MultiHeadAttentionOp, context ) {
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
         * @param input Input tensor of shape [B, TDataType, C] containing the input sequence, where B is batch size,
         *              TDataType is sequence length, and C is the input feature dimension.
         * @param parameters Vector of parameter tensors [weight, bias], where weight contains the query, key,
         *                   value projections and output projection, and bias contains the corresponding biases.
         * @param properties Additional attributes for the operation, such as number of attention heads.
         * @param output Output tensor of shape [B, TDataType, OC] containing the attention output, where OC is the
         *               output feature dimension.
         * @param output_state Intermediate results like attention scores and weights for
         *                     potential use in backward pass or visualization.
         */
        void forward(
            const Tensor<TPrecision, MR>& input,
            const std::vector<std::shared_ptr<Tensor<TPrecision, MR>>>& parameters,
            const OperationAttributes& properties,
            Tensor<TPrecision, MR>& output,
            std::vector<std::shared_ptr<Tensor<TPrecision, MR>>>& output_state ) const override {

            auto X = input.raw_data();
            auto Y = output.raw_data();

            auto weight = parameters[ 0 ]->raw_data();
            auto bias = parameters[ 1 ]->raw_data();

            int B = input.shape()[ 0 ];
            int T = input.shape()[ 1 ];
            int C = input.shape()[ 2 ];
            int OC = output.shape()[ 2 ];

            int num_heads = properties.get<int>( "num_heads", 1 );

            TPrecision* attn_scores = nullptr;
            TPrecision* attn_weights = nullptr;

            if ( output_state.size() >= 2 ) {
                attn_scores = output_state[ 0 ]->data();
                attn_weights = output_state[ 1 ]->data();
            }

            cudaStream_t stream = this->getDeviceContext()->getStream();

            //Detail::cuda_mha_impl<TPrecision>::forward( Y, X, weight, bias, attn_scores, attn_weights, B, T, C, OC, num_heads, stream );
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
         * @param output_state Cache tensors from forward pass (attention scores and weights).
         */
        void backward(
            const Tensor<TPrecision, MR>& input,
            const Tensor<TPrecision, MR>& output,
            const Tensor<TPrecision, MR>& output_gradient,
            const std::vector<std::shared_ptr<Tensor<TPrecision, MR>>>& parameters,
            std::vector<std::shared_ptr<Tensor<TPrecision, MR>>>& parameter_gradients,
            Tensor<TPrecision, MR>& input_gradient,
            const OperationAttributes& properties,
            const std::vector<std::shared_ptr<Tensor<TPrecision, MR>>>& output_state ) const {

            // Extract dimensions
            int B = input.shape()[ 0 ];
            int T = input.shape()[ 1 ];
            int C = input.shape()[ 2 ];
            int OC = output.shape()[ 2 ];

            // Get the number of attention heads from properties
            int num_heads = properties.get<int>( "num_heads", 1 );

            // Extract tensors
            const TPrecision* X = input.data();
            const TPrecision* dY = output_gradient.data();
            TPrecision* dX = input_gradient.data();

            const TPrecision* W = parameters[ 0 ]->data();
            const TPrecision* bias = parameters[ 1 ]->data();
            TPrecision* dW = parameter_gradients[ 0 ]->data();
            TPrecision* dBias = parameter_gradients[ 1 ]->data();

            // Get cached attention data if available
            const TPrecision* attn_scores = nullptr;
            const TPrecision* attn_weights = nullptr;
            if ( output_state.size() >= 2 ) {
                attn_scores = output_state[ 0 ]->data();
                attn_weights = output_state[ 1 ]->data();
            }

            // Get CUDA stream from device context
            cudaStream_t stream = this->getDeviceContext()->getStream();

            // Call CUDA backward kernels with stream
            // cuda_mha_backward(dX, dW, dBias, X, W, bias, dY, attn_scores, attn_weights, 
            //                   B, TDataType, C, OC, num_heads, stream);
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

            OperationRegistry::instance().registerUnaryOperation<DeviceType::Cuda, float, float, float>(
                opName,
                []( std::shared_ptr<DeviceContext> context ) -> std::shared_ptr<UnaryOperation<DeviceType::Cuda, float, float, float>> {
                    return context ? std::make_shared<CudaMultiHeadAttentionOp<float>>( context )
                        : std::make_shared<CudaMultiHeadAttentionOp<float>>();
                }
            );
            
            OperationRegistry::instance().registerUnaryOperation<DeviceType::Cuda, half, half, half>(
                opName,
                []( std::shared_ptr<DeviceContext> context ) -> std::shared_ptr<UnaryOperation<DeviceType::Cuda, half, half, half>> {
                    return context ? std::make_shared<CudaMultiHeadAttentionOp<half>>( context )
                        : std::make_shared<CudaMultiHeadAttentionOp<half>>();
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

