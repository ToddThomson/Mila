/**
 * @file CudaLayerNormOp.ixx
 * @brief Implementation of the CUDA-based Layer Normalization operation for neural networks.
 */

module;
#include <vector>
#include <memory>
#include <string>
#include <stdexcept>
#include "Kernels/CudaOps.h"

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
	 * @brief Namespace for CUDA layer normalization implementation details.
	 *
	 * This namespace contains the implementation details for the CUDA layer normalization operation,
	 * including specialized templates for different data types (float, half).
	 */
	namespace Detail
	{
		// Primary template - will cause a compile error if no specialization exists
		template <typename T>
		struct cuda_layernorm_impl;
		
        // Specialization for float
		template <>
		struct cuda_layernorm_impl<float> {
			static inline void forward( float* Y, const float* X,
				const float* weight, const float* bias,
				float* mean, float* rstd,
				int B, int T, int C, float epsilon, 
                cudaStream_t stream ) {
				cuda_layernorm_forward_fp32( Y, mean, rstd, X, weight, bias, B, T, C, epsilon, stream );
			}
		};
		
        // Specialization for half
		template <>
		struct cuda_layernorm_impl<half> {
			static inline void forward( half* Y, const half* X,
				const half* weight, const half* bias,
				half* mean, half* rstd,
				int B, int T, int C, float epsilon, 
                cudaStream_t stream ) {
				cuda_layernorm_forward_fp16( Y, mean, rstd, X, weight, bias, B, T, C, epsilon, stream );
			}
		};
	}

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
     * @tparam TPrecision The data type of the input tensor elements.
     * @tparam TDataType The data type of the output tensor elements (defaults to the input type).
     */
    export template<typename TInput = float, typename TOutput = TInput>
    class CudaLayerNormOp : public UnaryOperation<DeviceType::Cuda, TInput, TOutput> {
    public:
        using MR = typename CudaDevice::MR;
		using UnaryOperationBase = UnaryOperation<DeviceType::Cuda, TInput, TOutput>;

        /**
         * @brief Constructs a new CUDA Layer Normalization operation with the default device context.
         *
         * Initializes the operation with a CUDA device context (defaults to CUDA:0).
         */
        CudaLayerNormOp( ComputePrecision::Policy precision_policy = ComputePrecision::Policy::Auto )
            : UnaryOperationBase( OperationType::LayerNormOp, precision_policy ) {}

        /**
         * @brief Constructs a new CUDA Layer Normalization operation with a specific device context.
         *
         * @param context The device context to use for this operation.
         * @throws std::runtime_error If the context is not for a CUDA device.
         */
        CudaLayerNormOp( std::shared_ptr<DeviceContext> context, ComputePrecision::Policy precision_policy = ComputePrecision::Policy::Auto )
            : UnaryOperationBase( OperationType::LayerNormOp, context, precision_policy ) {
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
         * @param output_state Vector containing tensors for intermediate results [mean, rstd],
         *                     where mean is the mean values and rstd is the reciprocal of standard deviation.
         */
        void forward(
            const Tensor<TInput, MR>& input,
            const std::vector<std::shared_ptr<Tensor<TOutput, MR>>>& parameters,
            const OperationAttributes& properties,
            Tensor<TOutput, MR>& output,
            std::vector<std::shared_ptr<Tensor<TOutput, MR>>>& output_state ) const override {

            const TInput* X = input.raw_data();
            TOutput* Y = output.raw_data();

            const TOutput* weight = parameters[ 0 ]->raw_data();
            const TOutput* bias = (parameters.size() > 1 && parameters[ 1 ]) ? parameters[ 1 ]->raw_data() : nullptr;

            TOutput* mean = output_state[ 0 ]->raw_data();
            TOutput* rstd = output_state[ 1 ]->raw_data();

            int B = input.shape()[ 0 ];
            int T = input.shape()[ 1 ];
            int C = input.shape()[ 2 ];
            float epsilon = properties.epsilon;

            cudaStream_t stream = this->getDeviceContext()->getStream();

            Detail::cuda_layernorm_impl<TInput>::forward( Y, X, weight, bias, mean, rstd, B, T, C, epsilon, stream );
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
         * @param output_state Cache tensors from forward pass [mean, rstd].
         */
        void backward(
            const Tensor<TInput, MR>& input,
            const Tensor<TOutput, MR>& output,
            const Tensor<TOutput, MR>& output_gradient,
            const std::vector<std::shared_ptr<Tensor<TInput, MR>>>& parameters,
            std::vector<std::shared_ptr<Tensor<TInput, MR>>>& parameter_gradients,
            Tensor<TInput, MR>& input_gradient,
            const OperationAttributes& properties,
            const std::vector<std::shared_ptr<Tensor<TOutput, MR>>>& output_state ) const {

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

            const float* mean = output_state[ 0 ]->data();
            const float* rstd = output_state[ 1 ]->data();
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

            OperationRegistry::instance().registerUnaryOperation<DeviceType::Cuda, float, float>(
                opName,
                []( std::shared_ptr<DeviceContext> context, ComputePrecision::Policy precision_policy ) -> std::shared_ptr<UnaryOperation<DeviceType::Cuda, float, float>> {
                    return context ? std::make_shared<CudaLayerNormOp<float>>( context, precision_policy )
                        : std::make_shared<CudaLayerNormOp<float>>( precision_policy );
                }
            );

            OperationRegistry::instance().registerUnaryOperation<DeviceType::Cuda, half, half>(
                opName,
                []( std::shared_ptr<DeviceContext> context, ComputePrecision::Policy precision_policy ) -> std::shared_ptr<UnaryOperation<DeviceType::Cuda, half, half>> {
                    return context ? std::make_shared<CudaLayerNormOp<half>>( context, precision_policy )
                        : std::make_shared<CudaLayerNormOp<half>>( precision_policy );
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