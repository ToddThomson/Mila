/**
 * @file CudaResidualOp.ixx
 * @brief Implementation of the CUDA-based residual operation for neural networks.
 */

module;
#include <vector>
#include <iostream>
#include "Kernels/CudaOps.h"

export module Compute.CudaResidualOp;

import Dnn.Tensor;
import Compute.OperationBase;
import Compute.OperationAttributes;
import Compute.BinaryOperation;
import Compute.OperationRegistry;
import Compute.DeviceType;
import Compute.DeviceContext;
import Compute.OperationType;
import Compute.MemoryResource;
import Compute.CudaMemoryResource;
import Compute.CudaDevice;

namespace Mila::Dnn::Compute
{
    using namespace Mila::Dnn;

	/**
	 * @brief Namespace for CUDA residual implementation details.
	 *
	 * This namespace contains the implementation details for the CUDA residual operation,
	 * including specialized templates for different data types (float, half).
	 */
	namespace Detail
	{
		// Primary template - will cause a compile error if no specialization exists
		template <typename TPrecision>
		struct cuda_residual_impl;

		// Specialization for float
		template <>
		struct cuda_residual_impl<float> {
			static inline void forward( float* Y, const float* X1, const float* X2, int N, cudaStream_t stream ) {
				cuda_residual_forward_fp32( Y, X1, X2, N, stream );
			}
		};

		// Specialization for half
		template <>
		struct cuda_residual_impl<half> {
			static inline void forward( half* Y, const half* X1, const half* X2, int N, cudaStream_t stream ) {
				cuda_residual_forward_fp16( Y, X1, X2, N, stream );
			}
		};
	}

    /**
     * @brief CUDA implementation of the residual operation for neural networks.
     *
     * This class provides a CUDA-based implementation of the residual operation,
     * which performs element-wise addition of two input tensors.
     * It is commonly used in residual connections in neural network architectures
     * such as ResNet and Transformers to help with gradient flow and mitigate the
     * vanishing gradient problem. The implementation is optimized for NVIDIA GPUs
     * using CUDA for high-performance computation.
     *
     * @tparam TPrecision The data type of the input tensor elements.
     * @tparam TDataType The data type of the output tensor elements (defaults to the input type).
     */
	export template<typename TPrecision>
		requires (std::is_same_v<TPrecision, float> || std::is_same_v<TPrecision, half>)
    class CudaResidualOp : public BinaryOperation<TPrecision> {
    public:
        using MR = typename CudaDevice::MR;

        /**
         * @brief Constructs a new CUDA Residual operation with the default device context.
         *
         * Initializes the operation with a CUDA device context (defaults to CUDA:0).
         */
        CudaResidualOp() : BinaryOperation<TPrecision>( OperationType::ResidualOp ) {}

        /**
         * @brief Constructs a new CUDA Residual operation with a specific device context.
         *
         * @param context The device context to use for this operation.
         * @throws std::runtime_error If the context is not for a CUDA device.
         */
        CudaResidualOp( std::shared_ptr<DeviceContext> context )
            : BinaryOperation<TPrecision>( OperationType::ResidualOp, context ) {
        }

        /**
         * @brief Performs the forward pass of the residual operation on CUDA.
         *
         * Adds two input tensors element-wise and stores the result in the output tensor.
         * The computation is performed on the GPU using CUDA kernels for optimal performance.
         *
         * @param input1 The first input tensor to be added.
         * @param input2 The second input tensor to be added.
         * @param parameters Additional parameters (not used in this operation).
         * @param properties Additional attributes for the operation.
         * @param output The output tensor where the results will be stored.
         * @param output_state Cache for intermediate results (not used in this operation).
         */
        void forward(
            const Tensor<TPrecision, MR>& input1,
            const Tensor<TPrecision, MR>& input2,
            const std::vector<std::shared_ptr<Tensor<TPrecision, MR>>>& parameters,
            const OperationAttributes& properties,
            Tensor<TPrecision, MR>& output,
            std::vector<std::shared_ptr<Tensor<TPrecision, MR>>>& output_state ) const override {

            auto X1 = input1.raw_data();
            auto X2 = input2.raw_data();
            auto Y = output.raw_data();
            int N = input1.size();

            cudaStream_t stream = this->getDeviceContext()->getStream();
            
            Detail::cuda_residual_impl<TPrecision>::forward( Y, X1, X2, N, stream );
        }

        /**
         * @brief Performs the backward pass of the residual operation.
         *
         * Computes gradients with respect to both inputs by propagating the output
         * gradient to each input.
         *
         * @param input1 First input tensor from the forward pass.
         * @param input2 Second input tensor from the forward pass.
         * @param output Output tensor from the forward pass.
         * @param output_gradient Gradient of the loss with respect to the output.
         * @param parameters Parameters tensor from forward pass (not used in this operation).
         * @param parameter_gradients Gradients for parameters (not used in this operation).
         * @param input1_gradient Gradient of the loss with respect to input1.
         * @param input2_gradient Gradient of the loss with respect to input2.
         * @param properties Additional attributes for the operation.
         * @param output_state Cache tensors from forward pass (not used in this operation).
         */
        void backward(
            const Tensor<TPrecision, MR>& input1,
            const Tensor<TPrecision, MR>& input2,
            const Tensor<TPrecision, MR>& output,
            const Tensor<TPrecision, MR>& output_gradient,
            const std::vector<std::shared_ptr<Tensor<TPrecision, MR>>>& parameters,
            std::vector<std::shared_ptr<Tensor<TPrecision, MR>>>& parameter_gradients,
            Tensor<TPrecision, MR>& input1_gradient,
            Tensor<TPrecision, MR>& input2_gradient,
            const OperationAttributes& properties,
            const std::vector<std::shared_ptr<Tensor<TPrecision, MR>>>& output_state ) const {

            // Verify we're operating on CUDA memory
            if ( !this->getDeviceContext()->isDeviceType( DeviceType::Cuda ) ) {
                throw std::runtime_error( "CudaResidualOp::backward can only be executed on CUDA device" );
            }

            // Extract tensors
            const TPrecision* dY = output_gradient.data();
            TPrecision* dX1 = input1_gradient.data();
            TPrecision* dX2 = input2_gradient.data();
            int N = input1.size();

            cudaStream_t stream = this->getDeviceContext()->getStream();

            // For residual connection, the gradient just flows through to both inputs
            // FIXME: cuda_residual_backward( dX1, dX2, dY, N, stream );
        }

        /**
         * @brief Gets the name of this operation.
         *
         * @return std::string The name of the operation ("Cuda::ResidualOp").
         */
        std::string getName() const override {
            return "Cuda::ResidualOp";
        }
    };

    /**
     * @brief Class responsible for registering the CudaResidualOp operation.
     *
     * The CudaResidualOpRegistrar class registers the CudaResidualOp operation with the OperationRegistry.
     * It associates the operation name "Cuda::ResidualOp" with a factory function that creates
     * instances of CudaResidualOp.
     */
    export class CudaResidualOpRegistrar {
    public:
        /**
         * @brief Registers the CudaResidualOp operation with the OperationRegistry.
         *
         * This function registers the CudaResidualOp operation for the CUDA device type
         * with the OperationRegistry. It associates the operation name "Cuda::ResidualOp"
         * with a factory function that creates instances of CudaResidualOp.
         */
        static void registerOperations() {
            const std::string opName = "Cuda::ResidualOp";

            OperationRegistry::instance().registerBinaryOperation<float, float, float, DeviceType::Cuda>(
                opName,
                "Default",
                []( std::shared_ptr<DeviceContext> context ) -> std::shared_ptr<BinaryOperation<float, float, float, DeviceType::Cuda>> {
                    return context ? std::make_shared<CudaResidualOp<float>>( context )
                        : std::make_shared<CudaResidualOp<float>>();
                }
            );

            OperationRegistry::instance().registerBinaryOperation<half, half, half, DeviceType::Cuda>(
                opName,
                "Default",
                []( std::shared_ptr<DeviceContext> context ) -> std::shared_ptr<BinaryOperation<half, half, half, DeviceType::Cuda>> {
                    return context ? std::make_shared<CudaResidualOp<half>>( context )
                        : std::make_shared<CudaResidualOp<half>>();
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

