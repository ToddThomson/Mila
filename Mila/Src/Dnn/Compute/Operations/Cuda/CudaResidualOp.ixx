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
import Dnn.TensorTraits;
import Compute.Precision;
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
        template <typename TCompute>
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
     * vanishing gradient problem. The implementation is optimized for NVIDIA GPUs.
     *
     * @tparam TInput The data type of both input tensor elements.
     * @tparam TOutput The data type of the output tensor elements (defaults to TInput).
     * @tparam TCompute The data type used for computation (defaults to TOutput).
     */
    export template<typename TInput, typename TOutput = TInput, typename TCompute = TOutput>
        requires ValidFloatTensorType<TInput>&& ValidFloatTensorType<TOutput>&& ValidPrecisionType<TCompute>
    class CudaResidualOp : public BinaryOperation<TInput, TInput, TOutput, TCompute, DeviceType::Cuda> {
    public:
        using MR = typename CudaDevice::MR;
        using OperationBase = BinaryOperation<TInput, TInput, TOutput, TCompute, DeviceType::Cuda>;

        /**
         * @brief Constructs a new CUDA Residual operation with the default device context.
         *
         * Initializes the operation with a CUDA device context (defaults to CUDA:0).
         */
        CudaResidualOp() : OperationBase( OperationType::ResidualOp ) {}

        /**
         * @brief Constructs a new CUDA Residual operation with a specific device context.
         *
         * @param context The device context to use for this operation.
         * @throws std::runtime_error If the context is not for a CUDA device.
         */
        CudaResidualOp( std::shared_ptr<DeviceContext> context )
            : OperationBase( OperationType::ResidualOp, context ) {}

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
            const Tensor<TInput, MR>& input1,
            const Tensor<TInput, MR>& input2,
            const std::vector<std::shared_ptr<Tensor<TInput, MR>>>& parameters,
            const OperationAttributes& properties,
            Tensor<TOutput, MR>& output,
            std::vector<std::shared_ptr<Tensor<TOutput, MR>>>& output_state ) const override {

            auto X1 = input1.raw_data();
            auto X2 = input2.raw_data();
            auto Y = output.raw_data();
            int N = input1.size();

            cudaStream_t stream = this->getDeviceContext()->getStream();

            if constexpr ( std::is_same_v<TInput, TOutput> && std::is_same_v<TOutput, TCompute> ) {
                // All types are the same - direct computation
                Detail::cuda_residual_impl<TCompute>::forward( Y, X1, X2, N, stream );
            }
            else {
                // Handle mixed precision computation
                // For non-trivial mixed precision, we would need to implement 
                // type conversion here using cuda_convert_type or similar
                Detail::cuda_residual_impl<TCompute>::forward(
                    reinterpret_cast<TCompute*>(Y),
                    reinterpret_cast<const TCompute*>(X1),
                    reinterpret_cast<const TCompute*>(X2),
                    N, stream );
            }
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
            const Tensor<TInput, MR>& input1,
            const Tensor<TInput, MR>& input2,
            const Tensor<TOutput, MR>& output,
            const Tensor<TOutput, MR>& output_gradient,
            const std::vector<std::shared_ptr<Tensor<TInput, MR>>>& parameters,
            std::vector<std::shared_ptr<Tensor<TOutput, MR>>>& parameter_gradients,
            Tensor<TInput, MR>& input1_gradient,
            Tensor<TInput, MR>& input2_gradient,
            const OperationAttributes& properties,
            const std::vector<std::shared_ptr<Tensor<TOutput, MR>>>& output_state ) const {

            // Verify we're operating on CUDA memory
            if ( !this->getDeviceContext()->isDeviceType( DeviceType::Cuda ) ) {
                throw std::runtime_error( "CudaResidualOp::backward can only be executed on CUDA device" );
            }

            // Extract tensors
            const TOutput* dY = output_gradient.data();
            TInput* dX1 = input1_gradient.data();
            TInput* dX2 = input2_gradient.data();
            int N = input1.size();

            cudaStream_t stream = this->getDeviceContext()->getStream();

            // For residual connection, the gradient just flows through to both inputs
            // FIXME: cuda_residual_backward(dX1, dX2, dY, N, stream);
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
     */
    export class CudaResidualOpRegistrar {
    public:
        /**
         * @brief Registers the CudaResidualOp operation with the OperationRegistry.
         */
        static void registerOperations() {
            const std::string opName = "Cuda::ResidualOp";

            OperationRegistry::instance().registerBinaryOperation<float, float, float, float, DeviceType::Cuda>(
                opName,
                []( std::shared_ptr<DeviceContext> context ) -> std::shared_ptr<BinaryOperation<float, float, float, float, DeviceType::Cuda>> {
                    return context ? std::make_shared<CudaResidualOp<float>>( context )
                        : std::make_shared<CudaResidualOp<float>>();
                }
            );

            OperationRegistry::instance().registerBinaryOperation<half, half, half, half, DeviceType::Cuda>(
                opName,
                []( std::shared_ptr<DeviceContext> context ) -> std::shared_ptr<BinaryOperation<half, half, half, half, DeviceType::Cuda>> {
                    return context ? std::make_shared<CudaResidualOp<half>>( context )
                        : std::make_shared<CudaResidualOp<half>>();
                }
            );
        }

        /**
         * @brief Self-registration mechanism that registers the operation during startup.
         */
        static inline bool isRegistered = []() {
            registerOperations();
            return true;
            }();
    };
}