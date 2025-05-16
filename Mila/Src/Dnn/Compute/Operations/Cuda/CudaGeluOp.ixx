/**
 * @file CudaGeluOp.ixx
 * @brief Implementation of the CUDA-based GELU activation function for neural networks.
 */

module;
#include <vector>
#include <memory>
#include <iostream>
#include <cuda_fp16.h>
#include "Kernels/CudaOps.h"
#include <stdexcept>

export module Compute.CudaGeluOp;

import Dnn.Tensor;
import Dnn.TensorTraits;
import Compute.Precision;
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

namespace Mila::Dnn::Compute
{
    namespace Detail
    {
        // Primary template - will cause a compile error if no specialization exists
        template <typename TPrecision>
        struct cuda_gelu_impl;
        // Specialization for float
        template <>
        struct cuda_gelu_impl<float> {
            static inline void forward( float* Y, const float* X, int N, cudaStream_t stream ) {
                cuda_gelu_forward_fp32( Y, X, N, stream );
            }
        };
        // Specialization for half
        template <>
        struct cuda_gelu_impl<half> {
            static inline void forward( half* Y, const half* X, int N, cudaStream_t stream ) {
                cuda_gelu_forward_fp16( Y, X, N, stream );
            }
        };
    }

    using namespace Mila::Dnn;

    /**
     * @brief CUDA implementation of the GELU activation function for neural networks.
     *
     * This class provides a CUDA-based implementation of the Gaussian Error Linear Unit (GELU)
     * activation function, which is commonly used in transformer architectures. GELU is a smooth
     * approximation of the ReLU function that applies a non-linear transformation to its input.
     *
     * The implementation leverages CUDA for GPU acceleration, providing efficient computation
     * for large neural network models.
     *
     * @tparam TOutput The data type of the output tensor elements.
     * @tparam TInput The data type of the input tensor elements (defaults to TOutput).
     */
    export template<typename TInput, typename TOutput = TInput, typename TPrecision = TOutput>
    class CudaGeluOp : public UnaryOperation<TInput, TOutput, TPrecision, DeviceType::Cuda> {
        public:
            using MR = typename CudaDevice::MR;
			using OperationBase = UnaryOperation<TInput, TOutput, TPrecision, DeviceType::Cuda>;

            /**
             * @brief Constructs a new CUDA GELU operation with the default device context.
             *
             * Initializes the operation with a CUDA device context (defaults to CUDA:0).
             */
            CudaGeluOp() : UnaryOperation<TInput, TOutput, TPrecision, DeviceType::Cuda>( OperationType::GeluOp ) {}

            /**
             * @brief Constructs a new CUDA GELU operation with a specific device context.
             *
             * @param context The device context to use for this operation.
             * @throws std::runtime_error If the context is not for a CUDA device.
             */
            CudaGeluOp( std::shared_ptr<DeviceContext> context )
                : UnaryOperation<TInput, TOutput, TPrecision, DeviceType::Cuda>( OperationType::GeluOp, context ) {}

            /**
             * @brief Performs the forward pass of the GELU activation function on CUDA.
             *
             * Computes the GELU transformation of the input elements:
             * GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/?) * (x + 0.044715 * x^3)))
             *
             * @param input Input tensor containing the values to transform.
             * @param parameters Additional parameters (not used in this operation).
             * @param properties Additional attributes for the operation.
             * @param output Output tensor to store the transformed values.
             * @param output_state Cache for intermediate results (not used in this operation).
             */
            void forward(
                const Tensor<TInput, MR>& input,
                const std::vector<std::shared_ptr<Tensor<TOutput, MR>>>& parameters,
                const OperationAttributes& properties,
                Tensor<TOutput, MR>& output,
                std::vector<std::shared_ptr<Tensor<TOutput, MR>>>& output_state ) const override {

                // Verify we're operating on CUDA memory
                if ( !this->getDeviceContext()->isDeviceType( DeviceType::Cuda ) ) {
                    throw std::runtime_error( "CudaGeluOp::forward can only be executed on CUDA memory" );
                }

                auto X = input.data();
                auto Y = output.data();
                int N = input.size();

                cudaStream_t stream = this->getDeviceContext()->getStream();

                Detail::cuda_gelu_impl<TOutput>::forward( Y, X, N, stream );
            }

            /**
             * @brief Performs the backward pass of the GELU activation function.
             *
             * Computes gradients with respect to inputs for the GELU function.
             *
             * @param input Input tensor from the forward pass.
             * @param output Output tensor from the forward pass.
             * @param output_gradient Gradient of the loss with respect to the output.
             * @param parameters Parameters tensor from forward pass (not used).
             * @param parameter_gradients Gradients for parameters (not used).
             * @param input_gradient Gradient of the loss with respect to the input.
             * @param properties Additional attributes for the operation.
             * @param output_state Cache tensors from forward pass.
             */
            void backward(
                const Tensor<TInput, MR>& input,
                const Tensor<TOutput, MR>& output,
                const Tensor<TOutput, MR>& output_gradient,
                const std::vector<std::shared_ptr<Tensor<TOutput, MR>>>& parameters,
                std::vector<std::shared_ptr<Tensor<TOutput, MR>>>& parameter_gradients,
                Tensor<TInput, MR>& input_gradient,
                const OperationAttributes& properties,
                const std::vector<std::shared_ptr<Tensor<TOutput, MR>>>& output_state ) const {

                // Verify we're operating on CUDA memory
                if ( !this->getDeviceContext()->isDeviceType( DeviceType::Cuda ) ) {
                    throw std::runtime_error( "CudaGeluOp::backward can only be executed on CUDA memory" );
                }

                // Get tensor data pointers
                const TInput* X = input.data();
                const TOutput* dY = output_gradient.data();
                TInput* dX = input_gradient.data();
                int N = input.size();

                // Get CUDA stream from device context
                cudaStream_t stream = this->getDeviceContext()->getStream();

                // Call CUDA kernel with stream
                // FIXME: cuda_gelu_backward( dX, X, dY, N, stream );
            }

            /**
             * @brief Gets the name of this operation.
             *
             * @return std::string The name of the operation ("Cuda::GeluOp").
             */
            std::string getName() const override {
                return "Cuda::GeluOp";
            }
    };

    /**
     * @brief Class responsible for registering the CudaGeluOp operation.
     *
     * The CudaGeluOpRegistrar class registers the CudaGeluOp operation with the OperationRegistry.
     * It associates the operation name "Cuda::GeluOp" with a factory function that creates instances of CudaGeluOp.
     */
    export class CudaGeluOpRegistrar {
    public:
        /**
        * @brief Registers the CudaGeluOp operation with the OperationRegistry.
        *
        * This function registers the CudaGeluOp operation for the CUDA device type
        * with the OperationRegistry. It associates the operation name "Cuda::GeluOp"
        * with a factory function that creates instances of CudaGeluOp.
        */
        static void registerOperations() {
            const std::string opName = "Cuda::GeluOp";

            // Register float-to-float operation
            OperationRegistry::instance().registerUnaryOperation<float, float, float, DeviceType::Cuda>(
                opName,
                []( std::shared_ptr<DeviceContext> context ) -> std::shared_ptr<UnaryOperation<float, float, float, DeviceType::Cuda>> {
                    return context ? std::make_shared<CudaGeluOp<float, float>>( context )
                        : std::make_shared<CudaGeluOp<float, float>>();
                }
            );

            // Register half-to-half operation
            OperationRegistry::instance().registerUnaryOperation<half, half, half, DeviceType::Cuda>(
                opName,
                []( std::shared_ptr<DeviceContext> context ) -> std::shared_ptr<UnaryOperation<half, half, half, DeviceType::Cuda>> {
                    return context ? std::make_shared<CudaGeluOp<half>>( context )
                        : std::make_shared<CudaGeluOp<half>>();
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
