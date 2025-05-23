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
#include <type_traits>

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
     * for large neural network models. It also supports different precision modes via the
     * ComputePrecision policy.
     *
     * @tparam TOutput The data type of the output tensor elements.
     * @tparam TInput The data type of the input tensor elements (defaults to TOutput).
     */
    export template<typename TInput, typename TOutput = TInput>
    class CudaGeluOp : public UnaryOperation<DeviceType::Cuda, TInput, TOutput> {
    public:
        using MR = typename CudaDevice::MR;
        using UnaryOperationBase = UnaryOperation<DeviceType::Cuda, TInput, TOutput>;

        /**
        * @brief Constructs a new CUDA GELU operation with the default device context.
        *
        * Initializes the operation with a CUDA device context (defaults to CUDA:0).
        *
        * @param precision_policy The precision policy to use for mixed precision computation.
        */
        CudaGeluOp( ComputePrecision::Policy precision_policy = ComputePrecision::Policy::Auto )
            : UnaryOperationBase( OperationType::GeluOp, precision_policy ) {}

        /**
        * @brief Constructs a new CUDA GELU operation with a specific device context.
        *
        * @param context The device context to use for this operation.
        * @param precision_policy The precision policy to use for mixed precision computation.
        * @throws std::runtime_error If the context is not for a CUDA device.
        */
        CudaGeluOp( std::shared_ptr<DeviceContext> context,
            ComputePrecision::Policy precision_policy = ComputePrecision::Policy::Auto )
            : UnaryOperationBase( OperationType::GeluOp, context, precision_policy ) {}

        /**
        * @brief Performs the forward pass of the GELU activation function on CUDA.
        *
        * Computes the GELU transformation of the input elements:
        * GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/?) * (x + 0.044715 * x^3)))
        *
        * The precision policy affects how the computation is performed:
        * - Performance: May use faster but less precise algorithms
        * - Accuracy: Will use the most accurate algorithm available
        * - Auto: Will select an appropriate balance based on the hardware
        * - Disabled: Will use the standard precision of the input/output types
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

            ComputePrecision::Policy policy = this->getPrecisionPolicy();

            // Check if properties override the precision policy
			// This isn't needed for now, but could be useful in the future
            /*if ( properties.has( "precision_policy" ) ) {
                policy = static_cast<ComputePrecision::Policy>(properties.get( "precision_policy", static_cast<int>(policy) ));
            }*/

            auto X = input.raw_data();
            auto Y = output.raw_data();
            int N = input.size();
            cudaStream_t stream = this->getDeviceContext()->getStream();

            // For now, we use the same implementation regardless of policy
            // In a more advanced implementation, different kernels could be selected based on the policy
            if constexpr ( std::is_same_v<TInput, TOutput> ) {
                Detail::cuda_gelu_impl<TInput>::forward( Y, X, N, stream );
            }
        }

        /**
        * @brief Performs the backward pass of the GELU activation function.
        *
        * Computes gradients with respect to inputs for the GELU function.
        * The precision policy affects the computation in the same way as the forward pass.
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

            // Get precision policy from operation base class or override from properties
            ComputePrecision::Policy policy = this->getPrecisionPolicy();

            // Check if properties override the precision policy
            if ( properties.has( "precision_policy" ) ) {
                policy = static_cast<ComputePrecision::Policy>(properties.get( "precision_policy", static_cast<int>(policy) ));
            }

            // Get tensor data pointers
            const TInput* X = input.raw_data();
            const TOutput* dY = output_gradient.raw_data();
            TInput* dX = input_gradient.raw_data();
            int N = input.size();

            // Get CUDA stream from device context
            cudaStream_t stream = this->getDeviceContext()->getStream();

            // Call CUDA kernel with stream
            // FIXME: cuda_gelu_backward(dX, X, dY, N, stream);
            // Future implementation should respect the precision policy
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

            OperationRegistry::instance().registerUnaryOperation<DeviceType::Cuda, float, float>(
                opName,
                []( std::shared_ptr<DeviceContext> context, ComputePrecision::Policy precision_policy ) -> std::shared_ptr<UnaryOperation<DeviceType::Cuda, float, float>> {
                    return context ? std::make_shared<CudaGeluOp<float>>( context, precision_policy )
                        : std::make_shared<CudaGeluOp<float, float>>( precision_policy );
                }
            );

            OperationRegistry::instance().registerUnaryOperation<DeviceType::Cuda, half, half>(
                opName,
                []( std::shared_ptr<DeviceContext> context, ComputePrecision::Policy precision_policy ) -> std::shared_ptr<UnaryOperation<DeviceType::Cuda, half, half>> {
                    return context ? std::make_shared<CudaGeluOp<half>>( context, precision_policy )
                        : std::make_shared<CudaGeluOp<half>>( precision_policy );
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