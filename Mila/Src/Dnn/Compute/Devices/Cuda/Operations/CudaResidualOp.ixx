/**
 * @file CudaResidualOp.ixx
 * @brief Implementation of the CUDA-based residual operation for neural networks.
 */

module;
#include <vector>
#include <memory>
#include <iostream>
#include <cuda_fp16.h>
#include "Kernels/CudaOps.h"
#include <stdexcept>
#include <type_traits>

export module Compute.CudaResidualOp;

import Dnn.Modules.Residual;
import Dnn.Tensor;
import Dnn.ConfigurationBase;
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

        template <>
        struct cuda_residual_impl<float> {
            static inline void forward( float* Y, const float* X1, const float* X2, int N, cudaStream_t stream ) {
                cuda_residual_forward_fp32( Y, X1, X2, N, stream );
            }
        };

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
     * The implementation leverages CUDA for GPU acceleration, providing efficient computation
     * for large neural network models. It also supports different precision modes via the
     * ComputePrecision policy.
     *
     * @tparam TInput The data type of both input tensor elements.
     * @tparam TOutput The data type of the output tensor elements (defaults to TInput).
     */
    export template<typename TInput, typename TOutput = TInput>
        requires ValidFloatTensorTypes<TInput, TOutput>
    class CudaResidualOp : public BinaryOperation<DeviceType::Cuda, TInput, TOutput> {
    public:
        using MR = typename CudaDevice::MR;
        using BinaryOperationBase = BinaryOperation<DeviceType::Cuda, TInput, TOutput>;

        /**
         * @brief Constructs a new CUDA Residual operation with the default device context.
         *
         * Initializes the operation with a CUDA device context (defaults to CUDA:0).
         *
         * @param precision_policy The precision policy to use for mixed precision computation.
         */
        CudaResidualOp( const ResidualConfig& config )
            : BinaryOperationBase( OperationType::ResidualOp ), config_( config ) {}

        /**
         * @brief Constructs a new CUDA Residual operation with a specific device context.
         *
         * @param context The device context to use for this operation.
         * @param precision_policy The precision policy to use for mixed precision computation.
         * @throws std::runtime_error If the context is not for a CUDA device.
         */
        CudaResidualOp( std::shared_ptr<DeviceContext> context, const ResidualConfig& config )
            : BinaryOperationBase( OperationType::ResidualOp, context ), config_( config ) {}

        /**
         * @brief Performs the forward pass of the residual operation on CUDA.
         *
         * Adds two input tensors element-wise and stores the result in the output tensor.
         * The computation is performed on the GPU using CUDA kernels for optimal performance.
         *
         * The precision policy affects how the computation is performed:
         * - Performance: May use faster but less precise algorithms
         * - Accuracy: Will use the most accurate algorithm available
         * - Auto: Will select an appropriate balance based on the hardware
         * - Disabled: Will use the standard precision of the input/output types
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
            const std::vector<std::shared_ptr<ITensorData>>& parameters,
            Tensor<TOutput, MR>& output,
            std::vector<std::shared_ptr<Tensor<TOutput, MR>>>& output_state ) const override {

            // Get precision policy from operation base class
            ComputePrecision::Policy policy = config_.getPrecisionPolicy();

            auto X1 = input1.rawData();
            auto X2 = input2.rawData();
            auto Y = output.rawData();
            int N = input1.size();

            cudaStream_t stream = this->getDeviceContext()->getStream();

            // For now, we use the same implementation regardless of policy
            // In a more advanced implementation, different kernels could be selected based on the policy
            if constexpr ( std::is_same_v<TInput, TOutput> ) {
                // FIXME: Detail::cuda_residual_impl<TInput>::forward( Y, X1, X2, N, stream );
            }
            else {
                // Handle mixed precision computation based on the precision policy
                // Future implementations could have different paths for different policies
                // For example, we might choose different algorithms based on Performance vs Accuracy

                // Currently just use the default implementation - in the future this could be expanded
                // to handle mixed precision formats more efficiently

                // Note: This implementation might need to be updated when mixed precision kernels are available
                // For now, we would need to implement type conversion here
            }
        }

        /**
         * @brief Performs the backward pass of the residual operation.
         *
         * Computes gradients with respect to both inputs by propagating the output
         * gradient to each input.
         *
         * The precision policy affects the computation in the same way as the forward pass.
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
            const std::vector<std::shared_ptr<Tensor<TOutput, MR>>>& output_state ) const {

            // Get precision policy from operation base class or override from properties
            ComputePrecision::Policy policy = config_.getPrecisionPolicy();

            // Check if properties override the precision policy
            /* Fixme: if ( properties.has( "precision_policy" ) ) {
                policy = static_cast<ComputePrecision::Policy>(properties.get( "precision_policy", static_cast<int>(policy) ));
            }*/

            // Extract tensors
            const TOutput* dY = output_gradient.data();
            TInput* dX1 = input1_gradient.data();
            TInput* dX2 = input2_gradient.data();
            int N = input1.size();

            cudaStream_t stream = this->getDeviceContext()->getStream();

            // For residual connection, the gradient just flows through to both inputs
            // FIXME: cuda_residual_backward(dX1, dX2, dY, N, stream);
            // Future implementation should respect the precision policy
        }

        /**
         * @brief Gets the name of this operation.
         *
         * @return std::string The name of the operation ("Cuda::ResidualOp").
         */
        std::string getName() const override {
            return "Cuda::ResidualOp";
        }

    private:
        ResidualConfig config_;  ///< Configuration for the residual operation
    };

    /**
     * @brief Class responsible for registering the CudaResidualOp operation.
     *
     * The CudaResidualOpRegistrar class registers the CudaResidualOp operation with the OperationRegistry.
     * It associates the operation name "Cuda::ResidualOp" with factory functions that create instances of CudaResidualOp.
     */
    export class CudaResidualOpRegistrar {
    public:
        /**
         * @brief Registers the CudaResidualOp operation with the OperationRegistry.
         *
         * This function registers the CudaResidualOp operation for the CUDA device type
         * with the OperationRegistry. It associates the operation name "Cuda::ResidualOp"
         * with factory functions that create instances of CudaResidualOp.
         */
        static void registerOperations() {
            const std::string opName = "Cuda::ResidualOp";

            OperationRegistry::instance().registerBinaryOperation<DeviceType::Cuda, float, float>(
                opName,
                []( std::shared_ptr<DeviceContext> context, const ConfigurationBase& config ) -> std::shared_ptr<BinaryOperation<DeviceType::Cuda, float, float>> {
                    const auto& residualConfig = static_cast<const ResidualConfig&>( config );
                    return context ? std::make_shared<CudaResidualOp<float>>( context, residualConfig )
                        : std::make_shared<CudaResidualOp<float>>( residualConfig );
                }
            );

            OperationRegistry::instance().registerBinaryOperation<DeviceType::Cuda, half, half>(
                opName,
                []( std::shared_ptr<DeviceContext> context, const ConfigurationBase& config ) -> std::shared_ptr<BinaryOperation<DeviceType::Cuda, half, half>> {
                    const auto& residualConfig = static_cast<const ResidualConfig&>(config);
                    return context ? std::make_shared<CudaResidualOp<half>>( context, residualConfig )
                        : std::make_shared<CudaResidualOp<half>>( residualConfig );
                }
            );

            // Register float-to-half mixed precision operation (when input is float but output is half)
            /*OperationRegistry::instance().registerBinaryOperation<DeviceType::Cuda, float, float, half>(
                opName,
                []( std::shared_ptr<DeviceContext> context ) -> std::shared_ptr<BinaryOperation<DeviceType::Cuda, float, float, half>> {
                    return context ? std::make_shared<CudaResidualOp<float, half>>( context, ComputePrecision::Policy::Performance )
                        : std::make_shared<CudaResidualOp<float, half>>( ComputePrecision::Policy::Performance );
                }
            );*/
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