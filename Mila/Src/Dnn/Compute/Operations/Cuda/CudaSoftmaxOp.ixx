/**
 * @file CudaSoftmaxOp.ixx
 * @brief Implementation of the CUDA-based softmax operation for neural networks.
 */

module;
#include <vector>
#include <iostream>
#include <memory>
#include "Kernels/CudaOps.h"
#include <type_traits>

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
	 * @brief Namespace for CUDA softmax implementation details.
	 *
	 * This namespace contains the implementation details for the CUDA softmax operation,
	 * including specialized templates for different data types (float, half).
	 */
    namespace Detail
    {
        // Primary template - will cause a compile error if no specialization exists
        template <typename T>
        struct cuda_softmax_impl;

        // Specialization for float
        template <>
        struct cuda_softmax_impl<float> {
            static inline void forward( float* Y, const float* X, int N, int C, cudaStream_t stream ) {
                cuda_softmax_forward_fp32( Y, X, N, C, stream );
            }
        };

        // Specialization for half
        template <>
        struct cuda_softmax_impl<half> {
            static inline void forward( half* Y, const half* X, int N, int C, cudaStream_t stream ) {
                cuda_softmax_forward_fp16( Y, X, N, C, stream );
            }
        };

        // FUTURE: FP8 support
        /*
        template <>
        struct cuda_softmax_impl<__nv_fp8_e4m3> {
            static inline void forward(__nv_fp8_e4m3* Y, const __nv_fp8_e4m3* X, int N, int C, cudaStream_t stream) {
                cuda_softmax_forward_fp8(Y, X, N, C, stream);
            }
        };
        */
    }

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
     * @tparam TDataType The data type of the output tensor elements (defaults to the input type).
     */
    export template<typename TPrecision>
        requires (std::is_same_v<TPrecision, float> || std::is_same_v<TPrecision, half>)
    class CudaSoftmaxOp : public UnaryOperation<TPrecision> {
    public:
        using MR = typename CudaDevice::MR;
        /**
         * @brief Constructs a new CUDA Softmax operation with the default device context.
         *
         * Initializes the operation with a CUDA device context (defaults to CUDA:0).
         */
        CudaSoftmaxOp() : UnaryOperation<TPrecision>( OperationType::SoftmaxOp ) {}

        /**
         * @brief Constructs a new CUDA Softmax operation with a specific device context.
         *
         * @param context The device context to use for this operation.
         * @throws std::runtime_error If the context is not for a CUDA device.
         */
        CudaSoftmaxOp( std::shared_ptr<DeviceContext> context )
            : UnaryOperation<TPrecision>( OperationType::SoftmaxOp, context ) {
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
         * @param input Input tensor containing logits of shape [B, TDataType, V], where B is batch size,
         *              TDataType is sequence length, and V is vocabulary size.
         * @param parameters Additional parameters (not used in this operation).
         * @param properties Additional attributes for the operation.
         * @param output Output tensor of shape [B, TDataType, V] to store the resulting probability distribution.
         * @param output_cache Cache for intermediate results (not used in this operation).
         */
        void forward(
            const Tensor<TPrecision, MR>& input,
            const std::vector<std::shared_ptr<Tensor<TPrecision, MR>>>& parameters,
            const OperationAttributes& properties,
            Tensor<TPrecision, MR>& output,
            std::vector<std::shared_ptr<Tensor<TPrecision, MR>>>& output_cache ) const override {

            auto X = input.raw_data();
            auto Y = output.raw_data();
            int N = input.shape()[ 0 ];  // Batch size
            int C = input.shape()[ 2 ];  // Feature dimension size (vocabulary size)

            int axis = properties.axis;
            cudaStream_t stream = this->getDeviceContext()->getStream();

            Detail::cuda_softmax_impl<TPrecision>::forward( Y, X, N, C, stream );
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
            const Tensor<TPrecision, MR>& input,
            const Tensor<TPrecision, MR>& output,
            const Tensor<TPrecision, MR>& output_gradient,
            const std::vector<std::shared_ptr<Tensor<TPrecision, MR>>>& parameters,
            std::vector<std::shared_ptr<Tensor<TPrecision, MR>>>& parameter_gradients,
            Tensor<TPrecision, MR>& input_gradient,
            const OperationAttributes& properties,
            const std::vector<std::shared_ptr<Tensor<TPrecision, MR>>>& output_cache ) const {

            // Extract tensors
            const TPrecision* Y = output.data();
            const TPrecision* dY = output_gradient.data();
            TPrecision* dX = input_gradient.data();
            int N = input.size();

            // Get the axis parameter from properties
            int axis = properties.axis;

            // Get CUDA stream from device context
            cudaStream_t stream = this->getDeviceContext()->getStream();

            // Call CUDA kernel with stream
            //cuda_softmax_backward( dX, dY, Y, N, axis, stream );
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
                "Default",
                []( std::shared_ptr<DeviceContext> context ) -> std::shared_ptr<OperationBase<float, float, DeviceType::Cuda>> {
                    return context ? std::make_shared<CudaSoftmaxOp<float>>( context )
                        : std::make_shared<CudaSoftmaxOp<float>>();
                }
            );

            // Add additional precision variants if needed, for example:
            
            OperationRegistry::instance().registerOperation<half, half, DeviceType::Cuda>(
                opName,
                "half_precision",
                []( std::shared_ptr<DeviceContext> context ) -> std::shared_ptr<OperationBase<half, half, DeviceType::Cuda>> {
                    return context ? std::make_shared<CudaSoftmaxOp<half>>( context )
                        : std::make_shared<CudaSoftmaxOp<half>>();
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