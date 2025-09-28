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

import Dnn.Modules.Softmax;
import Dnn.Tensor;
import Dnn.TensorData;
import Dnn.TensorTraits;
import Dnn.ConfigurationBase;
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
    using namespace Mila::Dnn;

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
            static inline void forward_optimized( float* Y, const float* X, int N, int C, cudaStream_t stream ) {
                cuda_softmax_forward<float>( Y, X, N, C, stream );
            }

            static inline void forward_general(
                float* Y, const float* X,
                int outer_size, int dim_size, int inner_size,
                cudaStream_t stream )
            {
                cuda_softmax_forward_general<float>( Y, X, outer_size, dim_size, inner_size, stream );
            }
        };

        // Specialization for half
        template <>
        struct cuda_softmax_impl<half> {
            static inline void forward_optimized( half* Y, const half* X, int N, int C, cudaStream_t stream ) {
                cuda_softmax_forward<half>( Y, X, N, C, stream );
            }

            static inline void forward_general(
                half* Y, const half* X,
                int outer_size, int dim_size, int inner_size,
                cudaStream_t stream )
            {
                cuda_softmax_forward_general<half>( Y, X, outer_size, dim_size, inner_size, stream );
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
    export template<typename TInput = float, typename TOutput = TInput>
        requires ValidFloatTensorTypes<TInput, TOutput>
    class CudaSoftmaxOp : public UnaryOperation<DeviceType::Cuda, TInput, TOutput> {
    public:
        using MR = typename CudaDevice::MR;
        using UnaryOperationBase = UnaryOperation<DeviceType::Cuda, TInput, TOutput>;

        /**
         * @brief Constructs a new CUDA Softmax operation with the default device context.
         *
         * Initializes the operation with a CUDA device context (defaults to CUDA:0).
         */
        CudaSoftmaxOp( const SoftmaxConfig& config )
            : UnaryOperationBase( OperationType::SoftmaxOp ), config_( config ) {}

        /**
         * @brief Constructs a new CUDA Softmax operation with a specific device context.
         *
         * @param context The device context to use for this operation.
         * @throws std::runtime_error If the context is not for a CUDA device.
         */
        CudaSoftmaxOp( std::shared_ptr<DeviceContext> context, const SoftmaxConfig& config )
            : UnaryOperationBase( OperationType::SoftmaxOp, context ), config_( config ) {}

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
         * @param output_state Cache for intermediate results (not used in this operation).
         */
        void forward(
            const Tensor<TInput, MR>& input,
            const std::vector<std::shared_ptr<ITensorData>>& parameters,
            Tensor<TOutput, MR>& output,
            std::vector<std::shared_ptr<Tensor<TOutput, MR>>>& output_state ) const override {

            auto X = input.rawData();
            auto Y = output.rawData();

            int64_t axis = config_.getAxis();

            // Convert negative axis to positive for easier handling
            const int64_t ndim = input.shape().size();
            if ( axis < 0 ) {
                axis = ndim + axis;
            }

            // Validate the axis is within bounds
            if ( axis < 0 || axis >= ndim ) {
                throw std::runtime_error( "Softmax axis out of bounds" );
            }

            // Determine the shapes needed for the computation
            int64_t outer_size = 1;
            for ( int64_t i = 0; i < axis; ++i ) {
                outer_size *= input.shape()[ i ];
            }

            int64_t dim_size = input.shape()[ axis ];

            int64_t inner_size = 1;
            for ( int64_t i = axis + 1; i < ndim; ++i ) {
                inner_size *= input.shape()[ i ];
            }

            // If axis is the last dimension, we can use the optimized implementation
            if ( axis == ndim - 1 || inner_size == 1 ) {
                int N = outer_size;
                int C = dim_size;

                cudaStream_t stream = this->getDeviceContext()->getStream();

                // FIXME: Detail::cuda_softmax_impl<TInput>::forward_optimized( Y, X, N, C, stream );
            }
            else {
                // For other axes, use the generalized implementation
                cudaStream_t stream = this->getDeviceContext()->getStream();

                // FIXME:
                //Detail::cuda_softmax_impl<TInput>::forward_general(
                //    Y, X, outer_size, dim_size, inner_size, stream );
            }
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
         * @param output_state Cache tensors from forward pass.
         */
        void backward(
            const Tensor<TInput, MR>& input,
            const Tensor<TInput, MR>& output,
            const Tensor<TInput, MR>& output_gradient,
            const std::vector<std::shared_ptr<Tensor<TInput, MR>>>& parameters,
            std::vector<std::shared_ptr<Tensor<TInput, MR>>>& parameter_gradients,
            Tensor<TInput, MR>& input_gradient,
            const std::vector<std::shared_ptr<Tensor<TInput, MR>>>& output_state ) const {

            // Extract tensors
            const TInput* Y = output.data();
            const TInput* dY = output_gradient.data();
            TInput* dX = input_gradient.data();
            int N = input.size();

            // Get the axis parameter from properties
            int axis = config_.getAxis();

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

    private:
        SoftmaxConfig config_; ///< Configuration for the softmax operation, including axis and other parameters.
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

            OperationRegistry::instance().registerUnaryOperation<DeviceType::Cuda, float, float>(
                opName,
                []( std::shared_ptr<DeviceContext> context, const ConfigurationBase& config ) -> std::shared_ptr<UnaryOperation<DeviceType::Cuda, float, float>> {
                    const auto& softmaxConfig = static_cast<const SoftmaxConfig&>(config);
                    return context ? std::make_shared<CudaSoftmaxOp<float>>( context, softmaxConfig )
                        : std::make_shared<CudaSoftmaxOp<float>>( softmaxConfig );
                }
            );

           OperationRegistry::instance().registerUnaryOperation<DeviceType::Cuda, half, half>(
                opName,
                []( std::shared_ptr<DeviceContext> context, const ConfigurationBase& config ) -> std::shared_ptr<UnaryOperation<DeviceType::Cuda, half, half>> {
                    const auto& softmaxConfig = static_cast<const SoftmaxConfig&>(config);
                    return context ? std::make_shared<CudaSoftmaxOp<half>>( context, softmaxConfig )
                        : std::make_shared<CudaSoftmaxOp<half>>( softmaxConfig );
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