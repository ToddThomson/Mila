/**
 * @file MatMulBiasActivation.ixx
 * @brief Implementation of fused matrix multiplication, bias addition, and activation operations.
 */

module;
#include <memory>
#include <vector>
#include <stdexcept>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublasLt.h>

export module Compute.CudaMatMulBiasGeluOp;

import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTraits;
import Compute.OperationBase;
import Compute.UnaryOperation;
import Compute.OperationAttributes;
import Compute.OperationRegistry;
import Compute.DeviceType;
import Compute.DeviceContext;
import Compute.OperationType;
import Compute.CudaDevice;
import Compute.CudaDeviceMemoryResource;

// Forward declaration of CUDA kernel function from FusedMatMulBiasGelu.cu
//extern "C" {
//    template <typename TDataType>
//    void launchFusedMatmulBiasGelu(
//        const TDataType* A, const TDataType* B, const TDataType* bias, TDataType* C,
//        size_t M, size_t K, size_t N,
//        cublasLtHandle_t ltHandle, cudaStream_t stream );
//}

namespace Mila::Dnn::Compute
{
    /**
     * @brief CUDA implementation of the fused MatMul-Bias-GELU operation.
     *
     * This class provides a CUDA-based implementation of a fused operation that combines
     * matrix multiplication, bias addition, and GELU activation in a single operation.
     * Fusing these operations improves performance by reducing memory traffic and
     * kernel launch overhead.
     *
     * The implementation is optimized for NVIDIA GPUs using cuBLASLt for high-performance
     * computation of the fused operation.
     *
     * @tparam TPrecision The data type of the input tensor elements.
     * @tparam TDataType The data type for computation and output (defaults to the input type).
     */
    export template<typename TInput = float, typename TOutput = TInput>
		requires ValidFloatTensorTypes<TInput, TOutput>
    class CudaMatMulBiasGeluOp : public UnaryOperation<DeviceType::Cuda, TInput, TOutput> {
    public:
		using MR = typename CudaDevice::MR;
		using UnaryOperationBase = UnaryOperation<DeviceType::Cuda, TInput, TOutput>;

        /**
         * @brief Constructs a new CUDA MatMul-Bias-GELU fused operation with the default device context.
         *
         * Initializes the operation with a CUDA device context (defaults to CUDA:0).
         */
        CudaMatMulBiasGeluOp() : UnaryOperationBase( OperationType::FusedOp ) {}

        /**
         * @brief Constructs a new CUDA MatMul-Bias-GELU fused operation with a specific device context.
         *
         * @param context The device context to use for this operation.
         * @throws std::runtime_error If the context is not for a CUDA device.
         */
        CudaMatMulBiasGeluOp( std::shared_ptr<DeviceContext> context )
            : UnaryOperationBase( OperationType::FusedOp, context ) {
        }

        /**
         * @brief Performs the forward pass of the fused MatMul-Bias-GELU operation on CUDA.
         *
         * This method efficiently computes a matrix multiplication followed by a bias addition
         * and GELU activation in a single fused operation. The implementation uses cuBLASLt
         * for optimal performance on NVIDIA GPUs.
         *
         * @param input Input tensor of shape [B, S, K], where B is batch size, S is sequence length,
         *              and K is the input dimension.
         * @param parameters Vector of parameter tensors where:
         *                   - parameters[0]: Weights tensor of shape [K, N]
         *                   - parameters[1]: Bias tensor of shape [N]
         * @param properties Additional attributes for the operation.
         * @param output Output tensor of shape [B, S, N] to store the result.
         * @param output_state Cache for intermediate results (not used in this operation).
         */
        void forward(
            const Tensor<TInput, MR>& input,
            const std::vector<std::shared_ptr<ITensor>>& parameters,
            Tensor<TOutput, MR>& output,
            std::vector<std::shared_ptr<Tensor<TOutput, MR>>>& output_state ) const override {

            if ( parameters.size() < 2 ) {
                throw std::runtime_error( "CudaMatMulBiasGeluOp requires at least 2 parameters (weights and bias)" );
            }

            const auto& weights = *parameters[ 0 ];
            const auto& bias = *parameters[ 1 ];

            auto device_context = this->getDeviceContext();
            
            cudaStream_t stream = device_context->getStream();
            cublasLtHandle_t cuda_handle = device_context->getCublasLtHandle();

            const auto* input_data = input.data();
            const auto* weights_data = weights.data();
            const auto* bias_data = bias.data();
            auto* output_data = output.data();

            auto input_shape = input.shape();
            auto weights_shape = weights.shape();

            // Reshape for matmul: [batch_size * seq_len, input_dim]
            size_t M = input_shape[ 0 ] * input_shape[ 1 ];
            size_t K = input_shape[ 2 ];
            size_t N = weights_shape[ 1 ];

            launchFusedMatmulBiasGelu(
                input_data, weights_data, bias_data, output_data,
                M, K, N, cuda_handle, stream );
        }

        /**
         * @brief Performs the backward pass of the fused MatMul-Bias-GELU operation.
         *
         * Computes gradients with respect to inputs, weights, and biases.
         *
         * @param input Input tensor from the forward pass.
         * @param output Output tensor from the forward pass.
         * @param output_gradient Gradient of the loss with respect to the output.
         * @param parameters Parameters tensor from forward pass [weights, bias].
         * @param parameter_gradients Gradients for parameters [d_weights, d_bias].
         * @param input_gradient Gradient of the loss with respect to the input.
         * @param properties Additional attributes for the operation.
         * @param output_state Cache tensors from forward pass.
         */
        void backward(
            const Tensor<TInput, MR>& input,
            const Tensor<TOutput, MR>& output,
            const Tensor<TOutput, MR>& output_gradient,
            const std::vector<std::shared_ptr<ITensor>>& parameters,
            std::vector<std::shared_ptr<Tensor<TOutput, MR>>>& parameter_gradients,
            Tensor<TInput, MR>& input_gradient,
            const std::vector<std::shared_ptr<Tensor<TOutput, MR>>>& output_state ) const {

            if ( parameters.size() < 2 || parameter_gradients.size() < 2 ) {
                throw std::runtime_error( "CudaMatMulBiasGeluOp backward requires weights, bias and their gradients" );
            }

            auto device_context = this->getDeviceContext();
            
            cudaStream_t stream = device_context->getStream();
            cublasHandle_t cublas_handle = device_context->getCublasHandle();

            // Extract tensors
            const auto* X = input.data();
            const auto* dY = output_gradient.data();
            const auto* W = parameters[ 0 ]->data();

            auto* dX = input_gradient.data();
            auto* dW = parameter_gradients[ 0 ]->data();
            auto* dB = parameter_gradients[ 1 ]->data();

            // Extract dimensions
            auto input_shape = input.shape();
            auto weights_shape = parameters[ 0 ]->shape();

            size_t B = input_shape[ 0 ];
            size_t S = input_shape[ 1 ];
            size_t K = input_shape[ 2 ];
            size_t N = weights_shape[ 1 ];
            size_t M = B * S;

            // FIXME: Implement backward pass for fused operation
            // This would typically involve custom CUDA kernels to handle the 
            // backward pass of the fused operation efficiently
            // 
            // For now, we would need to implement something like:
            // launchFusedMatmulBiasGeluBackward(
            //     X, W, dY, dX, dW, dB, 
            //     M, K, N, cublas_handle, stream);
        }

        /**
         * @brief Gets the name of this operation.
         *
         * @return std::string The name of the operation ("Cuda::MatMulBiasGeluOp").
         */
        std::string getName() const override {
            return "Cuda::MatMulBiasGeluOp";
        }

        /**
         * @brief Gets the class name of this operation.
         *
         * @return const std::string& The class name of the operation.
         */
        static const std::string& className() {
            static std::string name = "Cuda::MatMulBiasGeluOp";
            return name;
        }
    };

    /**
     * @brief Class responsible for registering the CudaMatMulBiasGeluOp operation.
     *
     * The CudaMatMulBiasGeluOpRegistrar class registers the CudaMatMulBiasGeluOp operation
     * with the OperationRegistry. It associates the operation name "Cuda::MatMulBiasGeluOp"
     * with a factory function that creates instances of CudaMatMulBiasGeluOp.
     */
    export class CudaMatMulBiasGeluOpRegistrar {
    public:
        /**
         * @brief Registers the CudaMatMulBiasGeluOp operation with the OperationRegistry.
         *
         * This function registers the CudaMatMulBiasGeluOp operation for the CUDA device type
         * with the OperationRegistry. It associates the operation name "Cuda::MatMulBiasGeluOp"
         * with a factory function that creates instances of CudaMatMulBiasGeluOp.
         * It also registers patterns of modules that can be fused using this operation.
         */
        static void registerOperations() {
            auto& registry = OperationRegistry::instance();
            const std::string opName = "Cuda::MatMulBiasGeluOp";

            // Register the operation itself
            //registry.registerCudaOperation<float, float>(
            //    opName, "Default",
            //    []() { return std::make_shared<CudaMatMulBiasGeluOp<float, float>>(); } );

            //registry.registerCudaOperation<half, half, DeviceType::Cuda>(
            //    opName, "Default",
            //    []() { return std::make_shared<CudaMatMulBiasGeluOp<half, half>>(); } );

            //// Register the sequence that can be fused
            //registry.registerFusedOperation<float>(
            //    { OperationType::FullyConnectedOp, OperationType::GeluOp },
            //    opName );

            //registry.registerFusedOperation<half>(
            //    { OperationType::FullyConnectedOp, OperationType::GeluOp },
            //    opName );
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