/**
 * @file CudaFullyConnectedOp.ixx
 * @brief Implementation of the CUDA-based Fully Connected operation for neural networks.
 */

module;
#include <cublasLt.h>
#include <vector>
#include <memory>
#include <string>
#include <stdexcept>
#include <exception>
#include "Kernels/CudaOps.h"
#include <cuda_fp16.h>
#include <type_traits>

export module Compute.CudaMatMulOp;

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
import Compute.CublasLtMatMulBias;
import Utils.Logger;

using namespace Mila::Dnn;

namespace Mila::Dnn::Compute
{
	/**
	 * @brief Namespace for CUDA matrix multiplication implementation details.
	 *
	 * This namespace contains the implementation details for the CUDA matrix multiplication operation,
	 * including specialized templates for different data types (float, half).
	 */
    namespace Detail
    {
        // Primary template - will cause a compile error if no specialization exists
        template <typename T>
        struct cuda_matmul_impl;
        
        template <>
        struct cuda_matmul_impl<float> {
            static inline void forward( float* Y, const float* X,
                const float* weight, const float* bias,
                int B, int T, int C, int OC,
                cudaStream_t stream ) {
                cuda_matmul_forward_fp32( Y, X, weight, bias, B, T, C, OC, stream );
            }
        };

        template <>
        struct cuda_matmul_impl<half> {
            static inline void forward( half* Y, const half* X,
                const half* weight, const half* bias,
                int B, int T, int C, int OC,
                cudaStream_t stream ) {
                cuda_matmul_forward_fp16( Y, X, weight, bias, B, T, C, OC, stream );
            }
        };
        // Specialization for __nv_fp8_e4m3
    }

    /**
     * @brief CUDA implementation of the Fully Connected operation for neural networks.
     *
     * This class provides a CUDA-based implementation of the Fully Connected operation,
     * which performs a matrix multiplication between the input and a weight matrix,
     * optionally adds a bias, and produces an output. The operation is accelerated
     * on NVIDIA GPUs using CUDA for high-performance computation.
     *
     * The implementation delegates the actual matrix multiplication to optimized CUDA kernels
     * that may leverage libraries like cuBLAS or custom implementations for maximum performance.
     *
     * @tparam TPrecision The data type of the input tensor elements.
     * @tparam TDataType The data type of the output tensor elements (defaults to the input type).
     */
    export template<typename TPrecision>
        requires std::is_same_v<TPrecision, half> || std::is_same_v<TPrecision, float>
    class CudaFullyConnectedOp : public UnaryOperation<TPrecision> {
    public:
        using MR = typename CudaDevice::MR;
        /**
         * @brief Constructs a new CUDA Fully Connected operation with the default device context.
         *
         * Initializes the operation with a CUDA device context (defaults to CUDA:0).
         */
        CudaFullyConnectedOp() : UnaryOperation<TPrecision>( OperationType::FullyConnectedOp ) {}

        /**
         * @brief Constructs a new CUDA Fully Connected operation with a specific device context.
         *
         * @param context The device context to use for this operation.
         * @throws std::runtime_error If the context is not for a CUDA device.
         */
        CudaFullyConnectedOp( std::shared_ptr<DeviceContext> context )
            : UnaryOperation<TPrecision>( OperationType::FullyConnectedOp, context ) {
        }

        /**
         * @brief Performs the forward pass of the Fully Connected operation on CUDA.
         *
         * Computes the matrix multiplication between input and weights, adds bias if provided,
         * and stores the result in the output tensor. The computation is performed on the GPU
         * using CUDA kernels for optimal performance.
         *
         * @param input Input tensor of shape [B, TDataType, C] where B is batch size, TDataType is sequence length, and C is input feature dimension.
         * @param parameters Vector of parameter tensors [weight, bias] where weight is of shape [OC, C] and bias is of shape [OC].
         * @param properties Additional attributes for the operation.
         * @param output Output tensor of shape [B, TDataType, OC] where OC is output feature dimension.
         * @param output_state Cache for intermediate results (not used in this operation).
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
            TPrecision* bias = nullptr;

            if ( parameters.size() == 2 ) {
                bias = parameters[ 1 ]->raw_data();
            }

            int B = input.shape()[ 0 ];
            int T = input.shape()[ 1 ];
            int C = input.shape()[ 2 ];
            int OC = output.shape()[ 2 ];

            cudaStream_t stream = this->getDeviceContext()->getStream();

            // Try to use cuBLASLt MatMul first
            try {
                cublasLtHandle_t cublasLtHandle = this->getDeviceContext()->getCublasLtHandle();
                if ( cublasLtHandle ) {
                    cublaslt_matmul_forward<TPrecision>( Y, X, weight, bias, B, T, C, OC, stream, cublasLtHandle );
                    
                    return;
                }
            }
            catch ( const std::exception& e ) {
                Utils::Logger::warning( std::string( "cuBLASLt failed, falling back to custom CUDA kernel: " ) + e.what() );
            }

            // Fallback to custom cuda kernel...
            Detail::cuda_matmul_impl<TPrecision>::forward( Y, X, weight, bias, B, T, C, OC, stream );
        }

        /**
         * @brief Performs the backward pass of the Fully Connected operation.
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
         * @param output_state Cache tensors from forward pass.
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

            // Verify we're operating on CUDA memory
            if ( !this->getDeviceContext()->isDeviceType( DeviceType::Cuda ) ) {
                throw std::runtime_error( "CudaFullyConnectedOp::backward can only be executed on CUDA memory" );
            }

            // Extract dimensions
            int B = input.shape()[ 0 ];
            int T = input.shape()[ 1 ];
            int C = input.shape()[ 2 ];
            int OC = output.shape()[ 2 ];

            // Extract tensors
            auto X = input.data();
            auto dY = output_gradient.data();
            auto dX = input_gradient.data();
            auto W = parameters[ 0 ]->data();
            auto dW = parameter_gradients[ 0 ]->data();
            auto dB = parameter_gradients.size() > 1 ? parameter_gradients[ 1 ]->data() : nullptr;

            // Get CUDA stream from device context
            cudaStream_t stream = this->getDeviceContext()->getStream();

            // Call CUDA backward kernels with stream
            // FIXME: Implement these CUDA kernels
            // cuda_matmul_backward_input(dX, dY, W, B, TDataType, C, OC, stream);
            // cuda_matmul_backward_weight(dW, X, dY, B, TDataType, C, OC, stream);
            // if (dB != nullptr) {
            //     cuda_matmul_backward_bias(dB, dY, B, TDataType, OC, stream);
            // }
        }

        /**
         * @brief Gets the name of this operation.
         *
         * @return std::string The name of the operation ("Cuda::FullyConnectedOp").
         */
        std::string getName() const override {
            return "Cuda::FullyConnectedOp";
        }
    };

    /**
     * @brief Class responsible for registering the CudaFullyConnectedOp operation.
     *
     * The CudaFullyConnectedOpRegistrar class registers the CudaFullyConnectedOp operation with the OperationRegistry.
     * It associates the operation name "Cuda::FullyConnectedOp" with a factory function that creates instances of CudaFullyConnectedOp.
     */
    export class CudaFullyConnectedOpRegistrar {
    public:
        /**
         * @brief Registers the CudaFullyConnectedOp operation with the OperationRegistry.
         *
         * This function registers the CudaFullyConnectedOp operation for the CUDA device type
         * with the OperationRegistry. It associates the operation name "Cuda::FullyConnectedOp"
         * with a factory function that creates instances of CudaFullyConnectedOp.
         */
        static void registerOperations() {
            const std::string opName = "Cuda::FullyConnectedOp";

            OperationRegistry::instance().registerOperation<float, float, DeviceType::Cuda>(
                opName,
                "Default",
                []( std::shared_ptr<DeviceContext> context ) -> std::shared_ptr<OperationBase<float, float, DeviceType::Cuda>> {
                    return context ? std::make_shared<CudaFullyConnectedOp<float>>( context )
                        : std::make_shared<CudaFullyConnectedOp<float>>();
                }
            );

            /*OperationRegistry::instance().registerOperation<half, half, DeviceType::Cuda>(
                opName,
                "Default",
                []( std::shared_ptr<DeviceContext> context ) -> std::shared_ptr<OperationBase<half, half, DeviceType::Cuda>> {
                    return context ? std::make_shared<CudaFullyConnectedOp<half>>( context )
                        : std::make_shared<CudaFullyConnectedOp<half>>();
                }
            );*/
        }

        /**
         * @brief Self-registration mechanism that registers the operation during startup.
         *
         * This static member ensures the operation is registered when the program starts
         * without requiring explicit registration calls.
         */
        /*static inline bool isRegistered = []() {
            registerOperations();
            return true;
            }();*/
    };
}
