/**
 * @file CudaLinearOp.ixx
 * @brief Implementation of the CUDA-based Linear operation for neural networks.
 */

module;
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <vector>
#include <memory>
#include <string>
#include <stdexcept>
#include <exception>
#include "Kernels/CudaOps.h"
#include <type_traits>

export module Compute.CudaLinearOp;

import Dnn.Modules.Linear;
import Dnn.Tensor;
import Dnn.TensorData;
import Dnn.TensorTraits;
import Dnn.ConfigurationBase;
import Compute.OperationBase;
import Compute.UnaryOperation;
import Compute.Precision;
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

namespace Mila::Dnn::Compute
{
    using namespace Mila::Dnn;

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
                int outer_size, int C, int OC,
                cudaStream_t stream ) {
                cuda_matmul_forward_fp32( Y, X, weight, bias, outer_size, C, OC, stream );
            }
        };

        template <>
        struct cuda_matmul_impl<half> {
            static inline void forward( half* Y, const half* X,
                const half* weight, const half* bias,
                int outer_size, int C, int OC,
                cudaStream_t stream ) {
                
                cuda_matmul_forward_fp16( Y, X, weight, bias, outer_size, C, OC, stream );
            }
        };

        template <>
        struct cuda_matmul_impl<nv_bfloat16> {
            static inline void forward( nv_bfloat16* Y, const nv_bfloat16* X,
                const nv_bfloat16* weight, const nv_bfloat16* bias,
                int outer_size, int C, int OC,
                cudaStream_t stream ) {
                // Use the same kernel as FP16 for now - optimal implementation would use BF16-specific kernels
                // BF16 can typically use the same CUDA kernels as FP16 with minimal modifications
                
				// FIXME: Implemented BF16 with the dedicated kernel
                // cuda_matmul_forward_bf16( Y, X, weight, bias, outer_size, C, OC, stream );
            }
        };

        template <>
        struct cuda_matmul_impl<__nv_fp8_e4m3> {
            static inline void forward( __nv_fp8_e4m3* Y, const __nv_fp8_e4m3* X,
                const __nv_fp8_e4m3* weight, const __nv_fp8_e4m3* bias,
                int outer_size, int C, int OC,
                cudaStream_t stream ) {
                // Implemented FP8 E4M3 with the dedicated kernel
                
				// FIXME: Implemented FP8 E4M3 with the dedicated kernel
                // cuda_matmul_forward_fp8_e4m3( Y, X, weight, bias, outer_size, C, OC, stream );
            }
        };

        template <>
        struct cuda_matmul_impl<__nv_fp8_e5m2> {
            static inline void forward( __nv_fp8_e5m2* Y, const __nv_fp8_e5m2* X,
                const __nv_fp8_e5m2* weight, const __nv_fp8_e5m2* bias,
                int outer_size, int C, int OC,
                cudaStream_t stream ) {
                // Implemented FP8 E5M2 with the dedicated kernel
                
				// FIXME: Implemented FP8 E5M2 with the dedicated kernel
                // cuda_matmul_forward_fp8_e5m2( Y, X, weight, bias, outer_size, C, OC, stream );
            }
        };
    }

    /**
     * @brief CUDA implementation of the Linear operation for neural networks.
     *
     * This class provides a CUDA-based implementation of the Linear operation,
     * which performs a matrix multiplication between the input and a weight matrix,
     * optionally adds a bias, and produces an output. The operation is accelerated
     * on NVIDIA GPUs using CUDA for high-performance computation.
     *
     * The implementation delegates the actual matrix multiplication to optimized CUDA kernels
     * that may leverage libraries like cuBLAS or custom implementations for maximum performance.
     *
     * @tparam TDataType The data type used for tensor elements (float, half, etc.)
     */
    export template<typename TDataType = float>
        requires ValidFloatTensorType<TDataType>
    class CudaLinearOp : public UnaryOperation<DeviceType::Cuda, TDataType, TDataType> {
    public:
        using MR = typename CudaDevice::MR;
        using UnaryOperationBase = UnaryOperation<DeviceType::Cuda, TDataType, TDataType>;

        /**
         * @brief Constructs a new CUDA Linear operation with the default device context.
         *
         * Initializes the operation with a CUDA device context (defaults to CUDA:0).
         *
         * @param config Configuration parameters for the linear operation.
         */
        CudaLinearOp( const LinearConfig& config )
            : UnaryOperationBase( OperationType::LinearOp ), config_( config ) {}

        /**
         * @brief Constructs a new CUDA Linear operation with a specific device context.
         *
         * @param context The device context to use for this operation.
         * @param config Configuration parameters for the linear operation.
         * @throws std::runtime_error If the context is not for a CUDA device.
         */
        CudaLinearOp( std::shared_ptr<DeviceContext> context, const LinearConfig& config )
            : UnaryOperationBase( OperationType::LinearOp, context ), config_( config ) {}

        /**
         * @brief Performs the forward pass of the Linear operation on CUDA.
         *
         * Computes the matrix multiplication between input and weights, adds bias if provided,
         * and stores the result in the output tensor. The computation is performed on the GPU
         * using CUDA kernels for optimal performance.
         *
         * @param input Input tensor of shape [B, S, C] where B is batch size, S is sequence length, and C is input feature dimension.
         * @param parameters Vector of parameter tensors [weight, bias] where weight is of shape [OC, C] and bias is of shape [OC].
         * @param properties Additional attributes for the operation.
         * @param output Output tensor of shape [B, S, OC] where OC is output feature dimension.
         * @param output_state Cache for intermediate results (not used in this operation).
         */
        void forward(
            const Tensor<TDataType, MR>& input,
            const std::vector<std::shared_ptr<ITensorData>>& parameters,
            Tensor<TDataType, MR>& output,
            std::vector<std::shared_ptr<Tensor<TDataType, MR>>>& output_state ) const override {

            auto outer_dims = input.rank() - 1;

            if ( outer_dims <= 0 ) {
                throw std::runtime_error( "LinearOp requires input tensor with at least 2 dimensions" );
            }

            auto X = input.rawData();
            auto Y = output.rawData();

            // Safely get weight tensor and its raw data pointer
            const auto& weight_tensor = this->template getTypedParameter<TDataType>( parameters[ 0 ] );
            const TDataType* weight_data = weight_tensor.data();

            // Initialize bias pointer to nullptr
            const TDataType* bias_data = nullptr;

            // Check if bias is provided and extract its raw data pointer
            if ( parameters.size() > 1 ) {
                if constexpr ( std::is_same_v<TDataType, __nv_fp8_e4m3> ) {
                    // Special case for FP8: bias is in BF16 format
                    const auto& bias_tensor = this->template getTypedParameter<nv_bfloat16>( parameters[ 1 ] );
                    // FIXME: bias_data = bias_tensor.data();
                }
                else {
                    // Normal case: bias has same type as input/weight
                    const auto& bias_tensor = this->template getTypedParameter<TDataType>( parameters[ 1 ] );
                    bias_data = bias_tensor.data();
                }
            }

            int C = input.shape().back(); // Input channels/features
            int OC = output.shape().back(); // Output channels/features

            // Get the flattened input tensor outer size
            int outer_size = 1;
            for ( int i = 0; i < outer_dims; i++ ) {
                outer_size *= input.shape()[ i ];
            }

            cudaStream_t stream = this->getDeviceContext()->getStream();
            auto precision_policy = config_.getPrecisionPolicy();

            // Try to use cuBLASLt MatMul first
            try {
                cublasLtHandle_t cublasLtHandle = this->getDeviceContext()->getCublasLtHandle();
                if ( cublasLtHandle ) {
                    // FIXME
                    /*cublaslt_matmul_forward<TDataType>(
                        Y, X,
                        weight_data, bias_data,
                        outer_size, C, OC,
                        stream, cublasLtHandle,
                        precision_policy );*/

                    return;
                }
            }
            catch ( const std::exception& e ) {
                Utils::Logger::warning( std::string( "cuBLASLtMatMul failed, falling back to custom CUDA kernel: " ) + e.what() );
            }

            // FIXME:
            // Fallback to custom cuda kernel...
            //Detail::cuda_matmul_impl<TDataType>::forward(
            //    Y, X, weight_data, bias_data,
            //    outer_size, C, OC, stream
            //);
        }

        /**
         * @brief Performs the backward pass of the Linear operation.
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
            const Tensor<TDataType, MR>& input,
            const Tensor<TDataType, MR>& output,
            const Tensor<TDataType, MR>& output_gradient,
            const std::vector<std::shared_ptr<Tensor<TDataType, MR>>>& parameters,
            std::vector<std::shared_ptr<Tensor<TDataType, MR>>>& parameter_gradients,
            Tensor<TDataType, MR>& input_gradient,
            const std::vector<std::shared_ptr<Tensor<TDataType, MR>>>& output_state ) const {

            // Verify we're operating on CUDA memory
            if ( !this->getDeviceContext()->isDeviceType( DeviceType::Cuda ) ) {
                throw std::runtime_error( "CudaLinearOp::backward can only be executed on CUDA memory" );
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
            // cuda_matmul_backward_input(dX, dY, W, B, T, C, OC, stream);
            // cuda_matmul_backward_weight(dW, X, dY, B, T, C, OC, stream);
            // if (dB != nullptr) {
            //     cuda_matmul_backward_bias(dB, dY, B, T, OC, stream);
            // }
        }

        /**
         * @brief Gets the name of this operation.
         *
         * @return std::string The name of the operation ("Cuda::LinearOp").
         */
        std::string getName() const override {
            return "Cuda::LinearOp";
        }

    private:
        LinearConfig config_; ///< Configuration for the linear operation.
    };

    /**
     * @brief Class responsible for registering the CudaLinearOp operation.
     *
     * The CudaLinearOpRegistrar class registers the CudaLinearOp operation with the OperationRegistry.
     * It associates the operation name "Cuda::LinearOp" with a factory function that creates instances of CudaLinearOp.
     */
    export class CudaLinearOpRegistrar {
    public:
        /**
         * @brief Registers the CudaLinearOp operation with the OperationRegistry.
         *
         * This function registers the CudaLinearOp operation for the CUDA device type
         * with the OperationRegistry. It associates the operation name "Cuda::LinearOp"
         * with a factory function that creates instances of CudaLinearOp.
         */
        static void registerOperations() {
            const std::string opName = "Cuda::LinearOp";

            // Helper template function to register Linear ops for a specific data type
            auto registerForType = [ & ]<typename TDataType>() {
                OperationRegistry::instance().registerUnaryOperation<DeviceType::Cuda, TDataType, TDataType>(
                    opName,
                    []( std::shared_ptr<DeviceContext> context, const ConfigurationBase& config ) -> std::shared_ptr<UnaryOperation<DeviceType::Cuda, TDataType>> {
                        const auto& linearConfig = static_cast<const LinearConfig&>(config);
                        return context ? std::make_shared<CudaLinearOp<TDataType>>( context, linearConfig )
                            : std::make_shared<CudaLinearOp<TDataType>>( linearConfig );
                    }
                );
            };

            // Register for all supported floating point types
            registerForType.template operator()<float>();
            registerForType.template operator()<half>();
            registerForType.template operator()<nv_bfloat16>();
            registerForType.template operator()<__nv_fp8_e4m3>();
            registerForType.template operator()<__nv_fp8_e5m2>();

            /*OperationRegistry::instance().registerUnaryOperation<DeviceType::Cuda, float, float>(
                opName,
                []( std::shared_ptr<DeviceContext> context, const ConfigurationBase& config ) -> std::shared_ptr<UnaryOperation<DeviceType::Cuda, float>> {
                    const auto& linearConfig = static_cast<const LinearConfig&>(config);
                    return context ? std::make_shared<CudaLinearOp<float>>( context, linearConfig )
                        : std::make_shared<CudaLinearOp<float>>( linearConfig );
                }
            );

            OperationRegistry::instance().registerUnaryOperation<DeviceType::Cuda, half, half>(
                opName,
                []( std::shared_ptr<DeviceContext> context, const ConfigurationBase& config ) -> std::shared_ptr<UnaryOperation<DeviceType::Cuda, half>> {
                    const auto& linearConfig = static_cast<const LinearConfig&>(config);
                    return context ? std::make_shared<CudaLinearOp<half>>( context, linearConfig )
                        : std::make_shared<CudaLinearOp<half>>( linearConfig );
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