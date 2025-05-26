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
import Dnn.Modules.Gelu;
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
        // Function pointer types for dispatch
        using ForwardFp32Func = void (*)(float*, const float*, int, cudaStream_t);
        using BackwardFp32Func = void (*)(float*, const float*, const float*, int, cudaStream_t);
        using ForwardFp16Func = void (*)(half*, const half*, int, cudaStream_t);
        using BackwardFp16Func = void (*)(half*, const half*, const half*, int, cudaStream_t);

        //// Forward and Backward declarations of FP32 implementations
        ////void gelu_exact_forward_fp32( float* Y, const float* X, int N, cudaStream_t stream );
        //void gelu_tanh_forward_fp32( float* Y, const float* X, int N, cudaStream_t stream );
        ////void gelu_sigmoid_forward_fp32( float* Y, const float* X, int N, cudaStream_t stream );
        //
        ////void gelu_exact_backward_fp32( float* dX, const float* X, const float* dY, int N, cudaStream_t stream );
        //void gelu_tanh_backward_fp32( float* dX, const float* X, const float* dY, int N, cudaStream_t stream );
        ////void gelu_sigmoid_backward_fp32( float* dX, const float* X, const float* dY, int N, cudaStream_t stream );

        // Primary template - will cause a compile error if no specialization exists
        template <typename TDataType>
			requires ValidFloatTensorType<TDataType>
        struct cuda_gelu_impl;

        template <>
        struct cuda_gelu_impl<float> {
            ForwardFp32Func forward_func;
            BackwardFp32Func backward_func;

            cuda_gelu_impl( const GeluConfig& config ) {
                switch ( config.getApproximationMethod() ) {
                    /*case GeluConfig::ApproximationMethod::Exact:
                        forward_func = &gelu_exact_forward_fp32;
                        backward_func = &gelu_exact_backward_fp32;
                        break;
                    case GeluConfig::ApproximationMethod::Sigmoid:
                        forward_func = &gelu_sigmoid_forward_fp32;
                        backward_func = &gelu_sigmoid_backward_fp32;
                        break;*/
                    case GeluConfig::ApproximationMethod::Tanh:
                    default:
                        forward_func = &cuda_gelu_forward_fp32;
                        backward_func = &cuda_gelu_backward_fp32;
                        break;
                }
            }

            inline void forward( float* Y, const float* X, int N, cudaStream_t stream ) const {
                forward_func( Y, X, N, stream );
            }

            inline void backward( float* dX, const float* X, const float* dY, int N, cudaStream_t stream ) const {
                backward_func( dX, X, dY, N, stream );
            }
        };

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
     * @tparam TDataType The data type of the output tensor elements.
     * @tparam TInput The data type of the input tensor elements (defaults to TDataType).
     */
    export template<typename TDataType>
		requires ValidFloatTensorType<TDataType>
    class CudaGeluOp : public UnaryOperation<DeviceType::Cuda, TDataType, TDataType> {
    public:
        using MR = typename CudaDevice::MR;
        using UnaryOperationBase = UnaryOperation<DeviceType::Cuda, TDataType, TDataType>;

        CudaGeluOp( const GeluConfig& config )
            : UnaryOperationBase( OperationType::GeluOp ), config_( config ), impl_( config ) {
            config_.validate();
        }

        CudaGeluOp( std::shared_ptr<DeviceContext> context, const GeluConfig& config )
            : UnaryOperationBase( OperationType::GeluOp, context), config_( config ), impl_( config ) {
            config_.validate();
        }

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
            const Tensor<TDataType, MR>& input,
            const std::vector<std::shared_ptr<Tensor<TDataType, MR>>>& parameters,
            const OperationAttributes& properties,
            Tensor<TDataType, MR>& output,
            std::vector<std::shared_ptr<Tensor<TDataType, MR>>>& output_state ) const override {

            ComputePrecision::Policy policy = config_.getPrecision();

            auto X = input.raw_data();
            auto Y = output.raw_data();
            int N = input.size();
            cudaStream_t stream = this->getDeviceContext()->getStream();

            // For now, we use the same implementation regardless of policy
            // In a more advanced implementation, different kernels could be selected based on the policy
            if constexpr ( std::is_same_v<TDataType, TDataType> ) {
                impl_.forward( Y, X, N, stream );
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
            const Tensor<TDataType, MR>& input,
            const Tensor<TDataType, MR>& output,
            const Tensor<TDataType, MR>& output_gradient,
            const std::vector<std::shared_ptr<Tensor<TDataType, MR>>>& parameters,
            std::vector<std::shared_ptr<Tensor<TDataType, MR>>>& parameter_gradients,
            Tensor<TDataType, MR>& input_gradient,
            const OperationAttributes& properties,
            const std::vector<std::shared_ptr<Tensor<TDataType, MR>>>& output_state ) const {

            // Get precision policy from operation base class or override from properties
            ComputePrecision::Policy policy = this->getPrecisionPolicy();

            // Check if properties override the precision policy
            if ( properties.has( "precision_policy" ) ) {
                policy = static_cast<ComputePrecision::Policy>(properties.get( "precision_policy", static_cast<int>(policy) ));
            }

            // Get tensor data pointers
            const TDataType* X = input.raw_data();
            const TDataType* dY = output_gradient.raw_data();
            TDataType* dX = input_gradient.raw_data();
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

        const GeluConfig& getConfig() const {
            return config_;
		}
        
    private:
            GeluConfig config_; ///< Configuration for the GELU operation.
			Detail::cuda_gelu_impl<TDataType> impl_; ///< Implementation details for the GELU operation.
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
                []( std::shared_ptr<DeviceContext> context, const ComponentConfig& config ) -> std::shared_ptr<UnaryOperation<DeviceType::Cuda, float, float>> {
                    const auto& geluConfig = static_cast<const GeluConfig&>(config);
                    return context ? std::make_shared<CudaGeluOp<float>>( context, geluConfig )
                        : std::make_shared<CudaGeluOp<float>>( geluConfig );
                }
            );

			// FIXME: Uncomment when half precision is supported
            /*OperationRegistry::instance().registerUnaryOperation<DeviceType::Cuda, half, half>(
                opName,
                []( std::shared_ptr<DeviceContext> context, const ComponentConfig& config ) -> std::shared_ptr<UnaryOperation<DeviceType::Cuda, half, half>> {
                    const auto& geluConfig = static_cast<const GeluConfig&>( config );
                    return context ? std::make_shared<CudaGeluOp<half>>( context, geluConfig )
                        : std::make_shared<CudaGeluOp<half>>( geluConfig );
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