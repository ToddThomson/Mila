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

import Dnn.Modules.Gelu;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.ConfigurationBase;
import Compute.Precision;
import Compute.OperationBase;
import Compute.UnaryOperation;
import Compute.OperationRegistry;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.CudaExecutionContext;
import Compute.CudaDeviceResources;
import Compute.OperationType;
import Compute.OperationAttributes;
import Compute.MemoryResource;
import Compute.CudaDeviceMemoryResource;
import Compute.CudaTensorDataType;
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
        template <typename TNative>
            requires std::is_same_v<TNative, float> || std::is_same_v<TNative, half>
        struct cuda_gelu_impl;

        template <>
        struct cuda_gelu_impl<float> {
            ForwardFp32Func forward_func;
            BackwardFp32Func backward_func;

            cuda_gelu_impl( const GeluConfig& config ) {
                switch (config.getApproximationMethod())
                {
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
            cuda_gelu_impl( const GeluConfig& /*config*/ ) { /* Nothing to select for half yet */ }

            inline void forward( half* Y, const half* X, int N, cudaStream_t stream ) const {
                cuda_gelu_forward_fp16( Y, X, N, stream );
            }

            inline void backward( half* dX, const half* X, const half* dY, int N, cudaStream_t stream ) const {
                // FIXME: implement or call half backward kernel when available
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
     * @tparam TPrecision Abstract TensorDataType for the operation (FP32, FP16, BF16, etc.).
     */
    export template<TensorDataType TPrecision>
        requires ValidFloatTensorDataType<TPrecision>
    class CudaGeluOp : public UnaryOperation<DeviceType::Cuda, TPrecision>
    {
    public:
        using MR = CudaDeviceMemoryResource;
        using UnaryOperationBase = UnaryOperation<DeviceType::Cuda, TPrecision>;
        using TensorType = Tensor<TPrecision, MR>;
        using Parameters = std::vector<std::shared_ptr<TensorType>>;
        using OutputState = std::vector<std::shared_ptr<TensorType>>;
        using NativeType = typename Mila::Dnn::Compute::Cuda::TensorDataTypeMap<TPrecision>::native_type;
        using CudaExecutionContext = ExecutionContext<DeviceType::Cuda>;

        // TJT: Design Note:
		// Operations now use a 2 phase initialization:
		// 1) Constructor with config only - no context binding
		// 2) Optional context binding via bind_context() or with_context()
		// This allows operations to be created in a context-agnostic way
		/// and then bound to a specific execution context later (for device pinning).
		// This is especially useful for registries and factories.
		// The context can be set before execution.
		// The base class OperationBase now has a set_context() method for this purpose.
		// The constructor below does not take a context.
		/// The context can be set later via bind_context() or with_context().
		/// @param config Configuration for the GELU operation.
        /// 
           
        CudaGeluOp( const GeluConfig& config, std::shared_ptr<CudaExecutionContext> context = nullptr )
            : config_( config ), context_( context ), impl_( config ) {
            config_.validate();
        }

        /**
        * @brief Performs the forward pass of the GELU activation function on CUDA.
        *
        * Computes the GELU transformation of the input elements:
        * GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        *
        * The precision policy affects how the computation is performed:
        * - Performance: May use faster but less precise algorithms
        * - Accuracy: Will use the most accurate algorithm available
        * - Auto: Will select an appropriate balance based on the hardware
        * - Native: Use native precision of the tensor type
        *
        * @param input Input tensor containing the values to transform.
        * @param parameters Additional parameters (not used in this operation).
        * @param output Output tensor to store the transformed values.
        * @param output_state Cache for intermediate results (not used in this operation).
        */
        void forward(
            const ITensor& input,
            const Parameters& parameters,
            ITensor& output,
            OutputState& output_state ) const override {

            cudaStream_t stream;
            std::shared_ptr<CudaDeviceResources> resources;

            if (context_)
            {
                // Bound: dedicated non-blocking stream
                stream = this->context_->getStream();
                resources = this->context_->getResources();
            }
            else
            {
				// Unbound: per-thread stream from device
                auto* cuda_device = static_cast<CudaDevice*>(input.getDevice().get());
                
                //stream = cuda_device->getDefaultStream();
                resources = cuda_device->getResources();
            }

			// Set cuBLAS and cuDNN streams if needed

            //auto cublas = resources->getCublasHandle();
            //auto cudnn = resources->getCudnnHandle();
            
            //cublasSetStream( cublas, stream );
            //cudnnSetStream( cudnn, stream );

            ComputePrecision::Policy policy = config_.getPrecisionPolicy();

            // Get raw device pointers (native type)
            auto X = static_cast<const NativeType*>(input.rawData());
            auto Y = static_cast<NativeType*>(output.rawData());
            int N = static_cast<int>(input.size());
            
            //cudaStream_t stream = this->getExecutionContext()->getStream();

            // Dispatch to implementation based on native type
            impl_.forward( reinterpret_cast<NativeType*>(Y), reinterpret_cast<const NativeType*>(X), N, stream );
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
        * @param output_state Cache tensors from forward pass.
        */
        void backward(
            const ITensor& output_gradient,
            const ITensor& input,
            const Parameters& parameters,
            const OutputState& output_state,
            ITensor& input_gradient,
            Parameters& parameter_gradients ) const {

            // Get precision policy from operation base class or override from properties
            //ComputePrecision::Policy policy = this->getPrecisionPolicy();

            // Get tensor data pointers (native)
            const NativeType* X = static_cast<const NativeType*>(input.rawData());
            const NativeType* dY = static_cast<const NativeType*>(output_gradient.rawData());
            NativeType* dX = static_cast<NativeType*>(input_gradient.rawData());
            
            int N = static_cast<int>(input.size());

            // Get CUDA stream from execution context
            //cudaStream_t stream = this->getExecutionContext()->getStream();

            // Call CUDA kernel with stream (not implemented yet for all native types)
            // impl_.backward( dX, X, dY, N, stream );
        }

        OperationType getOperationType() const override {
            return OperationType::GeluOp;
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
		std::shared_ptr<CudaExecutionContext> context_; ///< Optional execution context for CUDA resources.
        Detail::cuda_gelu_impl<NativeType> impl_; ///< Implementation details for the GELU operation.
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

            // Register FP32 version
            OperationRegistry::instance().registerUnaryOperation<DeviceType::Cuda, TensorDataType::FP32>(
                opName,
                [](  const ConfigurationBase& config, std::shared_ptr<ExecutionContext<DeviceType::Cuda>> context ) -> std::shared_ptr<UnaryOperation<DeviceType::Cuda, TensorDataType::FP32>> {
                    const auto& geluConfig = static_cast<const GeluConfig&>(config);
                    return std::make_shared<CudaGeluOp<TensorDataType::FP32>>( geluConfig, context );
                }
            );

            // Register FP16 version (if/when supported)
            OperationRegistry::instance().registerUnaryOperation<DeviceType::Cuda, TensorDataType::FP16>(
                opName,
                [](  const ConfigurationBase& config, std::shared_ptr<ExecutionContext<DeviceType::Cuda>> context ) -> std::shared_ptr<UnaryOperation<DeviceType::Cuda, TensorDataType::FP16>> {
                    const auto& geluConfig = static_cast<const GeluConfig&>(config);
                    return std::make_shared<CudaGeluOp<TensorDataType::FP16>>( geluConfig, context );
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