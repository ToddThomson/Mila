/**
 * @file CudaGeluOp.ixx
 * @brief Implementation of the CUDA-based GELU activation function for neural networks.
 */

module;
#include <cuda_runtime.h>
//#include <vector>
#include <memory>
//#include <iostream>
//#include <cuda_fp16.h>
#include <stdexcept>
//#include <type_traits>
#include <string>
#include <cuda_runtime_api.h>  // cudaStream_t
//#include "Kernels/Gelu.cuh"

export module Compute.CudaGeluOp;
import :Dispatch;

import Dnn.Components.GeluConfig;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.ComponentConfig;
import Compute.OperationBase;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.IExecutionContext;
import Compute.ExecutionContextTemplate;
import Compute.OperationType;
import Compute.MemoryResource;
import Compute.CudaDeviceMemoryResource;
import Compute.CudaTensorDataType;
import Compute.CudaDevice;

namespace Mila::Dnn::Compute::Cuda::Gelu
{
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
    class CudaGeluOp : public Operation<DeviceType::Cuda, TPrecision>
    {
    public:
        using MR = CudaDeviceMemoryResource;
        using TensorType = Tensor<TPrecision, MR>;
        using NativeType = typename Mila::Dnn::Compute::Cuda::TensorDataTypeMap<TPrecision>::device_type;
        using CudaExecutionContext = ExecutionContext<DeviceType::Cuda>;

        CudaGeluOp( IExecutionContext* context, const GeluConfig& config )
            : context_( validateExecutionContext_<DeviceType::Cuda>( context, "CudaGeluOp")), config_(config), impl_(config)
        {
            config_.validate();
        }

		// ====================================================================
		// Compute
		// ====================================================================

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
        void forward( const ITensor& input, ITensor& output ) const
        {
            // REVIEW: This boilerplate code is fine for now but all ops should share a common helper for this.

            // ITensors must be of the same device type as the operation
            if ( input.getDeviceType() != DeviceType::Cuda || output.getDeviceType() != DeviceType::Cuda )
            {
                throw std::invalid_argument( "CudaGeluOp: Input and output tensors must be on CUDA device." );
            }

            if ( output.size() < input.size() )
            {
                throw std::invalid_argument( "CudaGeluOp: output tensors must be greater or equal to the input size." );
            }

            // TODO: Validate tensor data types match TPrecision

            cudaStream_t stream = context_->getStream();

            // TODO: Use precision policy from config
            // ComputePrecision::Policy policy = config_.getPrecisionPolicy();

            auto X = static_cast<const NativeType*>(input.rawData());
            auto Y = static_cast<NativeType*>(output.rawData());
            int N = static_cast<int>(input.size());

            impl_.forward( Y, X, N, stream );
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
            const ITensor& input,
            const ITensor& output_gradient,
            ITensor& input_gradient ) const
        {

            //ComputePrecision::Policy policy = this->getPrecisionPolicy();

            // Get tensor data pointers (native)
            const NativeType* X = static_cast<const NativeType*>(input.rawData());
            const NativeType* dY = static_cast<const NativeType*>(output_gradient.rawData());
            NativeType* dX = static_cast<NativeType*>(input_gradient.rawData());
            
            int N = static_cast<int>(input.size());

            // REVIEW: Cast and store during construction?
            auto* cuda_context = static_cast<CudaExecutionContext*>(context_);
            cudaStream_t stream = cuda_context->getStream();

            // NOTE: not implemented yet for all native types)
            impl_.backward( dX, X, dY, N, stream );
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

        /*const GeluConfig& getConfig() const {
            return config_;
        }*/

    private:

        GeluConfig config_; ///< Configuration for the GELU operation.
		CudaExecutionContext* context_; ///< Execution context for CUDA resources.
        Detail::cuda_gelu_impl<NativeType> impl_; ///< Implementation details for the GELU operation.
    };

}