/**
 * @file CudaResidualOp.ixx
 * @brief CUDA implementation of the residual (y = x + F(x)) binary operation.
 *
 * Device- and precision-specific implementation of the Residual operation for
 * CUDA devices. Implements the device-agnostic BinaryOperation interface and
 * dispatches to device kernels in the `Detail` namespace based on the native
 * compute type (float / half).
 *
 * @since Alpha
 */

module;
#include "Kernels/Residual.cuh"
#include <memory>
#include <cuda_fp16.h>
#include <stdexcept>
#include <cstdint>
#include <cassert>
#include <string>

export module Compute.CudaResidualOp;
import :Dispatch;

import Dnn.Components.Residual;
import Dnn.ITensor;
import Dnn.TensorDataType;
import Dnn.ComponentConfig;
import Compute.OperationBase;
import Compute.BinaryOperation;
import Compute.OperationRegistry;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.OperationType;
import Compute.CudaDeviceMemoryResource;
import Compute.CudaTensorDataType;
import Compute.OperationRegistrarHelpers;

namespace Mila::Dnn::Compute::Cuda::Residual
{
    /**
     * @brief CUDA Residual operation implementing the BinaryOperation interface.
     *
     * @tparam TInputA Precision of first input tensor
     * @tparam TInputB Precision of second input tensor
     * @tparam TPrecision Computation precision
     */
    export template <TensorDataType TInputA, TensorDataType TInputB = TInputA, TensorDataType TPrecision = TInputA>
    class CudaResidualOp : public BinaryOperation<DeviceType::Cuda, TInputA, TInputB, TPrecision>
    {
    public:
        using MR = CudaDeviceMemoryResource;
        using NativeType = typename Mila::Dnn::Compute::Cuda::TensorDataTypeMap<TPrecision>::native_type;
        using CudaExecutionContext = ExecutionContext<DeviceType::Cuda>;
        using ConfigType = ResidualConfig;

        /**
         * @brief Construct with a non-null CUDA execution context.
         *
         * The context parameter is required and validated at runtime. Callers must
         * pass a std::shared_ptr<ExecutionContext<DeviceType::Cuda>>.
         */
        CudaResidualOp( IExecutionContext* context, const ResidualConfig& config )
            : context_( validateExecutionContext_<DeviceType::Cuda>( context, "CudaResidualOp" ) ), config_( config )
        {
            if (!context_)
            {
                throw std::invalid_argument( "CudaResidualOp requires a non-null CUDA execution context" );
            }

            scale_ = config_.getScalingFactor();
        }

        /**
         * @brief Forward pass: element-wise addition Y = X1 + X2 using CUDA kernel.
         *
         * Input and output buffers are obtained from ITensor::rawData() and must be
         * device pointers compatible with the CUDA kernels.
         */
        void forward(
            const ITensor& input_A,
            const ITensor& input_B,
            ITensor& output ) const override
        {
            const NativeType* A = static_cast<const NativeType*>(input_A.rawData());
            const NativeType* B = static_cast<const NativeType*>(input_B.rawData());
            NativeType* Y = static_cast<NativeType*>(output.rawData());

            if ( !A || !B || !Y )
            {
                throw std::runtime_error( "CudaResidualOp::forward - null tensor data pointer" );
            }

            const int64_t actual_size = input_A.size();

            if ( input_B.size() != actual_size )
            {
                throw std::runtime_error( "CudaResidualOp::forward - input/output size mismatch" );
            }

            if ( actual_size > max_size_ )
            {
                throw std::runtime_error( "CudaResidualOp::forward - input size exceeds built max" );
            }

            cudaStream_t stream = context_->getStream();

            assert( scale_ == 1.0f && "Current implementation only supports scale=1.0f" );

            Detail::cuda_residual_impl<NativeType>::forward(
                Y, A, B, scale_, static_cast<int>(actual_size), stream );
        }

        /**
         * @brief Backward pass
         */
        void backward(
            const ITensor& input_A, // REVIEW: Unused
            const ITensor& input_B, // REVIEW: Unused
            const ITensor& output_grad,
            ITensor& A_grad,
            ITensor& B_grad ) const override
        {
            const NativeType* dY = static_cast<const NativeType*>(output_grad.rawData());
            NativeType* dA = static_cast<NativeType*>(A_grad.rawData());
            NativeType* dB = static_cast<NativeType*>(B_grad.rawData());

            if ( !dY || !dA || !dB )
            {
                throw std::runtime_error( "CudaResidualOp::backward - null gradient tensor data pointer" );
            }

            const int64_t actual_size = output_grad.size();

            if ( A_grad.size() != actual_size || B_grad.size() != actual_size )
            {
                throw std::runtime_error( "CudaResidualOp::backward - gradient size mismatch" );
            }

            if ( actual_size > max_size_ )
            {
                throw std::runtime_error( "CudaResidualOp::backward - gradient size exceeds built max" );
            }

            cudaStream_t stream = context_->getStream();

            Detail::cuda_residual_impl<NativeType>::backward(
                dA, dB, dY, static_cast<int>(actual_size), stream );
        }

        void build( const shape_t& input_shape )
        {
            max_size_ = 1;

            for ( const auto& dim : input_shape )
            {
                max_size_ *= dim;
            }
        }

        OperationType getOperationType() const override
        {
            return OperationType::ResidualOp;
        }

        std::string getName() const override
        {
            return "Cuda::ResidualOp";
        }

    private:
        
        CudaExecutionContext* context_{ nullptr };
        ResidualConfig config_;
        int64_t max_size_{ 0 };
        float scale_{ 1.0f };
    };

    export class CudaResidualOpRegistrar
    {
    public:

        static void registerOperations()
        {
            const std::string opName = "ResidualOp";

            // Register for FP32
            registerBinaryOpType<DeviceType::Cuda,
                CudaResidualOp<TensorDataType::FP32>,
                TensorDataType::FP32, TensorDataType::FP32,
                TensorDataType::FP32>(opName);

            // Register for FP16
            registerBinaryOpType<DeviceType::Cuda,
                CudaResidualOp<TensorDataType::FP16>,
                TensorDataType::FP16, TensorDataType::FP16,
                TensorDataType::FP16>(opName);
        }
    };
}