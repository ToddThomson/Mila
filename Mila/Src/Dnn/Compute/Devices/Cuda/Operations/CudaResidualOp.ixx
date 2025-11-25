/**
 * @file CudaResidualOp.ixx
 * @brief CUDA implementation of the residual (y = x + F(x)) binary operation.
 *
 * Device- and precision-specific implementation of the Residual operation for
 * CUDA devices. Implements the device-agnostic BinaryOperation interface and
 * dispatches to device kernels in the `Detail` namespace based on the native
 * compute type (float / half).
 *
 * Notes:
 *  - Forward is implemented using the CUDA kernels exposed in Kernels/CudaOps.h.
 *  - Backward pass is not implemented and will throw if invoked.
 *  - The class requires a non-null CUDA execution context at construction.
 *
 * @since Alpha
 */

module;
#include "Kernels/CudaOps.h"
#include <memory>
#include <cuda_fp16.h>
#include <stdexcept>
#include <string>

export module Compute.CudaResidualOp;

import Dnn.Components.Residual;
import Dnn.ITensor;
import Dnn.TensorDataType;
import Dnn.ComponentConfig;
import Compute.OperationBase;
import Compute.BinaryOperation;
import Compute.OperationRegistry;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.CudaExecutionContext;
import Compute.OperationType;
import Compute.CudaDeviceMemoryResource;
import Compute.CudaTensorDataType;

namespace Mila::Dnn::Compute
{
    //using namespace Mila::Dnn;

    /**
     * @brief CUDA residual kernel dispatch implementations.
     *
     * Specializations provided for float and half native types.
     */
    namespace Detail
    {
        template<typename TElementType>
        struct cuda_residual_impl;

        template<>
        struct cuda_residual_impl<float>
        {
            static inline void forward( float* Y, const float* X1, const float* X2, int N, cudaStream_t stream )
            {
                cuda_residual_forward_fp32( Y, X1, X2, N, stream );
            }

            static inline void backward( float* dX1, float* dX2, const float* dY, size_t N, cudaStream_t stream )
            {
                cuda_residual_backward_fp32( dX1, dX2, dY, N, stream );
            }
        };

        template<>
        struct cuda_residual_impl<half>
        {
            static inline void forward( half* Y, const half* X1, const half* X2, int N, cudaStream_t stream )
            {
                cuda_residual_forward_fp16( Y, X1, X2, N, stream );
            }
            
            static inline void backward( half* dX1, half* dX2, const half* dY, size_t N, cudaStream_t stream )
            {
                cuda_residual_backward_fp16( dX1, dX2, dY, N, stream );
			}
        };
    };

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

        /**
         * @brief Construct with a non-null CUDA execution context.
         *
         * The context parameter is required and validated at runtime. Callers must
         * pass a std::shared_ptr<ExecutionContext<DeviceType::Cuda>>.
         */
        CudaResidualOp( std::shared_ptr<CudaExecutionContext> context, const ResidualConfig& config )
            : context_( context )//, config_( config )
        {
            if (!context_)
            {
                throw std::invalid_argument( "CudaResidualOp requires a non-null CUDA execution context" );
            }
        }

        /**
         * @brief Forward pass: element-wise addition Y = X1 + X2 using CUDA kernel.
         *
         * Input and output buffers are obtained from ITensor::rawData() and must be
         * device pointers compatible with the CUDA kernels.
         */
        void forward(
            const ITensor& input_a,
            const ITensor& input_b,
            ITensor& output ) const override
        {
            const NativeType* X1 = static_cast<const NativeType*>(input_a.rawData());
            const NativeType* X2 = static_cast<const NativeType*>(input_b.rawData());
            NativeType* Y = static_cast<NativeType*>(output.rawData());

            if (!X1 || !X2 || !Y)
            {
                throw std::runtime_error( "CudaResidualOp::forward - null tensor data pointer" );
            }

            const size_t N = input_a.size();

            cudaStream_t stream = context_->getStream();

            Detail::cuda_residual_impl<NativeType>::forward( Y, X1, X2, N, stream );
        }

        /**
         * @brief Backward pass: not implemented for CUDA residual op.
         *
         * Throws std::runtime_error if invoked.
         */
        void backward(
            const ITensor& input_a,
            const ITensor& input_b,
            const ITensor& output_grad,
            ITensor& input_a_grad,
            ITensor& input_b_grad ) const override
        {
            const NativeType* dY = static_cast<const NativeType*>(output_grad.rawData());
            NativeType* dX1 = static_cast<NativeType*>(input_a_grad.rawData());
            NativeType* dX2 = static_cast<NativeType*>(input_b_grad.rawData());

            if (!dY || !dX1 || !dX2)
            {
                throw std::runtime_error( "CudaResidualOp::backward - null gradient tensor data pointer" );
            }

            const size_t N = input_a.size();
            cudaStream_t stream = context_->getStream();

            Detail::cuda_residual_impl<NativeType>::backward( dX1, dX2, dY, N, stream );
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
        
        std::shared_ptr<CudaExecutionContext> context_;
        //ResidualConfig config_;
    };

    export class CudaResidualOpRegistrar
    {
    public:
        template <TensorDataType TInputA, TensorDataType TInputB = TInputA, TensorDataType TPrecision = TInputA>
        static void registerForType( const std::string& opName )
        {
            OperationRegistry::instance().registerBinaryOperation<DeviceType::Cuda, TInputA, TInputB, TPrecision>(
                opName,
                []( std::shared_ptr<ExecutionContext<DeviceType::Cuda>> context, const ComponentConfig& config )
                    -> std::shared_ptr<BinaryOperation<DeviceType::Cuda, TInputA, TInputB, TPrecision>>
                {
                    const auto& residualConfig = static_cast<const ResidualConfig&>(config);
                    auto ctx = std::static_pointer_cast<ExecutionContext<DeviceType::Cuda>>(context);
                    
                    return std::make_shared<CudaResidualOp<TInputA, TInputB, TPrecision>>( ctx, residualConfig );
                }
            );
        }

        static void registerOperations()
        {
            const std::string opName = "ResidualOp";

            registerForType<TensorDataType::FP32>( opName );
            registerForType<TensorDataType::FP16>( opName );
        }

        /*static inline bool isRegistered = []() {
            registerOperations();
            return true;
            }();*/
    };
}