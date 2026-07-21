/**
 * @file CudaGegluOp.ixx
 * @brief CUDA GeGLU (GELU-gated linear unit) operation -- forward only.
 *
 * GeGLU(gate, up) = GeluTanh(gate) * up. Used by Gemma's GeGLU FFN. The SiLU
 * SwiGLU path (CudaSwigluOp + its optimized kernels) is a separate, untouched
 * unit; this op shares only the gate/up layout and the GeluTanh math source.
 *
 * Gemma is inference-only, so backward() is an honest throwing stub deferred to
 * Training Revival (mirrors the GQA inference-only contract).
 */

module;
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include "Kernels/Geglu.cuh"

export module Compute.CudaGegluOp;

import Dnn.Components.SwigluConfig;
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

namespace Mila::Dnn::Compute::Cuda::Geglu
{
    using namespace Mila::Dnn;

    export template<TensorDataType TPrecision>
        requires ValidFloatTensorDataType<TPrecision>
    class CudaGegluOp : public Operation<DeviceType::Cuda, TPrecision>
    {
    public:
        using MR = CudaDeviceMemoryResource;
        using NativeType = typename Mila::Dnn::Compute::Cuda::TensorDataTypeMap<TPrecision>::device_type;
        using CudaExecutionContext = ExecutionContext<DeviceType::Cuda>;

        CudaGegluOp( IExecutionContext* context, const SwigluConfig& config )
            : context_( validateExecutionContext_<DeviceType::Cuda>( context, "CudaGegluOp" ) ), config_( config )
        {
            config_.validate();
        }

        void forward( const ITensor& input, ITensor& output ) const
        {
            if ( input.size() % 2 != 0 )
            {
                throw std::invalid_argument( "CudaGegluOp: input must have an even element count (gate|up split)." );
            }

            const size_t out_size = input.size() / 2;
            if ( output.size() != out_size )
            {
                throw std::invalid_argument( "CudaGegluOp: output must be half the input size." );
            }

            int N = static_cast<int>(out_size);
            int half_width = static_cast<int>(input.shape().back() / 2);

            auto* cuda_context = static_cast<CudaExecutionContext*>(context_);
            cudaStream_t stream = cuda_context->getStream();

            auto X = static_cast<const NativeType*>(input.rawData());
            auto Y = static_cast<NativeType*>(output.rawData());

            if constexpr ( TPrecision == TensorDataType::BF16 )
            {
                cuda_geglu_forward_bf16( Y, X, N, half_width, stream );
            }
            else
            {
                cuda_geglu_forward_fp32( Y, X, N, half_width, stream );
            }
        }

        void backward( const ITensor&, const ITensor&, ITensor& ) const
        {
            throw std::runtime_error(
                "CudaGegluOp::backward is not implemented; GeGLU is inference-only in Mila "
                "(Gemma). The backward path is deferred to Training Revival." );
        }

        OperationType getOperationType() const override
        {
            return OperationType::GegluOp;
        }

        std::string getName() const override
        {
            return "Cuda::GegluOp";
        }

    private:
        SwigluConfig config_;
        CudaExecutionContext* context_;
    };
}
