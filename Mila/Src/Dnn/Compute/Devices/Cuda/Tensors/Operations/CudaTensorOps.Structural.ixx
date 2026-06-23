module;
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstring>
#include <algorithm>
#include <memory>
#include <span>
#include <type_traits>
#include <stdexcept>
#include <format>
#include "Kernels/Structural.h"

export module Compute.CudaTensorOps:Structural;

import Dnn.Tensor;
import Dnn.TensorTypes;
import Dnn.ITensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.TensorDataTypeMap;
import Compute.ExecutionContext;
import Compute.IExecutionContext;
import Compute.DeviceType;
//import Compute.DeviceTraits;
import Compute.DeviceId;
import Compute.CudaTensorDataType;
//import Cuda.Helpers;
import Cuda.Error;

namespace Mila::Dnn::Compute::Cuda
{
	using namespace Mila::Dnn::Compute;

	// Forward-declare CUDA-specific execution context type (concrete alias exists in ExecutionContext:Cuda).
	class CudaExecutionContext;

    namespace Detail
    {
        template<typename T>
        struct cuda_structural_kernels;

        // ====================================================================
        // FP32 specialization
        // ====================================================================

        template<>
        struct cuda_structural_kernels<float>
        {
            static void split(
                const float* src,
                float* out_a, float* out_b, float* out_c,
                int src_rows, int dim_a, int dim_b, int dim_c,
                cudaStream_t stream )
            {
                cuda_split3_fp32( 
                    src,
                    out_a, out_b, out_c,
                    src_rows, dim_a, dim_b, dim_c,
                    stream );
            }
        };

        template<>
        struct cuda_structural_kernels<nv_bfloat16>
        {
            static void split(
                const __nv_bfloat16* src,
                __nv_bfloat16* out_a, __nv_bfloat16* out_b, __nv_bfloat16* out_c,
                int src_rows, int dim_a, int dim_b, int dim_c,
                cudaStream_t stream )
            {
                cuda_split3_bf16(
                    src,
                    out_a, out_b, out_c,
                    src_rows, dim_a, dim_b, dim_c,
                    stream );
            }
        };
    }

    export struct StructuralOps
    {
        /**
         * @brief Split a tensor along its last dimension into two contiguous
         * output tensors.
         *
         * Input shape:  [B, T, D0+D1]
         * Output shapes: out0[B, T, D0], out1[B, T, D1]
         *
         * Used by the Gemma global (K=V) attention layer, whose fused projection
         * is [Q | K] with no V section. Implemented by reusing the 3-way kernel
         * with a zero-width third output: with D2 == 0 the input width is exactly
         * D0+D1, so the kernel's third-slice branch (col >= D0+D1) is never taken
         * and the aliased out_c is never written.
         *
         * Preconditions mirror the 3-way split: same device, rank-3 tensors,
         * matching B/T, D0+D1 == input last dim, and D0, D1 multiples of 4.
         */
        template<TensorDataType TDataType, typename TMemoryResource>
            requires isValidTensor<TDataType, TMemoryResource>
        static void split(
            const Dnn::Tensor<TDataType, TMemoryResource>& input,
            Dnn::Tensor<TDataType, TMemoryResource>& out0,
            Dnn::Tensor<TDataType, TMemoryResource>& out1,
            IExecutionContext* exec_context = nullptr )
        {
            using NativeType = typename Cuda::TensorDataTypeMap<TDataType>::device_type;

            const auto& in_shape = input.shape();
            const auto& s0 = out0.shape();
            const auto& s1 = out1.shape();

            if ( in_shape.size() != 3 || s0.size() != 3 || s1.size() != 3 )
            {
                throw std::invalid_argument(
                    "SplitOps::split: all tensors must be rank-3" );
            }

            const int B = static_cast<int>(in_shape[ 0 ]);
            const int T = static_cast<int>(in_shape[ 1 ]);
            const int D = static_cast<int>(in_shape[ 2 ]);

            const int D0 = static_cast<int>(s0[ 2 ]);
            const int D1 = static_cast<int>(s1[ 2 ]);

            if ( D0 + D1 != D )
            {
                throw std::invalid_argument( std::format(
                    "SplitOps::split: output dims {} + {} != input dim {}",
                    D0, D1, D ) );
            }

            if ( D0 % 4 != 0 || D1 % 4 != 0 )
            {
                throw std::invalid_argument( std::format(
                    "SplitOps::split: D0={}, D1={} must both be multiples of 4",
                    D0, D1 ) );
            }

            if ( s0[ 0 ] != B || s0[ 1 ] != T || s1[ 0 ] != B || s1[ 1 ] != T )
            {
                throw std::invalid_argument(
                    "SplitOps::split: B and T dims must match across all tensors" );
            }

            cudaStream_t stream = cudaStreamDefault;

            if ( exec_context )
            {
                auto* cuda_ctx = cast_context_<DeviceType::Cuda>( exec_context );
                stream = cuda_ctx->getStream();
            }

            const NativeType* src = static_cast<const NativeType*>(input.rawData());
            NativeType* dst0 = static_cast<NativeType*>(out0.rawData());
            NativeType* dst1 = static_cast<NativeType*>(out1.rawData());

            // Third output aliases dst1 with width 0: the branch that would write
            // it is unreachable (col < D0+D1 == D for every column).
            Detail::cuda_structural_kernels<NativeType>::split(
                src, dst0, dst1, dst1, B * T, D0, D1, 0, stream );
        }

        /**
         * @brief Split a tensor along its last dimension into three contiguous
         * output tensors.
         *
         * Input shape:  [B, T, D0+D1+D2]
         * Output shapes: out0[B, T, D0], out1[B, T, D1], out2[B, T, D2]
         *
         * Preconditions:
         *   - All tensors must be on the same CUDA device.
         *   - Input last dim must equal sum of output last dims.
         *   - D0, D1, D2 must be multiples of 4 (float4 vectorization).
         *   - Input and outputs must be rank-3 tensors.
         *
         * @tparam TDataType Tensor element type — drives vectorization width.
         * @tparam TMemRes   Memory resource type backing the tensors.
         */
        template<TensorDataType TDataType, typename TMemoryResource>
            requires isValidTensor<TDataType, TMemoryResource>
        static void split(
            const Dnn::Tensor<TDataType, TMemoryResource>& input,
            Dnn::Tensor<TDataType, TMemoryResource>& out0,
            Dnn::Tensor<TDataType, TMemoryResource>& out1,
            Dnn::Tensor<TDataType, TMemoryResource>& out2,
            IExecutionContext* exec_context = nullptr )
        {
            using NativeType = typename Cuda::TensorDataTypeMap<TDataType>::device_type;

            const auto& in_shape = input.shape();
            const auto& s0 = out0.shape();
            const auto& s1 = out1.shape();
            const auto& s2 = out2.shape();

            if ( in_shape.size() != 3 || s0.size() != 3 || s1.size() != 3 || s2.size() != 3 )
            {
                throw std::invalid_argument(
                    "SplitOps::split: all tensors must be rank-3" );
            }

            const int B = static_cast<int>(in_shape[ 0 ]);
            const int T = static_cast<int>(in_shape[ 1 ]);
            const int D = static_cast<int>(in_shape[ 2 ]);

            const int D0 = static_cast<int>(s0[ 2 ]);
            const int D1 = static_cast<int>(s1[ 2 ]);
            const int D2 = static_cast<int>(s2[ 2 ]);

            if ( D0 + D1 + D2 != D )
            {
                throw std::invalid_argument( std::format(
                    "SplitOps::split: output dims {} + {} + {} != input dim {}",
                    D0, D1, D2, D ) );
            }

            if ( D0 % 4 != 0 || D1 % 4 != 0 || D2 % 4 != 0 )
            {
                throw std::invalid_argument( std::format(
                    "SplitOps::split: D0={}, D1={}, D2={} must all be multiples of 4",
                    D0, D1, D2 ) );
            }

            if ( s0[ 0 ] != B || s0[ 1 ] != T ||
                s1[ 0 ] != B || s1[ 1 ] != T ||
                s2[ 0 ] != B || s2[ 1 ] != T )
            {
                throw std::invalid_argument(
                    "SplitOps::split: B and T dims must match across all tensors" );
            }

            cudaStream_t stream = cudaStreamDefault;

            if ( exec_context )
            {
                auto* cuda_ctx = cast_context_<DeviceType::Cuda>( exec_context );
                stream = cuda_ctx->getStream();
            }

            const NativeType* src = static_cast<const NativeType*>(input.rawData());
            NativeType* dst0 = static_cast<NativeType*>(out0.rawData());
            NativeType* dst1 = static_cast<NativeType*>(out1.rawData());
            NativeType* dst2 = static_cast<NativeType*>(out2.rawData());

            Detail::cuda_structural_kernels<NativeType>::split(
                src,
                dst0, dst1, dst2,
                B * T, D0, D1, D2,
                stream );
        }
    };
}