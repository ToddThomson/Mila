/**
 * @file CudaAttentionOp.ixx
 * @brief CUDA implementation of Multi-Head Attention cuBLASLt optimization.
 */

module;
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <vector>
#include <memory>
#include <string>
#include <stdexcept>
#include <cstdint>
#include <type_traits>
#include <sstream>
#include <cassert>
#include "Kernels/CudaAttention.cuh"

export module Compute.CudaAttentionOp;

import Dnn.Components.Attention;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.TensorOps;
import Dnn.ComponentConfig;
import Compute.OperationBase;
import Compute.UnaryOperation;
import Compute.Precision;
import Compute.OperationRegistry;
import Compute.Device;
import Compute.DeviceType;
import Compute.IExecutionContext;
import Compute.ExecutionContext;
import Compute.OperationType;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.CudaDeviceMemoryResource;
import Compute.CudaTensorDataType;
import Compute.CudaDevice;
import CublasLt.Error;
import CublasLtHelpers;
import Utils.Logger;

namespace Mila::Dnn::Compute::Cuda::Attention
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute::Cuda::Common::CublasLtHelpers;

    namespace Detail
    {
        /**
         * @brief cuBLASLt matmul execution plan for attention operations.
         */
        template <typename TNative>
        using CublasLtMatMulPlan = CublasLtMatMulPlan<TNative>;

        /**
         * @brief Build cuBLASLt plan for Q·K^T attention score computation (row-major).
         *
         * Row-major storage: Q[K] and K[K] are stored as [T, HS] (rows = sequence length, cols = head size).
         * Mathematical operation: preatt[T, T] = Q[T, HS] @ K^T[HS, T]
         */
        template <typename TNative>
        CublasLtMatMulPlan<TNative> build_qk_score_plan(
            cublasLtHandle_t handle,
            int batch_size,
            int num_heads,
            int seq_length,
            int head_size,
            cudaDataType_t cuda_data_type,
            cublasComputeType_t compute_type,
            cudaDataType_t scale_type )
        {
            const int batch_count = batch_size * num_heads;

            // Row-major element stride (elements per head per batch): rows * cols = T * HS
            const long long strideA = static_cast<long long>(seq_length) * static_cast<long long>(head_size);
            const long long strideB = static_cast<long long>(seq_length) * static_cast<long long>(head_size);
            const long long strideC = static_cast<long long>(seq_length) * static_cast<long long>(seq_length);

            // Row-major interpretation:
            // - A (Q): rows = T, cols = HS, ldA = HS
            // - B (K): rows = T, cols = HS, ldB = HS (we will use K^T via opB = CUBLAS_OP_T)
            // - C (preatt): rows = T, cols = T, ldC = T
            auto plan = build_strided_plan<TNative>(
                handle,
                /*A_rows=*/ seq_length, /*A_cols=*/ head_size, /*ldA=*/ head_size, /*strideA=*/ strideA,
                /*B_rows=*/ seq_length, /*B_cols=*/ head_size, /*ldB=*/ head_size, /*strideB=*/ strideB,
                /*C_rows=*/ seq_length, /*C_cols=*/ seq_length, /*ldC=*/ seq_length, /*strideC=*/ strideC,
                /*opA=*/ CUBLAS_OP_N, /*opB=*/ CUBLAS_OP_T,   // Q @ K^T
                batch_count,
                false,
                compute_type,
                cuda_data_type,
                scale_type );

            if ( !plan.isValid() || !plan.has_algorithm )
            {
                Utils::Logger::warning( "cuBLASLt QK score plan built without algorithm (will use default)" );
            }

            return plan;
        }

        template <typename TNative>
        CublasLtMatMulPlan<TNative> build_att_value_plan(
            cublasLtHandle_t handle,
            int batch_size,
            int num_heads,
            int seq_length,
            int head_size,
            cudaDataType_t cuda_data_type,
            cublasComputeType_t compute_type,
            cudaDataType_t scale_type )
        {
            const int batch_count = batch_size * num_heads;

            //
            // Row-major interpretation:
            // - A (att): rows = T, cols = T, ldA = T, strideA = T * T
            // - B (V)  : rows = T, cols = HS, ldB = HS, strideB = T * HS
            // - C (v_out): rows = T, cols = HS, ldC = HS, strideC = T * HS
            //
            // Mathematical operation: v_out[T, HS] = att[T, T] @ V[T, HS]
            //

            const long long strideA = static_cast<long long>(seq_length) * static_cast<long long>(seq_length); // T * T
            const long long strideB = static_cast<long long>(seq_length) * static_cast<long long>(head_size); // T * HS
            const long long strideC = static_cast<long long>(seq_length) * static_cast<long long>(head_size); // T * HS

            auto plan = build_strided_plan<TNative>(
                handle,
                /*A_rows=*/ seq_length, /*A_cols=*/ seq_length, /*ldA=*/ seq_length, /*strideA=*/ strideA, // att: [T, T]
                /*B_rows=*/ seq_length, /*B_cols=*/ head_size,  /*ldB=*/ head_size,  /*strideB=*/ strideB, // V:   [T, HS]
                /*C_rows=*/ seq_length, /*C_cols=*/ head_size,  /*ldC=*/ head_size,  /*strideC=*/ strideC, // v_out:[T, HS]
                /*opA=*/ CUBLAS_OP_N, /*opB=*/ CUBLAS_OP_N, // v_out = A @ B
                batch_count,
                false,
                compute_type,
                cuda_data_type,
                scale_type );

            if ( !plan.isValid() || !plan.has_algorithm )
            {
                Utils::Logger::warning( "cuBLASLt Att-Value plan built without algorithm (will use default)" );
            }

            return plan;
        }

        template <typename TNative>
        CublasLtMatMulPlan<TNative> build_backward_v_plan(
            cublasLtHandle_t handle,
            int batch_size,
            int num_heads,
            int seq_length,
            int head_size,
            cudaDataType_t cuda_data_type,
            cublasComputeType_t compute_type,
            cudaDataType_t scale_type )
        {
            const int batch_count = batch_size * num_heads;

            //
            // Row-major interpretation (refactored):
            // - A (att):   rows = T, cols = T,  ldA = T,  strideA = T * T
            // - B (dVout): rows = T, cols = HS, ldB = HS, strideB = T * HS
            // - C (dV):    rows = T, cols = HS, ldC = HS, strideC = T * HS
            //
            // Mathematical operation (row-major): dV[T,HS] = Att^T[T,T] @ dVout[T,HS]
            // Map to cuBLAS op flags: opA = CUBLAS_OP_T, opB = CUBLAS_OP_N
            //

            const long long strideA = static_cast<long long>(seq_length) * static_cast<long long>(seq_length); // T * T
            const long long strideB = static_cast<long long>(seq_length) * static_cast<long long>(head_size); // T * HS
            const long long strideC = static_cast<long long>(seq_length) * static_cast<long long>(head_size); // T * HS

            auto plan = build_strided_plan<TNative>(
                handle,
                /*A_rows=*/ seq_length, /*A_cols=*/ seq_length, /*ldA=*/ seq_length, /*strideA=*/ strideA, // att: [T, T]
                /*B_rows=*/ seq_length, /*B_cols=*/ head_size,  /*ldB=*/ head_size,  /*strideB=*/ strideB, // dVout: [T, HS]
                /*C_rows=*/ seq_length, /*C_cols=*/ head_size,  /*ldC=*/ head_size,  /*strideC=*/ strideC, // dV: [T, HS]
                /*opA=*/ CUBLAS_OP_T, /*opB=*/ CUBLAS_OP_N, // dV = Att^T @ dVout
                batch_count,
                false,
                compute_type,
                cuda_data_type,
                scale_type );

            if ( !plan.isValid() || !plan.has_algorithm )
            {
                Utils::Logger::warning( "cuBLASLt backward dV plan built without algorithm (will use default)" );
            }

            return plan;
        }

        template <typename TNative>
        CublasLtMatMulPlan<TNative> build_backward_att_plan(
            cublasLtHandle_t handle,
            int batch_size,
            int num_heads,
            int seq_length,
            int head_size,
            cudaDataType_t cuda_data_type,
            cublasComputeType_t compute_type,
            cudaDataType_t scale_type )
        {
            const int batch_count = batch_size * num_heads;

            //
            // Row-major interpretation (refactored):
            // - A (dVout): rows = T, cols = HS, ldA = HS, strideA = T * HS
            // - B (V)    : rows = T, cols = HS, ldB = HS, strideB = T * HS
            // - C (dAtt) : rows = T, cols = T,  ldC = T,  strideC = T * T
            //
            // Mathematical operation: dAtt[T, T] = dVout[T, HS] @ V[T, HS]^T
            // cuBLAS mapping: opA = CUBLAS_OP_N, opB = CUBLAS_OP_T
            //

            const long long strideA = static_cast<long long>(seq_length) * static_cast<long long>(head_size); // T * HS
            const long long strideB = static_cast<long long>(seq_length) * static_cast<long long>(head_size); // T * HS
            const long long strideC = static_cast<long long>(seq_length) * static_cast<long long>(seq_length); // T * T

            auto plan = build_strided_plan<TNative>(
                handle,
                /*A_rows=*/ seq_length, /*A_cols=*/ head_size, /*ldA=*/ head_size, /*strideA=*/ strideA, // dVout: [T, HS]
                /*B_rows=*/ seq_length, /*B_cols=*/ head_size, /*ldB=*/ head_size, /*strideB=*/ strideB, // V:     [T, HS]
                /*C_rows=*/ seq_length, /*C_cols=*/ seq_length, /*ldC=*/ seq_length, /*strideC=*/ strideC, // dAtt:  [T, T]
                /*opA=*/ CUBLAS_OP_N, /*opB=*/ CUBLAS_OP_T, // dAtt = A @ B^T
                batch_count,
                false,
                compute_type,
                cuda_data_type,
                scale_type );

            if ( !plan.isValid() || !plan.has_algorithm )
            {
                Utils::Logger::warning( "cuBLASLt backward dAtt plan built without algorithm (will use default)" );
            }

            return plan;
        }

        template <typename TNative>
        CublasLtMatMulPlan<TNative> build_backward_q_plan(
            cublasLtHandle_t handle,
            int batch_size,
            int num_heads,
            int seq_length,
            int head_size,
            cudaDataType_t cuda_data_type,
            cublasComputeType_t compute_type,
            cudaDataType_t scale_type )
        {
            const int batch_count = batch_size * num_heads;

            // Row-major interpretation:
            // - A (dPreatt): rows = T, cols = T, ldA = T, strideA = T * T
            // - B (K):       rows = T, cols = HS, ldB = HS, strideB = T * HS
            // - C (dQ):      rows = T, cols = HS, ldC = HS, strideC = T * HS
            //
            // Mathematical operation (row-major): dQ[T,HS] = dPreatt[T,T] @ K[T,HS]
            // cuBLAS mapping: opA = CUBLAS_OP_N, opB = CUBLAS_OP_N

            const long long strideA = static_cast<long long>(seq_length) * static_cast<long long>(seq_length); // T * T
            const long long strideB = static_cast<long long>(seq_length) * static_cast<long long>(head_size);  // T * HS
            const long long strideC = static_cast<long long>(seq_length) * static_cast<long long>(head_size);  // T * HS

            auto plan = build_strided_plan<TNative>(
                handle,
                /*A_rows=*/ seq_length, /*A_cols=*/ seq_length, /*ldA=*/ seq_length, /*strideA=*/ strideA, // dPreatt: [T, T]
                /*B_rows=*/ seq_length, /*B_cols=*/ head_size,  /*ldB=*/ head_size,  /*strideB=*/ strideB, // K/q: [T, HS]
                /*C_rows=*/ seq_length, /*C_cols=*/ head_size,  /*ldC=*/ head_size,  /*strideC=*/ strideC, // dQ: [T, HS]
                /*opA=*/ CUBLAS_OP_N, /*opB=*/ CUBLAS_OP_N, // C = A @ B
                batch_count,
                false,
                compute_type,
                cuda_data_type,
                scale_type );

            if ( !plan.isValid() || !plan.has_algorithm )
            {
                Utils::Logger::warning( "cuBLASLt backward dQ plan built without algorithm (will use default)" );
            }

            return plan;
        }

        template <typename TNative>
        CublasLtMatMulPlan<TNative> build_backward_k_plan(
            cublasLtHandle_t handle,
            int batch_size,
            int num_heads,
            int seq_length,
            int head_size,
            cudaDataType_t cuda_data_type,
            cublasComputeType_t compute_type,
            cudaDataType_t scale_type )
        {
            const int batch_count = batch_size * num_heads;

            //
            // Row-major interpretation (refactored):
            // - A (dPreatt): rows = T, cols = T, ldA = T, strideA = T * T
            // - B (Q)      : rows = T, cols = HS, ldB = HS, strideB = T * HS
            // - C (dK)     : rows = T, cols = HS, ldC = HS, strideC = T * HS
            //
            // Mathematical operation: dK[T, HS] = dPreatt^T[T, T] @ Q[T, HS]
            // cuBLAS mapping: opA = CUBLAS_OP_T, opB = CUBLAS_OP_N
            //

            const long long strideA = static_cast<long long>(seq_length) * static_cast<long long>(seq_length); // T * T
            const long long strideB = static_cast<long long>(seq_length) * static_cast<long long>(head_size); // T * HS
            const long long strideC = static_cast<long long>(seq_length) * static_cast<long long>(head_size); // T * HS

            auto plan = build_strided_plan<TNative>(
                handle,
                /*A_rows=*/ seq_length, /*A_cols=*/ seq_length, /*ldA=*/ seq_length, /*strideA=*/ strideA, // dPreatt: [T, T]
                /*B_rows=*/ seq_length, /*B_cols=*/ head_size,  /*ldB=*/ head_size,  /*strideB=*/ strideB, // Q:      [T, HS]
                /*C_rows=*/ seq_length, /*C_cols=*/ head_size,  /*ldC=*/ head_size,  /*strideC=*/ strideC, // dK:     [T, HS]
                /*opA=*/ CUBLAS_OP_T, /*opB=*/ CUBLAS_OP_N, // dK = A^T @ B
                batch_count,
                false,
                compute_type,
                cuda_data_type,
                scale_type );

            if ( !plan.isValid() || !plan.has_algorithm )
            {
                Utils::Logger::warning( "cuBLASLt backward dK plan built without algorithm (will use default)" );
            }

            return plan;
        }

        /**
         * @brief CUDA kernel dispatcher for attention non-matmul operations.
         */
        template <typename TNative>
            requires std::is_same_v<TNative, float> || std::is_same_v<TNative, half>
        struct cuda_mha_kernels;

        template <>
        struct cuda_mha_kernels<float>
        {
            cuda_mha_kernels() = default;

            static inline void permute_qkv(
                float* q, float* k, float* v,
                const float* inp,
                int B, int T, int NH, int HS,
                cudaStream_t stream )
            {
                cuda_permute_qkv_fp32( q, k, v, inp, B, T, NH, HS, stream );
            }

            static inline void unpermute_output(
                const float* vaccum, float* out,
                int B, int T, int NH, int HS,
                cudaStream_t stream )
            {
                cuda_unpermute_output_fp32( vaccum, out, B, T, NH, HS, stream );
            }

            static inline void softmax_forward(
                float* att, float scale, const float* preatt,
                int B, int NH, int T,
                cudaStream_t stream )
            {
                cuda_softmax_forward_fp32( att, scale, preatt, B, NH, T, stream );
            }

            static inline void softmax_backward(
                float* dpreatt, const float* datt, const float* att,
                float scale,
                int B, int NH, int T,
                cudaStream_t stream )
            {
                cuda_softmax_backward_fp32( dpreatt, datt, att, scale, B, NH, T, stream );
            }

            static inline void permute_backward(
                float* dinp,
                const float* dq, const float* dk, const float* dv,
                int B, int T, int NH, int HS,
                cudaStream_t stream )
            {
                cuda_permute_backward_fp32( dinp, dq, dk, dv, B, T, NH, HS, stream );
            }

            static inline void unpermute_backward(
                float* dvaccum, const float* dout,
                int B, int T, int NH, int HS,
                cudaStream_t stream )
            {
                cuda_unpermute_backward_fp32( dvaccum, dout, B, T, NH, HS, stream );
            }
        };

        template <>
        struct cuda_mha_kernels<half>
        {
            cuda_mha_kernels() = default;

            static inline void permute_qkv(
                half* q, half* k, half* v,
                const half* inp,
                int B, int T, int NH, int HS,
                cudaStream_t stream )
            {
                cuda_permute_qkv_fp16( q, k, v, inp, B, T, NH, HS, stream );
            }

            static inline void unpermute_output(
                const half* vaccum, half* out,
                int B, int T, int NH, int HS,
                cudaStream_t stream )
            {
                cuda_unpermute_output_fp16( vaccum, out, B, T, NH, HS, stream );
            }

            static inline void softmax_forward(
                half* att, float scale, const half* preatt,
                int B, int NH, int T,
                cudaStream_t stream )
            {
                cuda_softmax_forward_fp16( att, scale, preatt, B, NH, T, stream );
            }

            static inline void softmax_backward(
                half* dpreatt, const half* datt, const half* att,
                float scale,
                int B, int NH, int T,
                cudaStream_t stream )
            {
                cuda_softmax_backward_fp16( dpreatt, datt, att, scale, B, NH, T, stream );
            }

            static inline void permute_backward(
                half* dinp,
                const half* dq, const half* dk, const half* dv,
                int B, int T, int NH, int HS,
                cudaStream_t stream )
            {
                cuda_permute_backward_fp16( dinp, dq, dk, dv, B, T, NH, HS, stream );
            }

            static inline void unpermute_backward(
                half* dvaccum, const half* dout,
                int B, int T, int NH, int HS,
                cudaStream_t stream )
            {
                cuda_unpermute_backward_fp16( dvaccum, dout, B, T, NH, HS, stream );
            }
        };
    }

    /**
     * @brief CUDA implementation of Multi-Head Attention using column-major cuBLASLt optimization.
     *
     * Design philosophy:
     * - Two-phase initialization: build() creates cuBLASLt plans, forward()/backward() execute them
     * - Column-major layout eliminates most transpose operations in cuBLASLt
     * - All dimension computation and algorithm selection happens once in build()
     * - Forward/backward are hot-path methods with zero setup overhead
     * - Custom CUDA kernels handle permute/unpermute and softmax operations
     *
     * Forward pass:
     *  1. Permute QKV from [B, T, 3*C] to separate Q, K, V [B, NH, HS, T] (column-major)
     *  2. Compute attention scores: preatt = Q^T @ K (exploiting column-major layout)
     *  3. Apply softmax with causal masking: att = softmax(preatt / sqrt(HS))
     *  4. Compute values: vaccum = Att @ V^T
     *  5. Unpermute output from [B, NH, HS, T] to [B, T, C]
     *
     * Backward pass:
     *  1. Unpermute output gradient to [B, NH, HS, T]
     *  2. Compute dV = Att^T @ dvaccum^T
     *  3. Compute dAtt = dvaccum^T @ V
     *  4. Softmax backward: dPreatt = softmax_backward(dAtt, Att)
     *  5. Compute dQ = dPreatt @ K^T
     *  6. Compute dK = dPreatt^T @ Q^T
     *  7. Permute gradients back to concatenated QKV format
     */
    export template<TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, DeviceType::Cuda>
    class CudaAttentionOp : public UnaryOperation<DeviceType::Cuda, TPrecision>
    {
    public:
        using MR = CudaDeviceMemoryResource;
        using UnaryOperationBase = UnaryOperation<DeviceType::Cuda, TPrecision>;
        using TensorType = Tensor<TPrecision, MR>;
        using NativeType = typename Mila::Dnn::Compute::Cuda::TensorDataTypeMap<TPrecision>::native_type;
        using CudaExecutionContext = ExecutionContext<DeviceType::Cuda>;
        using ConfigType = AttentionConfig;

        CudaAttentionOp( IExecutionContext* context, const AttentionConfig& config )
            : context_( validateExecutionContext_<DeviceType::Cuda>( context, "CudaAttentionOp" ) ), config_( config )
        {
            config_.validate();
        }

        void setParameters( ITensor* /*unused1*/, ITensor* /*unused2*/ ) override
        {}

        void setGradients( ITensor* /*unused1*/, ITensor* /*unused2*/ ) override
        {}

        void build( const shape_t& input_shape ) override
        {
            // REVIEW: What happens when build fails partway through?
            validateInputShape( input_shape );

            B_ = static_cast<int>(input_shape[ 0 ]);
            T_ = static_cast<int>(input_shape[ 1 ]);
            qkv_dim_ = static_cast<int>(input_shape[ 2 ]);

            embedding_dim_ = qkv_dim_ / 3;
            NH_ = config_.getNumHeads();
            HS_ = embedding_dim_ / NH_;

            if ( embedding_dim_ % NH_ != 0 )
            {
                throw std::invalid_argument(
                    "CudaAttentionOp: embedding_dim must be divisible by num_heads" );
            }

            allocateStateTensors();

            cublaslt_handle_ = context_->getCublasLtHandle();

            if ( cublaslt_handle_ == nullptr )
            {
                throw std::runtime_error(
                    "CudaAttentionOp requires cuBLASLt support. "
                    "Ensure CUDA 10.1 or newer is installed." );
            }

            // REVIEW: Never used. PrecisionPolicy used in mixed-precision plan building in future?
            precision_policy_ = config_.getPrecisionPolicy();

            // REVIEW: Can definitely throw. What is the user experience here?
            buildCublasLtPlans();

            UnaryOperationBase::build( input_shape );
        }

        void forward( const ITensor& input, ITensor& output ) const override
        {
            const NativeType* X = static_cast<const NativeType*>(input.rawData());
            NativeType* Y = static_cast<NativeType*>(output.rawData());

            cudaStream_t stream = context_->getStream();

            const float alpha = 1.0f;
            const float beta = 0.0f;
            const float scale = 1.0f / sqrtf( static_cast<float>(HS_) );

            // Permute QKV from [B, T, 3*C] to separate Q, K, V with shape [B, NH, T, HS]
            Detail::cuda_mha_kernels<NativeType>::permute_qkv(
                q_, k_, v_,
                X,
                B_, T_, NH_, HS_,
                stream );

            context_->synchronize();
            {
                /*shape_t qkv_shape = { B_, NH_, T_, HS_ };
                
                std::string q_dump = dump_tensor<NativeType>(
                    q_, qkv_shape, this->getName() + ".dbg.Q", 4, stream );

                Utils::Logger::info( this->getName() + ": dbg.Q (device dump):\n" + q_dump );

                std::string k_dump = dump_tensor<NativeType>(
                    k_, qkv_shape, this->getName() + ".dbg.K", 4, stream );

                Utils::Logger::info( this->getName() + ": dbg.K (device dump):\n" + k_dump );*/

                /*std::string v_dump = dump_tensor<NativeType>(
                    v_, qkv_shape, this->getName() + ".dbg.V", 4, stream );

                Utils::Logger::info( this->getName() + ": dbg.V (device dump):\n" + v_dump );*/
            }

            // Compute attention scores: preatt = Q @ K^T
            execute_plan<NativeType>(
                cublaslt_handle_,
                qk_score_plan_,
                &scale,
                q_, k_,
                &beta,
                preatt_,
                nullptr,
                stream );

            /*context_->synchronize();
            {
                shape_t preatt_shape = { B_, NH_, T_, T_ };
                std::string preatt_dump = dump_tensor<NativeType>(
                    preatt_, preatt_shape, this->getName() + ".dbg.preatt", 4, stream );

                Utils::Logger::info( this->getName() + ": dbg.preatt (device dump):\n" + preatt_dump );
            }*/

            // att_: [B, NH, T, T] (after softmax)
            Detail::cuda_mha_kernels<NativeType>::softmax_forward(
                att_, 1.0f, preatt_,
                B_, NH_, T_,
                stream );

            /*context_->synchronize();
            {
                shape_t att_shape = { B_, NH_, T_, T_ };
                std::string att_dump = dump_tensor<NativeType>(
                    att_, att_shape, this->getName() + ".dbg.att", 4, stream );

                Utils::Logger::info( this->getName() + ": dbg.att (device dump):\n" + att_dump );
            }*/

            // Compute output values: v_out = Att @ V^T
            execute_plan<NativeType>(
                cublaslt_handle_,
                att_value_plan_,
                &alpha,
                att_, v_,
                &beta,
                v_out_,
                nullptr,
                stream );

            /*context_->synchronize();
            {
                shape_t v_out_shape = { B_, NH_, T_, HS_ };
                std::string v_out_dump = dump_tensor<NativeType>(
                    v_out_, v_out_shape, this->getName() + ".dbg.vaccum", 4, stream );

                Utils::Logger::info( this->getName() + ": dbg.vaccum (device dump):\n" + v_out_dump );
            }*/

            Detail::cuda_mha_kernels<NativeType>::unpermute_output(
                v_out_, Y,
                B_, T_, NH_, HS_,
                stream );
        }

        void backward(
            const ITensor& input,
            const ITensor& output_grad,
            ITensor& input_grad ) const override
        {
            assert( this->isBuilt() && "CudaAttentionOp must be built before calling backward()" );

            if ( !this->isTraining() )
            {
                throw std::runtime_error( "CudaAttentionOp::backward called in inference mode" );
            }

            const NativeType* dY = static_cast<const NativeType*>(output_grad.rawData());
            NativeType* dX = static_cast<NativeType*>(input_grad.rawData());

            cudaStream_t stream = context_->getStream();

            const float alpha = 1.0f;
            const float beta = 0.0f;
            const float scale = 1.0f / sqrtf( static_cast<float>(HS_) );

            Detail::cuda_mha_kernels<NativeType>::unpermute_backward(
                dVout_, dY,
                B_, T_, NH_, HS_,
                stream );

            /*context_->synchronize();
            {
                shape_t dVout_shape = { B_, NH_, T_, HS_ };
                std::string dVout_dump = dump_tensor<NativeType>(
                    dVout_, dVout_shape, this->getName() + ".dbg.dVout", 4, stream );
                Utils::Logger::info( this->getName() + ": dbg.dVout (device dump):\n" + dVout_dump );
            }*/

            /*context_->synchronize();
            {
                shape_t att_shape = { B_, NH_, T_, T_ };
                std::string att_dump = dump_tensor<NativeType>(
                    att_, att_shape, this->getName() + ".dbg.att", 4, stream );
                Utils::Logger::info( this->getName() + ": dbg.att (device dump):\n" + att_dump );
            }*/

            // Compute dV = Att^T @ dVout^T
            execute_plan<NativeType>(
                cublaslt_handle_,
                backward_v_plan_,
                &alpha,
                att_, dVout_,
                &beta,
                dV_,
                nullptr,
                stream );

            //context_->synchronize();
            //{
            //    // TJT: Verified correctness of dV via unit tests.

            //    shape_t dv_shape = { B_, NH_, T_, HS_ };
            //    std::string dv_dump = dump_tensor<NativeType>(
            //        dV_, dv_shape, this->getName() + ".dbg.dV", 4, stream );
            //    Utils::Logger::info( this->getName() + ": dbg.dV (device dump):\n" + dv_dump );
            //}

            /*context_->synchronize();
            {
                shape_t v_shape = { B_, NH_, T_, HS_ };
                std::string v_dump = dump_tensor<NativeType>(
                    v_, v_shape, this->getName() + ".dbg.V_backward", 4, stream );
                Utils::Logger::info( this->getName() + ": dbg.V (for backward, device dump):\n" + v_dump );
            }*/

            // Compute dAtt = dVout^T @ V
            execute_plan<NativeType>(
                cublaslt_handle_,
                backward_att_plan_,
                &alpha,
                dVout_, v_,
                &beta,
                datt_,
                nullptr,
                stream );

            //context_->synchronize();
            //{
            //    // TJT: Verified correctness of dAtt via unit tests.

            //    // Dump dAtt using CublasLtHelpers debug utility for column-major tensors.
            //    shape_t datt_shape = { B_, NH_, T_, T_ };
            //    std::string datt_dump = dump_tensor<NativeType>(
            //        datt_, datt_shape, this->getName() + ".dbg.dAtt", 4, stream );
            //    Utils::Logger::info( this->getName() + ": dbg.dAtt (device dump):\n" + datt_dump );
            //}

            Detail::cuda_mha_kernels<NativeType>::softmax_backward(
                dpreatt_, datt_, att_,
                scale,
                B_, NH_, T_,
                stream );

            //context_->synchronize();
            //{
            //    // TJT: Verified correctness of dPreatt via unit tests.

            //    shape_t dpreatt_shape = { B_, NH_, T_, T_ };
            //    std::string dpreatt_dump = dump_tensor<NativeType>(
            //        dpreatt_, dpreatt_shape, this->getName() + ".dbg.dPreatt", 4, stream );
            //    Utils::Logger::info( this->getName() + ": dbg.dPreatt (for backward, device dump):\n" + dpreatt_dump );
            //}

            /*context_->synchronize();
            {
                shape_t k_shape = { B_, NH_, T_, HS_ };
                std::string k_dump = dump_tensor<NativeType>(
                    k_, k_shape, this->getName() + ".dbg.K_backward", 4, stream );
                Utils::Logger::info( this->getName() + ": dbg.K (for backward, device dump):\n" + k_dump );
            }*/

            // Compute dQ = dPreatt @ K^T
            execute_plan<NativeType>(
                cublaslt_handle_,
                backward_q_plan_,
                &alpha,
                dpreatt_, k_,
                &beta,
                dq_,
                nullptr,
                stream );

            /*context_->synchronize();
            {
                shape_t dq_shape = { B_, NH_, HS_, T_ };
                
                std::string dq_dump = dump_tensor<NativeType>(
                    dq_, dq_shape, this->getName() + ".dbg.dQ", 4, stream );
                Utils::Logger::info( this->getName() + ": dbg.dQ (device dump):\n" + dq_dump );
            }*/

            /*context_->synchronize();
            {
                shape_t q_shape = { B_, NH_, HS_, T_ };
                std::string q_dump = dump_tensor<NativeType>(
                    q_, q_shape, this->getName() + ".dbg.Q_backward", 4, stream );
                Utils::Logger::info( this->getName() + ": dbg.Q (for backward, device dump):\n" + q_dump );
            }   */

            // Compute dK = dPreatt^T @ Q^T
            execute_plan<NativeType>(
                cublaslt_handle_,
                backward_k_plan_,
                &alpha,
                dpreatt_, q_,
                &beta,
                dk_,
                nullptr,
                stream );

            /*context_->synchronize();
            {
                shape_t dk_shape = { B_, NH_, HS_, T_ };
                std::string dk_dump = dump_tensor<NativeType>(
                    dk_, dk_shape, this->getName() + ".dbg.dK", 4, stream );
                Utils::Logger::info( this->getName() + ": dbg.dK (device dump):\n" + dk_dump );
            }*/

            // Permute gradients back to concatenated QKV format
            Detail::cuda_mha_kernels<NativeType>::permute_backward(
                dX,
                dq_, dk_, dV_,
                B_, T_, NH_, HS_,
                stream );
        }

        OperationType getOperationType() const override
        {
            return OperationType::AttentionOp;
        }

        std::string getName() const override
        {
            return "Cuda::AttentionOp";
        }

        const AttentionConfig& getConfig() const
        {
            return config_;
        }

    private:
        AttentionConfig config_;
        CudaExecutionContext* context_;

        int B_{ 0 };
        int T_{ 0 };
        int qkv_dim_{ 0 };
        int embedding_dim_{ 0 };
        int NH_{ 0 };
        int HS_{ 0 };

        cublasLtHandle_t cublaslt_handle_{ nullptr };
        ComputePrecision::Policy precision_policy_;

        Detail::CublasLtMatMulPlan<NativeType> qk_score_plan_;
        Detail::CublasLtMatMulPlan<NativeType> att_value_plan_;
        Detail::CublasLtMatMulPlan<NativeType> backward_v_plan_;
        Detail::CublasLtMatMulPlan<NativeType> backward_att_plan_;
        Detail::CublasLtMatMulPlan<NativeType> backward_q_plan_;
        Detail::CublasLtMatMulPlan<NativeType> backward_k_plan_;

        std::shared_ptr<TensorType> q_tensor_;
        std::shared_ptr<TensorType> k_tensor_;
        std::shared_ptr<TensorType> v_tensor_;
        std::shared_ptr<TensorType> preatt_tensor_;
        std::shared_ptr<TensorType> att_tensor_;
        std::shared_ptr<TensorType> v_out_tensor_;

        std::shared_ptr<TensorType> dq_tensor_;
        std::shared_ptr<TensorType> dk_tensor_;
        std::shared_ptr<TensorType> dV_tensor_;
        std::shared_ptr<TensorType> dpreatt_tensor_;
        std::shared_ptr<TensorType> datt_tensor_;
        std::shared_ptr<TensorType> dVout_tensor_;

        NativeType* q_{ nullptr };
        NativeType* k_{ nullptr };
        NativeType* v_{ nullptr };
        NativeType* preatt_{ nullptr };
        NativeType* att_{ nullptr };
        NativeType* v_out_{ nullptr };

        NativeType* dq_{ nullptr };
        NativeType* dk_{ nullptr };
        NativeType* dV_{ nullptr };
        NativeType* dpreatt_{ nullptr };
        NativeType* datt_{ nullptr };
        NativeType* dVout_{ nullptr };

        void validateInputShape( const shape_t& input_shape ) const
        {
            // REVIEW: Should input shape be just [ B, T ]?
            // We already know embedding_dim from config.
            if ( input_shape.size() != 3 )
            {
                throw std::invalid_argument(
                    "CudaAttentionOp: input must have rank 3 (batch_size, seq_length, 3*embedding_dim)" );
            }

            const int64_t expected_qkv_dim = 3 * config_.getEmbeddingDim();

            if ( input_shape[ 2 ] != expected_qkv_dim )
            {
                throw std::invalid_argument(
                    "CudaAttentionOp: input last dimension must be 3*embedding_dim (Q, K, V concatenated)" );
            }
        }

        void allocateStateTensors()
        {
            auto device = context_->getDeviceId();

            // Each of Q, K, V shape: [B, NH, T, HS]
            shape_t qkv_shape = { B_, NH_, T_, HS_ };

            q_tensor_ = std::make_shared<TensorType>( device, qkv_shape );
            q_tensor_->setName( "q_" );
            q_ = static_cast<NativeType*>(q_tensor_->rawData());

            k_tensor_ = std::make_shared<TensorType>( device, qkv_shape );
            k_tensor_->setName( "k_" );
            k_ = static_cast<NativeType*>(k_tensor_->rawData());

            v_tensor_ = std::make_shared<TensorType>( device, qkv_shape );
            v_tensor_->setName( "v_" );
            v_ = static_cast<NativeType*>(v_tensor_->rawData());

            // attention scores shape: [B, NH, T, T]
            shape_t att_shape = { B_, NH_, T_, T_ };

            preatt_tensor_ = std::make_shared<TensorType>( device, att_shape );
            preatt_tensor_->setName( "preatt_" );
            preatt_ = static_cast<NativeType*>(preatt_tensor_->rawData());

            att_tensor_ = std::make_shared<TensorType>( device, att_shape );
            att_tensor_->setName( "att_" );
            att_ = static_cast<NativeType*>(att_tensor_->rawData());

            // v_out_ shape: [B, NH, T, HS]
            shape_t v_out_shape = { B_, NH_, T_, HS_ };

            v_out_tensor_ = std::make_shared<TensorType>( device, v_out_shape );
            v_out_tensor_->setName( "v_out_" );
            v_out_ = static_cast<NativeType*>(v_out_tensor_->rawData());

            dq_tensor_ = std::make_shared<TensorType>( device, qkv_shape );
            dq_tensor_->setName( "dq_" );
            dq_ = static_cast<NativeType*>(dq_tensor_->rawData());

            dk_tensor_ = std::make_shared<TensorType>( device, qkv_shape );
            dk_tensor_->setName( "dk_" );
            dk_ = static_cast<NativeType*>(dk_tensor_->rawData());

            dV_tensor_ = std::make_shared<TensorType>( device, qkv_shape );
            dV_tensor_->setName( "dV_" );
            dV_ = static_cast<NativeType*>(dV_tensor_->rawData());

            dpreatt_tensor_ = std::make_shared<TensorType>( device, att_shape );
            dpreatt_tensor_->setName( "dpreatt_" );
            dpreatt_ = static_cast<NativeType*>(dpreatt_tensor_->rawData());

            datt_tensor_ = std::make_shared<TensorType>( device, att_shape );
            datt_tensor_->setName( "datt_" );
            datt_ = static_cast<NativeType*>(datt_tensor_->rawData());

            dVout_tensor_ = std::make_shared<TensorType>( device, v_out_shape );
            dVout_tensor_->setName( "dVout_" );
            dVout_ = static_cast<NativeType*>(dVout_tensor_->rawData());
        }

        void buildCublasLtPlans()
        {
            // REVIEW: This is poor code. Must be based on NativeType and likely done in build_XXX_plan<NativeType>() functions.
            
            cudaDataType_t cuda_data_type = getCudaDataType();
            cublasComputeType_t compute_type;
            cudaDataType_t scale_type;

            getComputeTypes( compute_type, scale_type );

            qk_score_plan_ = Detail::build_qk_score_plan<NativeType>(
                cublaslt_handle_,
                B_, NH_, T_, HS_,
                cuda_data_type,
                compute_type,
                scale_type );

            att_value_plan_ = Detail::build_att_value_plan<NativeType>(
                cublaslt_handle_,
                B_, NH_, T_, HS_,
                cuda_data_type,
                compute_type,
                scale_type );

            backward_v_plan_ = Detail::build_backward_v_plan<NativeType>(
                cublaslt_handle_,
                B_, NH_, T_, HS_,
                cuda_data_type,
                compute_type,
                scale_type );

            backward_att_plan_ = Detail::build_backward_att_plan<NativeType>(
                cublaslt_handle_,
                B_, NH_, T_, HS_,
                cuda_data_type,
                compute_type,
                scale_type );

            backward_q_plan_ = Detail::build_backward_q_plan<NativeType>(
                cublaslt_handle_,
                B_, NH_, T_, HS_,
                cuda_data_type,
                compute_type,
                scale_type );

            backward_k_plan_ = Detail::build_backward_k_plan<NativeType>(
                cublaslt_handle_,
                B_, NH_, T_, HS_,
                cuda_data_type,
                compute_type,
                scale_type );
        }

        cudaDataType_t getCudaDataType() const
        {
            if constexpr ( std::is_same_v<NativeType, float> )
            {
                return CUDA_R_32F;
            }
            else if constexpr ( std::is_same_v<NativeType, half> )
            {
                return CUDA_R_16F;
            }
        }

        void getComputeTypes( cublasComputeType_t& compute_type, cudaDataType_t& scale_type ) const
        {
            scale_type = CUDA_R_32F;

            switch ( precision_policy_ )
            {
                case ComputePrecision::Policy::Native:
                case ComputePrecision::Policy::Accuracy:
                    if constexpr ( std::is_same_v<NativeType, half> )
                    {
                        compute_type = CUBLAS_COMPUTE_16F;
                    }
                    else
                    {
                        compute_type = CUBLAS_COMPUTE_32F;
                    }
                    break;

                case ComputePrecision::Policy::Performance:
                case ComputePrecision::Policy::Auto:
                default:
                    if constexpr ( std::is_same_v<NativeType, half> )
                    {
                        compute_type = CUBLAS_COMPUTE_32F_FAST_16F;
                    }
                    else
                    {
                        compute_type = CUBLAS_COMPUTE_32F;
                    }
                    break;
            }
        }
    };

    export class CudaAttentionOpRegistrar
    {
    public:
        static void registerOperations()
        {
            const std::string opName = "AttentionOp";

            OperationRegistry::instance().registerUnaryOperation<DeviceType::Cuda, TensorDataType::FP32, TensorDataType::FP32>(
                opName,
                []( IExecutionContext* context,
                    const ComponentConfig& config ) -> std::shared_ptr<UnaryOperation<DeviceType::Cuda, TensorDataType::FP32>>
                {
                    const auto& attentionConfig = static_cast<const AttentionConfig&>(config);
                    return std::make_shared<CudaAttentionOp<TensorDataType::FP32>>( context, attentionConfig );
                }
            );

            OperationRegistry::instance().registerUnaryOperation<DeviceType::Cuda, TensorDataType::FP16, TensorDataType::FP16>(
                opName,
                []( IExecutionContext* context,
                    const ComponentConfig& config ) -> std::shared_ptr<UnaryOperation<DeviceType::Cuda, TensorDataType::FP16>>
                {
                    const auto& attentionConfig = static_cast<const AttentionConfig&>(config);
                    return std::make_shared<CudaAttentionOp<TensorDataType::FP16>>( context, attentionConfig );
                }
            );
        }
    };
}