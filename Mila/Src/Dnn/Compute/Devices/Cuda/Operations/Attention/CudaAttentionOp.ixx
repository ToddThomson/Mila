/**
 * @file CudaAttentionOp.ixx
 * @brief CUDA implementation of Multi-Head Attention with two-phase cuBLASLt optimization.
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
#include "Kernels/Attention.cuh"

export module Compute.CudaAttentionOp;

import Dnn.Components.Attention;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.ComponentConfig;
import Compute.OperationBase;
import Compute.UnaryOperation;
import Compute.Precision;
import Compute.OperationRegistry;
import Compute.DeviceType;
import Compute.IExecutionContext;
import Compute.ExecutionContext;
import Compute.OperationType;
import Compute.MemoryResource;
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
         *
         * Reuses the same RAII wrapper from CublasLtHelpers for consistency.
         * Plans are built once during build() and executed many times in forward/backward.
         */
        template <typename TNative>
        using CublasLtMatMulPlan = CublasLtMatMulPlan<TNative>;

        /**
         * @brief Build cuBLASLt plan for Q·K^T attention score computation.
         *
         * Computes: preatt[B, NH, T, T] = K^T[B, NH, HS, T] @ Q[B, NH, T, HS]
         *
         * After permutation, Q and K are both [B, NH, T, HS] in memory.
         * We need to compute batched matmul: preatt = K^T @ Q for each (B, NH) pair.
         *
         * Strided-batched configuration:
         * - Batch count: B * NH
         * - A matrix (K): [HS × T] per batch, ldA=HS, strideA=T*HS elements
         * - B matrix (Q): [HS × T] per batch, ldB=HS, strideB=T*HS elements
         * - C matrix (preatt): [T × T] per batch, ldC=T, strideC=T*T elements
         * - opA = CUBLAS_OP_T (transpose K)
         * - opB = CUBLAS_OP_N (no transpose Q)
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
            const long long strideA = static_cast<long long>(seq_length) * static_cast<long long>(head_size);
            const long long strideB = static_cast<long long>(seq_length) * static_cast<long long>(head_size);
            const long long strideC = static_cast<long long>(seq_length) * static_cast<long long>(seq_length);

            auto plan = build_strided_plan<TNative>(
                handle,
                head_size, seq_length, head_size, strideA,
                head_size, seq_length, head_size, strideB,
                seq_length, seq_length, seq_length, strideC,
                CUBLAS_OP_T, CUBLAS_OP_N,
                compute_type,
                scale_type,
                cuda_data_type,
                false );

            if ( plan.isValid() && plan.has_algorithm )
            {
                //Utils::Logger::info( "cuBLASLt QK score plan built successfully" );
            }
            else
            {
                Utils::Logger::warning( "cuBLASLt QK score plan built without algorithm (will use default)" );
            }

            return plan;
        }

        /**
         * @brief Build cuBLASLt plan for Att·V value computation.
         *
         * Computes: output[B, NH, T, HS] = V[B, NH, T, HS] @ Att[B, NH, T, T]
         *
         * In column-major cuBLAS notation: output = V @ Att
         *
         * Strided-batched configuration:
         * - Batch count: B * NH
         * - A matrix (V): [HS × T] per batch, ldA=HS, strideA=T*HS elements
         * - B matrix (Att): [T × T] per batch, ldB=T, strideB=T*T elements
         * - C matrix (output): [HS × T] per batch, ldC=HS, strideC=T*HS elements
         * - opA = CUBLAS_OP_N
         * - opB = CUBLAS_OP_N
         */
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
            const long long strideA = static_cast<long long>(seq_length) * static_cast<long long>(head_size);
            const long long strideB = static_cast<long long>(seq_length) * static_cast<long long>(seq_length);
            const long long strideC = static_cast<long long>(seq_length) * static_cast<long long>(head_size);

            auto plan = build_strided_plan<TNative>(
                handle,
                head_size, seq_length, head_size, strideA,
                seq_length, seq_length, seq_length, strideB,
                head_size, seq_length, head_size, strideC,
                CUBLAS_OP_N, CUBLAS_OP_N,
                compute_type,
                scale_type,
                cuda_data_type,
                false );

            if ( plan.isValid() && plan.has_algorithm )
            {
                //Utils::Logger::info( "cuBLASLt Att-Value plan built successfully" );
            }
            else
            {
                Utils::Logger::warning( "cuBLASLt Att-Value plan built without algorithm (will use default)" );
            }

            return plan;
        }

        /**
         * @brief Build cuBLASLt plan for backward dV computation.
         *
         * Computes: dV[B, NH, T, HS] = Att^T[B, NH, T, T] @ dvaccum[B, NH, T, HS]
         *
         * Memory layout (row-major storage):
         * - Att: [T, T] with stride HS between rows ? ldA = T
         * - dvaccum: [T, HS] with stride HS between rows ? ldB = HS  
         * - dV: [T, HS] with stride HS between rows ? ldC = HS
         *
         * cuBLAS column-major interpretation:
         * - A (Att): rows=T, cols=T, ldA=T
         * - B (dvaccum): rows=HS, cols=T, ldB=HS
         * - C (dV): rows=HS, cols=T, ldC=HS
         * - opA = CUBLAS_OP_T, opB = CUBLAS_OP_N
         *
         * Strided-batched configuration:
         * - Batch count: B * NH
         * - A matrix (Att): [T × T] per batch, ldA=T, strideA=T*T elements
         * - B matrix (dvaccum): [HS × T] per batch, ldB=HS, strideB=T*HS elements
         * - C matrix (dV): [HS × T] per batch, ldC=HS, strideC=T*HS elements
         */
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
            const long long strideA = static_cast<long long>(seq_length) * static_cast<long long>(seq_length);
            const long long strideB = static_cast<long long>(seq_length) * static_cast<long long>(head_size);
            const long long strideC = static_cast<long long>(seq_length) * static_cast<long long>(head_size);

            auto plan = build_strided_plan<TNative>(
                handle,
                seq_length, seq_length, seq_length, strideA,
                seq_length, head_size, seq_length, strideB,
                seq_length, head_size, seq_length, strideC,
                CUBLAS_OP_T, CUBLAS_OP_N,
                compute_type,
                scale_type,
                cuda_data_type,
                false );

            if ( plan.isValid() && plan.has_algorithm )
            {
                //Utils::Logger::info( "cuBLASLt backward dV plan built successfully" );
            }
            else
            {
                Utils::Logger::warning( "cuBLASLt backward dV plan built without algorithm (will use default)" );
            }

            return plan;
        }

        /**
         * @brief Build cuBLASLt plan for backward dAtt computation.
         *
         * Computes: dAtt[B, NH, T, T] = dvaccum[B, NH, T, HS] @ V^T[B, NH, HS, T]
         *
         * Mathematical operation: dAtt[T, T] = dvaccum[T, HS] @ V^T[HS, T]
         *
         * In cuBLAS column-major (matrices stored transposed):
         * - dvaccum stored as [HS × T] ? interpret as [T × HS] with ldA=T when transposed
         * - V stored as [HS × T] ? interpret as [T × HS] with ldB=T when transposed
         * - dAtt stored as [T × T], ldC=T
         *
         * Strided-batched configuration:
         * - Batch count: B * NH
         * - A matrix (dvaccum): [T × HS] per batch, ldA=T, strideA=T*HS elements (transposed view)
         * - B matrix (V): [HS × T] per batch, ldB=HS, strideB=T*HS elements (will be transposed)
         * - C matrix (dAtt): [T × T] per batch, ldC=T, strideC=T*T elements
         * - opA = CUBLAS_OP_T (transpose dvaccum for correct layout)
         * - opB = CUBLAS_OP_N (V is already in correct layout for transpose effect)
         */
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
            const long long strideA = static_cast<long long>(seq_length) * static_cast<long long>(head_size);
            const long long strideB = static_cast<long long>(seq_length) * static_cast<long long>(head_size);
            const long long strideC = static_cast<long long>(seq_length) * static_cast<long long>(seq_length);

            auto plan = build_strided_plan<TNative>(
                handle,
                head_size, seq_length, head_size, strideA,
                head_size, seq_length, head_size, strideB,
                seq_length, seq_length, seq_length, strideC,
                CUBLAS_OP_T, CUBLAS_OP_N,
                compute_type,
                scale_type,
                cuda_data_type,
                false );

            if ( plan.isValid() && plan.has_algorithm )
            {
                //Utils::Logger::info( "cuBLASLt backward dAtt plan built successfully" );
            }
            else
            {
                Utils::Logger::warning( "cuBLASLt backward dAtt plan built without algorithm (will use default)" );
            }

            return plan;
        }

        /**
         * @brief Build cuBLASLt plan for backward dQ computation.
         *
         * Computes: dQ[B, NH, T, HS] = dPreatt[B, NH, T, T] @ K[B, NH, T, HS]
         *
         * Mathematical operation: dQ[T, HS] = dPreatt[T, T] @ K[T, HS]
         *
         * In cuBLAS column-major (matrices stored transposed):
         * - dPreatt stored as [T × T], ldA=T
         * - K stored as [HS × T] ? interpret as [T × HS] with ldB=T when transposed
         * - dQ stored as [HS × T] ? result is [T × HS] with ldC=T when transposed
         *
         * Strided-batched configuration:
         * - Batch count: B * NH
         * - A matrix (dPreatt): [T × T] per batch, ldA=T, strideA=T*T elements
         * - B matrix (K): [T × HS] per batch, ldB=T, strideB=T*HS elements (transposed view)
         * - C matrix (dQ): [T × HS] per batch, ldC=T, strideC=T*HS elements (transposed view)
         * - opA = CUBLAS_OP_N
         * - opB = CUBLAS_OP_T (transpose K for correct layout)
         */
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
            const long long strideA = static_cast<long long>(seq_length) * static_cast<long long>(seq_length);
            const long long strideB = static_cast<long long>(seq_length) * static_cast<long long>(head_size);
            const long long strideC = static_cast<long long>(seq_length) * static_cast<long long>(head_size);

            auto plan = build_strided_plan<TNative>(
                handle,
                seq_length, seq_length, seq_length, strideA,
                seq_length, head_size, seq_length, strideB,
                seq_length, head_size, seq_length, strideC,
                CUBLAS_OP_T, CUBLAS_OP_N,
                compute_type,
                scale_type,
                cuda_data_type,
                false );

            if ( plan.isValid() && plan.has_algorithm )
            {
                //Utils::Logger::info( "cuBLASLt backward dQ plan built successfully" );
            }
            else
            {
                Utils::Logger::warning( "cuBLASLt backward dQ plan built without algorithm (will use default)" );
            }

            return plan;
        }

        /**
         * @brief Build cuBLASLt plan for backward dK computation.
         *
         * Computes: dK[B, NH, T, HS] = dPreatt^T[B, NH, T, T] @ Q[B, NH, T, HS]
         *
         * Mathematical operation: dK[T, HS] = dPreatt^T[T, T] @ Q[T, HS]
         *
         * In cuBLAS column-major (matrices stored transposed):
         * - dPreatt stored as [T × T], ldA=T (will be transposed)
         * - Q stored as [HS × T] ? interpret as [T × HS] with ldB=T when transposed
         * - dK stored as [HS × T] ? result is [T × HS] with ldC=T when transposed
         *
         * Strided-batched configuration:
         * - Batch count: B * NH
         * - A matrix (dPreatt): [T × T] per batch, ldA=T, strideA=T*T elements (will be transposed)
         * - B matrix (Q): [T × HS] per batch, ldB=T, strideB=T*HS elements (transposed view)
         * - C matrix (dK): [T × HS] per batch, ldC=T, strideC=T*HS elements (transposed view)
         * - opA = CUBLAS_OP_T (transpose dPreatt)
         * - opB = CUBLAS_OP_T (transpose Q for correct layout)
         */
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
            const long long strideA = static_cast<long long>(seq_length) * static_cast<long long>(seq_length);
            const long long strideB = static_cast<long long>(seq_length) * static_cast<long long>(head_size);
            const long long strideC = static_cast<long long>(seq_length) * static_cast<long long>(head_size);

            auto plan = build_strided_plan<TNative>(
                handle,
                seq_length, seq_length, seq_length, strideA,
                seq_length, head_size, seq_length, strideB,
                seq_length, head_size, seq_length, strideC,
                CUBLAS_OP_T, CUBLAS_OP_N,
                compute_type,
                scale_type,
                cuda_data_type,
                false );

            if ( plan.isValid() && plan.has_algorithm )
            {
                //Utils::Logger::info( "cuBLASLt backward dK plan built successfully" );
            }
            else
            {
                Utils::Logger::warning( "cuBLASLt backward dK plan built without algorithm (will use default)" );
            }

            return plan;
        }

        /**
         * @brief CUDA kernel dispatcher for attention non-matmul operations.
         *
         * Handles permute, unpermute, softmax operations that cannot be done with cuBLASLt.
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
                int B, int NH, int T,
                cudaStream_t stream )
            {
                cuda_softmax_backward_fp32( dpreatt, datt, att, B, NH, T, stream );
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
                int B, int NH, int T,
                cudaStream_t stream )
            {
                cuda_softmax_backward_fp16( dpreatt, datt, att, B, NH, T, stream );
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
     * @brief CUDA implementation of Multi-Head Attention using two-phase cuBLASLt optimization.
     *
     * Design philosophy (matches CudaLinearOp):
     * - Two-phase initialization: build() creates cuBLASLt plans, forward()/backward() execute them
     * - All dimension computation and algorithm selection happens once in build()
     * - Forward/backward are hot-path methods with zero setup overhead
     * - cuBLASLt plans cache descriptors, layouts, and optimal algorithms
     * - Custom CUDA kernels handle permute/unpermute and softmax operations
     *
     * Forward pass:
     *  1. Permute QKV from [B, T, 3*C] to separate Q, K, V [B, NH, T, HS]
     *  2. Compute attention scores: preatt = K^T @ Q
     *  3. Apply softmax with causal masking: att = softmax(preatt / sqrt(HS))
     *  4. Compute values: output = V @ att
     *  5. Unpermute output from [B, NH, T, HS] to [B, T, C]
     *
     * Backward pass:
     *  1. Unpermute output gradient
     *  2. Compute dV = Att^T @ dOut
     *  3. Compute dAtt = dOut @ V^T
     *  4. Softmax backward: dPreatt = softmax_backward(dAtt, Att)
     *  5. Compute dQ = dPreatt @ K
     *  6. Compute dK = dPreatt^T @ Q
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
            validateInputShape( input_shape );

            batch_size_ = static_cast<int>(input_shape[ 0 ]);
            seq_length_ = static_cast<int>(input_shape[ 1 ]);
            qkv_dim_ = static_cast<int>(input_shape[ 2 ]);

            embedding_dim_ = qkv_dim_ / 3;
            num_heads_ = config_.getNumHeads();
            head_size_ = embedding_dim_ / num_heads_;

            allocateStateTensors();

            cublaslt_handle_ = context_->getCublasLtHandle();

            if ( cublaslt_handle_ == nullptr )
            {
                throw std::runtime_error(
                    "CudaAttentionOp requires cuBLASLt support. "
                    "Ensure CUDA 10.1 or newer is installed." );
            }

            precision_policy_ = config_.getPrecisionPolicy();

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

            // Step 1: Split and permute concatenated QKV input
            // Input:  X [B, T, 3*C] - concatenated Q, K, V along feature dimension
            // Output: Q, K, V each [B, NH, T, HS] - separated and reshaped for multi-head processing
            // This transforms from model layout to multi-head layout for efficient batched matmuls
            Detail::cuda_mha_kernels<NativeType>::permute_qkv(
                q_, k_, v_,
                X,
                batch_size_, seq_length_, num_heads_, head_size_,
                stream );

            // Step 2: Compute attention scores using cuBLASLt
            // Computes: preatt = K^T @ Q for each (batch, head) pair
            // Input:  K [B, NH, T, HS], Q [B, NH, T, HS]
            // Output: preatt [B, NH, T, T] - raw attention scores before softmax
            // Each position attends to all previous positions (causal masking applied in softmax)
            execute_plan<NativeType>(
                cublaslt_handle_,
                qk_score_plan_,
                &alpha,
                k_, q_,
                &beta,
                preatt_,
                nullptr,
                stream );

            // Step 3: Apply scaled softmax with causal masking
            // Computes: att = softmax(preatt / sqrt(head_size)) with mask(t2 <= t)
            // Input:  preatt [B, NH, T, T] - raw attention scores
            // Output: att [B, NH, T, T] - normalized attention probabilities
            // Scaling by 1/sqrt(HS) prevents dot products from growing too large
            // Causal mask ensures position t can only attend to positions 0..t
            const float scale = 1.0f / sqrtf( static_cast<float>(head_size_) );

            Detail::cuda_mha_kernels<NativeType>::softmax_forward(
                att_, scale, preatt_,
                batch_size_, num_heads_, seq_length_,
                stream );

            // Step 4: Apply attention to values using cuBLASLt
            // Computes: vaccum = V @ Att for each (batch, head) pair
            // Input:  V [B, NH, T, HS], Att [B, NH, T, T]
            // Output: vaccum [B, NH, T, HS] - weighted sum of values by attention
            // Each output position is a weighted combination of all value vectors
            execute_plan<NativeType>(
                cublaslt_handle_,
                att_value_plan_,
                &alpha,
                v_, att_,
                &beta,
                vaccum_,
                nullptr,
                stream );

            // Step 5: Concatenate and permute multi-head outputs
            // Input:  vaccum [B, NH, T, HS] - separate head outputs
            // Output: Y [B, T, C] where C = NH * HS - concatenated output in model layout
            // Transforms from multi-head layout back to model layout for downstream layers
            Detail::cuda_mha_kernels<NativeType>::unpermute_output(
                vaccum_, Y,
                batch_size_, seq_length_, num_heads_, head_size_,
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

            // Step 1: Unpermute output gradient from model layout to multi-head layout
            // Input:  dY [B, T, C] - gradient from downstream layers (model layout)
            // Output: dvaccum [B, NH, T, HS] - gradient reshaped for multi-head processing
            // Reverse of forward's final unpermute step
            Detail::cuda_mha_kernels<NativeType>::unpermute_backward(
                dvaccum_, dY,
                batch_size_, seq_length_, num_heads_, head_size_,
                stream );

            // Step 2: Compute gradient w.r.t. values using cuBLASLt
            // Computes: dV = Att^T @ dvaccum for each (batch, head) pair
            // Input:  Att [B, NH, T, T] (forward attention weights), dvaccum [B, NH, T, HS]
            // Output: dV [B, NH, T, HS] - gradient w.r.t. value vectors
            // Backpropagates through: vaccum = V @ Att
            execute_plan<NativeType>(
                cublaslt_handle_,
                backward_v_plan_,
                &alpha,
                att_, dvaccum_,
                &beta,
                dv_,
                nullptr,
                stream );

            // Step 3: Compute gradient w.r.t. attention weights using cuBLASLt
            // Computes: dAtt = dvaccum @ V^T for each (batch, head) pair
            // Input:  dvaccum [B, NH, T, HS], V [B, NH, T, HS] (forward values)
            // Output: dAtt [B, NH, T, T] - gradient w.r.t. attention probabilities
            // Backpropagates through: vaccum = V @ Att
            execute_plan<NativeType>(
                cublaslt_handle_,
                backward_att_plan_,
                &alpha,
                dvaccum_, v_,
                &beta,
                datt_,
                nullptr,
                stream );

            // Step 4: Backpropagate through softmax with causal mask
            // Computes: dPreatt = softmax_backward(dAtt, Att)
            // Input:  dAtt [B, NH, T, T] (gradient w.r.t. attention probabilities)
            //         Att [B, NH, T, T] (forward attention probabilities)
            // Output: dPreatt [B, NH, T, T] - gradient w.r.t. pre-softmax scores
            // Accounts for softmax Jacobian and causal masking structure
            Detail::cuda_mha_kernels<NativeType>::softmax_backward(
                dpreatt_, datt_, att_,
                batch_size_, num_heads_, seq_length_,
                stream );

            // Step 5: Compute gradient w.r.t. queries using cuBLASLt
            // Computes: dQ = dPreatt @ K for each (batch, head) pair
            // Input:  dPreatt [B, NH, T, T], K [B, NH, T, HS] (forward keys)
            // Output: dQ [B, NH, T, HS] - gradient w.r.t. query vectors
            // Backpropagates through: preatt = K^T @ Q
            execute_plan<NativeType>(
                cublaslt_handle_,
                backward_q_plan_,
                &alpha,
                dpreatt_, k_,
                &beta,
                dq_,
                nullptr,
                stream );

            // Step 6: Compute gradient w.r.t. keys using cuBLASLt
            // Computes: dK = dPreatt^T @ Q for each (batch, head) pair
            // Input:  dPreatt [B, NH, T, T], Q [B, NH, T, HS] (forward queries)
            // Output: dK [B, NH, T, HS] - gradient w.r.t. key vectors
            // Backpropagates through: preatt = K^T @ Q
            execute_plan<NativeType>(
                cublaslt_handle_,
                backward_k_plan_,
                &alpha,
                dpreatt_, q_,
                &beta,
                dk_,
                nullptr,
                stream );

            // Step 7: Permute and concatenate QKV gradients back to model layout
            // Input:  dQ, dK, dV each [B, NH, T, HS] - separate head gradients
            // Output: dX [B, T, 3*C] - concatenated gradient in model layout
            // Reverse of forward's initial QKV permutation step
            Detail::cuda_mha_kernels<NativeType>::permute_backward(
                dX,
                dq_, dk_, dv_,
                batch_size_, seq_length_, num_heads_, head_size_,
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

        int batch_size_{ 0 };
        int seq_length_{ 0 };
        int qkv_dim_{ 0 };
        int embedding_dim_{ 0 };
        int num_heads_{ 0 };
        int head_size_{ 0 };

        cublasLtHandle_t cublaslt_handle_{ nullptr };
        ComputePrecision::Policy precision_policy_;

        Detail::CublasLtMatMulPlan<NativeType> qk_score_plan_;
        Detail::CublasLtMatMulPlan<NativeType> att_value_plan_;
        Detail::CublasLtMatMulPlan<NativeType> backward_v_plan_;
        Detail::CublasLtMatMulPlan<NativeType> backward_att_plan_;
        Detail::CublasLtMatMulPlan<NativeType> backward_q_plan_;
        Detail::CublasLtMatMulPlan<NativeType> backward_k_plan_;

        NativeType* q_{ nullptr };
        NativeType* k_{ nullptr };
        NativeType* v_{ nullptr };
        NativeType* preatt_{ nullptr };
        NativeType* att_{ nullptr };
        NativeType* vaccum_{ nullptr };

        NativeType* dq_{ nullptr };
        NativeType* dk_{ nullptr };
        NativeType* dv_{ nullptr };
        NativeType* dpreatt_{ nullptr };
        NativeType* datt_{ nullptr };
        NativeType* dvaccum_{ nullptr };

        /**
         * @brief Validate input tensor shape for attention.
         *
         * Expects input shape to be [batch_size, seq_length, 3*embedding_dim].
         *
         * @param input_shape Shape of the input tensor.
         *
         * @throws std::invalid_argument if shape has incorrect rank or last dimension.
         */
        void validateInputShape( const shape_t& input_shape ) const
        {
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

        /**
         * @brief Allocate device tensors used as intermediate state during forward/backward.
         *
         * Creates device tensors for Q, K, V, attention scores, probabilities and all
         * intermediate gradients.
         * Ownership is held by the local shared_ptr instances to keep lifetime tied to the
         * operation's state objects managed by the framework Tensor class.
         */
        void allocateStateTensors()
        {
            auto device = context_->getDeviceId();

            shape_t qkv_shape = {
                static_cast<int64_t>(batch_size_),
                static_cast<int64_t>(num_heads_),
                static_cast<int64_t>(seq_length_),
                static_cast<int64_t>(head_size_)
            };

            auto q_tensor = std::make_shared<TensorType>( device, qkv_shape );
            q_tensor->setName( "q_" );
            q_ = static_cast<NativeType*>(q_tensor->rawData());

            auto k_tensor = std::make_shared<TensorType>( device, qkv_shape );
            k_tensor->setName( "k_" );
            k_ = static_cast<NativeType*>(k_tensor->rawData());

            auto v_tensor = std::make_shared<TensorType>( device, qkv_shape );
            v_tensor->setName( "v_" );
            v_ = static_cast<NativeType*>(v_tensor->rawData());

            shape_t att_shape = {
                static_cast<int64_t>(batch_size_),
                static_cast<int64_t>(num_heads_),
                static_cast<int64_t>(seq_length_),
                static_cast<int64_t>(seq_length_)
            };

            auto preatt_tensor = std::make_shared<TensorType>( device, att_shape );
            preatt_tensor->setName( "preatt_" );
            preatt_ = static_cast<NativeType*>(preatt_tensor->rawData());

            auto att_tensor = std::make_shared<TensorType>( device, att_shape );
            att_tensor->setName( "att_" );
            att_ = static_cast<NativeType*>(att_tensor->rawData());

            auto vaccum_tensor = std::make_shared<TensorType>( device, qkv_shape );
            vaccum_tensor->setName( "vaccum_" );
            vaccum_ = static_cast<NativeType*>(vaccum_tensor->rawData());

            auto dq_tensor = std::make_shared<TensorType>( device, qkv_shape );
            dq_tensor->setName( "dq_" );
            dq_ = static_cast<NativeType*>(dq_tensor->rawData());

            auto dk_tensor = std::make_shared<TensorType>( device, qkv_shape );
            dk_tensor->setName( "dk_" );
            dk_ = static_cast<NativeType*>(dk_tensor->rawData());

            auto dv_tensor = std::make_shared<TensorType>( device, qkv_shape );
            dv_tensor->setName( "dv_" );
            dv_ = static_cast<NativeType*>(dv_tensor->rawData());

            auto dpreatt_tensor = std::make_shared<TensorType>( device, att_shape );
            dpreatt_tensor->setName( "dpreatt_" );
            dpreatt_ = static_cast<NativeType*>(dpreatt_tensor->rawData());

            auto datt_tensor = std::make_shared<TensorType>( device, att_shape );
            datt_tensor->setName( "datt_" );
            datt_ = static_cast<NativeType*>(datt_tensor->rawData());

            auto dvaccum_tensor = std::make_shared<TensorType>( device, qkv_shape );
            dvaccum_tensor->setName( "dvaccum_" );
            dvaccum_ = static_cast<NativeType*>(dvaccum_tensor->rawData());
        }

        /**
         * @brief Build all required cuBLASLt plans once per input configuration.
         *
         * This caches plan descriptors and algorithms so forward/backward are zero-overhead.
         */
        void buildCublasLtPlans()
        {
            cudaDataType_t cuda_data_type = getCudaDataType();
            cublasComputeType_t compute_type;
            cudaDataType_t scale_type;

            getComputeTypes( compute_type, scale_type );

            qk_score_plan_ = Detail::build_qk_score_plan<NativeType>(
                cublaslt_handle_,
                batch_size_,
                num_heads_,
                seq_length_,
                head_size_,
                cuda_data_type,
                compute_type,
                scale_type );

            att_value_plan_ = Detail::build_att_value_plan<NativeType>(
                cublaslt_handle_,
                batch_size_,
                num_heads_,
                seq_length_,
                head_size_,
                cuda_data_type,
                compute_type,
                scale_type );

            backward_v_plan_ = Detail::build_backward_v_plan<NativeType>(
                cublaslt_handle_,
                batch_size_,
                num_heads_,
                seq_length_,
                head_size_,
                cuda_data_type,
                compute_type,
                scale_type );

            backward_att_plan_ = Detail::build_backward_att_plan<NativeType>(
                cublaslt_handle_,
                batch_size_,
                num_heads_,
                seq_length_,
                head_size_,
                cuda_data_type,
                compute_type,
                scale_type );

            backward_q_plan_ = Detail::build_backward_q_plan<NativeType>(
                cublaslt_handle_,
                batch_size_,
                num_heads_,
                seq_length_,
                head_size_,
                cuda_data_type,
                compute_type,
                scale_type );

            backward_k_plan_ = Detail::build_backward_k_plan<NativeType>(
                cublaslt_handle_,
                batch_size_,
                num_heads_,
                seq_length_,
                head_size_,
                cuda_data_type,
                compute_type,
                scale_type );
        }

        /**
         * @brief Get CUDA data type for the configured template precision.
         *
         * @return cudaDataType_t CUDA runtime type for elements (CUDA_R_32F or CUDA_R_16F)
         */
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

        /**
         * @brief Select cuBLAS compute and scaling types according to precision policy.
         *
         * @param[out] compute_type Selected cublasComputeType_t
         * @param[out] scale_type Selected cudaDataType_t used for scaling parameters
         */
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