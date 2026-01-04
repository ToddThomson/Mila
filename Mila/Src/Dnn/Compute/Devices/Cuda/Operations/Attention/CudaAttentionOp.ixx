/**
 * @file CudaAttentionOp.ixx
 * @brief CUDA implementation of Multi-Head Attention with column-major cuBLASLt optimization.
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
         * @brief Build cuBLASLt plan for Q·K^T attention score computation.
         *
         * Each of Q, K, V shape: [B, NH, HS, T] in column-major layout
         * 
         * Column-major storage: Q[B, NH, HS, T] and K[B, NH, HS, T]
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
            const long long strideA = static_cast<long long>(head_size) * static_cast<long long>(seq_length);  // TJT: checked
            const long long strideB = static_cast<long long>(head_size) * static_cast<long long>(seq_length);  // TJT: checked
            const long long strideC = static_cast<long long>(seq_length) * static_cast<long long>(seq_length); // TJT: checked

            auto plan = build_strided_plan<TNative>(
                handle,
                head_size, seq_length, head_size, strideA,  // Q: (A_rows=HS, A_cols=T, ldA=HS, strideA)
                head_size, seq_length, head_size, strideB,  // K: (B_rows=HS, B_cols=T, ldB=HS, strideB)
                seq_length, seq_length, seq_length, strideC, // preatt: (C_rows=T, C_cols=T, ldC=T, strideC)
                CUBLAS_OP_T, CUBLAS_OP_N,                    // Q^T @ K
                compute_type,
                scale_type,
                cuda_data_type,
                batch_count,
                false );

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
            const long long strideA = static_cast<long long>(seq_length) * static_cast<long long>(seq_length);
            const long long strideB = static_cast<long long>(head_size) * static_cast<long long>(seq_length);
            const long long strideC = static_cast<long long>(head_size) * static_cast<long long>(seq_length);  // HS * T

            auto plan = build_strided_plan<TNative>(
                handle,
                seq_length, seq_length, seq_length, strideA,   // att: [T, T]
                head_size, seq_length, head_size, strideB,     // V: [HS, T]
                head_size, seq_length, head_size, strideC,     // CHANGED: [HS, T] with ldc=HS
                CUBLAS_OP_N, CUBLAS_OP_T,
                compute_type,
                scale_type,
                cuda_data_type,
                batch_count,
                false );

            if ( !plan.isValid() || !plan.has_algorithm )
            {
                Utils::Logger::warning( "cuBLASLt Att-Value plan built without algorithm (will use default)" );
            }

            return plan;
        }

        /**
         * @brief Build cuBLASLt plan for backward dV computation.
         *
         * Column-major storage: Att[B, NH, T, T] and dvaccum[B, NH, HS, T]
         * Mathematical operation: dV[T, HS] = Att^T[T, T] @ dvaccum[T, HS]
         * Rearranged: dV = dvaccum @ Att (since (A^T @ B)^T = B^T @ A)
         *
         * cuBLASLt column-major semantics:
         * - Matrix A (dvaccum): stored with HS varying fastest, so rows=HS, cols=T, ldA=HS
         * - Matrix B (Att): stored with T varying fastest, so rows=T, cols=T, ldB=T
         * - Operation: C[HS,T] = A[HS,T] @ B^T[T,T] using opA=N, opB=T
         * - Result C: rows=HS, cols=T, ldC=HS
         *
         * Strided-batched configuration:
         * - Batch count: B * NH
         * - A stride: HS*T elements per batch
         * - B stride: T*T elements per batch
         * - C stride: HS*T elements per batch
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
            const int batch_count = batch_size * num_heads;
            const long long strideA = static_cast<long long>(head_size) * static_cast<long long>(seq_length);
            const long long strideB = static_cast<long long>(seq_length) * static_cast<long long>(seq_length);
            const long long strideC = static_cast<long long>(head_size) * static_cast<long long>(seq_length);

            auto plan = build_strided_plan<TNative>(
                handle,
                head_size, seq_length, head_size, strideA,
                seq_length, seq_length, seq_length, strideB,
                head_size, seq_length, head_size, strideC,
                CUBLAS_OP_N, CUBLAS_OP_T,
                compute_type,
                scale_type,
                cuda_data_type,
                batch_count,
                false );

            if ( !plan.isValid() || !plan.has_algorithm )
            {
                Utils::Logger::warning( "cuBLASLt backward dV plan built without algorithm (will use default)" );
            }

            return plan;
        }

        /**
         * @brief Build cuBLASLt plan for backward dAtt computation.
         *
         * Column-major storage: dvaccum[B, NH, HS, T] and V[B, NH, HS, T]
         * Mathematical operation: dAtt[T, T] = dvaccum[T, HS] @ V^T[HS, T]
         *
         * cuBLASLt column-major semantics:
         * - Matrix A (dvaccum): stored with HS varying fastest, so rows=HS, cols=T, ldA=HS
         * - Matrix B (V): stored with HS varying fastest, so rows=HS, cols=T, ldB=HS
         * - Operation: C[T,T] = A^T[T,HS] @ B[HS,T] using opA=T, opB=N
         * - Result C: rows=T, cols=T, ldC=T
         *
         * Strided-batched configuration:
         * - Batch count: B * NH
         * - A stride: HS*T elements per batch
         * - B stride: HS*T elements per batch
         * - C stride: T*T elements per batch
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
            const int batch_count = batch_size * num_heads;
            const long long strideA = static_cast<long long>(head_size) * static_cast<long long>(seq_length);
            const long long strideB = static_cast<long long>(head_size) * static_cast<long long>(seq_length);
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
                batch_count,
                false );

            if ( !plan.isValid() || !plan.has_algorithm )
            {
                Utils::Logger::warning( "cuBLASLt backward dAtt plan built without algorithm (will use default)" );
            }

            return plan;
        }

        /**
         * @brief Build cuBLASLt plan for backward dQ computation.
         *
         * Column-major storage: dPreatt[B, NH, T, T] and K[B, NH, HS, T]
         * Mathematical operation: dQ[T, HS] = dPreatt[T, T] @ K[T, HS]
         * Rearranged: dQ = K @ dPreatt^T (exploiting column-major for efficiency)
         *
         * cuBLASLt column-major semantics:
         * - Matrix A (K): stored with HS varying fastest, so rows=HS, cols=T, ldA=HS
         * - Matrix B (dPreatt): stored with T varying fastest, so rows=T, cols=T, ldB=T
         * - Operation: C[HS,T] = A[HS,T] @ B^T[T,T] using opA=N, opB=T
         * - Result C: rows=HS, cols=T, ldC=HS
         *
         * Strided-batched configuration:
         * - Batch count: B * NH
         * - A stride: HS*T elements per batch
         * - B stride: T*T elements per batch
         * - C stride: HS*T elements per batch
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
            const int batch_count = batch_size * num_heads;
            const long long strideA = static_cast<long long>(head_size) * static_cast<long long>(seq_length);
            const long long strideB = static_cast<long long>(seq_length) * static_cast<long long>(seq_length);
            const long long strideC = static_cast<long long>(head_size) * static_cast<long long>(seq_length);

            auto plan = build_strided_plan<TNative>(
                handle,
                head_size, seq_length, head_size, strideA,
                seq_length, seq_length, seq_length, strideB,
                head_size, seq_length, head_size, strideC,
                CUBLAS_OP_N, CUBLAS_OP_T,
                compute_type,
                scale_type,
                cuda_data_type,
                batch_count,
                false );

            if ( !plan.isValid() || !plan.has_algorithm )
            {
                Utils::Logger::warning( "cuBLASLt backward dQ plan built without algorithm (will use default)" );
            }

            return plan;
        }

        /**
         * @brief Build cuBLASLt plan for backward dK computation.
         *
         * Column-major storage: dPreatt[B, NH, T, T] and Q[B, NH, HS, T]
         * Mathematical operation: dK[T, HS] = dPreatt^T[T, T] @ Q[T, HS]
         * Rearranged: dK = Q @ dPreatt (since (A^T @ B)^T = B^T @ A)
         *
         * cuBLASLt column-major semantics:
         * - Matrix A (Q): stored with HS varying fastest, so rows=HS, cols=T, ldA=HS
         * - Matrix B (dPreatt): stored with T varying fastest, so rows=T, cols=T, ldB=T
         * - Operation: C[HS,T] = A[HS,T] @ B[T,T] using opA=N, opB=N
         * - Result C: rows=HS, cols=T, ldC=HS
         *
         * Strided-batched configuration:
         * - Batch count: B * NH
         * - A stride: HS*T elements per batch
         * - B stride: T*T elements per batch
         * - C stride: HS*T elements per batch
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
            const int batch_count = batch_size * num_heads;
            const long long strideA = static_cast<long long>(head_size) * static_cast<long long>(seq_length);
            const long long strideB = static_cast<long long>(seq_length) * static_cast<long long>(seq_length);
            const long long strideC = static_cast<long long>(head_size) * static_cast<long long>(seq_length);

            auto plan = build_strided_plan<TNative>(
                handle,
                head_size, seq_length, head_size, strideA,
                seq_length, seq_length, seq_length, strideB,
                head_size, seq_length, head_size, strideC,
                CUBLAS_OP_N, CUBLAS_OP_N,
                compute_type,
                scale_type,
                cuda_data_type,
                batch_count,
                false );

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
            validateInputShape( input_shape );

            batch_size_ = static_cast<int>(input_shape[ 0 ]);
            seq_length_ = static_cast<int>(input_shape[ 1 ]);
            qkv_dim_ = static_cast<int>(input_shape[ 2 ]);

            embedding_dim_ = qkv_dim_ / 3;
            num_heads_ = config_.getNumHeads();
            head_size_ = embedding_dim_ / num_heads_;

            if ( embedding_dim_ % num_heads_ != 0 )
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
            const float scale = 1.0f / sqrtf( static_cast<float>(head_size_) );

            // Permute QKV from [B, T, 3*C] to separate Q, K, V with shape [B, NH, HS, T]
            // in column-major layout for efficient cuBLASLt matmul
            Detail::cuda_mha_kernels<NativeType>::permute_qkv(
                q_, k_, v_,
                X,
                batch_size_, seq_length_, num_heads_, head_size_,
                stream );

            // Ensure permutation completed before inspecting device buffers
            context_->synchronize();
            {
                // We have verified Q, K, V correctness via unit tests already.

                // Dump Q, K, V using CublasLtHelpers debug utility for column-major tensors.
                // Shapes: [B, NH, rows=HS, cols=T]
                /*std::vector<int> qkv_shape = {
                    static_cast<int>(batch_size_),
                    static_cast<int>(num_heads_),
                    static_cast<int>(head_size_),
                    static_cast<int>(seq_length_)
                };

                std::string q_dump = dump_colmajor_tensor<NativeType>(
                    q_, qkv_shape, this->getName() + ".dbg.Q", 4, stream );

                Utils::Logger::info( this->getName() + ": dbg.Q (device dump):\n" + q_dump );

                std::string k_dump = dump_colmajor_tensor<NativeType>(
                    k_, qkv_shape, this->getName() + ".dbg.K", 4, stream );

                Utils::Logger::info( this->getName() + ": dbg.K (device dump):\n" + k_dump );

                std::string v_dump = dump_colmajor_tensor<NativeType>(
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

            // Ensure preatt is available on device before dumping
            context_->synchronize();
            {
                // TJT: We have verified preatt correctness via unit tests already.

                // Dump preatt (column-major [B, NH, T, T]) using the same helper
                /*std::vector<int> preatt_shape = {
                    static_cast<int>(batch_size_),
                    static_cast<int>(num_heads_),
                    static_cast<int>(seq_length_),
                    static_cast<int>(seq_length_)
                };

                std::string preatt_dump = dump_colmajor_tensor<NativeType>(
                    preatt_, preatt_shape, this->getName() + ".dbg.preatt", 4, stream );

                Utils::Logger::info( this->getName() + ": dbg.preatt (device dump):\n" + preatt_dump );*/
            }

            // preatt_: [B, NH, T, T] in column-major
            // att_: [B, NH, T, T] in column-major
            Detail::cuda_mha_kernels<NativeType>::softmax_forward(
                att_, 1.0f, preatt_,
                batch_size_, num_heads_, seq_length_,
                stream );

            // Ensure att is available on device before dumping
            context_->synchronize();
            {
                // TJT: We have verified att correctness via unit tests already.

                /*std::vector<int> att_shape = {
                    static_cast<int>(batch_size_),
                    static_cast<int>(num_heads_),
                    static_cast<int>(seq_length_),
                    static_cast<int>(seq_length_)
                };

                std::string att_dump = dump_colmajor_tensor<NativeType>(
                    att_, att_shape, this->getName() + ".dbg.att", 4, stream );

                Utils::Logger::info( this->getName() + ": dbg.att (device dump):\n" + att_dump );*/
            }

            execute_plan<NativeType>(
                cublaslt_handle_,
                att_value_plan_,
                &alpha,
                att_, v_,
                &beta,
                vaccum_,
                nullptr,
                stream );

            // Ensure vaccum is available on device before dumping
            context_->synchronize();
            {
                //// vaccum shape: [B, NH, HS, T] in column-major
                //std::vector<int> vaccum_shape = {
                //    static_cast<int64_t>(batch_size_),
                //    static_cast<int64_t>(num_heads_),
                //    static_cast<int64_t>(head_size_),
                //    static_cast<int64_t>(seq_length_)
                //};

                //std::string vaccum_dump = dump_colmajor_tensor<NativeType>(
                //    vaccum_, vaccum_shape, this->getName() + ".dbg.vaccum", 4, stream );

                //Utils::Logger::info( this->getName() + ": dbg.vaccum (device dump):\n" + vaccum_dump );
            }

            // vaccum_: [B, NH, HS, T] in column-major
            // Y: [B, T, C] in row-major where C = NH * HS
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
            const float scale = 1.0f / sqrtf( static_cast<float>(head_size_) );

            Detail::cuda_mha_kernels<NativeType>::unpermute_backward(
                dvaccum_, dY,
                batch_size_, seq_length_, num_heads_, head_size_,  // Correct B, T, NH, HS
                stream );

            execute_plan<NativeType>(
                cublaslt_handle_,
                backward_v_plan_,
                &alpha,
                dvaccum_, att_,
                &beta,
                dv_,
                nullptr,
                stream );

            execute_plan<NativeType>(
                cublaslt_handle_,
                backward_att_plan_,
                &alpha,
                dvaccum_, v_,
                &beta,
                datt_,
                nullptr,
                stream );

            Detail::cuda_mha_kernels<NativeType>::softmax_backward(
                dpreatt_, datt_, att_,
                1.0f,
                batch_size_, num_heads_, seq_length_,
                stream );

            execute_plan<NativeType>(
                cublaslt_handle_,
                backward_q_plan_,
                &scale,
                k_, dpreatt_,
                &beta,
                dq_,
                nullptr,
                stream );

            execute_plan<NativeType>(
                cublaslt_handle_,
                backward_k_plan_,
                &scale,
                q_, dpreatt_,
                &beta,
                dk_,
                nullptr,
                stream );

            Detail::cuda_mha_kernels<NativeType>::permute_backward(
                dX,
                dq_, dk_, dv_,
                batch_size_, seq_length_, num_heads_, head_size_,  // Output is [B, T, 3*C]
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

        std::shared_ptr<TensorType> q_tensor_;
        std::shared_ptr<TensorType> k_tensor_;
        std::shared_ptr<TensorType> v_tensor_;
        std::shared_ptr<TensorType> preatt_tensor_;
        std::shared_ptr<TensorType> att_tensor_;
        std::shared_ptr<TensorType> vaccum_tensor_;

        std::shared_ptr<TensorType> dq_tensor_;
        std::shared_ptr<TensorType> dk_tensor_;
        std::shared_ptr<TensorType> dv_tensor_;
        std::shared_ptr<TensorType> dpreatt_tensor_;
        std::shared_ptr<TensorType> datt_tensor_;
        std::shared_ptr<TensorType> dvaccum_tensor_;

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

        void allocateStateTensors()
        {
            auto device = context_->getDeviceId();

            // Each of Q, K, V shape: [B, NH, HS, T] in column-major layout
            shape_t qkv_shape = {
                static_cast<int64_t>(batch_size_),
                static_cast<int64_t>(num_heads_),
                static_cast<int64_t>(head_size_),
                static_cast<int64_t>(seq_length_)
            };

            q_tensor_ = std::make_shared<TensorType>( device, qkv_shape );
            q_tensor_->setName( "q_" );
            q_ = static_cast<NativeType*>(q_tensor_->rawData());

            k_tensor_ = std::make_shared<TensorType>( device, qkv_shape );
            k_tensor_->setName( "k_" );
            k_ = static_cast<NativeType*>(k_tensor_->rawData());

            v_tensor_ = std::make_shared<TensorType>( device, qkv_shape );
            v_tensor_->setName( "v_" );
            v_ = static_cast<NativeType*>(v_tensor_->rawData());

            // attention scores shape: [B, NH, T, T] in column-major
            shape_t att_shape = {
                static_cast<int64_t>(batch_size_),
                static_cast<int64_t>(num_heads_),
                static_cast<int64_t>(seq_length_),
                static_cast<int64_t>(seq_length_)
            };

            preatt_tensor_ = std::make_shared<TensorType>( device, att_shape );
            preatt_tensor_->setName( "preatt_" );
            preatt_ = static_cast<NativeType*>(preatt_tensor_->rawData());

            att_tensor_ = std::make_shared<TensorType>( device, att_shape );
            att_tensor_->setName( "att_" );
            att_ = static_cast<NativeType*>(att_tensor_->rawData());

            // vaccum shape: [B, NH, HS, T] in column-major
            shape_t vaccum_shape = {
                static_cast<int64_t>(batch_size_),
                static_cast<int64_t>(num_heads_),
                static_cast<int64_t>(head_size_),
                static_cast<int64_t>(seq_length_)
            };

            vaccum_tensor_ = std::make_shared<TensorType>( device, vaccum_shape );
            vaccum_tensor_->setName( "vaccum_" );
            vaccum_ = static_cast<NativeType*>(vaccum_tensor_->rawData());

            dq_tensor_ = std::make_shared<TensorType>( device, qkv_shape );
            dq_tensor_->setName( "dq_" );
            dq_ = static_cast<NativeType*>(dq_tensor_->rawData());

            dk_tensor_ = std::make_shared<TensorType>( device, qkv_shape );
            dk_tensor_->setName( "dk_" );
            dk_ = static_cast<NativeType*>(dk_tensor_->rawData());

            dv_tensor_ = std::make_shared<TensorType>( device, qkv_shape );
            dv_tensor_->setName( "dv_" );
            dv_ = static_cast<NativeType*>(dv_tensor_->rawData());

            dpreatt_tensor_ = std::make_shared<TensorType>( device, att_shape );
            dpreatt_tensor_->setName( "dpreatt_" );
            dpreatt_ = static_cast<NativeType*>(dpreatt_tensor_->rawData());

            datt_tensor_ = std::make_shared<TensorType>( device, att_shape );
            datt_tensor_->setName( "datt_" );
            datt_ = static_cast<NativeType*>(datt_tensor_->rawData());

            dvaccum_tensor_ = std::make_shared<TensorType>( device, vaccum_shape );
            dvaccum_tensor_->setName( "dvaccum_" );
            dvaccum_ = static_cast<NativeType*>(dvaccum_tensor_->rawData());
        }

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