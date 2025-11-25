/**
 * @file CudaAttentionOp.ixx
 * @brief CUDA implementation of Multi-Head Attention operation (TensorDataType-based).
 *
 * Implements forward and backward passes for scaled dot-product attention
 * with multiple heads on CUDA devices. This is a stateful operation that caches
 * attention weights for the backward pass.
 */

module;
#include <vector>
#include <memory>
#include <string>
#include <stdexcept>
#include <cstdint>
#include <cuda_fp16.h>
#include "Kernels/CudaOps.h"

export module Compute.CudaAttentionOp;

import Dnn.Components.Attention;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.TensorHostTypeMap;
import Dnn.ComponentConfig;
import Compute.Precision;
import Compute.OperationBase;
import Compute.UnaryOperation;
import Compute.OperationRegistry;
import Compute.DeviceType;
import Compute.IExecutionContext;
import Compute.ExecutionContext;
import Compute.CudaExecutionContext;
import Compute.CudaDeviceResources;
import Compute.OperationType;
import Compute.MemoryResource;
import Compute.CudaDeviceMemoryResource;
import Compute.CudaTensorDataType;
import Compute.CudaDevice;

namespace Mila::Dnn::Compute
{
    namespace Detail
    {
        /**
         * @brief CUDA kernel dispatcher for Multi-Head Attention operations.
         *
         * Specialized for float (FP32) and half (FP16) native CUDA types.
         */
        template <typename TNative>
            requires std::is_same_v<TNative, float> || std::is_same_v<TNative, half>
        struct cuda_mha_impl;

        template <>
        struct cuda_mha_impl<float>
        {
            cuda_mha_impl() = default;

            static inline void forward(
                float* out, float* qkvr, float* att,
                const float* inp,
                int B, int T, int C, int NH,
                cudaStream_t stream )
            {
                //cuda_mha_forward_fp32( out, qkvr, att, inp, B, T, C, NH, stream );
            }

            static inline void backward(
                float* dinp, float* dqkvr, float* datt,
                const float* dout, const float* inp, const float* att,
                int B, int T, int C, int NH,
                cudaStream_t stream )
            {
                // FIXME: cuda_mha_backward_fp32( dinp, dqkvr, datt, dout, inp, att, B, T, C, NH, stream );
            }
        };

        template <>
        struct cuda_mha_impl<half>
        {
            cuda_mha_impl() = default;

            static inline void forward(
                half* out, half* qkvr, half* att,
                const half* inp,
                int B, int T, int C, int NH,
                cudaStream_t stream )
            {
                // FIXME: cuda_mha_forward_fp16( out, qkvr, att, inp, B, T, C, NH, stream );
            }

            static inline void backward(
                half* dinp, half* dqkvr, half* datt,
                const half* dout, const half* inp, const half* att,
                int B, int T, int C, int NH,
                cudaStream_t stream )
            {
                // FIXME: cuda_mha_backward_fp16( dinp, dqkvr, datt, dout, inp, att, B, T, C, NH, stream );
            }
        };
    }

    using namespace Mila::Dnn;

    /**
     * @brief CUDA implementation of Multi-Head Attention using abstract TensorDataType API.
     *
     * Template parameter TPrecision selects the abstract tensor precision (e.g. FP32, FP16).
     * NativeType is the corresponding CUDA device representation for that precision.
     *
     * Design philosophy:
     * - Two-phase initialization: build() does all setup, forward()/backward() are pure dispatch
     * - Module manages attention state caching for backward pass
     * - Operation allocates backend-owned pre-attention scores and attention weights
     * - All dimension computation and validation happens once in build()
     * - Forward/backward are hot-path methods with minimal overhead
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

        CudaAttentionOp( std::shared_ptr<CudaExecutionContext> context, const AttentionConfig& config )
            : config_( config ), context_( context ), impl_()
        {
            if (!context_)
            {
                throw std::runtime_error( "CudaAttentionOp requires a CUDA execution context" );
            }

            config_.validate();
        }

        // ====================================================================
        // Parameter binding (attention has no learnable parameters)
        // ====================================================================

        void setParameters( ITensor* /*unused1*/, ITensor* /*unused2*/ ) override
        {
            // Multi-head attention has no learnable parameters
        }

        void setGradients( ITensor* /*unused1*/, ITensor* /*unused2*/ ) override
        {
            // Multi-head attention has no learnable parameters
        }

        // ====================================================================
        // Lifecycle
        // ====================================================================

        /**
         * @brief Build the operation for a concrete input shape.
         *
         * This is the COLD PATH where all setup, validation, and computation happens ONCE.
         * After build() completes, forward() and backward() become pure dispatch methods.
         *
         * Responsibilities:
         *  1. Validate input shape (must be [B, T, 3*C] for concatenated Q, K, V)
         *  2. Compute and cache kernel dispatch dimensions [B, T, C, NH]
         *  3. Allocate backend-owned device storage for attention state
         *  4. Cache all device pointers for hot-path access
         *
         * After build(), the operation is ready for zero-overhead forward/backward dispatch.
         */
        void build( const shape_t& input_shape ) override
        {
            validateInputShape( input_shape );

            cached_batch_size_ = static_cast<int>(input_shape[0]);
            cached_seq_length_ = static_cast<int>(input_shape[1]);
            cached_qkv_dim_ = static_cast<int>(input_shape[2]);  // 3 * embedding_dim

            cached_embedding_dim_ = cached_qkv_dim_ / 3;
            cached_num_heads_ = config_.getNumHeads();
            cached_head_size_ = cached_embedding_dim_ / cached_num_heads_;

            // Allocate state tensors for forward pass caching
            allocateStateTensors();

            UnaryOperationBase::build( input_shape );
        }

        // ====================================================================
        // Forward pass
        // ====================================================================

        /**
         * @brief Forward pass - HOT PATH, pure dispatch to CUDA kernel.
         *
         * All setup, validation, and dimension computation was done in build().
         * This method extracts raw pointers and dispatches directly to the kernel
         * using pre-computed cached dimensions.
         *
         * Input shape: [B, T, 3*C] containing concatenated Q, K, V
         * Output shape: [B, T, C]
         *
         * For each head h and position t:
         * 1. Compute attention scores: scores[t, t2] = Q[t] · K[t2] / sqrt(d_k)
         * 2. Apply causal mask (t2 <= t)
         * 3. Apply softmax: att[t, t2] = exp(scores[t, t2]) / sum(exp(scores[t, :]))
         * 4. Compute output: out[t] = sum(att[t, t2] * V[t2])
         *
         * Zero redundant work - maximum performance.
         */
        void forward( const ITensor& input, ITensor& output ) const override
        {
            const NativeType* X = static_cast<const NativeType*>(input.rawData());
            NativeType* Y = static_cast<NativeType*>(output.rawData());

            cudaStream_t stream = context_->getStream();

            Detail::cuda_mha_impl<NativeType>::forward(
                Y,
                qkvr_cache_,
                att_cache_,
                X,
                cached_batch_size_,
                cached_seq_length_,
                cached_embedding_dim_,
                cached_num_heads_,
                stream
            );
        }

        // ====================================================================
        // Backward pass
        // ====================================================================

        /**
         * @brief Backward pass - HOT PATH, pure dispatch to CUDA kernel.
         *
         * Similar to forward(), this method does minimal work and dispatches
         * directly to the backward kernel using cached dimensions from build().
         *
         * Uses cached pre-attention scores and attention weights from forward pass.
         */
        void backward(
            const ITensor& input,
            const ITensor& output_grad,
            ITensor& input_grad ) const override
        {
            const NativeType* X = static_cast<const NativeType*>(input.rawData());
            const NativeType* dY = static_cast<const NativeType*>(output_grad.rawData());
            NativeType* dX = static_cast<NativeType*>(input_grad.rawData());

            cudaStream_t stream = context_->getStream();

            Detail::cuda_mha_impl<NativeType>::backward(
                dX,
                dqkvr_cache_,
                datt_cache_,
                dY,
                X,
                att_cache_,
                cached_batch_size_,
                cached_seq_length_,
                cached_embedding_dim_,
                cached_num_heads_,
                stream
            );
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
        std::shared_ptr<CudaExecutionContext> context_;
        Detail::cuda_mha_impl<NativeType> impl_;

        // Cached dimension values computed once in build() for hot-path dispatch
        int cached_batch_size_{ 0 };
        int cached_seq_length_{ 0 };
        int cached_qkv_dim_{ 0 };          // 3 * embedding_dim
        int cached_embedding_dim_{ 0 };
        int cached_num_heads_{ 0 };
        int cached_head_size_{ 0 };        // embedding_dim / num_heads

        // Backend-owned device runtime state storage (raw pointers for kernel dispatch)
        NativeType* qkvr_cache_{ nullptr };     // QKV projections cache [B, T, 3*C]
        NativeType* att_cache_{ nullptr };      // Attention weights [B, NH, T, T]
        NativeType* dqkvr_cache_{ nullptr };    // QKV gradients cache [B, T, 3*C]
        NativeType* datt_cache_{ nullptr };     // Attention gradients [B, NH, T, T]

        void validateInputShape( const shape_t& input_shape ) const
        {
            if (input_shape.size() != 3)
            {
                throw std::invalid_argument(
                    "CudaAttentionOp: input must have rank 3 (batch_size, seq_length, 3*embedding_dim)" );
            }

            const int64_t expected_qkv_dim = 3 * config_.getEmbeddingDim();

            if (input_shape[2] != expected_qkv_dim)
            {
                throw std::invalid_argument(
                    "CudaAttentionOp: input last dimension must be 3*embedding_dim (Q, K, V concatenated)" );
            }
        }

        void allocateStateTensors()
        {
            auto device = context_->getDevice();

            // QKV projections cache: [B, T, 3*C]
            shape_t qkvr_shape = {
                static_cast<int64_t>(cached_batch_size_),
                static_cast<int64_t>(cached_seq_length_),
                static_cast<int64_t>(cached_qkv_dim_)
            };

            auto qkvr_storage = std::make_shared<TensorType>( device, qkvr_shape );
            qkvr_storage->setName( "qkvr_cache" );
            qkvr_cache_ = static_cast<NativeType*>(qkvr_storage->rawData());

            // Attention weights: [B, NH, T, T]
            shape_t att_shape = {
                static_cast<int64_t>(cached_batch_size_),
                static_cast<int64_t>(cached_num_heads_),
                static_cast<int64_t>(cached_seq_length_),
                static_cast<int64_t>(cached_seq_length_)
            };

            auto att_storage = std::make_shared<TensorType>( device, att_shape );
            att_storage->setName( "att_cache" );
            att_cache_ = static_cast<NativeType*>(att_storage->rawData());

            // Allocate gradient caches (same shapes as forward caches)
            auto dqkvr_storage = std::make_shared<TensorType>( device, qkvr_shape );
            dqkvr_storage->setName( "dqkvr_cache" );
            dqkvr_cache_ = static_cast<NativeType*>(dqkvr_storage->rawData());

            auto datt_storage = std::make_shared<TensorType>( device, att_shape );
            datt_storage->setName( "datt_cache" );
            datt_cache_ = static_cast<NativeType*>(datt_storage->rawData());
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
                []( std::shared_ptr<ExecutionContext<DeviceType::Cuda>> context,
                    const ComponentConfig& config )
                -> std::shared_ptr<UnaryOperation<DeviceType::Cuda, TensorDataType::FP32, TensorDataType::FP32>>
                {
                    const auto& mha_config = dynamic_cast<const AttentionConfig&>(config);
                    return std::make_shared<CudaAttentionOp<TensorDataType::FP32>>( context, mha_config );
                }
            );

            // Register FP16 version
            OperationRegistry::instance().registerUnaryOperation<DeviceType::Cuda, TensorDataType::FP16, TensorDataType::FP16>(
                opName,
                []( std::shared_ptr<ExecutionContext<DeviceType::Cuda>> context,
                    const ComponentConfig& config ) -> std::shared_ptr<UnaryOperation<DeviceType::Cuda, TensorDataType::FP16>>
                {
                    const auto& mha_config = dynamic_cast<const AttentionConfig&>(config);
                    return std::make_shared<CudaAttentionOp<TensorDataType::FP16>>( context, mha_config );
                }
            );
        }

        static inline bool isRegistered = []()
            {
                registerOperations();
                return true;
            }();
    };
}