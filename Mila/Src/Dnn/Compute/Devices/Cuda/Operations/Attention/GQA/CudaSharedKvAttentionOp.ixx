/**
 * @file CudaSharedKvAttentionOp.ixx
 * @brief Single-query attention over a KV cache another layer owns, reading it and writing nothing.
 */

module;
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <format>
#include <stdexcept>
#include <cstdint>
#include "Kernels/CudaGqa.cuh"
#include "../../../Kernels/DecodePosition.cuh"

export module Compute.CudaSharedKvAttentionOp;

import Dnn.Components.GqaConfig;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.Component;
import Compute.OperationBase;
import Compute.OperationType;
import Compute.DeviceAllocation;
import Compute.DeviceType;
import Compute.IExecutionContext;
import Compute.ExecutionContext;
import Compute.CudaDeviceMemoryResource;
import Compute.KvCacheView;

namespace Mila::Dnn::Compute::Cuda::Gqa
{
    using namespace Mila::Dnn;

    /**
     * @brief One decode step's attention for a layer that projects queries only (Gemma4Mtp.md 4.3).
     *
     * The query sits at the context's decode position p and attends the keys the viewed cache holds at positions
     * up to p - 1, within its window: a one-thread launch derives p - 1 on the device each step, so the step stays
     * a pure function of device memory and replays (DecodeGraph.md). The fused decode kernels read the cache in
     * either format, BF16 or FP8, as the layer that wrote it does. BF16 only, as those kernels are.
     *
     * The geometry is the reader's: query heads, head size and scale from its config; key/value heads and window
     * must equal the cache's, which decode() checks against the view.
     */
    export template<TensorDataType TPrecision>
        requires ( TPrecision == TensorDataType::BF16 )
    class CudaSharedKvAttentionOp : public Operation<DeviceType::Cuda, TPrecision>
    {
    public:
        using MR = CudaDeviceMemoryResource;
        using CudaExecutionContext = ExecutionContext<DeviceType::Cuda>;
        using ConfigType = GqaConfig;

        CudaSharedKvAttentionOp( IExecutionContext* context, const GqaConfig& config )
            : context_( validateExecutionContext_<DeviceType::Cuda>( context, "CudaSharedKvAttentionOp" ) )
            , config_( config )
        {
            config_.validate();
        }

        /**
         * @brief Size for a batch and the longest context a query can attend: [B, T, num_heads * head_size].
         *
         * @throws std::invalid_argument When the fused decode kernel does not serve the geometry.
         */
        void build( const BuildContext& context ) override
        {
            validateInputShape( context.inputShape() );

            batch_ = static_cast<int>( context.inputShape()[ 0 ] );
            context_length_ = static_cast<int>( context.inputShape()[ 1 ] );

            key_end_ = std::make_unique<Tensor<TensorDataType::INT32, MR>>(
                context_->getDeviceId(), shape_t{ 1 }, "shared_kv_attention.key_end" );
            state_memory_size_ = occupiedTensorBytes( *key_end_ );

            Operation<DeviceType::Cuda, TPrecision>::build( context );
        }

        /**
         * @brief Attend `cache` with the query `q` [B, 1, num_heads * head_size] into `output`, same shape.
         *
         * @throws std::invalid_argument When the cache's key/value heads, head size or window differ from the config.
         */
        void decode( const ITensor& q, const KvCacheView& cache, ITensor& output )
        {
            if ( !this->isBuilt() )
                throw std::logic_error( "CudaSharedKvAttentionOp: build() before decode()" );

            const int heads = static_cast<int>( config_.getNumHeads() );
            const int kv_heads = static_cast<int>( config_.getNumKvHeads() );
            const int head_size = static_cast<int>( config_.getHeadDim() );
            const int window = static_cast<int>( config_.getWindow() );

            if ( cache.num_kv_heads != kv_heads || cache.head_size != head_size || cache.window != window
                 || cache.keys == nullptr || cache.values == nullptr )
            {
                throw std::invalid_argument( std::format(
                    "CudaSharedKvAttentionOp: the cache holds {} key/value heads of {} in window {}, and this attention "
                    "reads {} of {} in window {}", cache.num_kv_heads, cache.head_size, cache.window, kv_heads,
                    head_size, window ) );
            }

            cudaStream_t stream = context_->getStream();
            int* key_end = static_cast<int*>( key_end_->rawData() );

            // The query's own position holds no key yet: the band ends one before it.
            cuda_offset_decode_position( key_end, context_->getDecodePosition(), -1, stream );

            // Fetched on every call: the shared scratch may be reallocated on grow.
            float* split_scratch = static_cast<float*>( context_->getDeviceScratchBuffer(
                cuda_gqa_decode_attention_scratch_bytes( batch_, heads, head_size, 1 ) ) );

            const int max_band = ( window > 0 && window < context_length_ ) ? window : context_length_;
            const auto* query = static_cast<const __nv_bfloat16*>( q.rawData() );
            auto* out = static_cast<__nv_bfloat16*>( output.rawData() );

            if ( cache.fp8 )
            {
                cuda_gqa_decode_attention_fp8( query,
                    static_cast<const __nv_fp8_e4m3*>( cache.keys ), static_cast<const __nv_fp8_e4m3*>( cache.values ),
                    cache.key_scales, cache.value_scales, out, split_scratch,
                    batch_, heads, kv_heads, head_size, cache.capacity, key_end, 1, max_band, window,
                    config_.getAttentionScale(), stream );
            }
            else
            {
                cuda_gqa_decode_attention_bf16( query,
                    static_cast<const __nv_bfloat16*>( cache.keys ), static_cast<const __nv_bfloat16*>( cache.values ),
                    out, split_scratch,
                    batch_, heads, kv_heads, head_size, cache.capacity, key_end, 1, max_band, window,
                    config_.getAttentionScale(), stream );
            }
        }

        OperationType getOperationType() const override
        {
            return OperationType::SharedKvAttentionOp;
        }

        std::string getName() const override
        {
            return "Cuda::SharedKvAttentionOp";
        }

        const GqaConfig& getConfig() const noexcept
        {
            return config_;
        }

        std::size_t getStateMemorySize() const override
        {
            return state_memory_size_;
        }

        std::size_t getRequiredStateMemorySize( const BuildContext& context ) const override
        {
            validateInputShape( context.inputShape() );

            return occupiedDeviceBytes( storageBytes<TensorDataType::INT32>( 1 ), context.getAllocationGranularity() );
        }

        std::size_t getScratchBytes() const override
        {
            return cuda_gqa_decode_attention_scratch_bytes( batch_, static_cast<int>( config_.getNumHeads() ),
                static_cast<int>( config_.getHeadDim() ), 1 );
        }

        std::size_t getRequiredScratchBytes( const BuildContext& context ) const override
        {
            validateInputShape( context.inputShape() );

            return cuda_gqa_decode_attention_scratch_bytes( static_cast<int>( context.inputShape()[ 0 ] ),
                static_cast<int>( config_.getNumHeads() ), static_cast<int>( config_.getHeadDim() ), 1 );
        }

    private:

        CudaExecutionContext* context_{ nullptr };
        GqaConfig config_;

        int batch_{ 0 };
        int context_length_{ 0 };

        std::unique_ptr<Tensor<TensorDataType::INT32, MR>> key_end_{ nullptr };
        std::size_t state_memory_size_{ 0 };

        void validateInputShape( const shape_t& shape ) const
        {
            if ( shape.size() != 3 || shape[ 2 ] != config_.getModelDim() )
            {
                throw std::invalid_argument( std::format(
                    "CudaSharedKvAttentionOp: build shape must be [B, T, {}]", config_.getModelDim() ) );
            }

            const int heads = static_cast<int>( config_.getNumHeads() );
            const int kv_heads = static_cast<int>( config_.getNumKvHeads() );

            if ( !cuda_gqa_decode_attention_supported( static_cast<int>( config_.getHeadDim() ), heads / kv_heads ) )
            {
                throw std::invalid_argument( std::format(
                    "CudaSharedKvAttentionOp: the fused decode kernel does not serve head size {} with {} query heads "
                    "per key/value head", config_.getHeadDim(), heads / kv_heads ) );
            }
        }
    };
}
