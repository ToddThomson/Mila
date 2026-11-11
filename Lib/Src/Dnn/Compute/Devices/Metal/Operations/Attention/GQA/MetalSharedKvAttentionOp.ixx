/**
 * @file MetalSharedKvAttentionOp.ixx
 * @brief Single-query attention over a KV cache another layer owns, reading it and writing nothing, on Metal.
 */

export module Compute.MetalSharedKvAttentionOp;

import std;
import Compute.DeviceAllocation;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.IExecutionContext;
import Compute.KvCacheView;
import Compute.MetalExecutionContext;
import Compute.MetalKernelLibrary;
import Compute.MetalKernelLimits;
import Compute.MetalTensorDataTypeMap;
import Compute.OperationBase;
import Compute.OperationType;
import Dnn.Component;
import Dnn.Components.GqaConfig;
import Dnn.ITensor;
import Dnn.TensorDataType;
import Dnn.TensorTypes;

namespace Mila::Dnn::Compute::Metal::Gqa
{
    /**
     * @brief One decode step's attention for a layer that projects queries only (`Gemma4Mtp.md` 4.3, `MetalBackend.md` 4.6).
     *
     * The query sits at the context's decode position p and attends the keys the viewed cache holds at positions up to p - 1,
     * within its window. It runs the owning layer's decode kernels, partial pass and merge, with the band ending one before the
     * position the kernels read from the context, so the step stays a pure function of device memory. The cache's format, the
     * compute type or FP8, is read from the view at each call, as the layer that wrote it holds it.
     *
     * The geometry is the reader's: query heads, head size and scale from its config; key/value heads and window must equal the
     * cache's, which decode() checks against the view.
     *
     * @tparam TPrecision FP16 or BF16.
     */
    export template<TensorDataType TPrecision>
        requires ( TPrecision == TensorDataType::FP16 || TPrecision == TensorDataType::BF16 )
    class MetalSharedKvAttentionOp : public Operation<DeviceType::Metal, TPrecision>
    {
    public:

        using ConfigType = GqaConfig;

        MetalSharedKvAttentionOp( IExecutionContext* context, const GqaConfig& config )
            : context_( validateExecutionContext_<DeviceType::Metal>( context, "MetalSharedKvAttentionOp" ) ),
              config_( config )
        {
            config_.validate();

            // The cache, and so whether it is FP8, is seen only at decode(): both partial passes are resolved.
            MetalKernelLibrary& kernels = context_->device().kernels();

            partial_pipeline_ = kernels.pipeline( decodeKernelName( "gqa_decode_partial" ) );
            partial_fp8_pipeline_ = kernels.pipeline( decodeKernelName( "gqa_decode_partial_fp8" ) );
            merge_pipeline_ = kernels.pipeline( decodeKernelName( "gqa_decode_merge" ) );
        }

        /**
         * @brief Size for a batch: [B, T, num_heads * head_size], T the longest context a query can attend.
         *
         * @throws std::invalid_argument When a decode threadgroup cannot hold the query heads of one key/value head.
         */
        void build( const BuildContext& context ) override
        {
            validateInputShape( context.inputShape() );

            batches_ = narrowToKernelIndex( context.inputShape()[0] );

            // The cache, and so whether it is FP8, is seen only at decode(): both kernels must take the threadgroup.
            for ( const std::string_view family : { std::string_view( "gqa_decode_partial" ), std::string_view( "gqa_decode_partial_fp8" ) } )
            {
                const MetalKernelLimits limits = context_->device().kernels().limits( decodeKernelName( family ) );
                const std::size_t threads = static_cast<std::size_t>( queryHeadsPerKvHead() ) * limits.thread_execution_width;

                if ( threads > limits.max_total_threads_per_threadgroup )
                {
                    throw std::invalid_argument( std::format(
                        "MetalSharedKvAttentionOp: {} query heads to a key/value head take {} threads, past the {} this device's {} takes",
                        queryHeadsPerKvHead(), threads, limits.max_total_threads_per_threadgroup, decodeKernelName( family ) ) );
                }
            }

            Operation<DeviceType::Metal, TPrecision>::build( context );
        }

        /**
         * @brief Attend `cache` with the query `q` [B, 1, num_heads * head_size] into `output`, same shape.
         *
         * @throws std::invalid_argument When the cache's key/value heads, head size or window differ from the config.
         */
        void decode( const ITensor& q, const KvCacheView& cache, ITensor& output ) const
        {
            if ( !this->isBuilt() )
            {
                throw std::logic_error( "MetalSharedKvAttentionOp: build() before decode()" );
            }

            const int kv_heads = narrowToKernelIndex( config_.getNumKvHeads() );
            const int head_size = narrowToKernelIndex( config_.getHeadDim() );
            const int window = narrowToKernelIndex( config_.getWindow() );

            if ( cache.num_kv_heads != kv_heads || cache.head_size != head_size || cache.window != window
                 || cache.keys == nullptr || cache.values == nullptr )
            {
                throw std::invalid_argument( std::format(
                    "MetalSharedKvAttentionOp: the cache holds {} key/value heads of {} in window {}, and this attention reads {} of {} in "
                    "window {}", cache.num_kv_heads, cache.head_size, cache.window, kv_heads, head_size, window ) );
            }

            const int heads = narrowToKernelIndex( config_.getNumHeads() );
            const Geometry geometry{
                static_cast<std::uint32_t>( batches_ ), 1u, static_cast<std::uint32_t>( heads ), static_cast<std::uint32_t>( kv_heads ),
                static_cast<std::uint32_t>( head_size ), static_cast<std::uint32_t>( cache.capacity ), static_cast<std::uint32_t>( window ),
                0u, config_.getAttentionScale(), static_cast<std::uint32_t>( kDecodeBlocks ), 0u, 1u };

            const std::size_t slots = static_cast<std::size_t>( batches_ * heads ) * kDecodeBlocks;

            // Fetched on every call: the shared scratch may be reallocated on grow.
            auto* const partial_output = static_cast<float*>( context_->getDeviceScratchBuffer( scratchBytesFor( batches_ ) ) );
            float* const partial_largest = partial_output + slots * static_cast<std::size_t>( head_size );
            float* const partial_sum = partial_largest + slots;

            const void* const decode_position = context_->decodePositionAddress();
            const auto constants = std::as_bytes( std::span( &geometry, 1 ) );
            const std::size_t kv_heads_of_batches = static_cast<std::size_t>( batches_ * kv_heads );
            const std::size_t query_heads = static_cast<std::size_t>( queryHeadsPerKvHead() );

            if ( cache.fp8 )
            {
                context_->dispatchThreadgroups( partial_fp8_pipeline_,
                    std::array<const void*, 9>{ q.rawData(), cache.keys, cache.values, cache.key_scales, cache.value_scales, partial_output,
                        partial_largest, partial_sum, decode_position },
                    constants, kDecodeBlocks, kv_heads_of_batches, query_heads );
            } else {
                context_->dispatchThreadgroups( partial_pipeline_,
                    std::array<const void*, 7>{ q.rawData(), cache.keys, cache.values, partial_output, partial_largest, partial_sum,
                        decode_position },
                    constants, kDecodeBlocks, kv_heads_of_batches, query_heads );
            }

            context_->dispatchThreadgroups( merge_pipeline_,
                std::array<const void*, 4>{ partial_output, partial_largest, partial_sum, output.rawData() },
                constants, static_cast<std::size_t>( batches_ * heads ), 1 );
        }

        [[nodiscard]] OperationType getOperationType() const override
        {
            return OperationType::SharedKvAttentionOp;
        }

        [[nodiscard]] std::string getName() const override
        {
            return "Metal.SharedKvAttention";
        }

        [[nodiscard]] const GqaConfig& getConfig() const noexcept
        {
            return config_;
        }

        /// None: the band's end is derived in the kernel, and the cache is the owning layer's.
        [[nodiscard]] std::size_t getStateMemorySize() const override
        {
            return 0;
        }

        [[nodiscard]] std::size_t getRequiredStateMemorySize( const BuildContext& context ) const override
        {
            validateInputShape( context.inputShape() );

            return 0;
        }

        /// A step's partials, from the context's scratch.
        [[nodiscard]] std::size_t getScratchBytes() const override
        {
            return this->isBuilt() ? scratchBytesFor( batches_ ) : 0;
        }

        [[nodiscard]] std::size_t getRequiredScratchBytes( const BuildContext& context ) const override
        {
            validateInputShape( context.inputShape() );

            return scratchBytesFor( context.inputShape()[0] );
        }

    private:

        // The owning layer's split (MetalGqaOp), so that the same kernels and merge serve both.
        static constexpr std::size_t kDecodeBlocks = 32;

        // The kernels' GqaGeometry, field for field.
        struct Geometry
        {
            std::uint32_t batches;
            std::uint32_t tokens;
            std::uint32_t heads;
            std::uint32_t kv_heads;
            std::uint32_t head_size;
            std::uint32_t capacity;
            std::uint32_t window;
            std::uint32_t position_offset;
            float attention_scale;
            std::uint32_t decode_blocks;
            std::uint32_t score_stride;
            std::uint32_t keys_before;
        };

        MetalExecutionContext* context_{ nullptr };
        GqaConfig config_;
        MetalPipeline partial_pipeline_{ nullptr };
        MetalPipeline partial_fp8_pipeline_{ nullptr };
        MetalPipeline merge_pipeline_{ nullptr };

        int batches_{ 0 };

        [[nodiscard]] int queryHeadsPerKvHead() const
        {
            return narrowToKernelIndex( config_.getNumHeads() / config_.getNumKvHeads() );
        }

        [[nodiscard]] std::size_t scratchBytesFor( dim_t batches ) const
        {
            return static_cast<std::size_t>( batches * config_.getNumHeads() ) * kDecodeBlocks
                * static_cast<std::size_t>( config_.getHeadDim() + 2 ) * sizeof( float );
        }

        [[nodiscard]] std::string decodeKernelName( std::string_view family ) const
        {
            const dim_t head_size = config_.getHeadDim();
            const bool own = head_size == 64 || head_size == 128 || head_size == 256 || head_size == 512;

            return own
                ? std::format( "{}_{}_{}", family, MetalTensorDataTypeMap<TPrecision>::kernel_type, head_size )
                : std::format( "{}_{}_any", family, MetalTensorDataTypeMap<TPrecision>::kernel_type );
        }

        void validateInputShape( const shape_t& shape ) const
        {
            if ( shape.size() != 3 || shape[2] != config_.getModelDim() )
            {
                throw std::invalid_argument( std::format(
                    "MetalSharedKvAttentionOp: build shape must be [B, T, {}]", config_.getModelDim() ) );
            }

            if ( config_.getHeadDim() > 512 )
            {
                throw std::invalid_argument( std::format(
                    "MetalSharedKvAttentionOp: a head of {} channels is past the 512 that a decode step holds", config_.getHeadDim() ) );
            }
        }
    };
}
