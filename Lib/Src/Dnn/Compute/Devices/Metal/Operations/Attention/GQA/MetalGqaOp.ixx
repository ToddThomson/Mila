/**
 * @file MetalGqaOp.ixx
 * @brief Grouped-query attention over a KV cache, on Metal.
 */

export module Compute.MetalGqaOp;

import std;
import Compute.Device;
import Compute.DeviceAllocation;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.GqaState;
import Compute.IExecutionContext;
import Compute.IKvInference;
import Compute.KvCacheView;
import Compute.MetalExecutionContext;
import Compute.MetalKernelLibrary;
import Compute.MetalKernelLimits;
import Compute.MetalMatrixProduct;
import Compute.MetalMatrixView;
import Compute.MetalMemoryResource;
import Compute.MetalTensorDataTypeMap;
import Compute.OperationBase;
import Compute.OperationType;
import Dnn.Component;
import Dnn.Components.GqaConfig;
import Dnn.ITensor;
import Dnn.Quantization.KvCache.Policy;
import Dnn.Quantization.KvCache.PerTokenKvFp8;
import Dnn.Tensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.TensorTypes;

namespace Mila::Dnn::Compute::Metal::Gqa
{
    using namespace Mila::Dnn::Quant::KvCache;

    /**
     * @brief Attention with its own [B, NKV, capacity, HS] cache and the shared workspace of the stack it belongs to.
     *
     * Prefill writes the cache, then runs the fused kernel (`MetalBackend.md` section 4.5) when the transformer switches it on, or else
     * CUDA's non-flash pipeline: permute Q, Q * K^T, a causal softmax, attention * V, unpermute, a block of query positions at a time,
     * the block being the rows the score buffers it was wired hold. The fused prefill reads no score
     * buffer, which is why the transformer, which sizes those, is the one that chooses. A decode
     * writes the cache, then splits the keys over a fixed number of blocks and merges them, reading the position from the execution
     * context (`MetalBackend.md` section 4.4, `Notebooks/MetalDecodeAttention.md`); a decode of a few tokens in a row runs each at its
     * own position, the verify of a speculative round (`MetalBackend.md` section 4.6).
     *
     * An FP8 cache holds each row, one KV head at one token, as E4M3 codes with one FP32 scale, [B, NKV, capacity] for K and for V. A
     * decode step reads the codes and scales directly. Prefill expands one KV head's rows at a time to the compute precision in the
     * context's scratch, and multiplies them as an uncompressed cache's: folding V's scale into the probabilities would put them below
     * FP16's normal range.
     *
     * A sliding window's ring (SlidingWindowKvCache, `SlidingWindowKvCache.md`) holds the window and a prefill chunk less one rows, as
     * CUDA's does, position p in row p % rows; prefill's scores keep the context's stride, and only the ring's columns are filled.
     *
     * @tparam TPrecision FP16 or BF16.
     * @tparam TKvPolicy NoKvCompression, PerTokenKvFp8<>, or SlidingWindowKvCache.
     */
    export template<TensorDataType TPrecision, KvCachePolicy TKvPolicy = NoKvCompression>
    class MetalGqaOp : public Operation<DeviceType::Metal, TPrecision>, public IKvInference
    {
    public:

        using TensorType = Tensor<TPrecision, MetalMemoryResource>;

        static constexpr bool kFp8Cache = std::is_same_v<TKvPolicy, PerTokenKvFp8<>>;
        static constexpr bool kBounded = std::is_same_v<TKvPolicy, SlidingWindowKvCache>;

        static_assert( kFp8Cache || kBounded || std::is_same_v<TKvPolicy, NoKvCompression>,
            "Metal's attention carries an uncompressed cache, an FP8 one, or a sliding window's ring" );

        static constexpr TensorDataType kCacheType = kFp8Cache ? TensorDataType::FP8_E4M3 : TPrecision;

        using CacheTensorType = Tensor<kCacheType, MetalMemoryResource>;
        using ScaleTensorType = Tensor<TensorDataType::FP32, MetalMemoryResource>;

        MetalGqaOp( IExecutionContext* context, const GqaConfig& config )
            : context_( validateExecutionContext_<DeviceType::Metal>( context, "MetalGqaOp" ) ),
              config_( config )
        {
            config_.validate();
        }

        /// Wire the stack's shared workspace. Only non-null slots replace what was wired. With no `att` the softmax runs in place.
        void setState( const GqaState& state )
        {
            const auto wire = []( const ITensor*& slot, const ITensor* tensor )
            {
                if ( tensor )
                {
                    slot = tensor;
                }
            };

            wire( q_permute_, state.q_permute );
            wire( preatt_, state.preatt );
            wire( att_, state.att );
            wire( v_out_, state.v_out );
        }

        void build( const BuildContext& context ) override
        {
            const shape_t& input_shape = context.inputShape();

            validateInputShape( input_shape );

            batches_ = narrowToKernelIndex( input_shape[0] );
            capacity_ = narrowToKernelIndex( input_shape[1] );
            cache_rows_ = cacheRowsFor( context );
            prefill_chunk_ = prefillChunkOf( context );
            heads_ = narrowToKernelIndex( config_.getNumHeads() );
            kv_heads_ = narrowToKernelIndex( config_.getNumKvHeads() );
            head_size_ = narrowToKernelIndex( config_.getHeadDim() );
            decode_tokens_ = narrowToKernelIndex( context.getDecodeTokens() );

            if ( head_size_ > kMostDecodeChannels )
            {
                throw std::invalid_argument( std::format(
                    "MetalGqaOp: a head of {} channels is past the {} that a decode step holds", head_size_, kMostDecodeChannels ) );
            }

            // A decode threadgroup holds a SIMD-group for each query head of one KV head, and the kernel says how many threads it takes.
            const MetalKernelLimits limits = context_->device().kernels().limits( decodeKernelName( kFp8Cache ? "gqa_decode_partial_fp8" : "gqa_decode_partial" ) );
            const std::size_t decode_threads = static_cast<std::size_t>( heads_ / kv_heads_ ) * limits.thread_execution_width;

            if ( decode_threads > limits.max_total_threads_per_threadgroup )
            {
                throw std::invalid_argument( std::format(
                    "MetalGqaOp: {} query heads to a KV head take {} threads in a decode step, past the {} this device's kernel takes",
                    heads_ / kv_heads_, decode_threads, limits.max_total_threads_per_threadgroup ) );
            }

            if ( context.isInferenceMode() )
            {
                const shape_t cache_shape{ input_shape[0], config_.getNumKvHeads(), cache_rows_, config_.getHeadDim() };

                key_cache_ = std::make_unique<CacheTensorType>( context_->getDeviceId(), cache_shape, "gqa.key_cache" );
                value_cache_ = std::make_unique<CacheTensorType>( context_->getDeviceId(), cache_shape, "gqa.value_cache" );

                if constexpr ( kFp8Cache )
                {
                    const shape_t scale_shape{ input_shape[0], config_.getNumKvHeads(), cache_rows_ };

                    key_scales_ = std::make_unique<ScaleTensorType>( context_->getDeviceId(), scale_shape, "gqa.key_scales" );
                    value_scales_ = std::make_unique<ScaleTensorType>( context_->getDeviceId(), scale_shape, "gqa.value_scales" );
                }

                // The fused prefill reads whole tiles of rows, some not yet written, and multiplies them by zero; zero keeps them finite.
                clear( *key_cache_ );
                clear( *value_cache_ );

                if constexpr ( kFp8Cache )
                {
                    clear( *key_scales_ );
                    clear( *value_scales_ );
                }
            }

            Operation<DeviceType::Metal, TPrecision>::build( context );
        }

        void initializeKvCache( dim_t batch_size, dim_t max_sequence_length ) override
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "MetalGqaOp::initializeKvCache requires build() first" );
            }

            if ( batch_size != batches_ )
            {
                throw std::invalid_argument( "MetalGqaOp::initializeKvCache batch size mismatch" );
            }

            if ( max_sequence_length <= 0 || max_sequence_length > capacity_ )
            {
                throw std::invalid_argument( "MetalGqaOp::initializeKvCache max_sequence_length out of range" );
            }

            active_length_ = narrowToKernelIndex( max_sequence_length );
        }

        // The fill count is the network's (DecodeGraph.md 4.3), and the cache is only positional.
        void resetKvCache() override
        {
        }

        bool rewindKvCache( dim_t position, dim_t cached_length ) override
        {
            if ( position < 0 || position > cached_length )
            {
                return false;
            }

            // A ring holds the latest positions written; a continuation from `position` attends a window back from it, which the
            // positions written past it must not have wrapped over. CUDA's ring refuses the same.
            if constexpr ( kBounded )
            {
                // A ring as long as the context holds every position, as a short context's does.
                return cache_rows_ >= active_length_ || cached_length - position <= cache_rows_ - config_.getWindow();
            }

            return true;
        }

        /**
         * @brief Prefill through the fused kernel, or the unfused pipeline its tests compare it with. Off when built.
         *
         * The transformer sizes the shared score buffers by the same decision (`MetalBackend.md` section 4.5), so a narrow
         * workspace with the fused prefill off is refused at prefill.
         */
        void setUseFlashPrefill( bool enabled ) noexcept
        {
            use_flash_prefill_ = enabled;
        }

        /// Whether the fused prefill runs at `head_size`: the head sizes of the models Mila runs, in FP16 and BF16.
        static bool supportsFlashPrefill( dim_t head_size )
        {
            return head_size == 128 || head_size == 256 || head_size == 512;
        }

        /**
         * @brief Whether the unfused pipeline is the faster prefill at `head_size` where it is given a block of score rows.
         *
         * A head of 512 over an uncompressed cache, measured on an M1 Pro from 32 query positions a block up
         * (`Metal/MetalGlobalAttentionPrefill.md` fact 13). An FP8 cache is not measured unfused, and stays fused.
         */
        static bool prefersUnfusedPrefill( dim_t head_size )
        {
            return !kFp8Cache && head_size == 512;
        }

        /// @throws std::invalid_argument Past the cache. @throws std::logic_error When no cache or workspace is in place.
        void prefill( const ITensor& q, const ITensor& k, const ITensor& v, ITensor& output, dim_t position_offset ) override
        {
            requireReady();

            if ( !use_flash_prefill_ && ( q_permute_ == nullptr || preatt_ == nullptr || v_out_ == nullptr ) )
            {
                throw std::logic_error( "MetalGqaOp::prefill needs setState() first, with score buffers when the fused prefill is off" );
            }

            const int tokens = narrowToKernelIndex( q.shape()[1] );
            const int offset = narrowToKernelIndex( position_offset );

            if ( offset < 0 || offset + tokens > active_length_ )
            {
                throw std::invalid_argument( std::format(
                    "MetalGqaOp::prefill: position_offset {} + {} tokens exceeds the {} positions of the cache", offset, tokens, active_length_ ) );
            }

            // A ring holds a chunk's window only for a chunk no longer than it was built for.
            if ( kBounded && tokens > prefill_chunk_ )
            {
                throw std::invalid_argument( std::format(
                    "MetalGqaOp::prefill: {} tokens are past the {}-token prefill chunk the sliding window's ring was built for", tokens, prefill_chunk_ ) );
            }

            const Geometry geometry = geometryFor( tokens, offset );
            const std::size_t count = static_cast<std::size_t>( tokens ) * batches_;
            const int group_size = heads_ / kv_heads_;

            if constexpr ( kFp8Cache )
            {
                context_->dispatchThreadgroups( context_->device().kernels().pipeline( kernelName( "gqa_write_fp8_prefill" ) ),
                    std::array<const void*, 6>{ k.rawData(), v.rawData(), key_cache_->rawData(), value_cache_->rawData(), key_scales_->rawData(),
                        value_scales_->rawData() },
                    std::as_bytes( std::span( &geometry, 1 ) ), count * kv_heads_, 1 );
            } else {
                encode( "gqa_write_prefill", std::array<const void*, 4>{ k.rawData(), v.rawData(), key_cache_->rawData(), value_cache_->rawData() },
                    geometry, count * kv_heads_ * head_size_ );
            }

            if ( use_flash_prefill_ )
            {
                // The cache rows that hold the positions the chunk's queries see: all of them, or the whole ring once it has wrapped.
                flashPrefill( q, output, geometry, std::min( offset + tokens, cache_rows_ ) );

                return;
            }

            // A block of query positions at a time, each a prefill of its own rows at its own offset: every row's softmax is whole, so
            // nothing is carried between blocks (`Metal/MetalGlobalAttentionPrefill.md` section 7). The shape is MLX's
            // sdpa_full_self_attention_blocked (ml-explore/mlx v0.32.4, scaled_dot_product_attention.cpp). One batch at a time, since
            // a block's rows of Q and of the output are contiguous only within a batch.
            const int block = scoreBlockRows( tokens );
            const std::size_t element_bytes = metalElementBytes<TPrecision>( 1 );
            const std::size_t position_bytes = element_bytes * static_cast<std::size_t>( heads_ ) * static_cast<std::size_t>( head_size_ );

            for ( int first = 0; first < tokens; first += block )
            {
                const int block_tokens = std::min( block, tokens - first );
                const int block_attended = offset + first + block_tokens;
                const int block_columns = std::min( block_attended, cache_rows_ );
                const std::size_t block_elements = static_cast<std::size_t>( block_tokens ) * heads_ * head_size_;

                Geometry block_geometry = geometryFor( block_tokens, offset + first );
                block_geometry.batches = 1;

                for ( int b = 0; b < batches_; ++b )
                {
                    const std::size_t first_position = static_cast<std::size_t>( b ) * tokens + first;

                    encode( "gqa_permute_query",
                        std::array<const void*, 2>{ bytesOf( &q ) + position_bytes * first_position, q_permute_->rawData() },
                        block_geometry, block_elements );

                    for ( int group = 0; group < kv_heads_; ++group )
                    {
                        const std::size_t first_head = static_cast<std::size_t>( group * group_size );

                        const auto* const query = bytesOf( q_permute_ ) + element_bytes * first_head * block_tokens * head_size_;
                        const auto* const keys = cachedRows( key_cache_.get(), key_scales_.get(), b * kv_heads_ + group, block_columns, block_geometry );
                        const auto* const scores = bytesOf( preatt_ ) + element_bytes * first_head * block_tokens * capacity_;

                        encodeMatrixProduct<TPrecision>( *context_, MatrixProduct{
                            query, keys, scores, group_size * block_tokens, head_size_, block_columns, capacity_ } );
                    }

                    encodeSoftmax( block_geometry, block_attended, static_cast<std::size_t>( block_tokens ) * heads_ );

                    for ( int group = 0; group < kv_heads_; ++group )
                    {
                        const std::size_t first_head = static_cast<std::size_t>( group * group_size );

                        encodeValues(
                            bytesOf( attentionBuffer() ) + element_bytes * first_head * block_tokens * capacity_,
                            cachedRows( value_cache_.get(), value_scales_.get(), b * kv_heads_ + group, block_columns, block_geometry ),
                            bytesOf( v_out_ ) + element_bytes * first_head * block_tokens * head_size_,
                            group_size * block_tokens, block_columns );
                    }

                    encode( "gqa_unpermute_output",
                        std::array<const void*, 2>{ v_out_->rawData(), bytesOf( &output ) + position_bytes * first_position },
                        block_geometry, block_elements );
                }
            }
        }

        /**
         * One token, or a few in a row from the decode position (Gemma4Mtp.md 4.7), each attending through its own position.
         * `position` is checked here, and the kernels read the same value from the context.
         *
         * @throws std::invalid_argument When the tokens run past the cache, are more than the op was built for, or on the ring would
         *         overwrite a key the first of them still attends.
         */
        void decode( const ITensor& q, const ITensor& k, const ITensor& v, ITensor& output, dim_t position ) override
        {
            requireReady();

            const int tokens = decodeTokenCount( q, position );
            const Geometry geometry = geometryFor( tokens, 0 );
            const void* const decode_position = context_->decodePositionAddress();
            const std::size_t rows = static_cast<std::size_t>( batches_ * tokens * kv_heads_ );

            if constexpr ( kFp8Cache )
            {
                context_->dispatchThreadgroups( context_->device().kernels().pipeline( kernelName( "gqa_write_fp8_decode" ) ),
                    std::array<const void*, 7>{ k.rawData(), v.rawData(), key_cache_->rawData(), value_cache_->rawData(), key_scales_->rawData(),
                        value_scales_->rawData(), decode_position },
                    std::as_bytes( std::span( &geometry, 1 ) ), rows, 1 );
            } else {
                encode( "gqa_write_decode", std::array<const void*, 5>{ k.rawData(), v.rawData(), key_cache_->rawData(), value_cache_->rawData(), decode_position },
                    geometry, rows * static_cast<std::size_t>( head_size_ ) );
            }

            encodeDecodeAttention( q, output, geometry, decode_position );
        }

        /**
         * @brief This op's cache, for attention another component runs over it without writing (Gemma4Mtp.md 4.3).
         *
         * @throws std::logic_error Before the cache is initialized.
         */
        [[nodiscard]] KvCacheView getKvCacheView() const
        {
            requireReady();

            KvCacheView view;
            view.keys = key_cache_->rawData();
            view.values = value_cache_->rawData();
            view.key_scales = kFp8Cache ? static_cast<const float*>( key_scales_->rawData() ) : nullptr;
            view.value_scales = kFp8Cache ? static_cast<const float*>( value_scales_->rawData() ) : nullptr;
            view.capacity = cache_rows_;
            view.num_kv_heads = kv_heads_;
            view.head_size = head_size_;
            view.window = narrowToKernelIndex( config_.getWindow() );
            view.fp8 = kFp8Cache;

            return view;
        }

        /// A decode step's partials, and with an FP8 cache one KV head's rows expanded for prefill, from the context's scratch: none
        /// before the operation is built.
        [[nodiscard]] std::size_t getScratchBytes() const override
        {
            return this->isBuilt() ? scratchBytesFor( batches_, heads_, head_size_, capacity_, decode_tokens_ ) : 0;
        }

        [[nodiscard]] std::size_t getRequiredScratchBytes( const BuildContext& context ) const override
        {
            const shape_t& input_shape = context.inputShape();

            validateInputShape( input_shape );

            return scratchBytesFor( input_shape[0], config_.getNumHeads(), config_.getHeadDim(), input_shape[1], context.getDecodeTokens() );
        }

        /// @throws std::runtime_error Always: GQA is inference only, as on CUDA, and is driven by prefill() and decode().
        void forward( const ITensor&, ITensor& ) const
        {
            throw std::runtime_error( "MetalGqaOp::forward is not implemented; drive GQA through prefill() and decode()." );
        }

        [[nodiscard]] OperationType getOperationType() const override
        {
            return OperationType::GroupedQueryAttentionOp;
        }

        [[nodiscard]] std::string getName() const override
        {
            return "Metal.Gqa";
        }

        [[nodiscard]] std::size_t getStateMemorySize() const override
        {
            if ( !key_cache_ )
            {
                return 0;
            }

            std::size_t bytes = occupiedTensorBytes( *key_cache_ ) + occupiedTensorBytes( *value_cache_ );

            if constexpr ( kFp8Cache )
            {
                bytes += occupiedTensorBytes( *key_scales_ ) + occupiedTensorBytes( *value_scales_ );
            }

            return bytes;
        }

        /// K and V, each [B, NKV, capacity, HS], and with an FP8 cache their scales, each [B, NKV, capacity]. All transient scratch is
        /// the shared workspace's or the context's, and is counted there.
        [[nodiscard]] std::size_t getRequiredStateMemorySize( const BuildContext& context ) const override
        {
            const shape_t& input_shape = context.inputShape();

            validateInputShape( input_shape );

            const dim_t rows = input_shape[0] * config_.getNumKvHeads() * cacheRowsFor( context );
            std::size_t bytes = 2 * occupiedDeviceBytes( storageBytes<kCacheType>( rows * config_.getHeadDim() ), context.getAllocationGranularity() );

            if constexpr ( kFp8Cache )
            {
                bytes += 2 * occupiedDeviceBytes( storageBytes<TensorDataType::FP32>( rows ), context.getAllocationGranularity() );
            }

            return bytes;
        }

    private:

        // Fixed, so that a decode step's grid never changes; within 11% of the best count on average over the shapes and depths
        // measured, 27% at worst (Notebooks/MetalDecodeAttention.md section 2).
        static constexpr std::size_t kDecodeBlocks = 32;

        // A lane holds at most 16 of a head's channels in a decode step.
        static constexpr int kMostDecodeChannels = 512;

        static constexpr int kSmallTileRows = 8;
        static constexpr int kLargeTileRows = 128;
        static constexpr int kTileColumns = 16;

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

        /**
         * The rows a KV head's cache holds: every position of the context, or for a ring the window and a prefill chunk less one; and
         * never fewer than one tile of the fused prefill, which reads whole tiles. The rows past a short context are never written.
         */
        [[nodiscard]] int cacheRowsFor( const BuildContext& context ) const
        {
            const int sequence_length = narrowToKernelIndex( context.inputShape()[1] );
            const int one_tile = supportsFlashPrefill( config_.getHeadDim() ) ? flashTileFor( config_.getHeadDim() ).keys : 0;

            if constexpr ( kBounded )
            {
                const int window = narrowToKernelIndex( config_.getWindow() );

                if ( window <= 0 )
                {
                    throw std::invalid_argument( "MetalGqaOp: a sliding window's ring needs a positive window; a layer that attends every "
                        "position keeps the uncompressed cache" );
                }

                return std::max( one_tile, std::min( sequence_length, window + prefillChunkOf( context ) - 1 ) );
            } else {
                return std::max( one_tile, sequence_length );
            }
        }

        [[nodiscard]] static int prefillChunkOf( const BuildContext& context )
        {
            const dim_t prefill = context.getPrefillSize();

            return narrowToKernelIndex( prefill > 0 ? prefill : context.inputShape()[1] );
        }

        /**
         * The query positions one block of the unfused prefill takes: what the wired score buffers hold rows for, and never more than
         * the permuted Q and the output beside them do.
         *
         * @throws std::logic_error When the score buffers are narrower than the context their rows are strided by.
         */
        [[nodiscard]] int scoreBlockRows( int tokens ) const
        {
            dim_t rows = tokens;

            const ITensor* const attention = attentionBuffer();

            for ( const ITensor* const tensor : { q_permute_, preatt_, attention, v_out_ } )
            {
                rows = std::min( rows, tensor->shape()[2] );
            }

            if ( preatt_->shape()[3] < capacity_ || attention->shape()[3] < capacity_ )
            {
                throw std::logic_error( std::format(
                    "MetalGqaOp::prefill: the score buffers are {} wide, and the unfused prefill strides them by the {} positions of the context",
                    std::min( preatt_->shape()[3], attention->shape()[3] ), capacity_ ) );
            }

            return narrowToKernelIndex( rows );
        }

        /**
         * Where the softmax writes and attention * V reads: the scores' own buffer, in place, unless a second was wired. Each score
         * is read and written by the one thread that owns its column, after every thread of its row has its sum, so one buffer
         * serves as both.
         */
        [[nodiscard]] const ITensor* attentionBuffer() const noexcept
        {
            return att_ != nullptr ? att_ : preatt_;
        }

        void validateInputShape( const shape_t& shape ) const
        {
            const dim_t packed = ( config_.getNumHeads() + 2 * config_.getNumKvHeads() ) * config_.getHeadDim();

            if ( shape.size() != 3 || shape[2] != packed )
            {
                throw std::invalid_argument( std::format(
                    "MetalGqaOp: the input must be [B, T, {}], the packed query, key and value", packed ) );
            }
        }

        void requireReady() const
        {
            if ( active_length_ == 0 || !key_cache_ )
            {
                throw std::logic_error( "MetalGqaOp: the KV cache must be initialized first" );
            }
        }

        /**
         * The tokens of a decode call -- one, or a few in a row from the decode position -- after checking that the last of them is
         * inside the cache and, on the ring, that writing them evicts no key the first still attends.
         */
        [[nodiscard]] int decodeTokenCount( const ITensor& q, dim_t position ) const
        {
            const int tokens = narrowToKernelIndex( q.shape()[1] );

            if ( tokens < 1 || position < 0 || position + tokens > active_length_ )
            {
                throw std::invalid_argument( std::format(
                    "MetalGqaOp::decode: {} tokens from position {} do not fit the {} positions of the cache", tokens, position, active_length_ ) );
            }

            // More would grow the context's scratch past what the planner priced.
            if ( tokens > decode_tokens_ )
            {
                throw std::invalid_argument( std::format(
                    "MetalGqaOp::decode: {} tokens, and this op was built for decode calls of at most {} (BuildContext::withDecodeTokens)",
                    tokens, decode_tokens_ ) );
            }

            if constexpr ( kBounded )
            {
                const int window = narrowToKernelIndex( config_.getWindow() );

                if ( tokens > 1 && window + tokens - 1 > cache_rows_ )
                {
                    throw std::invalid_argument( std::format(
                        "MetalGqaOp::decode: {} tokens over a window of {} need {} ring rows; the ring holds {}",
                        tokens, window, window + tokens - 1, cache_rows_ ) );
                }
            }

            return tokens;
        }

        /// The partial pass over the cache for every query head of every token, then the merge, into `output`.
        void encodeDecodeAttention( const ITensor& q, ITensor& output, const Geometry& geometry, const void* decode_position ) const
        {
            const int tokens = static_cast<int>( geometry.tokens );

            // The partials: each block's output, then its largest score and its sum, for every head of every token of every batch.
            const std::size_t slots = static_cast<std::size_t>( batches_ * tokens * heads_ ) * kDecodeBlocks;
            auto* const partial_output = static_cast<float*>( context_->getDeviceScratchBuffer( decodeScratchBytes( tokens ) ) );
            float* const partial_largest = partial_output + slots * static_cast<std::size_t>( head_size_ );
            float* const partial_sum = partial_largest + slots;

            auto& pipelines = context_->device().kernels();
            const std::size_t query_heads = static_cast<std::size_t>( heads_ / kv_heads_ );
            const std::size_t kv_heads_of_tokens = static_cast<std::size_t>( batches_ * tokens * kv_heads_ );

            if constexpr ( kFp8Cache )
            {
                context_->dispatchThreadgroups( pipelines.pipeline( decodeKernelName( "gqa_decode_partial_fp8" ) ),
                    std::array<const void*, 9>{ q.rawData(), key_cache_->rawData(), value_cache_->rawData(), key_scales_->rawData(),
                        value_scales_->rawData(), partial_output, partial_largest, partial_sum, decode_position },
                    std::as_bytes( std::span( &geometry, 1 ) ), kDecodeBlocks, kv_heads_of_tokens, query_heads );
            } else {
                context_->dispatchThreadgroups( pipelines.pipeline( decodeKernelName( "gqa_decode_partial" ) ),
                    std::array<const void*, 7>{ q.rawData(), key_cache_->rawData(), value_cache_->rawData(), partial_output, partial_largest, partial_sum,
                        decode_position },
                    std::as_bytes( std::span( &geometry, 1 ) ), kDecodeBlocks, kv_heads_of_tokens, query_heads );
            }

            context_->dispatchThreadgroups( pipelines.pipeline( decodeKernelName( "gqa_decode_merge" ) ),
                std::array<const void*, 4>{ partial_output, partial_largest, partial_sum, output.rawData() },
                std::as_bytes( std::span( &geometry, 1 ) ), static_cast<std::size_t>( batches_ * tokens * heads_ ), 1 );
        }

        [[nodiscard]] Geometry geometryFor( int tokens, int position_offset ) const
        {
            return Geometry{
                static_cast<std::uint32_t>( batches_ ),
                static_cast<std::uint32_t>( tokens ),
                static_cast<std::uint32_t>( heads_ ),
                static_cast<std::uint32_t>( kv_heads_ ),
                static_cast<std::uint32_t>( head_size_ ),
                static_cast<std::uint32_t>( cache_rows_ ),
                static_cast<std::uint32_t>( config_.getWindow() ),
                static_cast<std::uint32_t>( position_offset ),
                config_.getAttentionScale(),
                static_cast<std::uint32_t>( kDecodeBlocks ),
                static_cast<std::uint32_t>( capacity_ ),
                0u };
        }

        /// A decode call's partials: every block of every query head of each of its tokens.
        [[nodiscard]] static std::size_t decodeScratchBytesFor( dim_t batches, dim_t heads, dim_t head_size, dim_t tokens )
        {
            return static_cast<std::size_t>( batches * tokens * heads ) * kDecodeBlocks * static_cast<std::size_t>( head_size + 2 ) * sizeof( float );
        }

        // An FP8 prefill expands one KV head's K, and for the fused prefill its V beside it, into the context's scratch.
        [[nodiscard]] static std::size_t scratchBytesFor( dim_t batches, dim_t heads, dim_t head_size, dim_t capacity, dim_t decode_tokens )
        {
            const std::size_t decode_bytes = decodeScratchBytesFor( batches, heads, head_size, decode_tokens );

            if constexpr ( kFp8Cache )
            {
                return std::max( decode_bytes, 2 * metalElementBytes<TPrecision>( capacity * head_size ) );
            } else {
                return decode_bytes;
            }
        }

        [[nodiscard]] std::size_t decodeScratchBytes( int tokens ) const
        {
            return decodeScratchBytesFor( batches_, heads_, head_size_, tokens );
        }

        /**
         * One KV head's first `rows` rows of K or V, as prefill's products read them: in place for a cache in the compute type, and for an
         * FP8 cache expanded into the context's scratch, where the next call's rows replace them once the work between has run.
         */
        [[nodiscard]] const std::byte* cachedRows( const CacheTensorType* cache, const ScaleTensorType* scales, int kv_head_of_batch, int rows,
            const Geometry& geometry ) const
        {
            const std::size_t first_row = static_cast<std::size_t>( kv_head_of_batch ) * cache_rows_;
            const auto* const codes = bytesOf( cache ) + storageBytes<kCacheType>( static_cast<dim_t>( first_row * head_size_ ) );

            if constexpr ( !kFp8Cache )
            {
                return codes;
            } else {
                void* const expanded = context_->getDeviceScratchBuffer( metalElementBytes<TPrecision>( static_cast<dim_t>( rows ) * head_size_ ) );
                const auto* const row_scales = static_cast<const float*>( static_cast<const ITensor*>( scales )->rawData() ) + first_row;

                encode( "gqa_expand_fp8_rows", std::array<const void*, 3>{ codes, row_scales, expanded }, geometry,
                    static_cast<std::size_t>( rows ) * static_cast<std::size_t>( head_size_ ) );

                return static_cast<const std::byte*>( expanded );
            }
        }

        /// The head sizes the models use have their own kernels; any other runs the interleaved one.
        [[nodiscard]] std::string decodeKernelName( std::string_view family ) const
        {
            const bool own = head_size_ == 64 || head_size_ == 128 || head_size_ == 256 || head_size_ == 512;

            return own
                ? std::format( "{}_{}_{}", family, MetalTensorDataTypeMap<TPrecision>::kernel_type, head_size_ )
                : std::format( "{}_{}_any", family, MetalTensorDataTypeMap<TPrecision>::kernel_type );
        }

        [[nodiscard]] static const std::byte* bytesOf( const ITensor* tensor )
        {
            return static_cast<const std::byte*>( tensor->rawData() );
        }

        [[nodiscard]] static std::string kernelName( std::string_view family )
        {
            return std::format( "{}_{}", family, MetalTensorDataTypeMap<TPrecision>::kernel_type );
        }

        template<std::size_t TBuffers>
        void encode( std::string_view family, const std::array<const void*, TBuffers>& buffers, const Geometry& geometry, std::size_t elements ) const
        {
            context_->dispatch( context_->device().kernels().pipeline( kernelName( family ) ), buffers, std::as_bytes( std::span( &geometry, 1 ) ), elements );
        }

        void encodeSoftmax( const Geometry& geometry, int attended, std::size_t rows ) const
        {
            // Two sets of constants: the geometry at one index and the attended width at the next.
            const std::array<const void*, 2> buffers{ preatt_->rawData(), attentionBuffer()->rawData() };
            const struct { Geometry geometry; std::uint32_t attended; } constants{ geometry, static_cast<std::uint32_t>( attended ) };

            context_->dispatchThreadgroupsWithTwoConstants(
                context_->device().kernels().pipeline( kernelName( "gqa_prefill_softmax" ) ), buffers,
                std::as_bytes( std::span( &constants.geometry, 1 ) ), std::as_bytes( std::span( &constants.attended, 1 ) ), rows, 1 );
        }

        void encodeValues( const std::byte* attention, const std::byte* values, const std::byte* output, int rows, int attended ) const
        {
            int next_row = 0;
            const int columns_in_tiles = rows < kSmallTileRows ? 0 : head_size_ - head_size_ % kTileColumns;
            const std::size_t element_bytes = metalElementBytes<TPrecision>( 1 );

            const auto tiles = [&]<int TTileRows, int TSimdgroups>()
            {
                const int tiles_down = ( rows - next_row ) / TTileRows;

                if ( tiles_down == 0 || columns_in_tiles == 0 )
                {
                    return;
                }

                const int rows_taken = tiles_down * TTileRows;

                const std::array<MetalMatrixView, 3> matrices{
                    MetalMatrixView{ attention + element_bytes * next_row * capacity_, TPrecision, rows_taken, attended, capacity_ },
                    MetalMatrixView{ values, TPrecision, attended, head_size_, head_size_ },
                    MetalMatrixView{ output + element_bytes * next_row * head_size_, TPrecision, rows_taken, head_size_, head_size_ } };

                context_->dispatchMatrices(
                    context_->device().kernels().pipeline( std::format(
                        "gqa_value_tiles_{}_{}_{}_{}", MetalTensorDataTypeMap<TPrecision>::kernel_type, TTileRows, kTileColumns, TSimdgroups ) ),
                    matrices, static_cast<std::size_t>( columns_in_tiles / kTileColumns ), static_cast<std::size_t>( tiles_down ), TSimdgroups );

                next_row += rows_taken;
            };

            tiles.template operator()<kLargeTileRows, 4>();
            tiles.template operator()<kSmallTileRows, 2>();

            encodeValueElements( attention, values, output, next_row, rows, 0, attended );
            encodeValueElements( attention, values, output, 0, next_row, columns_in_tiles, attended );
        }

        void encodeValueElements( const std::byte* attention, const std::byte* values, const std::byte* output, int first_row, int last_row, int first_column, int attended ) const
        {
            const int rows = last_row - first_row;
            const int columns = head_size_ - first_column;

            if ( rows <= 0 || columns <= 0 )
            {
                return;
            }

            const std::array<const void*, 3> buffers{ attention, values, output };
            const std::array<std::uint32_t, 6> rectangle{
                static_cast<std::uint32_t>( attended ), static_cast<std::uint32_t>( capacity_ ), static_cast<std::uint32_t>( head_size_ ),
                static_cast<std::uint32_t>( first_row ), static_cast<std::uint32_t>( first_column ), static_cast<std::uint32_t>( columns ) };

            context_->dispatch(
                context_->device().kernels().pipeline( kernelName( "gqa_value_elements" ) ), buffers,
                std::as_bytes( std::span( rectangle ) ), static_cast<std::size_t>( rows ) * static_cast<std::size_t>( columns ) );
        }

        // The kernels' GqaFlashConstants, field for field.
        struct FlashConstants
        {
            Geometry geometry;
            std::uint32_t kv_stride;
            std::uint32_t first_kv_head;
            std::uint32_t positions;
            std::uint32_t heads_per_threadgroup;
            std::uint32_t head_splits;
        };

        struct FlashTile
        {
            int rows;
            int keys;
            int most_simdgroups;

            // SIMD-groups a threadgroup is given when the query-head group is smaller, more blocks of rows reading each tile.
            int wanted_simdgroups;
        };

        // The kernel's instance for this head size: query rows a SIMD-group takes, cache rows a tile reads, and the most SIMD-groups its
        // threadgroup memory holds (GroupedQueryAttention.Flash.metal states the same).
        [[nodiscard]] static FlashTile flashTileFor( dim_t head_size )
        {
            switch ( head_size )
            {
                case 128:
                    return { 16, 64, 8, 4 };
                case 256:
                    return { 16, 64, 8, 4 };
                default:
                    return { 8, 64, 8, 4 };
            }
        }

        template<typename TTensor>
        void clear( TTensor& tensor ) const
        {
            context_->zeroBytes( tensor.rawData(), occupiedTensorBytes( tensor ) );
        }

        /// Q x K^T, the causal softmax and P x V in one kernel, from Q's own layout to the output's (`MetalBackend.md` section 4.5).
        void flashPrefill( const ITensor& q, ITensor& output, const Geometry& geometry, int columns ) const
        {
            if ( !supportsFlashPrefill( head_size_ ) )
            {
                throw std::logic_error( std::format( "MetalGqaOp: no fused prefill serves a head of {} channels", head_size_ ) );
            }

            const FlashTile tile = flashTileFor( head_size_ );

            if ( cache_rows_ < tile.keys )
            {
                throw std::logic_error( std::format(
                    "MetalGqaOp: the fused prefill reads {} cache rows a tile, more than the {} this cache was built with", tile.keys, cache_rows_ ) );
            }

            const int group_size = heads_ / kv_heads_;

            const std::string kernel = std::format( "gqa_flash_prefill_{}_{}", MetalTensorDataTypeMap<TPrecision>::kernel_type, head_size_ );
            const MetalKernelLimits limits = context_->device().kernels().limits( kernel );
            const std::size_t most_simdgroups = std::min<std::size_t>( static_cast<std::size_t>( tile.most_simdgroups ),
                limits.max_total_threads_per_threadgroup / limits.thread_execution_width );

            // The whole group to a threadgroup where its threads fit, else halves of it, then quarters.
            int heads_per_threadgroup = group_size;

            while ( static_cast<std::size_t>( heads_per_threadgroup ) > most_simdgroups && heads_per_threadgroup % 2 == 0 )
            {
                heads_per_threadgroup /= 2;
            }

            if ( static_cast<std::size_t>( heads_per_threadgroup ) > most_simdgroups )
            {
                throw std::invalid_argument( std::format(
                    "MetalGqaOp: the fused prefill cannot split {} query heads to a KV head into threadgroups of {} SIMD-groups, the most "
                    "this device gives {}", group_size, most_simdgroups, kernel ) );
            }

            const int head_splits = group_size / heads_per_threadgroup;
            const int blocks = std::max( 1, std::min( tile.wanted_simdgroups / heads_per_threadgroup,
                static_cast<int>( most_simdgroups ) / heads_per_threadgroup ) );
            const int positions = tile.rows * blocks;
            const std::size_t simdgroups = static_cast<std::size_t>( heads_per_threadgroup * blocks );
            const std::size_t position_groups = static_cast<std::size_t>( ( geometry.tokens + positions - 1 ) / positions ) * head_splits;

            auto* const pipeline = context_->device().kernels().pipeline( kernel );

            FlashConstants constants{ geometry, 0, 0, static_cast<std::uint32_t>( positions ),
                static_cast<std::uint32_t>( heads_per_threadgroup ), static_cast<std::uint32_t>( head_splits ) };

            if constexpr ( !kFp8Cache )
            {
                constants.kv_stride = static_cast<std::uint32_t>( cache_rows_ * head_size_ );

                context_->dispatchThreadgroups( pipeline,
                    std::array<const void*, 4>{ q.rawData(), key_cache_->rawData(), value_cache_->rawData(), output.rawData() },
                    std::as_bytes( std::span( &constants, 1 ) ), position_groups, static_cast<std::size_t>( batches_ * kv_heads_ ), simdgroups );
            } else {
                // Every row a tile can read, written or not: unwritten ones are cleared codes with a zero scale.
                const int rows_read = std::min( cache_rows_, std::max( columns, tile.keys ) );
                const std::size_t half_bytes = metalElementBytes<TPrecision>( static_cast<dim_t>( cache_rows_ ) * head_size_ );
                auto* const expanded = static_cast<std::byte*>( context_->getDeviceScratchBuffer( 2 * half_bytes ) );

                for ( int kv_head_of_batch = 0; kv_head_of_batch < batches_ * kv_heads_; ++kv_head_of_batch )
                {
                    expandRows( key_cache_.get(), key_scales_.get(), kv_head_of_batch, rows_read, expanded, geometry );
                    expandRows( value_cache_.get(), value_scales_.get(), kv_head_of_batch, rows_read, expanded + half_bytes, geometry );

                    constants.first_kv_head = static_cast<std::uint32_t>( kv_head_of_batch );

                    context_->dispatchThreadgroups( pipeline,
                        std::array<const void*, 4>{ q.rawData(), expanded, expanded + half_bytes, output.rawData() },
                        std::as_bytes( std::span( &constants, 1 ) ), position_groups, 1, simdgroups );
                }
            }
        }

        /// One KV head's first `rows` FP8 rows, widened to the compute type at `destination`.
        void expandRows( const CacheTensorType* cache, const ScaleTensorType* scales, int kv_head_of_batch, int rows, void* destination,
            const Geometry& geometry ) const
        {
            const std::size_t first_row = static_cast<std::size_t>( kv_head_of_batch ) * cache_rows_;
            const auto* const codes = bytesOf( cache ) + storageBytes<kCacheType>( static_cast<dim_t>( first_row * head_size_ ) );
            const auto* const row_scales = static_cast<const float*>( static_cast<const ITensor*>( scales )->rawData() ) + first_row;

            encode( "gqa_expand_fp8_rows", std::array<const void*, 3>{ codes, row_scales, destination }, geometry,
                static_cast<std::size_t>( rows ) * static_cast<std::size_t>( head_size_ ) );
        }

        MetalExecutionContext* context_;
        GqaConfig config_;

        bool use_flash_prefill_{ false };

        std::unique_ptr<CacheTensorType> key_cache_;
        std::unique_ptr<CacheTensorType> value_cache_;
        std::unique_ptr<ScaleTensorType> key_scales_;
        std::unique_ptr<ScaleTensorType> value_scales_;

        const ITensor* q_permute_{ nullptr };
        const ITensor* preatt_{ nullptr };
        const ITensor* att_{ nullptr };
        const ITensor* v_out_{ nullptr };

        int batches_{ 0 };

        // The context's positions, which prefill's scores are strided by, and the rows each KV head's cache holds.
        int capacity_{ 0 };
        int cache_rows_{ 0 };
        int prefill_chunk_{ 0 };
        int heads_{ 0 };
        int kv_heads_{ 0 };
        int head_size_{ 0 };
        int active_length_{ 0 };

        // The most tokens one decode call takes (BuildContext::withDecodeTokens), which its scratch reservation holds.
        int decode_tokens_{ 1 };
    };
}
