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
import Dnn.Tensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.TensorTypes;

namespace Mila::Dnn::Compute::Metal::Gqa
{
    /**
     * @brief Attention with its own [B, NKV, capacity, HS] cache and the shared workspace of the stack it belongs to.
     *
     * Prefill follows CUDA's non-flash pipeline: write the cache, permute Q, Q * K^T, a causal softmax, attention * V, unpermute. A decode
     * of one token writes the cache, then splits the keys over a fixed number of blocks and merges them, reading the position from the
     * execution context (`MetalBackend.md` section 4.4, `Notebooks/MetalDecodeAttention.md`). Several tokens in one decode are Phase 6's,
     * with the model that uses them, and are refused here.
     */
    export template<TensorDataType TPrecision>
    class MetalGqaOp : public Operation<DeviceType::Metal, TPrecision>, public IKvInference
    {
    public:

        using TensorType = Tensor<TPrecision, MetalMemoryResource>;

        MetalGqaOp( IExecutionContext* context, const GqaConfig& config )
            : context_( validateExecutionContext_<DeviceType::Metal>( context, "MetalGqaOp" ) ),
              config_( config )
        {
            config_.validate();
        }

        /// Wire the stack's shared workspace. Only non-null slots replace what was wired.
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
            heads_ = narrowToKernelIndex( config_.getNumHeads() );
            kv_heads_ = narrowToKernelIndex( config_.getNumKvHeads() );
            head_size_ = narrowToKernelIndex( config_.getHeadDim() );

            if ( head_size_ > kMostDecodeChannels )
            {
                throw std::invalid_argument( std::format(
                    "MetalGqaOp: a head of {} channels is past the {} that a decode step holds", head_size_, kMostDecodeChannels ) );
            }

            // A decode threadgroup holds a SIMD-group for each query head of one KV head, and the kernel says how many threads it takes.
            const MetalKernelLimits limits = context_->device().kernels().limits( decodeKernelName( "gqa_decode_partial" ) );
            const std::size_t decode_threads = static_cast<std::size_t>( heads_ / kv_heads_ ) * limits.thread_execution_width;

            if ( decode_threads > limits.max_total_threads_per_threadgroup )
            {
                throw std::invalid_argument( std::format(
                    "MetalGqaOp: {} query heads to a KV head take {} threads in a decode step, past the {} this device's kernel takes",
                    heads_ / kv_heads_, decode_threads, limits.max_total_threads_per_threadgroup ) );
            }

            if ( context.isInferenceMode() )
            {
                const shape_t cache_shape{ input_shape[0], config_.getNumKvHeads(), input_shape[1], config_.getHeadDim() };

                key_cache_ = std::make_unique<TensorType>( context_->getDeviceId(), cache_shape, "gqa.key_cache" );
                value_cache_ = std::make_unique<TensorType>( context_->getDeviceId(), cache_shape, "gqa.value_cache" );
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
            return position >= 0 && position <= cached_length;
        }

        /// @throws std::invalid_argument Past the cache. @throws std::logic_error When no cache or workspace is in place.
        void prefill( const ITensor& q, const ITensor& k, const ITensor& v, ITensor& output, dim_t position_offset ) override
        {
            requireReady();

            if ( q_permute_ == nullptr || preatt_ == nullptr || att_ == nullptr || v_out_ == nullptr )
            {
                throw std::logic_error( "MetalGqaOp::prefill needs setState() first" );
            }

            const int tokens = narrowToKernelIndex( q.shape()[1] );
            const int offset = narrowToKernelIndex( position_offset );

            if ( offset < 0 || offset + tokens > active_length_ )
            {
                throw std::invalid_argument( std::format(
                    "MetalGqaOp::prefill: position_offset {} + {} tokens exceeds the {} positions of the cache", offset, tokens, active_length_ ) );
            }

            const Geometry geometry = geometryFor( tokens, offset );
            const std::size_t count = static_cast<std::size_t>( tokens ) * batches_;
            const int group_size = heads_ / kv_heads_;
            const int attended = offset + tokens;

            encode( "gqa_write_prefill", std::array<const void*, 4>{ k.rawData(), v.rawData(), key_cache_->rawData(), value_cache_->rawData() },
                geometry, count * kv_heads_ * head_size_ );

            encode( "gqa_permute_query", std::array<const void*, 2>{ q.rawData(), q_permute_->rawData() },
                geometry, count * heads_ * head_size_ );

            const std::size_t element_bytes = metalElementBytes<TPrecision>( 1 );

            for ( int b = 0; b < batches_; ++b )
            {
                for ( int group = 0; group < kv_heads_; ++group )
                {
                    const std::size_t first_head = static_cast<std::size_t>( b * heads_ + group * group_size );
                    const std::size_t cache_row = static_cast<std::size_t>( b * kv_heads_ + group ) * capacity_ * head_size_;

                    const auto* const query = bytesOf( q_permute_ ) + element_bytes * first_head * tokens * head_size_;
                    const auto* const keys = bytesOf( key_cache_.get() ) + element_bytes * cache_row;
                    const auto* const scores = bytesOf( preatt_ ) + element_bytes * first_head * tokens * capacity_;

                    encodeMatrixProduct<TPrecision>( *context_, MatrixProduct{
                        query, keys, scores, group_size * tokens, head_size_, attended, capacity_ } );
                }
            }

            encodeSoftmax( geometry, attended, count * heads_ );

            for ( int b = 0; b < batches_; ++b )
            {
                for ( int group = 0; group < kv_heads_; ++group )
                {
                    const std::size_t first_head = static_cast<std::size_t>( b * heads_ + group * group_size );
                    const std::size_t cache_row = static_cast<std::size_t>( b * kv_heads_ + group ) * capacity_ * head_size_;

                    encodeValues(
                        bytesOf( att_ ) + element_bytes * first_head * tokens * capacity_,
                        bytesOf( value_cache_.get() ) + element_bytes * cache_row,
                        bytesOf( v_out_ ) + element_bytes * first_head * tokens * head_size_,
                        group_size * tokens, attended );
                }
            }

            encode( "gqa_unpermute_output", std::array<const void*, 2>{ v_out_->rawData(), output.rawData() },
                geometry, count * heads_ * head_size_ );
        }

        /// `position` is checked here, and the kernels read the same value from the context.
        void decode( const ITensor& q, const ITensor& k, const ITensor& v, ITensor& output, dim_t position ) override
        {
            requireReady();

            if ( q.shape()[1] != 1 )
            {
                throw std::logic_error( std::format( "MetalGqaOp::decode takes one token at a time, not {}", q.shape()[1] ) );
            }

            if ( position < 0 || position >= active_length_ )
            {
                throw std::invalid_argument( std::format(
                    "MetalGqaOp::decode: position {} is past the {} positions of the cache", position, active_length_ ) );
            }

            const Geometry geometry = geometryFor( 1, 0 );
            const void* const decode_position = context_->decodePositionAddress();

            encode( "gqa_write_decode", std::array<const void*, 5>{ k.rawData(), v.rawData(), key_cache_->rawData(), value_cache_->rawData(), decode_position },
                geometry, static_cast<std::size_t>( batches_ * kv_heads_ * head_size_ ) );

            // The partials: each block's output, then its largest score and its sum, for every head of every batch.
            const std::size_t slots = static_cast<std::size_t>( batches_ * heads_ ) * kDecodeBlocks;
            auto* const partial_output = static_cast<float*>( context_->getDeviceScratchBuffer( decodeScratchBytes() ) );
            float* const partial_largest = partial_output + slots * static_cast<std::size_t>( head_size_ );
            float* const partial_sum = partial_largest + slots;

            auto& pipelines = context_->device().kernels();

            context_->dispatchThreadgroups( pipelines.pipeline( decodeKernelName( "gqa_decode_partial" ) ),
                std::array<const void*, 7>{ q.rawData(), key_cache_->rawData(), value_cache_->rawData(), partial_output, partial_largest, partial_sum,
                    decode_position },
                std::as_bytes( std::span( &geometry, 1 ) ), kDecodeBlocks, static_cast<std::size_t>( batches_ * kv_heads_ ),
                static_cast<std::size_t>( heads_ / kv_heads_ ) );

            context_->dispatchThreadgroups( pipelines.pipeline( decodeKernelName( "gqa_decode_merge" ) ),
                std::array<const void*, 4>{ partial_output, partial_largest, partial_sum, output.rawData() },
                std::as_bytes( std::span( &geometry, 1 ) ), static_cast<std::size_t>( batches_ * heads_ ), 1 );
        }

        /// A decode step's partials, from the context's scratch: none before the operation is built.
        [[nodiscard]] std::size_t getScratchBytes() const override
        {
            return this->isBuilt() ? decodeScratchBytes() : 0;
        }

        [[nodiscard]] std::size_t getRequiredScratchBytes( const BuildContext& context ) const override
        {
            const shape_t& input_shape = context.inputShape();

            validateInputShape( input_shape );

            return decodeScratchBytesFor( input_shape[0], config_.getNumHeads(), config_.getHeadDim() );
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
            return key_cache_ ? occupiedTensorBytes( *key_cache_ ) + occupiedTensorBytes( *value_cache_ ) : 0;
        }

        /// K and V, each [B, NKV, capacity, HS]. All transient scratch is the shared workspace's, and is counted there.
        [[nodiscard]] std::size_t getRequiredStateMemorySize( const BuildContext& context ) const override
        {
            const shape_t& input_shape = context.inputShape();

            validateInputShape( input_shape );

            return 2 * occupiedDeviceBytes(
                storageBytes<TPrecision>( input_shape[0] * config_.getNumKvHeads() * input_shape[1] * config_.getHeadDim() ),
                context.getAllocationGranularity() );
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
        };

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

        [[nodiscard]] Geometry geometryFor( int tokens, int position_offset ) const
        {
            return Geometry{
                static_cast<std::uint32_t>( batches_ ),
                static_cast<std::uint32_t>( tokens ),
                static_cast<std::uint32_t>( heads_ ),
                static_cast<std::uint32_t>( kv_heads_ ),
                static_cast<std::uint32_t>( head_size_ ),
                static_cast<std::uint32_t>( capacity_ ),
                static_cast<std::uint32_t>( config_.getWindow() ),
                static_cast<std::uint32_t>( position_offset ),
                config_.getAttentionScale(),
                static_cast<std::uint32_t>( kDecodeBlocks ) };
        }

        [[nodiscard]] static std::size_t decodeScratchBytesFor( dim_t batches, dim_t heads, dim_t head_size )
        {
            return static_cast<std::size_t>( batches * heads ) * kDecodeBlocks * static_cast<std::size_t>( head_size + 2 ) * sizeof( float );
        }

        [[nodiscard]] std::size_t decodeScratchBytes() const
        {
            return decodeScratchBytesFor( batches_, heads_, head_size_ );
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
            const std::array<const void*, 2> buffers{ preatt_->rawData(), att_->rawData() };
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

        MetalExecutionContext* context_;
        GqaConfig config_;

        std::unique_ptr<TensorType> key_cache_;
        std::unique_ptr<TensorType> value_cache_;

        const ITensor* q_permute_{ nullptr };
        const ITensor* preatt_{ nullptr };
        const ITensor* att_{ nullptr };
        const ITensor* v_out_{ nullptr };

        int batches_{ 0 };
        int capacity_{ 0 };
        int heads_{ 0 };
        int kv_heads_{ 0 };
        int head_size_{ 0 };
        int active_length_{ 0 };
    };
}
