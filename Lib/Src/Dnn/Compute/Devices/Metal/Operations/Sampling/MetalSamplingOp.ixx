/**
 * @file MetalSamplingOp.ixx
 * @brief Token sampling on Metal.
 */

export module Compute.MetalSamplingOp;

import std;
import Compute.Device;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.IExecutionContext;
import Compute.MetalExecutionContext;
import Compute.MetalKernelLibrary;
import Compute.MetalMemoryResource;
import Compute.MetalTensorDataTypeMap;
import Compute.OperationBase;
import Compute.OperationType;
import Dnn.ITensor;
import Dnn.Samplers.SamplingConfig;
import Dnn.SamplingParams;
import Dnn.Tensor;
import Dnn.TensorDataType;
import Dnn.TensorOps;
import Dnn.TensorTypes;

namespace Mila::Dnn::Compute::Metal::Sampling
{
    /**
     * @brief Draws one token from the last row of logits, or one from each of several rows, greedy or by temperature, top-k and top-p.
     *
     * The multinomial is CUDA's multi-block pipeline in stages (Sampling.metal): a row spread over many threadgroups, each truncation's
     * cut found exactly by radix selection on the scaled logits' bits. Metal has no event for one small readback, so `awaitToken` synchronizes the context, which also finishes anything
     * encoded after the sample. The decode-ahead pipeline therefore runs one step at a time here, with the same tokens. Several rows are
     * a speculative round's (Gemma4Mtp.md 4.8): each row is drawn by the kernels that draw one, with its own uniform.
     */
    export template<TensorDataType TPrecision>
    class MetalSamplingOp : public Operation<DeviceType::Metal, TPrecision>
    {
    public:

        /// Most rows one call draws, as CUDA's sampler.
        static constexpr dim_t kMaximumRows = 8;

        /// @throws std::invalid_argument For a configuration of more than kMaximumRows rows.
        MetalSamplingOp( IExecutionContext* context, const SamplingConfig& config )
            : context_( validateExecutionContext_<DeviceType::Metal>( context, "MetalSamplingOp" ) ),
              config_( validated( config ) ),
              scratch_( context->getDeviceId(), shape_t{ config.getMaximumRows() * config.getVocabularySize() } ),
              reduction_scratch_( context->getDeviceId(), shape_t{ config.getMaximumRows() * kFloatScratchPerRow } ),
              count_scratch_( context->getDeviceId(), shape_t{ config.getMaximumRows() * kCountScratchPerRow } ),
              pipelines_( resolvePipelines( context_->device().kernels() ) )
        {
            uploadSuppressedTokens();
        }

        /**
         * @brief Encode one sample into `token_out`.
         *
         * @param logits   Metal logits; the last `vocab_size` elements are the row.
         * @param produced The reply's produced tokens, required with a presence penalty, and marked with the token drawn.
         * @throws std::invalid_argument For a penalty that is not finite, or one without the produced tokens.
         */
        void forward( const ITensor& logits, ITensor& token_out, const SamplingParams& params, float r, ITensor* produced = nullptr ) const
        {
            encodeLastRow( logits, token_out, params, r, produced );
        }

        /// As `forward`, remembering the token for `awaitToken`. One may be outstanding at a time.
        void enqueueForward( const ITensor& logits, ITensor& token_out, const SamplingParams& params, float r, ITensor* produced = nullptr ) const
        {
            encodeLastRow( logits, token_out, params, r, produced );
            pending_token_ = &token_out;
        }

        void enqueueForwardOnDevice( const ITensor& logits, ITensor& token_out, const SamplingParams& params, float r, ITensor* produced = nullptr ) const
        {
            encodeLastRow( logits, token_out, params, r, produced );
        }

        /**
         * @brief Encode a draw from each of the first `rows` rows of `logits` into element r of `tokens_out`, as `enqueueForwardOnDevice`
         *        draws that row alone with `uniforms[r]`. Nothing is marked produced: the acceptance walk marks what the round keeps.
         *
         * @param round_tokens The round's tokens, the known one then the drafts; with the penalty on, row r also penalizes drafts 1..r.
         * @throws std::invalid_argument For rows outside 1 to the configured maximum, or too few uniforms, logits, slots or round tokens.
         */
        void enqueueRowsOnDevice(
            const ITensor& logits, ITensor& tokens_out, dim_t rows, const SamplingParams& params, std::span<const float> uniforms,
            const ITensor* produced = nullptr, const ITensor* round_tokens = nullptr ) const
        {
            const dim_t vocab = config_.getVocabularySize();

            requireRows( rows, "enqueueRowsOnDevice" );

            if ( static_cast<dim_t>( uniforms.size() ) < rows || logits.size() < rows * vocab || tokens_out.size() < rows )
            {
                throw std::invalid_argument( std::format(
                    "MetalSamplingOp::enqueueRowsOnDevice: {} rows need {} uniforms, {} logits and {} token slots; given {}, {} and {}",
                    rows, rows, rows * vocab, rows, uniforms.size(), logits.size(), tokens_out.size() ) );
            }

            if ( round_tokens != nullptr && round_tokens->size() < rows )
            {
                throw std::invalid_argument( std::format(
                    "MetalSamplingOp::enqueueRowsOnDevice: {} rows need {} round tokens; given {}", rows, rows, round_tokens->size() ) );
            }

            encodeRows( logits.rawData(), tokens_out, rows, params, uniforms, produced, round_tokens );
        }

        /**
         * @brief Encode a verified round's acceptance walk: a draft is kept while the row before it chose it.
         *
         * Writes `[m, d_1 .. d_m, next]` into `result`, and the next token into slot 0 of `tokens`, where the next round's draft reads it.
         *
         * @param produced The reply's produced tokens, which gain the kept drafts and the next token; null when the penalty is off.
         */
        void enqueueAcceptOnDevice( ITensor& tokens, const ITensor& chosen, dim_t rows, ITensor& result, ITensor* produced = nullptr ) const
        {
            requireRows( rows, "enqueueAcceptOnDevice" );

            if ( tokens.size() < rows || chosen.size() < rows || result.size() < rows + 1 )
            {
                throw std::invalid_argument( std::format(
                    "MetalSamplingOp::enqueueAcceptOnDevice: {} rows need {} tokens, {} chosen and {} result slots; given {}, {} and {}",
                    rows, rows, rows, rows + 1, tokens.size(), chosen.size(), result.size() ) );
            }

            if ( produced != nullptr )
            {
                requireProduced( *produced );
            }

            const AcceptParameters parameters{ narrowToKernelIndex( rows ), produced != nullptr ? 1u : 0u };

            // An absent produced is bound to the result, which the kernel's flag keeps unwritten.
            const std::array<const void*, 4> buffers{
                tokens.rawData(), chosen.rawData(), result.rawData(), produced != nullptr ? produced->rawData() : result.rawData() };

            context_->dispatch( pipelines_.accept, buffers, std::as_bytes( std::span( &parameters, 1 ) ), 1 );
        }

        /// Wait for the sample `enqueueForward` encoded, and return its token. @throws std::logic_error When there is none.
        [[nodiscard]] std::int32_t awaitToken() const
        {
            if ( pending_token_ == nullptr )
            {
                throw std::logic_error( "MetalSamplingOp::awaitToken: no enqueueForward() outstanding" );
            }

            context_->synchronize();

            const std::int32_t token = *static_cast<const std::int32_t*>( static_cast<const ITensor*>( pending_token_ )->rawData() );

            pending_token_ = nullptr;

            return token;
        }

        /// Encode clearing the reply's produced tokens.
        void enqueueClearProduced( ITensor& produced ) const
        {
            requireProduced( produced );
            context_->zeroBytes( produced.rawData(), static_cast<std::size_t>( config_.getVocabularySize() ) );
        }

        [[nodiscard]] OperationType getOperationType() const override
        {
            return OperationType::SamplingOp;
        }

        [[nodiscard]] std::string getName() const override
        {
            return "Metal.Sampling";
        }

    private:

        // Sampling.metal's pipeline shape, which sizes the scratch and the grids.
        static constexpr std::size_t kSimdgroups = 8;
        static constexpr dim_t kThreads = 256;
        static constexpr dim_t kMaximumChunks = 1024;
        static constexpr dim_t kHistogramGroups = 64;
        static constexpr std::int32_t kRadixRounds = 3;
        static constexpr dim_t kFloatScratchPerRow = 16 + kMaximumChunks + 4096 + kMaximumChunks;
        static constexpr dim_t kCountScratchPerRow = 16 + 4096;

        // The kernels' SamplingParameters, field for field.
        struct Parameters
        {
            std::int32_t vocab;
            std::int32_t top_k;
            float top_p;
            float temperature;
            float softcap;
            float presence_penalty;
            std::uint32_t use_suppressed;
            std::uint32_t use_produced;
            std::uint32_t use_round;
            std::array<float, kMaximumRows> uniforms;
            std::int32_t round;
            std::int32_t partial_count;
            std::int32_t chunk_size;
            std::int32_t chunk_count;
        };

        // Every kernel this op dispatches, resolved once when it is built: a dispatch names none.
        struct Pipelines
        {
            MetalPipeline greedy;
            MetalPipeline scale;
            MetalPipeline prepare;
            MetalPipeline top_k_count;
            MetalPipeline top_k_select;
            MetalPipeline top_p_mass;
            MetalPipeline top_p_select;
            MetalPipeline probability;
            MetalPipeline draw;
            MetalPipeline mark_produced;
            MetalPipeline accept;
        };

        [[nodiscard]] static Pipelines resolvePipelines( MetalKernelLibrary& kernels )
        {
            const std::string type( MetalTensorDataTypeMap<TPrecision>::kernel_type );

            return Pipelines{
                kernels.pipeline( "sampling_greedy_" + type ),
                kernels.pipeline( "sampling_scale_" + type ),
                kernels.pipeline( "sampling_prepare" ),
                kernels.pipeline( "sampling_top_k_count" ),
                kernels.pipeline( "sampling_top_k_select" ),
                kernels.pipeline( "sampling_top_p_mass" ),
                kernels.pipeline( "sampling_top_p_select" ),
                kernels.pipeline( "sampling_probability" ),
                kernels.pipeline( "sampling_draw" ),
                kernels.pipeline( "sampling_mark_produced" ),
                kernels.pipeline( "sampling_accept" ) };
        }

        [[nodiscard]] static constexpr dim_t ceilingDivision( dim_t numerator, dim_t denominator )
        {
            return ( numerator + denominator - 1 ) / denominator;
        }

        // The kernels' SamplingAcceptParameters.
        struct AcceptParameters
        {
            std::int32_t rows;
            std::uint32_t use_produced;
        };

        [[nodiscard]] static SamplingConfig validated( const SamplingConfig& config )
        {
            config.validate();

            if ( config.getMaximumRows() > kMaximumRows )
            {
                throw std::invalid_argument( std::format(
                    "MetalSamplingOp: {} maximum rows, where one call samples at most {}", config.getMaximumRows(), kMaximumRows ) );
            }

            return config;
        }

        void requireRows( dim_t rows, const char* entry ) const
        {
            if ( rows < 1 || rows > config_.getMaximumRows() )
            {
                throw std::invalid_argument( std::format(
                    "MetalSamplingOp::{}: {} rows, where this sampler was built for 1 to {}", entry, rows, config_.getMaximumRows() ) );
            }
        }

        void uploadSuppressedTokens()
        {
            const auto& tokens = config_.getSuppressedTokens();

            if ( tokens.empty() )
            {
                return;
            }

            const std::int64_t vocab = config_.getVocabularySize();

            suppressed_ = std::make_unique<Tensor<TensorDataType::UINT8, MetalMemoryResource>>( context_->getDeviceId(), shape_t{ vocab } );

            std::vector<std::int32_t> flags( static_cast<std::size_t>( vocab ), 0 );

            for ( const std::int32_t token : tokens )
            {
                flags[static_cast<std::size_t>( token )] = 1;
            }

            fill( *suppressed_, std::span<const std::int32_t>( flags ) );
        }

        void requireProduced( const ITensor& produced ) const
        {
            if ( produced.getDataType() != TensorDataType::UINT8 || produced.size() < config_.getVocabularySize() )
            {
                throw std::invalid_argument( std::format(
                    "MetalSamplingOp: the produced tokens are one UINT8 flag per token of the {} in the vocabulary; given {} of {}",
                    config_.getVocabularySize(), produced.size(), produced.getDataTypeName() ) );
            }
        }

        void encodeLastRow( const ITensor& logits, ITensor& token_out, const SamplingParams& params, float r, ITensor* produced ) const
        {
            const auto* const row = static_cast<const std::byte*>( logits.rawData() )
                + metalElementBytes<TPrecision>( logits.size() - config_.getVocabularySize() );

            encodeRows( row, token_out, 1, params, std::span<const float>( &r, 1 ), produced, nullptr );

            if ( params.presence_penalty != 0.0f )
            {
                const std::array<const void*, 2> buffers{ token_out.rawData(), produced->rawData() };

                context_->dispatch( pipelines_.mark_produced, buffers, {}, 1 );
            }
        }

        void encodeRows(
            const void* first_row, ITensor& tokens_out, dim_t rows, const SamplingParams& params, std::span<const float> uniforms,
            const ITensor* produced, const ITensor* round_tokens ) const
        {
            if ( !std::isfinite( params.presence_penalty ) )
            {
                throw std::invalid_argument( std::format( "MetalSamplingOp: the presence penalty must be finite; given {}", params.presence_penalty ) );
            }

            const bool penalized = params.presence_penalty != 0.0f;

            if ( penalized )
            {
                if ( produced == nullptr )
                {
                    throw std::invalid_argument( "MetalSamplingOp: a presence penalty needs the reply's produced tokens" );
                }

                requireProduced( *produced );
            }

            const bool with_round = penalized && round_tokens != nullptr;

            const dim_t vocab = config_.getVocabularySize();
            const dim_t element_groups = ceilingDivision( vocab, kThreads );
            const dim_t scale_groups = std::min( element_groups, kMaximumChunks );
            const dim_t histogram_groups = std::min( element_groups, kHistogramGroups );
            const dim_t chunk_size = kThreads * ceilingDivision( vocab, kThreads * kMaximumChunks );
            const dim_t chunk_count = ceilingDivision( vocab, chunk_size );

            Parameters parameters{
                narrowToKernelIndex( vocab ), params.top_k, params.top_p, params.temperature,
                config_.getFinalLogitSoftcap(), params.presence_penalty, suppressed_ ? 1u : 0u, penalized ? 1u : 0u, with_round ? 1u : 0u, {},
                0, narrowToKernelIndex( scale_groups ), narrowToKernelIndex( chunk_size ), narrowToKernelIndex( chunk_count ) };

            for ( dim_t row = 0; row < rows; ++row )
            {
                parameters.uniforms[static_cast<std::size_t>( row )] = uniforms[static_cast<std::size_t>( row )];
            }

            // What is absent is bound to something valid, and the kernel's flags keep it unread.
            const void* const scratch = scratch_.rawData();
            const void* const suppressed = suppressed_ ? static_cast<const ITensor*>( suppressed_.get() )->rawData() : scratch;
            const void* const flags = penalized ? produced->rawData() : scratch;
            const void* const round = with_round ? round_tokens->rawData() : scratch;
            const auto height = static_cast<std::size_t>( rows );

            if ( params.temperature <= 0.0f || params.top_k == 1 )
            {
                const std::array<const void*, 5> buffers{ first_row, tokens_out.rawData(), suppressed, flags, round };

                context_->dispatchThreadgroups( pipelines_.greedy, buffers, std::as_bytes( std::span( &parameters, 1 ) ), 1, height, kSimdgroups );

                return;
            }

            const std::array<const void*, 8> buffers{
                first_row, tokens_out.rawData(), scratch, reduction_scratch_.rawData(), count_scratch_.rawData(), suppressed, flags, round };

            // Each stage is told its radix round; the rest of the parameters are the call's.
            auto stage = [&]( MetalPipeline pipeline, dim_t width, std::int32_t radix_round )
                {
                    Parameters staged = parameters;
                    staged.round = radix_round;

                    context_->dispatchThreadgroups(
                        pipeline, buffers, std::as_bytes( std::span( &staged, 1 ) ), static_cast<std::size_t>( width ), height, kSimdgroups );
                };

            stage( pipelines_.scale, scale_groups, 0 );
            stage( pipelines_.prepare, 1, 0 );

            if ( params.top_k > 0 && params.top_k < vocab )
            {
                for ( std::int32_t radix_round = 0; radix_round < kRadixRounds; ++radix_round )
                {
                    stage( pipelines_.top_k_count, histogram_groups, radix_round );
                    stage( pipelines_.top_k_select, 1, radix_round );
                }
            }

            if ( params.top_p < 1.0f )
            {
                for ( std::int32_t radix_round = 0; radix_round < kRadixRounds; ++radix_round )
                {
                    stage( pipelines_.top_p_mass, histogram_groups, radix_round );
                    stage( pipelines_.top_p_select, 1, radix_round );
                }
            }

            stage( pipelines_.probability, chunk_count, 0 );
            stage( pipelines_.draw, 1, 0 );
        }

        MetalExecutionContext* context_;
        SamplingConfig config_;
        mutable Tensor<TensorDataType::FP32, MetalMemoryResource> scratch_;
        mutable Tensor<TensorDataType::FP32, MetalMemoryResource> reduction_scratch_;
        mutable Tensor<TensorDataType::INT32, MetalMemoryResource> count_scratch_;
        Pipelines pipelines_;
        std::unique_ptr<Tensor<TensorDataType::UINT8, MetalMemoryResource>> suppressed_;
        mutable ITensor* pending_token_{ nullptr };
    };
}
