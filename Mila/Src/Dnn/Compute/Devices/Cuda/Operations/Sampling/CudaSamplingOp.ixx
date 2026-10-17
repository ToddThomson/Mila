/**
 * @file CudaSamplingOp.ixx
 * @brief CUDA device-side token sampling operation.
 *
 * Greedy (argmax) and stochastic top-k/top-p sampling on the device. The logits
 * never leave the device; the op writes the chosen int32 token into the caller's
 * device buffer. The stochastic path is a multi-block kernel pipeline; the original
 * single-block kernel is reachable via forwardReference() as the parity oracle.
 * See Specifications/TokenSampling.md.
 */

module;
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "Kernels/Sampling.cuh"

export module Compute.CudaSamplingOp;

import std;
import Dnn.Samplers.SamplingConfig;
import Dnn.SamplingParams;
import Dnn.Component;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Compute.OperationBase;
import Compute.DeviceType;
import Compute.IExecutionContext;
import Compute.ExecutionContext;
import Compute.OperationType;
import Compute.MemoryResource;
import Compute.CudaDeviceMemoryResource;
import Compute.CudaPinnedMemoryResource;
import Compute.CudaDevice;
import Compute.CudaTensorDataType;

namespace Mila::Dnn::Compute::Cuda::Sampling
{
    using namespace Mila::Dnn;

    /**
     * @brief CUDA token sampler op: maps a logits row to a single int32 token id.
     *
     * Reads the final `vocab_size` elements of @p logits (the last decode position)
     * and writes the sampled token into @p token_out on the device. Every entry runs on the
     * execution context's stream. forward() leaves the readback to the caller, who copies the
     * token with the same context; enqueueForward()/awaitToken() is the pipelined decode-ahead
     * contract (async pinned readback -- no host sync between forward pass and sampler), and
     * enqueueForwardOnDevice() leaves the token on the device with no readback at all. Internal
     * compute is FP32.
     *
     * The presence penalty reads the reply's produced tokens, a vocabulary-sized UINT8 device
     * tensor the caller owns and passes in; the op holds no state of its own across calls, and
     * each single-row entry marks the token it drew there when the penalty is on.
     *
     * @tparam TPrecision Logits precision (FP32 or BF16).
     */
    export template<TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, DeviceType::Cuda>
    class CudaSamplingOp : public Operation<DeviceType::Cuda, TPrecision>
    {
    public:
        using NativeType = typename Cuda::TensorDataTypeMap<TPrecision>::device_type;
        using CudaExecutionContext = ExecutionContext<DeviceType::Cuda>;

        CudaSamplingOp( IExecutionContext* context, const SamplingConfig& config )
            : context_( validateExecutionContext_<DeviceType::Cuda>( context, "CudaSamplingOp" ) ),
              config_( validatedConfig( config ) ),
              prob_scratch_( context->getDeviceId(), shape_t{ config.getMaximumRows() * config.getVocabularySize() } ),
              reduction_scratch_( context->getDeviceId(),
                  shape_t{ config.getMaximumRows() * kStochasticFloatScratchElements } ),
              index_scratch_( context->getDeviceId(),
                  shape_t{ config.getMaximumRows() * kStochasticIndexScratchElements } ),
              pinned_token_( context->getDeviceId(), shape_t{ 1, 1 } )
        {
            uploadSuppressedTokens();
        }

        ~CudaSamplingOp() override
        {
            if ( token_ready_event_ )
            {
                cudaEventDestroy( token_ready_event_ );
                token_ready_event_ = nullptr;
            }
        }

        /**
         * @brief Sample one token from the final logits row.
         *
         * @param logits    Device logits; the last `vocab_size` elements are the active row.
         * @param token_out Device INT32 [1,1] tensor; written in place on the stream.
         * @param params    Per-call sampling parameters.
         * @param r         Host-drawn uniform in [0,1) (unused by the greedy branch).
         * @param produced  The reply's produced tokens, required when the presence penalty is on and marked with the
         *                  token drawn.
         */
        void forward(
            const ITensor& logits,
            ITensor& token_out,
            const SamplingParams& params,
            float r,
            ITensor* produced = nullptr ) const
        {
            // On the context's stream, ordered after the forward pass that wrote the logits whatever
            // the stream's flags; the caller reads the token back by a copy on the same context.
            dispatchSample( logits, token_out, params, r, produced, context_->getStream() );
        }

        /**
         * @brief Enqueue one sampling step on the execution context's stream, without
         * any host synchronization.
         *
         * Stream-ordered after the forward pass that produced @p logits (same stream),
         * so no prior network synchronize is required. Besides the device token write,
         * an async 4-byte device->host copy into a pinned slot plus an event record are
         * enqueued; awaitToken() blocks on that event and returns the host value. At most
         * one enqueue may be outstanding -- awaitToken() must be called before the next
         * enqueueForward() (the decode-ahead loop's natural order).
         */
        void enqueueForward(
            const ITensor& logits,
            ITensor& token_out,
            const SamplingParams& params,
            float r,
            ITensor* produced = nullptr ) const
        {
            cudaStream_t stream = context_->getStream();

            dispatchSample( logits, token_out, params, r, produced, stream );

            if ( !token_ready_event_ )
            {
                cudaError_t status = cudaEventCreateWithFlags( &token_ready_event_, cudaEventDisableTiming );

                if ( status != cudaSuccess )
                {
                    throw std::runtime_error( std::format(
                        "CudaSamplingOp::enqueueForward: event creation failed: {}",
                        cudaGetErrorString( status ) ) );
                }
            }

            cudaError_t status = cudaMemcpyAsync(
                pinned_token_.rawData(), token_out.rawData(), sizeof( std::int32_t ),
                cudaMemcpyDeviceToHost, stream );

            if ( status != cudaSuccess )
            {
                throw std::runtime_error( std::format(
                    "CudaSamplingOp::enqueueForward: token readback enqueue failed: {}",
                    cudaGetErrorString( status ) ) );
            }

            status = cudaEventRecord( token_ready_event_, stream );

            if ( status != cudaSuccess )
            {
                throw std::runtime_error( std::format(
                    "CudaSamplingOp::enqueueForward: event record failed: {}",
                    cudaGetErrorString( status ) ) );
            }
        }

        /**
         * @brief Enqueue one sampling step on the execution context's stream, the token never read back.
         *
         * The token stays where the next pass on the stream reads it, as a draft step's does for the step after it
         * (Gemma4Mtp.md 4.6); the caller copies it to the host when it needs it. Any number may be outstanding, and
         * none touches the single slot enqueueForward()'s readback uses.
         */
        void enqueueForwardOnDevice(
            const ITensor& logits,
            ITensor& token_out,
            const SamplingParams& params,
            float r,
            ITensor* produced = nullptr ) const
        {
            dispatchSample( logits, token_out, params, r, produced, context_->getStream() );
        }

        /**
         * @brief Sample each of the first @p rows logits rows on the execution context's stream, all rows in each
         * launch, the tokens left on the device.
         *
         * Row r is drawn against `uniforms[r]` into element r of @p tokens_out, exactly as enqueueForwardOnDevice()
         * draws that row alone with that uniform -- the speculative sampler's first half (Gemma4Mtp.md 4.8). Nothing
         * is marked produced: the acceptance walk marks what the round keeps.
         *
         * @param logits       Device logits whose first `rows x vocab_size` elements are the rows, row-major.
         * @param tokens_out   Device INT32 of at least @p rows elements.
         * @param rows         1 to the configured maximum rows.
         * @param params       Per-call sampling parameters, the same for every row.
         * @param uniforms     One host-drawn uniform in [0, 1) per row; unused by the greedy branch.
         * @param produced     The reply's produced tokens, required when the presence penalty is on.
         * @param round_tokens The round's tokens, the known one then the drafts; with the penalty on, row r also
         *                     penalizes drafts 1..r. Null penalizes no draft.
         */
        void enqueueRowsOnDevice(
            const ITensor& logits,
            ITensor& tokens_out,
            dim_t rows,
            const SamplingParams& params,
            std::span<const float> uniforms,
            const ITensor* produced = nullptr,
            const ITensor* round_tokens = nullptr ) const
        {
            const std::int64_t vocab = config_.getVocabularySize();

            if ( rows < 1 || rows > config_.getMaximumRows() )
            {
                throw std::invalid_argument( std::format(
                    "CudaSamplingOp::enqueueRowsOnDevice: {} rows, where this sampler was built for 1 to {}",
                    rows, config_.getMaximumRows() ) );
            }

            if ( static_cast<dim_t>( uniforms.size() ) < rows || logits.size() < rows * vocab || tokens_out.size() < rows )
            {
                throw std::invalid_argument( std::format(
                    "CudaSamplingOp::enqueueRowsOnDevice: {} rows need {} uniforms, {} logits and {} token slots; "
                    "given {}, {} and {}",
                    rows, rows, rows * vocab, rows, uniforms.size(), logits.size(), tokens_out.size() ) );
            }

            if ( round_tokens && round_tokens->size() < rows )
            {
                throw std::invalid_argument( std::format(
                    "CudaSamplingOp::enqueueRowsOnDevice: {} rows need {} round tokens; given {}",
                    rows, rows, round_tokens->size() ) );
            }

            const SampleAdjustments adjustments = adjustmentsFor( params, produced, round_tokens );

            const NativeType* first_row = static_cast<const NativeType*>( logits.rawData() );
            std::int32_t* out = static_cast<std::int32_t*>( tokens_out.rawData() );
            const int row_count = narrowToKernelIndex( rows );
            cudaStream_t stream = context_->getStream();

            const bool greedy = (params.temperature <= 0.0f || params.top_k == 1);

            if ( greedy )
            {
                cuda_sample_rows_argmax<NativeType>( first_row, out, row_count, static_cast<int>( vocab ),
                    config_.getFinalLogitSoftcap(), adjustments, stream );
                return;
            }

            SampleRowUniforms row_uniforms{};

            for ( int row = 0; row < row_count; ++row )
                row_uniforms.value[ row ] = uniforms[ static_cast<std::size_t>( row ) ];

            cuda_sample_rows_stochastic<NativeType>(
                first_row, out,
                static_cast<float*>( prob_scratch_.rawData() ),
                static_cast<float*>( reduction_scratch_.rawData() ),
                static_cast<std::int32_t*>( index_scratch_.rawData() ),
                row_count, static_cast<int>( vocab ),
                config_.getFinalLogitSoftcap(), params.temperature,
                params.top_k, params.top_p, adjustments, row_uniforms, stream );
        }

        /**
         * @brief Walk a verified round's acceptance on the execution context's stream, the speculative sampler's
         * second half (Gemma4Mtp.md 4.8).
         *
         * A draft is kept while the row before it chose it. Writes `[m, d_1 .. d_m, next]` into @p result -- the
         * count kept, the kept drafts and the token chosen at row m -- and the next token into slot 0 of @p tokens,
         * where the next round's draft reads it.
         *
         * @param tokens   Device INT32 of the round's @p rows tokens: the known one, then the drafts.
         * @param chosen   Device INT32 of the token chosen at each of the @p rows rows (enqueueRowsOnDevice()).
         * @param rows     1 to the configured maximum rows.
         * @param result   Device INT32 of at least `rows + 1` elements.
         * @param produced The reply's produced tokens, which gain the kept drafts and the next token; null when the
         *                 presence penalty is off.
         */
        void enqueueAcceptOnDevice(
            ITensor& tokens,
            const ITensor& chosen,
            dim_t rows,
            ITensor& result,
            ITensor* produced = nullptr ) const
        {
            if ( rows < 1 || rows > config_.getMaximumRows() )
            {
                throw std::invalid_argument( std::format(
                    "CudaSamplingOp::enqueueAcceptOnDevice: {} rows, where this sampler was built for 1 to {}",
                    rows, config_.getMaximumRows() ) );
            }

            if ( tokens.size() < rows || chosen.size() < rows || result.size() < rows + 1 )
            {
                throw std::invalid_argument( std::format(
                    "CudaSamplingOp::enqueueAcceptOnDevice: {} rows need {} tokens, {} chosen and {} result slots; "
                    "given {}, {} and {}",
                    rows, rows, rows, rows + 1, tokens.size(), chosen.size(), result.size() ) );
            }

            if ( produced )
                requireProducedTokens( *produced, "enqueueAcceptOnDevice" );

            cuda_speculative_accept(
                static_cast<std::int32_t*>( tokens.rawData() ),
                static_cast<const std::int32_t*>( chosen.rawData() ),
                narrowToKernelIndex( rows ),
                static_cast<std::int32_t*>( result.rawData() ),
                produced ? static_cast<std::uint8_t*>( produced->rawData() ) : nullptr,
                context_->getStream() );
        }

        /**
         * @brief Clear the reply's produced tokens on the execution context's stream, so a new reply starts with none.
         */
        void enqueueClearProduced( ITensor& produced ) const
        {
            requireProducedTokens( produced, "enqueueClearProduced" );

            const cudaError_t status = cudaMemsetAsync(
                produced.rawData(), 0, static_cast<std::size_t>( config_.getVocabularySize() ), context_->getStream() );

            if ( status != cudaSuccess )
            {
                throw std::runtime_error( std::format(
                    "CudaSamplingOp::enqueueClearProduced: clearing failed: {}", cudaGetErrorString( status ) ) );
            }
        }

        /**
         * @brief Block until the last enqueueForward()'s token readback lands, then
         * return the host token id.
         *
         * The event is the final item enqueued, so returning also means every kernel of
         * that sampling step has completed. Work enqueued on the stream after the event
         * (the decode-ahead forward) is NOT waited on -- that is the point.
         */
        std::int32_t awaitToken() const
        {
            if ( !token_ready_event_ )
            {
                throw std::logic_error(
                    "CudaSamplingOp::awaitToken: no enqueueForward() outstanding" );
            }

            cudaError_t status = cudaEventSynchronize( token_ready_event_ );

            if ( status != cudaSuccess )
            {
                throw std::runtime_error( std::format(
                    "CudaSamplingOp::awaitToken: event synchronize failed: {}",
                    cudaGetErrorString( status ) ) );
            }

            return static_cast<const std::int32_t*>( pinned_token_.rawData() )[ 0 ];
        }

        /**
         * @brief forward() through the retained single-block reference kernel, on the execution context's stream.
         *
         * Parity oracle for the multi-block stochastic pipeline (same semantics up
         * to float reduction order at truncation/CDF boundaries) — tests only,
         * ~11 ms/token at a 262k vocab. The greedy branch is shared with forward().
         */
        void forwardReference(
            const ITensor& logits,
            ITensor& token_out,
            const SamplingParams& params,
            float r,
            ITensor* produced = nullptr ) const
        {
            const std::int64_t vocab = config_.getVocabularySize();
            const dim_t offset = logits.size() - vocab;

            const NativeType* row = static_cast<const NativeType*>( logits.rawData() ) + offset;
            std::int32_t* out = static_cast<std::int32_t*>( token_out.rawData() );

            cudaStream_t stream = context_->getStream();
            const SampleAdjustments adjustments = adjustmentsFor( params, produced, nullptr );

            const bool greedy = (params.temperature <= 0.0f || params.top_k == 1);

            if (greedy)
            {
                cuda_sample_argmax<NativeType>(
                    row, out, static_cast<int>( vocab ), config_.getFinalLogitSoftcap(), adjustments, stream );
            }
            else
            {
                float* scratch = static_cast<float*>( prob_scratch_.rawData() );

                cuda_sample_stochastic_reference<NativeType>(
                    row, out, scratch, static_cast<int>( vocab ),
                    config_.getFinalLogitSoftcap(), params.temperature,
                    params.top_k, params.top_p, adjustments, r, stream );
            }

            markProduced( token_out, params, produced, stream );
        }

        OperationType getOperationType() const override
        {
            return OperationType::SamplingOp;
        }

        std::string getName() const override
        {
            return "Cuda::SamplingOp";
        }

    private:

        static SamplingConfig validatedConfig( const SamplingConfig& config )
        {
            config.validate();

            if ( config.getMaximumRows() > kMaxSampleRows )
            {
                throw std::invalid_argument( std::format(
                    "CudaSamplingOp: {} maximum rows, where one call samples at most {}",
                    config.getMaximumRows(), kMaxSampleRows ) );
            }

            return config;
        }

        /// The suppressed tokens as a vocabulary flag per token on the device, built once; none allocated when empty.
        void uploadSuppressedTokens()
        {
            const auto& tokens = config_.getSuppressedTokens();

            if ( tokens.empty() )
                return;

            const std::int64_t vocab = config_.getVocabularySize();
            std::vector<std::uint8_t> flags( static_cast<std::size_t>( vocab ), 0 );

            for ( const std::int32_t token : tokens )
                flags[ static_cast<std::size_t>( token ) ] = 1;

            suppressed_ = std::make_unique<Tensor<TensorDataType::UINT8, CudaDeviceMemoryResource>>(
                context_->getDeviceId(), shape_t{ vocab } );

            const cudaError_t status = cudaMemcpy(
                suppressed_->rawData(), flags.data(), flags.size(), cudaMemcpyHostToDevice );

            if ( status != cudaSuccess )
            {
                throw std::runtime_error( std::format(
                    "CudaSamplingOp: uploading the suppressed tokens failed: {}", cudaGetErrorString( status ) ) );
            }
        }

        void requireProducedTokens( const ITensor& produced, const char* entry ) const
        {
            if ( produced.getDataType() != TensorDataType::UINT8 || produced.size() < config_.getVocabularySize() )
            {
                throw std::invalid_argument( std::format(
                    "CudaSamplingOp::{}: the produced tokens are one UINT8 flag per token of the {} in the vocabulary; "
                    "given {} of {}",
                    entry, config_.getVocabularySize(), produced.size(), produced.getDataTypeName() ) );
            }
        }

        SampleAdjustments adjustmentsFor(
            const SamplingParams& params, const ITensor* produced, const ITensor* round_tokens ) const
        {
            if ( !std::isfinite( params.presence_penalty ) )
            {
                throw std::invalid_argument( std::format(
                    "CudaSamplingOp: the presence penalty must be finite; given {}", params.presence_penalty ) );
            }

            SampleAdjustments adjustments{};
            adjustments.suppressed = suppressed_ ? static_cast<const std::uint8_t*>( suppressed_->rawData() ) : nullptr;

            if ( params.presence_penalty != 0.0f )
            {
                if ( !produced )
                {
                    throw std::invalid_argument(
                        "CudaSamplingOp: a presence penalty needs the reply's produced tokens" );
                }

                requireProducedTokens( *produced, "sample" );

                adjustments.produced = static_cast<const std::uint8_t*>( produced->rawData() );
                adjustments.round_tokens = round_tokens
                    ? static_cast<const std::int32_t*>( round_tokens->rawData() ) : nullptr;
                adjustments.presence_penalty = params.presence_penalty;
            }

            return adjustments;
        }

        /// With the presence penalty on, flags the token just drawn as produced, on the stream after the draw.
        void markProduced(
            ITensor& token_out, const SamplingParams& params, ITensor* produced, cudaStream_t stream ) const
        {
            if ( params.presence_penalty == 0.0f )
                return;

            cuda_mark_produced( static_cast<const std::int32_t*>( token_out.rawData() ),
                static_cast<std::uint8_t*>( produced->rawData() ), stream );
        }

        /// Kernel dispatch shared by every single-row entry; each passes the execution context's stream.
        void dispatchSample(
            const ITensor& logits,
            ITensor& token_out,
            const SamplingParams& params,
            float r,
            ITensor* produced,
            cudaStream_t stream ) const
        {
            const std::int64_t vocab = config_.getVocabularySize();
            const dim_t offset = logits.size() - vocab;

            const NativeType* row = static_cast<const NativeType*>( logits.rawData() ) + offset;
            std::int32_t* out = static_cast<std::int32_t*>( token_out.rawData() );

            const SampleAdjustments adjustments = adjustmentsFor( params, produced, nullptr );

            const bool greedy = (params.temperature <= 0.0f || params.top_k == 1);

            if (greedy)
            {
                cuda_sample_argmax<NativeType>(
                    row, out, static_cast<int>( vocab ), config_.getFinalLogitSoftcap(), adjustments, stream );
            }
            else
            {
                float* scratch = static_cast<float*>( prob_scratch_.rawData() );
                float* reduction_scratch = static_cast<float*>( reduction_scratch_.rawData() );
                std::int32_t* index_scratch = static_cast<std::int32_t*>( index_scratch_.rawData() );

                cuda_sample_stochastic<NativeType>(
                    row, out, scratch, reduction_scratch, index_scratch,
                    static_cast<int>( vocab ),
                    config_.getFinalLogitSoftcap(), params.temperature,
                    params.top_k, params.top_p, adjustments, r, stream );
            }

            markProduced( token_out, params, produced, stream );
        }

        CudaExecutionContext* context_;
        SamplingConfig config_;
        // Working stores for the stochastic pipeline (vocab FP32 probabilities plus the
        // fixed-size reduction/histogram scratch — see Sampling.cuh for the geometry).
        // Mutable so the const forward() can write through them (the op holds no
        // logical state).
        mutable Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> prob_scratch_;
        mutable Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> reduction_scratch_;
        mutable Tensor<TensorDataType::INT32, CudaDeviceMemoryResource> index_scratch_;

        // One flag per vocabulary token no draw may choose; null when the configuration names none.
        std::unique_ptr<Tensor<TensorDataType::UINT8, CudaDeviceMemoryResource>> suppressed_;

        // Enqueued-path readback state: the async 4-byte D2H lands in the pinned slot,
        // and the event marks it complete. Single-slot by contract -- one outstanding
        // enqueueForward() at a time (awaitToken() is called before the next enqueue,
        // so the slot is never overwritten while the host still needs it).
        mutable Tensor<TensorDataType::INT32, CudaPinnedMemoryResource> pinned_token_;
        mutable cudaEvent_t token_ready_event_{ nullptr };
    };
}
