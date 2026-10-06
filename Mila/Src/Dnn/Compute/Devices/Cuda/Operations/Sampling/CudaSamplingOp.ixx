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
#include <string>
#include <stdexcept>
#include <format>
#include <cstdint>
#include <span>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "Kernels/Sampling.cuh"

export module Compute.CudaSamplingOp;

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
     * and writes the sampled token into @p token_out on the device. Two entry points:
     * forward() is the synchronous contract (default stream; the caller synchronizes
     * the network first and reads the token back itself), enqueueForward()/awaitToken()
     * is the pipelined decode-ahead contract (execution context's stream, async pinned
     * readback -- no host sync between forward pass and sampler), and enqueueForwardOnDevice()
     * leaves the token on the device with no readback at all. Internal compute is FP32.
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
         */
        void forward(
            const ITensor& logits,
            ITensor& token_out,
            const SamplingParams& params,
            float r ) const
        {
            // Synchronous-contract path: runs on the default stream (0). The model
            // synchronizes the network before sampling, so the logits are complete, and
            // the TokenSampler reads the token back synchronously, so default-stream
            // ordering is correct. The pipelined decode path uses enqueueForward().
            dispatchSample( logits, token_out, params, r, 0 );
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
            float r ) const
        {
            cudaStream_t stream = context_->getStream();

            dispatchSample( logits, token_out, params, r, stream );

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
                pinned_token_.rawData(), token_out.rawData(), sizeof( int32_t ),
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
            float r ) const
        {
            dispatchSample( logits, token_out, params, r, context_->getStream() );
        }

        /**
         * @brief Sample each of the first @p rows logits rows on the execution context's stream, all rows in each
         * launch, the tokens left on the device.
         *
         * Row r is drawn against `uniforms[r]` into element r of @p tokens_out, exactly as enqueueForwardOnDevice()
         * draws that row alone with that uniform -- the speculative sampler's first half (Gemma4Mtp.md 4.8).
         *
         * @param logits     Device logits whose first `rows x vocab_size` elements are the rows, row-major.
         * @param tokens_out Device INT32 of at least @p rows elements.
         * @param rows       1 to the configured maximum rows.
         * @param params     Per-call sampling parameters, the same for every row.
         * @param uniforms   One host-drawn uniform in [0, 1) per row; unused by the greedy branch.
         */
        void enqueueRowsOnDevice(
            const ITensor& logits,
            ITensor& tokens_out,
            dim_t rows,
            const SamplingParams& params,
            std::span<const float> uniforms ) const
        {
            const int64_t vocab = config_.getVocabularySize();

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

            const NativeType* first_row = static_cast<const NativeType*>( logits.rawData() );
            int32_t* out = static_cast<int32_t*>( tokens_out.rawData() );
            const int row_count = narrowToKernelIndex( rows );
            cudaStream_t stream = context_->getStream();

            const bool greedy = (params.temperature <= 0.0f || params.top_k == 1);

            if ( greedy )
            {
                cuda_sample_rows_argmax<NativeType>( first_row, out, row_count, static_cast<int>( vocab ), stream );
                return;
            }

            SampleRowUniforms row_uniforms{};

            for ( int row = 0; row < row_count; ++row )
                row_uniforms.value[ row ] = uniforms[ static_cast<size_t>( row ) ];

            cuda_sample_rows_stochastic<NativeType>(
                first_row, out,
                static_cast<float*>( prob_scratch_.rawData() ),
                static_cast<float*>( reduction_scratch_.rawData() ),
                static_cast<int32_t*>( index_scratch_.rawData() ),
                row_count, static_cast<int>( vocab ),
                config_.getFinalLogitSoftcap(), params.temperature,
                params.top_k, params.top_p, row_uniforms, stream );
        }

        /**
         * @brief Walk a verified round's acceptance on the execution context's stream, the speculative sampler's
         * second half (Gemma4Mtp.md 4.8).
         *
         * A draft is kept while the row before it chose it. Writes `[m, d_1 .. d_m, next]` into @p result -- the
         * count kept, the kept drafts and the token chosen at row m -- and the next token into slot 0 of @p tokens,
         * where the next round's draft reads it.
         *
         * @param tokens Device INT32 of the round's @p rows tokens: the known one, then the drafts.
         * @param chosen Device INT32 of the token chosen at each of the @p rows rows (enqueueRowsOnDevice()).
         * @param rows   1 to the configured maximum rows.
         * @param result Device INT32 of at least `rows + 1` elements.
         */
        void enqueueAcceptOnDevice(
            ITensor& tokens,
            const ITensor& chosen,
            dim_t rows,
            ITensor& result ) const
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

            cuda_speculative_accept(
                static_cast<int32_t*>( tokens.rawData() ),
                static_cast<const int32_t*>( chosen.rawData() ),
                narrowToKernelIndex( rows ),
                static_cast<int32_t*>( result.rawData() ),
                context_->getStream() );
        }

        /**
         * @brief Block until the last enqueueForward()'s token readback lands, then
         * return the host token id.
         *
         * The event is the final item enqueued, so returning also means every kernel of
         * that sampling step has completed. Work enqueued on the stream after the event
         * (the decode-ahead forward) is NOT waited on -- that is the point.
         */
        int32_t awaitToken() const
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

            return static_cast<const int32_t*>( pinned_token_.rawData() )[ 0 ];
        }

        /**
         * @brief forward() through the retained single-block reference kernel.
         *
         * Parity oracle for the multi-block stochastic pipeline (same semantics up
         * to float reduction order at truncation/CDF boundaries) — tests only,
         * ~11 ms/token at a 262k vocab. The greedy branch is shared with forward().
         */
        void forwardReference(
            const ITensor& logits,
            ITensor& token_out,
            const SamplingParams& params,
            float r ) const
        {
            const int64_t vocab = config_.getVocabularySize();
            const dim_t offset = logits.size() - vocab;

            const NativeType* row = static_cast<const NativeType*>( logits.rawData() ) + offset;
            int32_t* out = static_cast<int32_t*>( token_out.rawData() );

            cudaStream_t stream = 0;

            const bool greedy = (params.temperature <= 0.0f || params.top_k == 1);

            if (greedy)
            {
                cuda_sample_argmax<NativeType>( row, out, static_cast<int>( vocab ), stream );
                return;
            }

            float* scratch = static_cast<float*>( prob_scratch_.rawData() );

            cuda_sample_stochastic_reference<NativeType>(
                row, out, scratch, static_cast<int>( vocab ),
                config_.getFinalLogitSoftcap(), params.temperature,
                params.top_k, params.top_p, r, stream );
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

        /// Kernel dispatch shared by the sync (default-stream) and enqueued (context-stream) paths.
        void dispatchSample(
            const ITensor& logits,
            ITensor& token_out,
            const SamplingParams& params,
            float r,
            cudaStream_t stream ) const
        {
            const int64_t vocab = config_.getVocabularySize();
            const dim_t offset = logits.size() - vocab;

            const NativeType* row = static_cast<const NativeType*>( logits.rawData() ) + offset;
            int32_t* out = static_cast<int32_t*>( token_out.rawData() );

            const bool greedy = (params.temperature <= 0.0f || params.top_k == 1);

            if (greedy)
            {
                cuda_sample_argmax<NativeType>( row, out, static_cast<int>( vocab ), stream );
                return;
            }

            float* scratch = static_cast<float*>( prob_scratch_.rawData() );
            float* reduction_scratch = static_cast<float*>( reduction_scratch_.rawData() );
            int32_t* index_scratch = static_cast<int32_t*>( index_scratch_.rawData() );

            cuda_sample_stochastic<NativeType>(
                row, out, scratch, reduction_scratch, index_scratch,
                static_cast<int>( vocab ),
                config_.getFinalLogitSoftcap(), params.temperature,
                params.top_k, params.top_p, r, stream );
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

        // Enqueued-path readback state: the async 4-byte D2H lands in the pinned slot,
        // and the event marks it complete. Single-slot by contract -- one outstanding
        // enqueueForward() at a time (awaitToken() is called before the next enqueue,
        // so the slot is never overwritten while the host still needs it).
        mutable Tensor<TensorDataType::INT32, CudaPinnedMemoryResource> pinned_token_;
        mutable cudaEvent_t token_ready_event_{ nullptr };
    };
}
