// Device-side token sampling kernels: greedy argmax and the stochastic multinomial
// with optional top-k / top-p truncation. The stochastic path is a multi-block
// pipeline (histogram threshold refinement + chunked inverse-CDF); the original
// single-block kernel is retained below as the parity reference oracle.

#include "Sampling.cuh"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cfloat>
#include <cstdint>

namespace Mila::Dnn::Compute::Cuda::Sampling
{
    namespace
    {
        __device__ inline float to_float( float v ) { return v; }
        __device__ inline float to_float( __nv_bfloat16 v ) { return __bfloat162float( v ); }

        // Single-block argmax: each thread grid-strides the vocab keeping its local
        // (max value, lowest index), then a shared-memory tree reduction picks the
        // global winner. Ties resolve to the lowest index to match std::max_element.
        template <typename TNative, int kBlock>
        __global__ void argmax_kernel( const TNative* logits, int32_t* token_out, int vocab )
        {
            __shared__ float s_val[kBlock];
            __shared__ int s_idx[kBlock];

            const int tid = threadIdx.x;
            float best = -FLT_MAX;
            int best_idx = 0;

            for ( int i = tid; i < vocab; i += kBlock )
            {
                const float v = to_float( logits[i] );

                if ( v > best )
                {
                    best = v;
                    best_idx = i;
                }
            }

            s_val[tid] = best;
            s_idx[tid] = best_idx;
            __syncthreads();

            for ( int stride = kBlock / 2; stride > 0; stride >>= 1 )
            {
                if ( tid < stride )
                {
                    const float other = s_val[tid + stride];

                    if ( other > s_val[tid] || ( other == s_val[tid] && s_idx[tid + stride] < s_idx[tid] ) )
                    {
                        s_val[tid] = other;
                        s_idx[tid] = s_idx[tid + stride];
                    }
                }

                __syncthreads();
            }

            if ( tid == 0 )
                token_out[0] = s_idx[0];
        }

        template <typename TNative>
        inline void launch_argmax( const TNative* logits, int32_t* token_out, int vocab, cudaStream_t stream )
        {
            constexpr int kBlock = 256;
            argmax_kernel<TNative, kBlock><<<1, kBlock, 0, stream>>>( logits, token_out, vocab );
        }
    }

    void cuda_sample_argmax_fp32( const float* logits, int32_t* token_out, int vocab, cudaStream_t stream )
    {
        launch_argmax( logits, token_out, vocab, stream );
    }

    void cuda_sample_argmax_bf16( const __nv_bfloat16* logits, int32_t* token_out, int vocab, cudaStream_t stream )
    {
        launch_argmax( logits, token_out, vocab, stream );
    }

    // ========================================================================
    // Stochastic multinomial — multi-block pipeline (production path)
    //
    // The single-block reference kernel below costs ~11 ms/token at a 262k vocab
    // (one SM; 40-pass bisection searches + a thread-0 serial CDF walk). The
    // pipeline computes the same truncation semantics with device-wide kernels:
    //
    //   1. scale:     softcap + temperature -> scratch; per-block max/min partials.
    //   2. prepare:   finalize max/min, reset thresholds/bins.
    //   3. top-k:     4 rounds of 4096-bin count-histogram range refinement
    //                 (4096^4 = 2^48 -- strictly finer than the reference's 2^40
    //                 bisection). Integer counts: exact and deterministic.
    //   4. top-p:     same machinery on per-bin exp(x - max) mass, over the top-k
    //                 survivor domain. Probability is monotonic in the scaled
    //                 logit, so both filters collapse to one value threshold.
    //   5. prob:      exp(x - max) for survivors (else 0) -> scratch; contiguous
    //                 per-chunk partial sums (index order preserved).
    //   6. cdf:       one block scans the chunk partials for the target chunk and
    //                 serially walks only that chunk -- token-index-order inverse
    //                 CDF with the reference kernel's survivor guard and vocab-1
    //                 fallback.
    //
    // Truncation boundaries match the reference up to float reduction order: the
    // top-p bin masses accumulate through atomics, so a token exactly on a
    // nucleus boundary may differ by one ulp of threshold across runs. Top-k is
    // count-based and exact. Distribution-level behavior is identical.
    // ========================================================================

    namespace
    {
        // Float-scratch header slots (reduction_scratch).
        constexpr int kSlotMax = 0;
        constexpr int kSlotMin = 1;
        constexpr int kSlotRangeLo = 2;
        constexpr int kSlotRangeHi = 3;
        constexpr int kSlotKThreshold = 4;
        constexpr int kSlotPThreshold = 5;
        constexpr int kSlotMassAbove = 6;
        constexpr int kSlotTotalMass = 7;

        constexpr int kOffsetMaxPartials = kStochasticHeaderFloats;
        constexpr int kOffsetMinPartials = kStochasticHeaderFloats + kStochasticMaxChunks;
        constexpr int kOffsetMassBins = kStochasticHeaderFloats + 2 * kStochasticMaxChunks;
        constexpr int kOffsetChunkPartials = kOffsetMassBins + kStochasticBins;

        // Index-scratch slots.
        constexpr int kSlotCountAbove = 0;
        constexpr int kOffsetCountBins = 16;

        constexpr int kRefinementRounds = 4;
        constexpr int kHistogramBlocks = 64;

        constexpr int ceil_div( int a, int b ) { return ( a + b - 1 ) / b; }

        // Pipeline stage 1: scaled logits -> scratch; per-block max/min partials.
        template <typename TNative, int kBlock>
        __global__ void stochastic_scale_kernel(
            const TNative* logits, float* scratch, int vocab,
            float softcap, float temperature, float* reduction_scratch )
        {
            __shared__ float s_max[kBlock];
            __shared__ float s_min[kBlock];

            const int tid = threadIdx.x;
            const int stride = gridDim.x * kBlock;
            float local_max = -FLT_MAX;
            float local_min = FLT_MAX;

            for ( int i = blockIdx.x * kBlock + tid; i < vocab; i += stride )
            {
                float x = to_float( logits[i] );

                if ( softcap > 0.0f )
                    x = softcap * tanhf( x / softcap );

                x = x / temperature;
                scratch[i] = x;
                local_max = fmaxf( local_max, x );
                local_min = fminf( local_min, x );
            }

            s_max[tid] = local_max;
            s_min[tid] = local_min;
            __syncthreads();

            for ( int s = kBlock / 2; s > 0; s >>= 1 )
            {
                if ( tid < s )
                {
                    s_max[tid] = fmaxf( s_max[tid], s_max[tid + s] );
                    s_min[tid] = fminf( s_min[tid], s_min[tid + s] );
                }

                __syncthreads();
            }

            if ( tid == 0 )
            {
                reduction_scratch[kOffsetMaxPartials + blockIdx.x] = s_max[0];
                reduction_scratch[kOffsetMinPartials + blockIdx.x] = s_min[0];
            }
        }

        // Pipeline stage 2: finalize max/min, reset thresholds, zero all bins.
        template <int kBlock>
        __global__ void stochastic_prepare_kernel(
            float* reduction_scratch, int32_t* index_scratch, int partial_count )
        {
            __shared__ float s_f[kBlock];

            const int tid = threadIdx.x;

            float local = -FLT_MAX;
            for ( int i = tid; i < partial_count; i += kBlock )
                local = fmaxf( local, reduction_scratch[kOffsetMaxPartials + i] );
            s_f[tid] = local;
            __syncthreads();
            for ( int s = kBlock / 2; s > 0; s >>= 1 )
            {
                if ( tid < s ) s_f[tid] = fmaxf( s_f[tid], s_f[tid + s] );
                __syncthreads();
            }
            const float max_val = s_f[0];
            __syncthreads();

            local = FLT_MAX;
            for ( int i = tid; i < partial_count; i += kBlock )
                local = fminf( local, reduction_scratch[kOffsetMinPartials + i] );
            s_f[tid] = local;
            __syncthreads();
            for ( int s = kBlock / 2; s > 0; s >>= 1 )
            {
                if ( tid < s ) s_f[tid] = fminf( s_f[tid], s_f[tid + s] );
                __syncthreads();
            }
            const float min_val = s_f[0];

            if ( tid == 0 )
            {
                reduction_scratch[kSlotMax] = max_val;
                reduction_scratch[kSlotMin] = min_val;
                // Refinement range for whichever truncation phase runs first. The
                // upper edge sits just above the max so the max itself always bins.
                reduction_scratch[kSlotRangeLo] = min_val;
                reduction_scratch[kSlotRangeHi] = nextafterf( max_val, FLT_MAX );
                reduction_scratch[kSlotKThreshold] = -FLT_MAX;
                reduction_scratch[kSlotPThreshold] = -FLT_MAX;
                reduction_scratch[kSlotMassAbove] = 0.0f;
                reduction_scratch[kSlotTotalMass] = 0.0f;
                index_scratch[kSlotCountAbove] = 0;
            }

            for ( int j = tid; j < kStochasticBins; j += kBlock )
            {
                index_scratch[kOffsetCountBins + j] = 0;
                reduction_scratch[kOffsetMassBins + j] = 0.0f;
            }
        }

        // Top-k refinement, count pass: bucket scratch values inside [lo, hi) into
        // kStochasticBins bins; values at or above hi accumulate the above-range
        // count. Recounted fresh every round so bin-edge float rounding stays
        // internally consistent within a round.
        template <int kBlock>
        __global__ void topk_count_kernel(
            const float* scratch, int vocab, const float* reduction_scratch, int32_t* index_scratch )
        {
            __shared__ int s_bins[kStochasticBins];
            __shared__ int s_above;

            const int tid = threadIdx.x;

            for ( int j = tid; j < kStochasticBins; j += kBlock )
                s_bins[j] = 0;
            if ( tid == 0 )
                s_above = 0;
            __syncthreads();

            const float lo = reduction_scratch[kSlotRangeLo];
            const float hi = reduction_scratch[kSlotRangeHi];

            if ( hi > lo )
            {
                const float inverse_width = static_cast<float>( kStochasticBins ) / ( hi - lo );
                const int stride = gridDim.x * kBlock;

                for ( int i = blockIdx.x * kBlock + tid; i < vocab; i += stride )
                {
                    const float x = scratch[i];

                    if ( x >= hi )
                    {
                        atomicAdd( &s_above, 1 );
                    }
                    else if ( x >= lo )
                    {
                        const int j = min( static_cast<int>( ( x - lo ) * inverse_width ), kStochasticBins - 1 );
                        atomicAdd( &s_bins[j], 1 );
                    }
                }
            }

            __syncthreads();

            if ( tid == 0 && s_above > 0 )
                atomicAdd( &index_scratch[kSlotCountAbove], s_above );

            for ( int j = tid; j < kStochasticBins; j += kBlock )
                if ( s_bins[j] > 0 )
                    atomicAdd( &index_scratch[kOffsetCountBins + j], s_bins[j] );
        }

        // Top-k refinement, select pass: walk bins from the top until the running
        // count reaches top_k; narrow the range to the boundary bin, or on the last
        // round emit the value threshold. An exact count at the boundary keeps the
        // boundary value; an overshoot means a tie cluster straddles the cut --
        // exclude it, matching the reference kernel's bisection behavior. Bins and
        // the above-count reset for the next round on the way out.
        template <int kBlock>
        __global__ void topk_select_kernel(
            float* reduction_scratch, int32_t* index_scratch, int top_k, int last_round )
        {
            __shared__ int s_bins[kStochasticBins];

            const int tid = threadIdx.x;

            for ( int j = tid; j < kStochasticBins; j += kBlock )
                s_bins[j] = index_scratch[kOffsetCountBins + j];
            __syncthreads();

            if ( tid == 0 )
            {
                const float lo = reduction_scratch[kSlotRangeLo];
                const float hi = reduction_scratch[kSlotRangeHi];

                if ( !( hi > lo ) )
                {
                    // Range collapsed below float resolution: everything left sits
                    // at lo; keep it.
                    if ( last_round )
                        reduction_scratch[kSlotKThreshold] = lo;
                }
                else
                {
                    const float width = ( hi - lo ) / static_cast<float>( kStochasticBins );
                    int cumulative = index_scratch[kSlotCountAbove];
                    int boundary = 0;

                    for ( int b = kStochasticBins - 1; b >= 0; --b )
                    {
                        if ( cumulative + s_bins[b] >= top_k )
                        {
                            boundary = b;
                            break;
                        }

                        cumulative += s_bins[b];
                    }

                    const float bin_lo = lo + width * static_cast<float>( boundary );
                    const float bin_hi = ( boundary == kStochasticBins - 1 )
                        ? hi : lo + width * static_cast<float>( boundary + 1 );

                    if ( last_round )
                    {
                        const int kept_with_boundary = cumulative + s_bins[boundary];
                        reduction_scratch[kSlotKThreshold] =
                            ( kept_with_boundary == top_k ) ? bin_lo : bin_hi;
                    }
                    else
                    {
                        reduction_scratch[kSlotRangeLo] = bin_lo;
                        reduction_scratch[kSlotRangeHi] = bin_hi;
                    }
                }

                if ( last_round )
                {
                    // Hand the refinement range to the top-p phase: its domain is
                    // the top-k survivor set.
                    reduction_scratch[kSlotRangeLo] = reduction_scratch[kSlotKThreshold];
                    reduction_scratch[kSlotRangeHi] = nextafterf( reduction_scratch[kSlotMax], FLT_MAX );
                }

                index_scratch[kSlotCountAbove] = 0;
            }

            __syncthreads();

            for ( int j = tid; j < kStochasticBins; j += kBlock )
                index_scratch[kOffsetCountBins + j] = 0;
        }

        // Top-p refinement, mass pass: same shape as the count pass but accumulates
        // exp(x - max) per bin. Bin masses merge through float atomics, so the
        // accumulation order (and a boundary threshold ulp) is not run-deterministic.
        template <int kBlock>
        __global__ void topp_mass_kernel(
            const float* scratch, int vocab, float* reduction_scratch )
        {
            __shared__ float s_bins[kStochasticBins];
            __shared__ float s_above;

            const int tid = threadIdx.x;

            for ( int j = tid; j < kStochasticBins; j += kBlock )
                s_bins[j] = 0.0f;
            if ( tid == 0 )
                s_above = 0.0f;
            __syncthreads();

            const float lo = reduction_scratch[kSlotRangeLo];
            const float hi = reduction_scratch[kSlotRangeHi];
            const float max_val = reduction_scratch[kSlotMax];

            if ( hi > lo )
            {
                const float inverse_width = static_cast<float>( kStochasticBins ) / ( hi - lo );
                const int stride = gridDim.x * kBlock;

                for ( int i = blockIdx.x * kBlock + tid; i < vocab; i += stride )
                {
                    const float x = scratch[i];

                    if ( x >= hi )
                    {
                        atomicAdd( &s_above, expf( x - max_val ) );
                    }
                    else if ( x >= lo )
                    {
                        const int j = min( static_cast<int>( ( x - lo ) * inverse_width ), kStochasticBins - 1 );
                        atomicAdd( &s_bins[j], expf( x - max_val ) );
                    }
                }
            }

            __syncthreads();

            if ( tid == 0 && s_above > 0.0f )
                atomicAdd( &reduction_scratch[kSlotMassAbove], s_above );

            for ( int j = tid; j < kStochasticBins; j += kBlock )
                if ( s_bins[j] > 0.0f )
                    atomicAdd( &reduction_scratch[kOffsetMassBins + j], s_bins[j] );
        }

        // Top-p refinement, select pass: the nucleus is the smallest top set whose
        // mass reaches top_p * total, so the boundary value is always kept -- the
        // last round emits the boundary bin's lower edge as the value threshold.
        // The survivor-set total is fixed on the first round.
        template <int kBlock>
        __global__ void topp_select_kernel(
            float* reduction_scratch, float top_p, int first_round, int last_round )
        {
            __shared__ float s_bins[kStochasticBins];

            const int tid = threadIdx.x;

            for ( int j = tid; j < kStochasticBins; j += kBlock )
                s_bins[j] = reduction_scratch[kOffsetMassBins + j];
            __syncthreads();

            if ( tid == 0 )
            {
                const float lo = reduction_scratch[kSlotRangeLo];
                const float hi = reduction_scratch[kSlotRangeHi];

                if ( first_round )
                {
                    float total = reduction_scratch[kSlotMassAbove];
                    for ( int b = 0; b < kStochasticBins; ++b )
                        total += s_bins[b];
                    reduction_scratch[kSlotTotalMass] = total;
                }

                const float target = top_p * reduction_scratch[kSlotTotalMass];

                if ( !( hi > lo ) )
                {
                    if ( last_round )
                        reduction_scratch[kSlotPThreshold] = lo;
                }
                else
                {
                    const float width = ( hi - lo ) / static_cast<float>( kStochasticBins );
                    float cumulative = reduction_scratch[kSlotMassAbove];
                    int boundary = 0;

                    for ( int b = kStochasticBins - 1; b >= 0; --b )
                    {
                        if ( cumulative + s_bins[b] >= target )
                        {
                            boundary = b;
                            break;
                        }

                        cumulative += s_bins[b];
                    }

                    if ( last_round )
                    {
                        reduction_scratch[kSlotPThreshold] = lo + width * static_cast<float>( boundary );
                    }
                    else
                    {
                        reduction_scratch[kSlotRangeLo] = lo + width * static_cast<float>( boundary );
                        reduction_scratch[kSlotRangeHi] = ( boundary == kStochasticBins - 1 )
                            ? hi : lo + width * static_cast<float>( boundary + 1 );
                    }
                }

                reduction_scratch[kSlotMassAbove] = 0.0f;
            }

            __syncthreads();

            for ( int j = tid; j < kStochasticBins; j += kBlock )
                reduction_scratch[kOffsetMassBins + j] = 0.0f;
        }

        // Pipeline stage 5: survivor probabilities (exp(x - max), else 0) written
        // back to scratch, with a contiguous-chunk partial sum per block so the
        // inverse CDF can locate its chunk without a full serial pass. When both
        // truncations are disabled the threshold is -FLT_MAX and everything
        // survives -- the full multinomial.
        template <int kBlock>
        __global__ void stochastic_prob_kernel(
            float* scratch, int vocab, float* reduction_scratch, int chunk_size )
        {
            __shared__ float s_f[kBlock];

            const int tid = threadIdx.x;
            const int begin = blockIdx.x * chunk_size;
            const int end = min( begin + chunk_size, vocab );
            const float threshold = fmaxf(
                reduction_scratch[kSlotKThreshold], reduction_scratch[kSlotPThreshold] );
            const float max_val = reduction_scratch[kSlotMax];

            float local = 0.0f;

            for ( int i = begin + tid; i < end; i += kBlock )
            {
                const float x = scratch[i];
                const float e = ( x >= threshold ) ? expf( x - max_val ) : 0.0f;
                scratch[i] = e;
                local += e;
            }

            s_f[tid] = local;
            __syncthreads();

            for ( int s = kBlock / 2; s > 0; s >>= 1 )
            {
                if ( tid < s ) s_f[tid] += s_f[tid + s];
                __syncthreads();
            }

            if ( tid == 0 )
                reduction_scratch[kOffsetChunkPartials + blockIdx.x] = s_f[0];
        }

        // Pipeline stage 6: token-index-order inverse CDF. Thread 0 scans the chunk
        // partials for the first chunk whose inclusive prefix reaches r * total,
        // then walks elementwise from that chunk's prefix base (tiles staged into
        // shared memory by the whole block). The walk continues into later chunks
        // when per-element rounding lands short, and keeps the reference kernel's
        // semantics: only a surviving (> 0) probability can be selected, and the
        // fallback is vocab - 1.
        template <int kBlock>
        __global__ void stochastic_cdf_kernel(
            const float* scratch, int vocab, const float* reduction_scratch,
            int chunk_count, int chunk_size, float r, int32_t* token_out )
        {
            __shared__ float s_partials[kStochasticMaxChunks];
            __shared__ float s_tile[kBlock];
            __shared__ float s_target;
            __shared__ float s_base;
            __shared__ int s_start;
            __shared__ int s_done;
            __shared__ int s_result;

            const int tid = threadIdx.x;
            const float* partials = reduction_scratch + kOffsetChunkPartials;

            for ( int j = tid; j < chunk_count; j += kBlock )
                s_partials[j] = partials[j];
            __syncthreads();

            if ( tid == 0 )
            {
                float total = 0.0f;
                for ( int c = 0; c < chunk_count; ++c )
                    total += s_partials[c];

                const float target = r * total;

                float prefix = 0.0f;
                int start = chunk_count - 1;
                float base = total - s_partials[chunk_count - 1];

                for ( int c = 0; c < chunk_count; ++c )
                {
                    if ( prefix + s_partials[c] >= target )
                    {
                        start = c;
                        base = prefix;
                        break;
                    }

                    prefix += s_partials[c];
                }

                s_target = target;
                s_base = base;
                s_start = start;
                s_done = 0;
                s_result = vocab - 1;
            }

            __syncthreads();

            float cumulative = s_base;

            for ( int c = s_start; c < chunk_count; ++c )
            {
                const int begin = c * chunk_size;
                const int end = min( begin + chunk_size, vocab );

                for ( int tile = begin; tile < end; tile += kBlock )
                {
                    const int index = tile + tid;
                    s_tile[tid] = ( index < end ) ? scratch[index] : 0.0f;
                    __syncthreads();

                    if ( tid == 0 )
                    {
                        const int count = min( kBlock, end - tile );

                        for ( int k = 0; k < count; ++k )
                        {
                            const float value = s_tile[k];
                            cumulative += value;

                            if ( value > 0.0f && cumulative >= s_target )
                            {
                                s_result = tile + k;
                                s_done = 1;
                                break;
                            }
                        }
                    }

                    __syncthreads();

                    if ( s_done )
                        break;
                }

                if ( s_done )
                    break;
            }

            if ( tid == 0 )
                token_out[0] = s_result;
        }

        template <typename TNative>
        inline void launch_stochastic(
            const TNative* logits, int32_t* token_out, float* scratch,
            float* reduction_scratch, int32_t* index_scratch,
            int vocab, float softcap, float temperature, int top_k, float top_p, float r, cudaStream_t stream )
        {
            constexpr int kBlock = kStochasticBlock;

            const int element_blocks = ceil_div( vocab, kBlock );
            const int scale_blocks = element_blocks < kStochasticMaxChunks ? element_blocks : kStochasticMaxChunks;
            const int histogram_blocks = element_blocks < kHistogramBlocks ? element_blocks : kHistogramBlocks;
            const int chunk_size = kBlock * ceil_div( vocab, kBlock * kStochasticMaxChunks );
            const int chunk_count = ceil_div( vocab, chunk_size );

            stochastic_scale_kernel<TNative, kBlock><<<scale_blocks, kBlock, 0, stream>>>(
                logits, scratch, vocab, softcap, temperature, reduction_scratch );
            stochastic_prepare_kernel<kBlock><<<1, kBlock, 0, stream>>>(
                reduction_scratch, index_scratch, scale_blocks );

            if ( top_k > 0 && top_k < vocab )
            {
                for ( int round = 0; round < kRefinementRounds; ++round )
                {
                    topk_count_kernel<kBlock><<<histogram_blocks, kBlock, 0, stream>>>(
                        scratch, vocab, reduction_scratch, index_scratch );
                    topk_select_kernel<kBlock><<<1, kBlock, 0, stream>>>(
                        reduction_scratch, index_scratch, top_k, round == kRefinementRounds - 1 );
                }
            }

            if ( top_p < 1.0f )
            {
                for ( int round = 0; round < kRefinementRounds; ++round )
                {
                    topp_mass_kernel<kBlock><<<histogram_blocks, kBlock, 0, stream>>>(
                        scratch, vocab, reduction_scratch );
                    topp_select_kernel<kBlock><<<1, kBlock, 0, stream>>>(
                        reduction_scratch, top_p, round == 0, round == kRefinementRounds - 1 );
                }
            }

            stochastic_prob_kernel<kBlock><<<chunk_count, kBlock, 0, stream>>>(
                scratch, vocab, reduction_scratch, chunk_size );
            stochastic_cdf_kernel<kBlock><<<1, kBlock, 0, stream>>>(
                scratch, vocab, reduction_scratch, chunk_count, chunk_size, r, token_out );
        }
    }

    void cuda_sample_stochastic_fp32(
        const float* logits, int32_t* token_out, float* scratch,
        float* reduction_scratch, int32_t* index_scratch,
        int vocab, float softcap, float temperature, int top_k, float top_p, float r, cudaStream_t stream )
    {
        launch_stochastic( logits, token_out, scratch, reduction_scratch, index_scratch,
            vocab, softcap, temperature, top_k, top_p, r, stream );
    }

    void cuda_sample_stochastic_bf16(
        const __nv_bfloat16* logits, int32_t* token_out, float* scratch,
        float* reduction_scratch, int32_t* index_scratch,
        int vocab, float softcap, float temperature, int top_k, float top_p, float r, cudaStream_t stream )
    {
        launch_stochastic( logits, token_out, scratch, reduction_scratch, index_scratch,
            vocab, softcap, temperature, top_k, top_p, r, stream );
    }

    // ========================================================================
    // Stochastic multinomial — single-block reference oracle
    //
    // The original correctness-first kernel, retained verbatim for parity tests
    // against the pipeline above (the bounded-ring oracle methodology). Never on
    // the production path.
    // ========================================================================

    namespace
    {
        // Block-wide reductions over s_f / s_i (single block, kBlock threads). Caller fills
        // s[tid] with its partial and calls these; the result lands in s[0] (all threads must
        // reach the post-call __syncthreads before reusing the buffer).
        template <int kBlock>
        __device__ inline float block_reduce_sum( float* s_f, int tid )
        {
            __syncthreads();
            for ( int s = kBlock / 2; s > 0; s >>= 1 )
            {
                if ( tid < s ) s_f[tid] += s_f[tid + s];
                __syncthreads();
            }
            return s_f[0];
        }

        template <int kBlock>
        __device__ inline int block_reduce_sum_i( int* s_i, int tid )
        {
            __syncthreads();
            for ( int s = kBlock / 2; s > 0; s >>= 1 )
            {
                if ( tid < s ) s_i[tid] += s_i[tid + s];
                __syncthreads();
            }
            return s_i[0];
        }

        // Single-block multinomial with optional top-k / top-p truncation. Correctness-first:
        // truncation thresholds are found by binary search on a value (top-k) or probability-mass
        // (top-p), each iteration a block reduction -- no global sort. The final inverse-CDF is a
        // thread-0 linear walk over the precomputed probabilities.
        template <typename TNative, int kBlock>
        __global__ void stochastic_reference_kernel(
            const TNative* logits, int32_t* token_out, float* scratch,
            int vocab, float softcap, float temperature, int top_k, float top_p, float r )
        {
            __shared__ float s_f[kBlock];
            __shared__ int s_i[kBlock];

            const int tid = threadIdx.x;

            // Pass 0: scaled logits -> scratch; reduce max and min.
            float local_max = -FLT_MAX;
            float local_min = FLT_MAX;

            for ( int i = tid; i < vocab; i += kBlock )
            {
                float x = to_float( logits[i] );

                if ( softcap > 0.0f )
                    x = softcap * tanhf( x / softcap );

                x = x / temperature;
                scratch[i] = x;
                local_max = fmaxf( local_max, x );
                local_min = fminf( local_min, x );
            }

            s_f[tid] = local_max;
            __syncthreads();
            for ( int s = kBlock / 2; s > 0; s >>= 1 )
            {
                if ( tid < s ) s_f[tid] = fmaxf( s_f[tid], s_f[tid + s] );
                __syncthreads();
            }
            const float max_val = s_f[0];
            __syncthreads();

            s_f[tid] = local_min;
            __syncthreads();
            for ( int s = kBlock / 2; s > 0; s >>= 1 )
            {
                if ( tid < s ) s_f[tid] = fminf( s_f[tid], s_f[tid + s] );
                __syncthreads();
            }
            const float min_val = s_f[0];
            __syncthreads();

            // Top-k: binary-search a value threshold keeping the top_k largest scaled logits,
            // then mask everything below it to -inf so it drops out of the softmax. `lo`
            // converges to count > top_k and `hi` to count <= top_k, so `hi` excludes the
            // (top_k+1)-th value -- keeping exactly top_k for distinct logits.
            if ( top_k > 0 && top_k < vocab )
            {
                float lo = min_val;
                float hi = max_val;

                for ( int iter = 0; iter < 40; ++iter )
                {
                    const float mid = 0.5f * ( lo + hi );

                    int local_count = 0;
                    for ( int i = tid; i < vocab; i += kBlock )
                        if ( scratch[i] >= mid ) ++local_count;

                    s_i[tid] = local_count;
                    const int count = block_reduce_sum_i<kBlock>( s_i, tid );
                    __syncthreads();

                    if ( count > top_k ) lo = mid; else hi = mid;
                }

                const float k_threshold = hi;
                for ( int i = tid; i < vocab; i += kBlock )
                    if ( scratch[i] < k_threshold ) scratch[i] = -FLT_MAX;
                __syncthreads();
            }

            // Pass 1: exp(scaled - max) -> scratch (masked entries -> 0); reduce sum.
            float local_sum = 0.0f;
            for ( int i = tid; i < vocab; i += kBlock )
            {
                const float e = expf( scratch[i] - max_val );
                scratch[i] = e;
                local_sum += e;
            }
            s_f[tid] = local_sum;
            float total = block_reduce_sum<kBlock>( s_f, tid );
            __syncthreads();

            // Top-p: binary-search a probability threshold keeping the highest-mass nucleus
            // (smallest set whose mass >= top_p * total), mask the rest to 0, recompute the sum.
            if ( top_p < 1.0f )
            {
                const float target_mass = top_p * total;

                float lo = 0.0f;
                float hi = 1.0f;   // max unnormalized prob is exp(0) = 1

                for ( int iter = 0; iter < 40; ++iter )
                {
                    const float mid = 0.5f * ( lo + hi );

                    float local_mass = 0.0f;
                    for ( int i = tid; i < vocab; i += kBlock )
                        if ( scratch[i] >= mid ) local_mass += scratch[i];

                    s_f[tid] = local_mass;
                    const float mass = block_reduce_sum<kBlock>( s_f, tid );
                    __syncthreads();

                    if ( mass > target_mass ) lo = mid; else hi = mid;
                }

                const float p_threshold = lo;

                float local_sum2 = 0.0f;
                for ( int i = tid; i < vocab; i += kBlock )
                {
                    if ( scratch[i] < p_threshold ) scratch[i] = 0.0f;
                    local_sum2 += scratch[i];
                }
                s_f[tid] = local_sum2;
                total = block_reduce_sum<kBlock>( s_f, tid );
                __syncthreads();
            }

            // Pass 2: thread-0 inverse-CDF against the host-drawn uniform. The scratch[i] > 0
            // guard keeps the result on a surviving (non-truncated) token.
            if ( tid == 0 )
            {
                const float target = r * total;
                float cumulative = 0.0f;
                int result = vocab - 1;

                for ( int i = 0; i < vocab; ++i )
                {
                    cumulative += scratch[i];

                    if ( scratch[i] > 0.0f && cumulative >= target )
                    {
                        result = i;
                        break;
                    }
                }

                token_out[0] = result;
            }
        }

        template <typename TNative>
        inline void launch_stochastic_reference(
            const TNative* logits, int32_t* token_out, float* scratch,
            int vocab, float softcap, float temperature, int top_k, float top_p, float r, cudaStream_t stream )
        {
            constexpr int kBlock = 256;
            stochastic_reference_kernel<TNative, kBlock><<<1, kBlock, 0, stream>>>(
                logits, token_out, scratch, vocab, softcap, temperature, top_k, top_p, r );
        }
    }

    void cuda_sample_stochastic_reference_fp32(
        const float* logits, int32_t* token_out, float* scratch,
        int vocab, float softcap, float temperature, int top_k, float top_p, float r, cudaStream_t stream )
    {
        launch_stochastic_reference( logits, token_out, scratch, vocab, softcap, temperature, top_k, top_p, r, stream );
    }

    void cuda_sample_stochastic_reference_bf16(
        const __nv_bfloat16* logits, int32_t* token_out, float* scratch,
        int vocab, float softcap, float temperature, int top_k, float top_p, float r, cudaStream_t stream )
    {
        launch_stochastic_reference( logits, token_out, scratch, vocab, softcap, temperature, top_k, top_p, r, stream );
    }
}
