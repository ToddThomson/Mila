// Device-side token sampling kernels. Phase A: greedy argmax over the logits row.

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
        // thread-0 linear walk over the precomputed probabilities. Kernel speed is not the win
        // here (the win is keeping the logits on the device); a parallel scan is a later option.
        template <typename TNative, int kBlock>
        __global__ void stochastic_kernel(
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
        inline void launch_stochastic(
            const TNative* logits, int32_t* token_out, float* scratch,
            int vocab, float softcap, float temperature, int top_k, float top_p, float r, cudaStream_t stream )
        {
            constexpr int kBlock = 256;
            stochastic_kernel<TNative, kBlock><<<1, kBlock, 0, stream>>>(
                logits, token_out, scratch, vocab, softcap, temperature, top_k, top_p, r );
        }
    }

    void cuda_sample_stochastic_fp32(
        const float* logits, int32_t* token_out, float* scratch,
        int vocab, float softcap, float temperature, int top_k, float top_p, float r, cudaStream_t stream )
    {
        launch_stochastic( logits, token_out, scratch, vocab, softcap, temperature, top_k, top_p, r, stream );
    }

    void cuda_sample_stochastic_bf16(
        const __nv_bfloat16* logits, int32_t* token_out, float* scratch,
        int vocab, float softcap, float temperature, int top_k, float top_p, float r, cudaStream_t stream )
    {
        launch_stochastic( logits, token_out, scratch, vocab, softcap, temperature, top_k, top_p, r, stream );
    }
}
