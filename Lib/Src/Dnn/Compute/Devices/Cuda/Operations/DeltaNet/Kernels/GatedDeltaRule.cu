/**
 * @file GatedDeltaRule.cu
 * @brief Gated delta rule -- two kernels, recurrent and chunked, both precisions.
 *
 * Written as one file rather than the Fp32/Bf16 pair used elsewhere because the kernels are
 * long and identical in both, differing only in the load/store conversion. Duplicating a
 * recurrence to change two casts is how the two copies drift.
 *
 * The recurrent kernel is the definition and the oracle; the chunked one is what prefill
 * runs. `launch` picks between them on the step count alone -- see the threshold below.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "device_launch_parameters.h"
#include "CudaUtils.h"

namespace Mila::Dnn::Compute::Cuda::DeltaNet
{
    namespace
    {
        __device__ __forceinline__ float toFloat( float x ) { return x; }
        __device__ __forceinline__ float toFloat( __nv_bfloat16 x ) { return __bfloat162float( x ); }

        __device__ __forceinline__ void fromFloat( float v, float& out ) { out = v; }
        __device__ __forceinline__ void fromFloat( float v, __nv_bfloat16& out ) { out = __float2bfloat16( v ); }

        // Matches torch.nn.functional.softplus, including its beta=1 threshold=20 cutover:
        // past 20 the exp overflows long before log1p could matter, and x IS the answer.
        // Precise expf, not __expf: this kernel is latency-bound on a sequential
        // recurrence, not on transcendental throughput, and the FP32 arm is checked against
        // a float32 oracle at 1e-5. The fast intrinsics buy nothing here and cost parity.
        __device__ __forceinline__ float softplus( float x )
        {
            return x > 20.0f ? x : log1pf( expf( x ) );
        }

        __device__ __forceinline__ float sigmoid( float x )
        {
            return 1.0f / (1.0f + expf( -x ) );
        }

        /**
         * @brief Which key row a thread's state slot holds.
         *
         * A part does NOT own a contiguous block of rows -- it owns every TKeySplit'th
         * GROUP OF FOUR. Contiguous slices are the obvious layout and they are the wrong
         * one: at head_key_dim 128 and four parts, the slices start 32 floats apart, which
         * is exactly the bank period, so all four parts of a column collide on every load.
         * Profiled that way the kernel ran the shared pipe at 78% of peak on 7.1 million
         * conflicts. Interleaving by float4 block puts the four parts four banks apart
         * instead, and keeps each load 16-byte aligned and vectorizable.
         */
        template<int TKeySplit>
        __device__ __forceinline__ int stateRowOf( int part, int block, int lane )
        {
            return 4 * (part + TKeySplit * block) + lane;
        }

        /**
         * @brief Dot a key or query row against this thread's share of the state column.
         *
         * FOUR VALUES PER LOAD AND FOUR ACCUMULATORS, both for the same reason. Profiled at
         * one scalar shared load per multiply, the chunked kernel issued on 9% of cycles and
         * spent 48% of them stalled on the shared-memory scoreboard, with the load-store
         * pipe doing twice the work of the arithmetic one. float4 turns four loads into one,
         * and the split accumulators give the scheduler four independent chains rather than
         * one dependency as deep as the share. Measured 2.3x on the published geometry.
         *
         * The loop walks all of TStateSlots and masks on the head width, because the state
         * has to stay constant-indexed to live in registers -- a runtime index spills it.
         */
        template<int TStateSlots, int TKeySplit>
        __device__ __forceinline__ float dotStateColumn(
            const float* row, const float (&column_state)[ TStateSlots ],
            int part, int head_k_dim )
        {
            float accumulator0 = 0.0f;
            float accumulator1 = 0.0f;
            float accumulator2 = 0.0f;
            float accumulator3 = 0.0f;

            #pragma unroll
            for ( int block = 0; block < TStateSlots / 4; ++block )
            {
                const int d = stateRowOf<TKeySplit>( part, block, 0 );
                const int slot = 4 * block;

                if ( d + 3 < head_k_dim )
                {
                    const float4 values = *reinterpret_cast<const float4*>( row + d );

                    accumulator0 += values.x * column_state[ slot ];
                    accumulator1 += values.y * column_state[ slot + 1 ];
                    accumulator2 += values.z * column_state[ slot + 2 ];
                    accumulator3 += values.w * column_state[ slot + 3 ];
                }
                else
                {
                    if ( d < head_k_dim ) accumulator0 += row[ d ] * column_state[ slot ];
                    if ( d + 1 < head_k_dim ) accumulator1 += row[ d + 1 ] * column_state[ slot + 1 ];
                    if ( d + 2 < head_k_dim ) accumulator2 += row[ d + 2 ] * column_state[ slot + 2 ];
                    if ( d + 3 < head_k_dim ) accumulator3 += row[ d + 3 ] * column_state[ slot + 3 ];
                }
            }

            return (accumulator0 + accumulator1) + (accumulator2 + accumulator3);
        }

        /// The outer-product half of the same trade: one load feeds four accumulations.
        template<int TStateSlots, int TKeySplit>
        __device__ __forceinline__ void accumulateStateColumn(
            const float* row, float (&column_state)[ TStateSlots ], float weight,
            int part, int head_k_dim )
        {
            #pragma unroll
            for ( int block = 0; block < TStateSlots / 4; ++block )
            {
                const int d = stateRowOf<TKeySplit>( part, block, 0 );
                const int slot = 4 * block;

                if ( d + 3 < head_k_dim )
                {
                    const float4 values = *reinterpret_cast<const float4*>( row + d );

                    column_state[ slot ] += values.x * weight;
                    column_state[ slot + 1 ] += values.y * weight;
                    column_state[ slot + 2 ] += values.z * weight;
                    column_state[ slot + 3 ] += values.w * weight;
                }
                else
                {
                    if ( d < head_k_dim ) column_state[ slot ] += row[ d ] * weight;
                    if ( d + 1 < head_k_dim ) column_state[ slot + 1 ] += row[ d + 1 ] * weight;
                    if ( d + 2 < head_k_dim ) column_state[ slot + 2 ] += row[ d + 2 ] * weight;
                    if ( d + 3 < head_k_dim ) column_state[ slot + 3 ] += row[ d + 3 ] * weight;
                }
            }
        }

        /**
         * @brief The block width the chunked kernel is COMPILED for, not merely launched at.
         *
         * head_value_dim * TKeySplit, 128 * 4 at the published geometry. It has to reach
         * ptxas: left to choose freely, ptxas picks a register count for a narrower block
         * and the launch is then refused outright -- "too many resources requested" rather
         * than a slow kernel -- which is how this was found. A geometry that would want a
         * wider block takes the recurrent kernel instead.
         *
         * The paired ONE BLOCK PER SM is not a wish, it is the fact: the chunk's working set
         * is over half of an SM's shared memory, so a second block could never be resident.
         * Saying so is what buys the register budget back -- without it ptxas squeezes to
         * 64 registers and spills, to fit a block that shared memory has already excluded.
         */
        constexpr int kChunkedBlockThreads = 512;

        /// Row-against-row, for the two triangles. Vectorized on both operands.
        __device__ __forceinline__ float dotRows(
            const float* left, const float* right, int head_k_dim )
        {
            float accumulator0 = 0.0f;
            float accumulator1 = 0.0f;
            int d = 0;

            for ( ; d + 3 < head_k_dim; d += 4 )
            {
                const float4 left_values = *reinterpret_cast<const float4*>( left + d );
                const float4 right_values = *reinterpret_cast<const float4*>( right + d );

                accumulator0 += left_values.x * right_values.x + left_values.z * right_values.z;
                accumulator1 += left_values.y * right_values.y + left_values.w * right_values.w;
            }

            for ( ; d < head_k_dim; ++d )
            {
                accumulator0 += left[ d ] * right[ d ];
            }

            return accumulator0 + accumulator1;
        }
    }

    /**
     * @tparam TMaxHeadK Compile-time bound on head_k_dim. The per-thread state array must
     *         be indexed by constants to live in registers, so the loops unroll to this
     *         bound and mask on the runtime width. Instantiated at the sizes actually used.
     */
    template<typename TElement, int TMaxHeadK>
    __global__ void gated_delta_rule_kernel(
        TElement* __restrict__       out,
        const TElement* __restrict__ q,
        const TElement* __restrict__ k,
        const TElement* __restrict__ v,
        const TElement* __restrict__ a,
        const TElement* __restrict__ b,
        const TElement* __restrict__ A_log,
        const TElement* __restrict__ dt_bias,
        float* __restrict__          state,
        int batch, int steps,
        int num_k_heads, int num_v_heads,
        int head_k_dim, int head_v_dim )
    {
        extern __shared__ float shared[];
        float* q_shared = shared;                 // [head_k_dim]
        float* k_shared = shared + head_k_dim;    // [head_k_dim]

        const int block = blockIdx.x;
        const int batch_index = block / num_v_heads;
        const int head = block % num_v_heads;
        const int group = num_v_heads / num_k_heads;
        const int k_head = head / group;

        const int column = threadIdx.x;
        const bool active = (column < head_v_dim);

        const int q_width = num_k_heads * head_k_dim;
        const int v_width = num_v_heads * head_v_dim;

        // Thread-local state column S[:, column]. Registers, for the whole chunk.
        float column_state[ TMaxHeadK ];

        const int state_base = (batch_index * num_v_heads + head) * head_k_dim * head_v_dim;

        #pragma unroll
        for ( int d = 0; d < TMaxHeadK; ++d )
        {
            if ( d < head_k_dim && active )
            {
                column_state[ d ] = state[ state_base + d * head_v_dim + column ];
            }
            else
            {
                column_state[ d ] = 0.0f;
            }
        }

        const float scale = rsqrtf( static_cast<float>( head_k_dim ) );

        const float head_A_log = toFloat( A_log[ head ] );
        const float head_dt_bias = toFloat( dt_bias[ head ] );

        for ( int t = 0; t < steps; ++t )
        {
            const int qk_offset = (batch_index * steps + t) * q_width + k_head * head_k_dim;

            // Every thread walks the same bound, so the syncs below are block-uniform.
            for ( int d = threadIdx.x; d < head_k_dim; d += blockDim.x )
            {
                q_shared[ d ] = toFloat( q[ qk_offset + d ] );
                k_shared[ d ] = toFloat( k[ qk_offset + d ] );
            }

            __syncthreads();

            // L2 norms, recomputed per thread from shared memory. A block reduction would
            // save the arithmetic and cost a second sync plus a partial-warp hazard; these
            // are broadcast reads and the loop is already unrolled.
            float q_square_sum = 0.0f;
            float k_square_sum = 0.0f;

            #pragma unroll
            for ( int d = 0; d < TMaxHeadK; ++d )
            {
                if ( d < head_k_dim )
                {
                    q_square_sum += q_shared[ d ] * q_shared[ d ];
                    k_square_sum += k_shared[ d ] * k_shared[ d ];
                }
            }

            const float q_inv_norm = rsqrtf( q_square_sum + 1e-6f ) * scale;
            const float k_inv_norm = rsqrtf( k_square_sum + 1e-6f );

            const int scalar_offset = (batch_index * steps + t) * num_v_heads + head;
            const float beta = sigmoid( toFloat( b[ scalar_offset ] ) );
            const float g = -expf( head_A_log )
                * softplus( toFloat( a[ scalar_offset ] ) + head_dt_bias );
            const float decay = expf( g );

            const int v_offset = (batch_index * steps + t) * v_width + head * head_v_dim + column;
            const float v_t = active ? toFloat( v[ v_offset ] ) : 0.0f;

            float kv_memory = 0.0f;

            #pragma unroll
            for ( int d = 0; d < TMaxHeadK; ++d )
            {
                if ( d < head_k_dim )
                {
                    column_state[ d ] *= decay;
                    kv_memory += column_state[ d ] * (k_shared[ d ] * k_inv_norm);
                }
            }

            const float delta = (v_t - kv_memory) * beta;

            float result = 0.0f;

            #pragma unroll
            for ( int d = 0; d < TMaxHeadK; ++d )
            {
                if ( d < head_k_dim )
                {
                    column_state[ d ] += (k_shared[ d ] * k_inv_norm) * delta;
                    result += column_state[ d ] * (q_shared[ d ] * q_inv_norm);
                }
            }

            if ( active )
            {
                fromFloat( result, out[ v_offset ] );
            }

            // The next iteration overwrites q_shared/k_shared.
            __syncthreads();
        }

        #pragma unroll
        for ( int d = 0; d < TMaxHeadK; ++d )
        {
            if ( d < head_k_dim && active )
            {
                state[ state_base + d * head_v_dim + column ] = column_state[ d ];
            }
        }
    }

    /**
     * @brief The same rule over a chunk of TChunk steps at a time -- the UT transform.
     *
     * WHAT IT IS. Unrolling the recurrence over a chunk and collecting terms turns the
     * per-step dependency into one triangular solve. With S0 the state at chunk entry and
     * c_t the running sum of the log-decays (c is non-increasing and at most 0):
     *
     *   u_t   = beta_t ( v_t - S0^T (k_t e^{c_t}) - sum(j<t) e^{c_t-c_j} (k_t.k_j) u_j )
     *   out_t = S0^T (q_t e^{c_t}) + sum(j<=t) e^{c_t-c_j} (q_t.k_j) u_j
     *   S_C   = e^{c_last} S0 + sum(j) e^{c_last-c_j} k_j (x) u_j
     *
     * u_t is exactly the recurrent form's delta_t, so this is a regrouping of the same
     * arithmetic rather than an approximation of it.
     *
     * EVERY EXPONENT IS A DIFFERENCE, and inside the causal triangle every difference is at
     * most 0. That is the whole numerical argument: the textbook derivation carries a
     * 1/e^{c_j} that overflows within a few dozen steps of a decaying gate, and never
     * forming it is what makes fp32 enough at this chunk width.
     *
     * WHAT IT IS WORTH, MEASURED: 1.10x against the recurrent kernel on the published
     * geometry, and that is the whole of it. Read the next two paragraphs before spending
     * any more on this form -- the ceiling is lower than the shape of the algorithm suggests.
     *
     * It does about 1.5x MORE arithmetic than the recurrence: three passes over the state
     * per token against two. What it buys back is that the recurrent form cannot fill the
     * machine -- profiled, that kernel holds FOUR warps on an SM with room for 48, because
     * one block per (batch, value head) and one thread per value column is 6144 threads for
     * the entire published geometry, so each scheduler has exactly one warp and every
     * latency is exposed. It runs at 0.68 TFLOPS on an RTX 4070, 2.3% of peak.
     *
     * THE KEY DIMENSION IS THEREFORE SPLIT ACROSS TKeySplit THREADS, each holding
     * head_k_dim/TKeySplit rows of one state column rather than all of them -- four times
     * the resident warps and a quarter of the register footprint. That costs a cross-thread
     * reduction per readout, which is precisely what the recurrent form cannot afford: there
     * it would land inside a per-token dependency chain, while here the chunk's C readouts
     * are independent. The parallelism and the regrouping are one idea; neither works alone.
     * The TKeySplit threads sharing a column are ADJACENT LANES, so the reduction is a warp
     * shuffle and the handoff of u_t between them a warp barrier, never a block one.
     *
     * The measured ladder, one mixer at the published geometry, against 4.4 us/token
     * recurrent -- kept because three of the four steps were worth more than the algorithm:
     *
     *   scalar shared loads, one thread per column      15.6 us/token   0.29x
     *   float4 loads and four accumulators               6.9            0.63x
     *   key split across four threads                    6.8            0.65x
     *   ... with the bank collision it exposed removed   4.1            1.07x
     *   ... one block per SM declared to ptxas           3.9            1.10x
     *
     * The third row is the lesson: four times the warps bought NOTHING until the layout
     * stopped putting all four parts of a column in one bank. The kernel now issues on 43%
     * of cycles against the recurrent form's 34%, with no stall above 0.24 and conflicts
     * down from 7.1M to 37K -- so 1.10x is close to what this structure yields, not a tuning
     * gap. What is left is the ALU pipe outrunning the FMA pipe (24% against 14%): address
     * arithmetic and mask predicates, not the mathematics.
     *
     * WHERE A REAL WIN WOULD COME FROM, if one is ever wanted: the triangles and the state
     * update are matmul-shaped and could go to tensor cores, which is reachable ONLY from
     * this form. Price it before building it -- and price it against the 13% of prefill the
     * whole mixer costs, not against the 68% Qwen3.8.md once claimed for it.
     *
     * A three-pass form that runs whole chunks concurrently was priced and rejected: its
     * middle pass is the same sequential scan at two thirds the work, so it can win at most
     * 1.5x, and it needs every chunk's entry state resident (~98 MB at T=1024 for this
     * geometry) against a card the whole project exists to fit inside.
     *
     * @tparam TChunk Steps per chunk. Fixed at compile time so the shared-memory triangle
     *         is a constant; 32 keeps the whole working set under a single block's share.
     * @tparam TKeySplit Threads per value column. Must divide 32 so the group is a lane
     *         quad, and TMaxHeadK, so the row split is compile-time.
     */
    template<typename TElement, int TMaxHeadK, int TChunk, int TKeySplit>
    __global__ void __launch_bounds__( kChunkedBlockThreads, 1 ) gated_delta_rule_chunked_kernel(
        TElement* __restrict__       out,
        const TElement* __restrict__ q,
        const TElement* __restrict__ k,
        const TElement* __restrict__ v,
        const TElement* __restrict__ a,
        const TElement* __restrict__ b,
        const TElement* __restrict__ A_log,
        const TElement* __restrict__ dt_bias,
        float* __restrict__          state,
        int batch, int steps,
        int num_k_heads, int num_v_heads,
        int head_k_dim, int head_v_dim )
    {
        extern __shared__ __align__( 16 ) float shared[];

        // Padded by FOUR, not one. The pad is there because the triangles read q and k down
        // a column -- every thread a different row at the same depth -- which an unpadded
        // 128-float stride would land entirely in one bank. Four rather than one keeps every
        // row 16-byte aligned so the float4 loads in the hot paths are legal, at the cost of
        // leaving a 4-way conflict in the triangles, which are a minority of the work.
        const int key_stride = head_k_dim + 4;

        float* k_shared = shared;                                   // [TChunk][key_stride]
        float* q_shared = k_shared + TChunk * key_stride;           // [TChunk][key_stride]
        float* u_shared = q_shared + TChunk * key_stride;           // [TChunk][head_v_dim]
        float* triangle = u_shared + TChunk * head_v_dim;           // [TChunk][TChunk]
        float* cumulative = triangle + TChunk * TChunk;             // [TChunk]
        float* beta_shared = cumulative + TChunk;                   // [TChunk]
        float* decay_shared = beta_shared + TChunk;                 // [TChunk] e^{c_t}
        float* row_scale = decay_shared + TChunk;                   // [2][TChunk]

        const int block = blockIdx.x;
        const int batch_index = block / num_v_heads;
        const int head = block % num_v_heads;
        const int group = num_v_heads / num_k_heads;
        const int k_head = head / group;

        // Lane layout: the TKeySplit threads sharing a value column are ADJACENT, so their
        // reduction is a shuffle within one warp rather than a trip through shared memory.
        const int part = threadIdx.x % TKeySplit;
        const int column = threadIdx.x / TKeySplit;

        // Rounded up to a multiple of four because the state is walked one float4 block at
        // a time and every loop names all four slots. Under-4 shares only arise at test
        // geometries, where the mask keeps the extra slots idle -- but an index past the end
        // of a register array is not something to leave to a runtime mask.
        constexpr int kStateSlots = ((TMaxHeadK / TKeySplit) + 3) / 4 * 4;

        // Naming a lane the block does not have is undefined, and the way it shows up is a
        // hang rather than a wrong answer -- so the mask is built from THIS warp's occupancy,
        // not the block's. head_value_dim * TKeySplit need not be a multiple of the warp
        // width, which makes the last warp short at geometries the config permits.
        // TKeySplit divides both 32 and the block width, so a column's group never straddles
        // two warps and every lane in the mask reaches every collective below.
        const unsigned int warp_first_lane = (threadIdx.x / 32u) * 32u;
        const unsigned int lanes_in_warp = (blockDim.x - warp_first_lane < 32u)
            ? (blockDim.x - warp_first_lane) : 32u;
        const unsigned int lane_mask = (lanes_in_warp >= 32u)
            ? 0xFFFFFFFFu : ((1u << lanes_in_warp) - 1u);

        const int q_width = num_k_heads * head_k_dim;
        const int v_width = num_v_heads * head_v_dim;

        float column_state[ kStateSlots ];

        const int state_base = (batch_index * num_v_heads + head) * head_k_dim * head_v_dim;

        #pragma unroll
        for ( int block = 0; block < kStateSlots / 4; ++block )
        {
            #pragma unroll
            for ( int lane = 0; lane < 4; ++lane )
            {
                const int d = stateRowOf<TKeySplit>( part, block, lane );

                column_state[ 4 * block + lane ] = (d < head_k_dim)
                    ? state[ state_base + d * head_v_dim + column ] : 0.0f;
            }
        }

        const float scale = rsqrtf( static_cast<float>( head_k_dim ) );

        const float head_A_log = toFloat( A_log[ head ] );
        const float head_dt_bias = toFloat( dt_bias[ head ] );

        for ( int base = 0; base < steps; base += TChunk )
        {
            const int span = (steps - base < TChunk) ? (steps - base) : TChunk;

            for ( int i = threadIdx.x; i < span * head_k_dim; i += blockDim.x )
            {
                const int t = i / head_k_dim;
                const int d = i - t * head_k_dim;
                const int offset =
                    (batch_index * steps + base + t) * q_width + k_head * head_k_dim + d;

                q_shared[ t * key_stride + d ] = toFloat( q[ offset ] );
                k_shared[ t * key_stride + d ] = toFloat( k[ offset ] );
            }

            // Strided rather than one row per thread: head_value_dim sets the block width
            // and a narrow head can be smaller than the chunk, which would leave the tail
            // of every row-indexed buffer below unwritten.
            for ( int t = threadIdx.x; t < span; t += blockDim.x )
            {
                const int scalar_offset =
                    (batch_index * steps + base + t) * num_v_heads + head;

                beta_shared[ t ] = sigmoid( toFloat( b[ scalar_offset ] ) );
                cumulative[ t ] = -expf( head_A_log )
                    * softplus( toFloat( a[ scalar_offset ] ) + head_dt_bias );
            }

            __syncthreads();

            for ( int t = threadIdx.x; t < span; t += blockDim.x )
            {
                float q_square_sum = 0.0f;
                float k_square_sum = 0.0f;

                for ( int d = 0; d < head_k_dim; ++d )
                {
                    const float q_value = q_shared[ t * key_stride + d ];
                    const float k_value = k_shared[ t * key_stride + d ];

                    q_square_sum += q_value * q_value;
                    k_square_sum += k_value * k_value;
                }

                row_scale[ t ] = rsqrtf( q_square_sum + 1e-6f ) * scale;
                row_scale[ TChunk + t ] = rsqrtf( k_square_sum + 1e-6f );
            }

            __syncthreads();

            // Serial over at most TChunk terms, and the only place the chunk is walked in
            // order. A parallel scan would save well under a hundred cycles per chunk.
            if ( threadIdx.x == 0 )
            {
                for ( int t = 1; t < span; ++t )
                {
                    cumulative[ t ] += cumulative[ t - 1 ];
                }

                for ( int t = 0; t < span; ++t )
                {
                    decay_shared[ t ] = expf( cumulative[ t ] );
                }
            }

            __syncthreads();

            for ( int i = threadIdx.x; i < span * head_k_dim; i += blockDim.x )
            {
                const int t = i / head_k_dim;
                const int d = i - t * head_k_dim;

                q_shared[ t * key_stride + d ] *= row_scale[ t ];
                k_shared[ t * key_stride + d ] *= row_scale[ TChunk + t ];
            }

            __syncthreads();

            // The strictly lower triangle of the solve: how much of u_j the row t update
            // has to undo. beta_t belongs here rather than on the right-hand side because
            // it multiplies the whole bracket in the recurrence.
            for ( int pair = threadIdx.x; pair < span * span; pair += blockDim.x )
            {
                const int t = pair / span;
                const int j = pair - t * span;

                float value = 0.0f;

                if ( j < t )
                {
                    const float dot = dotRows(
                        k_shared + t * key_stride, k_shared + j * key_stride, head_k_dim );

                    value = beta_shared[ t ]
                        * expf( cumulative[ t ] - cumulative[ j ] ) * dot;
                }

                triangle[ pair ] = value;
            }

            __syncthreads();

            // Forward substitution. The only barrier inside is a WARP one -- u_t is handed
            // between the TKeySplit lanes of one column, never across the block -- and the
            // loop bound is span for every thread, so both collectives below are reached by
            // every lane in the mask. A divergent bound here would hang, not misbehave.
            for ( int t = 0; t < span; ++t )
            {
                float readout = dotStateColumn<kStateSlots, TKeySplit>(
                    k_shared + t * key_stride, column_state, part, head_k_dim );

                #pragma unroll
                for ( int offset = 1; offset < TKeySplit; offset <<= 1 )
                {
                    readout += __shfl_xor_sync( lane_mask, readout, offset );
                }

                const int v_offset =
                    (batch_index * steps + base + t) * v_width + head * head_v_dim + column;
                const float v_t = toFloat( v[ v_offset ] );

                float value = beta_shared[ t ] * (v_t - decay_shared[ t ] * readout);

                for ( int j = 0; j < t; ++j )
                {
                    value -= triangle[ t * span + j ] * u_shared[ j * head_v_dim + column ];
                }

                // Every part computed the same value off the same reduced readout; one of
                // them publishes it, and the warp barrier is what makes it visible to the
                // others at step t+1.
                if ( part == 0 )
                {
                    u_shared[ t * head_v_dim + column ] = value;
                }

                __syncwarp( lane_mask );
            }

            __syncthreads();

            // The triangle is reused for the readout weights, inclusive of the diagonal:
            // out_t sees the update made at step t, exactly as the recurrent form does.
            for ( int pair = threadIdx.x; pair < span * span; pair += blockDim.x )
            {
                const int t = pair / span;
                const int j = pair - t * span;

                float value = 0.0f;

                if ( j <= t )
                {
                    const float dot = dotRows(
                        q_shared + t * key_stride, k_shared + j * key_stride, head_k_dim );

                    value = expf( cumulative[ t ] - cumulative[ j ] ) * dot;
                }

                triangle[ pair ] = value;
            }

            __syncthreads();

            for ( int t = 0; t < span; ++t )
            {
                float result = dotStateColumn<kStateSlots, TKeySplit>(
                    q_shared + t * key_stride, column_state, part, head_k_dim );

                #pragma unroll
                for ( int offset = 1; offset < TKeySplit; offset <<= 1 )
                {
                    result += __shfl_xor_sync( lane_mask, result, offset );
                }

                result *= decay_shared[ t ];

                for ( int j = 0; j <= t; ++j )
                {
                    result += triangle[ t * span + j ] * u_shared[ j * head_v_dim + column ];
                }

                if ( part == 0 )
                {
                    const int v_offset =
                        (batch_index * steps + base + t) * v_width + head * head_v_dim + column;

                    fromFloat( result, out[ v_offset ] );
                }
            }

            // Carry the state to the chunk boundary. Rolled over j and unrolled over the
            // part's rows, so the state array stays constant-indexed while u is read at
            // runtime depth.
            const float chunk_decay = decay_shared[ span - 1 ];

            #pragma unroll
            for ( int local = 0; local < kStateSlots; ++local )
            {
                column_state[ local ] *= chunk_decay;
            }

            for ( int j = 0; j < span; ++j )
            {
                const float weight = expf( cumulative[ span - 1 ] - cumulative[ j ] );
                const float u_value = u_shared[ j * head_v_dim + column ] * weight;

                accumulateStateColumn<kStateSlots, TKeySplit>(
                    k_shared + j * key_stride, column_state, u_value, part, head_k_dim );
            }

            // The next chunk overwrites every shared buffer above.
            __syncthreads();
        }

        #pragma unroll
        for ( int block = 0; block < kStateSlots / 4; ++block )
        {
            #pragma unroll
            for ( int lane = 0; lane < 4; ++lane )
            {
                const int d = stateRowOf<TKeySplit>( part, block, lane );

                if ( d < head_k_dim )
                {
                    state[ state_base + d * head_v_dim + column ] =
                        column_state[ 4 * block + lane ];
                }
            }
        }
    }

    namespace
    {
        constexpr int kChunkSteps = 32;

        /// Below one full chunk the chunked form is strictly more work, and decode is one
        /// step. The production prefill floor is 64 steps, so prefill always takes the
        /// chunked path and decode never does -- and both kernels stay reachable from tests.
        constexpr int kChunkedThreshold = kChunkSteps;

        /// Threads per value column. Divides 32, so the group is a lane quad and its
        /// reduction is a shuffle; divides every TMaxHeadK below, so the row split is
        /// compile-time. Four takes the published geometry from 4 resident warps to 16.
        constexpr int kKeySplit = 4;


        size_t chunkedSharedBytes( int head_k_dim, int head_v_dim )
        {
            const size_t floats =
                static_cast<size_t>( 2 * kChunkSteps ) * (head_k_dim + 4)   // q, k
                + static_cast<size_t>( kChunkSteps ) * head_v_dim           // u
                + static_cast<size_t>( kChunkSteps ) * kChunkSteps          // triangle
                + static_cast<size_t>( 5 * kChunkSteps );                   // gates, scales

            return floats * sizeof( float );
        }

        /**
         * @brief Whether this device will hand a block @p bytes of dynamic shared memory.
         *
         * The chunked kernel wants ~53 KB at the published geometry, past the 48 KB a block
         * gets without asking. Queried once per device rather than assumed, because a wide
         * head_value_dim can ask for more than any device allows -- in which case the
         * caller falls back to the recurrent kernel rather than failing the launch.
         */
        bool deviceAllowsSharedBytes( size_t bytes )
        {
            int device = 0;
            cudaCheck( cudaGetDevice( &device ) );

            static constexpr int kMaxDevices = 16;
            static int cached_limit[ kMaxDevices ] = {};

            if ( device < 0 || device >= kMaxDevices )
            {
                return false;
            }

            if ( cached_limit[ device ] == 0 )
            {
                int limit = 0;
                cudaCheck( cudaDeviceGetAttribute(
                    &limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, device ) );
                cached_limit[ device ] = limit;
            }

            return bytes <= static_cast<size_t>( cached_limit[ device ] );
        }

        template<typename TElement, int TMaxHeadK>
        void launchChunked(
            TElement* out, const TElement* q, const TElement* k, const TElement* v,
            const TElement* a, const TElement* b,
            const TElement* A_log, const TElement* dt_bias,
            float* state,
            int batch, int steps,
            int num_k_heads, int num_v_heads,
            int head_k_dim, int head_v_dim,
            size_t shared_bytes, cudaStream_t stream )
        {
            // Anything past the default 48 KB has to be asked for per kernel. Idempotent,
            // and the alternative is caching a flag per (kernel, device) pair to save a
            // driver call that costs a fraction of the launch it precedes.
            if ( shared_bytes > 48 * 1024 )
            {
                cudaCheck( cudaFuncSetAttribute(
                    gated_delta_rule_chunked_kernel<TElement, TMaxHeadK, kChunkSteps, kKeySplit>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>( shared_bytes ) ) );
            }

            gated_delta_rule_chunked_kernel<TElement, TMaxHeadK, kChunkSteps, kKeySplit>
                << <batch * num_v_heads, head_v_dim * kKeySplit, shared_bytes, stream >> > (
                    out, q, k, v, a, b, A_log, dt_bias, state,
                    batch, steps, num_k_heads, num_v_heads, head_k_dim, head_v_dim );
        }

        template<typename TElement>
        void launch(
            TElement* out, const TElement* q, const TElement* k, const TElement* v,
            const TElement* a, const TElement* b,
            const TElement* A_log, const TElement* dt_bias,
            float* state,
            int batch, int steps,
            int num_k_heads, int num_v_heads,
            int head_k_dim, int head_v_dim,
            cudaStream_t stream )
        {
            const int blocks = batch * num_v_heads;
            const int threads = head_v_dim;

            const size_t chunked_bytes = chunkedSharedBytes( head_k_dim, head_v_dim );
            const bool chunked = (steps >= kChunkedThreshold)
                && (head_v_dim * kKeySplit <= kChunkedBlockThreads)
                && deviceAllowsSharedBytes( chunked_bytes );

            if ( chunked )
            {
                if ( head_k_dim <= 8 )
                {
                    launchChunked<TElement, 8>( out, q, k, v, a, b, A_log, dt_bias, state,
                        batch, steps, num_k_heads, num_v_heads, head_k_dim, head_v_dim,
                        chunked_bytes, stream );
                }
                else if ( head_k_dim <= 64 )
                {
                    launchChunked<TElement, 64>( out, q, k, v, a, b, A_log, dt_bias, state,
                        batch, steps, num_k_heads, num_v_heads, head_k_dim, head_v_dim,
                        chunked_bytes, stream );
                }
                else
                {
                    launchChunked<TElement, 128>( out, q, k, v, a, b, A_log, dt_bias, state,
                        batch, steps, num_k_heads, num_v_heads, head_k_dim, head_v_dim,
                        chunked_bytes, stream );
                }

                cudaCheck( cudaGetLastError() );

                return;
            }

            const size_t shared_bytes = static_cast<size_t>( 2 * head_k_dim ) * sizeof( float );

            if ( head_k_dim <= 8 )
            {
                gated_delta_rule_kernel<TElement, 8> << <blocks, threads, shared_bytes, stream >> > (
                    out, q, k, v, a, b, A_log, dt_bias, state,
                    batch, steps, num_k_heads, num_v_heads, head_k_dim, head_v_dim );
            }
            else if ( head_k_dim <= 64 )
            {
                gated_delta_rule_kernel<TElement, 64> << <blocks, threads, shared_bytes, stream >> > (
                    out, q, k, v, a, b, A_log, dt_bias, state,
                    batch, steps, num_k_heads, num_v_heads, head_k_dim, head_v_dim );
            }
            else
            {
                gated_delta_rule_kernel<TElement, 128> << <blocks, threads, shared_bytes, stream >> > (
                    out, q, k, v, a, b, A_log, dt_bias, state,
                    batch, steps, num_k_heads, num_v_heads, head_k_dim, head_v_dim );
            }

            cudaCheck( cudaGetLastError() );
        }
    }

    void cuda_gated_delta_rule_fp32(
        float* out, const float* q, const float* k, const float* v,
        const float* a, const float* b, const float* A_log, const float* dt_bias,
        float* state,
        int batch, int steps, int num_k_heads, int num_v_heads,
        int head_k_dim, int head_v_dim, cudaStream_t stream )
    {
        launch<float>( out, q, k, v, a, b, A_log, dt_bias, state,
            batch, steps, num_k_heads, num_v_heads, head_k_dim, head_v_dim, stream );
    }

    void cuda_gated_delta_rule_bf16(
        __nv_bfloat16* out, const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
        const __nv_bfloat16* a, const __nv_bfloat16* b,
        const __nv_bfloat16* A_log, const __nv_bfloat16* dt_bias,
        float* state,
        int batch, int steps, int num_k_heads, int num_v_heads,
        int head_k_dim, int head_v_dim, cudaStream_t stream )
    {
        launch<__nv_bfloat16>( out, q, k, v, a, b, A_log, dt_bias, state,
            batch, steps, num_k_heads, num_v_heads, head_k_dim, head_v_dim, stream );
    }
}
