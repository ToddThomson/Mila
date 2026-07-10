#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h> // NVIDIA WMMA Header
#include <cmath>
#include <algorithm>

using namespace nvidia;

// HARDWARE TUNING CONSTANTS FOR WMMA TENSOR CORES
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define BLOCK_M 64
#define BLOCK_N 64

__global__ void mila_flash_gqa_wmma_kernel(
    const __nv_bfloat16* __restrict__ Q, // [B, ChunkLen, NH_Q, HS]
    const uint8_t* __restrict__ K_fp4, // [B, NKV, CacheCapacity, HS / 2]
    const uint8_t* __restrict__ V_fp4, // [B, NKV, CacheCapacity, HS / 2]
    const float* __restrict__ KV_scales, // [B, NKV, CacheCapacity, HS / 32]
    __nv_bfloat16* __restrict__ Y, // [B, ChunkLen, NH_Q, HS]
    const int B, const int chunk_len, const int total_seq_len,
    const int num_q_heads, const int num_kv_heads, const int head_size,
    const int window_size, const bool is_bounded, const int cache_capacity,
    const float scale )
{
    // Warp-level tracking inside the thread block
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    int block_m_start = blockIdx.x * BLOCK_M;
    int head_q = blockIdx.y;
    int batch = blockIdx.z;

    int g_size = num_q_heads / num_kv_heads;
    int head_kv = head_q / g_size;

    // Allocate Shared Memory (SRAM) for on-chip dequantization buffers
    // This serves as the uncompressed landing pad for WMMA fragment loading
    extern __shared__ __nv_bfloat16 s_buffer[];
    __nv_bfloat16* s_K_unpacked = &s_buffer[ 0 ]; // [BLOCK_N * head_size]
    __nv_bfloat16* s_V_unpacked = &s_buffer[ BLOCK_N * head_size ]; // [BLOCK_N * head_size]

    // Declare the Tensor Core Fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> q_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> k_frag; // Transposed on-the-fly
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> score_frag;

    // Accumulators for the final Value calculation
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> p_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> v_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> out_frag;

    // Local tracking registers for Online Softmax (per warp row)
    float m_i = -CUDART_INF_F;
    float l_i = 0.0f;

    // sliding window bounds
    int global_q_pos = total_seq_len - chunk_len + block_m_start;
    int start_kv_pos = is_bounded ? max( 0, global_q_pos - window_size ) : 0;
    int end_kv_pos = global_q_pos + BLOCK_M;

    // Loop over the historical KV Cache context blocks
    for ( int n_block = start_kv_pos; n_block < end_kv_pos; n_block += BLOCK_N )
    {

        // --- STEP 1: PARALLEL DEQUANTIZATION (FP4 -> SRAM BF16) ---
        // Threads cooperate to read packed bytes and unpack them into shared memory
        int total_elements_to_unpack = BLOCK_N * head_size;
        for ( int idx = threadIdx.x; idx < total_elements_to_unpack; idx += blockDim.x )
        {
            int n_local = idx / head_size;
            int d_local = idx % head_size;
            int kv_pos = n_block + n_local;

            if ( kv_pos < total_seq_len )
            {
                int physical_kv_pos = is_bounded ? ( kv_pos % cache_capacity ) : kv_pos;

                int cache_idx = batch * ( num_kv_heads * cache_capacity * ( head_size / 2 ) ) + head_kv * ( cache_capacity * ( head_size / 2 ) ) + physical_kv_pos * ( head_size / 2 ) + ( d_local / 2 );

                int scale_idx = batch * ( num_kv_heads * cache_capacity * ( head_size / 32 ) ) + head_kv * ( cache_capacity * ( head_size / 32 ) ) + physical_kv_pos * ( head_size / 32 ) + ( d_local / 32 );

                float scale_val = KV_scales[ scale_idx ];
                uint8_t packed_byte = K_fp4[ cache_idx ];

                // Isolate upper/lower nibble based on dimension alignment
                uint8_t raw_4bit = ( d_local % 2 == 0 ) ? ( packed_byte & 0x0F ) : ( ( packed_byte >> 4 ) & 0x0F );
                float dequant_k = static_cast<float>( static_cast<int>( raw_4bit ) - 8 ) * scale_val;
                s_K_unpacked[ idx ] = __float2bfloat16( dequant_k );

                // Unpack V simultaneously
                uint8_t packed_v_byte = V_fp4[ cache_idx ];
                uint8_t raw_v_4bit = ( d_local % 2 == 0 ) ? ( packed_v_byte & 0x0F ) : ( ( packed_v_byte >> 4 ) & 0x0F );
                float dequant_v = static_cast<float>( static_cast<int>( raw_v_4bit ) - 8 ) * scale_val;
                s_V_unpacked[ idx ] = __float2bfloat16( dequant_v );
            }
            else
            {
                s_K_unpacked[ idx ] = __float2bfloat16( 0.0f );
                s_V_unpacked[ idx ] = __float2bfloat16( 0.0f );
            }
        }
        __syncthreads(); // Memory barrier: Ensure SRAM is populated completely

        // --- STEP 2: TENSOR CORE GEMM 1 (Q @ K.T) via WMMA ---
        // Each warp processes a subset of rows (M) and columns (N)
        int warp_row_offset = block_m_start + ( warp_id * WMMA_M );

        if ( warp_row_offset < chunk_len )
        {
            wmma::fill_fragment( score_frag, 0.0f );

            // Loop along the Head Dimension (K-dim) in steps of 16
            for ( int k_step = 0; k_step < head_size; k_step += WMMA_K )
            {
                int q_stride_idx = batch * ( chunk_len * num_q_heads * head_size ) + warp_row_offset * ( num_q_heads * head_size ) + head_q * head_size + k_step;

                // Load vectors from global VRAM and SRAM into Tensor Core hardware registers
                wmma::load_matrix_sync( q_frag, Q + q_stride_idx, num_q_heads * head_size );
                wmma::load_matrix_sync( k_frag, s_K_unpacked + ( 0 * head_size ) + k_step, head_size );

                // Execute the Tensor Core Instruction
                wmma::mma_sync( score_frag, q_frag, k_frag, score_frag );
            }

            // --- STEP 3: REGISTER-LEVEL ONLINE SOFTMAX ---
            // Process the raw scores stored inside the accumulator fragment registers
            float local_max = -CUDART_INF_F;
            for ( int i = 0; i < score_frag.num_elements; ++i )
            {
                score_frag.x[ i ] *= scale;
                local_max = max( local_max, score_frag.x[ i ] );
            }

            // Synchronize max transitions and compute scaling elements
            float m_next = max( m_i, local_max );
            float alpha = expf( m_i - m_next );

            float local_sum = 0.0f;
            for ( int i = 0; i < score_frag.num_elements; ++i )
            {
                score_frag.x[ i ] = expf( score_frag.x[ i ] - m_next );
                local_sum += score_frag.x[ i ];
            }

            // --- STEP 4: TENSOR CORE GEMM 2 (P @ V) ---
            // Cast computed Softmax probabilities to BF16 fragment spaces
            wmma::fill_fragment( out_frag, 0.0f );

            // Re-interpret the float score fragment as a standard matrix_a input
            // (Assuming alignment matches layout mappings across hardware tiles)
            for ( int i = 0; i < p_frag.num_elements; ++i )
            {
                p_frag.x[ i ] = __float2bfloat16( score_frag.x[ i ] );
            }

            for ( int kv_step = 0; kv_step < head_size; kv_step += WMMA_K )
            {
                wmma::load_matrix_sync( v_frag, s_V_unpacked + ( 0 * head_size ) + kv_step, head_size );
                wmma::mma_sync( out_frag, p_frag, v_frag, out_frag );
            }

            // Combine local accumulators back into main thread states
            l_i = l_i * alpha + local_sum;
            m_i = m_next;

            // --- STEP 5: STREAM OUTPUT TO MILA GLOBAL TENSOR ---
            int y_stride_idx = batch * ( chunk_len * num_q_heads * head_size ) + warp_row_offset * ( num_q_heads * head_size ) + head_q * head_size;

            // Final normalization scaling directly as fragments write out back to VRAM
            for ( int i = 0; i < out_frag.num_elements; ++i )
            {
                out_frag.x[ i ] /= l_i;
            }
            wmma::store_matrix_sync( Y + y_stride_idx, out_frag, num_q_heads * head_size, wmma::mem_row_major );
        }
        __syncthreads(); // Re-sync thread block before moving to the next time tile
    }
}

// Host Launching Hook
void launch_mila_flash_gqa_wmma(
    cudaStream_t stream, const __nv_bfloat16* Q, const uint8_t* K_cache, const uint8_t* V_cache,
    const float* KV_scales, __nv_bfloat16* Y, int B, int chunk_len, int total_seq_len,
    int num_q_heads, int num_kv_heads, int head_size, int window_size, bool is_bounded, int cache_capacity )
{
    float scale = 1.0f / sqrtf( static_cast<float>( head_size ) );

    // Thread block configuration requires a minimum of 4 warps (128 threads) to handle tiling
    dim3 grid( ( chunk_len + BLOCK_M - 1 ) / BLOCK_M, num_q_heads, B );
    dim3 block( 128, 1, 1 );

    // Shared memory configuration for the uncompressed landing buffers
    size_t shared_mem_size = 2 * ( BLOCK_N * head_size ) * sizeof( __nv_bfloat16 );

    mila_flash_gqa_wmma_kernel<<<grid, block, shared_mem_size, stream>>>(
        Q, K_cache, V_cache, KV_scales, Y, B, chunk_len, total_seq_len,
        num_q_heads, num_kv_heads, head_size, window_size, is_bounded, cache_capacity, scale );
}
