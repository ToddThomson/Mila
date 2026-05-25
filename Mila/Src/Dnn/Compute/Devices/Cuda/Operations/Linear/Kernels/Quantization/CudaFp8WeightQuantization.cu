/**
 * @file CudaFp8WeightQuantization.cu
 * @brief Per-channel BF16→FP8_E4M3 weight quantization implementation.
 *
 * This plain CUDA translation unit is compiled exclusively by NVCC. It provides
 * the implementation of cuda_quantize_fp8_per_channel(), which is called from the
 * CudaLinearOp.Quantize.ixx module partition and ultimately from
 * CudaLinearOp::quantize() at model load time.
 *
 * No module imports or CXX_MODULES directives — this is a plain .cu TU that
 * compiles against the CUDA runtime headers only.
 *
 * Algorithm (GPU):
 *   1. Upload BF16 source blob to a temporary device buffer.
 *   2. Launch quantize_fp8_per_channel_kernel: one block per output channel.
 *      Each block does a two-phase pass over its row:
 *        Phase 1 — parallel warp + shared-memory reduction to find per-channel absmax.
 *        Phase 2 — parallel element-wise BF16→FP8 conversion using the channel scale.
 *   3. Synchronize and free the temporary device buffer.
 *
 * This replaces the previous single-threaded CPU loop, giving O(out_features)
 * parallel work on GPU vs O(out_features * in_features) sequential work on CPU.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdexcept>
#include <format>
#include <cstdint>
#include "CudaFp8WeightQuantization.cuh"

namespace Mila::Dnn::Compute::Cuda::Linear
{
    namespace
    {
        static constexpr float kFp8E4M3Max = 448.0f;
        static constexpr int   kBlockSize  = 256;

        /**
         * @brief GPU kernel: per-channel BF16→FP8_E4M3 quantization.
         *
         * Grid:  (out_features) blocks — one block per output channel.
         * Block: kBlockSize threads.
         * Smem:  (kBlockSize / 32) floats for the inter-warp absmax reduction.
         *
         * Phase 1 — find per-channel absmax:
         *   Each thread accumulates a local max over its strided elements.
         *   Warp-level reduction via __shfl_xor_sync collapses to one value per warp.
         *   Warp leaders write to shared memory; warp 0 reduces shared memory to a
         *   single absmax which is broadcast back via smem[0].
         *
         * Phase 2 — quantize:
         *   scale[channel] = absmax / kFp8E4M3Max  (or 1 if absmax == 0)
         *   dst[channel, i] = FP8_E4M3( src[channel, i] / scale )
         */
        __global__ void quantize_fp8_per_channel_kernel(
            const __nv_bfloat16* __restrict__ src,
            __nv_fp8_e4m3*       __restrict__ dst,
            float*               __restrict__ scales,
            int                              in_features )
        {
            const int channel = static_cast<int>( blockIdx.x );

            const __nv_bfloat16* row_src = src + static_cast<int64_t>( channel ) * in_features;
            __nv_fp8_e4m3*       row_dst = dst + static_cast<int64_t>( channel ) * in_features;

            // ----------------------------------------------------------------
            // Phase 1: parallel absmax reduction
            // ----------------------------------------------------------------

            float local_max = 0.0f;

            for ( int i = threadIdx.x; i < in_features; i += blockDim.x )
                local_max = fmaxf( local_max, fabsf( __bfloat162float( row_src[ i ] ) ) );

            // Warp-level reduction
            for ( int mask = 16; mask > 0; mask >>= 1 )
                local_max = fmaxf( local_max, __shfl_xor_sync( 0xffffffff, local_max, mask ) );

            // One float per warp in shared memory
            extern __shared__ float smem[];

            const int lane = threadIdx.x & 31;
            const int warp = threadIdx.x >> 5;

            if ( lane == 0 )
                smem[ warp ] = local_max;

            __syncthreads();

            // Final reduction across warps, done by warp 0
            const int num_warps = ( blockDim.x + 31 ) >> 5;

            if ( warp == 0 )
            {
                local_max = ( lane < num_warps ) ? smem[ lane ] : 0.0f;

                for ( int mask = 16; mask > 0; mask >>= 1 )
                    local_max = fmaxf( local_max, __shfl_xor_sync( 0xffffffff, local_max, mask ) );

                if ( lane == 0 )
                    smem[ 0 ] = local_max;
            }

            __syncthreads();

            const float absmax    = smem[ 0 ];
            const float scale     = ( absmax > 0.0f ) ? ( absmax / kFp8E4M3Max ) : 1.0f;
            const float inv_scale = 1.0f / scale;

            if ( threadIdx.x == 0 )
                scales[ channel ] = scale;

            // ----------------------------------------------------------------
            // Phase 2: element-wise quantization
            // ----------------------------------------------------------------

            for ( int i = threadIdx.x; i < in_features; i += blockDim.x )
                row_dst[ i ] = __nv_fp8_e4m3( __bfloat162float( row_src[ i ] ) * inv_scale );
        }

        /**
         * @brief Pass 1 — find the global absmax of a BF16 weight matrix.
         *
         * Grid: (out_features) blocks — one block per output channel.
         * Block: kBlockSize threads.
         * Smem:  (kBlockSize / 32) floats for the inter-warp absmax reduction.
         *
         * Each block reduces its row to a single per-channel absmax, then performs
         * an unsigned-integer atomicMax on global_max_bits to fold all channels into
         * a single global maximum. Positive IEEE 754 floats have the same ordering as
         * their unsigned bit representations, so atomicMax on float bits is valid here.
         */
        __global__ void find_global_absmax_kernel(
            const __nv_bfloat16* __restrict__ src,
            unsigned int*        __restrict__ global_max_bits,
            int                              in_features )
        {
            const int channel = static_cast<int>( blockIdx.x );
            const __nv_bfloat16* row = src + static_cast<int64_t>( channel ) * in_features;

            float local_max = 0.0f;
            for ( int i = threadIdx.x; i < in_features; i += blockDim.x )
                local_max = fmaxf( local_max, fabsf( __bfloat162float( row[ i ] ) ) );

            for ( int mask = 16; mask > 0; mask >>= 1 )
                local_max = fmaxf( local_max, __shfl_xor_sync( 0xffffffff, local_max, mask ) );

            extern __shared__ float smem[];

            const int lane = threadIdx.x & 31;
            const int warp = threadIdx.x >> 5;

            if ( lane == 0 )
                smem[ warp ] = local_max;

            __syncthreads();

            const int num_warps = ( blockDim.x + 31 ) >> 5;

            if ( warp == 0 )
            {
                local_max = ( lane < num_warps ) ? smem[ lane ] : 0.0f;

                for ( int mask = 16; mask > 0; mask >>= 1 )
                    local_max = fmaxf( local_max, __shfl_xor_sync( 0xffffffff, local_max, mask ) );

                if ( lane == 0 )
                    atomicMax( global_max_bits, __float_as_uint( local_max ) );
            }
        }

        /**
         * @brief Pass 2 — quantize BF16→FP8 using the global scale and fill scales[].
         *
         * Grid:  (out_features) blocks.
         * Block: kBlockSize threads.
         *
         * Reads global_max_bits (written by find_global_absmax_kernel), derives
         * global_scale = global_absmax / 448.0f, and:
         *   - Writes global_scale to scales[channel] (same value for every channel).
         *   - Converts each row element: dst[channel, i] = FP8_E4M3(src[channel, i] / global_scale).
         */
        __global__ void quantize_fp8_global_scale_kernel(
            const __nv_bfloat16*  __restrict__ src,
            __nv_fp8_e4m3*        __restrict__ dst,
            float*                __restrict__ scales,
            const unsigned int*   __restrict__ global_max_bits,
            int                               in_features )
        {
            const float global_absmax = __uint_as_float( *global_max_bits );
            const float global_scale  = ( global_absmax > 0.0f ) ? ( global_absmax / kFp8E4M3Max ) : 1.0f;
            const float inv_scale     = 1.0f / global_scale;

            const int channel = static_cast<int>( blockIdx.x );
            const __nv_bfloat16* row_src = src + static_cast<int64_t>( channel ) * in_features;
            __nv_fp8_e4m3*       row_dst = dst + static_cast<int64_t>( channel ) * in_features;

            if ( threadIdx.x == 0 )
                scales[ channel ] = global_scale;

            for ( int i = threadIdx.x; i < in_features; i += blockDim.x )
                row_dst[ i ] = __nv_fp8_e4m3( __bfloat162float( row_src[ i ] ) * inv_scale );
        }

    } // anonymous namespace

    void cuda_quantize_fp8_per_channel(
        const void* src_bf16,
        void*       dst_fp8,
        float*      dst_scales,
        int64_t     out_features,
        int64_t     in_features )
    {
        const size_t src_bytes = static_cast<size_t>( out_features * in_features )
                                 * sizeof( __nv_bfloat16 );

        // Upload BF16 source to a temporary device buffer.
        void* dev_src = nullptr;
        cudaError_t err = cudaMalloc( &dev_src, src_bytes );
        if ( err != cudaSuccess )
        {
            throw std::runtime_error( std::format(
                "cuda_quantize_fp8_per_channel - failed to allocate device staging buffer: {}",
                cudaGetErrorString( err ) ) );
        }

        err = cudaMemcpy( dev_src, src_bf16, src_bytes, cudaMemcpyHostToDevice );
        if ( err != cudaSuccess )
        {
            cudaFree( dev_src );
            throw std::runtime_error( std::format(
                "cuda_quantize_fp8_per_channel - BF16 source upload failed: {}",
                cudaGetErrorString( err ) ) );
        }

        // Launch: one block per output channel.
        const int smem_bytes = ( kBlockSize / 32 ) * static_cast<int>( sizeof( float ) );

        quantize_fp8_per_channel_kernel<<<
            static_cast<unsigned int>( out_features ),
            kBlockSize,
            smem_bytes >>>(
                static_cast<const __nv_bfloat16*>( dev_src ),
                static_cast<__nv_fp8_e4m3*>( dst_fp8 ),
                dst_scales,
                static_cast<int>( in_features ) );

        err = cudaGetLastError();
        if ( err != cudaSuccess )
        {
            cudaFree( dev_src );
            throw std::runtime_error( std::format(
                "cuda_quantize_fp8_per_channel - kernel launch failed: {}",
                cudaGetErrorString( err ) ) );
        }

        err = cudaDeviceSynchronize();
        if ( err != cudaSuccess )
        {
            cudaFree( dev_src );
            throw std::runtime_error( std::format(
                "cuda_quantize_fp8_per_channel - kernel execution failed: {}",
                cudaGetErrorString( err ) ) );
        }

        cudaFree( dev_src );
    }

    void cuda_quantize_fp8_per_tensor(
        const void* src_bf16,
        void*       dst_fp8,
        float*      dst_scales,
        int64_t     out_features,
        int64_t     in_features )
    {
        const size_t src_bytes = static_cast<size_t>( out_features * in_features ) * sizeof( __nv_bfloat16 );

        // Upload BF16 source to a temporary device buffer.
        void* dev_src = nullptr;
        cudaError_t err = cudaMalloc( &dev_src, src_bytes );
        if ( err != cudaSuccess )
        {
            throw std::runtime_error( std::format(
                "cuda_quantize_fp8_per_tensor - failed to allocate staging buffer: {}",
                cudaGetErrorString( err ) ) );
        }

        err = cudaMemcpy( dev_src, src_bf16, src_bytes, cudaMemcpyHostToDevice );
        if ( err != cudaSuccess )
        {
            cudaFree( dev_src );
            throw std::runtime_error( std::format(
                "cuda_quantize_fp8_per_tensor - source upload failed: {}",
                cudaGetErrorString( err ) ) );
        }

        // Allocate and zero-initialise the global max accumulator.
        // Stored as unsigned int for integer atomicMax over IEEE 754 float bits.
        unsigned int* dev_global_max = nullptr;
        err = cudaMalloc( &dev_global_max, sizeof( unsigned int ) );
        if ( err != cudaSuccess )
        {
            cudaFree( dev_src );
            throw std::runtime_error( std::format(
                "cuda_quantize_fp8_per_tensor - failed to allocate global max buffer: {}",
                cudaGetErrorString( err ) ) );
        }

        err = cudaMemset( dev_global_max, 0, sizeof( unsigned int ) );
        if ( err != cudaSuccess )
        {
            cudaFree( dev_src );
            cudaFree( dev_global_max );
            throw std::runtime_error( std::format(
                "cuda_quantize_fp8_per_tensor - failed to zero global max buffer: {}",
                cudaGetErrorString( err ) ) );
        }

        const int smem_bytes = ( kBlockSize / 32 ) * static_cast<int>( sizeof( float ) );

        // Pass 1: find global absmax (one block per channel, atomic fold across channels).
        find_global_absmax_kernel<<<
            static_cast<unsigned int>( out_features ),
            kBlockSize,
            smem_bytes >>>(
                static_cast<const __nv_bfloat16*>( dev_src ),
                dev_global_max,
                static_cast<int>( in_features ) );

        err = cudaGetLastError();
        if ( err != cudaSuccess )
        {
            cudaFree( dev_src );
            cudaFree( dev_global_max );
            throw std::runtime_error( std::format(
                "cuda_quantize_fp8_per_tensor - find_global_absmax_kernel launch failed: {}",
                cudaGetErrorString( err ) ) );
        }

        // Pass 2: quantize using global scale and fill scales[].
        // Kernel 1 and 2 are on the same (default) stream; kernel 2 sees the
        // fully-reduced global_max because sequential kernel launches are ordered.
        quantize_fp8_global_scale_kernel<<<
            static_cast<unsigned int>( out_features ),
            kBlockSize >>>(
                static_cast<const __nv_bfloat16*>( dev_src ),
                static_cast<__nv_fp8_e4m3*>( dst_fp8 ),
                dst_scales,
                dev_global_max,
                static_cast<int>( in_features ) );

        err = cudaGetLastError();
        if ( err != cudaSuccess )
        {
            cudaFree( dev_src );
            cudaFree( dev_global_max );
            throw std::runtime_error( std::format(
                "cuda_quantize_fp8_per_tensor - quantize_fp8_global_scale_kernel launch failed: {}",
                cudaGetErrorString( err ) ) );
        }

        err = cudaDeviceSynchronize();
        if ( err != cudaSuccess )
        {
            cudaFree( dev_src );
            cudaFree( dev_global_max );
            throw std::runtime_error( std::format(
                "cuda_quantize_fp8_per_tensor - kernel execution failed: {}",
                cudaGetErrorString( err ) ) );
        }

        cudaFree( dev_src );
        cudaFree( dev_global_max );
    }

} // namespace Mila::Dnn::Compute::Cuda::Linear
