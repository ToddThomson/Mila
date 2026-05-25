/**
 * @file CudaFp4WeightQuantization.cu
 * @brief Per-group BF16->FP4_E2M1 weight quantization kernel.
 *
 * Algorithm:
 *   Grid:  (K/kGroupSize, out_features)  — one block per (channel, group)
 *   Block: kGroupSize threads
 *   Smem:  kGroupSize floats for the per-group absmax reduction
 *
 *   Each block owns one (channel, group) pair covering K indices
 *   [g*kGroupSize, (g+1)*kGroupSize).
 *
 *   Phase 1 — absmax reduction:
 *     Each thread loads one BF16 element, takes abs(), writes to smem,
 *     then a standard binary-tree reduction finds the group absmax.
 *
 *   Phase 2 — quantize and pack:
 *     scale = absmax / 6.0f  (6.0 = max FP4 E2M1 representable)
 *     Each thread maps its element to a 4-bit FP4 E2M1 nibble.
 *     Pairs of threads (2i, 2i+1) combine their nibbles into one uint8 byte:
 *       packed_byte = nibble_even | (nibble_odd << 4)
 *
 * FP4 E2M1 nibble encoding:
 *   Positive (nibbles 0-7):  {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
 *   Negative (nibbles 8-15): sign-magnitude mirror (bit 3 = sign)
 *
 * Rounding breakpoints (midpoint between adjacent representable values):
 *   0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdexcept>
#include <format>
#include <cstdint>
#include "CudaFp4WeightQuantization.cuh"

namespace Mila::Dnn::Compute::Cuda::Linear
{
    namespace
    {
        static constexpr float kFp4E2M1Max = 6.0f;

        /**
         * @brief Map a float in [-6, 6] to the nearest FP4 E2M1 nibble (0-15).
         *
         * Positive representable magnitudes (nibble index 0-7):
         *   0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
         * Rounding breakpoints between adjacent values:
         *   0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0
         * Values with |x| >= 5.0 round to 6.0 (nibble 7).
         * Sign is encoded in bit 3 of the nibble.
         */
        __device__ __forceinline__ uint8_t fp4_e2m1_quantize( float x )
        {
            const uint8_t sign   = ( x < 0.0f ) ? 8u : 0u;
            const float   abs_x  = fabsf( x );

            uint8_t mag;
            if      ( abs_x < 0.25f ) mag = 0;
            else if ( abs_x < 0.75f ) mag = 1;
            else if ( abs_x < 1.25f ) mag = 2;
            else if ( abs_x < 1.75f ) mag = 3;
            else if ( abs_x < 2.5f  ) mag = 4;
            else if ( abs_x < 3.5f  ) mag = 5;
            else if ( abs_x < 5.0f  ) mag = 6;
            else                      mag = 7;

            return sign | mag;
        }

        /**
         * @brief Per-group BF16->FP4_E2M1 quantization kernel.
         *
         * Grid  (blockIdx.x, blockIdx.y): (K/kGroupSize, out_features)
         * Block (threadIdx.x):            kGroupSize threads
         * Smem:                           kGroupSize floats (absmax reduction workspace)
         *
         * Thread tid handles element k = blockIdx.x * kGroupSize + tid within output
         * channel blockIdx.y.
         */
        template<int kGroupSize>
        __global__ void quantize_fp4_per_group_kernel(
            const __nv_bfloat16* __restrict__ src,
            uint8_t*             __restrict__ dst_packed,
            float*               __restrict__ scales,
            int                              K )
        {
            const int channel   = static_cast<int>( blockIdx.y );
            const int group     = static_cast<int>( blockIdx.x );
            const int tid       = static_cast<int>( threadIdx.x );
            const int num_groups = K / kGroupSize;

            const int k = group * kGroupSize + tid;

            const __nv_bfloat16* row_src    = src       + static_cast<int64_t>( channel ) * K;
            uint8_t*             row_packed = dst_packed + static_cast<int64_t>( channel ) * ( K / 2 );
            float*               row_scales = scales    + static_cast<int64_t>( channel ) * num_groups;

            // ----------------------------------------------------------------
            // Phase 1: parallel absmax reduction within the group
            // ----------------------------------------------------------------

            extern __shared__ float smem[];  // kGroupSize floats

            const float val  = __bfloat162float( row_src[ k ] );
            smem[ tid ] = fabsf( val );

            __syncthreads();

            for ( int s = kGroupSize / 2; s > 0; s >>= 1 )
            {
                if ( tid < s )
                    smem[ tid ] = fmaxf( smem[ tid ], smem[ tid + s ] );
                __syncthreads();
            }

            const float absmax    = smem[ 0 ];
            const float scale     = ( absmax > 0.0f ) ? ( absmax / kFp4E2M1Max ) : 1.0f;
            const float inv_scale = 1.0f / scale;

            if ( tid == 0 )
                row_scales[ group ] = scale;

            // ----------------------------------------------------------------
            // Phase 2: quantize and pack pairs of nibbles into one byte
            // ----------------------------------------------------------------

            const uint8_t nibble = fp4_e2m1_quantize( val * inv_scale );

            // Store nibbles in smem for pair-wise packing (reuse smem as uint storage)
            __shared__ uint8_t smem_nibbles[ kGroupSize ];
            smem_nibbles[ tid ] = nibble;

            __syncthreads();

            // Each even thread writes one packed byte covering columns (2*tid, 2*tid+1).
            if ( tid % 2 == 0 )
            {
                const uint8_t n0 = smem_nibbles[ tid ];      // low  nibble — even column
                const uint8_t n1 = smem_nibbles[ tid + 1 ];  // high nibble — odd  column
                row_packed[ ( group * kGroupSize + tid ) / 2 ] = n0 | static_cast<uint8_t>( n1 << 4 );
            }
        }

    } // anonymous namespace

    // -------------------------------------------------------------------------
    // Host dispatch
    // -------------------------------------------------------------------------

    namespace
    {
        template<int kGroupSize>
        void launch_quantize_fp4_per_group(
            const __nv_bfloat16* dev_src,
            uint8_t*             dst_packed,
            float*               dst_scales,
            int64_t              out_features,
            int64_t              in_features )
        {
            const int64_t num_groups = in_features / kGroupSize;

            const dim3 grid(
                static_cast<unsigned>( num_groups ),
                static_cast<unsigned>( out_features ) );
            const dim3 block( kGroupSize );
            const int  smem_bytes = kGroupSize * static_cast<int>( sizeof( float ) );

            quantize_fp4_per_group_kernel<kGroupSize><<<grid, block, smem_bytes>>>(
                dev_src, dst_packed, dst_scales, static_cast<int>( in_features ) );

            const cudaError_t err = cudaGetLastError();
            if ( err != cudaSuccess )
            {
                throw std::runtime_error( std::format(
                    "cuda_quantize_fp4_per_group: kernel launch failed (group_size={}): {}",
                    kGroupSize, cudaGetErrorString( err ) ) );
            }
        }
    }

    void cuda_quantize_fp4_per_group(
        const void* src_bf16,
        void*       dst_packed,
        float*      dst_scales,
        int64_t     out_features,
        int64_t     in_features,
        int         group_size )
    {
        const size_t src_bytes = static_cast<size_t>( out_features * in_features )
                                 * sizeof( __nv_bfloat16 );

        void* dev_src = nullptr;
        cudaError_t err = cudaMalloc( &dev_src, src_bytes );
        if ( err != cudaSuccess )
        {
            throw std::runtime_error( std::format(
                "cuda_quantize_fp4_per_group: failed to allocate staging buffer: {}",
                cudaGetErrorString( err ) ) );
        }

        err = cudaMemcpy( dev_src, src_bf16, src_bytes, cudaMemcpyHostToDevice );
        if ( err != cudaSuccess )
        {
            cudaFree( dev_src );
            throw std::runtime_error( std::format(
                "cuda_quantize_fp4_per_group: source upload failed: {}",
                cudaGetErrorString( err ) ) );
        }

        switch ( group_size )
        {
            case 64:
                launch_quantize_fp4_per_group<64>(
                    static_cast<const __nv_bfloat16*>( dev_src ),
                    static_cast<uint8_t*>( dst_packed ),
                    dst_scales, out_features, in_features );
                break;
            case 128:
                launch_quantize_fp4_per_group<128>(
                    static_cast<const __nv_bfloat16*>( dev_src ),
                    static_cast<uint8_t*>( dst_packed ),
                    dst_scales, out_features, in_features );
                break;
            default:
                cudaFree( dev_src );
                throw std::runtime_error( std::format(
                    "cuda_quantize_fp4_per_group: unsupported group_size={}", group_size ) );
        }

        err = cudaDeviceSynchronize();
        if ( err != cudaSuccess )
        {
            cudaFree( dev_src );
            throw std::runtime_error( std::format(
                "cuda_quantize_fp4_per_group: kernel execution failed: {}",
                cudaGetErrorString( err ) ) );
        }

        cudaFree( dev_src );
    }

} // namespace Mila::Dnn::Compute::Cuda::Linear
