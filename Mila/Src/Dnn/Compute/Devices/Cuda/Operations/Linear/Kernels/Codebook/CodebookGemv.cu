/**
 * @file CodebookGemv.cu
 * @brief W2A16 / W3A16 decode GEMV kernels: BF16 activations, packed codebook weights.
 * Follows the matvec_decode_bf16_qfp4 pattern (CudaMatVecBias.Bf16.cu): a warp per group
 * of output rows, FP32 accumulation, warp-shuffle reduction. The codebook replaces the
 * FP4 level table -- on SM 8.9 both are a LUT applied in the tile load, which is the bet
 * the whole sub-4-bit plan rests on (Specifications/Qwen3.8.md section 4).
 */

#include "CodebookGemv.cuh"

#include <vector>

namespace Mila::Dnn::Compute::Cuda::Linear
{
    namespace
    {
        constexpr int kThreadsPerOutput = 32;
        constexpr int kWarpsPerBlock = 4;

        /**
         * Output rows a single warp accumulates at once.
         *
         * One row per warp leaves the kernel latency-bound rather than bandwidth-bound:
         * a warp issues one code word per iteration and then waits for it, and at these
         * shapes there are only six to ten iterations in which to build up loads. Walking
         * several rows together issues that many independent code loads before consuming
         * any of them, which is the memory-level parallelism the single-row form lacks.
         *
         * The counts differ because a 3-bit row costs two plane loads instead of one, so
         * it reaches the same number of loads in flight at half the rows and regresses
         * beyond that. Measured DRAM-resident on an RTX 4070 -- see Qwen3.8.md section 8.
         */
        template<int kEntries>
        constexpr int kRowsPerWarpFor = ( kEntries == 8 ) ? 2 : 4;

        __device__ inline void loadBf16x4(
            __nv_bfloat162& low, __nv_bfloat162& high, const __nv_bfloat16* pointer )
        {
            const int2 raw = *reinterpret_cast<const int2*>( pointer );
            low = *reinterpret_cast<const __nv_bfloat162*>( &raw.x );
            high = *reinterpret_cast<const __nv_bfloat162*>( &raw.y );
        }

        /**
         * Each thread processes 16 codes per iteration per row: one uint32 of the two-bit
         * plane, plus one uint16 of the high-bit plane when kEntries == 8. A chunk
         * of 16 starts at a multiple of 16, and every supported group size is a
         * multiple of 16, so a chunk never straddles a group -- one scale load per
         * chunk, applied to the chunk subtotal.
         *
         * A warp walks kRowsPerWarp adjacent output rows together over one shared pass of
         * the activation vector. The accumulation order within a row is unchanged, so the
         * result is bit-identical to the single-row form.
         */
        template<int kGroupSize, int kEntries, int kRowsPerWarp>
        __global__ void __launch_bounds__( kThreadsPerOutput * kWarpsPerBlock )
            codebook_matvec_decode_kernel(
                __nv_bfloat16* __restrict__ y,
                const __nv_bfloat16* __restrict__ x,
                const std::uint8_t* __restrict__ plane_two_bits,
                const std::uint8_t* __restrict__ plane_one_bit,
                const __half* __restrict__ scales,
                const float* __restrict__ codebook,
                const __nv_bfloat16* __restrict__ bias,
                int C, int OC )
        {
            const int oc_base = ( blockIdx.x * kWarpsPerBlock + threadIdx.y ) * kRowsPerWarp;

            if ( oc_base >= OC ) return;

            // The codebook lives ONE ENTRY PER LANE, read back with __shfl_sync rather
            // than by indexing a register array. nvcc cannot index registers with a
            // runtime value: it keeps the array in registers (ptxas reports no spill) but
            // lowers every lookup to a select chain, ~3 comparisons at 4 entries and ~7 at
            // 8. Measured on a 3072x8192 projection, that chain was 46% of the 2-bit
            // kernel's runtime and 68% of the 3-bit kernel's; the shuffle replaces it with
            // a single instruction and is worth 1.81x at 8 entries. The full mask is safe
            // because oc_base depends only on blockIdx.x and threadIdx.y, so the early
            // return above is warp-uniform and all 32 lanes reach here together.
            //
            // A shared-memory table indexed by several packed codes at once was measured
            // as the alternative and LOST at both widths (0.80x, 0.81x): a runtime-indexed
            // shared load costs more here than the shuffle it replaces.
            const float codebook_lane =
                ( threadIdx.x < kEntries ) ? codebook[threadIdx.x] : 0.0f;

            const int num_groups = C / kGroupSize;

            float acc[kRowsPerWarp];
#pragma unroll
            for ( int row = 0; row < kRowsPerWarp; ++row )
                acc[row] = 0.0f;

            const int c_start = threadIdx.x * 16;
            const int c_step = kThreadsPerOutput * 16;

            // The trip count is UNIFORM across the warp, and the tail is handled by
            // masking rather than by exiting early. The obvious `c < C` bound is
            // lane-dependent -- at C = 544 lanes 0-1 run twice and lanes 2-31 once -- and
            // the full-mask __shfl_sync below would then wait forever on lanes that had
            // already left the loop. That is a deadlock rather than a wrong answer, and it
            // hung the C = 544 tail case outright until this loop was made uniform.
            const int iterations = ( C + c_step - 1 ) / c_step;

            for ( int iteration = 0; iteration < iterations; ++iteration )
            {
                const int c = c_start + iteration * c_step;
                const bool active = ( c < C );

                // Inactive lanes read offset 0 instead of branching: always in bounds, and
                // the result is discarded below. Branching here would reintroduce the
                // divergence the uniform trip count exists to remove.
                const int c_read = active ? c : 0;

                __nv_bfloat162 x_pairs[8];
                loadBf16x4( x_pairs[0], x_pairs[1], x + c_read );
                loadBf16x4( x_pairs[2], x_pairs[3], x + c_read + 4 );
                loadBf16x4( x_pairs[4], x_pairs[5], x + c_read + 8 );
                loadBf16x4( x_pairs[6], x_pairs[7], x + c_read + 12 );

                float2 x_values[8];
#pragma unroll
                for ( int pair = 0; pair < 8; ++pair )
                    x_values[pair] = __bfloat1622float2( x_pairs[pair] );

                // Every row's operands are issued before any row is reduced, so the
                // kRowsPerWarp loads overlap instead of serializing one per iteration.
                // That overlap is the whole point of walking several rows at once.
                std::uint32_t low_bits[kRowsPerWarp];
                std::uint32_t high_bits[kRowsPerWarp];
                __half scale_bits[kRowsPerWarp];

#pragma unroll
                for ( int row = 0; row < kRowsPerWarp; ++row )
                {
                    // A row past the end reads row 0 and discards the result, for the same
                    // reason inactive lanes read column 0: oc_base and row are both
                    // warp-uniform, but branching would still cost a divergent path.
                    const int oc = oc_base + row;
                    const int oc_read = ( oc < OC ) ? oc : 0;

                    low_bits[row] = *reinterpret_cast<const std::uint32_t*>(
                        plane_two_bits + static_cast<std::size_t>( oc_read ) * ( C / 4 ) + c_read / 4 );

                    if constexpr ( kEntries == 8 )
                    {
                        high_bits[row] = *reinterpret_cast<const std::uint16_t*>(
                            plane_one_bit + static_cast<std::size_t>( oc_read ) * ( C / 8 ) + c_read / 8 );
                    }
                    else
                        high_bits[row] = 0u;

                    scale_bits[row] = scales[
                        static_cast<std::size_t>( oc_read ) * num_groups + c_read / kGroupSize];
                }

#pragma unroll
                for ( int row = 0; row < kRowsPerWarp; ++row )
                {
                    float chunk = 0.0f;
#pragma unroll
                    for ( int pair = 0; pair < 8; ++pair )
                    {
                        const float2 x_f = x_values[pair];
                        const int k0 = 2 * pair;
                        const int k1 = 2 * pair + 1;

                        std::uint32_t code0 = ( low_bits[row] >> ( 2 * k0 ) ) & 0x3u;
                        std::uint32_t code1 = ( low_bits[row] >> ( 2 * k1 ) ) & 0x3u;

                        if constexpr ( kEntries == 8 )
                        {
                            code0 |= ( ( high_bits[row] >> k0 ) & 0x1u ) << 2;
                            code1 |= ( ( high_bits[row] >> k1 ) & 0x1u ) << 2;
                        }

                        chunk += x_f.x * __shfl_sync( 0xffffffffu, codebook_lane, code0 )
                            + x_f.y * __shfl_sync( 0xffffffffu, codebook_lane, code1 );
                    }

                    const bool contributes = active && ( oc_base + row < OC );
                    acc[row] += contributes ? __half2float( scale_bits[row] ) * chunk : 0.0f;
                }
            }

#pragma unroll
            for ( int row = 0; row < kRowsPerWarp; ++row )
            {
#pragma unroll
                for ( int offset = kThreadsPerOutput / 2; offset > 0; offset >>= 1 )
                {
                    acc[row] += __shfl_down_sync( 0xffffffff, acc[row], offset );
                }
            }

            if ( threadIdx.x == 0 )
            {
#pragma unroll
                for ( int row = 0; row < kRowsPerWarp; ++row )
                {
                    const int oc = oc_base + row;

                    if ( oc >= OC ) continue;

                    const float bias_value = ( bias != nullptr ) ? __bfloat162float( bias[oc] ) : 0.0f;
                    y[oc] = __float2bfloat16( acc[row] + bias_value );
                }
            }
        }

        template<int kEntries>
        void dispatch_by_group_size(
            cudaStream_t stream,
            __nv_bfloat16* y, const __nv_bfloat16* x,
            const std::uint8_t* plane_two_bits, const std::uint8_t* plane_one_bit,
            const __half* scales, const float* codebook,
            const __nv_bfloat16* bias, int C, int OC, int group_size )
        {
            constexpr int kRowsPerWarp = kRowsPerWarpFor<kEntries>;

            const dim3 block( kThreadsPerOutput, kWarpsPerBlock );
            const int row_groups = ( OC + kRowsPerWarp - 1 ) / kRowsPerWarp;
            const dim3 grid( ( row_groups + kWarpsPerBlock - 1 ) / kWarpsPerBlock );

            switch ( group_size )
            {
                case 32:
                    codebook_matvec_decode_kernel<32, kEntries, kRowsPerWarp><<<grid, block, 0, stream>>>(
                        y, x, plane_two_bits, plane_one_bit, scales, codebook, bias, C, OC );
                    break;

                case 64:
                    codebook_matvec_decode_kernel<64, kEntries, kRowsPerWarp><<<grid, block, 0, stream>>>(
                        y, x, plane_two_bits, plane_one_bit, scales, codebook, bias, C, OC );
                    break;

                case 128:
                    codebook_matvec_decode_kernel<128, kEntries, kRowsPerWarp><<<grid, block, 0, stream>>>(
                        y, x, plane_two_bits, plane_one_bit, scales, codebook, bias, C, OC );
                    break;

                default:
                    break;
            }
        }
    }

    void launch_codebook2_matvec_decode(
        cudaStream_t stream,
        __nv_bfloat16* y, const __nv_bfloat16* x,
        const std::uint8_t* codes_packed, const __half* scales, const float* codebook,
        const __nv_bfloat16* bias, int C, int OC, int group_size )
    {
        dispatch_by_group_size<4>(
            stream, y, x, codes_packed, nullptr, scales, codebook, bias, C, OC, group_size );
    }

    void launch_codebook3_matvec_decode(
        cudaStream_t stream,
        __nv_bfloat16* y, const __nv_bfloat16* x,
        const std::uint8_t* plane_two_bits, const std::uint8_t* plane_one_bit,
        const __half* scales, const float* codebook,
        const __nv_bfloat16* bias, int C, int OC, int group_size )
    {
        dispatch_by_group_size<8>(
            stream, y, x, plane_two_bits, plane_one_bit, scales, codebook, bias, C, OC,
            group_size );
    }

    // -----------------------------------------------------------------------
    // Host-convenience entries (tests and bring-up)
    // -----------------------------------------------------------------------

    namespace
    {
        int run_host_common(
            int entries,
            const float* x, const std::uint8_t* plane_two_bits, const std::uint8_t* plane_one_bit,
            const std::uint16_t* scale_bits, const float* codebook, const float* bias, float* y,
            int C, int OC, int group_size )
        {
            const std::size_t plane_two_bytes = static_cast<std::size_t>( OC ) * ( C / 4 );
            const std::size_t plane_one_bytes = static_cast<std::size_t>( OC ) * ( C / 8 );
            const std::size_t scale_count = static_cast<std::size_t>( OC ) * ( C / group_size );

            std::vector<__nv_bfloat16> x_bf16( static_cast<std::size_t>( C ) );

            for ( int c = 0; c < C; ++c )
                x_bf16[static_cast<std::size_t>( c )] = __float2bfloat16( x[c] );

            std::vector<__nv_bfloat16> bias_bf16;

            if ( bias != nullptr )
            {
                bias_bf16.resize( static_cast<std::size_t>( OC ) );

                for ( int oc = 0; oc < OC; ++oc )
                    bias_bf16[static_cast<std::size_t>( oc )] = __float2bfloat16( bias[oc] );
            }

            __nv_bfloat16* device_y = nullptr;
            __nv_bfloat16* device_x = nullptr;
            __nv_bfloat16* device_bias = nullptr;
            std::uint8_t* device_plane_two = nullptr;
            std::uint8_t* device_plane_one = nullptr;
            __half* device_scales = nullptr;
            float* device_codebook = nullptr;

            cudaError_t status = cudaSuccess;

            auto check = [&status]( cudaError_t result )
            {
                if ( status == cudaSuccess && result != cudaSuccess )
                    status = result;
                return status == cudaSuccess;
            };

            check( cudaMalloc( &device_y, OC * sizeof( __nv_bfloat16 ) ) );
            check( cudaMalloc( &device_x, C * sizeof( __nv_bfloat16 ) ) );
            check( cudaMalloc( &device_plane_two, plane_two_bytes ) );
            check( cudaMalloc( &device_scales, scale_count * sizeof( __half ) ) );
            check( cudaMalloc( &device_codebook, entries * sizeof( float ) ) );

            if ( entries == 8 )
                check( cudaMalloc( &device_plane_one, plane_one_bytes ) );

            if ( bias != nullptr )
                check( cudaMalloc( &device_bias, OC * sizeof( __nv_bfloat16 ) ) );

            check( cudaMemcpy( device_x, x_bf16.data(), C * sizeof( __nv_bfloat16 ),
                cudaMemcpyHostToDevice ) );
            check( cudaMemcpy( device_plane_two, plane_two_bits, plane_two_bytes,
                cudaMemcpyHostToDevice ) );
            check( cudaMemcpy( device_scales, scale_bits, scale_count * sizeof( std::uint16_t ),
                cudaMemcpyHostToDevice ) );
            check( cudaMemcpy( device_codebook, codebook, entries * sizeof( float ),
                cudaMemcpyHostToDevice ) );

            if ( entries == 8 )
                check( cudaMemcpy( device_plane_one, plane_one_bit, plane_one_bytes,
                    cudaMemcpyHostToDevice ) );

            if ( bias != nullptr )
                check( cudaMemcpy( device_bias, bias_bf16.data(), OC * sizeof( __nv_bfloat16 ),
                    cudaMemcpyHostToDevice ) );

            if ( status == cudaSuccess )
            {
                if ( entries == 4 )
                    launch_codebook2_matvec_decode( nullptr, device_y, device_x, device_plane_two,
                        device_scales, device_codebook, device_bias, C, OC, group_size );
                else
                    launch_codebook3_matvec_decode( nullptr, device_y, device_x, device_plane_two,
                        device_plane_one, device_scales, device_codebook, device_bias, C, OC,
                        group_size );

                check( cudaGetLastError() );
                check( cudaDeviceSynchronize() );
            }

            std::vector<__nv_bfloat16> y_bf16( static_cast<std::size_t>( OC ) );
            check( cudaMemcpy( y_bf16.data(), device_y, OC * sizeof( __nv_bfloat16 ),
                cudaMemcpyDeviceToHost ) );

            if ( status == cudaSuccess )
            {
                for ( int oc = 0; oc < OC; ++oc )
                    y[oc] = __bfloat162float( y_bf16[static_cast<std::size_t>( oc )] );
            }

            cudaFree( device_y );
            cudaFree( device_x );
            cudaFree( device_bias );
            cudaFree( device_plane_two );
            cudaFree( device_plane_one );
            cudaFree( device_scales );
            cudaFree( device_codebook );

            return static_cast<int>( status );
        }
    }

    int run_codebook2_matvec_decode_host(
        const float* x, const std::uint8_t* codes_packed, const std::uint16_t* scale_bits,
        const float* codebook, const float* bias, float* y, int C, int OC, int group_size )
    {
        return run_host_common( 4, x, codes_packed, nullptr, scale_bits, codebook, bias, y,
            C, OC, group_size );
    }

    int run_codebook3_matvec_decode_host(
        const float* x, const std::uint8_t* plane_two_bits, const std::uint8_t* plane_one_bit,
        const std::uint16_t* scale_bits, const float* codebook, const float* bias, float* y,
        int C, int OC, int group_size )
    {
        return run_host_common( 8, x, plane_two_bits, plane_one_bit, scale_bits, codebook, bias,
            y, C, OC, group_size );
    }

    int codebook_gemv_device_available()
    {
        int device_count = 0;

        if ( cudaGetDeviceCount( &device_count ) != cudaSuccess )
            return 0;

        return device_count > 0 ? 1 : 0;
    }
}
