/**
 * @file CodebookDequantize.cu
 * @brief Packed codebook weights -> BF16 staging buffer, the phase-1 half of the
 * two-phase prefill path. Shares the decode GEMV's unpack: one block per output row,
 * sixteen codes per thread per iteration, the codebook read back by warp shuffle.
 */

#include "CodebookDequantize.cuh"

#include <format>
#include <stdexcept>

namespace Mila::Dnn::Compute::Cuda::Linear
{
    namespace
    {
        constexpr int kThreadsPerBlock = 64;
        constexpr int kElementsPerThread = 16;

        /**
         * One block per output row; each thread expands 16 codes per iteration.
         *
         * The shape is the decode GEMV's (CodebookGemv.cu), not this kernel's original
         * one-thread-per-element form. That form was measured at 43.3% of Qwen prefill --
         * more than the cuBLASLt GEMM it feeds -- and 7x off its own memory roof, for four
         * reasons this layout removes: a 64-bit `index / C` per element to recover the row,
         * byte-granularity plane loads (four threads sharing one byte of the two-bit plane,
         * eight sharing one byte of the one-bit plane, together 11 bytes of L1 traffic per
         * element to deliver three bits), one scale load per element where one scale serves
         * a whole group, and 2-byte stores. Measurements in Qwen3.8.md section 8.
         *
         * 64 threads x 16 elements covers 1024 columns per iteration, which divides both
         * Qwen widths (5120 and 17408) exactly, so no iteration runs masked on the shapes
         * that matter.
         *
         * The codebook lives ONE ENTRY PER LANE and is read back with __shfl_sync for the
         * reason CodebookGemv.cu records: nvcc cannot index registers with a runtime value
         * and lowers `table[code]` to a select chain, which measured at 46% of the 2-bit
         * kernel's runtime and 68% of the 3-bit kernel's.
         *
         * The trip count is UNIFORM across the block and the tail is masked rather than
         * exited, because the full-mask shuffle above needs every lane to arrive. A `c < C`
         * loop bound is lane-dependent and would hang rather than answer wrongly.
         *
         * Arithmetic is unchanged: codebook[code] * scale in an FP32 register, one
         * round-to-nearest to BF16. The staged bytes are bit-identical to the original
         * form's, which CodebookLinearOpCuda's *StagingMatchesHostCodec tests pin element
         * by element against the host codec.
         */
        template<int kGroupSize, int kEntries>
        __global__ void __launch_bounds__( kThreadsPerBlock )
            codebook_dequantize_to_bf16_kernel(
                __nv_bfloat16* __restrict__ staging,
                const std::uint8_t* __restrict__ plane_two_bits,
                const std::uint8_t* __restrict__ plane_one_bit,
                const __half* __restrict__ scales,
                const float* __restrict__ codebook,
                int C )
        {
            const int oc = static_cast<int>( blockIdx.x );
            const int lane = static_cast<int>( threadIdx.x ) & 31;

            const float codebook_lane = ( lane < kEntries ) ? codebook[lane] : 0.0f;

            const int num_groups = C / kGroupSize;

            const std::uint8_t* row_low =
                plane_two_bits + static_cast<std::size_t>( oc ) * ( C / 4 );
            const __half* row_scales =
                scales + static_cast<std::size_t>( oc ) * num_groups;
            __nv_bfloat16* row_destination =
                staging + static_cast<std::size_t>( oc ) * C;

            const std::uint8_t* row_high = nullptr;

            if constexpr ( kEntries == 8 )
            {
                row_high = plane_one_bit + static_cast<std::size_t>( oc ) * ( C / 8 );
            }

            const int c_start = static_cast<int>( threadIdx.x ) * kElementsPerThread;
            const int c_step = kThreadsPerBlock * kElementsPerThread;
            const int iterations = ( C + c_step - 1 ) / c_step;

            for ( int iteration = 0; iteration < iterations; ++iteration )
            {
                const int c = c_start + iteration * c_step;
                const bool active = ( c < C );

                // An inactive lane reads column 0 rather than branching: always in bounds,
                // and the store below is what actually drops it.
                const int c_read = active ? c : 0;

                const std::uint32_t low_bits =
                    *reinterpret_cast<const std::uint32_t*>( row_low + c_read / 4 );

                std::uint32_t high_bits = 0u;

                if constexpr ( kEntries == 8 )
                {
                    high_bits = *reinterpret_cast<const std::uint16_t*>( row_high + c_read / 8 );
                }

                // A 16-element chunk starts at a multiple of 16 and every supported group
                // size is a multiple of 16, so a chunk never straddles a group.
                const float scale = __half2float( row_scales[c_read / kGroupSize] );

                // uint4 for the 16-byte alignment the stores need; the pair view writes it.
                uint4 staged[2];
                __nv_bfloat162* pairs = reinterpret_cast<__nv_bfloat162*>( staged );

#pragma unroll
                for ( int pair = 0; pair < 8; ++pair )
                {
                    const int k0 = 2 * pair;
                    const int k1 = 2 * pair + 1;

                    std::uint32_t code0 = ( low_bits >> ( 2 * k0 ) ) & 0x3u;
                    std::uint32_t code1 = ( low_bits >> ( 2 * k1 ) ) & 0x3u;

                    if constexpr ( kEntries == 8 )
                    {
                        code0 |= ( ( high_bits >> k0 ) & 0x1u ) << 2;
                        code1 |= ( ( high_bits >> k1 ) & 0x1u ) << 2;
                    }

                    pairs[pair] = __floats2bfloat162_rn(
                        __shfl_sync( 0xffffffffu, codebook_lane, code0 ) * scale,
                        __shfl_sync( 0xffffffffu, codebook_lane, code1 ) * scale );
                }

                if ( active )
                {
                    uint4* destination = reinterpret_cast<uint4*>( row_destination + c );
                    destination[0] = staged[0];
                    destination[1] = staged[1];
                }
            }
        }

        /**
         * @brief Bind the group size at compile time so the scale index is a shift.
         *
         * An unsupported size THROWS rather than falling through to a silent no-op. The
         * sibling FP4 launcher's `default: break` leaves the staging buffer holding whatever
         * the previous strip wrote, which surfaces as wrong logits rather than as a failure;
         * the set here is the one CodebookDequantize.cuh documents, so anything outside it
         * is a caller error and should say so.
         */
        template<int kEntries>
        void launchByGroupSize(
            cudaStream_t stream,
            __nv_bfloat16* staging,
            const std::uint8_t* plane_two_bits, const std::uint8_t* plane_one_bit,
            const __half* scales, const float* codebook,
            int C, int OC, int group_size )
        {
            const unsigned int grid = static_cast<unsigned int>( OC );

            switch ( group_size )
            {
                case 32:
                    codebook_dequantize_to_bf16_kernel<32, kEntries>
                        <<<grid, kThreadsPerBlock, 0, stream>>>(
                            staging, plane_two_bits, plane_one_bit, scales, codebook, C );
                    break;

                case 64:
                    codebook_dequantize_to_bf16_kernel<64, kEntries>
                        <<<grid, kThreadsPerBlock, 0, stream>>>(
                            staging, plane_two_bits, plane_one_bit, scales, codebook, C );
                    break;

                case 128:
                    codebook_dequantize_to_bf16_kernel<128, kEntries>
                        <<<grid, kThreadsPerBlock, 0, stream>>>(
                            staging, plane_two_bits, plane_one_bit, scales, codebook, C );
                    break;

                default:
                    throw std::invalid_argument( std::format(
                        "codebook dequantize: group size {} is not one of 32, 64, 128",
                        group_size ) );
            }
        }
    }

    void launch_codebook2_dequantize_to_bf16(
        cudaStream_t stream,
        __nv_bfloat16* staging,
        const std::uint8_t* plane_two_bits, const __half* scales, const float* codebook,
        int C, int OC, int group_size )
    {
        launchByGroupSize<4>( stream, staging, plane_two_bits, nullptr, scales, codebook,
            C, OC, group_size );
    }

    void launch_codebook3_dequantize_to_bf16(
        cudaStream_t stream,
        __nv_bfloat16* staging,
        const std::uint8_t* plane_two_bits, const std::uint8_t* plane_one_bit,
        const __half* scales, const float* codebook,
        int C, int OC, int group_size )
    {
        launchByGroupSize<8>( stream, staging, plane_two_bits, plane_one_bit, scales, codebook,
            C, OC, group_size );
    }
}
