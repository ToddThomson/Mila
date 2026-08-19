/**
 * @file CodebookDequantize.cu
 * @brief Packed codebook weights -> BF16 staging buffer, the phase-1 half of the
 * two-phase prefill path. Mirrors cuda_fp4_dequantize_to_bf16 in structure so the
 * prefill branch of the codebook op is the proven shape rather than a new one.
 */

#include "CodebookDequantize.cuh"

namespace Mila::Dnn::Compute::Cuda::Linear
{
    namespace
    {
        constexpr int kThreadsPerBlock = 256;

        /**
         * One thread per weight element, grid-stride. The decode GEMV earns its
         * word-at-a-time loads because it is on the hot path; this kernel runs once per
         * forward against a GEMM that costs far more, so clarity wins over packing
         * tricks. Both read the same normative layout.
         */
        template<int kEntries>
        __global__ void __launch_bounds__( kThreadsPerBlock )
            codebook_dequantize_to_bf16_kernel(
                __nv_bfloat16* __restrict__ staging,
                const std::uint8_t* __restrict__ plane_two_bits,
                const std::uint8_t* __restrict__ plane_one_bit,
                const __half* __restrict__ scales,
                const float* __restrict__ codebook,
                int C, int OC, int group_size )
        {
            float codebook_registers[kEntries];
#pragma unroll
            for ( int entry = 0; entry < kEntries; ++entry )
                codebook_registers[entry] = codebook[entry];

            const int num_groups = C / group_size;
            const long long total = static_cast<long long>( OC ) * C;
            const long long stride = static_cast<long long>( gridDim.x ) * blockDim.x;

            for ( long long index = blockIdx.x * blockDim.x + threadIdx.x;
                  index < total; index += stride )
            {
                const int oc = static_cast<int>( index / C );
                const int c = static_cast<int>( index - static_cast<long long>( oc ) * C );

                const std::uint8_t low_bits = plane_two_bits[static_cast<long long>( oc ) * ( C / 4 ) + ( c >> 2 )];
                int code = ( low_bits >> ( ( c & 3 ) * 2 ) ) & 0x3;

                if constexpr ( kEntries == 8 )
                {
                    const std::uint8_t high_bits =
                        plane_one_bit[static_cast<long long>( oc ) * ( C / 8 ) + ( c >> 3 )];
                    code |= ( ( high_bits >> ( c & 7 ) ) & 0x1 ) << 2;
                }

                const float scale = __half2float(
                    scales[static_cast<long long>( oc ) * num_groups + c / group_size] );

                staging[index] = __float2bfloat16( codebook_registers[code] * scale );
            }
        }

        int gridForElements( long long total )
        {
            const long long blocks = ( total + kThreadsPerBlock - 1 ) / kThreadsPerBlock;

            // Cap the grid and let the stride loop cover the remainder: a 27B FFN matrix
            // is 89M elements, which is inside the 2^31 limit but well past the point
            // where more blocks buy anything.
            return static_cast<int>( blocks < 65535 ? blocks : 65535 );
        }
    }

    void launch_codebook2_dequantize_to_bf16(
        cudaStream_t stream,
        __nv_bfloat16* staging,
        const std::uint8_t* plane_two_bits, const __half* scales, const float* codebook,
        int C, int OC, int group_size )
    {
        const long long total = static_cast<long long>( OC ) * C;

        codebook_dequantize_to_bf16_kernel<4><<<gridForElements( total ), kThreadsPerBlock, 0, stream>>>(
            staging, plane_two_bits, nullptr, scales, codebook, C, OC, group_size );
    }

    void launch_codebook3_dequantize_to_bf16(
        cudaStream_t stream,
        __nv_bfloat16* staging,
        const std::uint8_t* plane_two_bits, const std::uint8_t* plane_one_bit,
        const __half* scales, const float* codebook,
        int C, int OC, int group_size )
    {
        const long long total = static_cast<long long>( OC ) * C;

        codebook_dequantize_to_bf16_kernel<8><<<gridForElements( total ), kThreadsPerBlock, 0, stream>>>(
            staging, plane_two_bits, plane_one_bit, scales, codebook, C, OC, group_size );
    }
}
