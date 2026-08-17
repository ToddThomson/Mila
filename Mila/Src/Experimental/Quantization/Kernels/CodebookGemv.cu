/**
 * @file CodebookGemv.cu
 * @brief W2A16 / W3A16 decode GEMV kernels: BF16 activations, packed codebook weights.
 * Follows the matvec_decode_bf16_qfp4 pattern (CudaMatVecBias.Bf16.cu): one warp per
 * output row, FP32 accumulation, warp-shuffle reduction. The codebook replaces the FP4
 * level table -- on SM 8.9 both are a LUT applied in the tile load, which is the bet
 * the whole sub-4-bit plan rests on (Specifications/Qwen3.8.md section 4).
 */

#include "CodebookGemv.cuh"

#include <vector>

namespace Mila::Dnn::Experimental::Quantization::Kernels
{
    namespace
    {
        constexpr int kThreadsPerOutput = 32;
        constexpr int kBlockOutputs = 8;

        __device__ inline void loadBf16x4(
            __nv_bfloat162& low, __nv_bfloat162& high, const __nv_bfloat16* pointer )
        {
            const int2 raw = *reinterpret_cast<const int2*>( pointer );
            low = *reinterpret_cast<const __nv_bfloat162*>( &raw.x );
            high = *reinterpret_cast<const __nv_bfloat162*>( &raw.y );
        }

        /**
         * Each thread processes 16 codes per iteration: one uint32 of the two-bit
         * plane, plus one uint16 of the high-bit plane when kEntries == 8. A chunk
         * of 16 starts at a multiple of 16, and every supported group size is a
         * multiple of 16, so a chunk never straddles a group -- one scale load per
         * chunk, applied to the chunk subtotal.
         */
        template<int kGroupSize, int kEntries>
        __global__ void __launch_bounds__( kThreadsPerOutput * kBlockOutputs )
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
            const int oc = blockIdx.x * kBlockOutputs + threadIdx.y;

            if ( oc >= OC ) return;

            float codebook_registers[kEntries];
#pragma unroll
            for ( int entry = 0; entry < kEntries; ++entry )
                codebook_registers[entry] = codebook[entry];

            const std::uint8_t* row_two_bits = plane_two_bits + oc * ( C / 4 );
            const std::uint8_t* row_one_bit =
                ( kEntries == 8 ) ? plane_one_bit + oc * ( C / 8 ) : nullptr;
            const int num_groups = C / kGroupSize;

            float acc = 0.0f;
            const int c_start = threadIdx.x * 16;
            const int c_step = kThreadsPerOutput * 16;

            for ( int c = c_start; c < C; c += c_step )
            {
                const std::uint32_t low_bits =
                    *reinterpret_cast<const std::uint32_t*>( row_two_bits + c / 4 );
                std::uint32_t high_bits = 0;

                if constexpr ( kEntries == 8 )
                    high_bits = *reinterpret_cast<const std::uint16_t*>( row_one_bit + c / 8 );

                __nv_bfloat162 x_pairs[8];
                loadBf16x4( x_pairs[0], x_pairs[1], x + c );
                loadBf16x4( x_pairs[2], x_pairs[3], x + c + 4 );
                loadBf16x4( x_pairs[4], x_pairs[5], x + c + 8 );
                loadBf16x4( x_pairs[6], x_pairs[7], x + c + 12 );

                float chunk = 0.0f;
#pragma unroll
                for ( int pair = 0; pair < 8; ++pair )
                {
                    const float2 x_f = __bfloat1622float2( x_pairs[pair] );
                    const int k0 = 2 * pair;
                    const int k1 = 2 * pair + 1;

                    std::uint32_t code0 = ( low_bits >> ( 2 * k0 ) ) & 0x3u;
                    std::uint32_t code1 = ( low_bits >> ( 2 * k1 ) ) & 0x3u;

                    if constexpr ( kEntries == 8 )
                    {
                        code0 |= ( ( high_bits >> k0 ) & 0x1u ) << 2;
                        code1 |= ( ( high_bits >> k1 ) & 0x1u ) << 2;
                    }

                    chunk += x_f.x * codebook_registers[code0]
                        + x_f.y * codebook_registers[code1];
                }

                const float scale = __half2float( scales[oc * num_groups + c / kGroupSize] );
                acc += scale * chunk;
            }

#pragma unroll
            for ( int offset = kThreadsPerOutput / 2; offset > 0; offset >>= 1 )
            {
                acc += __shfl_down_sync( 0xffffffff, acc, offset );
            }

            if ( threadIdx.x == 0 )
            {
                const float bias_value = ( bias != nullptr ) ? __bfloat162float( bias[oc] ) : 0.0f;
                y[oc] = __float2bfloat16( acc + bias_value );
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
            const dim3 block( kThreadsPerOutput, kBlockOutputs );
            const dim3 grid( ( OC + kBlockOutputs - 1 ) / kBlockOutputs );

            switch ( group_size )
            {
                case 32:
                    codebook_matvec_decode_kernel<32, kEntries><<<grid, block, 0, stream>>>(
                        y, x, plane_two_bits, plane_one_bit, scales, codebook, bias, C, OC );
                    break;

                case 64:
                    codebook_matvec_decode_kernel<64, kEntries><<<grid, block, 0, stream>>>(
                        y, x, plane_two_bits, plane_one_bit, scales, codebook, bias, C, OC );
                    break;

                case 128:
                    codebook_matvec_decode_kernel<128, kEntries><<<grid, block, 0, stream>>>(
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
