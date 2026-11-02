// What rotating keys on read costs a decode step (RopeInAttention.md section 5, fact 2).
//
// Gemma 4's global layers project K and V with one matrix: V = v_norm( x ) with v_norm weightless, and
// K = RoPE( V * w_k ), rotating only the first 128 of 512 dims (64 pairs). A cache could hold V alone. K's 384
// unrotated dims are then V times w_k, and w_k folds into the query once per step, so the MMA reads those dims of K
// from the V stage itself; only the 128 rotated dims need building, per key, per step. This prices that build on
// decode's tile loop -- cp.async a 16-key tile of E4M3 codes, widen into BF16 in shared memory, consume every element
// as the MMAs would -- for one global layer of the 26B-A4B (2 KV heads x 512 dims, FP8):
//
//   two tensors   read K and V codes, widen both                                      (today)
//   sincosf       read V codes, widen V, rotate 64 pairs per key with sincosf
//   recurrence    as sincosf, each thread's keys stepping from one sincosf at its first key
//   table         as sincosf, cos and sin per position and pair read from a BF16 table
//
// No MMA competes for issue here, so the K = V arms' extra work shows at least as plainly as it would in the real
// kernel. Gate, from RopeInAttention.md section 5, written before the run: at 64K a K = V arm no slower than today
// by more than 5%; the saving it buys is the cache's memory, not speed.
//
//   nvcc -gencode=arch=compute_120,code=sm_120 -O3 RotateOnRead.cu -o RotateOnRead.exe
//   CUDA_VISIBLE_DEVICES=GPU-<uuid> RotateOnRead.exe

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_pipeline.h>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error %s at line %d: %s\n", #x, __LINE__, cudaGetErrorString(e)); \
    return 1; } } while (0)

constexpr int kHeadSize = 512;
constexpr int kRotaryPairs = 64;                  // dims [0, 64) pair with [64, 128)
constexpr int kRotatedDims = 2 * kRotaryPairs;
constexpr int kKvHeads = 2;
constexpr int kTileKeys = 16;                     // double-buffered K and V codes and two BF16 stages fit 99 KB
constexpr int kKeysPerBlock = 512;
constexpr int kThreads = 256;
constexpr int kPad = kHeadSize + 8;
constexpr int kRotatedPad = kRotatedDims + 8;
constexpr int kChunksPerRow = kHeadSize / 16;      // 16 codes a chunk
constexpr int kKeysPerThread = kTileKeys * kRotaryPairs / kThreads;   // recurrence: each thread's run of keys

static_assert( kThreads % kRotaryPairs == 0, "a thread's pair is fixed across the tile loop" );

enum class Arm { TwoTensors, Sincos, Recurrence, Table };

__device__ __forceinline__ void widen16( const uint8_t* codes, float scale, __nv_bfloat16* out )
{
    const uint4 raw = *reinterpret_cast<const uint4*>( codes );
    const uint32_t words[ 4 ] = { raw.x, raw.y, raw.z, raw.w };
    uint32_t packed[ 8 ];

#pragma unroll
    for ( int w = 0; w < 4; ++w )
    {
#pragma unroll
        for ( int h = 0; h < 2; ++h )
        {
            const __nv_fp8x2_storage_t pair = static_cast<__nv_fp8x2_storage_t>( words[ w ] >> ( 16 * h ) );
            const float2 v = __half22float2( __half2( __nv_cvt_fp8x2_to_halfraw2( pair, __NV_E4M3 ) ) );
            const __nv_bfloat162 b = __floats2bfloat162_rn( v.x * scale, v.y * scale );
            packed[ 2 * w + h ] = *reinterpret_cast<const uint32_t*>( &b );
        }
    }

    reinterpret_cast<uint4*>( out )[ 0 ] = make_uint4( packed[ 0 ], packed[ 1 ], packed[ 2 ], packed[ 3 ] );
    reinterpret_cast<uint4*>( out )[ 1 ] = make_uint4( packed[ 4 ], packed[ 5 ], packed[ 6 ], packed[ 7 ] );
}

template<Arm kArm>
__global__ void __launch_bounds__( kThreads ) tile_loop_kernel(
    const uint8_t* __restrict__ k_codes, const uint8_t* __restrict__ v_codes, const float* __restrict__ scales,
    const float* __restrict__ k_weight, const float* __restrict__ inverse_frequency,
    const __nv_bfloat162* __restrict__ angle_table, int positions, float* __restrict__ sink )
{
    constexpr bool kTwoTensors = kArm == Arm::TwoTensors;

    extern __shared__ __align__( 16 ) char smem[];
    uint8_t* s_v_codes = reinterpret_cast<uint8_t*>( smem );                                  // [2][16][512]
    uint8_t* s_k_codes = s_v_codes + 2 * kTileKeys * kHeadSize;                               // [2][16][512], today only
    __nv_bfloat16* s_v = reinterpret_cast<__nv_bfloat16*>( s_k_codes + 2 * kTileKeys * kHeadSize );
    __nv_bfloat16* s_k = s_v + kTileKeys * kPad;     // today: [16][kPad]; K = V: [16][kRotatedPad], rotated dims only
    float* s_scale = reinterpret_cast<float*>( s_k + kTileKeys * kPad );                       // [2][16]

    const int tid = threadIdx.x;
    const int head = blockIdx.y;
    const int first_key = blockIdx.x * kKeysPerBlock;
    const std::size_t head_base = static_cast<std::size_t>( head ) * positions * kHeadSize;

    // A thread's rotated pair never changes: its weights and frequency live in registers.
    const int pair = tid % kRotaryPairs;
    [[maybe_unused]] const float weight_low = k_weight[ pair ];
    [[maybe_unused]] const float weight_high = k_weight[ kRotaryPairs + pair ];
    [[maybe_unused]] const float frequency = inverse_frequency[ pair ];

    const auto load = [&]( int tile, int stage )
    {
        const int key0 = first_key + tile * kTileKeys;

        for ( int chunk = tid; chunk < kTileKeys * kChunksPerRow; chunk += kThreads )
        {
            const int key = chunk / kChunksPerRow;
            const int column = ( chunk % kChunksPerRow ) * 16;
            const std::size_t source = head_base + static_cast<std::size_t>( key0 + key ) * kHeadSize + column;

            __pipeline_memcpy_async( s_v_codes + stage * kTileKeys * kHeadSize + key * kHeadSize + column, v_codes + source, 16 );

            if constexpr ( kTwoTensors )
                __pipeline_memcpy_async( s_k_codes + stage * kTileKeys * kHeadSize + key * kHeadSize + column, k_codes + source, 16 );
        }

        if ( tid < kTileKeys )
            __pipeline_memcpy_async( s_scale + stage * kTileKeys + tid, scales + head * positions + key0 + tid, 4 );
    };

    float acc = 0.0f;
    const int tiles = kKeysPerBlock / kTileKeys;

    load( 0, 0 );
    __pipeline_commit();

    for ( int tile = 0; tile < tiles; ++tile )
    {
        const int stage = tile & 1;
        [[maybe_unused]] const int key0 = first_key + tile * kTileKeys;

        __pipeline_wait_prior( 0 );
        __syncthreads();

        if ( tile + 1 < tiles )
        {
            load( tile + 1, stage ^ 1 );
            __pipeline_commit();
        }

        const uint8_t* vc = s_v_codes + stage * kTileKeys * kHeadSize;
        const float* sc = s_scale + stage * kTileKeys;

        for ( int chunk = tid; chunk < kTileKeys * kChunksPerRow; chunk += kThreads )
        {
            const int key = chunk / kChunksPerRow;
            const int column = ( chunk % kChunksPerRow ) * 16;
            widen16( vc + key * kHeadSize + column, sc[ key ], s_v + key * kPad + column );
        }

        if constexpr ( kTwoTensors )
        {
            const uint8_t* kc = s_k_codes + stage * kTileKeys * kHeadSize;

            for ( int chunk = tid; chunk < kTileKeys * kChunksPerRow; chunk += kThreads )
            {
                const int key = chunk / kChunksPerRow;
                const int column = ( chunk % kChunksPerRow ) * 16;
                widen16( kc + key * kHeadSize + column, sc[ key ], s_k + key * kPad + column );
            }
        }
        else
        {
            // The rotated dims from the codes directly: this thread's pair, keys tid / 64 + 4 k.
            const auto rotate = [&]( int key, float c, float s )
            {
                const uint8_t* row = vc + key * kHeadSize;
                const float scale = sc[ key ];
                const float x = static_cast<float>( *reinterpret_cast<const __nv_fp8_e4m3*>( row + pair ) ) * scale * weight_low;
                const float y = static_cast<float>( *reinterpret_cast<const __nv_fp8_e4m3*>( row + kRotaryPairs + pair ) ) * scale * weight_high;

                s_k[ key * kRotatedPad + pair ] = __float2bfloat16_rn( x * c - y * s );
                s_k[ key * kRotatedPad + kRotaryPairs + pair ] = __float2bfloat16_rn( y * c + x * s );
            };

            if constexpr ( kArm == Arm::Recurrence )
            {
                // A contiguous run of keys per thread: one sincosf at its first, then steps of the pair's angle.
                const int first = ( tid / kRotaryPairs ) * kKeysPerThread;
                float c, s, step_c, step_s;
                sincosf( static_cast<float>( key0 + first ) * frequency, &s, &c );
                sincosf( frequency, &step_s, &step_c );

#pragma unroll
                for ( int k = 0; k < kKeysPerThread; ++k )
                {
                    rotate( first + k, c, s );

                    const float next = c * step_c - s * step_s;
                    s = s * step_c + c * step_s;
                    c = next;
                }
            }
            else
            {
#pragma unroll
                for ( int key = tid / kRotaryPairs; key < kTileKeys; key += kThreads / kRotaryPairs )
                {
                    float c, s;

                    if constexpr ( kArm == Arm::Table )
                    {
                        const float2 cs = __bfloat1622float2( angle_table[ static_cast<std::size_t>( key0 + key ) * kRotaryPairs + pair ] );
                        c = cs.x;
                        s = cs.y;
                    }
                    else
                    {
                        sincosf( static_cast<float>( key0 + key ) * frequency, &s, &c );
                    }

                    rotate( key, c, s );
                }
            }
        }

        __syncthreads();

        // Consume every element of K and V, as the MMAs read them. Under K = V, K's dims past the rotated ones are
        // V's own (w_k folded into the query).
        for ( int i = tid; i < kTileKeys * kHeadSize; i += kThreads )
        {
            const int key = i / kHeadSize;
            const int dim = i % kHeadSize;
            const float v = __bfloat162float( s_v[ key * kPad + dim ] );
            float k;

            if constexpr ( kTwoTensors )
                k = __bfloat162float( s_k[ key * kPad + dim ] );
            else
                k = dim < kRotatedDims ? __bfloat162float( s_k[ key * kRotatedPad + dim ] ) : v;

            acc += k * v;
        }
    }

    if ( acc == 12345.678f )
        sink[ blockIdx.y * gridDim.x + blockIdx.x ] = acc;
}

template<Arm kArm>
int time_arm( int positions, const uint8_t* k, const uint8_t* v, const float* scales, const float* weight,
    const float* frequency, const __nv_bfloat162* table, float* sink, float* out_us )
{
    const std::size_t shared = 4 * kTileKeys * kHeadSize + 2 * kTileKeys * kPad * sizeof( __nv_bfloat16 )
        + 2 * kTileKeys * sizeof( float );
    CHECK( cudaFuncSetAttribute( tile_loop_kernel<kArm>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>( shared ) ) );

    const dim3 grid( positions / kKeysPerBlock, kKvHeads );
    cudaEvent_t start, stop;
    CHECK( cudaEventCreate( &start ) );
    CHECK( cudaEventCreate( &stop ) );

    for ( int i = 0; i < 20; ++i )
        tile_loop_kernel<kArm><<<grid, kThreads, shared>>>( k, v, scales, weight, frequency, table, positions, sink );

    CHECK( cudaGetLastError() );
    std::vector<float> times;

    for ( int i = 0; i < 200; ++i )
    {
        CHECK( cudaEventRecord( start ) );
        tile_loop_kernel<kArm><<<grid, kThreads, shared>>>( k, v, scales, weight, frequency, table, positions, sink );
        CHECK( cudaEventRecord( stop ) );
        CHECK( cudaEventSynchronize( stop ) );
        float ms;
        CHECK( cudaEventElapsedTime( &ms, start, stop ) );
        times.push_back( ms * 1000.0f );
    }

    std::sort( times.begin(), times.end() );
    *out_us = times[ times.size() / 2 ];

    return 0;
}

int main()
{
    cudaDeviceProp prop;
    CHECK( cudaGetDeviceProperties( &prop, 0 ) );
    printf( "%s, %d SMs, L2 %d MB\n", prop.name, prop.multiProcessorCount, prop.l2CacheSize >> 20 );

    for ( const int positions : { 32768, 65536 } )
    {
        // One layer's codes, 33 to 67 MB a tensor, are past the card's L2: every call reads DRAM.
        const std::size_t codes = static_cast<std::size_t>( kKvHeads ) * positions * kHeadSize;
        uint8_t *k, *v;
        float *scales, *weight, *frequency, *sink;
        __nv_bfloat162* table;
        CHECK( cudaMalloc( &k, codes ) );
        CHECK( cudaMalloc( &v, codes ) );
        CHECK( cudaMalloc( &scales, kKvHeads * positions * sizeof( float ) ) );
        CHECK( cudaMalloc( &weight, kHeadSize * sizeof( float ) ) );
        CHECK( cudaMalloc( &frequency, kRotaryPairs * sizeof( float ) ) );
        CHECK( cudaMalloc( &table, static_cast<std::size_t>( positions ) * kRotaryPairs * sizeof( __nv_bfloat162 ) ) );
        CHECK( cudaMalloc( &sink, 4096 * sizeof( float ) ) );
        CHECK( cudaMemset( k, 0x38, codes ) );
        CHECK( cudaMemset( v, 0x38, codes ) );
        CHECK( cudaMemset( table, 0, static_cast<std::size_t>( positions ) * kRotaryPairs * sizeof( __nv_bfloat162 ) ) );

        std::vector<float> host( kKvHeads * positions, 1.0f / 448.0f );
        CHECK( cudaMemcpy( scales, host.data(), host.size() * sizeof( float ), cudaMemcpyHostToDevice ) );
        std::vector<float> ones( kHeadSize, 1.0f );
        CHECK( cudaMemcpy( weight, ones.data(), ones.size() * sizeof( float ), cudaMemcpyHostToDevice ) );
        std::vector<float> frequencies( kRotaryPairs );

        for ( int i = 0; i < kRotaryPairs; ++i )
            frequencies[ i ] = std::pow( 1.0e6f, -2.0f * i / kHeadSize );

        CHECK( cudaMemcpy( frequency, frequencies.data(), frequencies.size() * sizeof( float ), cudaMemcpyHostToDevice ) );

        float two = 0, sincos = 0, recurrence = 0, tabled = 0;
        if ( time_arm<Arm::TwoTensors>( positions, k, v, scales, weight, frequency, table, sink, &two ) ) return 1;
        if ( time_arm<Arm::Sincos>( positions, k, v, scales, weight, frequency, table, sink, &sincos ) ) return 1;
        if ( time_arm<Arm::Recurrence>( positions, k, v, scales, weight, frequency, table, sink, &recurrence ) ) return 1;
        if ( time_arm<Arm::Table>( positions, k, v, scales, weight, frequency, table, sink, &tabled ) ) return 1;

        printf( "\n%d positions, one global layer (%.1f MB of codes a tensor), median of 200, against today:\n",
            positions, codes / 1e6 );
        printf( "  two tensors  %8.1f us  %6.1f GB/s\n", two, 2.0 * codes / two / 1e3 );
        printf( "  sincosf      %8.1f us  %6.1f GB/s  %.3f\n", sincos, 1.0 * codes / sincos / 1e3, sincos / two );
        printf( "  recurrence   %8.1f us  %6.1f GB/s  %.3f\n", recurrence, 1.0 * codes / recurrence / 1e3, recurrence / two );
        printf( "  table        %8.1f us  %6.1f GB/s  %.3f (the table is %.1f MB)\n", tabled, 1.0 * codes / tabled / 1e3,
            tabled / two, positions * kRotaryPairs * 4.0 / 1e6 );

        cudaFree( k ); cudaFree( v ); cudaFree( scales ); cudaFree( weight ); cudaFree( frequency ); cudaFree( table ); cudaFree( sink );
    }

    return 0;
}
