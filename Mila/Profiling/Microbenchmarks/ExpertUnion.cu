// What the Gemma 4 26B-A4B's Q4_0 expert bank costs a verify of R rows against one decode (Gemma4Mtp.md 4.7, step 5).
//
// Each row of a verify routes to its own 8 of 128 experts, so R rows read the union of their experts. The arms run the
// bank's two decode passes (gate and up, then down and combine) over real routing -- every reply row's experts as
// `Drafting routing --target 26b --output <dir>` writes them -- on random weights of the 26B's shapes, DRAM-resident
// (three banks of 428 MB in rotation, one per layer call), each timed over a stream of calls:
//
//   rows-apart   today's gather kernels (MoeGather.cu, copied verbatim) once per row: R one-row decodes' work
//   today        the same kernels at R tokens: grid (unit chunk, slot, token), so a token's blocks run after the
//                previous token's and a shared expert is read again unless L2 still holds it
//   token-first  the same kernel bodies with the block order changed so the R tokens' blocks for the same rows run
//                together: a chosen expert's rows come from DRAM once and from L2 for every other row that chose it
//   grouped      the gate and up pass over the union: a block per (unit chunk, distinct expert) loads the codes once
//                and applies them to every row that chose that expert; the combine pass as token-first
//
// Every arm keeps each row's arithmetic exactly the one-row kernel's, so each is checked bit for bit against
// rows-apart before it is timed. Direction, written before the run: Gemma4Mtp.md 5.1 prices the 26B's verify at 1.9
// decodes from a derived 3x expert union at R = 5, and 1.82x chat speedup at an ideal 1.0; the arm nearest union
// bytes at the bank's one-row bandwidth is the design.
//
//   nvcc -gencode=arch=compute_120,code=sm_120 -gencode=arch=compute_89,code=sm_89 -O3 ExpertUnion.cu -o ExpertUnion.exe
//   CUDA_VISIBLE_DEVICES=GPU-<uuid> ExpertUnion.exe <prompt>.routing ...

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error %s at line %d: %s\n", #x, __LINE__, cudaGetErrorString(e)); \
    std::exit(1); } } while (0)

constexpr int kHidden = 2816;
constexpr int kIntermediate = 704;
constexpr int kExperts = 128;
constexpr int kTopK = 8;
constexpr int kLayers = 30;
constexpr int kMaxRows = 8;
constexpr int kBanks = 3;
constexpr int kTrials = 5;

constexpr int kGroup = 32;
constexpr int kWarpsPerBlock = 8;
constexpr int kUnitsPerWarp = 2;
constexpr int kColumnsPerWarp = 2;
constexpr int kPaddedGroup = 36;

struct GeluTanh
{
    __device__ float fwd( float x ) const
    {
        float cube = 0.044715f * x * x * x;
        return 0.5f * x * ( 1.0f + tanhf( 0.7978845608f * ( x + cube ) ) );
    }
};

// ---------------------------------------------------------------------------------------------------------------
// MoeGather.cu's helpers, verbatim
// ---------------------------------------------------------------------------------------------------------------

__device__ __forceinline__ float centered_code( uint32_t word, int index )
{
    return __uint_as_float( 0x4B000000u | ( ( word >> ( 4 * index ) ) & 0xFu ) ) - 8388616.0f;
}

__device__ __forceinline__ float warp_sum( float value )
{
#pragma unroll
    for ( int offset = 16; offset > 0; offset >>= 1 )
    {
        value += __shfl_xor_sync( 0xffffffff, value, offset );
    }

    return value;
}

__device__ __forceinline__ void group_dot_pair_bf16(
    const int4 gate_codes, const int4 up_codes, const __nv_bfloat16* x, float& gate_sum, float& up_sum )
{
    const uint32_t gate_words[ 4 ] = {
        static_cast<uint32_t>( gate_codes.x ), static_cast<uint32_t>( gate_codes.y ),
        static_cast<uint32_t>( gate_codes.z ), static_cast<uint32_t>( gate_codes.w ) };
    const uint32_t up_words[ 4 ] = {
        static_cast<uint32_t>( up_codes.x ), static_cast<uint32_t>( up_codes.y ),
        static_cast<uint32_t>( up_codes.z ), static_cast<uint32_t>( up_codes.w ) };

    gate_sum = 0.0f;
    up_sum = 0.0f;

#pragma unroll
    for ( int word = 0; word < 4; ++word )
    {
        const int4 raw = *reinterpret_cast<const int4*>( x + word * 8 );
        const __nv_bfloat162* pairs = reinterpret_cast<const __nv_bfloat162*>( &raw );

#pragma unroll
        for ( int pair = 0; pair < 4; ++pair )
        {
            const float2 value = __bfloat1622float2( pairs[ pair ] );

            gate_sum += value.x * centered_code( gate_words[ word ], 2 * pair )
                      + value.y * centered_code( gate_words[ word ], 2 * pair + 1 );
            up_sum += value.x * centered_code( up_words[ word ], 2 * pair )
                    + value.y * centered_code( up_words[ word ], 2 * pair + 1 );
        }
    }
}

__device__ __forceinline__ float group_dot_fp32( const int4 codes, const float* g )
{
    const uint32_t words[ 4 ] = {
        static_cast<uint32_t>( codes.x ), static_cast<uint32_t>( codes.y ),
        static_cast<uint32_t>( codes.z ), static_cast<uint32_t>( codes.w ) };

    float even = 0.0f;
    float odd = 0.0f;

#pragma unroll
    for ( int word = 0; word < 4; ++word )
    {
        const float4 low = *reinterpret_cast<const float4*>( g + word * 8 );
        const float4 high = *reinterpret_cast<const float4*>( g + word * 8 + 4 );
        float& sum = ( word % 2 == 0 ) ? even : odd;

        sum += low.x * centered_code( words[ word ], 0 ) + low.y * centered_code( words[ word ], 1 )
             + low.z * centered_code( words[ word ], 2 ) + low.w * centered_code( words[ word ], 3 )
             + high.x * centered_code( words[ word ], 4 ) + high.y * centered_code( words[ word ], 5 )
             + high.z * centered_code( words[ word ], 6 ) + high.w * centered_code( words[ word ], 7 );
    }

    return even + odd;
}

// ---------------------------------------------------------------------------------------------------------------
// The gated pass, MoeGather.cu's body; kTokenFirst only changes which block is which
// ---------------------------------------------------------------------------------------------------------------

template<bool kTokenFirst>
__global__ void __launch_bounds__( 32 * kWarpsPerBlock, 2 ) gated_gather_kernel(
    const __nv_bfloat16* __restrict__ input, const uint8_t* __restrict__ gate_up,
    const __half* __restrict__ gate_up_scales, const int32_t* __restrict__ indices, float* __restrict__ gated,
    int tokens, int hidden, int intermediate, int experts, int top_k, GeluTanh functor )
{
    int chunk;
    int slot;
    int64_t token;

    if constexpr ( kTokenFirst )
    {
        const int block = blockIdx.x;
        token = block % tokens;
        slot = ( block / tokens ) % top_k;
        chunk = block / ( tokens * top_k );
    }
    else
    {
        chunk = blockIdx.x;
        slot = blockIdx.y;
        token = blockIdx.z;
    }

    const int first_unit = ( chunk * kWarpsPerBlock + threadIdx.y ) * kUnitsPerWarp;

    if ( first_unit >= intermediate )
    {
        return;
    }

    const int64_t first_cell = ( token * top_k + slot ) * intermediate + first_unit;
    const int32_t expert = indices[ token * top_k + slot ];

    if ( expert < 0 || expert >= experts )
    {
        if ( threadIdx.x == 0 )
        {
            for ( int u = 0; u < kUnitsPerWarp && first_unit + u < intermediate; ++u )
            {
                gated[ first_cell + u ] = NAN;
            }
        }

        return;
    }

    const int groups = hidden / kGroup;
    const __nv_bfloat16* x = input + token * hidden;

    const int4* gate_codes[ kUnitsPerWarp ];
    const int4* up_codes[ kUnitsPerWarp ];
    const __half* gate_scales[ kUnitsPerWarp ];
    const __half* up_scales[ kUnitsPerWarp ];

#pragma unroll
    for ( int u = 0; u < kUnitsPerWarp; ++u )
    {
        const int unit = min( first_unit + u, intermediate - 1 );
        const int64_t gate_row = static_cast<int64_t>( expert ) * 2 * intermediate + unit;
        const int64_t up_row = gate_row + intermediate;

        gate_codes[ u ] = reinterpret_cast<const int4*>( gate_up + gate_row * ( hidden / 2 ) );
        up_codes[ u ] = reinterpret_cast<const int4*>( gate_up + up_row * ( hidden / 2 ) );
        gate_scales[ u ] = gate_up_scales + gate_row * groups;
        up_scales[ u ] = gate_up_scales + up_row * groups;
    }

    float gate[ kUnitsPerWarp ] = {};
    float up[ kUnitsPerWarp ] = {};

    for ( int group = threadIdx.x; group < groups; group += 32 )
    {
        int4 gate_group[ kUnitsPerWarp ];
        int4 up_group[ kUnitsPerWarp ];

#pragma unroll
        for ( int u = 0; u < kUnitsPerWarp; ++u )
        {
            gate_group[ u ] = gate_codes[ u ][ group ];
            up_group[ u ] = up_codes[ u ][ group ];
        }

#pragma unroll
        for ( int u = 0; u < kUnitsPerWarp; ++u )
        {
            float gate_sum;
            float up_sum;

            group_dot_pair_bf16( gate_group[ u ], up_group[ u ], x + group * kGroup, gate_sum, up_sum );

            gate[ u ] = fmaf( __half2float( gate_scales[ u ][ group ] ), gate_sum, gate[ u ] );
            up[ u ] = fmaf( __half2float( up_scales[ u ][ group ] ), up_sum, up[ u ] );
        }
    }

#pragma unroll
    for ( int u = 0; u < kUnitsPerWarp; ++u )
    {
        const float gate_total = warp_sum( gate[ u ] );
        const float up_total = warp_sum( up[ u ] );

        if ( threadIdx.x == 0 && first_unit + u < intermediate )
        {
            gated[ first_cell + u ] = functor.fwd( gate_total ) * up_total;
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// The gated pass over the union: grid ( unit chunk, distinct expert ). Each block lists the rows' distinct experts in
// the order (row, slot) first meets them; block y takes the y-th and returns past the count. Each row's sums run in
// the one-row kernel's order, so its gated values are that kernel's bits.
// ---------------------------------------------------------------------------------------------------------------

__global__ void __launch_bounds__( 32 * kWarpsPerBlock, 2 ) gated_grouped_kernel(
    const __nv_bfloat16* __restrict__ input, const uint8_t* __restrict__ gate_up,
    const __half* __restrict__ gate_up_scales, const int32_t* __restrict__ indices, float* __restrict__ gated,
    int tokens, int hidden, int intermediate, int experts, int top_k, GeluTanh functor )
{
    __shared__ int entry_expert;
    __shared__ int row_slot[ kMaxRows ];

    if ( threadIdx.x == 0 && threadIdx.y == 0 )
    {
        int seen[ kMaxRows * kTopK ];
        int count = 0;
        int expert_found = -1;

        for ( int pair = 0; pair < tokens * top_k && expert_found < 0; ++pair )
        {
            const int candidate = indices[ pair ];
            bool fresh = true;

            for ( int s = 0; s < count; ++s )
            {
                fresh = fresh && seen[ s ] != candidate;
            }

            if ( fresh )
            {
                if ( count == static_cast<int>( blockIdx.y ) )
                {
                    expert_found = candidate;
                }

                seen[ count++ ] = candidate;
            }
        }

        entry_expert = expert_found;

        for ( int r = 0; r < kMaxRows; ++r )
        {
            row_slot[ r ] = -1;

            for ( int s = 0; r < tokens && s < top_k; ++s )
            {
                if ( expert_found >= 0 && indices[ r * top_k + s ] == expert_found )
                {
                    row_slot[ r ] = s;
                }
            }
        }
    }

    __syncthreads();

    const int expert = entry_expert;
    const int first_unit = ( blockIdx.x * kWarpsPerBlock + threadIdx.y ) * kUnitsPerWarp;

    if ( expert < 0 || expert >= experts || first_unit >= intermediate )
    {
        return;
    }

    const int groups = hidden / kGroup;

    const int4* gate_codes[ kUnitsPerWarp ];
    const int4* up_codes[ kUnitsPerWarp ];
    const __half* gate_scales[ kUnitsPerWarp ];
    const __half* up_scales[ kUnitsPerWarp ];

#pragma unroll
    for ( int u = 0; u < kUnitsPerWarp; ++u )
    {
        const int unit = min( first_unit + u, intermediate - 1 );
        const int64_t gate_row = static_cast<int64_t>( expert ) * 2 * intermediate + unit;
        const int64_t up_row = gate_row + intermediate;

        gate_codes[ u ] = reinterpret_cast<const int4*>( gate_up + gate_row * ( hidden / 2 ) );
        up_codes[ u ] = reinterpret_cast<const int4*>( gate_up + up_row * ( hidden / 2 ) );
        gate_scales[ u ] = gate_up_scales + gate_row * groups;
        up_scales[ u ] = gate_up_scales + up_row * groups;
    }

    int slots[ kMaxRows ];

#pragma unroll
    for ( int r = 0; r < kMaxRows; ++r )
    {
        slots[ r ] = row_slot[ r ];
    }

    float gate[ kMaxRows ][ kUnitsPerWarp ] = {};
    float up[ kMaxRows ][ kUnitsPerWarp ] = {};

    for ( int group = threadIdx.x; group < groups; group += 32 )
    {
        int4 gate_group[ kUnitsPerWarp ];
        int4 up_group[ kUnitsPerWarp ];
        float gate_scale[ kUnitsPerWarp ];
        float up_scale[ kUnitsPerWarp ];

#pragma unroll
        for ( int u = 0; u < kUnitsPerWarp; ++u )
        {
            gate_group[ u ] = gate_codes[ u ][ group ];
            up_group[ u ] = up_codes[ u ][ group ];
            gate_scale[ u ] = __half2float( gate_scales[ u ][ group ] );
            up_scale[ u ] = __half2float( up_scales[ u ][ group ] );
        }

#pragma unroll
        for ( int r = 0; r < kMaxRows; ++r )
        {
            if ( slots[ r ] < 0 )
            {
                continue;
            }

            const __nv_bfloat16* x = input + static_cast<int64_t>( r ) * hidden + group * kGroup;

#pragma unroll
            for ( int u = 0; u < kUnitsPerWarp; ++u )
            {
                float gate_sum;
                float up_sum;

                group_dot_pair_bf16( gate_group[ u ], up_group[ u ], x, gate_sum, up_sum );

                gate[ r ][ u ] = fmaf( gate_scale[ u ], gate_sum, gate[ r ][ u ] );
                up[ r ][ u ] = fmaf( up_scale[ u ], up_sum, up[ r ][ u ] );
            }
        }
    }

#pragma unroll
    for ( int r = 0; r < kMaxRows; ++r )
    {
        if ( slots[ r ] < 0 )
        {
            continue;
        }

#pragma unroll
        for ( int u = 0; u < kUnitsPerWarp; ++u )
        {
            const float gate_total = warp_sum( gate[ r ][ u ] );
            const float up_total = warp_sum( up[ r ][ u ] );

            if ( threadIdx.x == 0 && first_unit + u < intermediate )
            {
                gated[ ( static_cast<int64_t>( r ) * top_k + slots[ r ] ) * intermediate + first_unit + u ] =
                    functor.fwd( gate_total ) * up_total;
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// The combine pass, MoeGather.cu's body; kTokenFirst puts the R tokens' blocks for the same columns together
// ---------------------------------------------------------------------------------------------------------------

template<bool kTokenFirst>
__global__ void __launch_bounds__( 32 * kWarpsPerBlock, 2 ) combine_gather_kernel(
    const float* __restrict__ gated, const uint8_t* __restrict__ down, const __half* __restrict__ down_scales,
    const __nv_bfloat16* __restrict__ weights, const int32_t* __restrict__ indices,
    __nv_bfloat16* __restrict__ output, int tokens, int hidden, int intermediate, int experts, int top_k )
{
    extern __shared__ float4 shared_groups[];
    float* staged = reinterpret_cast<float*>( shared_groups );

    int chunk;
    int64_t token;

    if constexpr ( kTokenFirst )
    {
        token = blockIdx.x % tokens;
        chunk = blockIdx.x / tokens;
    }
    else
    {
        chunk = blockIdx.x;
        token = blockIdx.y;
    }

    const int groups_per_row = intermediate / kGroup;
    const int token_groups = top_k * groups_per_row;

    const float4* source = reinterpret_cast<const float4*>( gated + token * top_k * intermediate );
    const int thread = threadIdx.y * 32 + threadIdx.x;

    for ( int quad = thread; quad < token_groups * ( kGroup / 4 ); quad += 32 * kWarpsPerBlock )
    {
        const int group = quad / ( kGroup / 4 );

        reinterpret_cast<float4*>( staged + group * kPaddedGroup )[ quad % ( kGroup / 4 ) ] = source[ quad ];
    }

    __syncthreads();

    const int first_column = ( chunk * kWarpsPerBlock + threadIdx.y ) * kColumnsPerWarp;

    if ( first_column >= hidden )
    {
        return;
    }

    float sum[ kColumnsPerWarp ] = {};

    for ( int group = threadIdx.x; group < token_groups; group += 32 )
    {
        const int slot = group / groups_per_row;
        const int within = group - slot * groups_per_row;
        const int32_t expert = indices[ token * top_k + slot ];

        if ( expert < 0 || expert >= experts )
        {
#pragma unroll
            for ( int c = 0; c < kColumnsPerWarp; ++c )
            {
                sum[ c ] += NAN;
            }

            continue;
        }

        int4 codes[ kColumnsPerWarp ];
        float scale[ kColumnsPerWarp ];

#pragma unroll
        for ( int c = 0; c < kColumnsPerWarp; ++c )
        {
            const int64_t row = static_cast<int64_t>( expert ) * hidden + min( first_column + c, hidden - 1 );

            codes[ c ] = reinterpret_cast<const int4*>( down + row * ( intermediate / 2 ) )[ within ];
            scale[ c ] = __half2float( down_scales[ row * groups_per_row + within ] );
        }

        const float combine = __bfloat162float( weights[ token * top_k + slot ] );

#pragma unroll
        for ( int c = 0; c < kColumnsPerWarp; ++c )
        {
            sum[ c ] = fmaf( combine * scale[ c ], group_dot_fp32( codes[ c ], staged + group * kPaddedGroup ), sum[ c ] );
        }
    }

#pragma unroll
    for ( int c = 0; c < kColumnsPerWarp; ++c )
    {
        const float total = warp_sum( sum[ c ] );

        if ( threadIdx.x == 0 && first_column + c < hidden )
        {
            output[ token * hidden + first_column + c ] = __float2bfloat16( total );
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Host
// ---------------------------------------------------------------------------------------------------------------

struct Bank
{
    uint8_t* gate_up;
    __half* gate_up_scales;
    uint8_t* down;
    __half* down_scales;
};

enum class Arm { RowsApart, Today, TokenFirst, Grouped };

const char* armName( Arm arm )
{
    switch ( arm )
    {
        case Arm::RowsApart: return "rows-apart";
        case Arm::Today: return "today";
        case Arm::TokenFirst: return "token-first";
        default: return "grouped";
    }
}

constexpr int kUnitChunks = ( kIntermediate + kWarpsPerBlock * kUnitsPerWarp - 1 ) / ( kWarpsPerBlock * kUnitsPerWarp );
constexpr int kColumnChunks = ( kHidden + kWarpsPerBlock * kColumnsPerWarp - 1 ) / ( kWarpsPerBlock * kColumnsPerWarp );
constexpr int kCombineShared = kTopK * ( kIntermediate / kGroup ) * kPaddedGroup * static_cast<int>( sizeof( float ) );

// One layer's bank for R rows: input [R, H], indices and weights [R, top_k], gated [R, top_k, I], output [R, H].
void runArm( Arm arm, const Bank& bank, const __nv_bfloat16* input, const int32_t* indices,
    const __nv_bfloat16* weights, float* gated, __nv_bfloat16* output, int rows, cudaStream_t stream )
{
    const dim3 block( 32, kWarpsPerBlock );

    if ( arm == Arm::RowsApart )
    {
        for ( int r = 0; r < rows; ++r )
        {
            gated_gather_kernel<false><<<dim3( kUnitChunks, kTopK, 1 ), block, 0, stream>>>(
                input + r * kHidden, bank.gate_up, bank.gate_up_scales, indices + r * kTopK,
                gated + static_cast<int64_t>( r ) * kTopK * kIntermediate,
                1, kHidden, kIntermediate, kExperts, kTopK, GeluTanh{} );
            combine_gather_kernel<false><<<dim3( kColumnChunks, 1 ), block, kCombineShared, stream>>>(
                gated + static_cast<int64_t>( r ) * kTopK * kIntermediate, bank.down, bank.down_scales,
                weights + r * kTopK, indices + r * kTopK, output + r * kHidden,
                1, kHidden, kIntermediate, kExperts, kTopK );
        }

        return;
    }

    if ( arm == Arm::Today )
    {
        gated_gather_kernel<false><<<dim3( kUnitChunks, kTopK, rows ), block, 0, stream>>>(
            input, bank.gate_up, bank.gate_up_scales, indices, gated, rows, kHidden, kIntermediate, kExperts, kTopK,
            GeluTanh{} );
        combine_gather_kernel<false><<<dim3( kColumnChunks, rows ), block, kCombineShared, stream>>>(
            gated, bank.down, bank.down_scales, weights, indices, output, rows, kHidden, kIntermediate, kExperts, kTopK );

        return;
    }

    if ( arm == Arm::TokenFirst )
    {
        gated_gather_kernel<true><<<dim3( kUnitChunks * kTopK * rows ), block, 0, stream>>>(
            input, bank.gate_up, bank.gate_up_scales, indices, gated, rows, kHidden, kIntermediate, kExperts, kTopK,
            GeluTanh{} );
    }
    else
    {
        gated_grouped_kernel<<<dim3( kUnitChunks, kTopK * rows ), block, 0, stream>>>(
            input, bank.gate_up, bank.gate_up_scales, indices, gated, rows, kHidden, kIntermediate, kExperts, kTopK,
            GeluTanh{} );
    }

    combine_gather_kernel<true><<<dim3( kColumnChunks * rows ), block, kCombineShared, stream>>>(
        gated, bank.down, bank.down_scales, weights, indices, output, rows, kHidden, kIntermediate, kExperts, kTopK );
}

std::vector<int32_t> readRouting( const char* path, int& rows )
{
    FILE* file = std::fopen( path, "rb" );

    if ( file == nullptr )
    {
        printf( "cannot open %s\n", path );
        std::exit( 1 );
    }

    std::fseek( file, 0, SEEK_END );
    const long bytes = std::ftell( file );
    std::fseek( file, 0, SEEK_SET );

    std::vector<int32_t> routing( bytes / sizeof( int32_t ) );
    std::fread( routing.data(), sizeof( int32_t ), routing.size(), file );
    std::fclose( file );

    rows = static_cast<int>( routing.size() / ( kLayers * kTopK ) );

    return routing;
}

int main( int argc, char** argv )
{
    if ( argc < 2 )
    {
        printf( "ExpertUnion <prompt>.routing ...\n" );
        return 1;
    }

    cudaDeviceProp properties{};
    CHECK( cudaGetDeviceProperties( &properties, 0 ) );
    printf( "%s, %d SMs, L2 %d MB\n", properties.name, properties.multiProcessorCount, properties.l2CacheSize >> 20 );

    CHECK( cudaFuncSetAttribute( combine_gather_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, kCombineShared ) );
    CHECK( cudaFuncSetAttribute( combine_gather_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, kCombineShared ) );

    // Random banks: codes uniform, scales small enough that every sum stays finite.
    const size_t gate_up_bytes = size_t( kExperts ) * 2 * kIntermediate * ( kHidden / 2 );
    const size_t gate_up_scale_count = size_t( kExperts ) * 2 * kIntermediate * ( kHidden / kGroup );
    const size_t down_bytes = size_t( kExperts ) * kHidden * ( kIntermediate / 2 );
    const size_t down_scale_count = size_t( kExperts ) * kHidden * ( kIntermediate / kGroup );
    const double expert_bytes = double( gate_up_bytes + gate_up_scale_count * 2 + down_bytes + down_scale_count * 2 ) / kExperts;

    std::mt19937 generator( 20261006u );
    std::uniform_int_distribution<int> byte( 0, 255 );
    std::uniform_real_distribution<float> uniform( -1.0f, 1.0f );

    std::vector<uint8_t> host_codes( std::max( gate_up_bytes, down_bytes ) );
    std::vector<__half> host_scales( std::max( gate_up_scale_count, down_scale_count ) );

    for ( auto& value : host_codes )
        value = static_cast<uint8_t>( byte( generator ) );

    for ( auto& value : host_scales )
        value = __float2half( 0.002f + 0.002f * uniform( generator ) );

    Bank banks[ kBanks ];

    for ( Bank& bank : banks )
    {
        CHECK( cudaMalloc( &bank.gate_up, gate_up_bytes ) );
        CHECK( cudaMalloc( &bank.gate_up_scales, gate_up_scale_count * 2 ) );
        CHECK( cudaMalloc( &bank.down, down_bytes ) );
        CHECK( cudaMalloc( &bank.down_scales, down_scale_count * 2 ) );
        CHECK( cudaMemcpy( bank.gate_up, host_codes.data(), gate_up_bytes, cudaMemcpyHostToDevice ) );
        CHECK( cudaMemcpy( bank.gate_up_scales, host_scales.data(), gate_up_scale_count * 2, cudaMemcpyHostToDevice ) );
        CHECK( cudaMemcpy( bank.down, host_codes.data(), down_bytes, cudaMemcpyHostToDevice ) );
        CHECK( cudaMemcpy( bank.down_scales, host_scales.data(), down_scale_count * 2, cudaMemcpyHostToDevice ) );
    }

    std::vector<__nv_bfloat16> host_input( size_t( kMaxRows ) * kHidden );
    std::vector<__nv_bfloat16> host_weights( size_t( kMaxRows ) * kTopK );

    for ( auto& value : host_input )
        value = __float2bfloat16( uniform( generator ) );

    for ( auto& value : host_weights )
        value = __float2bfloat16( 0.125f + 0.05f * uniform( generator ) );

    __nv_bfloat16* input;
    __nv_bfloat16* weights;
    float* gated;
    __nv_bfloat16* output;

    CHECK( cudaMalloc( &input, host_input.size() * 2 ) );
    CHECK( cudaMalloc( &weights, host_weights.size() * 2 ) );
    CHECK( cudaMalloc( &gated, size_t( kMaxRows ) * kTopK * kIntermediate * sizeof( float ) ) );
    CHECK( cudaMalloc( &output, size_t( kMaxRows ) * kHidden * 2 ) );
    CHECK( cudaMemcpy( input, host_input.data(), host_input.size() * 2, cudaMemcpyHostToDevice ) );
    CHECK( cudaMemcpy( weights, host_weights.data(), host_weights.size() * 2, cudaMemcpyHostToDevice ) );

    cudaStream_t stream;
    CHECK( cudaStreamCreateWithFlags( &stream, cudaStreamNonBlocking ) );

    cudaEvent_t start;
    cudaEvent_t stop;
    CHECK( cudaEventCreate( &start ) );
    CHECK( cudaEventCreate( &stop ) );

    const Arm arms[] = { Arm::RowsApart, Arm::Today, Arm::TokenFirst, Arm::Grouped };

    for ( int file = 1; file < argc; ++file )
    {
        int reply_rows = 0;
        const std::vector<int32_t> routing = readRouting( argv[ file ], reply_rows );

        // Every window of R consecutive rows, as its indices [layer][R][top_k], uploaded once.
        printf( "\n## %s (%d reply rows)\n\n", argv[ file ], reply_rows );
        printf( "| R | union / 8, the floor | rows-apart | today | token-first | grouped |\n" );
        printf( "|---|---|---|---|---|---|\n" );

        double one_row_ms = 0.0;

        for ( int rows = 1; rows <= kMaxRows; ++rows )
        {
            const int windows = std::min( 64, reply_rows / rows );
            std::vector<int32_t> host_indices( size_t( windows ) * kLayers * rows * kTopK );
            double distinct_total = 0.0;

            for ( int w = 0; w < windows; ++w )
            {
                for ( int layer = 0; layer < kLayers; ++layer )
                {
                    std::vector<int> chosen;

                    for ( int r = 0; r < rows; ++r )
                    {
                        for ( int s = 0; s < kTopK; ++s )
                        {
                            const int32_t expert = routing[ ( size_t( w * rows + r ) * kLayers + layer ) * kTopK + s ];

                            host_indices[ ( ( size_t( w ) * kLayers + layer ) * rows + r ) * kTopK + s ] = expert;
                            chosen.push_back( expert );
                        }
                    }

                    std::sort( chosen.begin(), chosen.end() );
                    distinct_total += std::unique( chosen.begin(), chosen.end() ) - chosen.begin();
                }
            }

            const double union_per_layer = distinct_total / ( double( windows ) * kLayers );

            int32_t* indices;
            CHECK( cudaMalloc( &indices, host_indices.size() * sizeof( int32_t ) ) );
            CHECK( cudaMemcpy( indices, host_indices.data(), host_indices.size() * sizeof( int32_t ), cudaMemcpyHostToDevice ) );

            const auto window = [&]( int w, int layer )
            {
                return indices + ( size_t( w ) * kLayers + layer ) * rows * kTopK;
            };

            // Bits first: every arm's gated values and output against rows-apart, over every layer of window 0.
            std::vector<float> reference_gated( size_t( rows ) * kTopK * kIntermediate );
            std::vector<__nv_bfloat16> reference_output( size_t( rows ) * kHidden );
            std::vector<float> arm_gated( reference_gated.size() );
            std::vector<__nv_bfloat16> arm_output( reference_output.size() );

            for ( int layer = 0; layer < kLayers; ++layer )
            {
                for ( Arm arm : arms )
                {
                    CHECK( cudaMemsetAsync( gated, 0xFF, reference_gated.size() * sizeof( float ), stream ) );
                    CHECK( cudaMemsetAsync( output, 0xFF, reference_output.size() * 2, stream ) );
                    runArm( arm, banks[ layer % kBanks ], input, window( 0, layer ), weights, gated, output, rows, stream );
                    CHECK( cudaStreamSynchronize( stream ) );
                    CHECK( cudaGetLastError() );

                    auto& g = arm == Arm::RowsApart ? reference_gated : arm_gated;
                    auto& o = arm == Arm::RowsApart ? reference_output : arm_output;

                    CHECK( cudaMemcpy( g.data(), gated, g.size() * sizeof( float ), cudaMemcpyDeviceToHost ) );
                    CHECK( cudaMemcpy( o.data(), output, o.size() * 2, cudaMemcpyDeviceToHost ) );

                    if ( arm != Arm::RowsApart
                         && ( std::memcmp( g.data(), reference_gated.data(), g.size() * sizeof( float ) ) != 0
                              || std::memcmp( o.data(), reference_output.data(), o.size() * 2 ) != 0 ) )
                    {
                        printf( "MISMATCH: %s differs from rows-apart at R = %d, layer %d\n", armName( arm ), rows, layer );
                        return 1;
                    }
                }
            }

            double milliseconds[ 4 ];

            for ( int a = 0; a < 4; ++a )
            {
                std::vector<double> trials;

                for ( int trial = 0; trial < kTrials; ++trial )
                {
                    CHECK( cudaEventRecord( start, stream ) );

                    for ( int w = 0; w < windows; ++w )
                    {
                        for ( int layer = 0; layer < kLayers; ++layer )
                        {
                            runArm( arms[ a ], banks[ ( w * kLayers + layer ) % kBanks ], input, window( w, layer ),
                                weights, gated, output, rows, stream );
                        }
                    }

                    CHECK( cudaEventRecord( stop, stream ) );
                    CHECK( cudaEventSynchronize( stop ) );

                    float elapsed = 0.0f;
                    CHECK( cudaEventElapsedTime( &elapsed, start, stop ) );
                    trials.push_back( elapsed / windows );  // one token's 30 layers
                }

                std::sort( trials.begin(), trials.end() );
                milliseconds[ a ] = trials[ kTrials / 2 ];
            }

            if ( rows == 1 )
                one_row_ms = milliseconds[ 0 ];

            printf( "| %d | %.2f | %.3f ms (%.2f) | %.2f | %.2f | %.2f |\n", rows, union_per_layer / kTopK,
                milliseconds[ 0 ], milliseconds[ 0 ] / one_row_ms, milliseconds[ 1 ] / one_row_ms,
                milliseconds[ 2 ] / one_row_ms, milliseconds[ 3 ] / one_row_ms );

            CHECK( cudaFree( indices ) );
        }

        printf( "\none row: %.3f ms for 30 layers' banks, %.0f GB/s over its 8 experts a layer (%.2f MB each)\n",
            one_row_ms, kLayers * kTopK * expert_bytes / ( one_row_ms * 1e6 ), expert_bytes / 1e6 );
    }

    return 0;
}
