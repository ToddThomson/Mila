// What a weight's or a cache's codes mean, for every kernel that reads or writes them: FP8 E4M3 and FP4 E2M1, as
// Lib/Src/Dnn/Quantization lays them out. Compiled ahead of every operation's kernels, which name these.

#include <metal_stdlib>

using namespace metal;

// FP8 E4M3: 3 mantissa bits, bias 7, largest 448. The top code of each sign is not a number, and reads as zero here: no
// code Mila writes is one.
constant float kFp8E4M3Values[256] = {
    0.0f, 0.001953125f, 0.00390625f, 0.005859375f, 0.0078125f, 0.009765625f, 0.01171875f, 0.013671875f,
    0.015625f, 0.017578125f, 0.01953125f, 0.021484375f, 0.0234375f, 0.025390625f, 0.02734375f, 0.029296875f,
    0.03125f, 0.03515625f, 0.0390625f, 0.04296875f, 0.046875f, 0.05078125f, 0.0546875f, 0.05859375f,
    0.0625f, 0.0703125f, 0.078125f, 0.0859375f, 0.09375f, 0.1015625f, 0.109375f, 0.1171875f,
    0.125f, 0.140625f, 0.15625f, 0.171875f, 0.1875f, 0.203125f, 0.21875f, 0.234375f,
    0.25f, 0.28125f, 0.3125f, 0.34375f, 0.375f, 0.40625f, 0.4375f, 0.46875f,
    0.5f, 0.5625f, 0.625f, 0.6875f, 0.75f, 0.8125f, 0.875f, 0.9375f,
    1.0f, 1.125f, 1.25f, 1.375f, 1.5f, 1.625f, 1.75f, 1.875f,
    2.0f, 2.25f, 2.5f, 2.75f, 3.0f, 3.25f, 3.5f, 3.75f,
    4.0f, 4.5f, 5.0f, 5.5f, 6.0f, 6.5f, 7.0f, 7.5f,
    8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
    16.0f, 18.0f, 20.0f, 22.0f, 24.0f, 26.0f, 28.0f, 30.0f,
    32.0f, 36.0f, 40.0f, 44.0f, 48.0f, 52.0f, 56.0f, 60.0f,
    64.0f, 72.0f, 80.0f, 88.0f, 96.0f, 104.0f, 112.0f, 120.0f,
    128.0f, 144.0f, 160.0f, 176.0f, 192.0f, 208.0f, 224.0f, 240.0f,
    256.0f, 288.0f, 320.0f, 352.0f, 384.0f, 416.0f, 448.0f, 0.0f,
    -0.0f, -0.001953125f, -0.00390625f, -0.005859375f, -0.0078125f, -0.009765625f, -0.01171875f, -0.013671875f,
    -0.015625f, -0.017578125f, -0.01953125f, -0.021484375f, -0.0234375f, -0.025390625f, -0.02734375f, -0.029296875f,
    -0.03125f, -0.03515625f, -0.0390625f, -0.04296875f, -0.046875f, -0.05078125f, -0.0546875f, -0.05859375f,
    -0.0625f, -0.0703125f, -0.078125f, -0.0859375f, -0.09375f, -0.1015625f, -0.109375f, -0.1171875f,
    -0.125f, -0.140625f, -0.15625f, -0.171875f, -0.1875f, -0.203125f, -0.21875f, -0.234375f,
    -0.25f, -0.28125f, -0.3125f, -0.34375f, -0.375f, -0.40625f, -0.4375f, -0.46875f,
    -0.5f, -0.5625f, -0.625f, -0.6875f, -0.75f, -0.8125f, -0.875f, -0.9375f,
    -1.0f, -1.125f, -1.25f, -1.375f, -1.5f, -1.625f, -1.75f, -1.875f,
    -2.0f, -2.25f, -2.5f, -2.75f, -3.0f, -3.25f, -3.5f, -3.75f,
    -4.0f, -4.5f, -5.0f, -5.5f, -6.0f, -6.5f, -7.0f, -7.5f,
    -8.0f, -9.0f, -10.0f, -11.0f, -12.0f, -13.0f, -14.0f, -15.0f,
    -16.0f, -18.0f, -20.0f, -22.0f, -24.0f, -26.0f, -28.0f, -30.0f,
    -32.0f, -36.0f, -40.0f, -44.0f, -48.0f, -52.0f, -56.0f, -60.0f,
    -64.0f, -72.0f, -80.0f, -88.0f, -96.0f, -104.0f, -112.0f, -120.0f,
    -128.0f, -144.0f, -160.0f, -176.0f, -192.0f, -208.0f, -224.0f, -240.0f,
    -256.0f, -288.0f, -320.0f, -352.0f, -384.0f, -416.0f, -448.0f, -0.0f,
};

// FP4 E2M1, sign and magnitude: bit 3 is the sign, and bits 0-2 index {0, 0.5, 1, 1.5, 2, 3, 4, 6}.
constant float kFp4E2M1Values[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

constant float kFp8E4M3Largest = 448.0f;
constant float kFp4E2M1Largest = 6.0f;

// A BF16 value's exponent bits all set: an infinity or not a number. Read from the bits, which fast math cannot assume away.
static inline bool bf16IsNotFinite( ushort bits )
{
    return ( bits & 0x7F80u ) == 0x7F80u;
}

static inline float bf16Value( ushort bits )
{
    return as_type<float>( uint( bits ) << 16 );
}

// The E4M3 code nearest a finite value, ties to even, and the largest finite code past 448: CUDA's __nv_fp8_e4m3 with
// saturation, which Mila's CUDA quantize writes, so that the two devices write the same bytes.
static inline uchar fp8E4M3Code( float value )
{
    const uint bits = as_type<uint>( value );
    const uchar sign = uchar( ( bits >> 24 ) & 0x80u );
    const float magnitude = as_type<float>( bits & 0x7FFFFFFFu );

    if ( magnitude > kFp8E4M3Largest )
    {
        return sign | uchar( 0x7E );
    }

    // Below the smallest normal, 2^-6, codes are multiples of 2^-9, and scaling by 512 is exact.
    if ( magnitude < 0.015625f )
    {
        return sign | uchar( rint( magnitude * 512.0f ) );
    }

    const uint exponent = ( ( bits >> 23 ) & 0xFFu ) - 127u + 7u;
    const uint mantissa = bits & 0x7FFFFFu;
    const uint kept = mantissa >> 20;
    const uint rest = mantissa & 0xFFFFFu;
    const uint rounds_up = ( rest > 0x80000u || ( rest == 0x80000u && ( kept & 1u ) != 0u ) ) ? 1u : 0u;

    // A carry out of the mantissa steps the exponent, which the sum already does.
    const uint code = min( ( exponent << 3 ) + kept + rounds_up, 0x7Eu );

    return sign | uchar( code );
}

// The E2M1 code nearest a value, at the midpoints between codes that Mila's CUDA quantize uses.
static inline uchar fp4E2M1Code( float value )
{
    const uchar sign = value < 0.0f ? uchar( 8 ) : uchar( 0 );
    const float magnitude = fabs( value );

    uchar code = 7;

    if ( magnitude < 0.25f ) { code = 0; }
    else if ( magnitude < 0.75f ) { code = 1; }
    else if ( magnitude < 1.25f ) { code = 2; }
    else if ( magnitude < 1.75f ) { code = 3; }
    else if ( magnitude < 2.5f ) { code = 4; }
    else if ( magnitude < 3.5f ) { code = 5; }
    else if ( magnitude < 5.0f ) { code = 6; }

    return sign | code;
}
