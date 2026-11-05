/**
 * @file Metal.TensorOps.Metal.cpp
 * @brief Tensor fill, copy and zero on Metal, each checked by the values read back.
 *
 * References come from the formats' definitions: FP16 has 11 significand bits and BF16 has 8, so 1 + 2^-10 is exact in
 * FP16 and rounds to 1 in BF16, and 70000 overflows FP16 and BF16 rounds it to 70144.
 */

#include <gtest/gtest.h>

import Mila;
import std;
import Compute.CpuExecutionContext;
import Compute.ExecutionContext;
import Compute.MetalExecutionContext;

namespace Mila::Tests::Dnn::Tensors
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace
    {
        constexpr float kExactInFp16 = 1.0f + 1.0f / 1024.0f;
        constexpr dim_t kLarge = 100'000;

        template<TensorDataType TDataType>
        using MetalTensor = Tensor<TDataType, MetalMemoryResource>;

        using HostFloats = Tensor<TensorDataType::FP32, CpuMemoryResource>;
        using HostIntegers = Tensor<TensorDataType::INT32, CpuMemoryResource>;

        template<TensorDataType TDataType>
        std::vector<float> readFloats( const MetalTensor<TDataType>& tensor, IExecutionContext* context = nullptr )
        {
            const HostFloats host = toHost<TensorDataType::FP32>( tensor, context );

            return std::vector<float>( host.data(), host.data() + host.size() );
        }

        template<TensorDataType TDataType>
        std::vector<std::int32_t> readIntegers( const MetalTensor<TDataType>& tensor )
        {
            HostIntegers host( Device::Cpu(), tensor.shape() );
            copy( tensor, host );

            return std::vector<std::int32_t>( host.data(), host.data() + host.size() );
        }

        // Not callable on a Metal tensor: a host read has to wait for the GPU, and these would neither wait nor know
        // FP16's element. A parameterized concept, so that a constrained-away member is false rather than an error.
        template<typename TTensor>
        concept HasHostAccess = requires( TTensor tensor ) { tensor.data(); };
    }

    // ====================================================================
    // What the types allow
    // ====================================================================

    TEST( MetalTensorTypes, AMetalTensorHasNoHostAccessors )
    {
        EXPECT_FALSE( HasHostAccess<MetalTensor<TensorDataType::FP16>> );
        EXPECT_FALSE( HasHostAccess<MetalTensor<TensorDataType::FP32>> );
        EXPECT_TRUE( HasHostAccess<HostFloats> );
    }

    TEST( MetalTensorTypes, AModelMayComputeInFp16OrBf16AndNotFp32 )
    {
        EXPECT_TRUE( ( PrecisionSupportedOnDevice<TensorDataType::FP16, DeviceType::Metal> ) );
        EXPECT_TRUE( ( PrecisionSupportedOnDevice<TensorDataType::BF16, DeviceType::Metal> ) );
        EXPECT_FALSE( ( PrecisionSupportedOnDevice<TensorDataType::FP32, DeviceType::Metal> ) );
    }

    TEST( MetalTensorTypes, ATensorOnAnotherDeviceIsRefused )
    {
        EXPECT_THROW( MetalTensor<TensorDataType::FP16> tensor( Device::Cpu(), { 4 } ), std::runtime_error );
    }

    // ====================================================================
    // Scalar fill: the kernel
    // ====================================================================

    template<typename TParameter>
    class MetalScalarFillOfFloats : public testing::Test
    {
    };

    template<TensorDataType TDataType>
    struct OfType
    {
        static constexpr TensorDataType value = TDataType;
    };

    using FloatTypes = testing::Types<OfType<TensorDataType::FP32>, OfType<TensorDataType::FP16>, OfType<TensorDataType::BF16>>;
    TYPED_TEST_SUITE( MetalScalarFillOfFloats, FloatTypes );

    TYPED_TEST( MetalScalarFillOfFloats, WritesEveryElement )
    {
        MetalTensor<TypeParam::value> tensor( Device::Metal( 0 ), { kLarge } );

        fill( tensor, 2.5f );

        const std::vector<float> values = readFloats( tensor );

        ASSERT_EQ( values.size(), static_cast<std::size_t>( kLarge ) );
        EXPECT_EQ( std::ranges::count( values, 2.5f ), kLarge );
    }

    TEST( MetalScalarFill, Fp16HoldsAValueBf16RoundsAway )
    {
        MetalTensor<TensorDataType::FP16> half_tensor( Device::Metal( 0 ), { 8 } );
        MetalTensor<TensorDataType::BF16> bfloat_tensor( Device::Metal( 0 ), { 8 } );

        fill( half_tensor, kExactInFp16 );
        fill( bfloat_tensor, kExactInFp16 );

        EXPECT_EQ( readFloats( half_tensor )[7], kExactInFp16 );
        EXPECT_EQ( readFloats( bfloat_tensor )[7], 1.0f );
    }

    TEST( MetalScalarFill, Bf16HoldsARangeFp16Cannot )
    {
        MetalTensor<TensorDataType::FP16> half_tensor( Device::Metal( 0 ), { 8 } );
        MetalTensor<TensorDataType::BF16> bfloat_tensor( Device::Metal( 0 ), { 8 } );

        fill( half_tensor, 70000.0f );
        fill( bfloat_tensor, 70000.0f );

        EXPECT_TRUE( std::isinf( readFloats( half_tensor )[7] ) );
        EXPECT_EQ( readFloats( bfloat_tensor )[7], 70144.0f );
    }

    TEST( MetalScalarFill, EveryIntegerTypeHoldsItsValue )
    {
        MetalTensor<TensorDataType::INT32> int32_tensor( Device::Metal( 0 ), { 16 } );
        MetalTensor<TensorDataType::UINT32> uint32_tensor( Device::Metal( 0 ), { 16 } );
        MetalTensor<TensorDataType::INT16> int16_tensor( Device::Metal( 0 ), { 16 } );
        MetalTensor<TensorDataType::UINT16> uint16_tensor( Device::Metal( 0 ), { 16 } );
        MetalTensor<TensorDataType::INT8> int8_tensor( Device::Metal( 0 ), { 16 } );
        MetalTensor<TensorDataType::UINT8> uint8_tensor( Device::Metal( 0 ), { 16 } );

        fill( int32_tensor, std::int32_t{ -100000 } );
        fill( uint32_tensor, std::int32_t{ 100000 } );
        fill( int16_tensor, std::int32_t{ -30000 } );
        fill( uint16_tensor, std::int32_t{ 60000 } );
        fill( int8_tensor, std::int32_t{ -100 } );
        fill( uint8_tensor, std::int32_t{ 200 } );

        EXPECT_EQ( readIntegers( int32_tensor )[15], -100000 );
        EXPECT_EQ( readIntegers( uint32_tensor )[15], 100000 );
        EXPECT_EQ( readIntegers( int16_tensor )[15], -30000 );
        EXPECT_EQ( readIntegers( uint16_tensor )[15], 60000 );
        EXPECT_EQ( readIntegers( int8_tensor )[15], -100 );
        EXPECT_EQ( readIntegers( uint8_tensor )[15], 200 );
    }

    TEST( MetalScalarFill, WithAContextItRunsWhenSynchronized )
    {
        MetalExecutionContext context( Device::Metal( 0 ) );
        MetalTensor<TensorDataType::FP16> tensor( Device::Metal( 0 ), { 16 } );

        fill( tensor, 1.0f );
        fill( tensor, 4.0f, &context );

        // Without the context the read cannot know of the work, so what it sees is the value before it.
        EXPECT_EQ( readFloats( tensor )[15], 1.0f );

        context.synchronize();

        EXPECT_EQ( readFloats( tensor )[15], 4.0f );
    }

    TEST( MetalScalarFill, AViewFillsOnlyItsElements )
    {
        MetalTensor<TensorDataType::FP16> tensor( Device::Metal( 0 ), { 16 } );
        fill( tensor, 1.0f );

        auto window = tensor.view( { 4 }, 8 );
        fill( window, 9.0f );

        const std::vector<float> values = readFloats( tensor );

        EXPECT_EQ( values[7], 1.0f );
        EXPECT_EQ( values[8], 9.0f );
        EXPECT_EQ( values[11], 9.0f );
        EXPECT_EQ( values[12], 1.0f );
    }

    TEST( MetalScalarFill, AnEmptyTensorIsNothingToDo )
    {
        MetalTensor<TensorDataType::FP16> tensor( Device::Metal( 0 ), { 0 } );

        EXPECT_NO_THROW( fill( tensor, 1.0f ) );
    }

    TEST( MetalScalarFill, AContextThatIsNotMetalIsRefused )
    {
        MetalTensor<TensorDataType::FP16> tensor( Device::Metal( 0 ), { 4 } );
        CpuExecutionContext context;

        EXPECT_THROW( fill( tensor, 1.0f, &context ), std::invalid_argument );
    }

    // ====================================================================
    // Host values
    // ====================================================================

    TEST( MetalValuesFill, EachElementHoldsItsValue )
    {
        const std::vector<float> values{ 0.5f, -1.25f, 3.0f, kExactInFp16 };

        MetalTensor<TensorDataType::FP32> float_tensor( Device::Metal( 0 ), { 4 } );
        MetalTensor<TensorDataType::FP16> half_tensor( Device::Metal( 0 ), { 4 } );
        MetalTensor<TensorDataType::BF16> bfloat_tensor( Device::Metal( 0 ), { 4 } );

        fill( float_tensor, std::span<const float>( values ) );
        fill( half_tensor, std::span<const float>( values ) );
        fill( bfloat_tensor, std::span<const float>( values ) );

        EXPECT_EQ( readFloats( float_tensor ), values );
        EXPECT_EQ( readFloats( half_tensor ), values );
        EXPECT_EQ( readFloats( bfloat_tensor ), ( std::vector<float>{ 0.5f, -1.25f, 3.0f, 1.0f } ) );
    }

    TEST( MetalValuesFill, RoundsToEvenAtTheFormatsHalfway )
    {
        // Halfway between BF16's neighbours of 1: 1 + 2^-8 ties to 1 (even), 1 + 3 * 2^-8 ties to 1 + 2^-6 (even).
        const std::vector<float> ties{ 1.0f + 1.0f / 256.0f, 1.0f + 3.0f / 256.0f };

        MetalTensor<TensorDataType::BF16> tensor( Device::Metal( 0 ), { 2 } );
        fill( tensor, std::span<const float>( ties ) );

        EXPECT_EQ( readFloats( tensor ), ( std::vector<float>{ 1.0f, 1.0f + 1.0f / 64.0f } ) );
    }

    TEST( MetalValuesFill, FillsTheShorterOfTheTensorAndTheValues )
    {
        const std::vector<float> values{ 2.0f, 3.0f };

        MetalTensor<TensorDataType::FP16> tensor( Device::Metal( 0 ), { 4 } );
        fill( tensor, 1.0f );
        fill( tensor, std::span<const float>( values ) );

        EXPECT_EQ( readFloats( tensor ), ( std::vector<float>{ 2.0f, 3.0f, 1.0f, 1.0f } ) );
    }

    TEST( MetalValuesFill, RunsAfterTheContextsEarlierWork )
    {
        MetalExecutionContext context( Device::Metal( 0 ) );
        MetalTensor<TensorDataType::FP16> tensor( Device::Metal( 0 ), { 4 } );

        const std::vector<float> values{ 5.0f, 5.0f, 5.0f, 5.0f };

        fill( tensor, 1.0f, &context );
        fill( tensor, std::span<const float>( values ), &context );

        // Had the kernel run after the host write, it would have overwritten these with 1.
        EXPECT_EQ( readFloats( tensor, &context ), values );
    }

    // ====================================================================
    // Copy
    // ====================================================================

    TEST( MetalCopy, BetweenTensorsOfOneTypeKeepsEveryElement )
    {
        MetalTensor<TensorDataType::FP16> source( Device::Metal( 0 ), { kLarge } );
        MetalTensor<TensorDataType::FP16> destination( Device::Metal( 0 ), { kLarge } );

        fill( source, kExactInFp16 );
        copy( source, destination );

        EXPECT_EQ( std::ranges::count( readFloats( destination ), kExactInFp16 ), kLarge );
    }

    TEST( MetalCopy, RunsInOrderWithTheContextsOtherWork )
    {
        MetalExecutionContext context( Device::Metal( 0 ) );
        MetalTensor<TensorDataType::FP16> source( Device::Metal( 0 ), { 16 } );
        MetalTensor<TensorDataType::FP16> destination( Device::Metal( 0 ), { 16 } );

        fill( source, 1.0f, &context );
        copy( source, destination, &context );
        fill( source, 2.0f, &context );
        context.synchronize();

        EXPECT_EQ( readFloats( destination )[15], 1.0f );
        EXPECT_EQ( readFloats( source )[15], 2.0f );
    }

    TEST( MetalCopy, ReadingBackWaitsForTheContextsWork )
    {
        MetalExecutionContext context( Device::Metal( 0 ) );
        MetalTensor<TensorDataType::FP16> tensor( Device::Metal( 0 ), { kLarge } );

        fill( tensor, 3.0f, &context );

        EXPECT_EQ( std::ranges::count( readFloats( tensor, &context ), 3.0f ), kLarge );
    }

    TEST( MetalCopy, BetweenFormatsConvertsEachElement )
    {
        MetalTensor<TensorDataType::FP16> half_tensor( Device::Metal( 0 ), { 4 } );
        MetalTensor<TensorDataType::BF16> bfloat_tensor( Device::Metal( 0 ), { 4 } );

        fill( half_tensor, kExactInFp16 );
        copy( half_tensor, bfloat_tensor );

        EXPECT_EQ( readFloats( bfloat_tensor )[3], 1.0f );

        fill( bfloat_tensor, 70000.0f );
        copy( bfloat_tensor, half_tensor );

        EXPECT_TRUE( std::isinf( readFloats( half_tensor )[3] ) );
    }

    TEST( MetalCopy, FromTheHostConvertsToTheElement )
    {
        HostFloats host( Device::Cpu(), { 4 } );
        const std::vector<float> values{ 0.5f, -1.25f, 3.0f, kExactInFp16 };
        std::ranges::copy( values, host.data() );

        MetalTensor<TensorDataType::FP16> half_tensor( Device::Metal( 0 ), { 4 } );
        MetalTensor<TensorDataType::BF16> bfloat_tensor( Device::Metal( 0 ), { 4 } );
        MetalTensor<TensorDataType::FP32> float_tensor( Device::Metal( 0 ), { 4 } );

        copy( host, half_tensor );
        copy( host, bfloat_tensor );
        copy( host, float_tensor );

        EXPECT_EQ( readFloats( half_tensor ), values );
        EXPECT_EQ( readFloats( bfloat_tensor )[3], 1.0f );
        EXPECT_EQ( readFloats( float_tensor ), values );
    }

    TEST( MetalCopy, AViewCopiesOnlyItsElementsOnEveryPath )
    {
        MetalTensor<TensorDataType::FP16> source( Device::Metal( 0 ), { 16 } );
        MetalTensor<TensorDataType::FP16> destination( Device::Metal( 0 ), { 16 } );
        MetalTensor<TensorDataType::BF16> converted( Device::Metal( 0 ), { 16 } );

        fill( source, 1.0f );
        fill( destination, 0.0f );
        fill( converted, 0.0f );

        auto source_window = source.view( { 4 }, 2 );
        auto destination_window = destination.view( { 4 }, 8 );
        auto converted_window = converted.view( { 4 }, 8 );

        fill( source_window, 2.0f );
        copy( source_window, destination_window );
        copy( source_window, converted_window );

        const std::vector<float> expected{ 0.0f, 2.0f, 2.0f, 2.0f, 2.0f, 0.0f };
        const std::vector<float> copied = readFloats( destination );
        const std::vector<float> convert_copied = readFloats( converted );

        EXPECT_EQ( std::vector<float>( copied.begin() + 7, copied.begin() + 13 ), expected );
        EXPECT_EQ( std::vector<float>( convert_copied.begin() + 7, convert_copied.begin() + 13 ), expected );

        HostFloats host( Device::Cpu(), { 4 } );
        copy( source_window, host );

        EXPECT_EQ( host.data()[3], 2.0f );
    }

    TEST( MetalCopy, DifferentShapesAreRefused )
    {
        MetalTensor<TensorDataType::FP16> source( Device::Metal( 0 ), { 4 } );
        MetalTensor<TensorDataType::FP16> destination( Device::Metal( 0 ), { 8 } );

        EXPECT_THROW( copy( source, destination ), std::invalid_argument );
    }

    // ====================================================================
    // Zero
    // ====================================================================

    TEST( MetalZero, WritesZeroToEveryElementAndOnlyItsOwn )
    {
        MetalTensor<TensorDataType::FP16> tensor( Device::Metal( 0 ), { kLarge } );
        fill( tensor, 1.0f );

        zero( tensor );

        EXPECT_EQ( std::ranges::count( readFloats( tensor ), 0.0f ), kLarge );

        fill( tensor, 1.0f );

        auto window = tensor.view( { 4 }, 8 );
        zero( window );

        const std::vector<float> values = readFloats( tensor );

        EXPECT_EQ( values[7], 1.0f );
        EXPECT_EQ( values[8], 0.0f );
        EXPECT_EQ( values[11], 0.0f );
        EXPECT_EQ( values[12], 1.0f );
    }

    TEST( MetalZero, RunsInOrderWithTheContextsOtherWork )
    {
        MetalExecutionContext context( Device::Metal( 0 ) );
        MetalTensor<TensorDataType::FP16> tensor( Device::Metal( 0 ), { 16 } );

        fill( tensor, 1.0f, &context );
        zero( tensor, &context );
        context.synchronize();

        EXPECT_EQ( readFloats( tensor )[15], 0.0f );
    }
}
