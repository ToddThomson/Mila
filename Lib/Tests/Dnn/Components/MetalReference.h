/**
 * @file MetalReference.h
 * @brief What Metal's operation tests share: weights as a model hands them, and the distance to a double-precision reference.
 *
 * Included after `import std;` and `import Mila;`, so it names their types and includes nothing.
 */

#pragma once

namespace Mila::Tests::Dnn::MetalReference
{
    using Mila::Dnn::TensorDataType;
    using Mila::Dnn::Serialization::TensorBlobView;
    using Mila::Dnn::Serialization::TensorMetadata;
    using Mila::Dnn::shape_t;

    inline std::uint16_t toBf16Bits( float value )
    {
        std::uint32_t bits = std::bit_cast<std::uint32_t>( value );
        bits += 0x7FFFu + ( ( bits >> 16 ) & 1u );

        return static_cast<std::uint16_t>( bits >> 16 );
    }

    inline float fromBf16Bits( std::uint16_t bits )
    {
        return std::bit_cast<float>( static_cast<std::uint32_t>( bits ) << 16 );
    }

    /// The value one held at this precision really has: FP16 holds a normal BF16 value exactly, and rounds below its normal range.
    template<TensorDataType TPrecision>
    inline double asStored( float value )
    {
        if constexpr ( TPrecision == TensorDataType::FP16 )
        {
            return static_cast<double>( static_cast<float>( static_cast<__fp16>( value ) ) );
        } else {
            return static_cast<double>( fromBf16Bits( toBf16Bits( value ) ) );
        }
    }

    inline std::vector<std::uint16_t> randomBf16( std::size_t count, std::uint32_t seed, float sigma )
    {
        std::mt19937 engine( seed );
        std::normal_distribution<float> normal( 0.0f, sigma );

        std::vector<std::uint16_t> bits( count );

        for ( std::uint16_t& element : bits )
        {
            element = toBf16Bits( normal( engine ) );
        }

        return bits;
    }

    inline TensorBlobView blobOver( std::span<const std::uint16_t> bits, const shape_t& shape, TensorDataType data_type = TensorDataType::BF16 )
    {
        TensorMetadata metadata;
        metadata.dtype = data_type;
        metadata.shape = shape;
        metadata.total_bytes = bits.size_bytes();

        return TensorBlobView( std::move( metadata ), bits.data(), bits.size_bytes() );
    }

    inline double relativeError( std::span<const float> actual, std::span<const double> reference )
    {
        double difference = 0.0;
        double norm = 0.0;

        for ( std::size_t i = 0; i < reference.size(); ++i )
        {
            const double delta = static_cast<double>( actual[i] ) - reference[i];

            difference += delta * delta;
            norm += reference[i] * reference[i];
        }

        return norm == 0.0 ? std::sqrt( difference ) : std::sqrt( difference / norm );
    }

    /// The gate's bars: about twice one rounding of the output, which Phase 2 measured at 2.1e-4 and 1.7e-3.
    template<TensorDataType TPrecision>
    inline constexpr double kBar = TPrecision == TensorDataType::FP16 ? 5e-4 : 4e-3;

    template<TensorDataType TPrecision>
    using Precision = std::integral_constant<TensorDataType, TPrecision>;

    using Precisions = ::testing::Types<Precision<TensorDataType::FP16>, Precision<TensorDataType::BF16>>;

    struct PrecisionNames
    {
        template<typename T>
        static std::string GetName( int )
        {
            return T::value == TensorDataType::FP16 ? "Fp16" : "Bf16";
        }
    };

    template<TensorDataType TPrecision>
    inline Mila::Dnn::Tensor<TPrecision, Mila::Dnn::Compute::MetalMemoryResource> metalTensor( std::span<const float> values, const shape_t& shape )
    {
        Mila::Dnn::Tensor<TPrecision, Mila::Dnn::Compute::MetalMemoryResource> tensor( Mila::Dnn::Compute::Device::Metal( 0 ), shape );
        fill( tensor, values );

        return tensor;
    }

    template<TensorDataType TPrecision>
    inline std::vector<float> readFloats( const Mila::Dnn::Tensor<TPrecision, Mila::Dnn::Compute::MetalMemoryResource>& tensor )
    {
        const auto host = toHost<TensorDataType::FP32>( tensor );

        return std::vector<float>( host.data(), host.data() + host.size() );
    }

    /// Normal values as BF16 would have stored them, read as floats.
    inline std::vector<float> randomValues( std::size_t count, std::uint32_t seed, float sigma )
    {
        const std::vector<std::uint16_t> bits = randomBf16( count, seed, sigma );

        std::vector<float> values( bits.size() );
        std::ranges::transform( bits, values.begin(), fromBf16Bits );

        return values;
    }

    /// A value once it is in a tensor of this precision, for a reference to compute on what the kernel really read.
    template<TensorDataType TPrecision>
    inline std::vector<double> asStored( std::span<const float> values )
    {
        std::vector<double> stored( values.size() );
        std::ranges::transform( values, stored.begin(), []( float value ) { return asStored<TPrecision>( value ); } );

        return stored;
    }
}
