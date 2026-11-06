/**
 * @file QuantizedTiming.cpp
 * @brief Times Linear's operation on FP8 or FP4 weights on Metal, at a number of rows, as a network encodes it.
 *
 * The rows a path starts from are the operation's constants, so a crossing is timed by building this against each path in
 * turn. Notebooks/MetalQuantizedWeights.md holds the results.
 */

import std;
import Mila;
import Compute.MetalExecutionContext;
import Compute.MetalLinearOp;
import Dnn.Components.LinearConfig;
import Serialization.Tensor;

namespace
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Quant::Weight;

    constexpr int kWarmRounds = 3;

    std::uint16_t toBf16Bits( float value )
    {
        std::uint32_t bits = std::bit_cast<std::uint32_t>( value );
        bits += 0x7FFFu + ( ( bits >> 16 ) & 1u );

        return static_cast<std::uint16_t>( bits >> 16 );
    }

    template<TensorDataType TPrecision, typename TPolicy>
    int run( std::string_view label, int rows, int outputs, int inputs, int products, int rounds )
    {
        constexpr bool kIsFp4 = HasFp4E2M1Codes<TPolicy>;

        std::mt19937 engine( 2 );
        std::normal_distribution<float> normal( 0.0f, 0.02f );
        std::vector<std::uint16_t> bits( static_cast<std::size_t>( outputs ) * inputs );
        std::ranges::generate( bits, [&]() { return toBf16Bits( normal( engine ) ); } );

        Serialization::TensorMetadata metadata;
        metadata.dtype = TensorDataType::BF16;
        metadata.shape = shape_t{ outputs, inputs };
        metadata.total_bytes = bits.size() * sizeof( std::uint16_t );
        const Serialization::TensorBlobView blob( std::move( metadata ), bits.data(), bits.size() * sizeof( std::uint16_t ) );

        MetalExecutionContext context( Device::Metal( 0 ) );
        LinearConfig config( inputs, outputs );
        config.withBias( false );

        Metal::Linear::MetalLinearOp<TPrecision, TPolicy> op( &context, config );

        Tensor<kIsFp4 ? TensorDataType::UINT8 : TensorDataType::FP8_E4M3, MetalMemoryResource> codes(
            Device::Metal( 0 ), kIsFp4 ? shape_t{ outputs, inputs / 2 } : shape_t{ outputs, inputs } );
        const shape_t scale_shape = [&] {
            if constexpr ( kIsFp4 )
            {
                return shape_t{ outputs, inputs / TPolicy::kQuantizationGroupSize };
            } else {
                return shape_t{ outputs };
            }
        }();

        Tensor<TensorDataType::FP32, MetalMemoryResource> scales( Device::Metal( 0 ), scale_shape );

        op.setParameters( &codes, nullptr );
        op.setWeightScales( &scales );
        op.build( BuildContext( shape_t{ rows, inputs }, RuntimeMode::Inference, false ) );
        op.quantize( blob, codes, scales, shape_t{ outputs, inputs } );

        std::vector<float> input_values( static_cast<std::size_t>( rows ) * inputs );
        std::mt19937 input_engine( 1 );
        std::normal_distribution<float> unit( 0.0f, 1.0f );
        std::ranges::generate( input_values, [&]() { return unit( input_engine ); } );

        Tensor<TPrecision, MetalMemoryResource> input( Device::Metal( 0 ), shape_t{ rows, inputs } );
        Tensor<TPrecision, MetalMemoryResource> output( Device::Metal( 0 ), shape_t{ rows, outputs } );
        fill( input, std::span<const float>( input_values ) );

        std::vector<double> times;

        for ( int round = 0; round < kWarmRounds + rounds; ++round )
        {
            const auto start = std::chrono::steady_clock::now();

            for ( int i = 0; i < products; ++i )
            {
                op.forward( input, output );
            }

            context.synchronize();

            const std::chrono::duration<double, std::milli> elapsed = std::chrono::steady_clock::now() - start;

            if ( round >= kWarmRounds )
            {
                times.push_back( elapsed.count() / products );
            }
        }

        std::ranges::sort( times );

        std::println( "R {} M={} N={} K={} {:.4f} ms", label, rows, outputs, inputs, times[times.size() / 2] );

        return 0;
    }
}

/**
 * Usage: mila-metal-quantized-timing FP16|BF16 fp8|fp4 rows outputs inputs products rounds
 * Each result line starts with R. `products` are encoded and run together, as a network's layers are.
 */
int main( int argument_count, char** arguments )
{
    if ( argument_count != 8 )
    {
        std::println( std::cerr, "usage: mila-metal-quantized-timing FP16|BF16 fp8|fp4 rows outputs inputs products rounds" );

        return 2;
    }

    const std::string precision = arguments[1];
    const std::string format = arguments[2];
    const int rows = std::stoi( arguments[3] );
    const int outputs = std::stoi( arguments[4] );
    const int inputs = std::stoi( arguments[5] );
    const int products = std::stoi( arguments[6] );
    const int rounds = std::stoi( arguments[7] );
    const std::string label = precision + " " + format;

    try
    {
        Mila::initialize();

        if ( precision == "BF16" )
        {
            return format == "fp4" ? run<TensorDataType::BF16, PerGroupFp4<128>>( label, rows, outputs, inputs, products, rounds )
                : run<TensorDataType::BF16, PerChannelFp8<>>( label, rows, outputs, inputs, products, rounds );
        }

        return format == "fp4" ? run<TensorDataType::FP16, PerGroupFp4<128>>( label, rows, outputs, inputs, products, rounds )
            : run<TensorDataType::FP16, PerChannelFp8<>>( label, rows, outputs, inputs, products, rounds );
    }
    catch ( const std::exception& error )
    {
        std::println( std::cerr, "failed: {}", error.what() );

        return 1;
    }
}
