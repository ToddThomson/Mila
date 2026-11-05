/**
 * @file ProductTiming.cpp
 * @brief Times a Metal matrix product through the library's own function, padded and not, with its copies and encoder changes.
 *
 * mila-metal-matrix-timing times the Metal calls alone; this one answers what padding costs as Linear would really run it.
 * Notebooks/MetalLlamaOperations.md holds the results.
 */

import std;
import Mila;
import Compute.MetalExecutionContext;
import Compute.MetalMatrixProduct;

namespace
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Compute::Metal;

    constexpr int kWarmRounds = 3;

    template<TensorDataType TPrecision>
    using MetalTensor = Tensor<TPrecision, MetalMemoryResource>;

    std::vector<float> normalValues( std::size_t count, std::uint32_t seed, float sigma )
    {
        std::mt19937 engine( seed );
        std::normal_distribution<float> normal( 0.0f, sigma );

        std::vector<float> values( count );
        std::ranges::generate( values, [&]() { return normal( engine ); } );

        return values;
    }

    template<TensorDataType TPrecision>
    MetalTensor<TPrecision> filled( std::span<const float> values, const shape_t& shape )
    {
        MetalTensor<TPrecision> tensor( Device::Metal( 0 ), shape );
        fill( tensor, values );

        return tensor;
    }

    /// The middle of `rounds` rounds, in milliseconds per product, of `products` encoded and run together as a network's layers are.
    template<TensorDataType TPrecision>
    double millisecondsPerProduct( MetalExecutionContext& context, const MatrixProduct& product, int products, int rounds )
    {
        std::vector<double> times;
        times.reserve( static_cast<std::size_t>( rounds ) );

        for ( int round = 0; round < kWarmRounds + rounds; ++round )
        {
            const auto start = std::chrono::steady_clock::now();

            for ( int i = 0; i < products; ++i )
            {
                encodeMatrixProduct<TPrecision>( context, product );
            }

            context.synchronize();

            const std::chrono::duration<double, std::milli> elapsed = std::chrono::steady_clock::now() - start;

            if ( round >= kWarmRounds )
            {
                times.push_back( elapsed.count() / products );
            }
        }

        std::ranges::sort( times );

        return times[times.size() / 2];
    }

    template<TensorDataType TPrecision>
    int run( std::string_view precision, int rows, int outputs, int inputs, int products, int rounds )
    {
        const std::vector<float> input_values = normalValues( static_cast<std::size_t>( rows ) * inputs, 1, 1.0f );
        const std::vector<float> weight_values = normalValues( static_cast<std::size_t>( outputs ) * inputs, 2, 0.02f );

        auto input = filled<TPrecision>( input_values, shape_t{ rows, inputs } );
        auto weight = filled<TPrecision>( weight_values, shape_t{ outputs, inputs } );
        auto output = filled<TPrecision>( std::vector<float>( static_cast<std::size_t>( rows ) * outputs, 0.0f ), shape_t{ rows, outputs } );

        MetalExecutionContext context( Device::Metal( 0 ) );

        const MatrixProduct own{
            static_cast<const ITensor&>( input ).rawData(), static_cast<const ITensor&>( weight ).rawData(),
            static_cast<const ITensor&>( output ).rawData(), rows, inputs, outputs, outputs, kNeverPad };

        MatrixProduct padded = own;
        padded.fewest_multiplies_to_pad = 0;

        const double own_milliseconds = millisecondsPerProduct<TPrecision>( context, own, products, rounds );
        const auto own_output = toHost<TensorDataType::FP32>( output );

        const double padded_milliseconds = millisecondsPerProduct<TPrecision>( context, padded, products, rounds );
        const auto padded_output = toHost<TensorDataType::FP32>( output );

        double norm = 0.0;
        double difference = 0.0;

        for ( std::size_t i = 0; i < static_cast<std::size_t>( rows ) * outputs; ++i )
        {
            const double a = own_output.data()[i];
            const double b = padded_output.data()[i];

            norm += a * a;
            difference += ( a - b ) * ( a - b );
        }

        std::println( "R {} M={} K={} N={} own {:.4f} padded {:.4f} ms ratio {:.2f} difference {:.2e}",
            precision, rows, outputs, inputs, own_milliseconds, padded_milliseconds, own_milliseconds / padded_milliseconds,
            std::sqrt( difference / norm ) );

        return 0;
    }
}

int main( int argument_count, char** arguments )
{
    if ( argument_count != 7 )
    {
        std::println( std::cerr, "usage: mila-metal-product-timing FP16|BF16 rows outputs inputs products rounds" );

        return 2;
    }

    const std::string_view precision = arguments[1];
    const int rows = std::stoi( arguments[2] );
    const int outputs = std::stoi( arguments[3] );
    const int inputs = std::stoi( arguments[4] );
    const int products = std::stoi( arguments[5] );
    const int rounds = std::stoi( arguments[6] );

    try
    {
        Mila::initialize();

        return precision == "BF16"
            ? run<TensorDataType::BF16>( precision, rows, outputs, inputs, products, rounds )
            : run<TensorDataType::FP16>( precision, rows, outputs, inputs, products, rounds );
    }
    catch ( const std::exception& error )
    {
        std::println( std::cerr, "failed: {}", error.what() );

        return 1;
    }
}
