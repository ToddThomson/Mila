/**
 * @file DecodeReference.Cuda.cpp
 * @brief Decode logits against a reference dumped by an earlier build, bit for bit: DecodeGraph.md gate A1.
 *
 * A gate run by hand, not in CI: the reference is exact only on the card and build that wrote it, and the package
 * arms need the published Q4_0 weights. Write with MILA_DECODE_REFERENCE=write, compare without it.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

import Mila;

#include "Common/DecodeHarness.h"
#include "Measurement/LogLikelihoodHarness.h"
#include "Common/TinyDecodeNetworks.h"

namespace Mila::Tests::Dnn::Models
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace fs = std::filesystem;

    namespace
    {
        using LlamaBf16 = LlamaModel<DeviceType::Cuda, TensorDataType::BF16>;
        using GemmaBf16 = GemmaModel<DeviceType::Cuda, TensorDataType::BF16>;

        template<typename TNetwork>
        using Counted = Common::CountedDecodeNetwork<TNetwork>;

        using TinyLlamaBf16 = Counted<LlamaTransformer<DeviceType::Cuda, TensorDataType::BF16>>;
        using TinyLlamaBf16Fp8Kv = Counted<LlamaTransformer<DeviceType::Cuda, TensorDataType::BF16,
            Quant::Weight::NoWeightQuant, Quant::KvCache::PerTokenKvFp8<>>>;
        using TinyLlamaFp32 = Counted<LlamaTransformer<DeviceType::Cuda, TensorDataType::FP32>>;
        using TinyGemmaBf16 = Counted<GemmaTransformer<DeviceType::Cuda, TensorDataType::BF16,
            Quant::Weight::NoWeightQuant, GemmaBf16::GemmaSlidingKvPolicy>>;
        using TinyQwenBf16 = Counted<QwenTransformer<DeviceType::Cuda, TensorDataType::BF16>>;

        using LlamaQ4_0 = Counted<LlamaTransformer<DeviceType::Cuda, TensorDataType::BF16,
            Quant::Weight::PerGroupInt4<32>, Quant::KvCache::NoKvCompression>>;
        using LlamaQ4_0Fp8Kv = Counted<LlamaTransformer<DeviceType::Cuda, TensorDataType::BF16,
            Quant::Weight::PerGroupInt4<32>, Quant::KvCache::PerTokenKvFp8<>>>;
        using GemmaQ4_0 = Counted<GemmaTransformer<DeviceType::Cuda, TensorDataType::BF16,
            Quant::Weight::PerGroupInt4<32>, GemmaBf16::GemmaSlidingKvPolicy>>;

        // The positions span the split count's changes (one split to 64 positions, then one more per 64) and, on
        // the sliding layers, the ring's wrap. Every step is hashed; these are kept whole to show where a hash
        // mismatch lands.
        constexpr dim_t kLastPosition = 8200;
        constexpr dim_t kContextLength = kLastPosition + 8;
        const std::vector<dim_t> kKeptPositions{ 1, 17, 63, 64, 65, 1000, 8200 };

        struct DecodeTrace
        {
            std::vector<std::uint64_t> step_hashes;
            std::vector<dim_t> kept_positions;
            std::vector<std::vector<float>> kept_logits;
        };

        std::uint64_t hashOf( const std::vector<float>& logits )
        {
            std::uint64_t hash = 14695981039346656037ull;
            const auto* bytes = reinterpret_cast<const unsigned char*>( logits.data() );

            for ( std::size_t i = 0; i < logits.size() * sizeof( float ); ++i )
            {
                hash = ( hash ^ bytes[ i ] ) * 1099511628211ull;
            }

            return hash;
        }

        /**
         * Teacher-forced so a difference at one step cannot change the tokens of the next: one prompt token at
         * position 0, then a fixed token at every position from 1 to the last.
         */
        template<typename TNetwork>
        DecodeTrace traceDecode( TNetwork& network, std::int32_t vocabulary )
        {
            const auto tokenAt = [ vocabulary ]( dim_t position ) {
                return static_cast<std::int32_t>( ( 7 + 37 * position ) % vocabulary );
            };

            DecodeTrace trace;
            trace.step_hashes.reserve( static_cast<std::size_t>( kLastPosition ) );

            (void)Measurement::hostLogits( network, network.prefill( Measurement::deviceTokens( network, { tokenAt( 0 ) } ) ) );

            Common::DecodeInput<TNetwork> input( network );

            for ( dim_t position = 1; position <= kLastPosition; ++position )
            {
                std::vector<float> logits = Measurement::hostLogits(
                    network, network.decode( input.set( tokenAt( position ) ), position ) );

                trace.step_hashes.push_back( hashOf( logits ) );

                if ( std::ranges::contains( kKeptPositions, position ) )
                {
                    trace.kept_positions.push_back( position );
                    trace.kept_logits.push_back( std::move( logits ) );
                }
            }

            return trace;
        }

        std::string deviceName()
        {
            cudaDeviceProp properties{};
            cudaGetDeviceProperties( &properties, 0 );

            std::string name = properties.name;
            std::ranges::replace_if( name, []( char c ) { return !std::isalnum( static_cast<unsigned char>( c ) ); }, '_' );

            return name;
        }

        fs::path referencePath( const std::string& arm )
        {
            return fs::path( TEST_DATA_DIR ) / "References" / "DecodeGraph" / std::format( "{}.{}.bin", arm, deviceName() );
        }

        using FilePointer = std::unique_ptr<std::FILE, int ( * )( std::FILE* )>;

        void writeTrace( const fs::path& path, const DecodeTrace& trace )
        {
            fs::create_directories( path.parent_path() );
            FilePointer file( std::fopen( path.string().c_str(), "wb" ), &std::fclose );

            if ( !file )
            {
                throw std::runtime_error( "could not open " + path.string() );
            }

            bool written = true;

            const auto put = [ & ]( const void* data, std::size_t bytes ) {
                written = written && std::fwrite( data, 1, bytes, file.get() ) == bytes;
            };

            const std::uint64_t steps = trace.step_hashes.size();
            const std::uint64_t kept = trace.kept_positions.size();

            put( &steps, sizeof( steps ) );
            put( trace.step_hashes.data(), steps * sizeof( std::uint64_t ) );
            put( &kept, sizeof( kept ) );

            for ( std::size_t i = 0; i < kept; ++i )
            {
                const std::int64_t position = trace.kept_positions[ i ];
                const std::uint64_t size = trace.kept_logits[ i ].size();

                put( &position, sizeof( position ) );
                put( &size, sizeof( size ) );
                put( trace.kept_logits[ i ].data(), size * sizeof( float ) );
            }

            if ( !written )
            {
                throw std::runtime_error( "could not write " + path.string() );
            }
        }

        DecodeTrace readTrace( const fs::path& path )
        {
            FilePointer file( std::fopen( path.string().c_str(), "rb" ), &std::fclose );

            if ( !file )
            {
                throw std::runtime_error( "could not open " + path.string() );
            }

            bool read = true;

            const auto get = [ & ]( void* data, std::size_t bytes ) {
                read = read && std::fread( data, 1, bytes, file.get() ) == bytes;
            };

            DecodeTrace trace;
            std::uint64_t steps = 0;
            get( &steps, sizeof( steps ) );
            trace.step_hashes.resize( steps );
            get( trace.step_hashes.data(), steps * sizeof( std::uint64_t ) );

            std::uint64_t kept = 0;
            get( &kept, sizeof( kept ) );

            for ( std::uint64_t i = 0; i < kept && read; ++i )
            {
                std::int64_t position = 0;
                std::uint64_t size = 0;
                get( &position, sizeof( position ) );
                get( &size, sizeof( size ) );

                std::vector<float> logits( read ? size : 0 );
                get( logits.data(), logits.size() * sizeof( float ) );

                trace.kept_positions.push_back( static_cast<dim_t>( position ) );
                trace.kept_logits.push_back( std::move( logits ) );
            }

            if ( !read )
            {
                throw std::runtime_error( "could not read " + path.string() );
            }

            return trace;
        }

        /// Empty when the traces are equal bit for bit; otherwise where they first part and by how much.
        std::string differenceBetween( const DecodeTrace& expected, const DecodeTrace& actual )
        {
            if ( expected.step_hashes.size() != actual.step_hashes.size() )
            {
                return std::format( "{} steps against {}", actual.step_hashes.size(), expected.step_hashes.size() );
            }

            const auto mismatch = std::ranges::mismatch( expected.step_hashes, actual.step_hashes );

            if ( mismatch.in1 == expected.step_hashes.end() )
            {
                return {};
            }

            std::size_t differing_steps = 0;

            for ( std::size_t i = 0; i < expected.step_hashes.size(); ++i )
            {
                differing_steps += expected.step_hashes[ i ] != actual.step_hashes[ i ] ? 1 : 0;
            }

            std::string report = std::format( "first differing position {}, {} of {} steps differ",
                std::distance( expected.step_hashes.begin(), mismatch.in1 ) + 1, differing_steps,
                expected.step_hashes.size() );

            for ( std::size_t k = 0; k < expected.kept_positions.size() && k < actual.kept_logits.size(); ++k )
            {
                const auto& a = expected.kept_logits[ k ];
                const auto& b = actual.kept_logits[ k ];

                float largest = 0.0f;
                std::size_t differing = 0;

                for ( std::size_t i = 0; i < a.size() && i < b.size(); ++i )
                {
                    if ( std::memcmp( &a[ i ], &b[ i ], sizeof( float ) ) != 0 )
                    {
                        ++differing;
                        largest = std::max( largest, std::abs( a[ i ] - b[ i ] ) );
                    }
                }

                report += std::format( "\n    position {}: {} of {} logits differ, largest {:.3g}",
                    expected.kept_positions[ k ], differing, a.size(), largest );
            }

            return report;
        }

        void describe( const std::string& arm, const DecodeTrace& trace )
        {
            std::string line = std::format( "  {}:", arm );

            for ( std::size_t k = 0; k < trace.kept_positions.size(); ++k )
            {
                const auto& logits = trace.kept_logits[ k ];
                const auto finite = std::ranges::count_if( logits, []( float x ) { return std::isfinite( x ); } );

                line += std::format( " [{}] argmax {}{}", trace.kept_positions[ k ], Measurement::argMax( logits ),
                    finite == static_cast<std::ptrdiff_t>( logits.size() ) ? "" : std::format( " ({} non-finite)",
                        logits.size() - static_cast<std::size_t>( finite ) ) );
            }

            std::cout << line << "\n" << std::flush;
        }

        bool writing()
        {
            const char* mode = std::getenv( "MILA_DECODE_REFERENCE" );

            return mode != nullptr && std::string( mode ) == "write";
        }

        enum class Decoding
        {
            Called,
            Replayed
        };

        /**
         * Writing traces the arm twice from a fresh network and refuses when the two differ: a reference is only
         * worth comparing against if the build that wrote it reproduces itself.
         *
         * A replayed arm compares against the called arm's reference (DecodeGraph.md gate B2) and must still be
         * replaying at the end: a recording the self-check turned off would pass by calling every step.
         */
        template<typename TMakeNetwork>
        void checkAgainstReference(
            const std::string& arm, std::int32_t vocabulary, TMakeNetwork makeNetwork, Decoding decoding = Decoding::Called )
        {
            const fs::path path = referencePath( arm );
            bool still_replayed = true;
            int called_steps = 0;

            const auto trace = [ & ] {
                auto network = makeNetwork();
                network->setDecodeReplay( decoding == Decoding::Replayed );

                DecodeTrace result = traceDecode( *network, vocabulary );
                still_replayed = network->isDecodeReplayed();
                called_steps = network->calledDecodeSteps();

                return result;
            };

            if ( writing() )
            {
                if ( decoding == Decoding::Replayed )
                {
                    GTEST_SKIP() << "A replayed arm compares with the called arm's reference and writes none";
                }

                const DecodeTrace first = trace();
                const DecodeTrace second = trace();

                describe( arm, first );

                const std::string difference = differenceBetween( first, second );
                ASSERT_TRUE( difference.empty() ) << arm << " does not reproduce itself: " << difference;

                writeTrace( path, first );
                std::cout << "  wrote " << path.string() << "\n" << std::flush;

                return;
            }

            if ( !fs::exists( path ) )
            {
                GTEST_SKIP() << "No reference at " << path.string();
            }

            const DecodeTrace actual = trace();
            describe( arm, actual );

            const std::string difference = differenceBetween( readTrace( path ), actual );
            EXPECT_TRUE( difference.empty() ) << arm << " differs from its reference: " << difference;

            if ( decoding == Decoding::Replayed )
            {
                EXPECT_TRUE( still_replayed ) << arm << ": the self-check turned replay off";

                // Priming, recording and the self-check; every other step replayed.
                EXPECT_EQ( called_steps, 3 ) << arm << ": steps were called that should have been replayed";
            }
        }

        template<typename TNetwork, typename TConfig>
        auto tinyNetwork( const TConfig& config )
        {
            return [ config ] { return Common::buildTinyNetwork<TNetwork>( config, kContextLength ); };
        }

        template<typename TNetwork, typename TConfig>
        void checkPackage( const std::string& arm, const fs::path& weights, const TConfig& config,
            Decoding decoding = Decoding::Called )
        {
            checkAgainstReference( arm, static_cast<std::int32_t>( config.getVocabSize() ), [ & ] {
                return Measurement::buildMeasuredNetwork<TNetwork>( weights, config, DeviceId{ DeviceType::Cuda, 0 },
                    kContextLength );
            }, decoding );
        }

        fs::path llamaWeights()
        {
            return fs::path( TEST_DATA_DIR ) / "Models" / "LLaMa" / "llama31_8b_instruct_q4_0.safetensors";
        }

        fs::path gemmaWeights()
        {
            return fs::path( TEST_DATA_DIR ) / "Models" / "Gemma" / "gemma4_12b_it_qat_q4_0.safetensors";
        }

        LlamaConfig llamaConfig()
        {
            Serialization::WeightsReader reader( llamaWeights() );

            return LlamaBf16::configFromMetadata( reader.getWeightsMetadata() );
        }

        GemmaConfig gemmaConfig()
        {
            Serialization::WeightsReader reader( gemmaWeights() );

            return GemmaBf16::configFromMetadata( reader.getWeightsMetadata() );
        }
    }

    class DecodeReferenceCudaTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            if ( getDeviceCount( DeviceType::Cuda ) == 0 )
            {
                GTEST_SKIP() << "Needs a CUDA device";
            }
        }
    };

    class DecodeReferencePackageCudaTests : public DecodeReferenceCudaTests
    {
    protected:
        void SetUp() override
        {
            DecodeReferenceCudaTests::SetUp();

            if ( !fs::exists( llamaWeights() ) || !fs::exists( gemmaWeights() ) )
            {
                GTEST_SKIP() << "Needs " << llamaWeights().string() << " and " << gemmaWeights().string();
            }
        }
    };

    TEST_F( DecodeReferenceCudaTests, DISABLED_TinyLlamaBf16 )
    {
        checkAgainstReference( "tiny_llama_bf16", Common::kTinyVocabulary,
            tinyNetwork<TinyLlamaBf16>( Common::tinyLlamaConfig( kContextLength ) ) );
    }

    TEST_F( DecodeReferenceCudaTests, DISABLED_TinyLlamaBf16Fp8Kv )
    {
        checkAgainstReference( "tiny_llama_bf16_fp8kv", Common::kTinyVocabulary,
            tinyNetwork<TinyLlamaBf16Fp8Kv>( Common::tinyLlamaConfig( kContextLength ) ) );
    }

    TEST_F( DecodeReferenceCudaTests, DISABLED_TinyLlamaFp32 )
    {
        checkAgainstReference( "tiny_llama_fp32", Common::kTinyVocabulary,
            tinyNetwork<TinyLlamaFp32>( Common::tinyLlamaConfig( kContextLength ) ) );
    }

    TEST_F( DecodeReferenceCudaTests, DISABLED_TinyGemmaBf16 )
    {
        checkAgainstReference( "tiny_gemma_bf16", Common::kTinyVocabulary,
            tinyNetwork<TinyGemmaBf16>( Common::tinyGemmaConfig( kContextLength ) ) );
    }

    TEST_F( DecodeReferenceCudaTests, DISABLED_TinyQwenBf16 )
    {
        checkAgainstReference( "tiny_qwen_bf16", Common::kTinyVocabulary,
            tinyNetwork<TinyQwenBf16>( Common::tinyQwenConfig( kContextLength ) ) );
    }

    TEST_F( DecodeReferenceCudaTests, DISABLED_TinyLlamaBf16Replayed )
    {
        checkAgainstReference( "tiny_llama_bf16", Common::kTinyVocabulary,
            tinyNetwork<TinyLlamaBf16>( Common::tinyLlamaConfig( kContextLength ) ), Decoding::Replayed );
    }

    TEST_F( DecodeReferenceCudaTests, DISABLED_TinyLlamaBf16Fp8KvReplayed )
    {
        checkAgainstReference( "tiny_llama_bf16_fp8kv", Common::kTinyVocabulary,
            tinyNetwork<TinyLlamaBf16Fp8Kv>( Common::tinyLlamaConfig( kContextLength ) ), Decoding::Replayed );
    }

    TEST_F( DecodeReferenceCudaTests, DISABLED_TinyLlamaFp32Replayed )
    {
        checkAgainstReference( "tiny_llama_fp32", Common::kTinyVocabulary,
            tinyNetwork<TinyLlamaFp32>( Common::tinyLlamaConfig( kContextLength ) ), Decoding::Replayed );
    }

    TEST_F( DecodeReferenceCudaTests, DISABLED_TinyGemmaBf16Replayed )
    {
        checkAgainstReference( "tiny_gemma_bf16", Common::kTinyVocabulary,
            tinyNetwork<TinyGemmaBf16>( Common::tinyGemmaConfig( kContextLength ) ), Decoding::Replayed );
    }

    TEST_F( DecodeReferenceCudaTests, DISABLED_TinyQwenBf16Replayed )
    {
        checkAgainstReference( "tiny_qwen_bf16", Common::kTinyVocabulary,
            tinyNetwork<TinyQwenBf16>( Common::tinyQwenConfig( kContextLength ) ), Decoding::Replayed );
    }

    TEST_F( DecodeReferencePackageCudaTests, DISABLED_LlamaQ4_0 )
    {
        checkPackage<LlamaQ4_0>( "llama31_8b_q4_0", llamaWeights(), llamaConfig() );
    }

    TEST_F( DecodeReferencePackageCudaTests, DISABLED_LlamaQ4_0Fp8Kv )
    {
        checkPackage<LlamaQ4_0Fp8Kv>( "llama31_8b_q4_0_fp8kv", llamaWeights(), llamaConfig() );
    }

    TEST_F( DecodeReferencePackageCudaTests, DISABLED_GemmaQ4_0 )
    {
        checkPackage<GemmaQ4_0>( "gemma4_12b_q4_0", gemmaWeights(), gemmaConfig() );
    }

    TEST_F( DecodeReferencePackageCudaTests, DISABLED_LlamaQ4_0Replayed )
    {
        checkPackage<LlamaQ4_0>( "llama31_8b_q4_0", llamaWeights(), llamaConfig(), Decoding::Replayed );
    }

    TEST_F( DecodeReferencePackageCudaTests, DISABLED_LlamaQ4_0Fp8KvReplayed )
    {
        checkPackage<LlamaQ4_0Fp8Kv>( "llama31_8b_q4_0_fp8kv", llamaWeights(), llamaConfig(), Decoding::Replayed );
    }

    TEST_F( DecodeReferencePackageCudaTests, DISABLED_GemmaQ4_0Replayed )
    {
        checkPackage<GemmaQ4_0>( "gemma4_12b_q4_0", gemmaWeights(), gemmaConfig(), Decoding::Replayed );
    }
}
