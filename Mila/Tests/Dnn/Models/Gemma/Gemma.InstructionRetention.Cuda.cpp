/**
 * @file Gemma.InstructionRetention.Cuda.cpp
 * @brief Whether an instruction at position 0 still governs the reply after a book fills the context, BF16 against
 *        an FP8 global-layer KV cache: the behavioral arm beside the FP8 cache's loss gate (Quantization.md, Part III,
 *        decision 6).
 *
 * The Gemma counterpart of Llama.InstructionRetention.Cuda.cpp, with the same instructions, question and books.
 * Reports and does not gate.
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <format>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

import Mila;

#include "Measurement/LogLikelihoodHarness.h"
#include "Measurement/Pg19Books.h"

namespace Mila::Tests::Dnn::Models::GemmaInstructionRetention
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace fs = std::filesystem;

    namespace
    {
        using GemmaBf16 = GemmaModel<DeviceType::Cuda, TensorDataType::BF16>;

        using Bf16Cache12B = GemmaTransformer<DeviceType::Cuda, TensorDataType::BF16,
            Quant::Weight::PerGroupInt4<32>, GemmaBf16::GemmaSlidingKvPolicy>;
        using Fp8GlobalCache12B = GemmaTransformer<DeviceType::Cuda, TensorDataType::BF16,
            Quant::Weight::PerGroupInt4<32>, GemmaBf16::GemmaSlidingKvPolicy, GemmaFeedForward::Dense,
            Quant::KvCache::PerTokenKvFp8<>>;

        using Bf16Cache26B = GemmaTransformer<DeviceType::Cuda, TensorDataType::BF16,
            Quant::Weight::PerGroupInt4<32>, GemmaBf16::GemmaSlidingKvPolicy, GemmaFeedForward::Routed>;
        using Fp8GlobalCache26B = GemmaTransformer<DeviceType::Cuda, TensorDataType::BF16,
            Quant::Weight::PerGroupInt4<32>, GemmaBf16::GemmaSlidingKvPolicy, GemmaFeedForward::Routed,
            Quant::KvCache::PerTokenKvFp8<>>;

        // The tokenizer does not add <bos>; <eos> and <end_of_turn> are GemmaModel's own stop set.
        constexpr std::int32_t kBos = 2;
        const std::unordered_set<std::int32_t> kStopTokens{ 1, 106 };

        // A reply that reaches this without a stop token is reported as cut where the instruction is judged at its
        // end: at 32 every reply to "end with OVER" was cut mid-sentence, so that row measured the budget rather than
        // the model.
        constexpr int kReplyTokens = 256;

        fs::path weights12BPath()
        {
            return fs::path( TEST_DATA_DIR ) / "models" / "gemma" / "gemma4_12b_it_qat_q4_0.safetensors";
        }

        // Quantized on load to Q4_0 from Google's unquantized QAT checkpoint.
        fs::path weights26BPath()
        {
            return fs::path( TEST_DATA_DIR ) / "models" / "gemma" / "gemma4_26b_a4b_it_qat_bf16.bin";
        }

        fs::path tokenizerPath()
        {
            return fs::path( TEST_DATA_DIR ) / "models" / "gemma" / "gemma_tokenizer.bin";
        }

        /// The reply's words, letters only. No input stream: in a TU that does `import Mila;` its sentry is incomplete.
        std::vector<std::string> wordsOf( const std::string& reply )
        {
            std::vector<std::string> words;
            std::string letters;

            for ( std::size_t index = 0; index <= reply.size(); ++index )
            {
                const char c = index < reply.size() ? reply[ index ] : ' ';

                if ( std::isspace( static_cast<unsigned char>( c ) ) )
                {
                    if ( !letters.empty() )
                    {
                        words.push_back( letters );
                        letters.clear();
                    }
                }
                else if ( std::isalpha( static_cast<unsigned char>( c ) ) )
                {
                    letters.push_back( c );
                }
            }

            return words;
        }

        struct Instruction
        {
            std::string_view name;
            std::string_view text;
            bool ( *holds )( const std::string& reply );
            bool judged_at_end{ false };
        };

        const Instruction kInstructions[] = {
            { "three capital words", "Whatever the user asks, reply in exactly three words, all in capital letters.",
                []( const std::string& reply )
                {
                    const auto words = wordsOf( reply );

                    return words.size() == 3 && std::all_of( words.begin(), words.end(), []( const std::string& word )
                    {
                        return std::all_of( word.begin(), word.end(), []( char c ) { return std::isupper( static_cast<unsigned char>( c ) ) != 0; } );
                    } );
                } },
            { "begins BANANA", "Begin every reply with the word BANANA.",
                []( const std::string& reply )
                {
                    const auto words = wordsOf( reply );

                    return !words.empty() && words.front() == "BANANA";
                } },
            { "ends OVER", "End every reply with the word OVER.",
                []( const std::string& reply )
                {
                    const auto words = wordsOf( reply );

                    return !words.empty() && words.back() == "OVER";
                }, true },
        };

        constexpr std::string_view kQuestion = "Who is the main character of this book, and what do they want?";

        /// The system turn at position 0, the book in the user turn to `length` tokens in all, then the question and a
        /// model turn primed with thinking off, as Gemma::formatPrompt primes it.
        std::vector<std::int32_t> prompt( Mila::Data::BpeTokenizer& tokenizer, const Instruction& instruction,
            const fs::path& book, dim_t length )
        {
            namespace Protocol = ::Mila::Dnn::Gemma;

            std::vector<std::int32_t> head{ kBos };
            const auto system = tokenizer.encode( std::format( "{}system\n{}{}\n{}user\n",
                Protocol::kTurnOpen, instruction.text, Protocol::kTurnClose, Protocol::kTurnOpen ) );
            head.insert( head.end(), system.begin(), system.end() );

            const auto tail = tokenizer.encode( std::format( "\n\n{}{}\n{}model\n{}",
                kQuestion, Protocol::kTurnClose, Protocol::kTurnOpen, Protocol::kThoughtPrime ) );

            const std::size_t book_tokens = static_cast<std::size_t>( length ) - head.size() - tail.size() - kReplyTokens;
            const auto text = tokenizer.encode( Measurement::joinWraps( Measurement::readBook( book, book_tokens * 6 ) ) );

            if ( text.size() < book_tokens )
            {
                return {};
            }

            std::vector<std::int32_t> tokens = head;
            tokens.insert( tokens.end(), text.begin(), text.begin() + static_cast<std::ptrdiff_t>( book_tokens ) );
            tokens.insert( tokens.end(), tail.begin(), tail.end() );

            return tokens;
        }

        struct Reply
        {
            std::string text;
            bool holds;
            bool cut;

            std::string_view verdict() const
            {
                return cut ? "cut" : holds ? "holds" : "lost";
            }
        };

        template<typename TNetwork>
        std::vector<Reply> replies( const fs::path& weights, dim_t context, const std::vector<dim_t>& lengths,
            const std::vector<fs::path>& books )
        {
            const auto tokenizer = Mila::Data::BpeTokenizer::loadGemma( tokenizerPath() );

            Serialization::WeightsReader reader( weights );
            const GemmaConfig config = GemmaBf16::configFromMetadata( reader.getWeightsMetadata() );

            PrefillChunking chunking;
            auto network = Measurement::buildMeasuredNetwork<TNetwork>( weights, config, DeviceId{ DeviceType::Cuda, 0 },
                context, &chunking );

            std::cout << std::format( "  {}: context {}, prefill chunk {}\n", weights.filename().string(), context,
                chunking.chunk_rows ) << std::flush;

            std::vector<Reply> results;

            for ( const dim_t length : lengths )
            {
                for ( const fs::path& book : books )
                {
                    for ( const Instruction& instruction : kInstructions )
                    {
                        const auto tokens = prompt( *tokenizer, instruction, book, length );
                        const auto generated = Measurement::greedyContinuationOf( *network, tokens, kReplyTokens,
                            kStopTokens, context );
                        const std::string text = tokenizer->decode( generated.tokens );

                        const bool cut = instruction.judged_at_end
                            && generated.tokens.size() >= static_cast<std::size_t>( kReplyTokens );

                        results.push_back( { text, !cut && instruction.holds( text ), cut } );
                    }
                }
            }

            return results;
        }

        std::string oneLine( std::string text )
        {
            std::replace( text.begin(), text.end(), '\n', ' ' );

            // Both ends: an instruction can govern either.
            return text.size() > 60 ? text.substr( 0, 28 ) + "..." + text.substr( text.size() - 29 ) : text;
        }

        /// One row per length, book and instruction; the BF16 arm's lengths are a prefix of the FP8 arm's.
        void report( const std::vector<dim_t>& lengths, const std::vector<fs::path>& books, const std::vector<Reply>& bf16,
            const std::vector<Reply>& fp8 )
        {
            std::size_t index = 0;
            std::size_t bf16_holds = 0, fp8_holds = 0, cut = 0;

            std::cout << std::format( "  {:>7} {:>6} {:<20} {:<5} {:<5}  replies (BF16 | FP8 global)\n", "length", "book",
                "instruction", "BF16", "FP8" );

            for ( const dim_t length : lengths )
            {
                for ( const fs::path& book : books )
                {
                    for ( const Instruction& instruction : kInstructions )
                    {
                        const bool shared_length = index < bf16.size();
                        const Reply& fp8_reply = fp8[ index ];

                        std::cout << std::format( "  {:>7} {:>6} {:<20} {:<5} {:<5}  {} | {}\n", length,
                            book.stem().string(), instruction.name,
                            shared_length ? bf16[ index ].verdict() : "--",
                            fp8_reply.verdict(), shared_length ? oneLine( bf16[ index ].text ) : "",
                            oneLine( fp8_reply.text ) ) << std::flush;

                        bf16_holds += shared_length && bf16[ index ].holds ? 1 : 0;
                        fp8_holds += shared_length && fp8_reply.holds ? 1 : 0;
                        cut += ( shared_length && bf16[ index ].cut ? 1 : 0 ) + ( fp8_reply.cut ? 1 : 0 );
                        ++index;
                    }
                }
            }

            std::cout << std::format( "  at the lengths both reach: BF16 cache holds {} of {}, FP8 global cache {} of {}; "
                "{} replies cut at {} tokens in all\n", bf16_holds, bf16.size(), fp8_holds, bf16.size(), cut, kReplyTokens );
        }

        bool inputsPresent( const fs::path& weights )
        {
            return getDeviceCount( DeviceType::Cuda ) > 0 && fs::exists( weights ) && fs::exists( tokenizerPath() )
                && fs::exists( Measurement::pg19TestPath( TEST_DATA_DIR ) );
        }

        std::vector<fs::path> books()
        {
            return { Measurement::pg19TestPath( TEST_DATA_DIR ) / "30312.txt", Measurement::pg19TestPath( TEST_DATA_DIR ) / "3608.txt" };
        }
    }

    // MilaTests --gtest_also_run_disabled_tests --gtest_filter=GemmaInstructionRetentionCudaTests.*
    // Pin the 16 GB card by UUID. Reports only.
    // The 12B at 65536, where decision 6's loss arm ran both caches.
    TEST( GemmaInstructionRetentionCudaTests, DISABLED_Bf16AgainstFp8GlobalCache_12B_Q4_0 )
    {
        if ( !inputsPresent( weights12BPath() ) )
        {
            GTEST_SKIP() << "Needs a CUDA device, " << weights12BPath().string() << ", the Gemma tokenizer and PG-19";
        }

        const std::vector<dim_t> lengths = { 2048, 16384, 65536 - 1024 };

        const std::vector<Reply> bf16 = replies<Bf16Cache12B>( weights12BPath(), 65536, lengths, books() );
        const std::vector<Reply> fp8 = replies<Fp8GlobalCache12B>( weights12BPath(), 65536, lengths, books() );

        report( lengths, books(), bf16, fp8 );
    }

    // The 26B-A4B: the BF16 caches reach 32768 on the 16 GB card, the FP8 global cache 65536, which it alone is asked
    // about.
    TEST( GemmaInstructionRetentionCudaTests, DISABLED_Bf16AgainstFp8GlobalCache_26B_Q4_0 )
    {
        if ( !inputsPresent( weights26BPath() ) )
        {
            GTEST_SKIP() << "Needs a CUDA device, " << weights26BPath().string() << ", the Gemma tokenizer and PG-19";
        }

        const std::vector<dim_t> shared = { 2048, 16384, 32768 - 1024 };

        const std::vector<Reply> bf16 = replies<Bf16Cache26B>( weights26BPath(), 32768, shared, books() );

        std::vector<dim_t> fp8_lengths = shared;
        fp8_lengths.push_back( 65536 - 1024 );
        const std::vector<Reply> fp8 = replies<Fp8GlobalCache26B>( weights26BPath(), 65536, fp8_lengths, books() );

        report( fp8_lengths, books(), bf16, fp8 );
    }

    // The 26B-A4B past what the 16 GB card fits, for the K = V discussion (RopeInAttention.md, ContextProfile.md): with
    // the FP8 global cache, 131072 is about 406 MiB over the card's free memory, so the network is built past it. The
    // replies do not depend on the fit; nothing here is a rate. 64K again at this build, so the rows above it read
    // against one the card holds. Reports only.
    TEST( GemmaInstructionRetentionCudaTests, DISABLED_Fp8GlobalCache_26B_Q4_0_PastTheCard )
    {
        if ( !inputsPresent( weights26BPath() ) )
        {
            GTEST_SKIP() << "Needs a CUDA device, " << weights26BPath().string() << ", the Gemma tokenizer and PG-19";
        }

        const std::vector<dim_t> lengths = { 65536 - 1024, 98304 - 1024, 131072 - 1024 };
        const std::vector<Reply> fp8 = replies<Fp8GlobalCache26B>( weights26BPath(), 131072, lengths, books() );

        report( lengths, books(), {}, fp8 );

        const std::size_t per_length = books().size() * std::size( kInstructions );

        for ( std::size_t index = 0; index < lengths.size(); ++index )
        {
            const auto first = fp8.begin() + static_cast<std::ptrdiff_t>( index * per_length );
            const auto holds = std::count_if( first, first + static_cast<std::ptrdiff_t>( per_length ),
                []( const Reply& reply ) { return reply.holds; } );

            std::cout << std::format( "  at {}: FP8 global cache holds {} of {}\n", lengths[ index ], holds, per_length );
        }

        std::cout << std::flush;
    }
}
