/**
 * @file Llama.InstructionRetention.Cuda.cpp
 * @brief Whether an instruction at position 0 still governs the reply after a book fills the context, BF16 against
 *        FP8 KV cache: the behavioral arm beside the FP8 cache's loss gate (Quantization.md, Part III, decision 6).
 *
 * A loss averaged over book tokens cannot see an early instruction being forgotten; "The Pitfalls of KV Cache
 * Compression" (Chen, Geh, Grover, Van den Broeck and Israel, arXiv 2510.00231) shows compression degrading some
 * instructions far faster than others while aggregate scores hold. This reports and does not gate.
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

namespace Mila::Tests::Dnn::Models::LlamaInstructionRetention
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace fs = std::filesystem;

    namespace
    {
        using LlamaBf16 = LlamaModel<DeviceType::Cuda, TensorDataType::BF16>;

        using Bf16CacheNetwork = LlamaTransformer<DeviceType::Cuda, TensorDataType::BF16,
            Quant::Weight::PerGroupInt4<32>, Quant::KvCache::NoKvCompression>;

        using Fp8CacheNetwork = LlamaTransformer<DeviceType::Cuda, TensorDataType::BF16,
            Quant::Weight::PerGroupInt4<32>, Quant::KvCache::PerTokenKvFp8<>>;

        constexpr std::int32_t kBeginOfText = 128000;
        constexpr std::int32_t kEndOfTurn = 128009;
        constexpr std::int32_t kEndOfText = 128001;

        // A reply that reaches this without a stop token is reported as cut where the instruction is judged at its
        // end: at 32, Gemma's every reply to "end with OVER" was cut mid-sentence, so that row measured the budget
        // rather than the model.
        constexpr int kReplyTokens = 256;

        fs::path weightsPath()
        {
            return fs::path( TEST_DATA_DIR ) / "models" / "llama" / "llama31_8b_instruct_q4_0.safetensors";
        }

        fs::path tokenizerPath()
        {
            return fs::path( TEST_DATA_DIR ) / "models" / "llama" / "llama32_tokenizer.bin";
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

        /// The system instruction at position 0, the book in the user turn to `length` tokens in all, then the question.
        std::vector<std::int32_t> prompt( Mila::Data::BpeTokenizer& tokenizer, const Instruction& instruction,
            const fs::path& book, dim_t length )
        {
            std::vector<std::int32_t> head{ kBeginOfText };
            const auto system = tokenizer.encode( std::format(
                "<|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n",
                instruction.text ) );
            head.insert( head.end(), system.begin(), system.end() );

            const auto tail = tokenizer.encode( std::format(
                "\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", kQuestion ) );

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
        std::vector<Reply> replies( dim_t context, const std::vector<dim_t>& lengths, const std::vector<fs::path>& books )
        {
            const auto tokenizer = Mila::Data::BpeTokenizer::loadLlama32( tokenizerPath() );

            Serialization::WeightsReader reader( weightsPath() );
            LlamaConfig config = LlamaBf16::configFromMetadata( reader.getWeightsMetadata() );

            auto network = Measurement::buildMeasuredNetwork<TNetwork>( weightsPath(), config, DeviceId{ DeviceType::Cuda, 0 }, context );

            std::vector<Reply> results;

            for ( const dim_t length : lengths )
            {
                for ( const fs::path& book : books )
                {
                    for ( const Instruction& instruction : kInstructions )
                    {
                        const auto tokens = prompt( *tokenizer, instruction, book, length );
                        const auto generated = Measurement::greedyContinuationOf( *network, tokens, kReplyTokens,
                            { kEndOfTurn, kEndOfText }, context );
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
    }

    // MilaTests --gtest_also_run_disabled_tests --gtest_filter=LlamaInstructionRetentionCudaTests.*
    // Pin the 16 GB card by UUID. Reports only.
    TEST( LlamaInstructionRetentionCudaTests, DISABLED_Bf16AgainstFp8Cache_Q4_0 )
    {
        if ( getDeviceCount( DeviceType::Cuda ) == 0 || !fs::exists( weightsPath() ) || !fs::exists( tokenizerPath() )
            || !fs::exists( Measurement::pg19TestPath( TEST_DATA_DIR ) ) )
        {
            GTEST_SKIP() << "Needs a CUDA device, " << weightsPath().string() << ", the Llama tokenizer and PG-19";
        }

        const std::vector<fs::path> books = { Measurement::pg19TestPath( TEST_DATA_DIR ) / "30312.txt", Measurement::pg19TestPath( TEST_DATA_DIR ) / "3608.txt" };
        const std::vector<dim_t> shared = { 2048, 16384, 65536 };

        // The BF16 cache reaches 69632 on the 16 GB card; the FP8 cache 131072, which it alone is asked about.
        const std::vector<Reply> bf16 = replies<Bf16CacheNetwork>( 69632, shared, books );

        std::vector<dim_t> fp8_lengths = shared;
        fp8_lengths.push_back( 131072 - 1024 );
        const std::vector<Reply> fp8 = replies<Fp8CacheNetwork>( 131072, fp8_lengths, books );

        std::size_t index = 0;
        std::size_t bf16_holds = 0, fp8_holds = 0, cut = 0;

        std::cout << std::format( "  {:>7} {:>6} {:<20} {:<5} {:<5}  replies (BF16 | FP8)\n", "length", "book", "instruction",
            "BF16", "FP8" );

        for ( const dim_t length : fp8_lengths )
        {
            for ( const fs::path& book : books )
            {
                for ( const Instruction& instruction : kInstructions )
                {
                    const bool shared_length = index < bf16.size();
                    const Reply& fp8_reply = fp8[ index ];

                    std::cout << std::format( "  {:>7} {:>6} {:<20} {:<5} {:<5}  {} | {}\n", length, book.stem().string(),
                        instruction.name, shared_length ? bf16[ index ].verdict() : "--",
                        fp8_reply.verdict(), shared_length ? oneLine( bf16[ index ].text ) : "",
                        oneLine( fp8_reply.text ) ) << std::flush;

                    bf16_holds += shared_length && bf16[ index ].holds ? 1 : 0;
                    fp8_holds += shared_length && fp8_reply.holds ? 1 : 0;
                    cut += ( shared_length && bf16[ index ].cut ? 1 : 0 ) + ( fp8_reply.cut ? 1 : 0 );
                    ++index;
                }
            }
        }

        std::cout << std::format( "  at the lengths both reach: BF16 cache holds {} of {}, FP8 cache {} of {}; "
            "{} replies cut at {} tokens in all\n", bf16_holds, bf16.size(), fp8_holds, bf16.size(), cut, kReplyTokens );
    }
}
