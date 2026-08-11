/**
 * @file BpeTokenizer.ixx
 * @brief Unified BPE tokenizer for GPT-2, Llama 3.x, Mistral, and Gemma model families.
 *
 * Encode pipeline:
 *   1. Special token pre-pass: split input on registered special token strings
 *      (longest-first scan) and emit their IDs directly, bypassing BPE entirely.
 *      GPT-2 vocabularies with no registered special tokens skip this via fast path.
 *   2. Pre-tokenize each plain text segment with the configured regex pattern.
 *   3. Byte-encode each pre-token using the GPT-2 style byte encoder.
 *   4a. BPE path (GPT-2 / trained vocabularies): apply explicit merge rules
 *       greedily, lowest priority index first.
 *   4b. Max-munch path (Llama 3.x / TikToken): find the longest vocabulary match
 *       at each position in the encoded unit sequence. Used when no merge rules
 *       are present; the merge order is implicit in the token ID assignment.
 *   4c. SentencePiece path (Gemma): Metaspace pre-tokenization (space -> U+2581),
 *       UTF-8 character initial units with <0xNN> byte fallback, then the merge
 *       rules by rank. Selected when the vocabulary uses PreTokenizationMode::SentencePiece.
 *   5. Map final tokens to IDs; fall back to 0 on a miss.
 *
 * Decode pipeline:
 *   Concatenate token strings and reverse the byte encoding back to UTF-8.
 */

module;
#include <string>
#include <string_view>
#include <vector>
#include <span>
#include <memory>
#include <optional>
#include <filesystem>
#include <mutex>
#include <regex>
#include <limits>
#include <stdexcept>

export module Data.BpeTokenizer;

import Data.BpeVocabulary;
import Data.BpePreTokenizationMode;
import Data.Tokenizer;
import Data.TokenizerVocabulary;
import Logging.Logger;

namespace Mila::Data
{
    using Mila::Data::TokenId;
    using Mila::Data::Tokenizer;

    /**
     * @brief Unified BPE tokenizer targeting GPT-2, Llama 3.x, and Mistral model families.
     *
     * Construct from a pre-built vocabulary or via the convenience factory methods:
     *
     * @code
     * // GPT-2
     * auto tok = BpeTokenizer::loadGpt2( "gpt2_tokenizer.bin" );
     * auto ids = tok->encode( "Hello, world!" );
     *
     * // Llama 3.2
     * auto tok = BpeTokenizer::loadLlama32( "llama32_tokenizer.bin" );
     * auto ids = tok->encode( "<|begin_of_text|>Hello, world!" );
     * @endcode
     *
     * The special token pre-pass is enabled whenever the vocabulary registers at least
     * one special token. For GPT-2, this means "<|endoftext|>" is intercepted before
     * BPE runs; for Llama 3.x, the full set of named and extended tokens is intercepted.
     */
    export class BpeTokenizer : public Tokenizer
    {
    public:

        explicit BpeTokenizer( BpeVocabulary vocab )
            : vocab_( std::move( vocab ) )
        {
            is_sentencepiece_ =
                vocab_.getConfig().getPreTokenizationMode() == PreTokenizationMode::SentencePiece;

            initializePreTokenization();
        }

        // ====================================================================
        // Factory Methods
        // ====================================================================

        /**
         * @brief Load a tokenizer from a Mila binary vocabulary file.
         *
         * @param path Path to a vocabulary file written by BpeVocabulary::save().
         * @return Loaded BpeTokenizer instance.
         * @throws std::runtime_error on I/O or format errors.
         */
        static BpeTokenizer load( const std::filesystem::path& path )
        {
            return BpeTokenizer( BpeVocabulary::load( path ) );
        }

        /**
         * @brief Load a GPT-2 tokenizer from the binary produced by convert_gpt2_tokenizer.py.
         *
         * @param path Path to the GPT-2 tokenizer binary.
         * @return Shared tokenizer instance.
         * @throws std::runtime_error on I/O or format errors.
         */
        static std::shared_ptr<BpeTokenizer> loadGpt2( const std::filesystem::path& path )
        {
            return std::make_shared<BpeTokenizer>( BpeVocabulary::loadGpt2( path ) );
        }

        /**
         * @brief Load a Llama 3.2 tokenizer from the binary produced by convert_llama_tokenizer.py.
         *
         * @param path Path to the Llama 3.2 tokenizer binary.
         * @return Shared tokenizer instance.
         * @throws std::runtime_error on I/O or format errors.
         */
        static std::shared_ptr<BpeTokenizer> loadLlama32( const std::filesystem::path& path )
        {
            return std::make_shared<BpeTokenizer>( BpeVocabulary::loadLlama32( path ) );
        }

        /**
         * @brief Load a Gemma 4 tokenizer from the binary produced by convert_tokenizer.py.
         *
         * Gemma uses SentencePiece BPE: byte_level=false (raw UTF-8 pieces, U+2581
         * for spaces) with a Metaspace pre-tokenization and byte fallback.
         *
         * @param path Path to the Gemma tokenizer binary.
         * @return Shared tokenizer instance.
         * @throws std::runtime_error on I/O or format errors.
         */
        static std::shared_ptr<BpeTokenizer> loadGemma( const std::filesystem::path& path )
        {
            return std::make_shared<BpeTokenizer>( BpeVocabulary::loadGemma( path ) );
        }

        /**
         * @brief Load a Mistral tokenizer.
         *
         * @note Not yet implemented. Provide a Mila binary produced by save() as a workaround.
         * @throws std::runtime_error always.
         */
        static std::shared_ptr<BpeTokenizer> loadMistral(
            const std::filesystem::path& vocab_path,
            const std::filesystem::path& merges_path )
        {
            return std::make_shared<BpeTokenizer>(
                BpeVocabulary::loadMistral( vocab_path, merges_path ) );
        }

        // ====================================================================
        // Tokenizer Interface
        // ====================================================================

        /**
         * @brief Encode text to a sequence of token IDs.
         *
         * Performs the special token pre-pass first when the vocabulary has registered
         * special tokens. Plain text segments between special tokens are processed
         * through the standard pre-tokenization and BPE merge pipeline.
         *
         * @param text Input text (UTF-8).
         * @return Sequence of token IDs.
         */
        std::vector<TokenId> encode( const std::string& text ) override
        {
            std::vector<TokenId> out;
            const auto& special_list = vocab_.getSpecialTokenList();

            if ( special_list.empty() )
            {
                // Fast path: no special tokens registered, skip pre-pass entirely.
                encodeSegment( text, out );
            }
            else
            {
                size_t pos = 0;

                while ( pos < text.size() )
                {
                    // Try to match a special token at the current position (longest first).
                    bool matched = false;

                    for ( const auto& [token_str, token_id] : special_list )
                    {
                        if ( text.compare( pos, token_str.size(), token_str ) == 0 )
                        {
                            out.push_back( token_id );
                            pos += token_str.size();
                            matched = true;
                            break;
                        }
                    }

                    if ( !matched )
                    {
                        // Advance to the next special token boundary (or end of string).
                        size_t next_special = text.size();

                        for ( const auto& [token_str, token_id] : special_list )
                        {
                            const size_t found = text.find( token_str, pos );

                            if ( found != std::string::npos && found < next_special )
                            {
                                next_special = found;
                            }
                        }

                        encodeSegment( text.substr( pos, next_special - pos ), out );
                        pos = next_special;
                    }
                }
            }

            return out;
        }

        /**
         * @brief Decode a sequence of token IDs back to a UTF-8 string.
         *
         * Each token string is byte-decoded using the GPT-2 style byte mapping.
         * IDs with no vocabulary entry emit a '?' placeholder.
         *
         * @param tokens Sequence of token IDs.
         * @return Decoded UTF-8 string.
         */
        std::string decode( std::span<const TokenId> tokens ) override
        {
            std::string out;
            out.reserve( tokens.size() * 4 );

            for ( TokenId id : tokens )
            {
                auto tok_opt = vocab_.idToToken( id );

                if ( tok_opt )
                {
                    decodeToken( *tok_opt, out );
                }
                else
                {
                    out.push_back( '?' );
                }
            }

            return out;
        }

        size_t getVocabSize() const override
        {
            return vocab_.getSize();
        }

        std::optional<TokenId> getBosTokenId() const override
        {
            const auto& st = vocab_.getConfig().getSpecialTokens();

            if ( !st.use_bos )
            {
                return std::nullopt;
            }

            return vocab_.getSpecialTokenId( st.bos_token );
        }

        std::optional<TokenId> getEosTokenId() const override
        {
            const auto& st = vocab_.getConfig().getSpecialTokens();

            if ( !st.use_eos )
            {
                return std::nullopt;
            }

            return vocab_.getSpecialTokenId( st.eos_token );
        }

        std::optional<TokenId> getPadTokenId() const override
        {
            const auto& st = vocab_.getConfig().getSpecialTokens();

            if ( !st.use_pad )
            {
                return std::nullopt;
            }

            return vocab_.getSpecialTokenId( st.pad_token );
        }

        std::string tokenToString( TokenId tokenId ) const override
        {
            auto tok = vocab_.idToToken( tokenId );
            return tok ? *tok : std::string{};
        }

        bool isValidToken( TokenId tokenId ) const override
        {
            return static_cast<bool>(vocab_.idToToken( tokenId ));
        }

        const BpeVocabulary& getVocab() const
        {
            return vocab_;
        }

    private:

        BpeVocabulary              vocab_;
        std::optional<std::regex>  pre_tokenization_regex_;
        bool                       is_sentencepiece_ = false;

        // ====================================================================
        // Initialization
        // ====================================================================

        /**
         * @brief Build the pre-tokenization regex from the vocabulary config.
         *
         * Attempts to compile the Unicode pattern first. If std::regex rejects it
         * (MSVC ECMAScript mode does not support \p{L} / \p{N}), falls back to the
         * ASCII-only approximation for the detected mode. Llama3Regex and Gpt2Regex
         * each have a dedicated ASCII fallback; an unrecognised pattern that fails
         * compilation is treated as a hard error.
         */
        void initializePreTokenization()
        {
            // SentencePiece does not pre-tokenize with a regex; the Metaspace split
            // is handled directly in preTokenize().
            if ( is_sentencepiece_ )
            {
                return;
            }

            const auto& pattern = vocab_.getConfig().getPreTokenizationPattern();

            if ( pattern.empty() )
            {
                return;
            }

            try
            {
                pre_tokenization_regex_ = std::regex( pattern, std::regex::ECMAScript );
            }
            catch ( const std::regex_error& )
            {
                const auto mode = vocab_.getConfig().getPreTokenizationMode();
                const char* fallback = nullptr;

                if ( mode == PreTokenizationMode::Gpt2Regex )
                {
                    fallback = GPT2_PRETOKENIZATION_PATTERN_ASCII_FALLBACK;
                }
                else if ( mode == PreTokenizationMode::Llama3Regex )
                {
                    fallback = LLAMA3_PRETOKENIZATION_PATTERN_ASCII_FALLBACK;
                }

                if ( !fallback )
                {
                    throw std::runtime_error(
                        "BpeTokenizer: pre-tokenization pattern failed to compile "
                        "and no ASCII fallback is defined for this mode: " + pattern );
                }

                // Once per process, not once per tokenizer. This fires on EVERY
                // construction -- MSVC's std::regex has no \p{...}, so the Unicode
                // pattern never compiles there -- and a warning repeated that often
                // reads as background noise rather than as the standing limitation it
                // describes. Measured at 52 of 416 first-chance exceptions across the
                // test suite, a 100% hit rate.
                static std::once_flag fallback_warning_flag;

                std::call_once( fallback_warning_flag, []
                    {
                        Logging::Logger::warning(
                            "Unicode regex not supported by std::regex; using the ASCII fallback "
                            "for BPE pre-tokenization. On this toolchain that is permanent, not "
                            "occasional: non-ASCII text WILL tokenize differently from the "
                            "HuggingFace reference. Affects GPT-2 and Llama 3; SentencePiece "
                            "models (Gemma) do not use this path." );
                    } );

                pre_tokenization_regex_ = std::regex( fallback, std::regex::ECMAScript );
            }
        }

        // ====================================================================
        // Encode Helpers
        // ====================================================================

        /**
         * @brief Encode a plain text segment (guaranteed to contain no special tokens).
         *
         * Dispatches to the BPE merge path when explicit merge rules are present,
         * or to the max-munch path for TikToken-style vocabularies (Llama 3.x).
         *
         * @param text Plain text segment.
         * @param out  Accumulator for output token IDs.
         */
        void encodeSegment( const std::string& text, std::vector<TokenId>& out )
        {
            if ( text.empty() )
            {
                return;
            }

            const std::vector<std::string> words = preTokenize( text );

            if ( is_sentencepiece_ )
            {
                encodeSegmentSentencePiece( words, out );
            }
            else if ( vocab_.getMergeRules().empty() )
            {
                encodeSegmentMaxMunch( words, out );
            }
            else
            {
                encodeSegmentBpe( words, out );
            }
        }

        /**
         * @brief Max-munch encode for TikToken vocabularies (Llama 3.x).
         *
         * Byte-encodes each pre-token then scans for the longest vocabulary match
         * at each position. Falls back to ID 0 for unrecognised units.
         *
         * @param words Pre-tokenized segments from the regex pass.
         * @param out   Accumulator for output token IDs.
         */
        void encodeSegmentMaxMunch( const std::vector<std::string>& words, std::vector<TokenId>& out )
        {
            for ( const auto& word : words )
            {
                std::vector<std::string> units;
                units.reserve( word.size() );

                if ( vocab_.isByteLevel() )
                {
                    const auto& byte_encoder = BpeVocabulary::getByteEncoder();

                    for ( unsigned char byte : word )
                    {
                        units.push_back( byte_encoder.at( byte ) );
                    }
                }
                else
                {
                    for ( char c : word )
                    {
                        units.push_back( std::string( 1, c ) );
                    }
                }

                size_t pos = 0;

                while ( pos < units.size() )
                {
                    bool found = false;

                    for ( size_t len = units.size() - pos; len >= 1; --len )
                    {
                        std::string candidate;

                        for ( size_t j = pos; j < pos + len; ++j )
                        {
                            candidate += units[ j ];
                        }

                        auto id = vocab_.tokenToId( candidate );

                        if ( id )
                        {
                            out.push_back( *id );
                            pos += len;
                            found = true;
                            break;
                        }
                    }

                    if ( !found )
                    {
                        out.push_back( 0 );
                        ++pos;
                    }
                }
            }
        }

        /**
         * @brief BPE merge encode for GPT-2 style vocabularies.
         *
         * Byte-encodes each pre-token then applies explicit merge rules greedily
         * (lowest priority index first) until no more merges are possible.
         *
         * @param words Pre-tokenized segments from the regex pass.
         * @param out   Accumulator for output token IDs.
         */
        void encodeSegmentBpe( const std::vector<std::string>& words, std::vector<TokenId>& out )
        {
            for ( const auto& word : words )
            {
                std::vector<std::string> tokens;
                tokens.reserve( word.size() );

                if ( vocab_.isByteLevel() )
                {
                    const auto& byte_encoder = BpeVocabulary::getByteEncoder();

                    for ( unsigned char byte : word )
                    {
                        tokens.push_back( byte_encoder.at( byte ) );
                    }
                }
                else
                {
                    for ( char c : word )
                    {
                        tokens.push_back( std::string( 1, c ) );
                    }
                }

                applyMerges( tokens );

                for ( const auto& token : tokens )
                {
                    auto id = vocab_.tokenToId( token );
                    out.push_back( id ? *id : 0 );
                }
            }
        }

        /**
         * @brief Greedily apply merge rules to a unit sequence, lowest priority first.
         *
         * Shared by the GPT-2 byte-level BPE path and the Gemma SentencePiece BPE
         * path; only the construction of the initial unit sequence differs.
         *
         * @param tokens In/out unit sequence; merged in place until no rule applies.
         */
        void applyMerges( std::vector<std::string>& tokens )
        {
            while ( tokens.size() > 1 )
            {
                int    best_idx = -1;
                size_t best_priority = std::numeric_limits<size_t>::max();

                for ( size_t i = 0; i < tokens.size() - 1; ++i )
                {
                    auto priority = vocab_.getMergePriority( tokens[ i ], tokens[ i + 1 ] );

                    if ( priority && *priority < best_priority )
                    {
                        best_priority = *priority;
                        best_idx = static_cast<int>( i );
                    }
                }

                if ( best_idx == -1 )
                {
                    break;
                }

                tokens[ best_idx ] += tokens[ best_idx + 1 ];
                tokens.erase( tokens.begin() + best_idx + 1 );
            }
        }

        /**
         * @brief SentencePiece BPE encode (Gemma).
         *
         * Initial units are UTF-8 *characters* (not bytes): a character that is a
         * vocabulary piece is used directly; an unknown character decomposes to its
         * SentencePiece byte-fallback pieces (<0xNN> per byte). The shared merge
         * loop then builds up the final pieces by rank.
         *
         * @param words Pre-tokens from the Metaspace split (each is "<piece-run>").
         * @param out   Accumulator for output token IDs.
         */
        void encodeSegmentSentencePiece( const std::vector<std::string>& words, std::vector<TokenId>& out )
        {
            for ( const auto& word : words )
            {
                std::vector<std::string> tokens;
                tokens.reserve( word.size() );

                size_t i = 0;

                while ( i < word.size() )
                {
                    const size_t char_len = utf8CharLength( static_cast<unsigned char>( word[ i ] ) );
                    const std::string ch = word.substr( i, char_len );

                    if ( vocab_.containsToken( ch ) )
                    {
                        tokens.push_back( ch );
                    }
                    else
                    {
                        // Byte fallback: emit one <0xNN> piece per raw byte.
                        for ( size_t b = 0; b < char_len && (i + b) < word.size(); ++b )
                        {
                            tokens.push_back( byteFallbackToken( static_cast<unsigned char>( word[ i + b ] ) ) );
                        }
                    }

                    i += char_len;
                }

                applyMerges( tokens );

                for ( const auto& token : tokens )
                {
                    auto id = vocab_.tokenToId( token );
                    out.push_back( id ? *id : 0 );
                }
            }
        }

        /**
         * @brief SentencePiece byte-fallback piece for a raw byte, e.g. 0x0A -> "<0x0A>".
         */
        static std::string byteFallbackToken( unsigned char byte )
        {
            static const char* hex = "0123456789ABCDEF";
            std::string token = "<0x";
            token.push_back( hex[ (byte >> 4) & 0xF ] );
            token.push_back( hex[ byte & 0xF ] );
            token.push_back( '>' );
            return token;
        }
        /**
         * @brief Split text into pre-tokens using the configured regex.
         *
         * Returns the entire text as a single element when no regex is configured
         * (e.g., vocabularies built with PreTokenizationMode::None).
         *
         * @param text Input text segment.
         * @return Vector of pre-token strings.
         */
        std::vector<std::string> preTokenize( const std::string& text )
        {
            if ( is_sentencepiece_ )
            {
                return preTokenizeSentencePiece( text );
            }

            if ( !pre_tokenization_regex_ )
            {
                return { text };
            }

            std::vector<std::string> words;
            std::sregex_iterator it( text.begin(), text.end(), *pre_tokenization_regex_ );
            std::sregex_iterator end;

            for ( ; it != end; ++it )
            {
                words.push_back( it->str() );
            }

            return words.empty() ? std::vector<std::string>{ text } : words;
        }

        /**
         * @brief SentencePiece Metaspace pre-tokenization (Gemma).
         *
         * Replaces ASCII spaces with the metaspace mark U+2581 ('lower one eighth
         * block'), then splits so each mark begins a new pre-token. No leading mark
         * is prepended (Gemma's tokenizer does not add a prefix space -- confirmed by
         * the HF round-trip: "The capital" -> ["The", "_capital"]).
         *
         * @param text Plain text segment (no special tokens).
         * @return Pre-tokens, each a contiguous run beginning at a mark (or the lead).
         */
        std::vector<std::string> preTokenizeSentencePiece( const std::string& text ) const
        {
            static const std::string MARK = "\xE2\x96\x81";  // U+2581

            std::string marked;
            marked.reserve( text.size() );

            for ( char c : text )
            {
                if ( c == ' ' )
                {
                    marked += MARK;
                }
                else
                {
                    marked.push_back( c );
                }
            }

            std::vector<std::string> words;
            size_t i = 0;

            while ( i < marked.size() )
            {
                const size_t start = i;

                // Consume the first character of this pre-token (possibly the mark).
                i += utf8CharLength( static_cast<unsigned char>( marked[ i ] ) );

                // Extend until the next mark boundary.
                while ( i < marked.size() && marked.compare( i, MARK.size(), MARK ) != 0 )
                {
                    i += utf8CharLength( static_cast<unsigned char>( marked[ i ] ) );
                }

                words.push_back( marked.substr( start, i - start ) );
            }

            return words;
        }

        /**
         * @brief Reverse byte-encode a single token string and append to @p out.
         *
         * For byte-level vocabularies each UTF-8 character in the token string maps
         * back to one raw byte via the GPT-2 byte decoder. Characters without a
         * decoder entry emit '?'.
         *
         * @param token Token string from the vocabulary.
         * @param out   Output UTF-8 string to append to.
         */
        void decodeToken( const std::string& token, std::string& out )
        {
            if ( is_sentencepiece_ )
            {
                decodeTokenSentencePiece( token, out );
                return;
            }

            if ( !vocab_.isByteLevel() )
            {
                out.append( token );
                return;
            }

            const auto& byte_decoder = BpeVocabulary::getByteDecoder();
            size_t i = 0;

            while ( i < token.size() )
            {
                const size_t char_len = utf8CharLength( static_cast<unsigned char>( token[ i ] ) );
                const std::string utf8_char = token.substr( i, char_len );
                const auto it = byte_decoder.find( utf8_char );

                out.push_back( it != byte_decoder.end()
                    ? static_cast<char>( it->second )
                    : '?' );

                i += char_len;
            }
        }

        /**
         * @brief Decode a SentencePiece piece: U+2581 -> space, <0xNN> -> raw byte.
         *
         * Other characters are appended verbatim. Multi-byte characters that were
         * byte-fallback-encoded come back as consecutive <0xNN> pieces whose raw
         * bytes reassemble into the original UTF-8 sequence.
         */
        void decodeTokenSentencePiece( const std::string& token, std::string& out )
        {
            // Byte-fallback piece "<0xNN>" -> the raw byte.
            if ( token.size() == 6 && token.compare( 0, 3, "<0x" ) == 0 && token[ 5 ] == '>' )
            {
                const int hi = hexValue( token[ 3 ] );
                const int lo = hexValue( token[ 4 ] );

                if ( hi >= 0 && lo >= 0 )
                {
                    out.push_back( static_cast<char>( (hi << 4) | lo ) );
                    return;
                }
            }

            static const std::string MARK = "\xE2\x96\x81";  // U+2581
            size_t i = 0;

            while ( i < token.size() )
            {
                if ( token.compare( i, MARK.size(), MARK ) == 0 )
                {
                    out.push_back( ' ' );
                    i += MARK.size();
                }
                else
                {
                    out.push_back( token[ i ] );
                    ++i;
                }
            }
        }

        static int hexValue( char c )
        {
            if ( c >= '0' && c <= '9' ) return c - '0';
            if ( c >= 'A' && c <= 'F' ) return c - 'A' + 10;
            if ( c >= 'a' && c <= 'f' ) return c - 'a' + 10;
            return -1;
        }

        static size_t utf8CharLength( unsigned char first_byte )
        {
            if ( (first_byte & 0x80) == 0x00 ) return 1;
            if ( (first_byte & 0xE0) == 0xC0 ) return 2;
            if ( (first_byte & 0xF0) == 0xE0 ) return 3;
            if ( (first_byte & 0xF8) == 0xF0 ) return 4;
            return 1;
        }
    };
}
