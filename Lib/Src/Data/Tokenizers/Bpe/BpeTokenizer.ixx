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
 *       greedily, lowest priority index first, the leftmost of equal priority.
 *   4b. Rank path (Llama 3.x / TikToken): a pre-token held whole is one token;
 *       otherwise adjacent units merge lowest rank first, a token's rank being its
 *       ID. Used when no merge rules are present.
 *   4c. SentencePiece path (Gemma): space -> U+2581 with no split, so a segment between
 *       special tokens is one pre-token, UTF-8 character initial units with <0xNN> byte
 *       fallback, then the merge rules by rank. Selected when the vocabulary uses
 *       PreTokenizationMode::SentencePiece.
 *   5. Map final tokens to IDs; fall back to 0 on a miss.
 *
 * Decode pipeline:
 *   Concatenate token strings and reverse the byte encoding back to UTF-8.
 */

export module Data.BpeTokenizer;

import std;
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
         * for spaces, merged across the marks) with byte fallback.
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
         * @brief Load a Qwen 3.x tokenizer from the binary produced by Qwen/convert_tokenizer.py.
         *
         * Qwen uses GPT-2 style byte-level BPE with explicit merge ranks and Qwen's
         * own split pattern -- notably one token per digit.
         *
         * @param path Path to the Qwen tokenizer binary.
         * @return Shared tokenizer instance.
         * @throws std::runtime_error on I/O or format errors.
         */
        static std::shared_ptr<BpeTokenizer> loadQwen( const std::filesystem::path& path )
        {
            return std::make_shared<BpeTokenizer>( BpeVocabulary::loadQwen( path ) );
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
            warnIfNonAsciiUnderFallback( text );

            std::vector<TokenId> out;
            const auto& special_list = vocab_.getSpecialTokenList();

            if ( special_list.empty() )
            {
                // Fast path: no special tokens registered, skip pre-pass entirely.
                encodeSegment( text, out );
            }
            else
            {
                // Where each special token next occurs at or after `pos`, searched for again only once `pos` has
                // passed it: searching every token at every boundary rescans the rest of a long prompt once per
                // token it never contains.
                std::vector<std::size_t> next_found( special_list.size() );

                for ( std::size_t k = 0; k < special_list.size(); ++k )
                {
                    next_found[ k ] = text.find( special_list[ k ].first );
                }

                std::size_t pos = 0;

                while ( pos < text.size() )
                {
                    std::size_t next_special = text.size();

                    for ( std::size_t k = 0; k < special_list.size(); ++k )
                    {
                        if ( next_found[ k ] != std::string::npos && next_found[ k ] < pos )
                        {
                            next_found[ k ] = text.find( special_list[ k ].first, pos );
                        }

                        next_special = std::min( next_special, next_found[ k ] );
                    }

                    if ( next_special > pos )
                    {
                        encodeSegment( text.substr( pos, next_special - pos ), out );
                        pos = next_special;

                        continue;
                    }

                    // A special token starts here; the list is longest-first, so the first that does is the match.
                    for ( std::size_t k = 0; k < special_list.size(); ++k )
                    {
                        if ( next_found[ k ] == pos )
                        {
                            out.push_back( special_list[ k ].second );
                            pos += special_list[ k ].first.size();
                            break;
                        }
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

        std::size_t getVocabSize() const override
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
        bool                       ascii_fallback_active_ = false;

        /**
         * @brief Warn the first time text that the ASCII fallback cannot handle is encoded.
         *
         * The fallback is chosen at construction but only matters when a non-ASCII byte arrives,
         * so this is where the warning belongs: it then describes something that has just
         * happened rather than something that might. A session of English prompts stays silent.
         *
         * Once per process rather than per tokenizer or per call -- a standing limitation
         * repeated every turn reads as noise. The scan is a byte compare over one prompt and is
         * skipped entirely when the Unicode pattern compiled.
         */
        void warnIfNonAsciiUnderFallback( std::string_view text ) const
        {
            if ( !ascii_fallback_active_ )
                return;

            bool has_non_ascii = false;

            for ( const unsigned char byte : text )
            {
                if ( byte >= 0x80 )
                {
                    has_non_ascii = true;
                    break;
                }
            }

            if ( !has_non_ascii )
                return;

            static std::once_flag fallback_warning_flag;

            std::call_once( fallback_warning_flag, []
                {
                    Logging::Logger::warning(
                        "Non-ASCII text tokenized through the ASCII fallback: std::regex on this "
                        "toolchain has no Unicode property support, so this text WILL tokenize "
                        "differently from the HuggingFace reference. Permanent, not occasional. "
                        "Affects GPT-2, Llama 3 and Qwen; SentencePiece models (Gemma) do not use "
                        "this path." );
                } );
        }

        // ====================================================================
        // Initialization
        // ====================================================================

        /**
         * @brief Build the pre-tokenization regex from the vocabulary config.
         *
         * Attempts to compile the Unicode pattern first. If std::regex rejects it
         * (no standard std::regex supports \p{L} / \p{N} -- Unicode property escapes are an
         * ECMAScript 2018 feature, and the grammar C++ adopted predates them), falls back to the
         * ASCII-only approximation for the detected mode. Gpt2Regex, Llama3Regex and
         * Qwen3Regex each have a dedicated ASCII fallback; an unrecognised pattern
         * that fails compilation is treated as a hard error.
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
                else if ( mode == PreTokenizationMode::Qwen3Regex )
                {
                    fallback = QWEN3_PRETOKENIZATION_PATTERN_ASCII_FALLBACK;
                }

                if ( !fallback )
                {
                    throw std::runtime_error(
                        "BpeTokenizer: pre-tokenization pattern failed to compile "
                        "and no ASCII fallback is defined for this mode: " + pattern );
                }

                // Recorded, not announced. The fallback is selected on EVERY construction and on
                // EVERY platform -- observed 2026-08-16 in the published Linux container under
                // clang-21, not just under MSVC, because \p{...} is absent from the ECMAScript
                // grammar C++ adopted rather than from any one vendor's library -- but it only
                // CHANGES anything once non-ASCII text is actually tokenized.
                // Warning at construction told every user of an English prompt about a
                // divergence they will never meet, as the first line of a session. See
                // warnIfNonAsciiUnderFallback, which fires when the claim becomes true.
                ascii_fallback_active_ = true;

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
         * or to the rank path for TikToken-style vocabularies (Llama 3.x).
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
                encodeSegmentByRank( words, out );
            }
            else
            {
                encodeSegmentBpe( words, out );
            }
        }

        /**
         * @brief TikToken encode (Llama 3.x): a pre-token the vocabulary holds whole is one token; otherwise
         *        adjacent units merge by rank, where a token's rank is its id.
         *
         * This is tiktoken's byte-pair merge, and what HuggingFace's `ignore_merges` tokenizer computes from
         * the merge list the ranks were written out as. Falls back to ID 0 for an unrecognised unit.
         *
         * @param words Pre-tokenized segments from the regex pass.
         * @param out   Accumulator for output token IDs.
         */
        void encodeSegmentByRank( const std::vector<std::string>& words, std::vector<TokenId>& out )
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

                std::string whole;

                for ( const auto& unit : units )
                {
                    whole += unit;
                }

                if ( auto id = vocab_.tokenToId( whole ) )
                {
                    out.push_back( *id );
                    continue;
                }

                while ( units.size() > 1 )
                {
                    std::size_t best_index = units.size();
                    TokenId best_rank = std::numeric_limits<TokenId>::max();

                    for ( std::size_t index = 0; index + 1 < units.size(); ++index )
                    {
                        auto rank = vocab_.tokenToId( units[ index ] + units[ index + 1 ] );

                        if ( rank && *rank < best_rank )
                        {
                            best_rank = *rank;
                            best_index = index;
                        }
                    }

                    if ( best_index == units.size() )
                    {
                        break;
                    }

                    units[ best_index ] += units[ best_index + 1 ];
                    units.erase( units.begin() + static_cast<std::ptrdiff_t>( best_index ) + 1 );
                }

                for ( const auto& unit : units )
                {
                    auto id = vocab_.tokenToId( unit );
                    out.push_back( id ? *id : 0 );
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
         * @brief Greedily apply merge rules to a unit sequence, lowest priority first and, among equal
         *        priorities, leftmost first -- HuggingFace's BPE order.
         *
         * Shared by the GPT-2 byte-level BPE path and the Gemma SentencePiece BPE
         * path; only the construction of the initial unit sequence differs. Candidate
         * pairs wait in a priority queue and the units in a linked list, so a Gemma
         * segment -- a whole tool result between two special tokens -- merges in
         * O(n log n) rather than one rescan per merge.
         *
         * @param tokens In/out unit sequence; merged in place until no rule applies.
         */
        void applyMerges( std::vector<std::string>& tokens )
        {
            if ( tokens.size() < 2 )
            {
                return;
            }

            constexpr std::size_t kNone = std::numeric_limits<std::size_t>::max();
            const std::size_t count = tokens.size();

            std::vector<std::size_t> next( count );
            std::vector<std::size_t> previous( count );
            std::vector<bool> alive( count, true );

            for ( std::size_t i = 0; i < count; ++i )
            {
                next[ i ] = i + 1 < count ? i + 1 : kNone;
                previous[ i ] = i > 0 ? i - 1 : kNone;
            }

            // (priority, left unit's original index): positions keep their order, so the smallest index of a
            // priority is its leftmost pair.
            using Candidate = std::pair<std::size_t, std::size_t>;
            std::priority_queue<Candidate, std::vector<Candidate>, std::greater<Candidate>> queue;

            auto enqueue = [&]( std::size_t left )
                {
                    if ( left == kNone || next[ left ] == kNone )
                    {
                        return;
                    }

                    if ( auto priority = vocab_.getMergePriority( tokens[ left ], tokens[ next[ left ] ] ) )
                    {
                        queue.emplace( *priority, left );
                    }
                };

            for ( std::size_t i = 0; i < count; ++i )
            {
                enqueue( i );
            }

            while ( !queue.empty() )
            {
                const auto [priority, left] = queue.top();
                queue.pop();

                const std::size_t right = next[ left ];

                if ( !alive[ left ] || right == kNone )
                {
                    continue;
                }

                // A priority names exactly one pair, so a candidate whose pair has since changed is stale.
                const auto current = vocab_.getMergePriority( tokens[ left ], tokens[ right ] );

                if ( !current || *current != priority )
                {
                    continue;
                }

                tokens[ left ] += tokens[ right ];
                tokens[ right ].clear();
                alive[ right ] = false;

                next[ left ] = next[ right ];

                if ( next[ right ] != kNone )
                {
                    previous[ next[ right ] ] = left;
                }

                enqueue( previous[ left ] );
                enqueue( left );
            }

            // Unit 0 is never merged away: a merge removes the right unit of its pair.
            std::vector<std::string> merged;

            for ( std::size_t i = 0; i != kNone; i = next[ i ] )
            {
                merged.push_back( std::move( tokens[ i ] ) );
            }

            tokens = std::move( merged );
        }

        /**
         * @brief SentencePiece BPE encode (Gemma).
         *
         * Initial units are UTF-8 *characters* (not bytes): a character that is a
         * vocabulary piece is used directly; an unknown character decomposes to its
         * SentencePiece byte-fallback pieces (<0xNN> per byte). The shared merge
         * loop then builds up the final pieces by rank.
         *
         * @param words Pre-tokens from preTokenizeSentencePiece (one per segment).
         * @param out   Accumulator for output token IDs.
         */
        void encodeSegmentSentencePiece( const std::vector<std::string>& words, std::vector<TokenId>& out )
        {
            for ( const auto& word : words )
            {
                std::vector<std::string> tokens;
                tokens.reserve( word.size() );

                std::size_t i = 0;

                while ( i < word.size() )
                {
                    const std::size_t char_len = utf8CharLength( static_cast<unsigned char>( word[ i ] ) );
                    const std::string ch = word.substr( i, char_len );

                    if ( vocab_.containsToken( ch ) )
                    {
                        tokens.push_back( ch );
                    }
                    else
                    {
                        // Byte fallback: emit one <0xNN> piece per raw byte.
                        for ( std::size_t b = 0; b < char_len && (i + b) < word.size(); ++b )
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
         * @brief SentencePiece pre-tokenization (Gemma): the whole segment as one pre-token.
         *
         * Replaces ASCII spaces with the metaspace mark U+2581 ('lower one eighth
         * block') and does not split, which is what Gemma 4's tokenizer.json computes:
         * its normalizer makes the replacement, and its split on ' ' then has nothing
         * to match. Merges therefore cross the marks, so a run of spaces -- code's
         * indentation -- becomes one piece (four spaces are id 140) where a split
         * before every mark would give one piece per space. No leading mark is
         * prepended (Gemma's tokenizer adds no prefix space: "The capital" ->
         * ["The", "_capital"]).
         *
         * @param text Plain text segment (no special tokens).
         * @return One pre-token, the segment with its spaces marked.
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

            return { std::move( marked ) };
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
            std::size_t i = 0;

            while ( i < token.size() )
            {
                const std::size_t char_len = utf8CharLength( static_cast<unsigned char>( token[ i ] ) );
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
            std::size_t i = 0;

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

        static std::size_t utf8CharLength( unsigned char first_byte )
        {
            if ( (first_byte & 0x80) == 0x00 ) return 1;
            if ( (first_byte & 0xE0) == 0xC0 ) return 2;
            if ( (first_byte & 0xF0) == 0xE0 ) return 3;
            if ( (first_byte & 0xF8) == 0xF0 ) return 4;
            return 1;
        }
    };
}
