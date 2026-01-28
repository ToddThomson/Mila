/**
 * @file BPEVocabulary.ixx
 * @brief Byte-Pair Encoding vocabulary implementation that can be built from text.
 *
 * Implements a simple, deterministic BPE learning loop suitable for small
 * corpora and preprocessing tooling. The implementation is intentionally
 * minimal and safe for repository-level tooling (not optimized for large-scale
 * production BPE training).
 */

module;
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <optional>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <cstdint>

export module Data.BpeVocabulary;

import Data.BpeSpecialTokens;
import Data.BpeTrainerConfig;
import Data.TokenizerVocabulary;

namespace Mila::Data
{
    namespace fs = std::filesystem;
    using Mila::Dnn::Data::TokenizerVocabulary;

    /**
     * @brief Byte Pair Encoding (BPE) vocabulary implementation.
     *
     * This implementation follows modern BPE practices used in GPT-2, GPT-3, and similar models:
     * - Byte-level BPE: operates on UTF-8 bytes, not characters
     * - Pre-tokenization: splits on whitespace and punctuation
     * - Merge rules: stores and applies learned merge operations
     * - Special tokens: supports configurable special tokens (PAD, UNK, BOS, EOS, etc.)
     *
     * Key differences from character-level tokenization:
     * - Can represent any Unicode text without unknown tokens at byte level
     * - Learns subword units that balance vocabulary size and sequence length
     * - More efficient than character-level for most languages
     *
     * Algorithm overview:
     * 1. Initialize vocabulary with all 256 possible byte values (byte-level BPE)
     * 2. Add special tokens if requested
     * 3. Pre-tokenize text into words/subwords
     * 4. Iteratively merge the most frequent adjacent byte pair
     * 5. Continue until target vocabulary size is reached
     *
     * Thread safety: Not thread-safe. External synchronization required for concurrent access.
     */
    export class BpeVocabulary : public TokenizerVocabulary
    {
    public:
        
        // ========================================================================
        // Loading
        // ========================================================================

        /**
         * @brief Load vocabulary from disk.
         *
         * Reads binary format written by save(). Validates version compatibility.
         *
         * @param path Input file path.
         *
         * @throws std::runtime_error on I/O errors or format incompatibility.
         */
        static BpeVocabulary load( const std::filesystem::path& path )
        {
            std::ifstream file( path, std::ios::binary );
            
            if ( !file ) {
                throw std::runtime_error( "Cannot open vocabulary file: " + path.string() );
            }

            BpeVocabulary vocab;

            vocab.token_to_id_.clear();
            vocab.id_to_token_.clear();
            vocab.merges_.clear();
            vocab.special_token_ids_.clear();

            // Version check
            uint32_t version = 0;
            file.read( reinterpret_cast<char*>(&version), sizeof( version ) );
            if ( version != 1 ) {
                throw std::runtime_error( "Unsupported vocabulary file version: " + std::to_string( version ) );
            }

            // Vocabulary size
            uint32_t vocab_size = 0;
            file.read( reinterpret_cast<char*>(&vocab_size), sizeof( vocab_size ) );

            // Special tokens
            uint32_t num_special = 0;
            file.read( reinterpret_cast<char*>(&num_special), sizeof( num_special ) );

            for ( uint32_t i = 0; i < num_special; ++i ) {
                uint8_t type_byte = 0;
                file.read( reinterpret_cast<char*>( &type_byte ), sizeof( type_byte ) );

                uint32_t token_id = 0;
                file.read( reinterpret_cast<char*>( &token_id ), sizeof( token_id ) );

                uint32_t len = 0;
                file.read( reinterpret_cast<char*>( &len ), sizeof( len ) );

                std::string token_str;
                if ( len > 0 ) {
                    token_str.resize( len );
                    file.read( token_str.data(), len );
                }

                vocab.special_token_ids_[ static_cast<char>(type_byte) ] = token_id;
            }

            // Merge rules
            uint32_t num_merges = 0;
            file.read( reinterpret_cast<char*>(&num_merges), sizeof( num_merges ) );

            vocab.merges_.reserve( num_merges );
            for ( uint32_t i = 0; i < num_merges; ++i ) {
                uint32_t left_len = 0;
                file.read( reinterpret_cast<char*>( &left_len ), sizeof( left_len ) );

                std::string left;
                if ( left_len > 0 ) {
                    left.resize( left_len );
                    file.read( left.data(), left_len );
                }

                uint32_t right_len = 0;
                file.read( reinterpret_cast<char*>(&right_len), sizeof( right_len ) );

                std::string right;
                if ( right_len > 0 ) {
                    right.resize( right_len );
                    file.read( right.data(), right_len );
                }

                vocab.merges_.emplace_back( left, right );
            }

            // Token strings
            vocab.id_to_token_.resize( vocab_size );
            for ( uint32_t i = 0; i < vocab_size; ++i ) {
                uint32_t len = 0;
                file.read( reinterpret_cast<char*>( &len ), sizeof( len ) );

                std::string token;
                if ( len > 0 ) {
                    token.resize( len );
                    file.read( token.data(), len );
                }

                vocab.id_to_token_[ i ] = std::move( token );
                vocab.token_to_id_[ vocab.id_to_token_[ i ] ] = i;
            }

            return vocab;
        }

        // Load pre-trained vocabularies from external sources
        static BpeVocabulary loadGpt2(
            const std::filesystem::path& tokenizerPath
        );

        static BpeVocabulary loadLlama(
            const std::filesystem::path& modelPath
        );

        static BpeVocabulary loadMistral(
            const std::filesystem::path& vocabPath,
            const std::filesystem::path& mergesPath
        );

        BpeVocabulary() = default;

        /**
         * @brief Build a BPE vocabulary from raw text corpus using trainer config.
         *
         * The trainer config controls:
         * - target vocabulary size (getVocabSize())
         * - special tokens (getSpecialTokens())
         * - minimum pair frequency (getMinFrequency())
         * - byte-level vs char-level (isByteLevel())
         * - max merges (getMaxMerges())
         *
         * Returns the final vocabulary size.
         */
        size_t buildFromText( const std::string& text, const BpeTrainerConfig& config )
        {
            token_to_id_.clear();
            id_to_token_.clear();
            merges_.clear();
            special_token_ids_.clear();

            uint32_t current_id = 0;

            const auto& special_tokens = config.getSpecialTokens();
            const size_t target_vocab_size = config.getVocabSize();
            const size_t min_frequency = config.getMinFrequency();
            const bool byte_level = config.isByteLevel();
            const size_t max_merges = config.getMaxMerges();

            // Validate special token configuration (throws on invalid config)
            if ( special_tokens.enabled ) {
                special_tokens.validate();
            }

            // Step 1: Initialize base vocabulary
            if ( byte_level ) {
                // Add all 256 byte values as base vocabulary (byte-level BPE)
                for ( int i = 0; i < 256; ++i ) {
                    std::string byte_token( 1, static_cast<char>( i ) );
                    id_to_token_.push_back( byte_token );
                    token_to_id_[ byte_token ] = current_id++;
                }
            }
            else {
                // Character-level base vocabulary derived from the input text
                std::unordered_set<unsigned char> unique_chars;
                for ( unsigned char cu : text ) {
                    unique_chars.insert( cu );
                }

                // Deterministic ordering
                std::vector<unsigned char> sorted_chars( unique_chars.begin(), unique_chars.end() );
                std::sort( sorted_chars.begin(), sorted_chars.end() );

                for ( unsigned char cu : sorted_chars ) {
                    std::string tok( 1, static_cast<char>( cu ) );
                    id_to_token_.push_back( tok );
                    token_to_id_[ tok ] = current_id++;
                }
            }

            // Step 2: Add special tokens (BPE tokens are full strings; the token strings matter)
            if ( special_tokens.enabled ) {
                // Core tokens
                addSpecialToken( special_tokens.pad_token, current_id++, "pad" );
                addSpecialToken( special_tokens.unk_token, current_id++, "unk" );
                addSpecialToken( special_tokens.bos_token, current_id++, "bos" );
                addSpecialToken( special_tokens.eos_token, current_id++, "eos" );

                // Extended tokens (optional)
                if ( !special_tokens.mask_token.empty() ) {
                    addSpecialToken( special_tokens.mask_token, current_id++, "mask" );
                }
                if ( !special_tokens.sep_token.empty() ) {
                    addSpecialToken( special_tokens.sep_token, current_id++, "sep" );
                }
                if ( !special_tokens.cls_token.empty() ) {
                    addSpecialToken( special_tokens.cls_token, current_id++, "cls" );
                }
            }

            // If target is zero or already satisfied, return current size
            if ( target_vocab_size == 0 || current_id >= static_cast<uint32_t>( target_vocab_size ) ) {
                return id_to_token_.size();
            }

            // Step 3: Pre-tokenize text into words
            auto words = preTokenize( text );

            // Step 4: Convert words into token sequences (initial tokens are bytes or chars)
            std::vector<std::vector<std::string>> corpus_tokens;
            corpus_tokens.reserve( words.size() );

            for ( const auto& word : words ) {
                std::vector<std::string> tokens;
                tokens.reserve( word.size() );

                if ( byte_level ) {
                    for ( unsigned char byte : word ) {
                        tokens.push_back( std::string( 1, static_cast<char>( byte ) ) );
                    }
                }
                else {
                    // character-level: treat each char (byte) as token for now
                    for ( unsigned char cu : word ) {
                        tokens.push_back( std::string( 1, static_cast<char>( cu ) ) );
                    }
                }

                if ( !tokens.empty() ) {
                    corpus_tokens.push_back( std::move( tokens ) );
                }
            }

            // Step 5: BPE merge loop with frequency and max merge limits
            size_t merges_performed = 0;
            while ( current_id < static_cast<uint32_t>( target_vocab_size ) ) {
                auto pair_counts = countPairs( corpus_tokens );

                if ( pair_counts.empty() ) {
                    break;  // No more pairs to merge
                }

                auto best = getMostFrequentPair( pair_counts );

                const auto& best_pair = best.first;
                size_t best_count = best.second;

                if ( best_count < min_frequency ) {
                    break;  // No pair meets the minimum frequency threshold
                }

                // Create merged token string
                std::string merged = best_pair.first + best_pair.second;

                // Skip if token already exists
                if ( token_to_id_.find( merged ) != token_to_id_.end() ) {
                    // Remove this pair and continue; but to keep simplicity, break if duplicated
                    break;
                }

                // Apply merge to corpus
                applyMerge( corpus_tokens, best_pair.first, best_pair.second, merged );

                // Add merged token to vocabulary and record the merge rule
                id_to_token_.push_back( merged );
                token_to_id_[ merged ] = current_id;
                merges_.push_back( best_pair );

                ++current_id;
                ++merges_performed;

                if ( max_merges > 0 && merges_performed >= max_merges ) {
                    break;
                }
            }

            // Final vocabulary size
            
            return id_to_token_.size();
        }

        /**
         * @brief Serialize vocabulary to disk.
         *
         * File format (binary):
         * - [uint32_t] Version number (for future compatibility)
         * - [uint32_t] Vocabulary size
         * - [uint32_t] Number of special tokens
         * - For each special token:
         *   - [uint8_t] Type (0=pad, 1=unk, 2=bos, 3=eos, etc.)
         *   - [uint32_t] Token ID
         *   - [uint32_t] Token string length
         *   - [bytes] Token string data
         * - [uint32_t] Number of merge rules
         * - For each merge rule:
         *   - [uint32_t] Left token length
         *   - [bytes] Left token data
         *   - [uint32_t] Right token length
         *   - [bytes] Right token data
         * - For each vocabulary token:
         *   - [uint32_t] Token string length
         *   - [bytes] Token string data
         *
         * @param path Output file path. Parent directory must exist.
         *
         * @throws std::runtime_error on I/O errors.
         */
        void save( const fs::path& path ) const override
        {
            std::ofstream file( path, std::ios::binary );
            if ( !file ) {
                throw std::runtime_error( "Cannot open vocabulary file for writing: " + path.string() );
            }

            // Version for future compatibility
            uint32_t version = 1;
            file.write( reinterpret_cast<const char*>(&version), sizeof( version ) );

            // Vocabulary size
            uint32_t vocab_size = static_cast<uint32_t>(id_to_token_.size());
            file.write( reinterpret_cast<const char*>(&vocab_size), sizeof( vocab_size ) );

            // Special tokens
            uint32_t num_special = static_cast<uint32_t>(special_token_ids_.size());
            file.write( reinterpret_cast<const char*>(&num_special), sizeof( num_special ) );

            for ( const auto& [type, id] : special_token_ids_ ) {
                uint8_t type_byte = static_cast<uint8_t>(type);
                file.write( reinterpret_cast<const char*>(&type_byte), sizeof( type_byte ) );

                uint32_t token_id = static_cast<uint32_t>(id);
                file.write( reinterpret_cast<const char*>(&token_id), sizeof( token_id ) );

                const std::string& token_str = id_to_token_[ id ];
                uint32_t len = static_cast<uint32_t>(token_str.size());
                file.write( reinterpret_cast<const char*>(&len), sizeof( len ) );
                file.write( token_str.data(), len );
            }

            // Merge rules
            uint32_t num_merges = static_cast<uint32_t>(merges_.size());
            file.write( reinterpret_cast<const char*>(&num_merges), sizeof( num_merges ) );

            for ( const auto& [left, right] : merges_ ) {
                uint32_t left_len = static_cast<uint32_t>(left.size());
                file.write( reinterpret_cast<const char*>(&left_len), sizeof( left_len ) );
                file.write( left.data(), left_len );

                uint32_t right_len = static_cast<uint32_t>(right.size());
                file.write( reinterpret_cast<const char*>(&right_len), sizeof( right_len ) );
                file.write( right.data(), right_len );
            }

            // Token strings
            for ( const auto& token : id_to_token_ ) {
                uint32_t len = static_cast<uint32_t>(token.size());
                file.write( reinterpret_cast<const char*>(&len), sizeof( len ) );
                if ( len > 0 ) {
                    file.write( token.data(), len );
                }
            }

            if ( !file ) {
                throw std::runtime_error( "Error writing vocabulary file: " + path.string() );
            }
        }

        size_t getSize() const override
        {
            return id_to_token_.size();
        }

        std::optional<uint32_t> tokenToId( const std::string& token ) const override
        {
            auto it = token_to_id_.find( token );
            if ( it != token_to_id_.end() ) {
                return it->second;
            }

            // Return UNK token if available
            auto unk_it = special_token_ids_.find( 'u' );  // 'u' for unk
            if ( unk_it != special_token_ids_.end() ) {
                return unk_it->second;
            }

            return std::nullopt;
        }

        std::optional<std::string> idToToken( uint32_t id ) const override
        {
            if ( id < id_to_token_.size() ) {
                return id_to_token_[ id ];
            }
            return std::nullopt;
        }

        /**
         * @brief Get merge rules learned during training.
         *
         * Merge rules define the order in which byte pairs should be merged
         * during encoding. Rules are ordered by training priority (most
         * frequent pairs merged first become earlier rules).
         *
         * @return const std::vector<std::pair<std::string, std::string>>&
         *         Ordered list of (left, right) token pairs to merge.
         */
        const std::vector<std::pair<std::string, std::string>>& getMergeRules() const
        {
            return merges_;
        }

        /**
         * @brief Get special token ID by type.
         *
         * @param type Special token type ('p'=pad, 'u'=unk, 'b'=bos, 'e'=eos)
         * @return std::optional<uint32_t> Token ID if present, nullopt otherwise.
         */
        std::optional<uint32_t> getSpecialTokenId( char type ) const
        {
            auto it = special_token_ids_.find( type );
            if ( it != special_token_ids_.end() ) {
                return it->second;
            }
            return std::nullopt;
        }

        std::optional<size_t> getMergePriority( const std::string& left, const std::string& right ) const
        {
            // Linear search - can optimize with a map later
            for ( size_t i = 0; i < merges_.size(); ++i ) {
                if ( merges_[ i ].first == left && merges_[ i ].second == right ) {
                    return i;  // Index is the priority
                }
            }
            return std::nullopt;  // This pair doesn't have a merge rule
        }

        // TODO: Alternative implementation using a map for faster lookup

        /*std::optional<size_t> getMergePriority( const std::string& left, const std::string& right ) const
        {
            auto it = merge_map_.find( { left, right } );
            if ( it != merge_map_.end() ) {
                return it->second;
            }
            return std::nullopt;
        }*/

        // Add this to BpeVocabulary or as a utility function
        static const std::unordered_map<unsigned char, std::string>& getByteEncoder() {
            static std::unordered_map<unsigned char, std::string> encoder = []() {
                std::unordered_map<unsigned char, std::string> enc;

                // Build the bytes-to-unicode mapping (matches GPT-2's bytes_to_unicode)
                std::vector<int> bs;
                for ( int i = int( '!' ); i <= int( '~' ); ++i ) bs.push_back( i );
                for ( int i = 0xA1; i <= 0xAC; ++i ) bs.push_back( i );
                for ( int i = 0xAE; i <= 0xFF; ++i ) bs.push_back( i );

                std::vector<int> cs = bs;
                int n = 0;
                for ( int b = 0; b < 256; ++b ) {
                    if ( std::find( bs.begin(), bs.end(), b ) == bs.end() ) {
                        bs.push_back( b );
                        cs.push_back( 256 + n );
                        ++n;
                    }
                }

                for ( size_t i = 0; i < bs.size(); ++i ) {
                    char byte = static_cast<char>( bs[ i ] );
                    char32_t unicode_point = static_cast<char32_t>( cs[ i ] );

                    // Convert unicode point to UTF-8 string
                    std::string utf8_str;
                    if ( unicode_point < 0x80 ) {
                        utf8_str += static_cast<char>( unicode_point );
                    }
                    else if ( unicode_point < 0x800 ) {
                        utf8_str += static_cast<char>( 0xC0 | (unicode_point >> 6) );
                        utf8_str += static_cast<char>( 0x80 | (unicode_point & 0x3F) );
                    }
                    else {
                        utf8_str += static_cast<char>( 0xE0 | (unicode_point >> 12) );
                        utf8_str += static_cast<char>( 0x80 | ((unicode_point >> 6) & 0x3F) );
                        utf8_str += static_cast<char>( 0x80 | (unicode_point & 0x3F) );
                    }

                    enc[ static_cast<unsigned char>( byte ) ] = utf8_str;
                }
                return enc;
                }();
            return encoder;
        }

        // Add this alongside getByteEncoder()
        static const std::unordered_map<std::string, unsigned char>& getByteDecoder() {
            static std::unordered_map<std::string, unsigned char> decoder = []() {
                std::unordered_map<std::string, unsigned char> dec;
                const auto& encoder = getByteEncoder();
                for ( const auto& [byte, utf8_str] : encoder ) {
                    dec[ utf8_str ] = byte;
                }
                return dec;
                }();
            return decoder;
        }

        bool isByteLevel() const {
            return byte_level_;
        }

    private:

        /**
         * @brief Hash function for string pairs (for unordered_map).
         */
        struct PairHash {
            size_t operator()( const std::pair<std::string, std::string>& p ) const {
                return std::hash<std::string>{}(p.first) ^
                    (std::hash<std::string>{}(p.second) << 1);
            }
        };

        /**
         * @brief Pre-tokenize text into words/units for BPE processing.
         *
         * Current implementation: Simple whitespace splitting.
         *
         * TODO: Implement GPT-2 style regex pattern:
         * - Splits on whitespace and punctuation
         * - Preserves contractions (don't, can't)
         * - Handles numbers specially
         * Pattern: r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
         */
        std::vector<std::string> preTokenize( const std::string& text ) const
        {
            std::vector<std::string> words;
            std::istringstream stream( text );
            std::string word;

            while ( stream >> word ) {
                if ( !word.empty() ) {
                    words.push_back( word );
                }
            }

            return words;
        }

        /**
         * @brief Count all adjacent token pairs in the corpus.
         */
        std::unordered_map<std::pair<std::string, std::string>, size_t, PairHash>
            countPairs( const std::vector<std::vector<std::string>>& corpus ) const
        {
            std::unordered_map<std::pair<std::string, std::string>, size_t, PairHash> counts;

            for ( const auto& tokens : corpus ) {
                for ( size_t i = 0; i + 1 < tokens.size(); ++i ) {
                    counts[ {tokens[ i ], tokens[ i + 1 ]} ]++;
                }
            }

            return counts;
        }

        /**
         * @brief Find the most frequent pair in the counts map.
         */
        std::pair<std::pair<std::string, std::string>, size_t>
            getMostFrequentPair( const std::unordered_map<std::pair<std::string, std::string>,
                size_t, PairHash>& counts ) const
        {
            if ( counts.empty() ) {
                return std::make_pair( std::make_pair( std::string(), std::string() ), size_t( 0 ) );
                //return { {{"", ""}, 0} };
            }

            auto best = std::max_element(
                counts.begin(), counts.end(),
                []( const auto& a, const auto& b ) { return a.second < b.second; }
            );

            return *best;
        }

        /**
         * @brief Apply a merge operation to the entire corpus.
         */
        void applyMerge( std::vector<std::vector<std::string>>& corpus,
            const std::string& left,
            const std::string& right,
            const std::string& merged ) const
        {
            for ( auto& tokens : corpus ) {
                std::vector<std::string> new_tokens;
                new_tokens.reserve( tokens.size() );

                size_t i = 0;
                while ( i < tokens.size() ) {
                    if ( i + 1 < tokens.size() &&
                        tokens[ i ] == left &&
                        tokens[ i + 1 ] == right ) {
                        new_tokens.push_back( merged );
                        i += 2;
                    }
                    else {
                        new_tokens.push_back( tokens[ i ] );
                        ++i;
                    }
                }

                tokens = std::move( new_tokens );
            }
        }

        /**
         * @brief Add a special token to the vocabulary.
         */
        void addSpecialToken( const std::string& token, uint32_t id, const std::string& type )
        {
            id_to_token_.push_back( token );
            token_to_id_[ token ] = id;

            // Map type to char: pad='p', unk='u', bos='b', eos='e'
            char type_char = type.empty() ? '?' : type[ 0 ];
            special_token_ids_[ type_char ] = id;
        }

        bool byte_level_ = false;

        std::unordered_map<std::string, uint32_t> token_to_id_;
        std::vector<std::string> id_to_token_;
        std::vector<std::pair<std::string, std::string>> merges_;
        std::unordered_map<char, uint32_t> special_token_ids_;  // type -> id
    };

    // ------------------------------------------------------------------------
    // Basic external-format loaders (stubs)
    //
    // These provide definitions for pre-trained loader entry points that are
    // referenced by BpeTokenizer::load* helper functions. The current project
    // uses a compact binary `save()`/`load()` format; if callers provide the
    // binary file written by `save()` we forward to `load()`. For other
    // external formats (GPT-2/LLAMA/Mistral) a proper parser should be
    // implemented here in the future. For now we raise a clear runtime error
    // explaining the limitation.
    // ------------------------------------------------------------------------

    BpeVocabulary BpeVocabulary::loadGpt2( const std::filesystem::path& tokenizerPath )
    {
        // Attempt to read a converter-produced binary produced by
        // convert_gpt2_tokenizer.py (single output file).
        std::ifstream file( tokenizerPath, std::ios::binary );

        if ( !file ) {
            throw std::runtime_error( "Cannot open GPT-2 tokenizer file: " + tokenizerPath.string() );
        }

        auto read_u32 = [&]( uint32_t &out ) {
            file.read( reinterpret_cast<char*>( &out ), sizeof( out ) );
            if ( !file ) {
                throw std::runtime_error( "Unexpected EOF or read error while parsing GPT-2 converter file: " + tokenizerPath.string() );
            }
        };

        BpeVocabulary vocab;
        vocab.token_to_id_.clear();
        vocab.id_to_token_.clear();
        vocab.merges_.clear();
        vocab.special_token_ids_.clear();

        // File format produced by convert_gpt2_tokenizer.py:
        // [uint32] vocab_size
        // [uint32] num_merges
        // For each vocab entry:
        //   [uint32] token_length
        //   [bytes] token_bytes (utf-8)
        //   [uint32] token_id
        // For each merge:
        //   [uint32] token1_length
        //   [bytes] token1
        //   [uint32] token2_length
        //   [bytes] token2
        // Special tokens flags:
        //   [uint32] has_eos (0/1)
        //   [uint32] eos_id (if has_eos)
        //   [uint32] has_bos (0/1)
        //   [uint32] bos_id (if has_bos)
        //   [uint32] has_pad (0/1)
        //   [uint32] pad_id (if has_pad)

        uint32_t vocab_size = 0;
        read_u32( vocab_size );

        uint32_t num_merges = 0;
        read_u32( num_merges );

        // Read vocabulary entries (token + id)
        vocab.id_to_token_.resize( vocab_size );
        for ( uint32_t i = 0; i < vocab_size; ++i ) {
            uint32_t len = 0;
            read_u32( len );

            std::string token;
            if ( len > 0 ) {
                token.resize( len );
                file.read( token.data(), static_cast<std::streamsize>( len ) );
                if ( !file ) {
                    throw std::runtime_error( "Failed reading token bytes from GPT-2 converter file: " + tokenizerPath.string() );
                }
            }

            uint32_t token_id = 0;
            read_u32( token_id );

            if ( token_id >= vocab_size ) {
                throw std::runtime_error( "Invalid token id in GPT-2 converter file: " + std::to_string( token_id ) );
            }

            vocab.id_to_token_[ token_id ] = token;
            vocab.token_to_id_[ token ] = token_id;
        }

        // Read merges (ordered by rank)
        vocab.merges_.reserve( num_merges );
        for ( uint32_t i = 0; i < num_merges; ++i ) {
            uint32_t llen = 0;
            read_u32( llen );

            std::string left;
            if ( llen > 0 ) {
                left.resize( llen );
                file.read( left.data(), static_cast<std::streamsize>( llen ) );
                if ( !file ) {
                    throw std::runtime_error( "Failed reading merge left token from GPT-2 converter file: " + tokenizerPath.string() );
                }
            }

            uint32_t rlen = 0;
            read_u32( rlen );

            std::string right;
            if ( rlen > 0 ) {
                right.resize( rlen );
                file.read( right.data(), static_cast<std::streamsize>( rlen ) );
                if ( !file ) {
                    throw std::runtime_error( "Failed reading merge right token from GPT-2 converter file: " + tokenizerPath.string() );
                }
            }

            vocab.merges_.emplace_back( std::move( left ), std::move( right ) );
        }

        // Read special token flags and IDs (if present)
        uint32_t has_eos = 0;
        read_u32( has_eos );
        if ( has_eos ) {
            uint32_t eos_id = 0;
            read_u32( eos_id );
            vocab.special_token_ids_[ 'e' ] = eos_id; // 'e' = eos
        }

        uint32_t has_bos = 0;
        read_u32( has_bos );
        if ( has_bos ) {
            uint32_t bos_id = 0;
            read_u32( bos_id );
            vocab.special_token_ids_[ 'b' ] = bos_id; // 'b' = bos
        }

        uint32_t has_pad = 0;
        read_u32( has_pad );
        if ( has_pad ) {
            uint32_t pad_id = 0;
            read_u32( pad_id );
            vocab.special_token_ids_[ 'p' ] = pad_id; // 'p' = pad
        }

        vocab.byte_level_ = true;  // GPT-2 is always byte-level!

        return vocab;
    }

    BpeVocabulary BpeVocabulary::loadLlama( const std::filesystem::path& modelPath )
    {
        try {
            return load( modelPath );
        }
        catch ( const std::exception& e ) {
            (void)e;
            throw std::runtime_error(
                "BpeVocabulary::loadLlama not implemented for model formats; "
                "provide a binary vocabulary produced by BpeVocabulary::save(), or implement LLAMA parsing."
            );
        }
    }

    BpeVocabulary BpeVocabulary::loadMistral(
        const std::filesystem::path& vocabPath,
        const std::filesystem::path& mergesPath
    )
    {
        try {
            return load( vocabPath );
        }
        catch ( const std::exception& e ) {
            (void)e;
            throw std::runtime_error(
                "BpeVocabulary::loadMistral not implemented for non-binary vocab files. "
                "Provide a binary vocabulary produced by BpeVocabulary::save(), or implement Mistral parsing."
            );
        }
    }
}