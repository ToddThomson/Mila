/**
 * @file Gpt4BpeVocabulary.ixx
 * @brief GPT-4 style BPE vocabulary for Llama 3.x models.
 *
 * Mirrors BpeVocabulary from the BpeTokenizer module but targets the
 * GPT-4 / Llama 3.x BPE family. Key differences:
 *
 *  - special_token_ids_ is keyed on std::string (token string) rather than
 *    char (type code), enabling the encode pre-pass in Gpt4BpeTokenizer to
 *    match Llama 3.x's large special token set by string before BPE runs.
 *
 *  - loadLlama32() reads the binary format produced by convert_llama_tokenizer.py.
 *    The format is identical to the GPT-2 binary except for the extra UNK field
 *    and the absence of BPE merges (Llama 3.x uses a pure vocabulary lookup
 *    after pre-tokenization; merges are encoded implicitly in the vocab).
 *
 *  - Training from scratch is intentionally not supported (alpha.2 scope).
 *    Only load() and loadLlama32() factory methods are provided.
 *
 *  - getByteEncoder() / getByteDecoder() are identical to BpeVocabulary
 *    (GPT-2 style byte mapping is shared across both BPE families).
 */

module;
#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>
#include <optional>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <cstdint>
#include <iostream>
#include <iomanip>

export module Data.Tokenizers.Bpe.Gpt4Vocabulary;

import Data.SpecialTokens;
import Data.Tokenizer;
import Data.TokenizerVocabulary;
import Data.Tokenizers.Bpe.Gpt4VocabularyConfig;
import Data.FileHeader;
import Serialization.Metadata;

namespace Mila::Data
{
    namespace fs = std::filesystem;
    using Mila::Dnn::Data::TokenizerVocabulary;
    using Mila::Dnn::Data::TokenId;
    using Mila::Dnn::Serialization::SerializationMetadata;

    /**
     * @brief GPT-4 style BPE vocabulary for Llama 3.x models.
     *
     * Immutable after construction. Thread-safe for concurrent reads.
     */
    export class Gpt4Vocabulary : public TokenizerVocabulary
    {
    public:

        // ====================================================================
        // Factory Methods
        // ====================================================================

        /**
         * @brief Load vocabulary from Mila binary format.
         *
         * Reads a Gpt4Vocabulary file written by save().
         *
         * @param path Input file path.
         * @return Loaded Gpt4BpeVocabulary instance.
         * @throws std::runtime_error on I/O errors or format mismatch.
         */
        static Gpt4Vocabulary load( const fs::path& path )
        {
            std::ifstream file( path, std::ios::binary );
            if ( !file )
            {
                throw std::runtime_error( "Cannot open vocabulary file: " + path.string() );
            }

            MilaFileHeader header = MilaFileHeader::read( file );

            if ( header.getFileType() != MilaFileType::Gpt4BpeVocabulary )
            {
                throw std::runtime_error(
                    "File is not a Gpt4BpeVocabulary (got " +
                    std::string( toString( header.getFileType() ) ) + "): " + path.string() );
            }

            Gpt4VocabularyConfig config;
            config.fromMetadata( header.getMetadata() );

            Gpt4Vocabulary vocab( config );
            vocab.loadContent( file );

            return vocab;
        }

        /**
         * @brief Load Llama 3.2 vocabulary from the binary format produced by
         *        convert_llama_tokenizer.py.
         *
         * Binary format (matches convert_llama_tokenizer.py output):
         *   Header:
         *     - vocab_size       (uint32)
         *     - use_byte_fallback (uint8)
         *   Vocabulary:
         *     - For each token:
         *       - token_length   (uint32)
         *       - token_bytes    (utf-8)
         *       - score          (float32)  -- always 0.0 for BPE, ignored
         *       - token_id       (uint32)
         *   Special tokens:
         *     - has_bos          (uint32)
         *     - bos_token_id     (uint32, if has_bos)   -- 128000
         *     - has_eos          (uint32)
         *     - eos_token_id     (uint32, if has_eos)   -- 128001
         *     - has_pad          (uint32)
         *     - pad_token_id     (uint32, if has_pad)
         *     - has_unk          (uint32)
         *     - unk_token_id     (uint32, if has_unk)   -- absent for Llama 3.2
         *
         * @param path Path to converted Llama 3.2 tokenizer binary.
         * @return Loaded Gpt4BpeVocabulary instance.
         * @throws std::runtime_error on I/O or format errors.
         */
        static Gpt4Vocabulary loadLlama32( const fs::path& path )
        {
            std::ifstream file( path, std::ios::binary );
            if ( !file )
            {
                throw std::runtime_error( "Cannot open Llama 3.2 tokenizer file: " + path.string() );
            }

            auto read_u32 = [&]( uint32_t& out )
                {
                    file.read( reinterpret_cast<char*>(&out), sizeof( out ) );
                    if ( !file )
                    {
                        throw std::runtime_error(
                            "Unexpected EOF reading Llama 3.2 tokenizer: " + path.string() );
                    }
                };

            // --- Header ---
            uint32_t vocab_size = 0;
            read_u32( vocab_size );

            uint8_t use_byte_fallback = 0;
            file.read( reinterpret_cast<char*>(&use_byte_fallback), sizeof( use_byte_fallback ) );
            if ( !file )
            {
                throw std::runtime_error( "Failed reading use_byte_fallback" );
            }

            Gpt4VocabularyConfig config = Gpt4VocabularyConfig()
                .withVocabSize( vocab_size )
                .withByteLevel( true )
                .withPreTokenization( PreTokenizationMode::Llama3Regex )
                .withPreTokenizationPattern( LLAMA3_PRETOKENIZATION_PATTERN )
                .withSpecialTokens( SpecialTokens::llamaStyle() );

            Gpt4Vocabulary vocab( config );
            vocab.id_to_token_.resize( vocab_size );

            // --- Vocabulary ---
            for ( uint32_t i = 0; i < vocab_size; ++i )
            {
                uint32_t len = 0;
                read_u32( len );

                std::string token;
                if ( len > 0 )
                {
                    token.resize( len );
                    file.read( token.data(), static_cast<std::streamsize>(len) );
                    if ( !file )
                    {
                        throw std::runtime_error( "Failed reading token string at index " + std::to_string( i ) );
                    }
                }

                float score = 0.0f;  // BPE scores are unused; read and discard
                file.read( reinterpret_cast<char*>(&score), sizeof( score ) );
                if ( !file )
                {
                    throw std::runtime_error( "Failed reading token score at index " + std::to_string( i ) );
                }

                uint32_t token_id = 0;
                read_u32( token_id );

                if ( token_id >= vocab_size )
                {
                    throw std::runtime_error(
                        "Invalid token_id " + std::to_string( token_id ) +
                        " at vocab position " + std::to_string( i ) );
                }

                vocab.id_to_token_[ token_id ] = token;
                vocab.token_to_id_[ token ] = static_cast<TokenId>(token_id);
            }

            // --- Special tokens ---
            // Order matches convert_llama_tokenizer.py: BOS, EOS, PAD, UNK

            uint32_t has_bos = 0;
            read_u32( has_bos );
            if ( has_bos )
            {
                uint32_t bos_id = 0;
                read_u32( bos_id );
                const std::string& bos_str = vocab.config_.getSpecialTokens().bos_token;
                vocab.special_token_ids_[ bos_str ] = static_cast<TokenId>(bos_id);
                std::cout << "  BOS: '" << bos_str << "' (ID: " << bos_id << ")\n";
            }

            uint32_t has_eos = 0;
            read_u32( has_eos );
            if ( has_eos )
            {
                uint32_t eos_id = 0;
                read_u32( eos_id );
                const std::string& eos_str = vocab.config_.getSpecialTokens().eos_token;
                vocab.special_token_ids_[ eos_str ] = static_cast<TokenId>(eos_id);
                std::cout << "  EOS: '" << eos_str << "' (ID: " << eos_id << ")\n";
            }

            uint32_t has_pad = 0;
            read_u32( has_pad );
            if ( has_pad )
            {
                uint32_t pad_id = 0;
                read_u32( pad_id );
                const std::string& pad_str = vocab.config_.getSpecialTokens().pad_token;
                vocab.special_token_ids_[ pad_str ] = static_cast<TokenId>(pad_id);
                std::cout << "  PAD: '" << pad_str << "' (ID: " << pad_id << ")\n";
            }

            uint32_t has_unk = 0;
            read_u32( has_unk );
            if ( has_unk )
            {
                uint32_t unk_id = 0;
                read_u32( unk_id );
                const std::string& unk_str = vocab.config_.getSpecialTokens().unk_token;
                vocab.special_token_ids_[ unk_str ] = static_cast<TokenId>(unk_id);
                std::cout << "  UNK: '" << unk_str << "' (ID: " << unk_id << ")\n";
            }

            // Build the sorted special token list for O(n) pre-pass scanning
            // (sorted longest-first to handle prefix collisions correctly)
            vocab.buildSpecialTokenList();

            std::cout << "Loaded Llama 3.2 vocabulary: "
                << vocab_size << " tokens, "
                << vocab.special_token_ids_.size() << " special tokens\n";

            return vocab;
        }

        Gpt4Vocabulary() = delete;

        // ====================================================================
        // Configuration Access
        // ====================================================================

        const Gpt4VocabularyConfig& getConfig() const
        {
            return config_;
        }

        // ====================================================================
        // Serialization
        // ====================================================================

        void save( const fs::path& path ) const override
        {
            std::ofstream file( path, std::ios::binary );
            if ( !file )
            {
                throw std::runtime_error( "Cannot open vocabulary file for writing: " + path.string() );
            }

            SerializationMetadata meta = config_.toMetadata();
            meta.set( "actual_vocab_size", static_cast<int64_t>(id_to_token_.size()) )
                .set( "num_merges", static_cast<int64_t>(merges_.size()) );

            MilaFileHeader header( MilaFileType::Gpt4BpeVocabulary, std::move( meta ) );
            header.write( file );

            saveContent( file );
        }

        // ====================================================================
        // TokenizerVocabulary Interface
        // ====================================================================

        size_t getSize() const override
        {
            return id_to_token_.size();
        }

        std::optional<TokenId> tokenToId( const std::string& token ) const override
        {
            auto it = token_to_id_.find( token );
            if ( it != token_to_id_.end() )
            {
                return it->second;
            }
            return std::nullopt;  // No UNK fallback for Llama 3.x
        }

        std::optional<std::string> idToToken( TokenId id ) const override
        {
            if ( id >= 0 && static_cast<size_t>(id) < id_to_token_.size() )
            {
                return id_to_token_[ static_cast<size_t>(id) ];
            }
            return std::nullopt;
        }

        // ====================================================================
        // Gpt4Bpe-Specific Methods
        // ====================================================================

        const std::vector<std::pair<std::string, std::string>>& getMergeRules() const
        {
            return merges_;
        }

        /**
         * @brief Look up a special token ID by its string representation.
         *
         * Used by Gpt4BpeTokenizer's encode pre-pass to resolve special tokens
         * directly without going through BPE merges.
         *
         * @param token_str Token string, e.g. "<|begin_of_text|>"
         * @return Token ID if registered as a special token, nullopt otherwise.
         */
        std::optional<TokenId> getSpecialTokenId( const std::string& token_str ) const
        {
            auto it = special_token_ids_.find( token_str );
            if ( it != special_token_ids_.end() )
            {
                return it->second;
            }
            return std::nullopt;
        }

        /**
         * @brief Get the sorted special token list for encode pre-pass scanning.
         *
         * Tokens are sorted longest-first to prevent shorter tokens from
         * matching as prefixes of longer ones during linear scanning.
         *
         * @return Vector of (token_string, token_id) pairs, longest string first.
         */
        const std::vector<std::pair<std::string, TokenId>>& getSpecialTokenList() const
        {
            return special_token_list_;
        }

        std::optional<size_t> getMergePriority( const std::string& left, const std::string& right ) const
        {
            auto it = merge_map_.find( { left, right } );
            if ( it != merge_map_.end() )
            {
                return it->second;
            }
            return std::nullopt;
        }

        static const std::unordered_map<unsigned char, std::string>& getByteEncoder();
        static const std::unordered_map<std::string, unsigned char>& getByteDecoder();

        bool isByteLevel() const
        {
            return config_.isByteLevel();
        }

    private:

        explicit Gpt4Vocabulary( const Gpt4VocabularyConfig& config )
            : config_( config )
        {}

        // ====================================================================
        // Serialization Implementation
        // ====================================================================

        void saveContent( std::ostream& file ) const
        {
            uint32_t version = 1;
            file.write( reinterpret_cast<const char*>(&version), sizeof( version ) );

            uint32_t vocab_size = static_cast<uint32_t>(id_to_token_.size());
            file.write( reinterpret_cast<const char*>(&vocab_size), sizeof( vocab_size ) );

            // Special tokens: write as (string_length, string, id) triples
            uint32_t num_special = static_cast<uint32_t>(special_token_ids_.size());
            file.write( reinterpret_cast<const char*>(&num_special), sizeof( num_special ) );

            for ( const auto& [token_str, id] : special_token_ids_ )
            {
                uint32_t str_len = static_cast<uint32_t>(token_str.size());
                file.write( reinterpret_cast<const char*>(&str_len), sizeof( str_len ) );
                file.write( token_str.data(), str_len );

                uint32_t token_id = static_cast<uint32_t>(id);
                file.write( reinterpret_cast<const char*>(&token_id), sizeof( token_id ) );
            }

            uint32_t num_merges = static_cast<uint32_t>(merges_.size());
            file.write( reinterpret_cast<const char*>(&num_merges), sizeof( num_merges ) );

            for ( const auto& [left, right] : merges_ )
            {
                uint32_t llen = static_cast<uint32_t>(left.size());
                file.write( reinterpret_cast<const char*>(&llen), sizeof( llen ) );
                file.write( left.data(), llen );

                uint32_t rlen = static_cast<uint32_t>(right.size());
                file.write( reinterpret_cast<const char*>(&rlen), sizeof( rlen ) );
                file.write( right.data(), rlen );
            }

            for ( const auto& token : id_to_token_ )
            {
                uint32_t len = static_cast<uint32_t>(token.size());
                file.write( reinterpret_cast<const char*>(&len), sizeof( len ) );
                if ( len > 0 )
                {
                    file.write( token.data(), len );
                }
            }

            if ( !file )
            {
                throw std::runtime_error( "Error writing Gpt4BpeVocabulary content" );
            }
        }

        void loadContent( std::istream& file )
        {
            uint32_t version = 0;
            file.read( reinterpret_cast<char*>(&version), sizeof( version ) );
            if ( version != 1 )
            {
                throw std::runtime_error(
                    "Unsupported Gpt4BpeVocabulary content version: " + std::to_string( version ) );
            }

            uint32_t vocab_size = 0;
            file.read( reinterpret_cast<char*>(&vocab_size), sizeof( vocab_size ) );

            uint32_t num_special = 0;
            file.read( reinterpret_cast<char*>(&num_special), sizeof( num_special ) );

            for ( uint32_t i = 0; i < num_special; ++i )
            {
                uint32_t str_len = 0;
                file.read( reinterpret_cast<char*>( &str_len ), sizeof( str_len ) );

                std::string token_str;
                if ( str_len > 0 )
                {
                    token_str.resize( str_len );
                    file.read( token_str.data(), str_len );
                }

                uint32_t token_id = 0;
                file.read( reinterpret_cast<char*>(&token_id), sizeof( token_id ) );

                special_token_ids_[ token_str ] = static_cast<TokenId>(token_id);
            }

            uint32_t num_merges = 0;
            file.read( reinterpret_cast<char*>(&num_merges), sizeof( num_merges ) );

            merges_.reserve( num_merges );
            for ( uint32_t i = 0; i < num_merges; ++i )
            {
                uint32_t llen = 0;
                file.read( reinterpret_cast<char*>( &llen ), sizeof( llen ) );
                std::string left( llen, '\0' );
                if ( llen > 0 ) file.read( left.data(), llen );

                uint32_t rlen = 0;
                file.read( reinterpret_cast<char*>(&rlen), sizeof( rlen ) );
                std::string right( rlen, '\0' );
                if ( rlen > 0 ) file.read( right.data(), rlen );

                merges_.emplace_back( std::move( left ), std::move( right ) );
            }

            id_to_token_.resize( vocab_size );
            for ( uint32_t i = 0; i < vocab_size; ++i )
            {
                uint32_t len = 0;
                file.read( reinterpret_cast<char*>( &len ), sizeof( len ) );
                std::string token( len, '\0' );
                if ( len > 0 ) file.read( token.data(), len );

                id_to_token_[ i ] = token;
                token_to_id_[ token ] = static_cast<TokenId>(i);
            }

            buildMergeMap();
            buildSpecialTokenList();

            if ( !file )
            {
                throw std::runtime_error( "Error reading Gpt4BpeVocabulary content" );
            }
        }

        // ====================================================================
        // Helper Methods
        // ====================================================================

        struct PairHash
        {
            size_t operator()( const std::pair<std::string, std::string>& p ) const
            {
                return std::hash<std::string>{}(p.first) ^ (std::hash<std::string>{}(p.second) << 1);
            }
        };

        void buildMergeMap()
        {
            merge_map_.clear();
            merge_map_.reserve( merges_.size() );
            for ( size_t i = 0; i < merges_.size(); ++i )
            {
                merge_map_[ merges_[ i ] ] = i;
            }
        }

        /**
         * @brief Build the sorted special token list for O(n) encode pre-pass.
         *
         * Sorted longest-first so that e.g. "<|begin_of_text|>" is matched
         * before any shorter prefix could be.
         */
        void buildSpecialTokenList()
        {
            special_token_list_.clear();
            special_token_list_.reserve( special_token_ids_.size() );

            for ( const auto& [token_str, id] : special_token_ids_ )
            {
                special_token_list_.emplace_back( token_str, id );
            }

            std::sort(
                special_token_list_.begin(), special_token_list_.end(),
                []( const auto& a, const auto& b )
                {
                    return a.first.size() > b.first.size();  // longest first
                }
            );
        }

        // ====================================================================
        // Member Variables
        // ====================================================================

        Gpt4VocabularyConfig config_;

        std::unordered_map<std::string, TokenId> token_to_id_;
        std::vector<std::string> id_to_token_;
        std::vector<std::pair<std::string, std::string>> merges_;
        std::unordered_map<std::pair<std::string, std::string>, size_t, PairHash> merge_map_;

        // Keyed on token string (e.g. "<|begin_of_text|>") rather than char type.
        // This is the critical difference from BpeVocabulary's special_token_ids_,
        // enabling the encode pre-pass to match Llama 3.x's full special token set.
        std::unordered_map<std::string, TokenId> special_token_ids_;

        // Sorted (longest first) for safe linear scanning in the encode pre-pass.
        std::vector<std::pair<std::string, TokenId>> special_token_list_;
    };

    // ========================================================================
    // Static Byte Encoder/Decoder (GPT-2 style — shared with BpeVocabulary)
    // ========================================================================

    const std::unordered_map<unsigned char, std::string>& Gpt4Vocabulary::getByteEncoder()
    {
        static std::unordered_map<unsigned char, std::string> encoder = []()
            {
                std::unordered_map<unsigned char, std::string> enc;

                std::vector<int> bs;
                for ( int i = int( '!' ); i <= int( '~' ); ++i ) bs.push_back( i );
                for ( int i = 0xA1; i <= 0xAC; ++i )        bs.push_back( i );
                for ( int i = 0xAE; i <= 0xFF; ++i )        bs.push_back( i );

                std::vector<int> cs = bs;
                int n = 0;
                for ( int b = 0; b < 256; ++b )
                {
                    if ( std::find( bs.begin(), bs.end(), b ) == bs.end() )
                    {
                        bs.push_back( b );
                        cs.push_back( 256 + n );
                        ++n;
                    }
                }

                for ( size_t i = 0; i < bs.size(); ++i )
                {
                    char32_t unicode_point = static_cast<char32_t>( cs[ i ] );

                    std::string utf8_str;
                    if ( unicode_point < 0x80 )
                    {
                        utf8_str += static_cast<char>( unicode_point );
                    }
                    else if ( unicode_point < 0x800 )
                    {
                        utf8_str += static_cast<char>( 0xC0 | (unicode_point >> 6) );
                        utf8_str += static_cast<char>( 0x80 | (unicode_point & 0x3F) );
                    }
                    else
                    {
                        utf8_str += static_cast<char>( 0xE0 | (unicode_point >> 12) );
                        utf8_str += static_cast<char>( 0x80 | ((unicode_point >> 6) & 0x3F) );
                        utf8_str += static_cast<char>( 0x80 | (unicode_point & 0x3F) );
                    }

                    enc[ static_cast<unsigned char>(bs[ i ]) ] = utf8_str;
                }
                return enc;
            }();
        return encoder;
    }

    const std::unordered_map<std::string, unsigned char>& Gpt4Vocabulary::getByteDecoder()
    {
        static std::unordered_map<std::string, unsigned char> decoder = []()
            {
                std::unordered_map<std::string, unsigned char> dec;
                const auto& encoder = getByteEncoder();
                for ( const auto& [byte, utf8_str] : encoder )
                {
                    dec[ utf8_str ] = byte;
                }
                return dec;
            }();
        return decoder;
    }
}
