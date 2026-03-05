/**
 * @file BpeVocabulary.ixx
 * @brief BPE vocabulary for GPT-2, Llama 3.x, and Mistral model families.
 */

module;
#include <string>
#include <string_view>
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
#include <chrono>
#include <iostream>
#include <iomanip>

export module Data.BpeVocabulary;

import Data.SpecialTokens;
import Data.Tokenizer;
import Data.TokenizerVocabulary;
import Data.BpeVocabularyConfig;
import Data.BpePreTokenizationMode;
import Data.FileHeader;
import Serialization.Metadata;

namespace Mila::Data
{
    namespace fs = std::filesystem;
    using Mila::Data::TokenizerVocabulary;
    using Mila::Data::TokenId;
    using Mila::Dnn::Serialization::SerializationMetadata;

    /**
     * @brief Unified Byte Pair Encoding (BPE) vocabulary.
     *
     * Immutable after construction; safe for concurrent reads. Supports training
     * from scratch via BpeTrainer, or loading pretrained vocabularies from:
     *   - Mila binary format produced by save() (load)
     *   - GPT-2 binary produced by convert_gpt2_tokenizer.py (loadGpt2)
     *   - Llama 3.2 binary produced by convert_llama_tokenizer.py (loadLlama32)
     *
     * Special tokens are keyed on their string representation (e.g., "<|endoftext|>",
     * "<|begin_of_text|>") and exposed via getSpecialTokenList() for O(n) pre-pass
     * scanning in BpeTokenizer. Extended special tokens from SpecialTokens are
     * registered automatically.
     */
    export class BpeVocabulary : public TokenizerVocabulary
    {
    public:

        // ====================================================================
        // Factory Methods
        // ====================================================================

        /**
         * @brief Train a BPE vocabulary from a text corpus.
         *
         * @param corpus Training text.
         * @param config Vocabulary configuration; config.validate() is called internally.
         * @return Trained BpeVocabulary instance.
         * @throws std::invalid_argument if config fails validation.
         */
        static BpeVocabulary train( const std::string& corpus, const BpeVocabularyConfig& config )
        {
            config.validate();

            BpeVocabulary vocab( config );
            vocab.buildFromText( corpus );

            return vocab;
        }

        /**
         * @brief Train a BPE vocabulary from a corpus file.
         *
         * @param corpus_path Path to a UTF-8 text corpus file.
         * @param config Vocabulary configuration.
         * @return Trained BpeVocabulary instance.
         * @throws std::runtime_error if the file cannot be opened.
         * @throws std::invalid_argument if config fails validation.
         */
        static BpeVocabulary trainFromFile( const fs::path& corpus_path, const BpeVocabularyConfig& config )
        {
            std::ifstream file( corpus_path, std::ios::binary );

            if ( !file )
            {
                throw std::runtime_error( "Cannot open corpus file: " + corpus_path.string() );
            }

            std::string corpus( (std::istreambuf_iterator<char>( file )), std::istreambuf_iterator<char>() );

            return train( corpus, config );
        }

        /**
         * @brief Load a vocabulary from Mila binary format (version 2).
         *
         * Reads a file written by save(). Special tokens are restored from
         * the serialized (string, id) pairs and the special token list is
         * rebuilt automatically.
         *
         * @param path Input file path.
         * @return Loaded BpeVocabulary instance.
         * @throws std::runtime_error on I/O errors or format mismatch.
         */
        static BpeVocabulary load( const fs::path& path )
        {
            std::ifstream file( path, std::ios::binary );

            if ( !file )
            {
                throw std::runtime_error( "Cannot open vocabulary file: " + path.string() );
            }

            MilaFileHeader header = MilaFileHeader::read( file );

            if ( header.getFileType() != MilaFileType::BpeVocabulary )
            {
                throw std::runtime_error(
                    "File is not a BpeVocabulary (got " +
                    std::string( toString( header.getFileType() ) ) + "): " + path.string() );
            }

            BpeVocabularyConfig config;
            config.fromMetadata( header.getMetadata() );

            BpeVocabulary vocab( config );
            vocab.loadContent( file );

            return vocab;
        }

        /**
         * @brief Load a pretrained GPT-2 vocabulary.
         *
         * Reads the binary format produced by convert_gpt2_tokenizer.py:
         * @code
         *   vocab_size   (uint32)
         *   num_merges   (uint32)
         *   For each token: token_length (uint32), token_bytes (utf-8), token_id (uint32)
         *   For each merge: left_length (uint32), left, right_length (uint32), right
         *   has_eos (uint32), eos_id (uint32, conditional)
         *   has_bos (uint32), bos_id (uint32, conditional)
         *   has_pad (uint32), pad_id (uint32, conditional)
         * @endcode
         *
         * @param tokenizer_path Path to the converted GPT-2 tokenizer binary.
         * @return Loaded BpeVocabulary instance.
         * @throws std::runtime_error on I/O or format errors.
         */
        static BpeVocabulary loadGpt2( const fs::path& tokenizer_path );

        /**
         * @brief Load a pretrained Llama 3.2 vocabulary.
         *
         * Reads the binary format produced by convert_llama_tokenizer.py:
         * @code
         *   Header: vocab_size (uint32), use_byte_fallback (uint8)
         *   For each token: token_length (uint32), token_bytes, score (float32), token_id (uint32)
         *   has_bos (uint32), bos_id (uint32, conditional)  -- 128000
         *   has_eos (uint32), eos_id (uint32, conditional)  -- 128001
         *   has_pad (uint32), pad_id (uint32, conditional)
         *   has_unk (uint32), unk_id (uint32, conditional)  -- absent for Llama 3.2
         * @endcode
         *
         * Llama 3.x vocabularies carry no explicit BPE merges; the merge order is
         * encoded implicitly in the token ID assignment.
         *
         * @param path Path to the converted Llama 3.2 tokenizer binary.
         * @return Loaded BpeVocabulary instance.
         * @throws std::runtime_error on I/O or format errors.
         */
        static BpeVocabulary loadLlama32( const fs::path& path );

        /**
         * @brief Load a pretrained Mistral vocabulary.
         *
         * @note Not yet implemented for external Mistral formats.
         *       Provide a Mila binary produced by save() as a workaround.
         *
         * @throws std::runtime_error always.
         */
        static BpeVocabulary loadMistral( const fs::path& vocab_path, const fs::path& merges_path );

        BpeVocabulary() = delete;

        // ====================================================================
        // Configuration
        // ====================================================================

        const BpeVocabularyConfig& getConfig() const
        {
            return config_;
        }

        // ====================================================================
        // Serialization
        // ====================================================================

        /**
         * @brief Serialize the vocabulary to Mila binary format (content version 2).
         *
         * Writes a MilaFileHeader followed by the vocabulary content. Special tokens
         * are stored as (string_length, string, token_id) triples, eliminating the
         * char-key indirection used in the former GPT-2-only format.
         *
         * @param path Output file path. Parent directory must exist.
         * @throws std::runtime_error on I/O errors.
         */
        void save( const fs::path& path ) const override
        {
            std::ofstream file( path, std::ios::binary );

            if ( !file )
            {
                throw std::runtime_error( "Cannot open vocabulary file for writing: " + path.string() );
            }

            SerializationMetadata meta = config_.toMetadata();
            meta.set( "content_version", static_cast<int64_t>(2) )
                .set( "actual_vocab_size", static_cast<int64_t>(id_to_token_.size()) )
                .set( "num_merges", static_cast<int64_t>(merges_.size()) );

            MilaFileHeader header( MilaFileType::BpeVocabulary, std::move( meta ) );
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

        /**
         * @brief Convert a token string to its ID.
         *
         * Falls back to the UNK token ID when the token is not found and use_unk
         * is enabled (GPT-2 style). Llama 3.x vocabularies return nullopt on a miss
         * because they rely on byte-level fallback rather than an UNK token.
         *
         * @param token UTF-8 encoded token string.
         * @return Token ID, UNK ID (if enabled), or nullopt on miss.
         */
        std::optional<TokenId> tokenToId( const std::string& token ) const override
        {
            auto it = token_to_id_.find( token );

            if ( it != token_to_id_.end() )
            {
                return it->second;
            }

            if ( config_.getSpecialTokens().use_unk )
            {
                auto unk_it = special_token_ids_.find( config_.getSpecialTokens().unk_token );

                if ( unk_it != special_token_ids_.end() )
                {
                    return unk_it->second;
                }
            }

            return std::nullopt;
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
        // BPE-Specific Interface
        // ====================================================================

        const std::vector<std::pair<std::string, std::string>>& getMergeRules() const
        {
            return merges_;
        }

        /**
         * @brief Look up a special token ID by its string representation.
         *
         * Used by BpeTokenizer's encode pre-pass to resolve tokens such as
         * "<|endoftext|>" or "<|begin_of_text|>" directly to IDs before BPE runs.
         *
         * @param token_str Token string to look up.
         * @return Token ID if registered as special, nullopt otherwise.
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
         * @brief Return the special token list sorted longest-first.
         *
         * Ordered longest-first so BpeTokenizer's linear scan matches longer tokens
         * before any of their prefixes (e.g., "<|begin_of_text|>" before "<|").
         *
         * @return Vector of (token_string, token_id) pairs.
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

        // ====================================================================
        // Private Constructor
        // ====================================================================

        explicit BpeVocabulary( const BpeVocabularyConfig& config )
            : config_( config ), current_id_( 0 )
        {}

        // ====================================================================
        // Training Implementation (declarations)
        // ====================================================================

        void buildFromText( const std::string& corpus );
        void initializeBaseVocabulary();
        void addSpecialTokensFromConfig();
        std::vector<std::string> preTokenizeCorpus( const std::string& text );
        std::vector<std::vector<std::string>> convertToTokenSequences( const std::vector<std::string>& words );
        void runBpeMergeLoop( std::vector<std::vector<std::string>>& corpus_tokens, std::chrono::steady_clock::time_point start_time );
        void logTrainingComplete( std::chrono::steady_clock::time_point start_time );

        // ====================================================================
        // Serialization Implementation (declarations)
        // ====================================================================

        void saveContent( std::ostream& file ) const;
        void loadContent( std::istream& file );

        // ====================================================================
        // Helpers
        // ====================================================================

        struct PairHash
        {
            size_t operator()( const std::pair<std::string, std::string>& p ) const
            {
                return std::hash<std::string>{}(p.first) ^ (std::hash<std::string>{}(p.second) << 1);
            }
        };

        struct PairViewHash
        {
            size_t operator()( const std::pair<std::string_view, std::string_view>& p ) const
            {
                size_t h1 = std::hash<std::string_view>{}(p.first);
                size_t h2 = std::hash<std::string_view>{}(p.second);
                return h1 ^ (h2 << 1);
            }
        };

        void buildMergeMap();
        void buildSpecialTokenList();
        std::vector<std::string> preTokenize( const std::string& text ) const;

        std::unordered_map<std::pair<std::string, std::string>, size_t, PairHash>
            countPairs( const std::vector<std::vector<std::string>>& corpus ) const;

        std::pair<std::pair<std::string, std::string>, size_t>
            getMostFrequentPair( const std::unordered_map<std::pair<std::string, std::string>, size_t, PairHash>& counts ) const;

        void applyMergeAndUpdateCounts(
            std::vector<std::vector<std::string>>& corpus,
            const std::string& left,
            const std::string& right,
            const std::string& merged,
            std::unordered_map<std::pair<std::string, std::string>, size_t, PairHash>& pair_counts );

        void addSpecialToken( const std::string& token, TokenId id );

        // ====================================================================
        // Member Variables
        // ====================================================================

        BpeVocabularyConfig config_;
        TokenId             current_id_;

        std::unordered_map<std::string, TokenId>  token_to_id_;
        std::vector<std::string>                  id_to_token_;
        std::vector<std::pair<std::string, std::string>> merges_;
        std::unordered_map<std::pair<std::string, std::string>, size_t, PairHash> merge_map_;

        // String-keyed to support the encode pre-pass across all BPE families.
        // GPT-2 registers "<|endoftext|>"; Llama 3.x registers "<|begin_of_text|>" etc.
        std::unordered_map<std::string, TokenId>        special_token_ids_;

        // Sorted longest-first for safe prefix-collision-free scanning in the pre-pass.
        std::vector<std::pair<std::string, TokenId>>    special_token_list_;
    };

    // ========================================================================
    // Training Implementation
    // ========================================================================

    void BpeVocabulary::buildFromText( const std::string& corpus )
    {
        const auto start_time = std::chrono::steady_clock::now();

        token_to_id_.clear();
        id_to_token_.clear();
        merges_.clear();
        merge_map_.clear();
        special_token_ids_.clear();
        special_token_list_.clear();
        current_id_ = 0;

        initializeBaseVocabulary();
        addSpecialTokensFromConfig();

        const size_t target_vocab_size = config_.getVocabSize();

        if ( target_vocab_size == 0 || static_cast<size_t>(current_id_) >= target_vocab_size )
        {
            buildMergeMap();
            return;
        }

        auto words = preTokenizeCorpus( corpus );
        auto corpus_tokens = convertToTokenSequences( words );

        runBpeMergeLoop( corpus_tokens, start_time );
        buildMergeMap();
        logTrainingComplete( start_time );
    }

    void BpeVocabulary::initializeBaseVocabulary()
    {
        if ( !config_.isByteLevel() )
        {
            throw std::runtime_error( "BpeVocabulary: character-level BPE is not yet implemented; use byte_level=true." );
        }

        for ( int i = 0; i < 256; ++i )
        {
            std::string byte_token( 1, static_cast<char>( i ) );
            id_to_token_.push_back( byte_token );
            token_to_id_[ byte_token ] = current_id_++;
        }
    }

    void BpeVocabulary::addSpecialTokensFromConfig()
    {
        const auto& st = config_.getSpecialTokens();

        if ( st.use_pad )  addSpecialToken( st.pad_token, current_id_++ );
        if ( st.use_unk )  addSpecialToken( st.unk_token, current_id_++ );
        if ( st.use_bos )  addSpecialToken( st.bos_token, current_id_++ );
        if ( st.use_eos )  addSpecialToken( st.eos_token, current_id_++ );
        if ( st.use_mask ) addSpecialToken( st.mask_token, current_id_++ );
        if ( st.use_sep )  addSpecialToken( st.sep_token, current_id_++ );
        if ( st.use_cls )  addSpecialToken( st.cls_token, current_id_++ );

        // Extended tokens (e.g., Llama 3.x reserved set) carry explicit IDs
        // and must not consume slots from current_id_.
        for ( const auto& [token_str, id] : st.extended_special_tokens )
        {
            special_token_ids_[ token_str ] = static_cast<TokenId>(id);
        }

        buildSpecialTokenList();
    }

    std::vector<std::string> BpeVocabulary::preTokenizeCorpus( const std::string& text )
    {
        return preTokenize( text );
    }

    std::vector<std::vector<std::string>> BpeVocabulary::convertToTokenSequences(
        const std::vector<std::string>& words )
    {
        std::vector<std::vector<std::string>> corpus_tokens;
        corpus_tokens.reserve( words.size() );

        for ( const auto& word : words )
        {
            std::vector<std::string> tokens;
            tokens.reserve( word.size() );

            for ( unsigned char byte : word )
            {
                tokens.push_back( std::string( 1, static_cast<char>(byte) ) );
            }

            if ( !tokens.empty() )
            {
                corpus_tokens.push_back( std::move( tokens ) );
            }
        }

        return corpus_tokens;
    }

    void BpeVocabulary::runBpeMergeLoop(
        std::vector<std::vector<std::string>>& corpus_tokens,
        std::chrono::steady_clock::time_point  start_time )
    {
        const size_t target_vocab_size = config_.getVocabSize();
        const size_t min_frequency = config_.getMinFrequency();
        const size_t max_merges = config_.getMaxMerges();

        size_t merges_performed = 0;
        auto pair_counts = countPairs( corpus_tokens );

        while ( static_cast<size_t>(current_id_) < target_vocab_size )
        {
            if ( pair_counts.empty() )
            {
                break;
            }

            auto [best_pair, best_count] = getMostFrequentPair( pair_counts );

            if ( best_count < min_frequency )
            {
                break;
            }

            std::string merged;
            merged.reserve( best_pair.first.size() + best_pair.second.size() );
            merged.append( best_pair.first );
            merged.append( best_pair.second );

            if ( token_to_id_.find( merged ) != token_to_id_.end() )
            {
                break;
            }

            applyMergeAndUpdateCounts( corpus_tokens, best_pair.first, best_pair.second, merged, pair_counts );
            pair_counts.erase( best_pair );

            id_to_token_.push_back( merged );
            token_to_id_[ merged ] = current_id_;
            merges_.push_back( best_pair );

            ++current_id_;
            ++merges_performed;

            if ( merges_performed % 100 == 0 )
            {
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::steady_clock::now() - start_time).count();
                float progress = static_cast<float>(merges_performed) /
                    static_cast<float>(target_vocab_size) * 100.0f;

                std::cout << "\r[" << elapsed << "s] "
                    << merges_performed << "/" << target_vocab_size
                    << " (" << std::fixed << std::setprecision( 1 ) << progress << "%)"
                    << " | freq: " << best_count
                    << "          " << std::flush;
            }

            if ( max_merges > 0 && merges_performed >= max_merges )
            {
                break;
            }
        }

        std::cout << '\n';
    }

    void BpeVocabulary::logTrainingComplete( std::chrono::steady_clock::time_point start_time )
    {
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - start_time).count();

        std::cout << "Training completed in " << elapsed << "s\n"
            << "Final vocabulary size: " << id_to_token_.size() << '\n';
    }

    // ========================================================================
    // Serialization Implementation
    // ========================================================================

    void BpeVocabulary::saveContent( std::ostream& file ) const
    {
        // Content version 2: special tokens stored as (str_len, string, token_id).
        // Version 1 used a char type-code key and is no longer supported.
        uint32_t version = 2;
        file.write( reinterpret_cast<const char*>(&version), sizeof( version ) );

        uint32_t vocab_size = static_cast<uint32_t>(id_to_token_.size());
        file.write( reinterpret_cast<const char*>(&vocab_size), sizeof( vocab_size ) );

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
            uint32_t left_len = static_cast<uint32_t>(left.size());
            file.write( reinterpret_cast<const char*>(&left_len), sizeof( left_len ) );
            file.write( left.data(), left_len );

            uint32_t right_len = static_cast<uint32_t>(right.size());
            file.write( reinterpret_cast<const char*>(&right_len), sizeof( right_len ) );
            file.write( right.data(), right_len );
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
            throw std::runtime_error( "Error writing BpeVocabulary content" );
        }
    }

    void BpeVocabulary::loadContent( std::istream& file )
    {
        uint32_t version = 0;
        file.read( reinterpret_cast<char*>(&version), sizeof( version ) );

        if ( version != 2 )
        {
            throw std::runtime_error(
                "Unsupported BpeVocabulary content version: " + std::to_string( version ) +
                " (expected 2; re-save with the current build)" );
        }

        uint32_t vocab_size = 0;
        file.read( reinterpret_cast<char*>(&vocab_size), sizeof( vocab_size ) );

        uint32_t num_special = 0;
        file.read( reinterpret_cast<char*>(&num_special), sizeof( num_special ) );

        for ( uint32_t i = 0; i < num_special; ++i )
        {
            uint32_t str_len = 0;
            file.read( reinterpret_cast<char*>( &str_len ), sizeof( str_len ) );

            std::string token_str( str_len, '\0' );

            if ( str_len > 0 )
            {
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
            uint32_t left_len = 0;
            file.read( reinterpret_cast<char*>( &left_len ), sizeof( left_len ) );
            std::string left( left_len, '\0' );

            if ( left_len > 0 )
            {
                file.read( left.data(), left_len );
            }

            uint32_t right_len = 0;
            file.read( reinterpret_cast<char*>(&right_len), sizeof( right_len ) );
            std::string right( right_len, '\0' );

            if ( right_len > 0 )
            {
                file.read( right.data(), right_len );
            }

            merges_.emplace_back( std::move( left ), std::move( right ) );
        }

        id_to_token_.resize( vocab_size );

        for ( uint32_t i = 0; i < vocab_size; ++i )
        {
            uint32_t len = 0;
            file.read( reinterpret_cast<char*>( &len ), sizeof( len ) );

            std::string token( len, '\0' );

            if ( len > 0 )
            {
                file.read( token.data(), len );
            }

            id_to_token_[ i ] = std::move( token );
            token_to_id_[ id_to_token_[ i ] ] = static_cast<TokenId>(i);
        }

        buildMergeMap();
        buildSpecialTokenList();

        if ( !file )
        {
            throw std::runtime_error( "Error reading BpeVocabulary content" );
        }
    }

    // ========================================================================
    // Helper Implementations
    // ========================================================================

    void BpeVocabulary::buildMergeMap()
    {
        merge_map_.clear();
        merge_map_.reserve( merges_.size() );

        for ( size_t i = 0; i < merges_.size(); ++i )
        {
            merge_map_[ merges_[ i ] ] = i;
        }
    }

    void BpeVocabulary::buildSpecialTokenList()
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
                return a.first.size() > b.first.size();
            }
        );
    }

    std::vector<std::string> BpeVocabulary::preTokenize( const std::string& text ) const
    {
        std::vector<std::string> words;
        std::istringstream stream( text );
        std::string word;

        while ( stream >> word )
        {
            if ( !word.empty() )
            {
                words.push_back( word );
            }
        }

        return words;
    }

    std::unordered_map<std::pair<std::string, std::string>, size_t, BpeVocabulary::PairHash>
        BpeVocabulary::countPairs( const std::vector<std::vector<std::string>>& corpus ) const
    {
        std::unordered_map<std::pair<std::string_view, std::string_view>, size_t, PairViewHash> temp_counts;

        for ( const auto& tokens : corpus )
        {
            for ( size_t i = 0; i + 1 < tokens.size(); ++i )
            {
                temp_counts[ { std::string_view{ tokens[ i ] }, std::string_view{ tokens[ i + 1 ] } } ]++;
            }
        }

        std::unordered_map<std::pair<std::string, std::string>, size_t, PairHash> counts;
        counts.reserve( temp_counts.size() );

        for ( const auto& [pair_view, count] : temp_counts )
        {
            counts[ { std::string{ pair_view.first }, std::string{ pair_view.second } } ] = count;
        }

        return counts;
    }

    std::pair<std::pair<std::string, std::string>, size_t>
        BpeVocabulary::getMostFrequentPair(
            const std::unordered_map<std::pair<std::string, std::string>, size_t, PairHash>& counts ) const
    {
        if ( counts.empty() )
        {
            return { { std::string{}, std::string{} }, size_t{ 0 } };
        }

        auto best = std::max_element(
            counts.begin(), counts.end(),
            []( const auto& a, const auto& b ) { return a.second < b.second; }
        );

        return *best;
    }

    void BpeVocabulary::applyMergeAndUpdateCounts(
        std::vector<std::vector<std::string>>& corpus,
        const std::string& left,
        const std::string& right,
        const std::string& merged,
        std::unordered_map<std::pair<std::string, std::string>, size_t, PairHash>& pair_counts )
    {
        std::string_view left_view{ left };
        std::string_view right_view{ right };

        for ( auto& word : corpus )
        {
            for ( size_t i = 0; i + 1 < word.size(); )
            {
                if ( std::string_view{ word[ i ] } == left_view &&
                    std::string_view{ word[ i + 1 ] } == right_view )
                {
                    if ( i > 0 )
                    {
                        auto old_left = std::make_pair( word[ i - 1 ], word[ i ] );

                        if ( --pair_counts[ old_left ] == 0 )
                        {
                            pair_counts.erase( old_left );
                        }
                    }

                    if ( i + 2 < word.size() )
                    {
                        auto old_right = std::make_pair( word[ i + 1 ], word[ i + 2 ] );

                        if ( --pair_counts[ old_right ] == 0 )
                        {
                            pair_counts.erase( old_right );
                        }
                    }

                    word[ i ] = merged;
                    word.erase( word.begin() + static_cast<std::ptrdiff_t>(i + 1) );

                    if ( i > 0 )
                    {
                        pair_counts[ { word[ i - 1 ], word[ i ] } ]++;
                    }

                    if ( i + 1 < word.size() )
                    {
                        pair_counts[ { word[ i ], word[ i + 1 ] } ]++;
                    }
                }
                else
                {
                    ++i;
                }
            }
        }
    }

    void BpeVocabulary::addSpecialToken( const std::string& token, TokenId id )
    {
        id_to_token_.push_back( token );
        token_to_id_[ token ] = id;
        special_token_ids_[ token ] = id;
    }

    // ========================================================================
    // Static Byte Encoder / Decoder  (GPT-2 style — shared across all families)
    // ========================================================================

    const std::unordered_map<unsigned char, std::string>& BpeVocabulary::getByteEncoder()
    {
        static const std::unordered_map<unsigned char, std::string> encoder = []()
            {
                std::unordered_map<unsigned char, std::string> enc;

                std::vector<int> bs;
                for ( int i = static_cast<int>('!'); i <= static_cast<int>('~'); ++i ) bs.push_back( i );
                for ( int i = 0xA1; i <= 0xAC; ++i ) bs.push_back( i );
                for ( int i = 0xAE; i <= 0xFF; ++i ) bs.push_back( i );

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
                    char32_t cp = static_cast<char32_t>( cs[ i ] );

                    std::string utf8;
                    if ( cp < 0x80 )
                    {
                        utf8 += static_cast<char>( cp );
                    }
                    else if ( cp < 0x800 )
                    {
                        utf8 += static_cast<char>( 0xC0 | (cp >> 6) );
                        utf8 += static_cast<char>( 0x80 | (cp & 0x3F) );
                    }
                    else
                    {
                        utf8 += static_cast<char>( 0xE0 | (cp >> 12) );
                        utf8 += static_cast<char>( 0x80 | ((cp >> 6) & 0x3F) );
                        utf8 += static_cast<char>( 0x80 | (cp & 0x3F) );
                    }

                    enc[ static_cast<unsigned char>(bs[ i ]) ] = std::move( utf8 );
                }

                return enc;
            }();

        return encoder;
    }

    const std::unordered_map<std::string, unsigned char>& BpeVocabulary::getByteDecoder()
    {
        static const std::unordered_map<std::string, unsigned char> decoder = []()
            {
                std::unordered_map<std::string, unsigned char> dec;

                for ( const auto& [byte, utf8] : getByteEncoder() )
                {
                    dec[ utf8 ] = byte;
                }

                return dec;
            }();

        return decoder;
    }

    // ========================================================================
    // External Format Loaders
    // ========================================================================

    BpeVocabulary BpeVocabulary::loadGpt2( const fs::path& tokenizer_path )
    {
        std::ifstream file( tokenizer_path, std::ios::binary );

        if ( !file )
        {
            throw std::runtime_error( "Cannot open GPT-2 tokenizer file: " + tokenizer_path.string() );
        }

        auto read_u32 = [&]( uint32_t& out )
            {
                file.read( reinterpret_cast<char*>(&out), sizeof( out ) );

                if ( !file )
                {
                    throw std::runtime_error(
                        "Unexpected EOF reading GPT-2 tokenizer: " + tokenizer_path.string() );
                }
            };

        uint32_t vocab_size = 0;
        read_u32( vocab_size );

        uint32_t num_merges = 0;
        read_u32( num_merges );

        BpeVocabularyConfig config = BpeVocabularyConfig()
            .withVocabSize( vocab_size )
            .withByteLevel( true )
            .withPreTokenization( PreTokenizationMode::Gpt2Regex )
            .withPreTokenizationPattern( GPT2_PRETOKENIZATION_PATTERN )
            .withSpecialTokens( SpecialTokens::gptStyle() );

        BpeVocabulary vocab( config );
        vocab.id_to_token_.resize( vocab_size );

        for ( uint32_t i = 0; i < vocab_size; ++i )
        {
            uint32_t len = 0;
            read_u32( len );

            std::string token( len, '\0' );

            if ( len > 0 )
            {
                file.read( token.data(), static_cast<std::streamsize>(len) );

                if ( !file )
                {
                    throw std::runtime_error(
                        "Failed reading token at position " + std::to_string( i ) );
                }
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

        vocab.merges_.reserve( num_merges );

        for ( uint32_t i = 0; i < num_merges; ++i )
        {
            uint32_t llen = 0;
            read_u32( llen );
            std::string left( llen, '\0' );

            if ( llen > 0 )
            {
                file.read( left.data(), static_cast<std::streamsize>(llen) );
                if ( !file ) throw std::runtime_error( "Failed reading merge left at " + std::to_string( i ) );
            }

            uint32_t rlen = 0;
            read_u32( rlen );
            std::string right( rlen, '\0' );

            if ( rlen > 0 )
            {
                file.read( right.data(), static_cast<std::streamsize>(rlen) );
                if ( !file ) throw std::runtime_error( "Failed reading merge right at " + std::to_string( i ) );
            }

            vocab.merges_.emplace_back( std::move( left ), std::move( right ) );
        }

        // GPT-2 uses a single "<|endoftext|>" string for all roles; gptStyle() sets all four
        // token strings to that value so a single map entry covers BOS, EOS, PAD, and UNK.
        const auto& st = vocab.config_.getSpecialTokens();

        uint32_t has_eos = 0;
        read_u32( has_eos );

        if ( has_eos )
        {
            uint32_t eos_id = 0;
            read_u32( eos_id );
            vocab.special_token_ids_[ st.eos_token ] = static_cast<TokenId>(eos_id);
        }

        uint32_t has_bos = 0;
        read_u32( has_bos );

        if ( has_bos )
        {
            uint32_t bos_id = 0;
            read_u32( bos_id );
            vocab.special_token_ids_[ st.bos_token ] = static_cast<TokenId>(bos_id);
        }

        uint32_t has_pad = 0;
        read_u32( has_pad );

        if ( has_pad )
        {
            uint32_t pad_id = 0;
            read_u32( pad_id );
            vocab.special_token_ids_[ st.pad_token ] = static_cast<TokenId>(pad_id);
        }

        vocab.buildMergeMap();
        vocab.buildSpecialTokenList();

        return vocab;
    }

    BpeVocabulary BpeVocabulary::loadLlama32( const fs::path& path )
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

        uint32_t vocab_size = 0;
        read_u32( vocab_size );

        uint8_t use_byte_fallback = 0;
        file.read( reinterpret_cast<char*>(&use_byte_fallback), sizeof( use_byte_fallback ) );

        if ( !file )
        {
            throw std::runtime_error( "Failed reading use_byte_fallback field" );
        }

        BpeVocabularyConfig config = BpeVocabularyConfig()
            .withVocabSize( vocab_size )
            .withByteLevel( true )
            .withPreTokenization( PreTokenizationMode::Llama3Regex )
            .withPreTokenizationPattern( LLAMA3_PRETOKENIZATION_PATTERN )
            .withSpecialTokens( SpecialTokens::llamaStyle() );

        BpeVocabulary vocab( config );
        vocab.id_to_token_.resize( vocab_size );

        for ( uint32_t i = 0; i < vocab_size; ++i )
        {
            uint32_t len = 0;
            read_u32( len );

            std::string token( len, '\0' );

            if ( len > 0 )
            {
                file.read( token.data(), static_cast<std::streamsize>(len) );

                if ( !file )
                {
                    throw std::runtime_error(
                        "Failed reading token string at index " + std::to_string( i ) );
                }
            }

            float score = 0.0f;
            file.read( reinterpret_cast<char*>(&score), sizeof( score ) );

            if ( !file )
            {
                throw std::runtime_error(
                    "Failed reading token score at index " + std::to_string( i ) );
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

        // Special token order matches convert_llama_tokenizer.py: BOS, EOS, PAD, UNK.
        const auto& st = vocab.config_.getSpecialTokens();

        uint32_t has_bos = 0;
        read_u32( has_bos );

        if ( has_bos )
        {
            uint32_t bos_id = 0;
            read_u32( bos_id );
            vocab.special_token_ids_[ st.bos_token ] = static_cast<TokenId>(bos_id);
            std::cout << "  BOS: '" << st.bos_token << "' (ID: " << bos_id << ")\n";
        }

        uint32_t has_eos = 0;
        read_u32( has_eos );

        if ( has_eos )
        {
            uint32_t eos_id = 0;
            read_u32( eos_id );
            vocab.special_token_ids_[ st.eos_token ] = static_cast<TokenId>(eos_id);
            std::cout << "  EOS: '" << st.eos_token << "' (ID: " << eos_id << ")\n";
        }

        uint32_t has_pad = 0;
        read_u32( has_pad );

        if ( has_pad )
        {
            uint32_t pad_id = 0;
            read_u32( pad_id );
            vocab.special_token_ids_[ st.pad_token ] = static_cast<TokenId>(pad_id);
            std::cout << "  PAD: '" << st.pad_token << "' (ID: " << pad_id << ")\n";
        }

        uint32_t has_unk = 0;
        read_u32( has_unk );

        if ( has_unk )
        {
            uint32_t unk_id = 0;
            read_u32( unk_id );
            vocab.special_token_ids_[ st.unk_token ] = static_cast<TokenId>(unk_id);
            std::cout << "  UNK: '" << st.unk_token << "' (ID: " << unk_id << ")\n";
        }

        vocab.buildSpecialTokenList();

        std::cout << "Loaded Llama 3.2 vocabulary: "
            << vocab_size << " tokens, "
            << vocab.special_token_ids_.size() << " special tokens\n";

        return vocab;
    }

    BpeVocabulary BpeVocabulary::loadMistral(
        const fs::path& /*vocab_path*/,
        const fs::path& /*merges_path*/ )
    {
        throw std::runtime_error(
            "BpeVocabulary::loadMistral is not yet implemented. "
            "Provide a Mila binary produced by save() as a workaround." );
    }
}