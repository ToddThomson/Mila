/**
 * @file BpeVocabulary.ixx
 * @brief Byte-Pair Encoding vocabulary with factory-based construction.
 *
 * Vocabularies are immutable after construction and store their configuration
 * for full provenance tracking. Use static factory methods to create instances.
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

import Data.BpeSpecialTokens;
import Data.BpeVocabularyConfig;
import Data.TokenizerVocabulary;
import Data.FileHeader;
import Serialization.Metadata;
import Data.Tokenizer;

namespace Mila::Data
{
    namespace fs = std::filesystem;
    using Mila::Dnn::Data::TokenizerVocabulary;
    using Mila::Dnn::Data::TokenId;
    using Mila::Dnn::Serialization::SerializationMetadata;

    /**
     * @brief Byte Pair Encoding (BPE) vocabulary implementation.
     *
     * Immutable vocabulary created via static factory methods. Stores configuration
     * for full provenance tracking and serialization.
     *
     * Thread safety: Immutable after construction, safe for concurrent reads.
     */
    export class BpeVocabulary : public TokenizerVocabulary
    {
    public:
        
        // ====================================================================
        // Factory Methods
        // ====================================================================

        /**
         * @brief Train a BPE vocabulary from text corpus.
         *
         * @param corpus Training text corpus.
         * @param config Vocabulary configuration.
         * @return Trained BpeVocabulary instance.
         * @throws std::invalid_argument if config is invalid.
         */
        static BpeVocabulary train( const std::string& corpus, const BpeVocabularyConfig& config )
        {
            config.validate();
            
            BpeVocabulary vocab( config );
            vocab.buildFromTextImpl( corpus );
            
            return vocab;
        }

        /**
         * @brief Train a BPE vocabulary from corpus file.
         *
         * @param corpus_path Path to training corpus text file.
         * @param config Vocabulary configuration.
         * @return Trained BpeVocabulary instance.
         * @throws std::runtime_error if file cannot be opened.
         * @throws std::invalid_argument if config is invalid.
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
         * @brief Load vocabulary from Mila binary format.
         *
         * Reads vocabulary and configuration from file written by save().
         *
         * @param path Input file path.
         * @return Loaded BpeVocabulary instance.
         * @throws std::runtime_error on I/O errors or format incompatibility.
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
                throw std::runtime_error( "File is not a BPE vocabulary: " + path.string() );
            }

            BpeVocabularyConfig config;
            config.fromMetadata( header.getMetadata() );

            BpeVocabulary vocab( config );
            vocab.loadContentImpl( file );

            return vocab;
        }

        /**
         * @brief Load pre-trained GPT-2 vocabulary.
         *
         * @param tokenizer_path Path to converted GPT-2 tokenizer binary.
         * @return Loaded BpeVocabulary instance.
         * @throws std::runtime_error on I/O or format errors.
         */
        static BpeVocabulary loadGpt2( const fs::path& tokenizer_path );

        /**
         * @brief Load pre-trained LLAMA vocabulary.
         *
         * @param model_path Path to LLAMA model file.
         * @return Loaded BpeVocabulary instance.
         * @throws std::runtime_error - Currently not implemented.
         */
        static BpeVocabulary loadLlama( const fs::path& model_path );

        /**
         * @brief Load pre-trained Mistral vocabulary.
         *
         * @param vocab_path Path to Mistral vocabulary file.
         * @param merges_path Path to Mistral merges file.
         * @return Loaded BpeVocabulary instance.
         * @throws std::runtime_error - Currently not implemented.
         */
        static BpeVocabulary loadMistral( const fs::path& vocab_path, const fs::path& merges_path );

        // Delete default constructor - force use of factories
        BpeVocabulary() = delete;

        // ====================================================================
        // Configuration Access
        // ====================================================================

        /**
         * @brief Get the configuration used to create this vocabulary.
         *
         * @return const BpeVocabularyConfig& Configuration reference.
         */
        const BpeVocabularyConfig& getConfig() const { return config_; }

        // ====================================================================
        // Serialization
        // ====================================================================

        /**
         * @brief Serialize vocabulary to disk with configuration.
         *
         * File format includes MilaFileHeader with config metadata followed
         * by binary vocabulary content.
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
            meta.set( "format_version", static_cast<int64_t>( 1 ) )
                .set( "actual_vocab_size", static_cast<int64_t>( id_to_token_.size() ) )
                .set( "num_merges", static_cast<int64_t>( merges_.size() ) );

            MilaFileHeader header( MilaFileType::BpeVocabulary, std::move( meta ) );
            header.write( file );

            saveContentImpl( file );
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

            auto unk_it = special_token_ids_.find( 'u' );
            if ( unk_it != special_token_ids_.end() )
            {
                return unk_it->second;
            }

            return std::nullopt;
        }

        std::optional<std::string> idToToken( TokenId id ) const override
        {
            if ( id >= 0 && static_cast<size_t>( id ) < id_to_token_.size() )
            {
                return id_to_token_[ static_cast<size_t>( id ) ];
            }
            return std::nullopt;
        }

        // ====================================================================
        // BPE-Specific Methods
        // ====================================================================

        const std::vector<std::pair<std::string, std::string>>& getMergeRules() const
        {
            return merges_;
        }

        std::optional<TokenId> getSpecialTokenId( char type ) const
        {
            auto it = special_token_ids_.find( type );
            if ( it != special_token_ids_.end() )
            {
                return it->second;
            }
            return std::nullopt;
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
            : config_( config )
            , current_id_( 0 )
        {
        }

        // ====================================================================
        // Training Implementation
        // ====================================================================

        void buildFromTextImpl( const std::string& corpus );
        
        void initializeBaseVocabulary();
        void addSpecialTokensFromConfig();
        std::vector<std::string> preTokenizeCorpus( const std::string& text );
        std::vector<std::vector<std::string>> convertToTokenSequences( const std::vector<std::string>& words );
        void runBpeMergeLoop( std::vector<std::vector<std::string>>& corpus_tokens, std::chrono::steady_clock::time_point start_time );
        void logTrainingComplete( std::chrono::steady_clock::time_point start_time );

        // ====================================================================
        // Serialization Implementation
        // ====================================================================

        void saveContentImpl( std::ostream& file ) const;
        void loadContentImpl( std::istream& file );

        // ====================================================================
        // Helper Methods
        // ====================================================================

        struct PairHash
        {
            size_t operator()( const std::pair<std::string, std::string>& p ) const
            {
                return std::hash<std::string>{}( p.first ) ^ (std::hash<std::string>{}( p.second ) << 1);
            }
        };

        struct PairViewHash
        {
            std::size_t operator()( const std::pair<std::string_view, std::string_view>& p ) const
            {
                std::size_t h1 = std::hash<std::string_view>{}( p.first );
                std::size_t h2 = std::hash<std::string_view>{}( p.second );
                return h1 ^ (h2 << 1);
            }
        };

        void buildMergeMap();
        std::vector<std::string> preTokenize( const std::string& text ) const;
        
        std::unordered_map<std::pair<std::string, std::string>, size_t, PairHash> countPairs(
            const std::vector<std::vector<std::string>>& corpus ) const;
            
        std::pair<std::pair<std::string, std::string>, size_t> getMostFrequentPair(
            const std::unordered_map<std::pair<std::string, std::string>, size_t, PairHash>& counts ) const;
            
        void applyMergeAndUpdateCounts(
            std::vector<std::vector<std::string>>& corpus,
            const std::string& left,
            const std::string& right,
            const std::string& merged,
            std::unordered_map<std::pair<std::string, std::string>, size_t, PairHash>& pair_counts );
            
        void addSpecialToken( const std::string& token, TokenId id, const std::string& type );

        // ====================================================================
        // Member Variables
        // ====================================================================

        BpeVocabularyConfig config_;
        TokenId current_id_;

        std::unordered_map<std::string, TokenId> token_to_id_;
        std::vector<std::string> id_to_token_;
        std::vector<std::pair<std::string, std::string>> merges_;
        std::unordered_map<std::pair<std::string, std::string>, size_t, PairHash> merge_map_;
        std::unordered_map<char, TokenId> special_token_ids_;
    };

    // ========================================================================
    // Training Implementation
    // ========================================================================

    void BpeVocabulary::buildFromTextImpl( const std::string& corpus )
    {
        const auto start_time = std::chrono::steady_clock::now();

        token_to_id_.clear();
        id_to_token_.clear();
        merges_.clear();
        special_token_ids_.clear();
        current_id_ = 0;

        initializeBaseVocabulary();
        addSpecialTokensFromConfig();

        const size_t target_vocab_size = config_.getVocabSize();
        if ( target_vocab_size == 0 || static_cast<size_t>( current_id_ ) >= target_vocab_size )
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
        if ( config_.isByteLevel() )
        {
            for ( int i = 0; i < 256; ++i )
            {
                std::string byte_token( 1, static_cast<char>( i ) );
                id_to_token_.push_back( byte_token );
                token_to_id_[ byte_token ] = current_id_++;
            }
        }
        else
        {
            throw std::runtime_error( "BpeVocabulary: Character-level BPE not yet implemented. Use byte_level=true." );
        }
    }

    void BpeVocabulary::addSpecialTokensFromConfig()
    {
        const auto& special_tokens = config_.getSpecialTokens();
        if ( !special_tokens.enabled )
        {
            return;
        }

        special_tokens.validate();

        if ( !special_tokens.pad_token.empty() )
        {
            addSpecialToken( special_tokens.pad_token, current_id_++, "pad" );
        }
        if ( !special_tokens.unk_token.empty() )
        {
            addSpecialToken( special_tokens.unk_token, current_id_++, "unk" );
        }
        if ( !special_tokens.bos_token.empty() )
        {
            addSpecialToken( special_tokens.bos_token, current_id_++, "bos" );
        }
        if ( !special_tokens.eos_token.empty() )
        {
            addSpecialToken( special_tokens.eos_token, current_id_++, "eos" );
        }
        if ( !special_tokens.mask_token.empty() )
        {
            addSpecialToken( special_tokens.mask_token, current_id_++, "mask" );
        }
        if ( !special_tokens.sep_token.empty() )
        {
            addSpecialToken( special_tokens.sep_token, current_id_++, "sep" );
        }
        if ( !special_tokens.cls_token.empty() )
        {
            addSpecialToken( special_tokens.cls_token, current_id_++, "cls" );
        }
    }

    std::vector<std::string> BpeVocabulary::preTokenizeCorpus( const std::string& text )
    {
        return preTokenize( text );
    }

    std::vector<std::vector<std::string>> BpeVocabulary::convertToTokenSequences( const std::vector<std::string>& words )
    {
        std::vector<std::vector<std::string>> corpus_tokens;
        corpus_tokens.reserve( words.size() );

        for ( const auto& word : words )
        {
            std::vector<std::string> tokens;
            tokens.reserve( word.size() );

            for ( unsigned char byte : word )
            {
                tokens.push_back( std::string( 1, static_cast<char>( byte ) ) );
            }

            if ( !tokens.empty() )
            {
                corpus_tokens.push_back( std::move( tokens ) );
            }
        }

        return corpus_tokens;
    }

    void BpeVocabulary::runBpeMergeLoop( std::vector<std::vector<std::string>>& corpus_tokens, std::chrono::steady_clock::time_point start_time )
    {
        const size_t target_vocab_size = config_.getVocabSize();
        const size_t min_frequency = config_.getMinFrequency();
        const size_t max_merges = config_.getMaxMerges();

        size_t merges_performed = 0;
        auto pair_counts = countPairs( corpus_tokens );

        while ( static_cast<size_t>( current_id_ ) < target_vocab_size )
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
                    std::chrono::steady_clock::now() - start_time
                ).count();

                float progress = static_cast<float>( merges_performed ) / static_cast<float>( target_vocab_size ) * 100.0f;

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

        std::cout << std::endl;
    }

    void BpeVocabulary::logTrainingComplete( std::chrono::steady_clock::time_point start_time )
    {
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - start_time
        ).count();

        std::cout << "Training completed in " << elapsed << "s" << std::endl;
        std::cout << "Final vocabulary size: " << id_to_token_.size() << std::endl;
    }

    // ========================================================================
    // Serialization Implementation
    // ========================================================================

    void BpeVocabulary::saveContentImpl( std::ostream& file ) const
    {
        uint32_t version = 1;
        file.write( reinterpret_cast<const char*>( &version ), sizeof( version ) );

        uint32_t vocab_size = static_cast<uint32_t>( id_to_token_.size() );
        file.write( reinterpret_cast<const char*>( &vocab_size ), sizeof( vocab_size ) );

        uint32_t num_special = static_cast<uint32_t>( special_token_ids_.size() );
        file.write( reinterpret_cast<const char*>( &num_special ), sizeof( num_special ) );

        for ( const auto& [type, id] : special_token_ids_ )
        {
            uint8_t type_byte = static_cast<uint8_t>( type );
            file.write( reinterpret_cast<const char*>( &type_byte ), sizeof( type_byte ) );

            uint32_t token_id = static_cast<uint32_t>( id );
            file.write( reinterpret_cast<const char*>( &token_id ), sizeof( token_id ) );

            const std::string& token_str = id_to_token_[ static_cast<size_t>( id ) ];
            uint32_t len = static_cast<uint32_t>( token_str.size() );
            file.write( reinterpret_cast<const char*>( &len ), sizeof( len ) );
            file.write( token_str.data(), len );
        }

        uint32_t num_merges = static_cast<uint32_t>( merges_.size() );
        file.write( reinterpret_cast<const char*>( &num_merges ), sizeof( num_merges ) );

        for ( const auto& [left, right] : merges_ )
        {
            uint32_t left_len = static_cast<uint32_t>( left.size() );
            file.write( reinterpret_cast<const char*>( &left_len ), sizeof( left_len ) );
            file.write( left.data(), left_len );

            uint32_t right_len = static_cast<uint32_t>( right.size() );
            file.write( reinterpret_cast<const char*>( &right_len ), sizeof( right_len ) );
            file.write( right.data(), right_len );
        }

        for ( const auto& token : id_to_token_ )
        {
            uint32_t len = static_cast<uint32_t>( token.size() );
            file.write( reinterpret_cast<const char*>( &len ), sizeof( len ) );
            if ( len > 0 )
            {
                file.write( token.data(), len );
            }
        }

        if ( !file )
        {
            throw std::runtime_error( "Error writing vocabulary content" );
        }
    }

    void BpeVocabulary::loadContentImpl( std::istream& file )
    {
        uint32_t version = 0;
        file.read( reinterpret_cast<char*>( &version ), sizeof( version ) );
        if ( version != 1 )
        {
            throw std::runtime_error( "Unsupported vocabulary content version: " + std::to_string( version ) );
        }

        uint32_t vocab_size = 0;
        file.read( reinterpret_cast<char*>( &vocab_size ), sizeof( vocab_size ) );

        uint32_t num_special = 0;
        file.read( reinterpret_cast<char*>( &num_special ), sizeof( num_special ) );

        for ( uint32_t i = 0; i < num_special; ++i )
        {
            uint8_t type_byte = 0;
            file.read( reinterpret_cast<char*>( &type_byte ), sizeof( type_byte ) );

            uint32_t token_id = 0;
            file.read( reinterpret_cast<char*>( &token_id ), sizeof( token_id ) );

            uint32_t len = 0;
            file.read( reinterpret_cast<char*>( &len ), sizeof( len ) );

            std::string token_str;
            if ( len > 0 )
            {
                token_str.resize( len );
                file.read( token_str.data(), len );
            }

            special_token_ids_[ static_cast<char>( type_byte ) ] = static_cast<TokenId>( token_id );
        }

        uint32_t num_merges = 0;
        file.read( reinterpret_cast<char*>( &num_merges ), sizeof( num_merges ) );

        merges_.reserve( num_merges );
        for ( uint32_t i = 0; i < num_merges; ++i )
        {
            uint32_t left_len = 0;
            file.read( reinterpret_cast<char*>( &left_len ), sizeof( left_len ) );

            std::string left;
            if ( left_len > 0 )
            {
                left.resize( left_len );
                file.read( left.data(), left_len );
            }

            uint32_t right_len = 0;
            file.read( reinterpret_cast<char*>( &right_len ), sizeof( right_len ) );

            std::string right;
            if ( right_len > 0 )
            {
                right.resize( right_len );
                file.read( right.data(), right_len );
            }

            merges_.emplace_back( left, right );
        }

        id_to_token_.resize( vocab_size );
        for ( uint32_t i = 0; i < vocab_size; ++i )
        {
            uint32_t len = 0;
            file.read( reinterpret_cast<char*>( &len ), sizeof( len ) );

            std::string token;
            if ( len > 0 )
            {
                token.resize( len );
                file.read( token.data(), len );
            }

            id_to_token_[ i ] = std::move( token );
            token_to_id_[ id_to_token_[ i ] ] = static_cast<TokenId>( i );
        }

        buildMergeMap();

        if ( !file )
        {
            throw std::runtime_error( "Error reading vocabulary content" );
        }
    }

    // ========================================================================
    // Helper Methods Implementation
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
            return std::make_pair( std::make_pair( std::string(), std::string() ), size_t( 0 ) );
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
                        auto old_left_pair = std::make_pair( word[ i - 1 ], word[ i ] );
                        if ( --pair_counts[ old_left_pair ] == 0 )
                        {
                            pair_counts.erase( old_left_pair );
                        }
                    }
                    if ( i + 2 < word.size() )
                    {
                        auto old_right_pair = std::make_pair( word[ i + 1 ], word[ i + 2 ] );
                        if ( --pair_counts[ old_right_pair ] == 0 )
                        {
                            pair_counts.erase( old_right_pair );
                        }
                    }

                    word[ i ] = merged;
                    word.erase( word.begin() + i + 1 );

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

    void BpeVocabulary::addSpecialToken( const std::string& token, TokenId id, const std::string& type )
    {
        id_to_token_.push_back( token );
        token_to_id_[ token ] = id;

        char type_char = type.empty() ? '?' : type[ 0 ];
        special_token_ids_[ type_char ] = id;
    }

    // ========================================================================
    // Static Byte Encoder/Decoder (GPT-2 style)
    // ========================================================================

    const std::unordered_map<unsigned char, std::string>& BpeVocabulary::getByteEncoder()
    {
        static std::unordered_map<unsigned char, std::string> encoder = []()
        {
            std::unordered_map<unsigned char, std::string> enc;

            std::vector<int> bs;
            for ( int i = int( '!' ); i <= int( '~' ); ++i ) bs.push_back( i );
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
                char byte = static_cast<char>( bs[ i ] );
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

                enc[ static_cast<unsigned char>( byte ) ] = utf8_str;
            }
            return enc;
        }();
        return encoder;
    }

    const std::unordered_map<std::string, unsigned char>& BpeVocabulary::getByteDecoder()
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

    // ========================================================================
    // External Format Loaders
    // ========================================================================

    BpeVocabulary BpeVocabulary::loadGpt2( const std::filesystem::path& tokenizerPath )
    {
        std::ifstream file( tokenizerPath, std::ios::binary );

        if ( !file )
        {
            throw std::runtime_error( "Cannot open GPT-2 tokenizer file: " + tokenizerPath.string() );
        }

        auto read_u32 = [&]( uint32_t& out )
        {
            file.read( reinterpret_cast<char*>( &out ), sizeof( out ) );
            if ( !file )
            {
                throw std::runtime_error( "Unexpected EOF while parsing GPT-2 file: " + tokenizerPath.string() );
            }
        };

        uint32_t vocab_size = 0;
        read_u32( vocab_size );

        uint32_t num_merges = 0;
        read_u32( num_merges );

        BpeVocabularyConfig config = BpeVocabularyConfig()
            .withVocabSize( vocab_size )
            .withByteLevel( true )
            .withSpecialTokens( BpeSpecialTokens::gptStyle() );

        BpeVocabulary vocab( config );
        vocab.id_to_token_.resize( vocab_size );

        for ( uint32_t i = 0; i < vocab_size; ++i )
        {
            uint32_t len = 0;
            read_u32( len );

            std::string token;
            if ( len > 0 )
            {
                token.resize( len );
                file.read( token.data(), static_cast<std::streamsize>( len ) );
                if ( !file )
                {
                    throw std::runtime_error( "Failed reading token from GPT-2 file" );
                }
            }

            uint32_t token_id = 0;
            read_u32( token_id );

            if ( token_id >= vocab_size )
            {
                throw std::runtime_error( "Invalid token id: " + std::to_string( token_id ) );
            }

            vocab.id_to_token_[ token_id ] = token;
            vocab.token_to_id_[ token ] = static_cast<TokenId>( token_id );
        }

        vocab.merges_.reserve( num_merges );
        for ( uint32_t i = 0; i < num_merges; ++i )
        {
            uint32_t llen = 0;
            read_u32( llen );

            std::string left;
            if ( llen > 0 )
            {
                left.resize( llen );
                file.read( left.data(), static_cast<std::streamsize>( llen ) );
                if ( !file )
                {
                    throw std::runtime_error( "Failed reading merge left token" );
                }
            }

            uint32_t rlen = 0;
            read_u32( rlen );

            std::string right;
            if ( rlen > 0 )
            {
                right.resize( rlen );
                file.read( right.data(), static_cast<std::streamsize>( rlen ) );
                if ( !file )
                {
                    throw std::runtime_error( "Failed reading merge right token" );
                }
            }

            vocab.merges_.emplace_back( std::move( left ), std::move( right ) );
        }

        uint32_t has_eos = 0;
        read_u32( has_eos );
        if ( has_eos )
        {
            uint32_t eos_id = 0;
            read_u32( eos_id );
            vocab.special_token_ids_[ 'e' ] = static_cast<TokenId>( eos_id );
        }

        uint32_t has_bos = 0;
        read_u32( has_bos );
        if ( has_bos )
        {
            uint32_t bos_id = 0;
            read_u32( bos_id );
            vocab.special_token_ids_[ 'b' ] = static_cast<TokenId>( bos_id );
        }

        uint32_t has_pad = 0;
        read_u32( has_pad );
        if ( has_pad )
        {
            uint32_t pad_id = 0;
            read_u32( pad_id );
            vocab.special_token_ids_[ 'p' ] = static_cast<TokenId>( pad_id );
        }

        vocab.buildMergeMap();

        return vocab;
    }

    BpeVocabulary BpeVocabulary::loadLlama( const std::filesystem::path& modelPath )
    {
        try
        {
            return load( modelPath );
        }
        catch ( const std::exception& )
        {
            throw std::runtime_error(
                "BpeVocabulary::loadLlama not implemented for model formats; "
                "provide a binary vocabulary produced by BpeVocabulary::save(), or implement LLAMA parsing." );
        }
    }

    BpeVocabulary BpeVocabulary::loadMistral(
        const std::filesystem::path& vocabPath,
        const std::filesystem::path& mergesPath )
    {
        try
        {
            return load( vocabPath );
        }
        catch ( const std::exception& )
        {
            throw std::runtime_error(
                "BpeVocabulary::loadMistral not implemented for non-binary vocab files. "
                "Provide a binary vocabulary produced by BpeVocabulary::save(), or implement Mistral parsing." );
        }
    }
}