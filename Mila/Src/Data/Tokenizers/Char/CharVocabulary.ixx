/** @file CharVocabulary.ixx
 * @brief Character vocabulary with factory-based construction.
 *
 * Vocabularies are immutable after construction and store their configuration
 * for full provenance tracking. Use static factory methods to create instances.
 */

module;
#include <vector>
#include <unordered_map>
#include <string>
#include <fstream>
#include <algorithm>
#include <stdexcept>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <cctype>

export module Data.CharVocabulary;

import Data.SpecialTokens;
import Data.TokenizerVocabulary;
import Data.CharVocabularyConfig;
import Data.FileHeader;
import Serialization.Metadata;
import Data.Tokenizer;

namespace Mila::Data
{
    using Mila::Data::TokenizerVocabulary;
    using Mila::Data::TokenId;
    using Mila::Dnn::Serialization::SerializationMetadata;
    namespace fs = std::filesystem;

    /**
     * @brief Character vocabulary for tokenization.
     *
     * Immutable vocabulary created via static factory methods. Stores configuration
     * for full provenance tracking and serialization.
     *
     * Thread safety: Immutable after construction, safe for concurrent reads.
     */
    export class CharVocabulary : public TokenizerVocabulary
    {
    public:

        // ====================================================================
        // Factory Methods
        // ====================================================================

        /**
         * @brief Build a character vocabulary from text corpus.
         *
         * @param corpus Training text corpus.
         * @param config Vocabulary configuration.
         * @return Trained CharVocabulary instance.
         * @throws std::invalid_argument if config is invalid.
         */
        static CharVocabulary train( const std::string& corpus, const CharVocabularyConfig& config )
        {
            config.validate();

            CharVocabulary vocab( config );
            vocab.buildFromText( corpus );

            return vocab;
        }

        /**
         * @brief Build a character vocabulary from corpus file.
         *
         * @param corpus_path Path to training corpus text file.
         * @param config Vocabulary configuration.
         * @return Trained CharVocabulary instance.
         * @throws std::runtime_error if file cannot be opened.
         * @throws std::invalid_argument if config is invalid.
         */
        static CharVocabulary trainFromFile( const fs::path& corpus_path, const CharVocabularyConfig& config )
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
         * @return Loaded CharVocabulary instance.
         * @throws std::runtime_error on I/O errors or format incompatibility.
         */
        static CharVocabulary load( const fs::path& path )
        {
            std::ifstream file( path, std::ios::binary );

            if ( !file )
            {
                throw std::runtime_error( "Cannot open vocabulary file: " + path.string() );
            }

            MilaFileHeader header = MilaFileHeader::read( file );

            if ( header.getFileType() != MilaFileType::CharVocabulary )
            {
                throw std::runtime_error( "File is not a character vocabulary: " + path.string() );
            }

            CharVocabularyConfig config;
            config.fromMetadata( header.getMetadata() );

            CharVocabulary vocab( config );
            vocab.loadContent( file );

            return vocab;
        }

        CharVocabulary() = delete;

        // ====================================================================
        // Configuration Access
        // ====================================================================

        /**
         * @brief Get the configuration used to create this vocabulary.
         *
         * @return const CharVocabularyConfig& Configuration reference.
         */
        const CharVocabularyConfig& getConfig() const { return config_; }

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
                .set( "actual_vocab_size", static_cast<int64_t>( idx_to_char_.size() ) )
                .set( "has_special_tokens", pad_token_id_ >= 0 );

            MilaFileHeader header( MilaFileType::CharVocabulary, std::move( meta ) );
            header.write( file );

            saveContent( file );
        }

        // ====================================================================
        // TokenizerVocabulary Interface
        // ====================================================================

        size_t getSize() const override
        {
            return idx_to_char_.size();
        }

        std::optional<TokenId> tokenToId( const std::string& token ) const override
        {
            if ( token.empty() )
            {
                return std::nullopt;
            }

            unsigned char c = static_cast<unsigned char>( token[ 0 ] );
            auto it = char_to_idx_.find( static_cast<char>( c ) );

            if ( it != char_to_idx_.end() )
            {
                return it->second;
            }

            if ( unk_token_id_ >= 0 )
            {
                return unk_token_id_;
            }

            return std::nullopt;
        }

        std::optional<std::string> idToToken( TokenId id ) const override
        {
            if ( id >= 0 && static_cast<size_t>( id ) < idx_to_char_.size() )
            {
                return std::string( 1, idx_to_char_[ static_cast<size_t>( id ) ] );
            }

            return std::nullopt;
        }

        // ====================================================================
        // Character-Specific API
        // ====================================================================

        TokenId charToIndex( char c ) const
        {
            auto it = char_to_idx_.find( c );

            if ( it != char_to_idx_.end() )
            {
                return it->second;
            }

            return unk_token_id_ >= 0 ? unk_token_id_ : static_cast<TokenId>(0);
        }

        char indexToChar( TokenId idx ) const
        {
            if ( idx >= 0 && static_cast<size_t>( idx ) < idx_to_char_.size() )
            {
                return idx_to_char_[ static_cast<size_t>( idx ) ];
            }

            return '?';
        }

        TokenId padTokenId() const
        {
            return pad_token_id_;
        }

        TokenId unkTokenId() const
        {
            return unk_token_id_;
        }

        bool hasSpecialTokens() const
        {
            return pad_token_id_ >= 0;
        }

    private:

        // ====================================================================
        // Private Constructor
        // ====================================================================

        explicit CharVocabulary( const CharVocabularyConfig& config )
            : config_( config )
            , pad_token_id_( -1 )
            , unk_token_id_( -1 )
        {
        }

        // ====================================================================
        // Training Implementation
        // ====================================================================

        void buildFromText( const std::string& corpus )
        {
            char_to_idx_.clear();
            idx_to_char_.clear();

            std::string normalized = normalizeText( corpus );
            auto unique_bytes = extractUniqueBytes( normalized );
            auto sorted_bytes = sortBytes( unique_bytes );

            addSpecialTokensFromConfig();
            addRegularTokens( sorted_bytes );
        }

        std::string normalizeText( const std::string& text ) const
        {
            std::string norm;
            norm.reserve( text.size() );

            if ( config_.isCaseSensitive() )
            {
                for ( size_t i = 0; i < text.size(); ++i )
                {
                    char c = text[ i ];

                    if ( c == '\r' )
                    {
                        if ( i + 1 < text.size() && text[ i + 1 ] == '\n' )
                        {
                            continue;
                        }

                        norm.push_back( '\n' );
                    }
                    else
                    {
                        norm.push_back( c );
                    }
                }
            }
            else
            {
                for ( size_t i = 0; i < text.size(); ++i )
                {
                    unsigned char cu = static_cast<unsigned char>( text[ i ] );

                    if ( cu == '\r' )
                    {
                        if ( i + 1 < text.size() && static_cast<unsigned char>( text[ i + 1 ] ) == '\n' )
                        {
                            continue;
                        }

                        norm.push_back( '\n' );
                    }
                    else
                    {
                        char lowered = static_cast<char>( std::tolower( cu ) );
                        norm.push_back( lowered );
                    }
                }
            }

            return norm;
        }

        std::unordered_map<unsigned char, bool> extractUniqueBytes( const std::string& text ) const
        {
            std::unordered_map<unsigned char, bool> unique_bytes;

            for ( char c : text )
            {
                unique_bytes[ static_cast<unsigned char>( c ) ] = true;
            }

            return unique_bytes;
        }

        std::vector<unsigned char> sortBytes( const std::unordered_map<unsigned char, bool>& unique_bytes ) const
        {
            std::vector<unsigned char> sorted_bytes;
            sorted_bytes.reserve( unique_bytes.size() );

            for ( const auto& kv : unique_bytes )
            {
                sorted_bytes.push_back( kv.first );
            }

            std::sort( sorted_bytes.begin(), sorted_bytes.end() );

            return sorted_bytes;
        }

        void addSpecialTokensFromConfig()
        {
            const SpecialTokens& st = config_.getSpecialTokens();
            TokenId idx = 0;

            if ( st.use_pad )
            {
                pad_token_id_ = idx;
                idx_to_char_.push_back( '\0' );
                char_to_idx_[ '\0' ] = idx++;
            }

            if ( st.use_unk )
            {
                unk_token_id_ = idx;
                idx_to_char_.push_back( '\1' );
                char_to_idx_[ '\1' ] = idx++;
            }

            if ( st.use_bos )
            {
                idx_to_char_.push_back( '\2' );
                char_to_idx_[ '\2' ] = idx++;
            }

            if ( st.use_eos )
            {
                idx_to_char_.push_back( '\3' );
                char_to_idx_[ '\3' ] = idx++;
            }

            if ( st.use_mask )
            {
                idx_to_char_.push_back( '\4' );
                char_to_idx_[ '\4' ] = idx++;
            }

            if ( st.use_sep )
            {
                idx_to_char_.push_back( '\5' );
                char_to_idx_[ '\5' ] = idx++;
            }

            if ( st.use_cls )
            {
                idx_to_char_.push_back( '\6' );
                char_to_idx_[ '\6' ] = idx++;
            }
        }

        void addRegularTokens( const std::vector<unsigned char>& sorted_bytes )
        {
            TokenId idx = static_cast<TokenId>( idx_to_char_.size() );

            std::vector<unsigned char> reserved = { '\0', '\1', '\2', '\3', '\4', '\5', '\6' };

            for ( unsigned char ub : sorted_bytes )
            {
                bool is_reserved = false;

                for ( unsigned char rb : reserved )
                {
                    if ( rb == ub )
                    {
                        is_reserved = true;
                        break;
                    }
                }

                if ( !is_reserved )
                {
                    char c = static_cast<char>( ub );
                    idx_to_char_.push_back( c );
                    char_to_idx_[ c ] = idx++;
                }
            }
        }

        // ====================================================================
        // Serialization Implementation
        // ====================================================================

        void saveContent( std::ostream& file ) const
        {
            size_t vocab_size = idx_to_char_.size();
            file.write( reinterpret_cast<const char*>( &vocab_size ), sizeof( vocab_size ) );

            bool has_special = (pad_token_id_ >= 0);
            file.write( reinterpret_cast<const char*>( &has_special ), sizeof( has_special ) );

            if ( has_special )
            {
                file.write( reinterpret_cast<const char*>( &pad_token_id_ ), sizeof( pad_token_id_ ) );
                file.write( reinterpret_cast<const char*>( &unk_token_id_ ), sizeof( unk_token_id_ ) );
            }

            for ( char c : idx_to_char_ )
            {
                file.write( &c, sizeof( char ) );
            }

            if ( !file )
            {
                throw std::runtime_error( "Error writing vocabulary content" );
            }
        }

        void loadContent( std::istream& file )
        {
            size_t vocab_size;
            file.read( reinterpret_cast<char*>( &vocab_size ), sizeof( vocab_size ) );

            bool has_special;
            file.read( reinterpret_cast<char*>( &has_special ), sizeof( has_special ) );

            if ( has_special )
            {
                file.read( reinterpret_cast<char*>( &pad_token_id_ ), sizeof( pad_token_id_ ) );
                file.read( reinterpret_cast<char*>( &unk_token_id_ ), sizeof( unk_token_id_ ) );
            }
            else
            {
                pad_token_id_ = static_cast<TokenId>( -1 );
                unk_token_id_ = static_cast<TokenId>( -1 );
            }

            idx_to_char_.resize( vocab_size );
            for ( size_t i = 0; i < vocab_size; ++i )
            {
                char c;
                file.read( &c, sizeof( char ) );
                idx_to_char_[ i ] = c;
                char_to_idx_[ c ] = static_cast<TokenId>( i );
            }

            if ( !file )
            {
                throw std::runtime_error( "Error reading vocabulary content" );
            }
        }

        // ====================================================================
        // Member Variables
        // ====================================================================

        CharVocabularyConfig config_;

        std::unordered_map<char, TokenId> char_to_idx_;
        std::vector<char> idx_to_char_;
        TokenId pad_token_id_;
        TokenId unk_token_id_;
    };
}