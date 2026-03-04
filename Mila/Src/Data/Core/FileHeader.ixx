/**
 * @file FileHeader.ixx
 * @brief Common file header structure for Mila data files.
 *
 * Provides standardized file identification, versioning, and metadata storage.
 */

module;
#include <cstdint>
#include <string>
#include <vector>
#include <stdexcept>
#include <fstream>

export module Data.FileHeader;

import Serialization.Metadata;

namespace Mila::Data
{
    using Mila::Dnn::Serialization::SerializationMetadata;

    /**
     * @brief File type identifiers for Mila data files.
     */
    export enum class MilaFileType : uint32_t
    {
        Unknown = 0,
        BpeVocabulary = 1,
        CharVocabulary = 2,
        TokenizedCorpus = 3,
        Dataset = 4,
        TrainingCheckpoint = 5,
        Gpt4BpeVocabulary = 6,      // GPT-4 style BPE vocabulary (Llama 3.x, GPT-4, etc.)
    };

    export constexpr const char* toString( MilaFileType type )
    {
        switch ( type )
        {
            case MilaFileType::BpeVocabulary:     return "BpeVocabulary";
            case MilaFileType::CharVocabulary:    return "CharVocabulary";
            case MilaFileType::TokenizedCorpus:   return "TokenizedCorpus";
            case MilaFileType::Dataset:           return "Dataset";
            case MilaFileType::TrainingCheckpoint: return "TrainingCheckpoint";
            case MilaFileType::Gpt4BpeVocabulary: return "Gpt4BpeVocabulary";
            default:                              return "Unknown";
        }
    }

    /**
     * @brief Common file header for Mila data files.
     *
     * Structure:
     * - Magic bytes: "MILA" (4 bytes) for file identification
     * - Header version: uint32_t for future compatibility
     * - File type: MilaFileType enum identifying content
     * - Metadata size: uint32_t for variable-length metadata
     * - Metadata: JSON-serialized SerializationMetadata
     *
     * Followed by file-specific binary content.
     */
    export class MilaFileHeader
    {
    public:
        static constexpr uint32_t MAGIC = 0x4D494C41;  // "MILA" in little-endian
        static constexpr uint32_t VERSION = 1;

        MilaFileHeader() = default;

        MilaFileHeader( MilaFileType type, SerializationMetadata metadata = {} )
            : file_type_( type )
            , metadata_( std::move( metadata ) )
        {}

        void write( std::ostream& out ) const
        {
            uint32_t magic = MAGIC;
            out.write( reinterpret_cast<const char*>(&magic), sizeof( magic ) );

            uint32_t version = VERSION;
            out.write( reinterpret_cast<const char*>(&version), sizeof( version ) );

            uint32_t file_type = static_cast<uint32_t>(file_type_);
            out.write( reinterpret_cast<const char*>(&file_type), sizeof( file_type ) );

            std::string metadata_json = metadata_.toJson().dump();
            uint32_t metadata_size = static_cast<uint32_t>(metadata_json.size());
            out.write( reinterpret_cast<const char*>(&metadata_size), sizeof( metadata_size ) );

            if ( metadata_size > 0 )
            {
                out.write( metadata_json.data(), metadata_size );
            }

            if ( !out )
            {
                throw std::runtime_error( "MilaFileHeader: Failed to write header" );
            }
        }

        static MilaFileHeader read( std::istream& in )
        {
            uint32_t magic = 0;
            in.read( reinterpret_cast<char*>(&magic), sizeof( magic ) );

            if ( magic != MAGIC )
            {
                throw std::runtime_error( "MilaFileHeader: Invalid magic bytes (not a Mila file)" );
            }

            uint32_t version = 0;
            in.read( reinterpret_cast<char*>(&version), sizeof( version ) );

            if ( version != VERSION )
            {
                throw std::runtime_error(
                    "MilaFileHeader: Unsupported version " + std::to_string( version ) );
            }

            uint32_t file_type = 0;
            in.read( reinterpret_cast<char*>(&file_type), sizeof( file_type ) );

            uint32_t metadata_size = 0;
            in.read( reinterpret_cast<char*>(&metadata_size), sizeof( metadata_size ) );

            SerializationMetadata metadata;

            if ( metadata_size > 0 )
            {
                std::string metadata_json( metadata_size, '\0' );
                in.read( metadata_json.data(), metadata_size );

                if ( !in )
                {
                    throw std::runtime_error( "MilaFileHeader: Failed to read metadata" );
                }

                auto json_obj = nlohmann::json::parse( metadata_json );
                metadata = SerializationMetadata::fromJson( json_obj );
            }

            MilaFileHeader header;
            header.file_type_ = static_cast<MilaFileType>(file_type);
            header.metadata_ = std::move( metadata );

            return header;
        }

        MilaFileType getFileType() const
        {
            return file_type_;
        }
        const SerializationMetadata& getMetadata() const
        {
            return metadata_;
        }
        SerializationMetadata& getMetadata()
        {
            return metadata_;
        }

    private:
        MilaFileType file_type_ = MilaFileType::Unknown;
        SerializationMetadata metadata_;
    };
}
