/**
 * @file PublishedModel.cpp
 * @brief The bodies of PublishedModel's members: the listing, and the weights header read from the hub into the
 *        store. The interface is PublishedModel.ixx.
 */

module Distribution.PublishedModel;

import std;
import Distribution.ModelCoordinate;
import Distribution.ModelManifest;
import Distribution.ModelHub;
import Distribution.ModelStore;

namespace Mila::Distribution
{
    namespace
    {
        /// The bound WeightsReader puts on a safetensors header; a larger length is not a header.
        constexpr std::uint64_t kMostHeaderBytes = static_cast<std::uint64_t>( 128 ) * 1024 * 1024;

        std::uint64_t readLittleEndian( const std::string& bytes )
        {
            std::uint64_t value = 0;

            for ( std::size_t index = 0; index < sizeof( value ); ++index )
            {
                value |= static_cast<std::uint64_t>( static_cast<unsigned char>( bytes[ index ] ) ) << ( 8 * index );
            }

            return value;
        }
    }

    PublishedModel::PublishedModel( ModelCoordinate coordinate, std::optional<ModelManifest> manifest )
        : coordinate_( std::move( coordinate ) ), manifest_( std::move( manifest ) )
    {}

    std::vector<PublishedModel> PublishedModel::list( const IModelHub& hub, const std::string& owner )
    {
        std::vector<PublishedModel> published;

        for ( const HubModel& model : hub.listModels( owner ) )
        {
            if ( !model.hasManifest() )
            {
                continue;
            }

            ModelCoordinate coordinate{ owner, model.repository };

            // The revision listed is the one the manifest is read at, so the header read later matches it.
            if ( !model.revision.empty() )
            {
                coordinate.revision = model.revision;
            }

            std::optional<ModelManifest> manifest;

            try
            {
                manifest = parseModelManifest( hub.fetchManifest( coordinate ), model.repository );
            }
            catch ( const std::exception& )
            {
                // One manifest unread costs that model its description, not the whole list.
            }

            published.emplace_back( std::move( coordinate ), std::move( manifest ) );
        }

        return published;
    }

    const std::string& PublishedModel::name() const noexcept
    {
        return coordinate_.repository;
    }

    const ModelCoordinate& PublishedModel::coordinate() const noexcept
    {
        return coordinate_;
    }

    const std::optional<ModelManifest>& PublishedModel::manifest() const noexcept
    {
        return manifest_;
    }

    std::uint64_t PublishedModel::downloadBytes() const noexcept
    {
        std::uint64_t bytes = 0;

        if ( manifest_ )
        {
            for ( const ModelFile& file : manifest_->files )
            {
                bytes += file.bytes;
            }
        }

        return bytes;
    }

    std::filesystem::path PublishedModel::weightsHeader( const IModelHub& hub, const ModelStore& store ) const
    {
        if ( !manifest_ )
        {
            throw std::runtime_error( std::format( "'{}': its manifest could not be read", name() ) );
        }

        const ModelFile* weights = manifest_->file( kWeightsRole );

        if ( weights == nullptr || weights->sha256.empty() )
        {
            throw std::runtime_error( std::format( "'{}': its manifest names no weights with a digest", name() ) );
        }

        const std::filesystem::path path = store.root() / "headers" / ( "sha256-" + weights->sha256 + ".header" );
        std::error_code ignored;

        if ( std::filesystem::exists( path, ignored ) )
        {
            return path;
        }

        const std::string length_bytes = hub.fetchFilePart( coordinate_, weights->path, 0, sizeof( std::uint64_t ) );
        const std::uint64_t header_length = readLittleEndian( length_bytes );

        if ( header_length == 0 || header_length > kMostHeaderBytes
            || ( weights->bytes != 0 && sizeof( std::uint64_t ) + header_length > weights->bytes ) )
        {
            throw std::runtime_error( std::format(
                "'{}': its weights do not begin with a safetensors header (length {})", name(), header_length ) );
        }

        const std::string header = hub.fetchFilePart( coordinate_, weights->path, sizeof( std::uint64_t ), header_length );

        std::filesystem::create_directories( path.parent_path() );

        // Written beside its final name and renamed, so a header cut short is never read as a whole one.
        const std::filesystem::path partial = path.string() + ".partial";

        {
            std::ofstream output( partial, std::ios::binary | std::ios::trunc );
            output.write( length_bytes.data(), static_cast<std::streamsize>( length_bytes.size() ) );
            output.write( header.data(), static_cast<std::streamsize>( header.size() ) );

            if ( !output )
            {
                throw std::runtime_error( std::format( "'{}': cannot write {}", name(), partial.string() ) );
            }
        }

        std::filesystem::rename( partial, path );

        return path;
    }
}
