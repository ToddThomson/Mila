/**
 * @file LocalDirectoryHub.cpp
 * @brief The bodies of LocalDirectoryHub's members. The interface is LocalDirectoryHub.ixx.
 */

module;
#include <cstdio>
#include <stdio.h>
#ifndef _WIN32
#  include <sys/types.h>
#endif

module Distribution.LocalDirectoryHub;

import std;
import Distribution.ModelCoordinate;
import Distribution.FetchReport;
import Distribution.ModelHub;

namespace Mila::Distribution
{
    namespace
    {
        using File = std::unique_ptr<std::FILE, int( * )( std::FILE* )>;

        // C stdio rather than streams, as WeightsReader reads: MSVC does not complete an istream's sentry instantiated
        // from a module.
        File openForReading( const std::filesystem::path& path )
        {
#ifdef _MSC_VER
#pragma warning( suppress : 4996 )
#endif
            return File( std::fopen( path.string().c_str(), "rb" ), &std::fclose );
        }

        bool seekTo( std::FILE* file, std::uint64_t offset )
        {
#ifdef _WIN32
            return ::_fseeki64( file, static_cast<long long>( offset ), SEEK_SET ) == 0;
#else
            return ::fseeko( file, static_cast<off_t>( offset ), SEEK_SET ) == 0;
#endif
        }

        /// A repository is one folder's name: nothing that could name another folder.
        bool isFolderName( const std::string& name )
        {
            return !name.empty() && name != "." && name != ".."
                && name.find_first_of( "/\\" ) == std::string::npos;
        }
    }

    LocalDirectoryHub::LocalDirectoryHub( std::filesystem::path root, Progress progress )
        : root_( std::move( root ) ), progress_( std::move( progress ) )
    {}

    std::string LocalDirectoryHub::name() const
    {
        return "directory";
    }

    std::vector<HubModel> LocalDirectoryHub::listModels( const std::string& owner ) const
    {
        if ( !std::filesystem::is_directory( root_ ) )
        {
            throw std::runtime_error( std::format( "{} names no folder of models: {}", kHubDirectoryVariable, root_.string() ) );
        }

        std::vector<HubModel> listed;

        for ( const auto& entry : std::filesystem::directory_iterator( root_ ) )
        {
            if ( !entry.is_directory() || !std::filesystem::exists( entry.path() / "mila.json" ) )
            {
                continue;
            }

            HubModel model;
            model.owner = owner;
            model.repository = entry.path().filename().string();
            model.library = "mila";

            for ( const auto& file : std::filesystem::directory_iterator( entry.path() ) )
            {
                if ( file.is_regular_file() )
                {
                    model.files.push_back( file.path().filename().string() );
                }
            }

            listed.push_back( std::move( model ) );
        }

        std::ranges::sort( listed, []( const HubModel& left, const HubModel& right ) { return left.repository < right.repository; } );

        return listed;
    }

    std::string LocalDirectoryHub::fetchManifest( const ModelCoordinate& coordinate ) const
    {
        const std::filesystem::path path = fileOf( coordinate, "mila.json" );
        File file = openForReading( path );

        if ( file == nullptr )
        {
            throw std::runtime_error( std::format( "{}: no such model in {}", coordinate.repository, root_.string() ) );
        }

        std::string text;
        char buffer[ 4096 ];
        std::size_t read = 0;

        while ( ( read = std::fread( buffer, 1, sizeof( buffer ), file.get() ) ) > 0 )
        {
            text.append( buffer, read );
        }

        return text;
    }

    FetchReport LocalDirectoryHub::fetchFile(
        const ModelCoordinate& coordinate,
        const std::string& path,
        std::uint64_t resume_from,
        const std::function<bool( const char*, std::size_t )>& sink ) const
    {
        const std::filesystem::path source = fileOf( coordinate, path );
        std::error_code missing;
        const std::uint64_t total = std::filesystem::file_size( source, missing );

        if ( missing )
        {
            return { FetchOutcome::Failed, std::format( "{}: {}", source.string(), missing.message() ) };
        }

        if ( resume_from > total )
        {
            return { FetchOutcome::RangeNotSatisfiable, "the resume offset is past the end of the file" };
        }

        File file = openForReading( source );

        if ( file == nullptr || !seekTo( file.get(), resume_from ) )
        {
            return { FetchOutcome::Failed, std::format( "{}: cannot be read", source.string() ) };
        }

        constexpr std::size_t kChunkBytes = static_cast<std::size_t>( 8 ) * 1024 * 1024;
        std::vector<char> buffer( kChunkBytes );
        std::uint64_t received = resume_from;
        std::size_t read = 0;

        while ( ( read = std::fread( buffer.data(), 1, buffer.size(), file.get() ) ) > 0 )
        {
            if ( !sink( buffer.data(), read ) )
            {
                return { FetchOutcome::Failed, "the store stopped taking the file" };
            }

            received += read;

            if ( progress_ && !progress_( received, total ) )
            {
                return { FetchOutcome::Failed, "the copy was stopped" };
            }
        }

        if ( received != total )
        {
            return { FetchOutcome::Failed, std::format( "{}: read {} of {} bytes", source.string(), received, total ) };
        }

        return { FetchOutcome::Complete, {} };
    }

    std::string LocalDirectoryHub::fetchFilePart(
        const ModelCoordinate& coordinate,
        const std::string& path,
        std::uint64_t first,
        std::uint64_t count ) const
    {
        const std::filesystem::path source = fileOf( coordinate, path );
        File file = openForReading( source );

        if ( file == nullptr || !seekTo( file.get(), first ) )
        {
            throw std::runtime_error( std::format( "{}: cannot be read from byte {}", source.string(), first ) );
        }

        std::string part( static_cast<std::size_t>( count ), '\0' );

        if ( std::fread( part.data(), 1, part.size(), file.get() ) != part.size() )
        {
            throw std::runtime_error( std::format( "{}: fewer than {} bytes from byte {}", source.string(), count, first ) );
        }

        return part;
    }

    std::filesystem::path LocalDirectoryHub::fileOf( const ModelCoordinate& coordinate, const std::string& path ) const
    {
        const std::filesystem::path relative( path );

        const bool leaves = relative.empty() || relative.is_absolute() || relative.has_root_name()
            || std::ranges::any_of( relative, []( const std::filesystem::path& part ) { return part == ".."; } );

        if ( !isFolderName( coordinate.repository ) || leaves )
        {
            throw std::runtime_error( std::format( "'{}' in '{}' names a file outside its package", path, coordinate.repository ) );
        }

        return root_ / coordinate.repository / relative;
    }
}
